#!/usr/bin/env python3
"""Share one CommunicationActivityTracker across all offload managers of a model.

Models with several offloaded ModuleLists (Flux, HunyuanVideo: double- and
single-stream blocks) get one manager per list. The forward context only carries
the first manager's tracker, so the other managers' prefetch workers never saw an
active collective and never yielded. With this patch configure_layerwise_offload
creates a single tracker and hands it to every manager.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py"
)
src = path.read_text()
if "_owns_comm_tracker" in src:
    print("already patched:", path)
    sys.exit(0)

edits = [
    (
        '''        comm_active_window_mode: str = "kernel",
        prefetch_chunk_size_mb: int = 32,
''',
        '''        comm_active_window_mode: str = "kernel",
        comm_tracker: CommunicationActivityTracker | None = None,
        prefetch_chunk_size_mb: int = 32,
''',
    ),
    (
        '''        self.comm_tracker = (
            CommunicationActivityTracker(window_mode=self.comm_active_window_mode)
            if self.comm_aware
            else None
        )
''',
        '''        self._owns_comm_tracker = False
        if not self.comm_aware:
            self.comm_tracker = None
        elif comm_tracker is not None:
            # One tracker per model, shared by every offloaded ModuleList.
            self.comm_tracker = comm_tracker
        else:
            self.comm_tracker = CommunicationActivityTracker(
                window_mode=self.comm_active_window_mode
            )
            self._owns_comm_tracker = True
''',
    ),
    (
        '''        if self.comm_tracker is not None:
            self.comm_tracker.unpatch_torch_distributed()
            self.comm_tracker.close()
''',
        '''        if self.comm_tracker is not None and self._owns_comm_tracker:
            self.comm_tracker.unpatch_torch_distributed()
            self.comm_tracker.close()
''',
    ),
    (
        '''        for layer_name in self.layer_names:
            module_list = getattr(self, layer_name, None)
            if module_list is None or not isinstance(module_list, torch.nn.ModuleList):
                continue
''',
        '''        shared_comm_tracker = (
            CommunicationActivityTracker(window_mode=comm_active_window_mode)
            if comm_aware and torch.cuda.is_available()
            else None
        )
        for layer_name in self.layer_names:
            module_list = getattr(self, layer_name, None)
            if module_list is None or not isinstance(module_list, torch.nn.ModuleList):
                continue
''',
    ),
    (
        '''                comm_active_window_mode=comm_active_window_mode,
                prefetch_chunk_size_mb=prefetch_chunk_size_mb,
''',
        '''                comm_active_window_mode=comm_active_window_mode,
                comm_tracker=shared_comm_tracker,
                prefetch_chunk_size_mb=prefetch_chunk_size_mb,
''',
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"pattern not found exactly once:\n{old}"
    src = src.replace(old, new)
# the comm_tracker attribute must exist before _initialize() may consult it
assert src.index("self._owns_comm_tracker = False") < src.index("self._initialize()")
path.write_text(src)
print("patched:", path)
