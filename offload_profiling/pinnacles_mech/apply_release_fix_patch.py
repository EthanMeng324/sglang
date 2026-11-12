#!/usr/bin/env python3
"""Stream-safe release of a layer's GPU weight buffers in the chunk-wise offload manager.

Bug (2026-11-12, found by black/noise outputs of the gate arm on H100): the post-forward hook
releases a layer's GPU buffers as soon as the layer's forward has been *enqueued*; the caching
allocator hands the same blocks to the next prefetched layer and the copy stream overwrites them
while the compute stream may still be executing the released layer. The manager already has a
deferred-release path (release event recorded on the compute stream, tensors kept alive until it
completes) but enabled it only for the bucketed phase mode. This patch turns it on for every
mode (opt out with SGLANG_DIT_OFFLOAD_IMMEDIATE_RELEASE=1).

usage: apply_release_fix_patch.py [repo_root]
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = root / "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py"
MARKER = "SGLANG_DIT_OFFLOAD_IMMEDIATE_RELEASE"
OLD = '''        self._use_record_stream_protection = self._auto_bucket_phases
        self._use_deferred_buffer_release = bool(
            self._auto_bucket_phases
            and _env_bool("SGLANG_DIT_OFFLOAD_DEFER_BUCKET_RELEASE", False)
        )
'''
NEW = '''        self._use_record_stream_protection = self._auto_bucket_phases
        # A layer's GPU buffers are released in the post-forward hook, i.e. when the layer's
        # forward has been enqueued, not executed. Keeping them alive until an event recorded
        # on the compute stream at that point completes stops the caching allocator from
        # handing the blocks to the next prefetched layer while the GPU may still be reading
        # them (otherwise the copy stream overwrites weights in use: black/noise outputs).
        self._use_deferred_buffer_release = not _env_bool(MARKER_PLACEHOLDER, False)
'''.replace("MARKER_PLACEHOLDER", f'"{MARKER}"')

src = path.read_text()
if MARKER in src:
    print("already patched:", path)
    sys.exit(0)
if src.count(OLD) != 1:
    print(f"expected exactly one match of the release block in {path}, found {src.count(OLD)}")
    sys.exit(1)
path.write_text(src.replace(OLD, NEW))
print("patched:", path)
