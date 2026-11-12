#!/usr/bin/env python3
"""Chunk-wise offload manager: take GPU weight buffers from the copy stream's allocator pool and
register the compute stream's use of them.

Root cause of the wrong outputs of every chunk-wise arm (unit test offload_unit_test.py: the data
path is bit-exact, the hooked forward is not): the per-layer GPU weight buffers were allocated with
torch.empty on the worker thread's current stream (the default/compute stream), so the caching
allocator handed them blocks that the forward had just freed (activations) and whose kernels were
still pending; the copy stream then wrote weights into memory the compute stream was still using,
and pending compute wrote into memory that had become weights. Allocating under the copy stream
keeps weight buffers in the copy stream's pool, and record_stream(compute stream) makes the
allocator wait for the compute stream before reusing a freed weight buffer.

usage: apply_alloc_stream_patch.py [repo_root]
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = root / "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_chunkwise.py"
MARKER = "allocated on the copy stream: its own allocator pool"

OLD_ALLOC = '''                if phase_idx not in self._prefetch_gpu_buffers[layer_idx]:
                    cpu_phase_buffers = self._consolidated_cpu_weights[layer_idx].get(
                        phase_idx, {}
                    )
                    self._prefetch_gpu_buffers[layer_idx][phase_idx] = {
                        dtype: torch.empty(
                            cpu_buffer.shape,
                            dtype=dtype,
                            device=self.device,
                        )
                        for dtype, cpu_buffer in cpu_phase_buffers.items()
                    }
'''
NEW_ALLOC = '''                if phase_idx not in self._prefetch_gpu_buffers[layer_idx]:
                    cpu_phase_buffers = self._consolidated_cpu_weights[layer_idx].get(
                        phase_idx, {}
                    )
                    # Weight buffers are allocated on the copy stream: its own allocator pool,
                    # so a block the forward just freed (an activation with kernels still
                    # pending on the compute stream) is never handed to a weight buffer that
                    # the copy stream fills right away.
                    with torch.cuda.stream(self.copy_stream):
                        self._prefetch_gpu_buffers[layer_idx][phase_idx] = {
                            dtype: torch.empty(
                                cpu_buffer.shape,
                                dtype=dtype,
                                device=self.device,
                            )
                            for dtype, cpu_buffer in cpu_phase_buffers.items()
                        }
'''
OLD_RS = "        self._use_record_stream_protection = self._auto_bucket_phases\n"
NEW_RS = ("        # Every materialized buffer is read by the compute stream: record that use so the\n"
          "        # allocator waits for it before the freed block is reused (any mode).\n"
          "        self._use_record_stream_protection = True\n")

src = path.read_text()
if MARKER in src:
    print("already patched:", path)
    sys.exit(0)
for old, what in ((OLD_ALLOC, "allocation block"), (OLD_RS, "record_stream flag")):
    if src.count(old) != 1:
        print(f"expected exactly one match of the {what}, found {src.count(old)}")
        sys.exit(1)
src = src.replace(OLD_ALLOC, NEW_ALLOC).replace(OLD_RS, NEW_RS)
path.write_text(src)
print("patched:", path)
