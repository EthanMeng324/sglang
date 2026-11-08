#!/usr/bin/env python3
"""Add SGLANG_DIT_OFFLOAD_COMM_QUIESCE to the whole-layer (baseline) offload manager.

With the variable set to 0 the baseline no longer makes the compute stream wait for the
copy stream before every collective, i.e. it behaves like upstream SGLang's layerwise
offload ("stock" arm). Default (unset / 1) keeps the paper's baseline behaviour.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "python/sglang/multimodal_gen/runtime/utils/layerwise_offload_original.py"
)
src = path.read_text()

old_fn = '''    @torch.compiler.disable
    def quiesce_copy_stream_for_comm(self, after_drain=None) -> None:
        if not self.enabled or self.copy_stream is None:
            if after_drain is not None:
                after_drain()
            return
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        if after_drain is not None:
            after_drain()
'''
new_fn = '''    @torch.compiler.disable
    def quiesce_copy_stream_for_comm(self, after_drain=None) -> None:
        if not self.enabled or self.copy_stream is None or not _COMM_QUIESCE_ENABLED:
            if after_drain is not None:
                after_drain()
            return
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        if after_drain is not None:
            after_drain()
'''
flag = '''
# SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0 disables the wait-for-copy-stream before every
# collective, which upstream SGLang does not have ("stock" whole-layer offload).
_COMM_QUIESCE_ENABLED = os.getenv("SGLANG_DIT_OFFLOAD_COMM_QUIESCE", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
'''

if "_COMM_QUIESCE_ENABLED" in src:
    print("already patched:", path)
    sys.exit(0)
assert src.count(old_fn) == 1, "quiesce_copy_stream_for_comm not found exactly once"
src = src.replace(old_fn, new_fn)
if "\nimport os\n" not in src:
    src = src.replace("import torch\n", "import os\n\nimport torch\n", 1)
    assert "\nimport os\n" in src
# place the flag after the import block (before the first top-level def/class)
marker = "\nlogger = "
if marker in src:
    src = src.replace(marker, flag + marker, 1)
else:
    first_def = min(i for i in (src.find("\nclass "), src.find("\ndef ")) if i >= 0)
    src = src[:first_def] + flag + src[first_def:]
path.write_text(src)
print("patched:", path)
