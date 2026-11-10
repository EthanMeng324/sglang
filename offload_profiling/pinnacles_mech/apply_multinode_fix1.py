#!/usr/bin/env python3
"""Multi-node fix 1: the CUDA device index must be the local device, not the global rank.

platforms/cuda.py get_available_gpu_memory() used torch.distributed.get_rank() as the device id; on a
worker-only node the global ranks exceed the local GPU count ("Invalid device id" while loading the
text encoder). The worker made its local device current before loading, so use that.
usage: apply_multinode_fix1.py [repo_root]"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
path = root / "python/sglang/multimodal_gen/runtime/platforms/cuda.py"
src = path.read_text()
old = '''        if torch.distributed.is_initialized():
            device_id = torch.distributed.get_rank()

        device_props = torch.cuda.get_device_properties(device_id)
'''
new = '''        if torch.distributed.is_initialized():
            # the CUDA device index is the LOCAL device (multi-node: global ranks beyond
            # gpus_per_node live on other nodes); the worker made its device current
            device_id = torch.cuda.current_device()

        device_props = torch.cuda.get_device_properties(device_id)
'''
if new in src:
    print("already patched:", path)
elif src.count(old) == 1:
    path.write_text(src.replace(old, new))
    print("patched:", path)
else:
    raise SystemExit(f"pattern not found exactly once in {path}")
