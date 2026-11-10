#!/usr/bin/env python3
"""Guard the Pinnacles-specific bootstrap-interface defaults: only default to en0 / mlx5_0 when those devices exist
on this node, so the same script runs unchanged on other clusters. usage: apply_mn_ifname_guard.py <workspace_root>"""
import pathlib
import sys

ws = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
old = ('    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}" NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"\n')
new = ('    if [[ -z "${NCCL_SOCKET_IFNAME:-}" && -d /sys/class/net/en0 ]]; then export NCCL_SOCKET_IFNAME=en0 GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}"; fi\n'
       '    if [[ -z "${NCCL_IB_HCA:-}" && -d /sys/class/infiniband/mlx5_0 ]]; then export NCCL_IB_HCA=mlx5_0; fi\n')
for rel in ("run_mn_arm.sh", "sglang/offload_profiling/pinnacles_mech/run_mn_arm.sh"):
    p = ws / rel
    s = p.read_text()
    if "/sys/class/net/en0" in s:
        print("already guarded:", p)
        continue
    assert s.count(old) == 1, f"{p}: pattern not found exactly once"
    p.write_text(s.replace(old, new))
    print("guarded:", p)
