#!/usr/bin/env python3
"""Pinnacles: the IPoIB (ib0) addresses are not routable between nodes, so the NCCL/gloo bootstrap must use
the Ethernet interface (en0); the data path still uses the IB HCA (verbs). Patches run_mn_arm.sh and sb_mn.sh
(both the workspace copies and the repo copies). usage: apply_mn_ifname_patch.py <workspace_root>"""
import pathlib
import sys

ws = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
arm_old = 'export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"\n'
arm_new = (
    'export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"\n'
    'if [[ "$NNODES" -gt 1 ]]; then\n'
    '    # Pinnacles: the IPoIB (ib0) addresses are not routable between nodes, so the NCCL/gloo bootstrap\n'
    '    # must use the Ethernet interface; the data path still uses the IB HCA (verbs).\n'
    '    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-en0}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-en0}" NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}"\n'
    'fi\n'
)
bw_old = "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET torchrun"
bw_new = ("NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-en0} "
          "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-en0} NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0} torchrun")
for rel, old, new in [
    ("run_mn_arm.sh", arm_old, arm_new),
    ("sglang/offload_profiling/pinnacles_mech/run_mn_arm.sh", arm_old, arm_new),
    ("sb_mn.sh", bw_old, bw_new),
    ("sglang/offload_profiling/pinnacles_mech/sb_mn.sh", bw_old, bw_new),
]:
    p = ws / rel
    s = p.read_text()
    if "NCCL_SOCKET_IFNAME:-en0" in s:
        print("already patched:", p)
        continue
    assert s.count(old) == 1, f"{p}: pattern not found exactly once"
    p.write_text(s.replace(old, new))
    print("patched:", p)
