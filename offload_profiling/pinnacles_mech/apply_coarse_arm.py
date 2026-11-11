#!/usr/bin/env python3
"""Add the harness arm `coarse` (GPU-side gate held across each attention's communication phase) to run_mech.sh
(workspace copy and repo copy). usage: apply_coarse_arm.py <workspace_root>"""
import pathlib
import sys

ws = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
old = "        paced)  env_vars+=("
new = ('        coarse) env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE=gate SGLANG_DIT_COMM_GATE_COARSE=1 SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=)\n'
       '                flags+=("${offload[@]}") ;;\n' + old)
for rel in ("run_mech.sh", "sglang/offload_profiling/pinnacles_mech/run_mech.sh"):
    p = ws / rel
    s = p.read_text()
    if "coarse)" in s:
        print("already has coarse:", p)
        continue
    assert s.count(old) == 1, f"{p}: anchor not found exactly once"
    p.write_text(s.replace(old, new))
    print("patched:", p)
