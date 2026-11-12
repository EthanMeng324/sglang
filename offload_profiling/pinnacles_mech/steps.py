#!/usr/bin/env python3
"""Per-step denoising times from perf_<arm>.json files: usage steps.py <run_dir> <arm>..."""
import json
import sys

d = sys.argv[1]
for a in sys.argv[2:]:
    p = f"{d}/profiles/perf_{a}.json"
    try:
        j = json.load(open(p))
        st = [s["duration_ms"] for s in j["denoise_steps_ms"]]
        m8 = sum(st[1:9]) / 8
        m9 = sum(st[1:]) / 9
        print(f"{a:6s} steps1-8 {m8:7.0f}  1-9 {m9:7.0f}  steps: " + " ".join(f"{x:.0f}" for x in st))
    except Exception as e:  # noqa: BLE001
        print(a, "no perf:", e)
