#!/usr/bin/env python3
"""Mean SM clock (pclk) and power per GPU over the steady steps 1-9 of one arm.
Window = [t_end - sum(steps 1..9), t_end], t_end = timestamp of the rank-0 '[DenoisingStage] finished in' line
(1 s log resolution). dmon was started with 'nvidia-smi dmon -s pc -d 1 -o T'.
usage: python dmon_avg.py <OUT dir> <arm>"""
import json, re, sys, datetime as dt
out, arm = sys.argv[1], sys.argv[2]
p = json.load(open(f"{out}/profiles/perf_{arm}.json"))
st = sorted(p["denoise_steps_ms"], key=lambda s: s["step"]); win = sum(s["duration_ms"] for s in st[1:10]) / 1000
log = open(f"{out}/logs/{arm}.log", errors="ignore").read()
m = re.findall(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] \[DenoisingStage\] finished in", log)
t_end = dt.datetime.strptime(m[-1], "%m-%d %H:%M:%S").time()
def sec(t): return t.hour * 3600 + t.minute * 60 + t.second
e = sec(t_end); s = e - win
hdr, rows = None, []
for line in open(f"{out}/logs/dmon_{arm}.txt"):
    f = line.split()
    if line.startswith("#"):
        if line.startswith("#Time"): hdr = [f[0].lstrip("#")] + f[1:]
        continue
    if hdr and len(f) == len(hdr): rows.append(dict(zip(hdr, f)))
sel = [r for r in rows if s - 0.5 <= sec(dt.datetime.strptime(r["Time"], "%H:%M:%S").time()) <= e + 0.5]
for g in sorted({r["gpu"] for r in sel}):
    rr = [r for r in sel if r["gpu"] == g]
    num = lambda k: [float(r[k]) for r in rr if r.get(k, "-") not in ("-", "")]
    pclk, pwr, mclk = num("pclk"), num("pwr"), num("mclk")
    print(f"{arm} gpu{g}: window {s:.0f}-{e}s ({win:.1f}s, {len(rr)} samples) pclk_MHz={sum(pclk)/len(pclk):.0f} pwr_W={sum(pwr)/len(pwr):.0f} mclk_MHz={sum(mclk)/len(mclk):.0f}")
