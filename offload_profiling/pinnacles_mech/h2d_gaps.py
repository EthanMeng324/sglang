#!/usr/bin/env python3
"""H2D prefetch stream duty cycle and inter-copy gaps in steady-state steps.
usage: h2d_gaps.py <trace.sqlite> [min_mb]"""
import sqlite3, statistics, sys

db = sqlite3.connect(sys.argv[1])
min_bytes = int(float(sys.argv[2]) * 1048576) if len(sys.argv) > 2 else 1048576
rows = db.execute("SELECT text,start,end FROM NVTX_EVENTS WHERE text GLOB 'SGL_DENOISING_STEP_*' AND end IS NOT NULL").fetchall()
steps = {}
for t, s, e in rows:
    i = int(t.split("_")[-1])
    if i >= 1:
        steps.setdefault(i, []).append((s, e))
windows = sorted((min(a for a, _ in v), max(b for _, b in v)) for v in steps.values())
g0, g1 = windows[0][0], windows[-1][1]
for dev, in db.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1 AND start<? AND end>? ORDER BY deviceId", (g1, g0)):
    cp = db.execute("SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=? AND copyKind=1 AND bytes>=? AND start<? AND end>? ORDER BY start", (dev, min_bytes, g1, g0)).fetchall()
    durs, gaps, sizes = [], [], []
    busy = 0
    for (ws, we) in windows:
        prev_end = None
        for s, e, b in cp:
            if e <= ws or s >= we:
                continue
            durs.append((e - s) / 1e6); sizes.append(b); busy += min(e, we) - max(s, ws)
            if prev_end is not None and s > prev_end:
                gaps.append((s - prev_end) / 1e6)
            prev_end = e
    span = sum(we - ws for ws, we in windows)
    n = len(durs)
    tot_gb = sum(sizes) / 1e9
    print(f"device {dev}: {n} copies, {tot_gb:.1f} GB in {len(windows)} steps; busy {busy/1e6/len(windows):.0f} ms/step = {100*busy/span:.0f}% of step;"
          f" eff bw while busy {tot_gb*1e9/ (busy/1e9) /1e9:.1f} GB/s; avg over step {tot_gb/(span/1e9):.1f} GB/s")
    if durs:
        print(f"  copy dur ms: p50 {statistics.median(durs):.2f} mean {statistics.mean(durs):.2f}; size MB p50 {statistics.median(sizes)/1048576:.1f}")
    if gaps:
        q = statistics.quantiles(gaps, n=10)
        print(f"  inter-copy gap ms: p50 {q[4]:.3f} p90 {q[8]:.2f} mean {statistics.mean(gaps):.3f}; total gap {sum(gaps)/len(windows):.0f} ms/step; gaps<0.2ms {100*sum(1 for g in gaps if g<0.2)/len(gaps):.0f}%")
