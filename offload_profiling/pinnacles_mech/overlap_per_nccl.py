#!/usr/bin/env python3
"""Per-NCCL-kernel view of H2D overlap in one nsys sqlite export.

For every ncclDevKernel in steady-state steps (>=1) on each device: how many prefetch
H2D memcpys overlap it, how much of the kernel is overlapped, and where the overlapping
copies start relative to the kernel start. usage: overlap_per_nccl.py <trace.sqlite>
"""
import sqlite3, statistics, sys
from pathlib import Path

db = sqlite3.connect(sys.argv[1])
rows = db.execute("SELECT text,start,end FROM NVTX_EVENTS WHERE text GLOB 'SGL_DENOISING_STEP_*' AND end IS NOT NULL").fetchall()
steps = {}
for t, s, e in rows:
    i = int(t.split("_")[-1])
    if i >= 1:
        steps.setdefault(i, []).append((s, e))
windows = [(min(a for a, _ in v), max(b for _, b in v)) for v in steps.values()]
g0, g1 = min(w[0] for w in windows), max(w[1] for w in windows)
devs = [r[0] for r in db.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? AND end>? ORDER BY deviceId", (g1, g0))]
for dev in devs:
    nccl = db.execute("""SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
                          WHERE k.deviceId=? AND k.start<? AND k.end>? AND s.value LIKE '%ncclDevKernel_SendRecv%' ORDER BY k.start""", (dev, g1, g0)).fetchall()
    h2d = db.execute("""SELECT start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=? AND copyKind=1 AND bytes>=1048576
                         AND start<? AND end>? ORDER BY start""", (dev, g1, g0)).fetchall()
    j = 0
    counts, fracs, lags = [], [], []
    tot_ov = 0
    for ks, ke in nccl:
        if not any(ws <= ks and ke <= we for ws, we in windows):
            continue
        while j < len(h2d) and h2d[j][1] <= ks:
            j += 1
        k = j; n = 0; ov = 0
        while k < len(h2d) and h2d[k][0] < ke:
            cs, ce, _ = h2d[k]
            o = min(ce, ke) - max(cs, ks)
            if o > 0:
                n += 1; ov += o; lags.append((cs - ks) / 1e6)
            k += 1
        counts.append(n); fracs.append(ov / (ke - ks)); tot_ov += ov
    nk = len(counts)
    hist = {c: counts.count(c) for c in sorted(set(counts))}
    print(f"device {dev}: {nk} NCCL kernels in steady steps; overlap total {tot_ov/1e6/len(windows):.1f} ms/step;"
          f" overlapping chunks per kernel: mean {statistics.mean(counts):.2f}, hist {hist}")
    print(f"  overlapped fraction of kernel: mean {statistics.mean(fracs):.2f}, median {statistics.median(fracs):.2f}")
    if lags:
        q = statistics.quantiles(lags, n=10)
        print(f"  chunk start - kernel start (ms): p10 {q[0]:.2f} p50 {q[4]:.2f} p90 {q[8]:.2f}  (negative = chunk already in flight)")
    kd = [(ke - ks) / 1e6 for ks, ke in nccl]
    print(f"  NCCL kernel duration ms: median {statistics.median(kd):.2f}, mean {statistics.mean(kd):.2f}")
