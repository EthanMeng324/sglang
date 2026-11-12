#!/usr/bin/env python3
"""Per-kernel-name summary of device 0 inside SGL_DENOISING_STEP_<k> for each nsys sqlite export.
usage: python kernels.py <step> no.sqlite gate.sqlite  -> markdown on stdout"""
import sqlite3, sys
step = sys.argv[1]; dbs = sys.argv[2:]
res = {}
for db in dbs:
    c = sqlite3.connect(db)
    wins = c.execute("select start,end from NVTX_EVENTS where text=?", (f"SGL_DENOISING_STEP_{step}",)).fetchall()
    best = None
    for s, e in wins:  # one range per rank process; keep the one that holds device-0 kernels
        n = c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0 and start>=? and end<=?", (s, e)).fetchone()[0]
        if best is None or n > best[2]: best = (s, e, n)
    s, e, _ = best
    rows = c.execute("""select st.value, count(*), sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds st on st.id=k.demangledName
                        where k.deviceId=0 and k.start>=? and k.end<=? group by st.value order by 3 desc""", (s, e)).fetchall()
    tot = c.execute("select sum(end-start)/1e6, count(*) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0 and start>=? and end<=?", (s, e)).fetchone()
    res[db] = dict(step_ms=(e - s) / 1e6, kern_ms=tot[0], kern_n=tot[1], rows=rows, windows=len(wins))
names = [d.split("/")[-1].split("_nsys")[0] for d in dbs]
print(f"## step {step}, device 0\n\n| arm | step window ms | kernel total ms | kernel count | NVTX windows found |\n|---|---|---|---|---|")
for n, d in zip(names, dbs): r = res[d]; print(f"| {n} | {r['step_ms']:.2f} | {r['kern_ms']:.2f} | {r['kern_n']} | {r['windows']} |")
print("\n### top 25 kernels per arm (side by side, by total ms)\n")
print("| rank | " + " | ".join(f"{n} kernel | n | ms" for n in names) + " |"); print("|---|" + "---|---|---|" * len(names))
for i in range(25):
    cells = []
    for d in dbs:
        r = res[d]["rows"]
        cells.append(f"`{r[i][0][:90]}` | {r[i][1]} | {r[i][2]:.3f}" if i < len(r) else " | | ")
    print(f"| {i+1} | " + " | ".join(cells) + " |")
print("\n### same kernel name in both arms (top 40 by ms in first arm): count and ms\n")
m = [{k: (n, ms) for k, n, ms in res[d]["rows"]} for d in dbs]
print("| kernel | " + " | ".join(f"{n} n | {n} ms" for n in names) + " |"); print("|---|" + "---|---|" * len(names))
for k, _, _ in res[dbs[0]]["rows"][:40]:
    print(f"| `{k[:90]}` | " + " | ".join(f"{mm[k][0]} | {mm[k][1]:.3f}" if k in mm else "- | -" for mm in m) + " |")
only = [(i, sorted(set(m[i]) - set(m[1 - i]))) for i in range(2)] if len(dbs) == 2 else []
for i, ks in only:
    print(f"\n### kernels only in {names[i]} ({len(ks)}): " + ", ".join(f"`{k[:70]}` ({m[i][k][0]}x, {m[i][k][1]:.3f} ms)" for k in sorted(ks, key=lambda k: -m[i][k][1])[:20]))
