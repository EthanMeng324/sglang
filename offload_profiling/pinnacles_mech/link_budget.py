#!/usr/bin/env python3
"""Per-step PCIe link budget from an nsys sqlite export of one arm.

usage: link_budget.py <run_dir> [step=3] [arms=all]
Reads <run_dir>/profiles/sqlite_cache/<arm>_nsys_summary_export.sqlite and prints, per device:
step ms, NCCL kernel ms, H2D GB / ms / GB/s, H2D bytes that fall inside NCCL kernel windows (the leak),
H2D copies, and H2D time outside NCCL windows."""
import pathlib
import sqlite3
import sys


def budget(db_path: pathlib.Path, step: int):
    c = sqlite3.connect(str(db_path)).cursor()
    c.execute("select id, value from StringIds")
    S = dict(c.fetchall())
    nccl_ids = [i for i, v in S.items() if "nccl" in v.lower()]
    c.execute("select start, end from NVTX_EVENTS where text=? order by start", (f"SGL_DENOISING_STEP_{step}",))
    rows = c.fetchall()
    if not rows:
        return None
    s, e = rows[0]
    out = []
    for dev in (0, 1):
        q = ",".join(str(i) for i in nccl_ids) or "-1"
        c.execute(f"select start, end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and start>=? and end<=? and demangledName in ({q}) order by start", (dev, s, e))
        nk = c.fetchall()
        if not nk:
            continue
        nccl = sum(b - a for a, b in nk) / 1e6
        c.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and copyKind=1 and start>=? and end<=? order by start", (dev, s, e))
        h = c.fetchall()
        gb = sum(x[2] for x in h) / 1e9
        ms = sum(b - a for a, b, _ in h) / 1e6
        gb_in = 0.0
        out_ms = 0.0
        j = 0
        for a, b, by in h:
            ov = 0
            while j > 0 and nk[j - 1][1] > a:
                j -= 1
            k = j
            while k < len(nk) and nk[k][0] < b:
                ka, kb = nk[k]
                if kb > a:
                    ov += min(b, kb) - max(a, ka)
                k += 1
            out_ms += (b - a - ov)
            gb_in += by / 1e9 * (ov / (b - a) if b > a else 0)
        out.append((dev, (e - s) / 1e6, nccl, len(nk), gb, ms, (gb / ms * 1e3 if ms else 0.0), gb_in, len(h), out_ms / 1e6))
    return out


def main():
    run = pathlib.Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    cache = run / "profiles" / "sqlite_cache"
    print(f"# {run.name} step {step}")
    print(f"{'arm':8s} {'dev':>3s} {'step_ms':>8s} {'nccl_ms':>8s} {'n_nccl':>6s} {'h2d_GB':>7s} {'h2d_ms':>7s} {'h2d_GBps':>8s} {'h2d_in_nccl_GB':>14s} {'n_h2d':>6s} {'h2d_outside_ms':>14s}")
    for db in sorted(cache.glob("*_nsys_summary_export.sqlite")):
        arm = db.name[: -len("_nsys_summary_export.sqlite")]
        if arms and arm not in arms:
            continue
        res = budget(db, step)
        if not res:
            print(f"{arm:8s} (no step {step})")
            continue
        for dev, st, nccl, nn, gb, ms, bw, gb_in, nh, out_ms in res:
            print(f"{arm:8s} {dev:3d} {st:8.0f} {nccl:8.0f} {nn:6d} {gb:7.2f} {ms:7.0f} {bw:8.1f} {gb_in:14.2f} {nh:6d} {out_ms:14.0f}")


if __name__ == "__main__":
    main()
