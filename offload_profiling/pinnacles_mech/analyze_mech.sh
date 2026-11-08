#!/usr/bin/env bash
# Summarize one run_mech.sh result directory: step time, NCCL/H2D metrics (nsys) and peak memory (perf JSON).
# usage: bash analyze_mech.sh <results_dir>
set -uo pipefail
OUT="${1:?usage: analyze_mech.sh <results_dir>}"
WS=/data/hanmeng/chunkflow
# Toolchain: do NOT `module load nvhpc` (it exports CC=nvc and puts a CUDA-less nvcc on PATH).
# nsys: a wrapper in $WS/bin; nvcc/libcudart for JIT builds (flashinfer, inductor): the nvhpc-bundled CUDA 12.9.
export CUDA_HOME=/opt/spack-v1.1.1/opt/spack/linux-cascadelake/nvhpc-25.7-kfehydowyjvjmze7hn7po2jkarxfml77/Linux_x86_64/25.7/cuda/12.9
export PATH="$WS/bin:$CUDA_HOME/bin:$PATH"
export CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST="8.9"
source "$WS/sglang/.venv/bin/activate"
PROF="$OUT/profiles"
reps=( "$PROF"/*_nsys.nsys-rep )
if [[ -e "${reps[0]}" ]]; then
    python "$WS/sglang/offload_profiling/analyze_nsys_step_metrics.py" "${reps[@]}" \
        --output "$OUT/step_metrics.md" --export-dir "$PROF/sqlite_cache" 2>&1 | tail -3
fi
python - "$OUT" <<'PY'
import json, re, sys, pathlib, statistics
out = pathlib.Path(sys.argv[1]); prof = out / "profiles"
metrics = {}
md = out / "step_metrics.md"
if md.exists():
    for line in md.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| Trace") or line.startswith("| ---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        arm = cells[0].replace("_nsys", "")
        try:
            metrics[arm] = dict(step=float(cells[1]), nccl=float(cells[2]), overlap=float(cells[3]),
                                uncovered=float(cells[4]), wait_bg=float(cells[5]))
        except ValueError:
            continue
def peak(perf):
    pr = pa = None
    cps = perf.get("memory_checkpoints") or {}
    for key in ("mem_analysis", "after_forward", "before_forward"):
        v = cps.get(key) if isinstance(cps, dict) else None
        if isinstance(v, dict):
            if pr is None and isinstance(v.get("peak_reserved_mb"), (int, float)): pr = float(v["peak_reserved_mb"])
            if pa is None and isinstance(v.get("peak_allocated_mb"), (int, float)): pa = float(v["peak_allocated_mb"])
    if pr is None and isinstance(perf.get("peak_memory_mb"), (int, float)): pr = float(perf["peak_memory_mb"])
    return pr, pa
def step_from_perf(perf):
    # fall back to the perf dump's per-step times when nsys is off
    for key in ("denoising_step_times_ms", "step_times_ms"):
        v = perf.get(key)
        if isinstance(v, list) and len(v) > 1:
            return statistics.mean(v[1:])
    return None
failed = {}
ff = prof / "failed_arms.txt"
if ff.exists():
    for line in ff.read_text().splitlines():
        if line.strip():
            failed[line.split()[0]] = line.strip()
arms = set(metrics) | {pj.stem[len("perf_"):] for pj in prof.glob("perf_*.json")}
rows = []
for arm in sorted(arms):
    pj = prof / f"perf_{arm}.json"
    perf = {}
    if pj.exists():
        try:
            perf = json.loads(pj.read_text())
        except Exception:
            perf = {}
    pr, pa = peak(perf)
    m = metrics.get(arm, {})
    step = m.get("step") or step_from_perf(perf)
    label = arm + (" (failed after denoising)" if arm in failed and step else (" (failed)" if arm in failed else ""))
    rows.append((label, step, m.get("nccl"), m.get("overlap"), m.get("uncovered"), m.get("wait_bg"), pr, pa,
                 perf.get("total_duration_ms")))
order = {"no": 0, "old": 1, "stock": 2, "none": 3, "launch": 4, "kernel": 5, "gpu": 6, "auto": 7, "ratio": 8}
rows.sort(key=lambda r: order.get(r[0].split()[0], 9))
no_step = next((r[1] for r in rows if r[0].split()[0] == "no" and r[1]), None)
f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
lines = [f"# {out.name}", "",
         "| arm | step ms | vs no | NCCL ms | NCCL/H2D overlap ms | uncovered H2D ms | WAIT_COMM_BG ms | peak reserved GB | peak alloc GB | e2e s |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for arm, step, nccl, ov, unc, wb, pr, pa, e2e in rows:
    vs = "-" if not (step and no_step) else f"{100*(step/no_step-1):+.1f}%"
    lines.append(f"| {arm} | {f(step)} | {vs} | {f(nccl)} | {f(ov)} | {f(unc)} | {f(wb)} | "
                 f"{f(pr/1024 if pr else None, 2)} | {f(pa/1024 if pa else None, 2)} | {f(e2e/1000 if e2e else None, 2)} |")
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
