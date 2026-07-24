#!/usr/bin/env bash
# =============================================================================
# ChunkFlow rebuttal benchmark — single node (no slurm), WanVideo only.
#
# Runs the paper's mode matrix at one (SP degree, frame count) point, with all
# paper settings baked in. Produces the same artifact layout as the original
# pipeline so analyze_nsys.sh / summarize_profile_matrix.py work unchanged.
#
# Usage:
#   bash offload_profiling/run_rebuttal.sh <SP> <FRAMES> [MODES]
#
#     SP      number of GPUs = Ulysses SP degree: 2 or 4
#     FRAMES  video frames; Wan needs 4k+1 (41, 81, 121, 161, 201, ...)
#     MODES   comma list, default: warmup,no,old,new,ratio,analyze
#               warmup   first requested mode without nsys (torch.compile cache)
#               no       No Offload  (all DiT weights resident)
#               old      Layerwise   (SGLang whole-layer prefetch baseline)
#               new      ChunkFlow   (comm-aware chunked prefetch)
#               ratio    ChunkFlow + partial residency (RESIDENT_RATIO)
#               analyze  step-time + peak-memory summary (markdown/csv)
#
# A mode that fails (e.g. deliberate OOM probe of "no") does NOT abort the
# remaining modes; per-mode exit codes are reported at the end.
#
# Optional env overrides:
#   RESIDENT_RATIO=0.4    residency fraction for "ratio"
#   TAG=r2                suffix for the results dir (repeat runs)
#   NSYS=1                set 0 to skip nsys wrapping everywhere (no analyze)
#   CUDA_VISIBLE_DEVICES  SP=2 defaults to "0,1"; SP=4 uses all four
#
# Outputs:
#   /dev/shm/chunkflow_results/sp<SP>_f<FRAMES>[_TAG]/     traces + logs (RAM)
#   offload_profiling/results_5090/sp<SP>_f<FRAMES>[_TAG]/ json/md/csv/logs
# =============================================================================
set -euo pipefail

SP="${1:?usage: run_rebuttal.sh <SP:2|4> <FRAMES> [MODES]}"
FRAMES="${2:?usage: run_rebuttal.sh <SP:2|4> <FRAMES> [MODES]}"
MODES="${3:-warmup,no,old,new,analyze}"   # Figure 3 arms; pass "ratio" explicitly for residency runs
RESIDENT_RATIO="${RESIDENT_RATIO:-0.4}"
NSYS="${NSYS:-1}"
TAG="${TAG:-}"

# ---- fixed paper settings (Sec 4.1) -----------------------------------------
MODEL_PATH="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT="A cat walks on the grass, realistic"
HEIGHT=704 WIDTH=1280            # Wan2.2's 16:9 grid; 720 breaks patchify
NUM_INFERENCE_STEPS=10
GUIDANCE_SCALE=4.0 GUIDANCE_SCALE_2=3.0
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa}"   # paper runs did not use sageattention; fa -> Torch SDPA on sm_120
CHUNK_SIZE_MB=16                 # paper C = 16 MB
COMM_WINDOW_MODE=launch          # paper-era window mode (kernel = post-submission)

# ---- machine setup ----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
# Unconditional: the base image exports HF_HOME=/workspace/.hf_home in the ambient
# env, and the 32GB root disk cannot hold a second model copy.
export HF_HOME=/dev/shm/hf
# nsys writes multi-GB .qdstrm intermediates to TMPDIR; keep them off the tiny disk.
export TMPDIR=/dev/shm/nsys-tmp
mkdir -p "$TMPDIR"
# ... but /dev/shm is noexec: compiled inductor/triton .so files must stay on a
# filesystem that allows mapping executable pages.
export TORCHINDUCTOR_CACHE_DIR=/workspace/sglang/.cache/torchinductor
export TRITON_CACHE_DIR=/workspace/sglang/.cache/triton
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [[ "$SP" == "2" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
fi

RUN_TAG="sp${SP}_f${FRAMES}${TAG:+_${TAG}}"
RESULTS_DIR="/dev/shm/chunkflow_results/${RUN_TAG}"
PROF_DIR="$RESULTS_DIR/profiles/wanvideo"
LOG_DIR="$RESULTS_DIR/logs"
PERSIST_DIR="$SCRIPT_DIR/results_5090/${RUN_TAG}"
mkdir -p "$PROF_DIR" "$LOG_DIR" "$PERSIST_DIR"

export PYTHONUNBUFFERED=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_WAN_MOCK_COMM_ENABLE=0     # real NCCL collectives only
# Original (paper) config: untiled VAE decode. A no-offload OOM at large frame
# counts on 32GB cards is a *result* (offloading necessity), not a failure.
export SGLANG_WAN_VAE_TILING="${SGLANG_WAN_VAE_TILING:-0}"

# ---- one benchmark run ------------------------------------------------------
# run_one <mode: no|old|new|ratio> <use_nsys: 0|1> <port_offset>
run_one() {
    local mode="$1" use_nsys="$2" off="$3"
    local stem env_vars=() flags=()

    case "$mode" in
        no)    stem="no_offload"
               env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0)
               flags+=(--dit-layerwise-offload false) ;;
        old)   stem="old_offload"
               env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0)
               flags+=(--dit-layerwise-offload true --dit-offload-prefetch-size 1) ;;
        new|ratio)
               [[ "$mode" == new ]] && stem="new_offload" || stem="ratio_resident_offload"
               env_vars+=(
                   SGLANG_DIT_COMM_AWARE_OFFLOAD=1
                   SGLANG_DIT_PHASE_AWARE_PREFETCH=0
                   SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE="$COMM_WINDOW_MODE"
                   SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB="$CHUNK_SIZE_MB"
               )
               [[ "$mode" == ratio ]] && env_vars+=(SGLANG_DIT_OFFLOAD_RESIDENT_RATIO="$RESIDENT_RATIO")
               flags+=(--dit-layerwise-offload true --dit-offload-prefetch-size 1) ;;
        *)     echo "ERROR: unknown mode '$mode'"; return 2 ;;
    esac

    local cmd=(sglang generate
        --model-path "$MODEL_PATH"
        --text-encoder-cpu-offload
        --pin-cpu-memory
        --num-gpus "$SP"
        --ulysses-degree "$SP"
        --attention-backend "$ATTENTION_BACKEND"
        --enable-torch-compile
        --prompt "$PROMPT"
        --num-outputs-per-prompt 1
        --num-frames "$FRAMES"
        --height "$HEIGHT" --width "$WIDTH"
        --num-inference-steps "$NUM_INFERENCE_STEPS"
        --guidance-scale "$GUIDANCE_SCALE"
        --guidance-scale-2 "$GUIDANCE_SCALE_2"
        --master-port "$((29500 + off))"
        --port "$((30000 + off))"
        --scheduler-port "$((5700 + off))"
        --perf-dump-path "$PROF_DIR/perf_${stem}_profiled.json"
        --output-path "$RESULTS_DIR"
        --output-file-name "${stem}.mp4"
        --no-save-output
        "${flags[@]}"
    )
    if [[ "$use_nsys" == "1" ]]; then
        cmd=(nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true
             --trace-fork-before-exec=true --kill=none --force-overwrite=true
             --output="$PROF_DIR/${stem}_nsys" "${cmd[@]}")
    fi

    echo "------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] MODE=$mode nsys=$use_nsys  (log: $LOG_DIR/${mode}.log)"
    echo "  env : ${env_vars[*]}"
    echo "------------------------------------------------------------"
    rm -f "$PROF_DIR/perf_${stem}_profiled.json"   # stale JSON would defeat the success check
    local rc=0
    env "${env_vars[@]}" "${cmd[@]}" >"$LOG_DIR/${mode}.log" 2>&1 || rc=$?
    # The CLI can swallow generation failures (exit 0 with no perf dump);
    # treat a missing perf JSON as a failure.
    if (( rc == 0 )) && [[ ! -f "$PROF_DIR/perf_${stem}_profiled.json" ]]; then
        echo "!! mode '$mode': perf JSON missing despite exit 0 (generation failed internally?)"
        rc=90
    fi
    if (( rc != 0 )); then
        # Drop partial artifacts of a failed run so analyze only sees clean runs.
        rm -f "$PROF_DIR/${stem}_nsys.nsys-rep" "$PROF_DIR/perf_${stem}_profiled.json"
        grep -m1 -iE "out of memory|OOM" "$LOG_DIR/${mode}.log" || true
        return "$rc"
    fi
    return 0
}

# ---- drive the requested modes ----------------------------------------------
IFS=',' read -r -a MODE_LIST <<<"$MODES"
declare -A MODE_STATUS=()
port_off=0
first_bench_mode=""
for m in "${MODE_LIST[@]}"; do
    [[ "$m" == warmup || "$m" == analyze ]] && continue
    first_bench_mode="$m"; break
done

echo "== ChunkFlow rebuttal: SP=$SP frames=$FRAMES modes=$MODES =="
echo "== results: $RESULTS_DIR =="

for m in "${MODE_LIST[@]}"; do
    case "$m" in
        warmup)
            if [[ -z "$first_bench_mode" ]]; then continue; fi
            if run_one "$first_bench_mode" 0 "$((port_off += 7))"; then
                MODE_STATUS[warmup]=0
            else
                MODE_STATUS[warmup]=$?
            fi
            mv "$LOG_DIR/${first_bench_mode}.log" "$LOG_DIR/warmup.log" 2>/dev/null || true
            sleep 5
            ;;
        no|old|new|ratio)
            if run_one "$m" "$NSYS" "$((port_off += 7))"; then
                MODE_STATUS[$m]=0
            else
                MODE_STATUS[$m]=$?
                echo "!! mode '$m' FAILED (exit ${MODE_STATUS[$m]}) — continuing; see $LOG_DIR/${m}.log"
            fi
            sleep 5
            ;;
        analyze)
            if [[ "$NSYS" == "1" ]]; then
                if RESULTS_DIR="$RESULTS_DIR" PROFILES_DIR="$RESULTS_DIR/profiles" \
                    bash "$SCRIPT_DIR/analyze_nsys.sh" wanvideo >"$LOG_DIR/analyze.log" 2>&1; then
                    MODE_STATUS[analyze]=0
                else
                    MODE_STATUS[analyze]=$?
                    echo "!! analyze FAILED — see $LOG_DIR/analyze.log"
                fi
            else
                echo "(NSYS=0 — skipping analyze)"
            fi
            ;;
        *) echo "ERROR: unknown mode '$m'"; exit 2 ;;
    esac
done

# ---- persist small artifacts + report ---------------------------------------
(cd "$RESULTS_DIR" && find . \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.log' \) \
    -exec cp --parents --no-preserve=mode -t "$PERSIST_DIR" {} + 2>/dev/null) || true

echo ""
echo "== DONE sp${SP} f${FRAMES} =="
overall=0
for m in "${!MODE_STATUS[@]}"; do
    echo "  $m: exit ${MODE_STATUS[$m]}"
    [[ "${MODE_STATUS[$m]}" != 0 && "$m" != warmup ]] && overall=1
done
summary="$PROF_DIR/analysis/analysis_summary_matrix.md"
[[ -f "$summary" ]] && { echo ""; cat "$summary"; }
echo ""
echo "persisted: $PERSIST_DIR"
exit "$overall"
