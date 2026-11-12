#!/usr/bin/env bash
# ChunkFlow mechanism study on Pinnacles (2xL40S, Ulysses SP=2).
# One (model, config) point, a chosen list of arms, each wrapped in nsys.
#
# Usage: bash run_mech.sh <wan|flux|hunyuan> <cfg> [ARMS]
#   cfg   frames for wan/hunyuan, batch size (num outputs per prompt) for flux
#   ARMS  comma list from: warmup no old stock launch kernel ratio analyze
#         default: warmup,no,old,stock,launch,kernel,analyze
#     no      No Offload (all DiT weights resident)
#     old     Layerwise baseline as in the paper: whole-layer prefetch + wait-for-copy-stream before every collective
#     stock   same whole-layer prefetch, quiesce disabled (SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0) = upstream SGLang behaviour
#     launch  ChunkFlow, comm window = Python wrapper lifetime (the paper's numbers)
#     kernel  ChunkFlow, comm window = until the NCCL Work completes on the GPU
#     none    ChunkFlow chunking without any yield (window mode none) = chunked, no yield
#     gpu     ChunkFlow, GPU-timed window: active from the compute stream reaching the collective until the NCCL Work completes
#     auto    ChunkFlow, GPU-timed window + slack-aware decision: yield only if the pending prefetch fits the block's remaining compute time
#     ratio   ChunkFlow + partial residency (RESIDENT_RATIO, window RATIO_WINDOW)
# Env overrides: TAG NSYS=1 RESIDENT_RATIO=0.4 CHUNK_MB=16 NUM_GPUS=2 STEPS=10 RATIO_WINDOW=kernel ATTENTION_BACKEND=fa
set -uo pipefail

MODEL="${1:?usage: run_mech.sh <wan|flux|hunyuan> <cfg> [ARMS]}"
CFG="${2:?usage: run_mech.sh <wan|flux|hunyuan> <cfg> [ARMS]}"
ARMS="${3:-warmup,no,old,stock,launch,kernel,analyze}"
WS=/data/hanmeng/chunkflow
TAG="${TAG:-}"; NSYS="${NSYS:-1}"; RESIDENT_RATIO="${RESIDENT_RATIO:-0.4}"; CHUNK_MB="${CHUNK_MB:-16}"
NUM_GPUS="${NUM_GPUS:-2}"; STEPS="${STEPS:-10}"; RATIO_WINDOW="${RATIO_WINDOW:-kernel}"
ATTN="${ATTENTION_BACKEND:-fa}"

# Toolchain: do NOT `module load nvhpc` (it exports CC=nvc and puts a CUDA-less nvcc on PATH).
# nsys: a wrapper in $WS/bin; nvcc/libcudart for JIT builds (flashinfer, inductor): the nvhpc-bundled CUDA 12.9.
export CUDA_HOME=/opt/spack-v1.1.1/opt/spack/linux-cascadelake/nvhpc-25.7-kfehydowyjvjmze7hn7po2jkarxfml77/Linux_x86_64/25.7/cuda/12.9
export PATH="$WS/bin:$CUDA_HOME/bin:$PATH"
export CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST="8.9"
source "$WS/sglang/.venv/bin/activate"
export HF_HOME=/data/hanmeng/hf
export TORCHINDUCTOR_CACHE_DIR="$WS/.cache/torchinductor" TRITON_CACHE_DIR="$WS/.cache/triton"
export TMPDIR="/tmp/$USER/nsys-tmp"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export PYTHONUNBUFFERED=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1 SGLANG_DIFFUSION_STAGE_LOGGING=1
export SGLANG_WAN_MOCK_COMM_ENABLE=0
PROMPT="A cat walks on the grass, realistic"

case "$MODEL" in
    wan)
        MP="$WS/models/Wan2.2-TI2V-5B-Diffusers"
        MFLAGS=(--num-frames "$CFG" --height 704 --width 1280 --guidance-scale 4.0 --guidance-scale-2 3.0 --num-outputs-per-prompt 1) ;;
    hunyuan)
        MP="$WS/models/HunyuanVideo"
        MFLAGS=(--num-frames "$CFG" --height 720 --width 1280 --guidance-scale 1.0 --num-outputs-per-prompt 1) ;;
    flux)
        MP="$WS/models/FLUX.1-dev"
        MFLAGS=(--num-frames 1 --height 1024 --width 1024 --guidance-scale 1.0 --num-outputs-per-prompt "$CFG"
                --dit-cpu-offload true --vae-cpu-offload true)
        export SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING=1 ;;
    *) echo "ERROR: unknown model '$MODEL'"; exit 2 ;;
esac

RUN="${MODEL}_c${CFG}${TAG:+_$TAG}"
OUT="$WS/results/$RUN"; PROF="$OUT/profiles"; LOG="$OUT/logs"
mkdir -p "$PROF" "$LOG"
echo "== run_mech: $RUN arms=$ARMS gpus=$NUM_GPUS attn=$ATTN chunk=${CHUNK_MB}MB node=$(hostname) $(date) =="
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader | tee "$LOG/gpus.txt"
nvidia-smi topo -m > "$LOG/topo.txt" 2>&1 || true

# run_one <arm> <use_nsys 0|1> <port_offset>
run_one() {
    local arm="$1" use_nsys="$2" off="$3"
    local env_vars=() flags=()
    local offload=(--dit-layerwise-offload true --dit-offload-prefetch-size 1)
    local cf=(SGLANG_DIT_COMM_AWARE_OFFLOAD=1 SGLANG_DIT_PHASE_AWARE_PREFETCH=0
              SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB="$CHUNK_MB" SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST=0)
    case "$arm" in
        no)     env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0); flags+=(--dit-layerwise-offload false) ;;
        old)    env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0 SGLANG_DIT_OFFLOAD_COMM_QUIESCE=1); flags+=("${offload[@]}") ;;
        stock)  env_vars+=(SGLANG_DIT_COMM_AWARE_OFFLOAD=0 SGLANG_DIT_OFFLOAD_COMM_QUIESCE=0); flags+=("${offload[@]}") ;;
        launch|kernel|none|gpu|auto|gate)
                env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE="$arm" SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=)
                flags+=("${offload[@]}") ;;
        coarse) env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE=gate SGLANG_DIT_COMM_GATE_COARSE=1 SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=)
                flags+=("${offload[@]}") ;;
        paced)  env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE=none SGLANG_DIT_PREFETCH_PACE_GBPS="${PACE_GBPS:-16}" SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=)
                flags+=("${offload[@]}") ;;
        ratio)  env_vars+=("${cf[@]}" SGLANG_DIT_COMM_ACTIVE_WINDOW_MODE="$RATIO_WINDOW" SGLANG_DIT_OFFLOAD_RESIDENT_RATIO="$RESIDENT_RATIO")
                flags+=("${offload[@]}") ;;
        *) echo "ERROR: unknown arm '$arm'"; return 2 ;;
    esac
    local cmd=(sglang generate
        --model-path "$MP" --text-encoder-cpu-offload --pin-cpu-memory
        --num-gpus "$NUM_GPUS" --ulysses-degree "$NUM_GPUS" --attention-backend "$ATTN" --enable-torch-compile
        --prompt "$PROMPT" --num-inference-steps "$STEPS" "${MFLAGS[@]}"
        --master-port "$((29500 + off))" --port "$((30000 + off))" --scheduler-port "$((5700 + off))"
        --perf-dump-path "$PROF/perf_${arm}.json"
        --output-path "$OUT" --output-file-name "${arm}.out" --seed 42
        "${flags[@]}")
    if [[ "$use_nsys" == "1" ]]; then
        cmd=(nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --trace-fork-before-exec=true
             --kill=none --force-overwrite=true --output="$PROF/${arm}_nsys" "${cmd[@]}")
    fi
    echo "[$(date +%H:%M:%S)] ARM=$arm nsys=$use_nsys env: ${env_vars[*]}"
    rm -f "$PROF/perf_${arm}.json"
    local t0 rc=0; t0=$(date +%s)
    # a crashed sglang server can hang instead of exiting; bound every arm
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT:-1500}" env "${env_vars[@]}" "${cmd[@]}" > "$LOG/${arm}.log" 2>&1 || rc=$?
    if (( rc == 0 )) && [[ ! -f "$PROF/perf_${arm}.json" ]]; then rc=90; fi
    echo "[$(date +%H:%M:%S)] ARM=$arm rc=$rc wall=$(( $(date +%s) - t0 ))s"
    if (( rc != 0 )); then
        grep -m2 -iE "out of memory|OOM|Traceback|Error" "$LOG/${arm}.log" | head -2
        # keep the trace: a run that fails after denoising (e.g. VAE-decode OOM) still has valid step windows
        echo "$arm rc=$rc" >> "$PROF/failed_arms.txt"
    fi
    return "$rc"
}

IFS=',' read -r -a LIST <<<"$ARMS"
declare -A ST=()
off=0
for a in "${LIST[@]}"; do
    case "$a" in
        warmup)
            first=""
            for b in "${LIST[@]}"; do [[ "$b" != warmup && "$b" != analyze ]] && { first="$b"; break; }; done
            [[ -z "$first" ]] && continue
            run_one "$first" 0 "$((off += 7))"; ST[warmup]=$?
            mv "$LOG/${first}.log" "$LOG/warmup.log" 2>/dev/null || true
            rm -f "$PROF/perf_${first}.json"
            sleep 3 ;;
        analyze)
            bash "$WS/analyze_mech.sh" "$OUT" > "$LOG/analyze.log" 2>&1; ST[analyze]=$?
            [[ -f "$OUT/summary.md" ]] && cat "$OUT/summary.md" ;;
        *)
            run_one "$a" "$NSYS" "$((off += 7))"; ST[$a]=$?
            sleep 3 ;;
    esac
done
echo "== DONE $RUN $(date): $(for k in "${!ST[@]}"; do printf '%s=%s ' "$k" "${ST[$k]}"; done)"
