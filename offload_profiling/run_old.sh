#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"

mkdir -p /tmp/data/tmp
chmod 777 /tmp/data/tmp
export TMPDIR=/tmp/data/tmp
export TEMP=/tmp/data/tmp
export TMP=/tmp/data/tmp

mkdir -p "$RESULTS_DIR" "$PROFILES_DIR"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$REPO_ROOT"

if ! command -v sglang >/dev/null 2>&1; then
    echo "ERROR: sglang not found in PATH (activate env first)."
    exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not found in PATH."
    exit 1
fi

# GPU info
nvidia-smi
echo ""

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
echo ""

MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
PROMPT="${PROMPT:-A cat walks on the grass, realistic}"
NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-9}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-4.0}"
NUM_GPUS="${NUM_GPUS:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage_attn}"
NSYS_OUTPUT_PREFIX="${NSYS_OUTPUT_PREFIX:-$PROFILES_DIR/old_offload_nsys}"
VIDEO_OUT="${VIDEO_OUT:-$RESULTS_DIR/nsys_old_offload_profiled.mp4}"
PERF_OUT="${PERF_OUT:-$PROFILES_DIR/perf_old_offload_profiled.json}"

# Common generation flags
COMMON_FLAGS=(
    --model-path "$MODEL_PATH"
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --dit-layerwise-offload true
    --dit-offload-prefetch-size 1
    --num-gpus "$NUM_GPUS"
    --ulysses-degree "$ULYSSES_DEGREE"
    --attention-backend "$ATTENTION_BACKEND"
    --prompt "$PROMPT"
    --num-frames "$NUM_FRAMES"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
    --guidance-scale-2 "$GUIDANCE_SCALE_2"
)

# if [[ "${SGLANG_ENABLE_TORCH_COMPILE:-0}" == "1" ]]; then
    COMMON_FLAGS+=(--enable-torch-compile)
# fi

export PYTHONUNBUFFERED=1

# Enable sync profiling for accurate stage timing
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1

# Force-disable comm-aware offload path for baseline comparison.
export SGLANG_DIT_COMM_AWARE_OFFLOAD=0
export SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST=0
export SGLANG_WAN_MOCK_COMM_ENABLE="${SGLANG_WAN_MOCK_COMM_ENABLE:-1}"
export SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS="${SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS:-1}"
export SGLANG_WAN_MOCK_COMM_DEBUG="${SGLANG_WAN_MOCK_COMM_DEBUG:-0}"
export SGLANG_WAN_MOCK_COMM_PCIE_MB="${SGLANG_WAN_MOCK_COMM_PCIE_MB:-512}"
export SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE="${SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE:-4}"
export SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE="${SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE:-1.0}"
export SGLANG_WAN_MOCK_COMM_MAX_MB="${SGLANG_WAN_MOCK_COMM_MAX_MB:-256}"
export SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES="${SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES:-0}"

echo "=========================================="
echo "RUN: Profiled run (nsys)"
echo "=========================================="
echo "Start: $(date)"
echo "Mock PCIe config: enabled=${SGLANG_WAN_MOCK_COMM_ENABLE}, pcie_mb=${SGLANG_WAN_MOCK_COMM_PCIE_MB}, every_n_blocks=${SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS}, comm_aware=0"

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --force-overwrite=true \
    --output="$NSYS_OUTPUT_PREFIX" \
    sglang generate "${COMMON_FLAGS[@]}" \
    --perf-dump-path "$PERF_OUT" \
    --output-path "$VIDEO_OUT"

echo "End: $(date)"

echo "=========================================="
echo "Job complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Video: ${VIDEO_OUT}"
echo "  - nsys profile: ${NSYS_OUTPUT_PREFIX}.nsys-rep"
echo "  - Perf JSON: ${PERF_OUT}"
echo ""
