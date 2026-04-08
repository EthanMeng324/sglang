#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"

module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
module load GCC/12.3.0

mkdir -p "$RESULTS_DIR" "$PROFILES_DIR"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$REPO_ROOT"
source "$SCRIPT_DIR/model_profile_common.sh"

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

resolve_profile_model no "${1:-}"
apply_profile_model_defaults
apply_profile_output_layout

if profile_model_is_image; then
    export SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING="${SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING:-1}"
fi

# Common generation flags
COMMON_FLAGS=(
    --model-path "$MODEL_PATH"
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --dit-layerwise-offload false
    # --dit-offload-prefetch-size 1
    --num-gpus "$NUM_GPUS"
    --ulysses-degree "$ULYSSES_DEGREE"
    --attention-backend "$ATTENTION_BACKEND"
    --prompt "$PROMPT"
    --num-outputs-per-prompt "$NUM_OUTPUTS_PER_PROMPT"
    --num-frames "$NUM_FRAMES"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
)

if [[ -n "${MASTER_PORT_OVERRIDE:-}" ]]; then
    COMMON_FLAGS+=(--master-port "$MASTER_PORT_OVERRIDE")
fi

if [[ -n "${SERVER_PORT_OVERRIDE:-}" ]]; then
    COMMON_FLAGS+=(--port "$SERVER_PORT_OVERRIDE")
fi

if [[ -n "${SCHEDULER_PORT_OVERRIDE:-}" ]]; then
    COMMON_FLAGS+=(--scheduler-port "$SCHEDULER_PORT_OVERRIDE")
fi

if [[ -n "${GUIDANCE_SCALE_2:-}" ]]; then
    COMMON_FLAGS+=(--guidance-scale-2 "$GUIDANCE_SCALE_2")
fi

if [[ -n "${DIT_CPU_OFFLOAD_OVERRIDE:-}" ]]; then
    COMMON_FLAGS+=(--dit-cpu-offload "$DIT_CPU_OFFLOAD_OVERRIDE")
fi

DIT_CPU_OFFLOAD_OVERRIDE_DISPLAY="${DIT_CPU_OFFLOAD_OVERRIDE:-<unset>}"
DIT_CPU_OFFLOAD_FLAG_DISPLAY="default"
if [[ -n "${DIT_CPU_OFFLOAD_OVERRIDE:-}" ]]; then
    DIT_CPU_OFFLOAD_FLAG_DISPLAY="--dit-cpu-offload ${DIT_CPU_OFFLOAD_OVERRIDE}"
fi

ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-}"
if [[ -z "$ENABLE_TORCH_COMPILE" ]]; then
    if [[ "${DISABLE_TORCH_COMPILE:-0}" == "1" ]]; then
        ENABLE_TORCH_COMPILE="0"
    else
        ENABLE_TORCH_COMPILE="1"
    fi
fi

if [[ "$ENABLE_TORCH_COMPILE" == "1" ]]; then
    COMMON_FLAGS+=(--enable-torch-compile)
fi

export PYTHONUNBUFFERED=1

# Enable sync profiling for accurate stage timing
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1
export SGLANG_DIFFUSION_STAGE_LOGGING=1

# Force-disable comm-aware offload path for baseline comparison.
export SGLANG_DIT_COMM_AWARE_OFFLOAD=0
export SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST=0
export SGLANG_WAN_MOCK_COMM_ENABLE="${SGLANG_WAN_MOCK_COMM_ENABLE:-0}"
export SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS="${SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS:-1}"
export SGLANG_WAN_MOCK_COMM_DEBUG="${SGLANG_WAN_MOCK_COMM_DEBUG:-0}"
export SGLANG_WAN_MOCK_COMM_PCIE_MB="${SGLANG_WAN_MOCK_COMM_PCIE_MB:-512}"
export SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE="${SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE:-4}"
export SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE="${SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE:-1.0}"
export SGLANG_WAN_MOCK_COMM_MAX_MB="${SGLANG_WAN_MOCK_COMM_MAX_MB:-256}"
export SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES="${SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DEFAULT_WARMUP_EXT="mp4"
    if profile_model_is_image; then
        DEFAULT_WARMUP_EXT="png"
    fi
    DEFAULT_WARMUP_FILE_NAME="warmup_${PROFILE_MODEL}.${DEFAULT_WARMUP_EXT}"
    if [[ -n "${WARMUP_OUT:-}" ]]; then
        WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-$(dirname "$WARMUP_OUT")}"
        WARMUP_OUTPUT_FILE_NAME="${WARMUP_OUTPUT_FILE_NAME:-$(basename "$WARMUP_OUT")}"
    else
        WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-${MODEL_RESULTS_DIR:-$RESULTS_DIR}}"
        WARMUP_OUTPUT_FILE_NAME="${WARMUP_OUTPUT_FILE_NAME:-$DEFAULT_WARMUP_FILE_NAME}"
    fi
    mkdir -p "$WARMUP_OUTPUT_DIR"
    WARMUP_OUT="${WARMUP_OUTPUT_DIR%/}/${WARMUP_OUTPUT_FILE_NAME}"
    WARMUP_PERF_OUT="${WARMUP_PERF_OUT:-${MODEL_PROFILES_DIR:-$PROFILES_DIR}/perf_${PROFILE_MODEL}_warmup_no_offload_profiled.json}"

    echo "=========================================="
    echo "RUN: Warmup dry run (no offload)"
    echo "=========================================="
    echo "Start: $(date)"
    echo "Profile model: ${PROFILE_MODEL}"
    echo "Batch size: ${NUM_OUTPUTS_PER_PROMPT}"
    echo "Output: $(format_output_display_path "$WARMUP_OUT" "$NUM_OUTPUTS_PER_PROMPT")"
    echo "DIT_CPU_OFFLOAD_OVERRIDE=${DIT_CPU_OFFLOAD_OVERRIDE_DISPLAY}"
    echo "DIT CPU offload flag: ${DIT_CPU_OFFLOAD_FLAG_DISPLAY}"
    echo "Server port override: ${SERVER_PORT_OVERRIDE:-<default>}"
    echo "Scheduler port override: ${SCHEDULER_PORT_OVERRIDE:-<default>}"
    echo "Master port override: ${MASTER_PORT_OVERRIDE:-<default>}"
    echo "Force diffusers timestep embedding: ${SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING:-0}"

    time sglang generate "${COMMON_FLAGS[@]}" \
        --perf-dump-path "$WARMUP_PERF_OUT" \
        --output-path "$WARMUP_OUTPUT_DIR" \
        --output-file-name "$WARMUP_OUTPUT_FILE_NAME"

    echo "End: $(date)"
    echo ""
    echo "Warmup complete."
    echo "  - Artifact: $(format_output_display_path "$WARMUP_OUT" "$NUM_OUTPUTS_PER_PROMPT")"
    echo "  - Perf JSON: ${WARMUP_PERF_OUT}"
    exit 0
fi

echo "=========================================="
echo "RUN: Profiled run (nsys)"
echo "=========================================="
echo "Start: $(date)"
echo "Profile model: ${PROFILE_MODEL}"
echo "Batch size: ${NUM_OUTPUTS_PER_PROMPT}"
echo "Artifact: $(format_output_display_path "$VIDEO_OUT" "$NUM_OUTPUTS_PER_PROMPT")"
echo "DIT_CPU_OFFLOAD_OVERRIDE=${DIT_CPU_OFFLOAD_OVERRIDE_DISPLAY}"
echo "DIT CPU offload flag: ${DIT_CPU_OFFLOAD_FLAG_DISPLAY}"
echo "Server port override: ${SERVER_PORT_OVERRIDE:-<default>}"
echo "Scheduler port override: ${SCHEDULER_PORT_OVERRIDE:-<default>}"
echo "Master port override: ${MASTER_PORT_OVERRIDE:-<default>}"
echo "Force diffusers timestep embedding: ${SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING:-0}"
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
    --output-path "$OUTPUT_DIR" \
    --output-file-name "$OUTPUT_FILE_NAME"

echo "End: $(date)"

echo "=========================================="
echo "Job complete!"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""
echo "Outputs:"
echo "  - Artifact: $(format_output_display_path "$VIDEO_OUT" "$NUM_OUTPUTS_PER_PROMPT")"
echo "  - nsys profile: ${NSYS_OUTPUT_PREFIX}.nsys-rep"
echo "  - Perf JSON: ${PERF_OUT}"
echo ""
