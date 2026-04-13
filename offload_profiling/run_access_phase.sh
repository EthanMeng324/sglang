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

resolve_profile_model phase "${1:-}"
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
    --dit-layerwise-offload true
    --dit-offload-prefetch-size 1
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

if [[ -n "${GUIDANCE_SCALE_2:-}" ]]; then
    COMMON_FLAGS+=(--guidance-scale-2 "$GUIDANCE_SCALE_2")
fi

if [[ -n "${DIT_CPU_OFFLOAD_OVERRIDE:-}" ]]; then
    COMMON_FLAGS+=(--dit-cpu-offload "$DIT_CPU_OFFLOAD_OVERRIDE")
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

# Enable comm-aware offload.
export SGLANG_DIT_COMM_AWARE_OFFLOAD="${SGLANG_DIT_COMM_AWARE_OFFLOAD:-1}"
export SGLANG_DIT_PHASE_AWARE_PREFETCH="${SGLANG_DIT_PHASE_AWARE_PREFETCH:-1}"
export SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST="${SGLANG_DIT_COMM_AWARE_PATCH_TORCH_DIST:-0}"
export SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB="${SGLANG_DIT_COMM_PREFETCH_CHUNK_SIZE_MB:-16}"
export SGLANG_DIT_COMM_PREFETCH_SUBMODULE_GRANULARITY=1
export SGLANG_WAN_MOCK_COMM_ENABLE="${SGLANG_WAN_MOCK_COMM_ENABLE:-0}"
export SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS="${SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS:-1}"
export SGLANG_WAN_MOCK_COMM_DEBUG="${SGLANG_WAN_MOCK_COMM_DEBUG:-0}"
export SGLANG_WAN_MOCK_COMM_PCIE_MB="${SGLANG_WAN_MOCK_COMM_PCIE_MB:-512}"
export SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE="${SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE:-4}"
export SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE="${SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE:-1.0}"
export SGLANG_WAN_MOCK_COMM_MAX_MB="${SGLANG_WAN_MOCK_COMM_MAX_MB:-256}"
export SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES="${SGLANG_DIFFUSION_LOG_DENOISING_STEP_TIMES:-0}"
PROFILE_SAVE_OUTPUT_ARTIFACTS="${PROFILE_SAVE_OUTPUT_ARTIFACTS:-0}"

OUTPUT_FLAGS=(
    --output-path "$OUTPUT_DIR"
    --output-file-name "$OUTPUT_FILE_NAME"
)
if [[ "$PROFILE_SAVE_OUTPUT_ARTIFACTS" != "1" ]]; then
    OUTPUT_FLAGS+=(--no-save-output)
fi

echo "=========================================="
echo "RUN: Profiled run (nsys)"
echo "=========================================="
echo "Start: $(date)"
echo "Profile model: ${PROFILE_MODEL}"
echo "Batch size: ${NUM_OUTPUTS_PER_PROMPT}"
echo "Artifact: $(format_output_display_path "$VIDEO_OUT" "$NUM_OUTPUTS_PER_PROMPT")"
echo "Save artifact: ${PROFILE_SAVE_OUTPUT_ARTIFACTS}"
echo "Force diffusers timestep embedding: ${SGLANG_FORCE_DIFFUSERS_TIMESTEP_EMBEDDING:-0}"
echo "Mock PCIe config: enabled=${SGLANG_WAN_MOCK_COMM_ENABLE}, pcie_mb=${SGLANG_WAN_MOCK_COMM_PCIE_MB}, every_n_blocks=${SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS}, comm_aware=${SGLANG_DIT_COMM_AWARE_OFFLOAD}"

time nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --kill=none \
    --force-overwrite=true \
    --output="$NSYS_OUTPUT_PREFIX" \
    sglang generate "${COMMON_FLAGS[@]}" \
    --perf-dump-path "$PERF_OUT" \
    "${OUTPUT_FLAGS[@]}"

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
