#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash offload_profiling/run_profile.sh <model> [num_frames]
  bash offload_profiling/run_profile.sh --model <model> [--num-frames <n>]

Supported models:
  wanvideo
  hunyuanvideo
  flux

Notes:
  - A no-offload dry run with 1 denoising step is executed first for warmup
  - This wrapper runs: no -> old -> new -> phase -> analyze
  - Extra settings can still be passed through env vars, e.g. HEIGHT/WIDTH/NUM_GPUS
EOF
}

PROFILE_MODEL="${PROFILE_MODEL:-}"
NUM_FRAMES_OVERRIDE="${NUM_FRAMES:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            PROFILE_MODEL="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$PROFILE_MODEL" ]]; then
                PROFILE_MODEL="$1"
            elif [[ -z "$NUM_FRAMES_OVERRIDE" ]]; then
                NUM_FRAMES_OVERRIDE="$1"
            else
                echo "ERROR: unexpected argument '$1'"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

PROFILE_MODEL="${PROFILE_MODEL:-wanvideo}"
export PROFILE_MODEL

if [[ -n "${NUM_FRAMES_OVERRIDE:-}" ]]; then
    export NUM_FRAMES="$NUM_FRAMES_OVERRIDE"
fi

echo "=========================================="
echo "RUN PROFILE MATRIX"
echo "=========================================="
echo "Model      : ${PROFILE_MODEL}"
echo "Num frames : ${NUM_FRAMES_OVERRIDE:-default}"
echo "Start      : $(date)"
echo ""

run_step() {
    local label="$1"
    local script_name="$2"
    echo "------------------------------------------"
    echo "STEP: ${label}"
    echo "------------------------------------------"
    bash "$SCRIPT_DIR/$script_name" "$PROFILE_MODEL"
    echo ""
}

run_step_with_env() {
    local label="$1"
    local script_name="$2"
    shift 2
    echo "------------------------------------------"
    echo "STEP: ${label}"
    echo "------------------------------------------"
    env "$@" bash "$SCRIPT_DIR/$script_name" "$PROFILE_MODEL"
    echo ""
}

run_step_with_env "Warmup Dry Run" "run_access_no.sh" \
    DRY_RUN=1 \
    NUM_INFERENCE_STEPS=1
run_step "No Offload" "run_access_no.sh"
run_step "Old Offload" "run_access_old.sh"
run_step "Comm-Aware Offload" "run_access.sh"
run_step "Phase-Aware Offload" "run_access_phase.sh"
run_step "Analyze NSYS" "analyze_nsys.sh"

echo "=========================================="
echo "PROFILE MATRIX COMPLETE"
echo "=========================================="
echo "Model : ${PROFILE_MODEL}"
echo "End   : $(date)"
