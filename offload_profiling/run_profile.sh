#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_profile_common.sh"

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
  - A full-resident no-offload dry run is executed first for warmup
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

resolve_profile_model no "$PROFILE_MODEL" >/dev/null
apply_profile_model_defaults
WARMUP_NUM_INFERENCE_STEPS="${WARMUP_NUM_INFERENCE_STEPS:-$NUM_INFERENCE_STEPS}"
WARMUP_SERVER_PORT="${WARMUP_SERVER_PORT:-30100}"
WARMUP_SCHEDULER_PORT="${WARMUP_SCHEDULER_PORT:-5748}"
WARMUP_MASTER_PORT="${WARMUP_MASTER_PORT:-30170}"
WARMUP_COOLDOWN_SEC="${WARMUP_COOLDOWN_SEC:-5}"

if [[ "$PROFILE_MODEL" == "flux" ]]; then
    export DIT_CPU_OFFLOAD_OVERRIDE=false
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
    SERVER_PORT_OVERRIDE="$WARMUP_SERVER_PORT" \
    SCHEDULER_PORT_OVERRIDE="$WARMUP_SCHEDULER_PORT" \
    MASTER_PORT_OVERRIDE="$WARMUP_MASTER_PORT" \
    NUM_INFERENCE_STEPS="$WARMUP_NUM_INFERENCE_STEPS"
sleep "$WARMUP_COOLDOWN_SEC"
if [[ "$PROFILE_MODEL" == "flux" ]]; then
    run_step "No Offload" "run_access_no.sh"
    run_step "Old Offload" "run_access_old.sh"
    run_step "Comm-Aware Offload" "run_access.sh"
    run_step "Phase-Aware Offload" "run_access_phase.sh"
else
    run_step "Comm-Aware Offload" "run_access.sh"
    run_step "Phase-Aware Offload" "run_access_phase.sh"
    run_step "No Offload" "run_access_no.sh"
    run_step "Old Offload" "run_access_old.sh"
fi
run_step "Analyze NSYS" "analyze_nsys.sh"

echo "=========================================="
echo "PROFILE MATRIX COMPLETE"
echo "=========================================="
echo "Model : ${PROFILE_MODEL}"
echo "End   : $(date)"
