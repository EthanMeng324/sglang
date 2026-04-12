#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_profile_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash offload_profiling/run_profile.sh <model> [num_frames]
  bash offload_profiling/run_profile.sh --model <model> [--num-frames <n>] [--batch-size <n>]

Supported models:
  wanvideo
  hunyuanvideo
  flux
  flux_2

Notes:
  - A full-resident no-offload dry run is executed first for warmup
  - This wrapper runs: no -> old -> new -> phase -> analyze
  - For flux/flux_2, --batch-size maps to --num-outputs-per-prompt and generates multiple images from the same prompt
  - Phase resident phases can be overridden with SGLANG_DIT_OFFLOAD_RESIDENT_PHASES=<csv>
  - Phase resident bytes can also be requested with SGLANG_DIT_OFFLOAD_RESIDENT_RATIO=<0..1>
  - Phase prefetch lookahead can also be requested with SGLANG_DIT_OFFLOAD_PHASE_PREFETCH_RATIO=<0..1>
    wanvideo: entry,self_attn_tail,cross_attn,ffn
    hunyuanvideo:
      double_blocks -> entry,self_attn_tail,ffn
      single_blocks -> entry,tail
    flux / flux_2:
      transformer_blocks -> entry,self_attn_tail,ffn
      single_transformer_blocks -> entry,tail
  - Extra settings can still be passed through env vars, e.g. HEIGHT/WIDTH/NUM_GPUS
EOF
}

PROFILE_MODEL="${PROFILE_MODEL:-}"
NUM_FRAMES_OVERRIDE="${NUM_FRAMES:-}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE:-${NUM_OUTPUTS_PER_PROMPT:-}}"

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
        --batch-size|--num-outputs-per-prompt)
            BATCH_SIZE_OVERRIDE="$2"
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
if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
    export NUM_OUTPUTS_PER_PROMPT="$BATCH_SIZE_OVERRIDE"
fi

resolve_profile_model no "$PROFILE_MODEL" >/dev/null
apply_profile_model_defaults
WARMUP_NUM_INFERENCE_STEPS="${WARMUP_NUM_INFERENCE_STEPS:-$NUM_INFERENCE_STEPS}"
WARMUP_SERVER_PORT="${WARMUP_SERVER_PORT:-30100}"
WARMUP_SCHEDULER_PORT="${WARMUP_SCHEDULER_PORT:-5748}"
WARMUP_MASTER_PORT="${WARMUP_MASTER_PORT:-30170}"
WARMUP_COOLDOWN_SEC="${WARMUP_COOLDOWN_SEC:-5}"
TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-}"

if profile_model_is_image; then
    export DIT_CPU_OFFLOAD_OVERRIDE=false
fi

timeline_log() {
    if [[ -z "${TIMELINE_LOG_PATH:-}" ]]; then
        return 0
    fi
    local event="$1"
    local label="${2:-}"
    local detail="${3:-}"
    mkdir -p "$(dirname "$TIMELINE_LOG_PATH")"
    printf '%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(date +%s.%N)" \
        "$event" \
        "$label" \
        "$detail" >>"$TIMELINE_LOG_PATH"
}

if [[ -n "${TIMELINE_LOG_PATH:-}" ]]; then
    : >"$TIMELINE_LOG_PATH"
    printf 'utc_iso,epoch_s,event,label,detail\n' >>"$TIMELINE_LOG_PATH"
    timeline_log "run_profile_start" "$PROFILE_MODEL" "num_frames=${NUM_FRAMES_OVERRIDE:-default};batch_size=${BATCH_SIZE_OVERRIDE:-default}"
fi

echo "=========================================="
echo "RUN PROFILE MATRIX"
echo "=========================================="
echo "Model      : ${PROFILE_MODEL}"
echo "Num frames : ${NUM_FRAMES_OVERRIDE:-default}"
echo "Batch size : ${BATCH_SIZE_OVERRIDE:-default}"
echo "Resident phases : ${SGLANG_DIT_OFFLOAD_RESIDENT_PHASES:-<default>}"
echo "Resident ratio  : ${SGLANG_DIT_OFFLOAD_RESIDENT_RATIO:-<default>}"
echo "Prefetch ratio  : ${SGLANG_DIT_OFFLOAD_PHASE_PREFETCH_RATIO:-<default>}"
echo "Start      : $(date)"
echo ""

run_step() {
    local label="$1"
    local script_name="$2"
    echo "------------------------------------------"
    echo "STEP: ${label}"
    echo "------------------------------------------"
    timeline_log "step_start" "$label" "$script_name"
    if bash "$SCRIPT_DIR/$script_name" "$PROFILE_MODEL"; then
        timeline_log "step_end" "$label" "$script_name"
    else
        timeline_log "step_failed" "$label" "$script_name"
        return 1
    fi
    echo ""
}

run_step_with_env() {
    local label="$1"
    local script_name="$2"
    shift 2
    echo "------------------------------------------"
    echo "STEP: ${label}"
    echo "------------------------------------------"
    timeline_log "step_start" "$label" "$script_name"
    if env "$@" bash "$SCRIPT_DIR/$script_name" "$PROFILE_MODEL"; then
        timeline_log "step_end" "$label" "$script_name"
    else
        timeline_log "step_failed" "$label" "$script_name"
        return 1
    fi
    echo ""
}

run_step_with_env "Warmup Dry Run" "run_access_no.sh" \
    DRY_RUN=1 \
    SERVER_PORT_OVERRIDE="$WARMUP_SERVER_PORT" \
    SCHEDULER_PORT_OVERRIDE="$WARMUP_SCHEDULER_PORT" \
    MASTER_PORT_OVERRIDE="$WARMUP_MASTER_PORT" \
    NUM_INFERENCE_STEPS="$WARMUP_NUM_INFERENCE_STEPS"
timeline_log "cooldown_start" "warmup" "sleep=${WARMUP_COOLDOWN_SEC}"
sleep "$WARMUP_COOLDOWN_SEC"
timeline_log "cooldown_end" "warmup" "sleep=${WARMUP_COOLDOWN_SEC}"
if profile_model_is_image; then
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
timeline_log "run_profile_end" "$PROFILE_MODEL" ""
