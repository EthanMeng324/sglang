#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_profile_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash offload_profiling/run_profile.sh <model> [num_frames]
  bash offload_profiling/run_profile.sh --model <model> [--num-frames <n>] [--batch-size <n>] [--resident-ratio <r>] [--steps <csv>]

Supported models:
  wanvideo
  hunyuanvideo
  flux
  flux_2

Notes:
  - A full-resident no-offload dry run is executed first for warmup
  - This wrapper runs: no -> old -> new -> ratio-resident -> analyze
  - --steps accepts a comma-separated subset of: warmup,no,old,new,ratio,analyze
    aliases: phase -> ratio, analysis -> analyze, dryrun/dry-run -> warmup
  - For flux/flux_2, --batch-size maps to --num-outputs-per-prompt and generates multiple images from the same prompt
  - The final ratio-resident run uses the comm-aware new path plus SGLANG_DIT_OFFLOAD_RESIDENT_RATIO
  - SGLANG_DIT_OFFLOAD_RESIDENT_RATIO defaults to 0.4 for the ratio-resident run
  - Profiling defaults to PROFILE_SAVE_OUTPUT_ARTIFACTS=0 to avoid video/image save failures affecting perf/nsys
    set PROFILE_SAVE_OUTPUT_ARTIFACTS=1 if you explicitly want the generated artifact
  - Extra settings can still be passed through env vars, e.g. HEIGHT/WIDTH/NUM_GPUS
EOF
}

PROFILE_MODEL="${PROFILE_MODEL:-}"
NUM_FRAMES_OVERRIDE="${NUM_FRAMES:-}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE:-${NUM_OUTPUTS_PER_PROMPT:-}}"
RESIDENT_RATIO_OVERRIDE="${SGLANG_DIT_OFFLOAD_RESIDENT_RATIO:-}"
PROFILE_STEPS_OVERRIDE="${PROFILE_STEPS:-all}"

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
        --resident-ratio)
            RESIDENT_RATIO_OVERRIDE="$2"
            shift 2
            ;;
        --steps)
            PROFILE_STEPS_OVERRIDE="$2"
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
RESIDENT_RATIO_EFFECTIVE="${RESIDENT_RATIO_OVERRIDE:-${SGLANG_DIT_OFFLOAD_RESIDENT_RATIO:-0.4}}"

normalize_step_name() {
    local raw="$1"
    local step="${raw//[[:space:]]/}"
    step="$(printf '%s' "$step" | tr '[:upper:]' '[:lower:]')"
    case "$step" in
        ""|all) printf '%s' "all" ;;
        warmup|dryrun|dry-run) printf '%s' "warmup" ;;
        no|nooffload|no-offload|no_offload) printf '%s' "no" ;;
        old|oldoffload|old-offload|old_offload) printf '%s' "old" ;;
        new|comm|commaware|comm-aware|comm_aware) printf '%s' "new" ;;
        ratio|resident|ratioresident|ratio-resident|ratio_resident|phase) printf '%s' "ratio" ;;
        analyze|analysis) printf '%s' "analyze" ;;
        *)
            echo "ERROR: unsupported step '$raw'. Supported: warmup,no,old,new,ratio,analyze"
            exit 1
            ;;
    esac
}

RUN_WARMUP=0
RUN_NO=0
RUN_OLD=0
RUN_NEW=0
RUN_RATIO=0
RUN_ANALYZE=0

if [[ "$(normalize_step_name "$PROFILE_STEPS_OVERRIDE")" == "all" ]]; then
    RUN_WARMUP=1
    RUN_NO=1
    RUN_OLD=1
    RUN_NEW=1
    RUN_RATIO=1
    RUN_ANALYZE=1
else
    IFS=',' read -r -a _requested_steps <<<"$PROFILE_STEPS_OVERRIDE"
    for _step in "${_requested_steps[@]}"; do
        case "$(normalize_step_name "$_step")" in
            warmup) RUN_WARMUP=1 ;;
            no) RUN_NO=1 ;;
            old) RUN_OLD=1 ;;
            new) RUN_NEW=1 ;;
            ratio) RUN_RATIO=1 ;;
            analyze) RUN_ANALYZE=1 ;;
        esac
    done
fi

SELECTED_STEPS_DISPLAY=()
[[ "$RUN_WARMUP" == "1" ]] && SELECTED_STEPS_DISPLAY+=("warmup")
[[ "$RUN_NO" == "1" ]] && SELECTED_STEPS_DISPLAY+=("no")
[[ "$RUN_OLD" == "1" ]] && SELECTED_STEPS_DISPLAY+=("old")
[[ "$RUN_NEW" == "1" ]] && SELECTED_STEPS_DISPLAY+=("new")
[[ "$RUN_RATIO" == "1" ]] && SELECTED_STEPS_DISPLAY+=("ratio")
[[ "$RUN_ANALYZE" == "1" ]] && SELECTED_STEPS_DISPLAY+=("analyze")

if [[ "${#SELECTED_STEPS_DISPLAY[@]}" -eq 0 ]]; then
    echo "ERROR: no steps selected."
    exit 1
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
echo "Steps      : $(IFS=,; echo "${SELECTED_STEPS_DISPLAY[*]}")"
echo "Resident ratio  : ${RESIDENT_RATIO_EFFECTIVE}"
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

if [[ "$RUN_WARMUP" == "1" ]]; then
    run_step_with_env "Warmup Dry Run" "run_access_no.sh" \
        DRY_RUN=1 \
        DIT_CPU_OFFLOAD_OVERRIDE=false \
        SERVER_PORT_OVERRIDE="$WARMUP_SERVER_PORT" \
        SCHEDULER_PORT_OVERRIDE="$WARMUP_SCHEDULER_PORT" \
        MASTER_PORT_OVERRIDE="$WARMUP_MASTER_PORT" \
        NUM_INFERENCE_STEPS="$WARMUP_NUM_INFERENCE_STEPS"
    if [[ "$RUN_NO" == "1" || "$RUN_OLD" == "1" || "$RUN_NEW" == "1" || "$RUN_RATIO" == "1" ]]; then
        timeline_log "cooldown_start" "warmup" "sleep=${WARMUP_COOLDOWN_SEC}"
        sleep "$WARMUP_COOLDOWN_SEC"
        timeline_log "cooldown_end" "warmup" "sleep=${WARMUP_COOLDOWN_SEC}"
    fi
fi
if [[ "$RUN_NO" == "1" ]]; then
    run_step_with_env "No Offload" "run_access_no.sh" \
        DIT_CPU_OFFLOAD_OVERRIDE=false
fi
if [[ "$RUN_OLD" == "1" ]]; then
    run_step_with_env "Old Offload" "run_access_old.sh" \
        DIT_CPU_OFFLOAD_OVERRIDE=false
fi
if [[ "$RUN_NEW" == "1" ]]; then
    run_step_with_env "Comm-Aware Offload" "run_access.sh" \
        DIT_CPU_OFFLOAD_OVERRIDE=false \
        SGLANG_DIT_OFFLOAD_RESIDENT_RATIO= \
        SGLANG_DIT_PHASE_AWARE_PREFETCH=0
fi
if [[ "$RUN_RATIO" == "1" ]]; then
    run_step_with_env "Ratio-Resident Offload" "run_access_phase.sh" \
        DIT_CPU_OFFLOAD_OVERRIDE=false \
        SGLANG_DIT_OFFLOAD_RESIDENT_RATIO="$RESIDENT_RATIO_EFFECTIVE"
fi
if [[ "$RUN_ANALYZE" == "1" ]]; then
    run_step "Analyze NSYS" "analyze_nsys.sh"
fi

echo "=========================================="
echo "PROFILE MATRIX COMPLETE"
echo "=========================================="
echo "Model : ${PROFILE_MODEL}"
echo "End   : $(date)"
timeline_log "run_profile_end" "$PROFILE_MODEL" ""
