#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/model_profile_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"

usage() {
    cat <<'EOF'
Usage:
  bash offload_profiling/run_profile_dmon.sh <model> [num_frames]
  bash offload_profiling/run_profile_dmon.sh --model <model> [--num-frames <n>] [--batch-size <n>]

Supported models:
  wanvideo
  hunyuanvideo
  flux

This wrapper:
  1. starts nvidia-smi dmon in the background
  2. runs the existing run_profile.sh flow
  3. stops dmon and stores the CSV under results/profiles/<model>/

Optional env vars:
  DMON_INTERVAL_SEC   Sampling interval for dmon. Default: 1
  DMON_METRICS        dmon metric groups. Default: pucvmt
  DMON_GPU_IDS        Optional GPU ids passed to `nvidia-smi dmon -i`
  DMON_OUTPUT_PATH    Optional explicit output CSV path
  BATCH_SIZE / NUM_OUTPUTS_PER_PROMPT  Number of outputs per prompt (useful for flux)
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
apply_profile_output_layout

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found in PATH."
    exit 1
fi

DMON_INTERVAL_SEC="${DMON_INTERVAL_SEC:-1}"
DMON_METRICS="${DMON_METRICS:-pucvmt}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DMON_OUTPUT_PATH="${DMON_OUTPUT_PATH:-${MODEL_PROFILES_DIR}/dmon_${PROFILE_MODEL}_${TIMESTAMP}.csv}"
TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${MODEL_PROFILES_DIR}/timeline_${PROFILE_MODEL}_${TIMESTAMP}.log}"
mkdir -p "$(dirname "$DMON_OUTPUT_PATH")"

DMON_PID=""
cleanup() {
    if [[ -n "${DMON_PID:-}" ]]; then
        if kill -0 "$DMON_PID" >/dev/null 2>&1; then
            kill "$DMON_PID" >/dev/null 2>&1 || true
            wait "$DMON_PID" >/dev/null 2>&1 || true
        fi
    fi
}
trap cleanup EXIT INT TERM

DMON_CMD=(
    nvidia-smi dmon
    -s "$DMON_METRICS"
    -d "$DMON_INTERVAL_SEC"
    -o DT
    -f "$DMON_OUTPUT_PATH"
)

if [[ -n "${DMON_GPU_IDS:-}" ]]; then
    DMON_CMD+=(-i "$DMON_GPU_IDS")
fi

echo "=========================================="
echo "RUN PROFILE MATRIX WITH DMON"
echo "=========================================="
echo "Model        : ${PROFILE_MODEL}"
echo "Num frames   : ${NUM_FRAMES_OVERRIDE:-default}"
echo "Batch size   : ${BATCH_SIZE_OVERRIDE:-default}"
echo "DMON output  : ${DMON_OUTPUT_PATH}"
echo "Timeline log : ${TIMELINE_LOG_PATH}"
echo "DMON metrics : ${DMON_METRICS}"
echo "DMON period  : ${DMON_INTERVAL_SEC}s"
if [[ -n "${DMON_GPU_IDS:-}" ]]; then
    echo "DMON GPUs    : ${DMON_GPU_IDS}"
fi
echo "Start        : $(date)"
echo ""

"${DMON_CMD[@]}" >/dev/null 2>&1 &
DMON_PID="$!"
sleep 1

if ! kill -0 "$DMON_PID" >/dev/null 2>&1; then
    echo "ERROR: failed to start nvidia-smi dmon"
    exit 1
fi

echo "Started nvidia-smi dmon with PID ${DMON_PID}"
echo ""

mkdir -p "$(dirname "$TIMELINE_LOG_PATH")"
printf 'utc_iso,epoch_s,event,label,detail\n' >"$TIMELINE_LOG_PATH"
printf '%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(date +%s.%N)" \
    "dmon_start" \
    "$PROFILE_MODEL" \
    "$DMON_OUTPUT_PATH" >>"$TIMELINE_LOG_PATH"

RUN_PROFILE_CMD=(
    bash "$SCRIPT_DIR/run_profile.sh"
    --model "$PROFILE_MODEL"
)
if [[ -n "${NUM_FRAMES_OVERRIDE:-}" ]]; then
    RUN_PROFILE_CMD+=(--num-frames "$NUM_FRAMES_OVERRIDE")
fi
if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
    RUN_PROFILE_CMD+=(--batch-size "$BATCH_SIZE_OVERRIDE")
fi
TIMELINE_LOG_PATH="$TIMELINE_LOG_PATH" "${RUN_PROFILE_CMD[@]}"

cleanup
DMON_PID=""

printf '%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(date +%s.%N)" \
    "dmon_stop" \
    "$PROFILE_MODEL" \
    "$DMON_OUTPUT_PATH" >>"$TIMELINE_LOG_PATH"

echo ""
echo "=========================================="
echo "RUN PROFILE MATRIX WITH DMON COMPLETE"
echo "=========================================="
echo "Model       : ${PROFILE_MODEL}"
echo "DMON output : ${DMON_OUTPUT_PATH}"
echo "End         : $(date)"
