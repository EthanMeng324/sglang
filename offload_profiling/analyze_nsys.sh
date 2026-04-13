#!/usr/bin/env bash
set -euo pipefail

module purge
module load Miniforge3/25.3.0-3
module load CUDA/12.8.0
module load GCC/12.3.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"
PROFILE_MODEL="${1:-${PROFILE_MODEL:-wanvideo}}"
PROFILE_MODEL="$(printf '%s' "$PROFILE_MODEL" | tr '[:upper:]' '[:lower:]')"

resolve_trace_path() {
    local preferred="$1"
    local fallback="$2"
    if [[ -f "$preferred" ]]; then
        printf '%s' "$preferred"
    else
        printf '%s' "$fallback"
    fi
}

case "$PROFILE_MODEL" in
    wan|wanvideo)
        PROFILE_MODEL="wanvideo"
        MODEL_PROFILES_DIR="${PROFILES_DIR%/}/wanvideo"
        ANALYSIS_DIR="${ANALYSIS_DIR:-$MODEL_PROFILES_DIR/analysis}"
        NEW_NSYS="${NEW_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/new_offload_nsys.nsys-rep" "$PROFILES_DIR/new_offload_nsys.nsys-rep")}"
        OLD_NSYS="${OLD_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/old_offload_nsys.nsys-rep" "$PROFILES_DIR/old_offload_nsys.nsys-rep")}"
        NO_NSYS="${NO_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/no_offload_nsys.nsys-rep" "$PROFILES_DIR/no_offload_nsys.nsys-rep")}"
        RATIO_NSYS="${RATIO_NSYS:-${PHASE_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/ratio_resident_offload_nsys.nsys-rep" "$MODEL_PROFILES_DIR/phase_offload_nsys.nsys-rep")}}"
        ;;
    flux|flux1|flux_1|fluximage|flux_image)
        PROFILE_MODEL="flux"
        MODEL_PROFILES_DIR="${PROFILES_DIR%/}/flux"
        ANALYSIS_DIR="${ANALYSIS_DIR:-$MODEL_PROFILES_DIR/analysis}"
        NEW_NSYS="${NEW_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_new_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_new_offload_nsys.nsys-rep")}"
        OLD_NSYS="${OLD_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_old_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_old_offload_nsys.nsys-rep")}"
        NO_NSYS="${NO_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_no_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_no_offload_nsys.nsys-rep")}"
        RATIO_NSYS="${RATIO_NSYS:-${PHASE_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_ratio_resident_offload_nsys.nsys-rep" "$MODEL_PROFILES_DIR/flux_phase_offload_nsys.nsys-rep")}}"
        ;;
    flux_2|flux2|flux-2|flux2dev|flux_2_dev)
        PROFILE_MODEL="flux_2"
        MODEL_PROFILES_DIR="${PROFILES_DIR%/}/flux_2"
        ANALYSIS_DIR="${ANALYSIS_DIR:-$MODEL_PROFILES_DIR/analysis}"
        NEW_NSYS="${NEW_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_2_new_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_2_new_offload_nsys.nsys-rep")}"
        OLD_NSYS="${OLD_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_2_old_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_2_old_offload_nsys.nsys-rep")}"
        NO_NSYS="${NO_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_2_no_offload_nsys.nsys-rep" "$PROFILES_DIR/flux_2_no_offload_nsys.nsys-rep")}"
        RATIO_NSYS="${RATIO_NSYS:-${PHASE_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/flux_2_ratio_resident_offload_nsys.nsys-rep" "$MODEL_PROFILES_DIR/flux_2_phase_offload_nsys.nsys-rep")}}"
        ;;
    hunyuan|hunyuanvideo)
        PROFILE_MODEL="hunyuanvideo"
        MODEL_PROFILES_DIR="${PROFILES_DIR%/}/hunyuanvideo"
        ANALYSIS_DIR="${ANALYSIS_DIR:-$MODEL_PROFILES_DIR/analysis}"
        NEW_NSYS="${NEW_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/hunyuanvideo_new_offload_nsys.nsys-rep" "$PROFILES_DIR/hunyuanvideo_new_offload_nsys.nsys-rep")}"
        OLD_NSYS="${OLD_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/hunyuanvideo_old_offload_nsys.nsys-rep" "$PROFILES_DIR/hunyuanvideo_old_offload_nsys.nsys-rep")}"
        NO_NSYS="${NO_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/hunyuanvideo_no_offload_nsys.nsys-rep" "$PROFILES_DIR/hunyuanvideo_no_offload_nsys.nsys-rep")}"
        RATIO_NSYS="${RATIO_NSYS:-${PHASE_NSYS:-$(resolve_trace_path "$MODEL_PROFILES_DIR/hunyuanvideo_ratio_resident_offload_nsys.nsys-rep" "$MODEL_PROFILES_DIR/hunyuanvideo_phase_offload_nsys.nsys-rep")}}"
        ;;
    *)
        echo "ERROR: unsupported PROFILE_MODEL='$PROFILE_MODEL'. Supported: wanvideo, flux, flux_2, hunyuanvideo."
        exit 1
        ;;
esac


mkdir -p "$ANALYSIS_DIR"
rm -f "$ANALYSIS_DIR"/analysis_summary.md "$ANALYSIS_DIR"/profile_matrix.csv

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not found in PATH."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH."
    exit 1
fi

if [[ ! -f "$NEW_NSYS" ]]; then
    echo "ERROR: missing new profile: $NEW_NSYS"
    exit 1
fi

if [[ ! -f "$OLD_NSYS" ]]; then
    echo "ERROR: missing old profile: $OLD_NSYS"
    exit 1
fi

if [[ ! -f "$NO_NSYS" ]]; then
    echo "ERROR: missing no-offload profile: $NO_NSYS"
    exit 1
fi

echo "=========================================="
echo "NSYS PROFILE MATRIX"
echo "=========================================="
echo "model: $PROFILE_MODEL"
echo "new  : $NEW_NSYS"
echo "old  : $OLD_NSYS"
echo "no   : $NO_NSYS"
echo "ratio: $RATIO_NSYS"
echo "out  : $ANALYSIS_DIR"
echo ""
echo "Optional overrides:"
echo "  NO_STEP_TIME_S / OLD_STEP_TIME_S / NEW_STEP_TIME_S / RATIO_STEP_TIME_S"
echo "  Legacy alias still accepted: PHASE_STEP_TIME_S"
echo "  NO_PEAK_RESERVED_MB / OLD_PEAK_RESERVED_MB / NEW_PEAK_RESERVED_MB / RATIO_PEAK_RESERVED_MB"
echo "  NO_PEAK_ALLOCATED_MB / OLD_PEAK_ALLOCATED_MB / NEW_PEAK_ALLOCATED_MB / RATIO_PEAK_ALLOCATED_MB"
echo "  Legacy aliases: *_PEAK_MEMORY_MB -> *_PEAK_RESERVED_MB"
echo ""

python3 "$SCRIPT_DIR/summarize_profile_matrix.py" \
    --no "$NO_NSYS" \
    --old "$OLD_NSYS" \
    --new "$NEW_NSYS" \
    --ratio "$RATIO_NSYS" \
    --output "$ANALYSIS_DIR/analysis_summary.md" \
    --csv-output "$ANALYSIS_DIR/profile_matrix.csv"

echo ""
echo "Analysis completed."
echo "Summary:"
echo "  - $ANALYSIS_DIR/analysis_summary.md"
echo "CSV outputs:"
ls -1 "$ANALYSIS_DIR"/profile_matrix.csv 2>/dev/null || true
