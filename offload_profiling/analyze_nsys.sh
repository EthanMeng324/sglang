#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$PROFILES_DIR/analysis}"

NEW_NSYS="${NEW_NSYS:-$PROFILES_DIR/new_offload_nsys.nsys-rep}"
OLD_NSYS="${OLD_NSYS:-$PROFILES_DIR/old_offload_nsys.nsys-rep}"
NO_NSYS="${NO_NSYS:-$PROFILES_DIR/no_offload_nsys.nsys-rep}"
PHASE_NSYS="${PHASE_NSYS:-$PROFILES_DIR/phase_offload_nsys.nsys-rep}"

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
echo "new  : $NEW_NSYS"
echo "old  : $OLD_NSYS"
echo "no   : $NO_NSYS"
echo "phase: $PHASE_NSYS"
echo "out  : $ANALYSIS_DIR"
echo ""
echo "Optional overrides:"
echo "  NO_STEP_TIME_S / OLD_STEP_TIME_S / NEW_STEP_TIME_S / PHASE_STEP_TIME_S"
echo "  NO_PEAK_RESERVED_MB / OLD_PEAK_RESERVED_MB / NEW_PEAK_RESERVED_MB / PHASE_PEAK_RESERVED_MB"
echo "  NO_PEAK_ALLOCATED_MB / OLD_PEAK_ALLOCATED_MB / NEW_PEAK_ALLOCATED_MB / PHASE_PEAK_ALLOCATED_MB"
echo "  Legacy aliases: *_PEAK_MEMORY_MB -> *_PEAK_RESERVED_MB"
echo ""

python3 "$SCRIPT_DIR/summarize_profile_matrix.py" \
    --no "$NO_NSYS" \
    --old "$OLD_NSYS" \
    --new "$NEW_NSYS" \
    --phase "$PHASE_NSYS" \
    --output "$ANALYSIS_DIR/analysis_summary.md" \
    --csv-output "$ANALYSIS_DIR/profile_matrix.csv"

echo ""
echo "Analysis completed."
echo "Summary:"
echo "  - $ANALYSIS_DIR/analysis_summary.md"
echo "CSV outputs:"
ls -1 "$ANALYSIS_DIR"/profile_matrix.csv 2>/dev/null || true
