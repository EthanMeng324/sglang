#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="${TRACE_DIR:-$SCRIPT_DIR/results/traces}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/trace_analysis}"

mkdir -p "$OUT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH."
    exit 1
fi

if [[ ! -d "$TRACE_DIR" ]]; then
    echo "ERROR: trace dir not found: $TRACE_DIR"
    echo "Put *.trace.json.gz under: $SCRIPT_DIR/results/traces/"
    exit 1
fi

shopt -s nullglob
trace_files=("$TRACE_DIR"/*.trace.json.gz)
shopt -u nullglob
if [[ ${#trace_files[@]} -eq 0 ]]; then
    echo "ERROR: no *.trace.json.gz found under $TRACE_DIR"
    exit 1
fi

echo "=========================================="
echo "TRACE ANALYSIS (CPU-side)"
echo "=========================================="
echo "trace_dir: $TRACE_DIR"
echo "out_dir:   $OUT_DIR"
echo ""

python3 "$SCRIPT_DIR/trace_analysis/cpu_trace_overlap.py" \
    --traces-dir "$TRACE_DIR" \
    | tee "$OUT_DIR/cpu_trace_overlap.txt"

python3 "$SCRIPT_DIR/trace_analysis/analyze_copy_events.py" \
    --traces-dir "$TRACE_DIR" \
    | tee "$OUT_DIR/analyze_copy_events.txt"

python3 "$SCRIPT_DIR/trace_analysis/analyze_pcie_bandwidth.py" \
    --traces-dir "$TRACE_DIR" \
    | tee "$OUT_DIR/analyze_pcie_bandwidth.txt"

python3 "$SCRIPT_DIR/trace_analysis/analyze_pcie_from_copies.py" \
    --traces-dir "$TRACE_DIR" \
    | tee "$OUT_DIR/analyze_pcie_from_copies.txt"

echo ""
echo "Done. Reports:"
ls -1 "$OUT_DIR"/*.txt 2>/dev/null || true
