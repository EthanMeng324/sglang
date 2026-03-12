#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
PROFILES_DIR="${PROFILES_DIR:-$RESULTS_DIR/profiles}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$PROFILES_DIR/analysis}"

NEW_NSYS="${NEW_NSYS:-$PROFILES_DIR/new_offload_nsys.nsys-rep}"
OLD_NSYS="${OLD_NSYS:-$PROFILES_DIR/old_offload_nsys.nsys-rep}"
COMM_MODE="${1:-${COMM_MODE:-}}"

if [[ -z "$COMM_MODE" ]]; then
    if [[ "${SGLANG_WAN_MOCK_COMM_ENABLE:-1}" == "1" ]]; then
        COMM_MODE="mock"
    else
        COMM_MODE="real"
    fi
fi

if [[ "$COMM_MODE" != "mock" && "$COMM_MODE" != "real" ]]; then
    echo "ERROR: unsupported comm mode '$COMM_MODE' (expected: mock or real)"
    echo "Usage: $0 [mock|real]"
    exit 1
fi

mkdir -p "$ANALYSIS_DIR"
rm -f "$ANALYSIS_DIR"/*.csv "$ANALYSIS_DIR"/analysis_summary.md

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
    echo "Run: $SCRIPT_DIR/run.sh"
    exit 1
fi

if [[ ! -f "$OLD_NSYS" ]]; then
    echo "ERROR: missing old profile: $OLD_NSYS"
    echo "Run: $SCRIPT_DIR/run_old.sh"
    exit 1
fi

export_to_sqlite() {
    local rep_file="$1"
    local sqlite_file="${rep_file%.nsys-rep}.sqlite"
    local need_export=0

    if [[ "${FORCE_EXPORT:-0}" == "1" ]]; then
        need_export=1
    fi

    if [[ ! -f "$sqlite_file" ]]; then
        need_export=1
    elif [[ "$rep_file" -nt "$sqlite_file" ]]; then
        # nsys-rep is newer than sqlite: must re-export to avoid stale analysis.
        need_export=1
    fi

    if [[ "$need_export" == "1" ]]; then
        rm -f "$sqlite_file"
        echo "Export SQLite: $rep_file -> $sqlite_file" >&2
        nsys export --type=sqlite --output="$sqlite_file" "$rep_file" >&2
    else
        echo "Reuse SQLite: $sqlite_file" >&2
    fi

    echo "$sqlite_file"
}

echo "=========================================="
echo "NSYS EXPORT + ANALYSIS"
echo "=========================================="
echo "new: $NEW_NSYS"
echo "old: $OLD_NSYS"
echo "out: $ANALYSIS_DIR"
echo "comm mode: $COMM_MODE"
echo ""

NEW_DB="$(export_to_sqlite "$NEW_NSYS")"
OLD_DB="$(export_to_sqlite "$OLD_NSYS")"

python3 "$SCRIPT_DIR/nsys_query.py" "$NEW_DB" "$OLD_DB" "$ANALYSIS_DIR" "$COMM_MODE"
python3 "$SCRIPT_DIR/summarize_nsys_csv.py" \
    --analysis-dir "$ANALYSIS_DIR" \
    --comm-mode "$COMM_MODE" \
    --output "$ANALYSIS_DIR/analysis_summary.md"

echo ""
echo "Analysis completed."
echo "Summary:"
echo "  - $ANALYSIS_DIR/analysis_summary.md"
echo "CSV outputs:"
ls -1 "$ANALYSIS_DIR"/*.csv 2>/dev/null || true
