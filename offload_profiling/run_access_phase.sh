#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env \
    PROFILE_MODE=ratio \
    SGLANG_DIT_PHASE_AWARE_PREFETCH=0 \
    SGLANG_DIT_OFFLOAD_RESIDENT_RATIO="${SGLANG_DIT_OFFLOAD_RESIDENT_RATIO:-0.4}" \
    bash "$SCRIPT_DIR/run_access.sh" "$@"
