#!/usr/bin/env bash
# Stage 06: native-kernel speedup bench (F6 panel A).
#
# Runs the existing realistic-size bench suite committed under
# bench/native_speedups.py and persists per-kernel timings to
# paper/reproducibility/output/native_speedups.json for F6 panel A.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/output"
mkdir -p "$OUT_DIR"

if [[ -f "$REPO_ROOT/bench/native_speedups.py" ]]; then
    echo "[stage 06] ==> bench/native_speedups.py"
    cd "$REPO_ROOT"
    python3 bench/native_speedups.py \
        ${REPRODUCE_QUICK:+--quick} \
        --output "$OUT_DIR/native_speedups.json"
else
    echo "[stage 06] WARN: bench/native_speedups.py not present; skipping"
fi

echo "[stage 06] complete."
