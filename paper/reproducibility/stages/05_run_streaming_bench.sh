#!/usr/bin/env bash
# Stage 05: streaming memory bench (F6 panel B).
#
# Until the dedicated bench/streaming_p_sweep.py is authored in Tier 4 D2.8,
# this stage runs the streaming-memory regression test which captures the
# slope across p ∈ {10^4, 3·10^4, 10^5, 3·10^5, 10^6} at fixed n.
#
# Output:  paper/reproducibility/output/streaming_memory.json
#          (rendered into F6 panel B by 08_render_figures.py)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")/output"
mkdir -p "$OUT_DIR"

if [[ -f "$REPO_ROOT/bench/streaming_p_sweep.py" ]]; then
    echo "[stage 05] ==> bench/streaming_p_sweep.py"
    python3 "$REPO_ROOT/bench/streaming_p_sweep.py" \
        --output "$OUT_DIR/streaming_memory.json"
elif [[ -f "$REPO_ROOT/tests/test_streaming_memory.py" ]]; then
    echo "[stage 05] WARN: bench/streaming_p_sweep.py not yet authored;"
    echo "[stage 05]       falling back to tests/test_streaming_memory.py output"
    pytest "$REPO_ROOT/tests/test_streaming_memory.py" -v --tb=short \
        > "$OUT_DIR/streaming_memory_pytest.log" 2>&1 || true
else
    echo "[stage 05] WARN: no streaming bench source found; skipping"
fi

echo "[stage 05] complete."
