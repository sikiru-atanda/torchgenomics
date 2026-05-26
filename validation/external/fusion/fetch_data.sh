#!/usr/bin/env bash
# FUSION harness fixture generator --- simulates a measured-expression
# TWAS dataset (300 samples × 15 genes, planted causal at gene index 7).
# No external network calls; the fixture is fully deterministic from
# --seed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

preflight_check_with_data_size "fusion-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

python3 "${HERE}/simulate_fixture.py" --output-dir "${DATA_DIR}" --seed 42
echo "[fusion fetch] fixture ready at ${DATA_DIR}"
