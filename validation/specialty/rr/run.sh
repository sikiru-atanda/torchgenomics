#!/usr/bin/env bash
# run.sh -- orchestrate the RR reference-tool comparison end-to-end.
#
# Pre-flight gate, then:
#   1. Re-generate fixture via fetch_data.sh (deterministic).
#   2. Run the lme4 reference fit via run_reference.R.
#   3. Run the TG RandomRegressionLMM fit via run_torchgenomics.py.
#   4. Diff outputs via compare.py and emit results/.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

preflight_check "rr-run" 1 2

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
RES_DIR="${HERE}/results"
mkdir -p "${OUT_DIR}" "${RES_DIR}"

echo "[rr run] 1/4 install gate"
bash "${HERE}/install.sh"

echo "[rr run] 2/4 fixture"
bash "${HERE}/fetch_data.sh"

echo "[rr run] 3/4 lme4 reference"
Rscript "${HERE}/run_reference.R" "${DATA_DIR}" "${OUT_DIR}"

echo "[rr run] 4/4 torchgenomics"
python3 "${HERE}/run_torchgenomics.py" --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}"

echo "[rr run] 5/5 compare"
python3 "${HERE}/compare.py" --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}" --results-dir "${RES_DIR}"

echo "[rr run] done. Results in ${RES_DIR}/"
