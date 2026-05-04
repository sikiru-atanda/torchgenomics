#!/usr/bin/env bash
# Bash wrapper that sources preflight then calls run_twosamplemr.R.
#
# Idempotent: rerun overwrites previous outputs/twosamplemr_results.json.
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Locate install marker ---------------------------------------------------
INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[twosamplemr run] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
SUMSTATS="${DATA_DIR}/sumstats.tsv"
TRUTH="${DATA_DIR}/sim_truth.json"
OUT="${HERE}/outputs"

if [[ ! -f "${SUMSTATS}" || ! -f "${TRUTH}" ]]; then
    echo "[twosamplemr run] ABORT: simulated sumstats not found in ${DATA_DIR}."
    echo "[twosamplemr run] Run fetch_data.sh first."
    exit 1
fi

# --- Pre-flight (peak ~1 GB on 30 instruments × 1000 perms) ------------------
# MRPRESSO holds a K x n_perms matrix in memory plus per-iteration LOO
# residuals. On K=30 / n_perms=1000 this is ~250 KB, plus the R interpreter's
# own ~200 MB footprint. We pre-flight 2 GB for safety.
preflight_check_with_data_size "twosamplemr-run" 1 2

mkdir -p "${OUT}"

# --- Run ---------------------------------------------------------------------
echo "[twosamplemr run] running TwoSampleMR + MRPRESSO via Rscript"
Rscript --vanilla "${HERE}/run_twosamplemr.R" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --n-perms 1000 \
    --seed 42

ls "${OUT}"/
echo "[twosamplemr run] reference output produced in ${OUT}"
