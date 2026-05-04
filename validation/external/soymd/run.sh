#!/usr/bin/env bash
# Bash wrapper that sources preflight then calls run_mediation.R.
#
# Idempotent: rerun overwrites previous outputs/mediation_results.json.
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
    echo "[soymd run] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
TRIPLE="${DATA_DIR}/triple.tsv"
TRUTH="${DATA_DIR}/sim_truth.json"
OUT="${HERE}/outputs"

if [[ ! -f "${TRIPLE}" || ! -f "${TRUTH}" ]]; then
    echo "[soymd run] ABORT: simulated triple not found in ${DATA_DIR}."
    echo "[soymd run] Run fetch_data.sh first."
    exit 1
fi

# --- Pre-flight (peak ~300 MB on n=200 with 1000 sims) ----------------------
# `mediation::mediate` holds an n × sims matrix of per-replicate predictions
# in memory plus the R interpreter's ~200 MB footprint. We pre-flight 2 GB.
preflight_check_with_data_size "soymd-run" 1 2

mkdir -p "${OUT}"

# --- Run ---------------------------------------------------------------------
echo "[soymd run] running mediation::mediate via Rscript"
Rscript --vanilla "${HERE}/run_mediation.R" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --n-sims 1000 \
    --seed 42

ls "${OUT}"/
echo "[soymd run] reference output produced in ${OUT}"
