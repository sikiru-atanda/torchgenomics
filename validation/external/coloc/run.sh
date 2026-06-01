#!/usr/bin/env bash
# Bash wrapper that sources preflight then calls run.R.
#
# Idempotent: rerun overwrites previous outputs/coloc_results.json.
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
    echo "[coloc run] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT="${HERE}/outputs"

if [[ ! -f "${DATA_DIR}/scenarios.json" ]]; then
    echo "[coloc run] ABORT: scenarios manifest not found in ${DATA_DIR}."
    echo "[coloc run] Run fetch_data.sh first."
    exit 1
fi

# --- Pre-flight (peak ~200 MB R interpreter + tiny per-scenario buffers) -----
preflight_check_with_data_size "coloc-run" 1 1

mkdir -p "${OUT}"

# --- Run ---------------------------------------------------------------------
echo "[coloc run] running coloc.abf via Rscript"
Rscript --vanilla "${HERE}/run.R" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}"

ls "${OUT}"/
echo "[coloc run] reference output produced in ${OUT}"
