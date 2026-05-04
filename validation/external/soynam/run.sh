#!/usr/bin/env bash
# Bash wrapper that sources preflight then calls run_soynam.R.
#
# Idempotent: rerun overwrites previous outputs/rrblup_results.json + A_mat.tsv.
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[soynam run] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
GENO="${DATA_DIR}/geno.tsv"
PHENO="${DATA_DIR}/pheno.tsv"
MAP="${DATA_DIR}/marker_map.tsv"
OUT="${HERE}/outputs"

if [[ ! -f "${GENO}" || ! -f "${PHENO}" || ! -f "${MAP}" ]]; then
    echo "[soynam run] ABORT: extracted SoyNAM TSVs not found in ${DATA_DIR}."
    echo "[soynam run] Run fetch_data.sh first."
    exit 1
fi

# --- Pre-flight (rrBLUP::GWAS on 547 × 4275 needs ~3 GB peak RAM) -----------
preflight_check_with_data_size "soynam-run" 2 4

mkdir -p "${OUT}"

# --- Run ---------------------------------------------------------------------
echo "[soynam run] running rrBLUP::A.mat + rrBLUP::GWAS via Rscript"
Rscript --vanilla "${HERE}/run_soynam.R" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT}" \
    --trait-col y_blup

ls "${OUT}"/
echo "[soynam run] reference output produced in ${OUT}"
