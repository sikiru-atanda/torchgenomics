#!/usr/bin/env bash
# FUSION measured-expression harness reference run.
# Invokes run_reference.R against the simulated fixture and writes
# outputs/fusion_results.tsv with one row per gene.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[fusion run] ABORT: ${INSTALL_MARKER} missing. Run install.sh first."
    exit 1
fi
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
EXPR_TSV="${DATA_DIR}/expression.tsv"
PHENO_TSV="${DATA_DIR}/phenotype.tsv"
COV_TSV="${DATA_DIR}/covariates.tsv"
OUT_TSV="${OUT_DIR}/fusion_results.tsv"

for f in "${EXPR_TSV}" "${PHENO_TSV}" "${COV_TSV}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[fusion run] ABORT: ${f} missing. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "fusion-run" 1 1
mkdir -p "${OUT_DIR}"

Rscript "${HERE}/run_reference.R" \
    --expression "${EXPR_TSV}" \
    --phenotype  "${PHENO_TSV}" \
    --covariates "${COV_TSV}" \
    --out        "${OUT_TSV}"

if [[ ! -f "${OUT_TSV}" ]]; then
    echo "[fusion run] ABORT: reference did not produce ${OUT_TSV}"
    exit 1
fi
echo "[fusion run] OK: $(wc -l < ${OUT_TSV}) lines in ${OUT_TSV}"
