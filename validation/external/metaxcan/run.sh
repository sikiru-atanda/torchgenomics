#!/usr/bin/env bash
# MetaXcan harness reference run --- sources ../_lib/preflight.sh (pre-flight gate).
# Invokes MetaXcan/software/SPrediXcan.py against the simulated fixture and
# writes the gene-level association CSV to outputs/sprediXcan_results.csv.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[metaxcan run] ABORT: ${INSTALL_MARKER} missing. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
MODEL_DB="${DATA_DIR}/model.db"
COV_GZ="${DATA_DIR}/model.txt.gz"
SUMSTATS_GZ="${DATA_DIR}/gwas_sumstats.txt.gz"

for f in "${MODEL_DB}" "${COV_GZ}" "${SUMSTATS_GZ}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[metaxcan run] ABORT: ${f} missing. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "metaxcan-run" 1 1

mkdir -p "${OUT_DIR}"

OUT_CSV="${OUT_DIR}/sprediXcan_results.csv"

echo "[metaxcan run] invoking SPrediXcan.py"
echo "[metaxcan run]   model_db:   ${MODEL_DB}"
echo "[metaxcan run]   covariance: ${COV_GZ}"
echo "[metaxcan run]   gwas:       ${SUMSTATS_GZ}"
echo "[metaxcan run]   output:     ${OUT_CSV}"

# shellcheck disable=SC2086
PYTHONPATH="${METAXCAN_REPO_DIR}/software" python3 "${METAXCAN_SPREDIXCAN}" \
    --model_db_path "${MODEL_DB}" \
    --covariance "${COV_GZ}" \
    --gwas_file "${SUMSTATS_GZ}" \
    --snp_column SNP \
    --effect_allele_column effect_allele \
    --non_effect_allele_column non_effect_allele \
    --zscore_column zscore \
    --beta_column beta \
    --se_column se \
    --pvalue_column pvalue \
    --output_file "${OUT_CSV}" \
    --overwrite \
    --verbosity 7

echo "[metaxcan run] done. Output: ${OUT_CSV}"
head -3 "${OUT_CSV}"
