#!/usr/bin/env bash
# Survival GWAS reference comparison run (Tier 3 C1).
#
# Pipeline:
#   1. Reference: per-SNP coxme on the fixture (Wald beta/SE/p; ~0.5 s).
#   2. Comparator: TorchGWAS SurvivalGLMM PQL+score test, with and
#      without SPACox SPA tail correction.
#   3. compare.py merges, computes agreement metrics, writes
#      results/{summary.tsv, agreement.json}.
#
# Pre-flight: 1 GB disk + 4 GB RAM headroom (TG GRM eigendecomposition
# is the main RAM peak for n=500).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
RESULTS_DIR="${HERE}/results"
INFRA_BLOCKER="${HERE}/.infra_blocker"
INSTALL_MARKER="${HERE}/.install_marker"

if [[ -f "${INFRA_BLOCKER}" ]]; then
    echo "[survival run] ABORT: install was marked infra-blocker."
    cat "${INFRA_BLOCKER}"
    exit 1
fi

if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[survival run] ABORT: ${INSTALL_MARKER} missing.  Run install.sh first."
    exit 1
fi

for f in "${DATA_DIR}/pheno.csv" "${DATA_DIR}/geno.csv" "${DATA_DIR}/kinship.csv"; do
    if [[ ! -f "${f}" ]]; then
        echo "[survival run] ABORT: ${f} missing.  Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check "survival-run" 1 4

mkdir -p "${OUT_DIR}" "${RESULTS_DIR}"

echo "[survival run] step 1/2: R coxme per-SNP fits ..."
Rscript "${HERE}/run.R" \
    "${DATA_DIR}/pheno.csv" \
    "${DATA_DIR}/geno.csv" \
    "${DATA_DIR}/kinship.csv" \
    "${OUT_DIR}/coxme_results.csv"

echo "[survival run] step 2/2: TG SurvivalGLMM + agreement comparison ..."
python3 "${HERE}/compare.py" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --results-dir "${RESULTS_DIR}"

# Compute SHA256 manifest for committed results (per Tier 3 acceptance gate).
echo "[survival run] writing manifest.sha256"
(cd "${RESULTS_DIR}" && sha256sum agreement.json summary.tsv > manifest.sha256)

echo "[survival run] done. Results:"
ls -la "${RESULTS_DIR}"
