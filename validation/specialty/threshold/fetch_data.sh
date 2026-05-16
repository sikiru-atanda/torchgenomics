#!/usr/bin/env bash
# Threshold-linear harness data fetch -- sources ../../external/_lib/preflight.sh.
#
# Generates a deterministic 3-trait simulated fixture (seed=42):
#   trait 1: ordinal, 2 categories (binary)
#   trait 2: ordinal, 3 categories
#   trait 3: continuous (Gaussian)
# Liabilities are drawn from a known correlated residual covariance R and
# additive genetic variance G (Vg = sigma2_g * I via diploid kinship).
# Sample size and marker count are chosen so a Gibbs chain converges in
# under ~3 wall-clock minutes on a single core.
#
# Outputs (data/):
#   pheno.txt     -- BLUPF90 / TG joint phenotype + covariate file
#   geno.012      -- additive dosages 0/1/2, ASCII (BLUPF90 format)
#   marker.map    -- 1-based marker positions
#   pedigree.dat  -- self-pedigree (genomic-only run; required by gibbsf90)
#   sim_truth.json -- true thresholds, R, G, betas
#   X0.tsv        -- intercept + sex covariate matrix (TG side)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"
PHENO="${DATA_DIR}/pheno.txt"
GENO="${DATA_DIR}/geno.012"
MAP="${DATA_DIR}/marker.map"
PEDI="${DATA_DIR}/pedigree.dat"
TRUTH="${DATA_DIR}/sim_truth.json"
X0="${DATA_DIR}/X0.tsv"

NEED_REGEN=0
for f in "${PHENO}" "${GENO}" "${MAP}" "${PEDI}" "${TRUTH}" "${X0}"; do
    if [[ ! -f "${f}" ]]; then
        NEED_REGEN=1
        break
    fi
done

if (( NEED_REGEN == 0 )); then
    echo "[thr fetch] data already staged at ${DATA_DIR}; skipping"
    exit 0
fi

preflight_check_with_data_size "thr-fetch" 1 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "[thr fetch] ABORT: python3 not on PATH."
    exit 1
fi
python3 -c "import numpy, scipy, json" >/dev/null 2>&1 || {
    echo "[thr fetch] ABORT: python deps numpy+scipy required."
    exit 1
}

mkdir -p "${DATA_DIR}"
echo "[thr fetch] generating 3-trait simulated fixture (seed=42)"
python3 "${HERE}/generate.py" --output-dir "${DATA_DIR}" --seed 42
echo "[thr fetch] done."
ls -la "${DATA_DIR}"

