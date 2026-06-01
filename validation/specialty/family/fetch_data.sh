#!/usr/bin/env bash
# Fetch + stage the within-family fixture for Plan B Tier 3 / Agent C3.
#
# Fixture design: simulated sib-pair design (Young et al. 2022 §2.1 setup).
#   - n_families = 500
#   - sibs_per_family = 2 (canonical sib-pair design)
#   - n_snps = 100 causal markers + ~200 null
#   - seed = 42 (reproducible)
#   - planted direct-effect beta_d and indirect-effect (parental NTC) beta_i
#     with a known ratio (alpha = beta_i / beta_d); see simulate_sibpair.py.
#
# Why simulation, not a public dataset:
#   - The Young 2022 method requires either parental genotypes or sib-pairs
#     with known family structure. No widely-redistributable public dataset
#     pairs sibling genotypes with the level of metadata needed to evaluate
#     the direct/indirect decomposition at the sig-fig level. Snipar tutorials
#     ship simulators for exactly this reason; we re-implement the same
#     statistical model in-process and seed it (seed=42) for reproducibility.
#
# Pillar-B contract:
#   - Memory pre-flight before any work.
#   - Idempotent (skip if data/*.tsv already exists).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

source "${ROOT}/validation/external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

# Sib-pair simulation: 500 families x 2 sibs x 300 SNPs x 8 bytes ~ 2 MB; trivial.
preflight_check_with_data_size "family-fetch" 1 1

SIM_SCRIPT="${HERE}/simulate_sibpair.py"
PHENO_FILE="${DATA_DIR}/pheno.tsv"
GENO_FILE="${DATA_DIR}/geno.tsv"
PARENT_FILE="${DATA_DIR}/parent_geno.tsv"
TRUTH_FILE="${DATA_DIR}/truth.tsv"

if [[ -f "${PHENO_FILE}" && -f "${GENO_FILE}" && -f "${PARENT_FILE}" && -f "${TRUTH_FILE}" ]]; then
    echo "[family fetch] fixtures already staged at ${DATA_DIR}; skipping simulation"
    PHENO_ROWS=$(wc -l < "${PHENO_FILE}")
    GENO_ROWS=$(wc -l < "${GENO_FILE}")
    echo "[family fetch]   pheno rows: ${PHENO_ROWS}, geno rows: ${GENO_ROWS}"
    exit 0
fi

if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "[family fetch] ABORT: simulator missing: ${SIM_SCRIPT}"
    exit 1
fi

echo "[family fetch] running sib-pair simulator (Young et al. 2022 design, seed=42)"
python3 "${SIM_SCRIPT}" \
    --n-families 500 \
    --sibs-per-family 2 \
    --n-snps 300 \
    --n-causal 50 \
    --beta-d 0.20 \
    --alpha 0.50 \
    --h2 0.40 \
    --seed 42 \
    --output-dir "${DATA_DIR}"

echo "[family fetch] outputs:"
ls -la "${DATA_DIR}"
echo
echo "[family fetch] truth-table head:"
head -5 "${TRUTH_FILE}"
