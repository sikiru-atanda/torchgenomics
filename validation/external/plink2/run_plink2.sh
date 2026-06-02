#!/usr/bin/env bash
# Run PLINK 2.0 against the MDP fixture to produce three reference outputs:
#   1. GLM linear regression on EarHT  → outputs/glm.EarHT.glm.linear
#   2. GRM (--make-rel)                → outputs/kinship.rel{,.id}
#   3. Pairwise r² (--r2-unphased ...) → outputs/r2.vcor (square matrix)
#
# Idempotent: rerun overwrites previous outputs/.
#
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).
#
# Peak RAM: PLINK 2 on 281×3093 typically uses < 50 MB. We pre-flight 1 GB
# headroom for safety.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

PLINK2="${HERE}/bin/plink2"
DATA="${HERE}/data/mdp"
PHENO="${HERE}/data/mdp_pheno.txt"
OUT="${HERE}/outputs"

if [[ ! -x "${PLINK2}" ]]; then
    echo "[plink2 run] ABORT: ${PLINK2} not found. Run install.sh first."
    exit 1
fi
for ext in bed bim fam; do
    if [[ ! -f "${DATA}.${ext}" ]]; then
        echo "[plink2 run] ABORT: ${DATA}.${ext} not found. Run fetch_data.sh first."
        exit 1
    fi
done
if [[ ! -f "${PHENO}" ]]; then
    echo "[plink2 run] ABORT: ${PHENO} not found. Run fetch_data.sh first."
    exit 1
fi

# Pre-flight: 1 GB working set, 1 GB peak RAM (very conservative — actual peak
# on this fixture is < 50 MB).
preflight_check_with_data_size "plink2-run" 1 1

mkdir -p "${OUT}"

# --- 1. GLM linear regression on EarHT --------------------------------------
echo "[plink2 run] (1/3) GLM linear regression on EarHT"
"${PLINK2}" \
    --bfile "${DATA}" \
    --pheno "${PHENO}" \
    --pheno-name EarHT \
    --glm allow-no-covars \
    --out "${OUT}/glm" \
    --threads 1 \
    --seed 1
ls "${OUT}"/glm.* 2>/dev/null

# --- 2. Kinship matrix via --make-rel cov (centered VanRaden-style) ----------
# PLINK 2's default --make-rel does GCTA-style per-SNP variance standardization;
# its diagonal correction is implementation-specific and doesn't match a clean
# closed-form. The `cov` modifier instead computes the centered cross-product
# K = (G - 2p)(G - 2p)^T / m, which differs from VanRaden's
# K_vr = (G - 2p)(G - 2p)^T / sum(2p(1-p)) by a single global scalar — so the
# two correlate at > 0.9999 element-wise. That gives us a clean PLINK 2 vs
# grm_vanraden reference.
#
# We still need --maf 1e-6 to drop truly monomorphic SNPs (PLINK 2 refuses
# --make-rel on those); both sides are then operating on the same SNP set.
echo "[plink2 run] (2/3) GRM via --make-rel cov (matches grm_vanraden up to scalar)"
"${PLINK2}" \
    --bfile "${DATA}" \
    --maf 1e-6 \
    --make-rel triangle cov \
    --out "${OUT}/kinship" \
    --threads 1
# Also dump the SNP list used for kinship so TorchGenomics scores against the same set
"${PLINK2}" \
    --bfile "${DATA}" \
    --maf 1e-6 \
    --write-snplist \
    --out "${OUT}/kinship_snps" \
    --threads 1 >/dev/null
ls "${OUT}"/kinship.* 2>/dev/null

# --- 3. Pairwise r² (square matrix; allows full-set comparison) -------------
echo "[plink2 run] (3/3) pairwise r² via --r2-unphased square"
# --r2-unphased gives the unphased (allele-count) r² that matches the formula
# implemented in TorchGenomics' LD module. We request `square` output so we get
# an n_var × n_var matrix that's directly comparable to TorchGenomics' tensor.
"${PLINK2}" \
    --bfile "${DATA}" \
    --r2-unphased square \
    --out "${OUT}/r2" \
    --threads 1
ls "${OUT}"/r2.* 2>/dev/null

echo "[plink2 run] all 3 reference outputs produced in ${OUT}"
