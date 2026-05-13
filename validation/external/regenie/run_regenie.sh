#!/usr/bin/env bash
# Run regenie Step 1 + Step 2 against the MDP fixture for both quantitative
# and binary phenotypes:
#
#   Step 1 (per-block ridge → LOCO predictor)
#     - one Step 1 run produces predictors for all phenotypes in the file
#       (regenie writes a `*_pred.list` and per-chromosome predictor files).
#
#   Step 2 (per-SNP association)
#     - one run on the quantitative trait → outputs/step2_qt_EarHT.regenie
#     - one run on the binary trait + Firth correction → outputs/step2_bin_EarHT_bin.regenie
#
# Idempotent: rerun overwrites previous outputs/.
#
# Pillar B contract:
#   - Memory pre-flight before run (Step 1 has the largest peak; on this
#     fixture < 200 MB but we pre-flight 1 GB headroom).
#   - Outputs go to ${HERE}/outputs (.gitignored).
#
# regenie flags worth knowing:
#   --bsize 100      block size for ridge regression in Step 1. With m=3093
#                    SNPs this gives ~31 ridge blocks. The default for
#                    biobank data is 1000, but smaller blocks are needed
#                    when m is small.
#   --lowmem         disables the in-memory genotype cache; uses a temp file
#                    instead. Keeps peak memory predictable.
#   --pred           in Step 2, point at Step 1's `*_pred.list`.
#   --bt --firth     binary trait with Firth correction for low-count cells.
#   --pThresh 0.01   apply Firth only when standard logistic p < 0.01
#                    (regenie default; faster + numerically stable).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

REGENIE="${HERE}/bin/regenie"
DATA="${HERE}/data/mdp"
PHENO="${HERE}/data/regenie_pheno.tsv"
COVAR="${HERE}/data/regenie_covar.tsv"
OUT="${HERE}/outputs"

if [[ ! -x "${REGENIE}" ]]; then
    echo "[regenie run] ABORT: ${REGENIE} not found. Run install.sh first."
    exit 1
fi
for ext in bed bim fam; do
    if [[ ! -f "${DATA}.${ext}" ]]; then
        echo "[regenie run] ABORT: ${DATA}.${ext} not found. Run fetch_data.sh first."
        exit 1
    fi
done
for f in "${PHENO}" "${COVAR}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[regenie run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

# Pre-flight: 1 GB working set, 2 GB peak RAM. Step 1 on 281 × 3093 fits in
# < 200 MB, but we leave 1× MDP-equivalent headroom for the lowmem temp file.
preflight_check_with_data_size "regenie-run" 1 2

mkdir -p "${OUT}"
LOWMEM_DIR="${OUT}/.step1_tmp"
mkdir -p "${LOWMEM_DIR}"

# --- Step 1: whole-genome ridge predictor (LOCO) -----------------------------
# The block-ridge prediction step. With N=281 and m=3093, the whole-genome
# fit is in regenie's small-data regime — the predictor will essentially
# reduce to per-CV-fold residual means after ridge shrinkage. That's fine
# for the harness: we only need the prediction file to exist and contain
# finite values for Step 2 to consume.
#
# `--qt` so EarHT is fit as quantitative (regenie auto-detects but we're
# explicit). The phenotype file has both a quant and a binary trait; we
# split into two Step 1 runs to keep their LOCO predictors separate.
echo "[regenie run] (1/3) Step 1 quantitative ridge → outputs/step1_qt_*"
"${REGENIE}" \
    --step 1 \
    --bed "${DATA}" \
    --phenoFile "${PHENO}" \
    --phenoCol EarHT \
    --covarFile "${COVAR}" \
    --covarColList PC1,PC2 \
    --bsize 100 \
    --qt \
    --lowmem \
    --lowmem-prefix "${LOWMEM_DIR}/qt_" \
    --threads 1 \
    --out "${OUT}/step1_qt"

echo "[regenie run] (2/3) Step 1 binary ridge → outputs/step1_bin_*"
"${REGENIE}" \
    --step 1 \
    --bed "${DATA}" \
    --phenoFile "${PHENO}" \
    --phenoCol EarHT_bin \
    --covarFile "${COVAR}" \
    --covarColList PC1,PC2 \
    --bsize 100 \
    --bt \
    --lowmem \
    --lowmem-prefix "${LOWMEM_DIR}/bin_" \
    --threads 1 \
    --out "${OUT}/step1_bin"

# --- Step 2: per-SNP association --------------------------------------------
# Quantitative path: per-SNP linear regression with the LOCO predictor
# subtracted from the phenotype.
echo "[regenie run] (3/3a) Step 2 quantitative association"
"${REGENIE}" \
    --step 2 \
    --bed "${DATA}" \
    --phenoFile "${PHENO}" \
    --phenoCol EarHT \
    --covarFile "${COVAR}" \
    --covarColList PC1,PC2 \
    --pred "${OUT}/step1_qt_pred.list" \
    --qt \
    --bsize 200 \
    --threads 1 \
    --out "${OUT}/step2_qt"

# Binary path: Firth-corrected logistic regression. We disable SPA so the
# comparison against TG's BinaryGLMM (PQL + score test, optionally + SPA)
# isolates the Firth penalty from the SPA tail correction. The TG test we
# pair this with sets `firth=True, use_spa=False` — same shape.
echo "[regenie run] (3/3b) Step 2 binary (Firth) association"
"${REGENIE}" \
    --step 2 \
    --bed "${DATA}" \
    --phenoFile "${PHENO}" \
    --phenoCol EarHT_bin \
    --covarFile "${COVAR}" \
    --covarColList PC1,PC2 \
    --pred "${OUT}/step1_bin_pred.list" \
    --bt \
    --firth --pThresh 0.01 \
    --bsize 200 \
    --threads 1 \
    --out "${OUT}/step2_bin"

# --- Cleanup the lowmem temp dir but leave outputs/* in place ----------------
rm -rf "${LOWMEM_DIR}"

echo "[regenie run] all reference outputs produced in ${OUT}"
ls "${OUT}"
