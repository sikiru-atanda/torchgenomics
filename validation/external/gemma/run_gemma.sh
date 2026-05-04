#!/usr/bin/env bash
# Run GEMMA against the MDP fixture to produce reference outputs:
#   1. Centered kinship (cXX) via -gk 1                  → outputs/mdp_kinship.cXX.txt
#   2. Single-trait LMM (Wald + LRT + score) via -lmm 4  → outputs/mdp_lmm_all.assoc.txt
#   3. Multi-trait mvLMM (Wald) via -lmm 1, -n 1 2       → outputs/mdp_mvlmm_wald.assoc.txt
#
# Idempotent: if all expected outputs already exist, exit 0 immediately. To
# force a re-run, delete outputs/ first.
#
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).
#
# Peak RAM: GEMMA on 276×3093 typically uses < 200 MB. We pre-flight 2 GB
# headroom for safety.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

GEMMA="${HERE}/bin/gemma"
DATA="${HERE}/data"
OUT="${HERE}/outputs"

if [[ ! -x "${GEMMA}" ]]; then
    echo "[gemma run] ABORT: ${GEMMA} not found. Run install.sh first."
    exit 1
fi

required_inputs=(
    "${DATA}/mdp_geno_bimbam.txt"
    "${DATA}/mdp_pheno1.txt"
    "${DATA}/mdp_pheno2.txt"
    "${DATA}/mdp_snps_bimbam.txt"
)
for f in "${required_inputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[gemma run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

# Pre-flight: 2 GB peak RAM (very conservative).
preflight_check_with_data_size "gemma-run" 1 2

mkdir -p "${OUT}"

# --- Idempotence: skip if all outputs already exist --------------------------
expected_outputs=(
    "${OUT}/mdp_kinship.cXX.txt"
    "${OUT}/mdp_lmm_all.assoc.txt"
    "${OUT}/mdp_lmm_all.log.txt"
    "${OUT}/mdp_mvlmm_wald.assoc.txt"
    "${OUT}/mdp_mvlmm_wald.log.txt"
)
all_present=1
for f in "${expected_outputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        all_present=0
        break
    fi
done
if (( all_present == 1 )); then
    echo "[gemma run] all outputs already present in ${OUT}; skipping"
    ls -la "${OUT}"
    exit 0
fi

# --- GEMMA writes outputs to ./output/ relative to invocation dir ------------
# We invoke from ${OUT}/.. so its "output/" subdir lands inside ${OUT}, then
# we flatten by moving files out.
WORK="${OUT}/_workdir"
mkdir -p "${WORK}"

cd "${WORK}"
ln -sf "${DATA}/mdp_geno_bimbam.txt" mdp_geno_bimbam.txt
ln -sf "${DATA}/mdp_pheno1.txt" mdp_pheno1.txt
ln -sf "${DATA}/mdp_pheno2.txt" mdp_pheno2.txt
ln -sf "${DATA}/mdp_snps_bimbam.txt" mdp_snps_bimbam.txt
mkdir -p output

echo "[gemma run] (1/3) computing centered kinship (-gk 1)"
"${GEMMA}" \
    -g mdp_geno_bimbam.txt \
    -p mdp_pheno1.txt \
    -a mdp_snps_bimbam.txt \
    -gk 1 \
    -o mdp_kinship 2>&1 | tail -5

echo "[gemma run] (2/3) single-trait LMM (-lmm 4: Wald + LRT + score)"
"${GEMMA}" \
    -g mdp_geno_bimbam.txt \
    -p mdp_pheno1.txt \
    -a mdp_snps_bimbam.txt \
    -k output/mdp_kinship.cXX.txt \
    -lmm 4 \
    -o mdp_lmm_all 2>&1 | tail -5

echo "[gemma run] (3/3) multi-trait mvLMM (-lmm 1: Wald, -n 1 2)"
"${GEMMA}" \
    -g mdp_geno_bimbam.txt \
    -p mdp_pheno2.txt \
    -a mdp_snps_bimbam.txt \
    -k output/mdp_kinship.cXX.txt \
    -n 1 2 \
    -lmm 1 \
    -o mdp_mvlmm_wald 2>&1 | tail -5

# --- Move outputs from output/ → ${OUT}/ -------------------------------------
mv -f output/mdp_kinship.cXX.txt "${OUT}/"
mv -f output/mdp_kinship.log.txt "${OUT}/"
mv -f output/mdp_lmm_all.assoc.txt "${OUT}/"
mv -f output/mdp_lmm_all.log.txt "${OUT}/"
mv -f output/mdp_mvlmm_wald.assoc.txt "${OUT}/"
mv -f output/mdp_mvlmm_wald.log.txt "${OUT}/"

# Clean up the workdir links
rm -rf "${WORK}"

echo "[gemma run] all outputs in ${OUT}:"
ls -la "${OUT}"
