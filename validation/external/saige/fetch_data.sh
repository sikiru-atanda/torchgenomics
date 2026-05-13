#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the SAIGE harness.
#
# Strategy: re-use the MDP fileset already converted to PLINK BED for the
# PLINK 2 harness, run the same MAF/geno QC the regenie harness uses, then
# generate SAIGE-format phenotype + covariate tables via simulate_phenotype.py.
#
# Why MDP (281 samples, ~3000 SNPs, polyploid maize) instead of a
# biobank-scale dataset?
#   - SAIGE was designed for N >> 10k case-control studies. On N=281 the
#     PCG-based variance ratio estimator can be near-degenerate; SAIGE
#     defaults to a sparse-GRM approximation that's algorithmically
#     similar to TG's BinaryGLMM PQL path on small N.
#   - We accept this and tune the harness as a structural agreement check
#     (β/SE/p sign + rank + magnitude) rather than a full numerical
#     equivalence check. The README documents that the small-N gates are
#     looser than the §16 LMM contract.
#   - Re-using a single fixture across all Pillar B harnesses gives clean
#     cross-tool consistency.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Source files are committed fixtures (no external download).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

PLINK2_DATA="${HERE}/../plink2/data"
PLINK2_BIN="${HERE}/../plink2/bin/plink2"

# --- Source fixture paths ----------------------------------------------------
SRC_BED="${PLINK2_DATA}/mdp.bed"
SRC_BIM="${PLINK2_DATA}/mdp.bim"
SRC_FAM="${PLINK2_DATA}/mdp.fam"

for f in "${SRC_BED}" "${SRC_BIM}" "${SRC_FAM}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[saige fetch] ABORT: source fixture missing: ${f}"
        echo "[saige fetch] Run validation/external/plink2/fetch_data.sh first to convert MDP to PLINK BED."
        exit 1
    fi
done

if [[ ! -x "${PLINK2_BIN}" ]]; then
    echo "[saige fetch] ABORT: PLINK 2 binary missing at ${PLINK2_BIN}"
    echo "[saige fetch] Run validation/external/plink2/install.sh first (we use it for QC)."
    exit 1
fi

# --- Pre-flight (~10 MB working set, peak RAM trivial) -----------------------
preflight_check_with_data_size "saige-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

DST_PHENO="${DATA_DIR}/saige_pheno.tsv"
DST_PREFIX="${DATA_DIR}/sample"
DST_BED="${DST_PREFIX}.bed"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${DST_BED}" && -f "${DST_PHENO}" ]]; then
    echo "[saige fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[saige fetch]   $(wc -l < "${DST_PREFIX}.fam") samples, "\
"$(wc -l < "${DST_PREFIX}.bim") variants"
    exit 0
fi

# --- QC the BED fileset ------------------------------------------------------
# SAIGE's Step 1 PCG iteration becomes unstable on monomorphic / very-rare
# variants. We mirror the regenie harness QC: --maf 0.01 --geno 0.1.
echo "[saige fetch] running PLINK 2 QC (--maf 0.01 --geno 0.1) on MDP → ${DST_PREFIX}"
"${PLINK2_BIN}" \
    --bfile "${PLINK2_DATA}/mdp" \
    --maf 0.01 \
    --geno 0.1 \
    --make-bed \
    --threads 1 \
    --out "${DST_PREFIX}" >/dev/null
rm -f "${DST_PREFIX}.log"

# --- Generate SAIGE-format phenotype + covariate file ------------------------
echo "[saige fetch] generating SAIGE-format phenotype (logistic on causal SNPs + PC1)"
python3 "${HERE}/simulate_phenotype.py" \
    --plink-prefix "${DST_PREFIX}" \
    --output-dir "${DATA_DIR}" \
    --seed 42 \
    --n-causal 5 \
    --beta 0.5

# --- Sanity ------------------------------------------------------------------
echo "[saige fetch] outputs:"
ls -la "${DATA_DIR}"
echo
echo "[saige fetch] phenotype head:"
head -3 "${DST_PHENO}"
