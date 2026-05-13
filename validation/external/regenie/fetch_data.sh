#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the regenie harness.
#
# Strategy: re-use the MDP fileset already converted to PLINK BED for the
# PLINK 2 harness (validation/external/plink2/data/mdp.{bed,bim,fam}). We
# symlink (or copy if symlinks fail) those into our `data/` and run
# simulate_phenotype.py to produce regenie-format pheno + covar tables.
#
# Why we re-use MDP:
#   - MDP has 281 samples × 3093 SNPs after QC. That is far below the scale
#     where regenie's Step 1 LOCO ridge predictor is statistically
#     meaningful (regenie was designed for biobank-scale N >> 10k). On this
#     fixture the LOCO predictor is essentially the per-block phenotype
#     mean. We accept that and document it: the harness exercises the
#     *API* (Step 1 + Step 2 pipeline + per-SNP β/SE/χ²/p extraction), not
#     the LOCO accuracy. The tolerance gates are loosened accordingly.
#   - Re-using a single fixture across all Pillar B harnesses means we
#     compare every reference tool against TG on the *same* genotype +
#     phenotype data, which gives clean cross-tool consistency.
#
# Idempotent: if `data/mdp.{bed,bim,fam}` and `data/regenie_pheno.tsv`
# already exist, exit 0.
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

# --- Source fixture paths ----------------------------------------------------
SRC_BED="${PLINK2_DATA}/mdp.bed"
SRC_BIM="${PLINK2_DATA}/mdp.bim"
SRC_FAM="${PLINK2_DATA}/mdp.fam"
SRC_PHENO="${PLINK2_DATA}/mdp_pheno.txt"

for f in "${SRC_BED}" "${SRC_BIM}" "${SRC_FAM}" "${SRC_PHENO}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[regenie fetch] ABORT: source fixture missing: ${f}"
        echo "[regenie fetch] Run validation/external/plink2/fetch_data.sh first to convert MDP to PLINK BED."
        exit 1
    fi
done

# --- Pre-flight (~10 MB working set, peak RAM trivial) -----------------------
preflight_check_with_data_size "regenie-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

DST_BED="${DATA_DIR}/mdp.bed"
DST_BIM="${DATA_DIR}/mdp.bim"
DST_FAM="${DATA_DIR}/mdp.fam"
DST_PHENO_REGENIE="${DATA_DIR}/regenie_pheno.tsv"
DST_COVAR_REGENIE="${DATA_DIR}/regenie_covar.tsv"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${DST_BED}" && -f "${DST_BIM}" && -f "${DST_FAM}" \
      && -f "${DST_PHENO_REGENIE}" && -f "${DST_COVAR_REGENIE}" ]]; then
    echo "[regenie fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[regenie fetch]   $(wc -l < "${DST_FAM}") samples, $(wc -l < "${DST_BIM}") variants"
    exit 0
fi

# --- Stage + QC the BED fileset ----------------------------------------------
# regenie aborts hard on monomorphic SNPs ("Uh-oh, SNP X has low variance").
# The MDP fixture has a handful of effectively-monomorphic SNPs after our
# mean-imputation, so we run a quick PLINK 2 MAF filter before handing the
# fileset to regenie.
PLINK2_BIN="${HERE}/../plink2/bin/plink2"
if [[ ! -x "${PLINK2_BIN}" ]]; then
    echo "[regenie fetch] ABORT: PLINK 2 binary missing at ${PLINK2_BIN}"
    echo "[regenie fetch] Run validation/external/plink2/install.sh first (we use it for QC)."
    exit 1
fi

# We use --maf 0.01 to drop low-frequency variants regenie would refuse,
# and --geno 0.1 to drop variants with > 10% missing genotype calls. These
# thresholds match the regenie tutorial's default QC for biobank scans.
echo "[regenie fetch] running PLINK 2 QC (--maf 0.01 --geno 0.1) on MDP → ${DATA_DIR}/mdp"
"${PLINK2_BIN}" \
    --bfile "${PLINK2_DATA}/mdp" \
    --maf 0.01 \
    --geno 0.1 \
    --make-bed \
    --threads 1 \
    --out "${DATA_DIR}/mdp" >/dev/null
# Drop the PLINK log file: it leaks the filtered-out variant list and we
# don't need it for downstream comparison.
rm -f "${DATA_DIR}/mdp.log"

# --- Generate regenie-format phenotype + covariate tables -------------------
echo "[regenie fetch] generating regenie-format phenotype + covariates"
python3 "${HERE}/simulate_phenotype.py" \
    --plink-prefix "${DATA_DIR}/mdp" \
    --src-pheno "${SRC_PHENO}" \
    --output-dir "${DATA_DIR}" \
    --seed 42

# --- Sanity ------------------------------------------------------------------
echo "[regenie fetch] outputs:"
ls -la "${DATA_DIR}"
echo
echo "[regenie fetch] phenotype head:"
head -3 "${DST_PHENO_REGENIE}"
echo
echo "[regenie fetch] covariate head:"
head -3 "${DST_COVAR_REGENIE}"
