#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the BOLT-LMM harness.
#
# Strategy: re-use the MDP fileset already converted to PLINK BED for the
# PLINK 2 harness (validation/external/plink2/data/mdp.{bed,bim,fam}).
# We copy those into our `data/` and run simulate_phenotype.py to produce
# a BOLT-format pheno + covar table.
#
# Why MDP and not a "biobank-scale" fixture:
#   - BOLT-LMM was designed for biobank-scale N (>>10k); on N=281 the
#     ridge-regression-based "BoltLMM-mixed" approximation operates well
#     outside its design regime. The harness uses --lmmInfOnly (textbook
#     infinitesimal model) which is closer to TG's SingleTraitLMM and
#     does not require the LD-scores tail-calibration step.
#   - Re-using a single fixture across all Pillar B harnesses gives
#     clean cross-tool consistency: every reference tool sees the *same*
#     genotype + phenotype data.
#
# BOLT-LMM input expectations:
#   --bfile=<prefix>     PLINK BED/BIM/FAM (we already have it).
#   --phenoFile=<tsv>    Whitespace TSV with FID, IID, then phenotype cols.
#   --phenoCol=<name>    Continuous trait. Missing values = "NA".
#   --covarFile=<tsv>    Optional; same shape (FID, IID, then covars).
#   --qCovarCol=<name>   One flag per quantitative covariate.
#
# Idempotent: if `data/mdp.{bed,bim,fam}` and `data/bolt_pheno.tsv`
# already exist, exit 0.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

PLINK2_DATA="${HERE}/../plink2/data"

# --- Source fixture paths ---------------------------------------------------
SRC_BED="${PLINK2_DATA}/mdp.bed"
SRC_BIM="${PLINK2_DATA}/mdp.bim"
SRC_FAM="${PLINK2_DATA}/mdp.fam"
SRC_PHENO="${PLINK2_DATA}/mdp_pheno.txt"

for f in "${SRC_BED}" "${SRC_BIM}" "${SRC_FAM}" "${SRC_PHENO}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[bolt fetch] ABORT: source fixture missing: ${f}"
        echo "[bolt fetch] Run validation/external/plink2/fetch_data.sh first to convert MDP to PLINK BED."
        exit 1
    fi
done

# --- Pre-flight (~10 MB working set, peak RAM trivial) ----------------------
preflight_check_with_data_size "bolt-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

DST_BED="${DATA_DIR}/mdp.bed"
DST_BIM="${DATA_DIR}/mdp.bim"
DST_FAM="${DATA_DIR}/mdp.fam"
DST_PHENO="${DATA_DIR}/bolt_pheno.tsv"
DST_COVAR="${DATA_DIR}/bolt_covar.tsv"

# --- Idempotence -----------------------------------------------------------
if [[ -f "${DST_BED}" && -f "${DST_BIM}" && -f "${DST_FAM}" \
      && -f "${DST_PHENO}" && -f "${DST_COVAR}" ]]; then
    echo "[bolt fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[bolt fetch]   $(wc -l < "${DST_FAM}") samples, $(wc -l < "${DST_BIM}") variants"
    exit 0
fi

# --- Stage + QC the BED fileset --------------------------------------------
# BOLT-LMM, like regenie, is allergic to monomorphic SNPs. The MDP fixture
# has a handful of effectively-monomorphic SNPs after our mean-imputation,
# so we run a quick PLINK 2 MAF filter before handing the fileset to BOLT.
# Match the regenie harness: --maf 0.01 --geno 0.1.
PLINK2_BIN="${HERE}/../plink2/bin/plink2"
if [[ ! -x "${PLINK2_BIN}" ]]; then
    echo "[bolt fetch] ABORT: PLINK 2 binary missing at ${PLINK2_BIN}"
    echo "[bolt fetch] Run validation/external/plink2/install.sh first (we use it for QC)."
    exit 1
fi

echo "[bolt fetch] running PLINK 2 QC (--maf 0.01 --geno 0.1) on MDP → ${DATA_DIR}/mdp"
"${PLINK2_BIN}" \
    --bfile "${PLINK2_DATA}/mdp" \
    --maf 0.01 \
    --geno 0.1 \
    --make-bed \
    --threads 1 \
    --out "${DATA_DIR}/mdp" >/dev/null
rm -f "${DATA_DIR}/mdp.log"

# --- Generate BOLT-format phenotype + covariate tables ----------------------
echo "[bolt fetch] generating BOLT-format phenotype + covariates"
python3 "${HERE}/simulate_phenotype.py" \
    --plink-prefix "${DATA_DIR}/mdp" \
    --src-pheno "${SRC_PHENO}" \
    --output-dir "${DATA_DIR}" \
    --seed 42

# --- Sanity ----------------------------------------------------------------
echo "[bolt fetch] outputs:"
ls -la "${DATA_DIR}"
echo
echo "[bolt fetch] phenotype head:"
head -3 "${DST_PHENO}"
echo
echo "[bolt fetch] covariate head:"
head -3 "${DST_COVAR}"
