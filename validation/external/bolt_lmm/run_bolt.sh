#!/usr/bin/env bash
# Run BOLT-LMM against the MDP fixture for the EarHT continuous trait.
#
# We use --lmmInfOnly (the textbook infinitesimal LMM) because:
#   1. The full BoltLMM-mixed approximation requires LD-scores that ship
#      only for hg19 / hg38 human assemblies. MDP is *Zea mays*; the
#      shipped LD-scores file is incompatible (chromosome IDs don't
#      match, and the LD pattern is different).
#   2. --lmmInfOnly is the closest BOLT analog to TG's SingleTraitLMM
#      with LOCO via grm_loco — both fit a single VC null and score per
#      SNP via the standard LMM Wald test. The "BoltLMM-mixed" mode adds
#      a Gaussian-mixture spread approximation that's specific to BOLT
#      and not a textbook construct; comparing TG to that would be a
#      methods comparison, not an equivalence check.
#
# BOLT performs LOCO internally (it always residualizes the test SNP's
# chromosome out of the GRM predictor in --lmmInf mode). The output stats
# file therefore corresponds to the LOCO-corrected per-SNP test, which we
# pair with TG's per-chromosome SingleTraitLMM + grm_loco.
#
# Idempotent: rerun overwrites previous outputs/.
#
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

BOLT="${HERE}/bin/bolt"
DATA_DIR="${HERE}/data"
DATA_PREFIX="${DATA_DIR}/mdp"
PHENO="${DATA_DIR}/bolt_pheno.tsv"
COVAR="${DATA_DIR}/bolt_covar.tsv"
OUT_DIR="${HERE}/outputs"
INSTALL_MARKER="${HERE}/.install_marker"
INFRA_BLOCKER="${HERE}/.infra_blocker"

if [[ -f "${INFRA_BLOCKER}" ]]; then
    echo "[bolt run] ABORT: BOLT-LMM install was marked infra-blocker."
    cat "${INFRA_BLOCKER}"
    echo "[bolt run] To proceed, re-run install.sh after fixing the underlying issue."
    exit 1
fi
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[bolt run] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi
if [[ ! -x "${BOLT}" ]]; then
    echo "[bolt run] ABORT: ${BOLT} not found. Run install.sh first."
    exit 1
fi
for ext in bed bim fam; do
    if [[ ! -f "${DATA_PREFIX}.${ext}" ]]; then
        echo "[bolt run] ABORT: ${DATA_PREFIX}.${ext} not found. Run fetch_data.sh first."
        exit 1
    fi
done
for f in "${PHENO}" "${COVAR}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[bolt run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

# Pre-flight: 1 GB working set, 4 GB peak RAM. BOLT on 281 × 2897 fits in
# < 200 MB but BOLT internally allocates large buffers; we leave headroom.
preflight_check_with_data_size "bolt-run" 1 4

mkdir -p "${OUT_DIR}"

# --- Run BOLT-LMM (--lmmInfOnly, the textbook infinitesimal LMM) ------------
# Notes on flags:
#   --bfile=...                   PLINK BED/BIM/FAM prefix.
#   --phenoFile=...               whitespace TSV with FID IID + traits.
#   --phenoCol=EarHT              continuous trait.
#   --covarFile=...               whitespace TSV with FID IID + covars.
#   --qCovarCol=PC1 --qCovarCol=PC2
#                                 quantitative covariates (one flag each).
#   --lmmInfOnly                  textbook infinitesimal LMM (no Gaussian-
#                                 mixture spread approximation, no LD-score
#                                 calibration). Output: BETA, SE, p-value
#                                 from the standard LMM score test with
#                                 internal LOCO.
#   --statsFile=...               main per-SNP output.
#   --numThreads=2                modest parallelism for the 281×2897 fixture.
#   --maxModelSnps=50000          guard rail (no effect at our scale).
#   --LDscoresMatchBp             match LD-scores by BP rather than RSID
#                                 (ignored under --lmmInfOnly but harmless).
#
# We do *not* pass --LDscoresFile because (a) --lmmInfOnly does not need
# LD-score calibration, and (b) the shipped LD-scores file in
# bin/tables/LDSCORE.1000G_EUR.tab.gz is for hg19 humans only. MDP maize
# chromosomes 1-10 do not match. Documenting this limitation in README.md.
echo "[bolt run] running BOLT-LMM v$(${BOLT} 2>&1 | head -1 | awk '{print $NF}' || echo unknown) "
echo "[bolt run] mode: --lmmInfOnly (textbook infinitesimal LMM)"

set +e
"${BOLT}" \
    --bfile="${DATA_PREFIX}" \
    --phenoFile="${PHENO}" \
    --phenoCol=EarHT \
    --covarFile="${COVAR}" \
    --qCovarCol=PC1 \
    --qCovarCol=PC2 \
    --lmmInfOnly \
    --statsFile="${OUT_DIR}/bolt_stats.tsv" \
    --numThreads=2 \
    2>&1 | tee "${OUT_DIR}/bolt.log"
RC="${PIPESTATUS[0]}"
set -e

if [[ "${RC}" -ne 0 ]]; then
    echo "[bolt run] BOLT-LMM exited with status ${RC}"
    echo "[bolt run] tail of log:"
    tail -30 "${OUT_DIR}/bolt.log"
    exit "${RC}"
fi

echo "[bolt run] outputs:"
ls -la "${OUT_DIR}"
echo
echo "[bolt run] stats head:"
head -3 "${OUT_DIR}/bolt_stats.tsv"
