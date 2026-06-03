#!/usr/bin/env bash
# Run LDSC against the simulated chr22 sumstats to produce reference outputs:
#   1. h² for trait 1                  → outputs/h2_trait1.log
#   2. h² for trait 2                  → outputs/h2_trait2.log
#   3. rg between trait 1 and trait 2  → outputs/rg.log
#
# All three runs use the same 1000G Phase 3 LD scores and weights files
# fetched by fetch_data.sh. Only chr22 LD scores are needed because we
# simulated chr22-only sumstats.
#
# Idempotent: rerun overwrites previous outputs/.
#
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).
#
# Peak RAM: LDSC on 17 K SNPs × 200 jackknife blocks uses < 500 MB. We
# pre-flight 4 GB headroom for safety (the Python 2.7 numpy stack carries
# its own footprint).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Locate the conda env ----------------------------------------------------
ENV_MARKER="${HERE}/.env_marker"
if [[ ! -f "${ENV_MARKER}" ]]; then
    echo "[ldsc run] ABORT: ${ENV_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_MARKER}"

DATA_DIR="${HERE}/data"
LDSCORES_DIR="${DATA_DIR}/ld_scores"
WEIGHTS_DIR="${DATA_DIR}/weights"
SUMSTATS_1="${DATA_DIR}/sim_trait1.sumstats.gz"
SUMSTATS_2="${DATA_DIR}/sim_trait2.sumstats.gz"
OUT="${HERE}/outputs"

for f in "${SUMSTATS_1}" "${SUMSTATS_2}" "${LDSCORES_DIR}" "${WEIGHTS_DIR}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[ldsc run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

# Pre-flight: 1 GB working set, 4 GB peak RAM. Actual LDSC peak is < 500 MB
# on 17 K SNPs but we leave generous headroom for the Python 2.7 numpy stack.
preflight_check_with_data_size "ldsc-run" 1 4

mkdir -p "${OUT}"

# LDSC's --ref-ld takes a single .l2.ldscore.gz prefix. Because we only
# simulated chr22 sumstats, we point LDSC at the chr22 LD score file only —
# `--ref-ld-chr` (which sums M across all 22 chromosomes from .M_5_50 files)
# would mis-scale h² by ~72× since our truth uses chr22-local M.
LDSCORE_PREFIX="${LDSCORES_DIR}/LDscore.22"
WEIGHTS_PREFIX="${WEIGHTS_DIR}/weights.hm3_noMHC.22"

# --- Activate env via subshell so failure modes propagate cleanly -----------
_run_ldsc() {
    (
        # `set +eu` because conda's compiler-toolchain hooks (gxx_linux-64
        # deactivate.d) trip `set -u` on CONDA_BACKUP_CXX. The same trick
        # used in install.sh.
        set +eu
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV_NAME}"
        set -e
        "$@"
    )
}

# --- Why --two-step 99999 ----------------------------------------------------
# LDSC's default `--h2` uses a two-step estimator (cutoff at 30):
#   step 1: WLS on all SNPs → estimate intercept
#   step 2: WLS on chi² < 30 SNPs with intercept FIXED → estimate slope
# TorchGenomics' `ldsc_h2` is structured differently:
#   step 1: WLS on all SNPs (warm-up; result discarded)
#   step 2: WLS on chi² < 30 SNPs, both slope & intercept refit
# These two-step parameterizations diverge when many SNPs have chi² > 30
# (which is our case: simulated mean chi² ≈ 45). To get an apples-to-apples
# reference, we run LDSC with `--two-step 99999` (no SNPs filtered out),
# i.e. single-pass WLS — which is what TorchGenomics' single-pass coef computes.
# This is documented in compare.py.

LDSC_COMMON=(
    --ref-ld "${LDSCORE_PREFIX}"
    --w-ld "${WEIGHTS_PREFIX}"
    --no-check-alleles
    --two-step 99999
)

# --- 1. h² for trait 1 -------------------------------------------------------
echo "[ldsc run] (1/3) h² for trait 1 (single-pass; --two-step 99999)"
_run_ldsc ldsc.py \
    --h2 "${SUMSTATS_1}" \
    "${LDSC_COMMON[@]}" \
    --out "${OUT}/h2_trait1"

# --- 2. h² for trait 2 -------------------------------------------------------
echo "[ldsc run] (2/3) h² for trait 2 (single-pass; --two-step 99999)"
_run_ldsc ldsc.py \
    --h2 "${SUMSTATS_2}" \
    "${LDSC_COMMON[@]}" \
    --out "${OUT}/h2_trait2"

# --- 3. rg between trait 1 and trait 2 --------------------------------------
echo "[ldsc run] (3/3) genetic correlation (single-pass)"
_run_ldsc ldsc.py \
    --rg "${SUMSTATS_1},${SUMSTATS_2}" \
    "${LDSC_COMMON[@]}" \
    --out "${OUT}/rg"

ls "${OUT}"/
echo "[ldsc run] all 3 reference outputs produced in ${OUT}"
