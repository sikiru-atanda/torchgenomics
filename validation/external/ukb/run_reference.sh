#!/usr/bin/env bash
# Run REGENIE Step 1 + Step 2 + LDSC h² on the same UKB chr22 fixture as
# run_torchgwas.sh.
#
# Comparison anchor (per Task NA3 spec; SESSION_HANDOFF lines 91-104):
#   - REGENIE quantitative-trait LMM (Step 1 ridge LOCO predictor + Step 2
#     per-SNP score test). Pin = REGENIE_VERSION_USED from .env_marker
#     (v4.2 if available, else v4.1; UNVERIFIED at scaffold time).
#   - LDSC 1.0.1 `--h2` on the REGENIE Step 2 output (REGENIE writes a
#     `.regenie` file we munge to LDSC sumstats format on the fly).
#
# Pillar B contract (mirrored from
# validation/external/regenie/run_regenie.sh +
# validation/external/ldsc/run_ldsc.sh on the validation branch):
#   - Memory pre-flight before each call.
#   - Outputs go to reference_outputs/ (cached; re-running compare.py
#     reuses these without re-invoking REGENIE / LDSC unless --force).
#   - Peak RSS recorded via `/usr/bin/time -v` for fair comparison vs
#     run_torchgwas.sh.
#
# Outputs:
#   reference_outputs/regenie_step1_pred.list  (LOCO predictor manifest)
#   reference_outputs/regenie_step2_<trait>.regenie
#   reference_outputs/regenie_step2.time.log
#   reference_outputs/ldsc_h2.log              (canonical LDSC log)
#   reference_outputs/ldsc_h2.time.log
#   reference_outputs/regenie_munged.sumstats.gz  (LDSC-format sumstats)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

ENV_MARKER="${HERE}/.env_marker"
if [[ ! -f "${ENV_MARKER}" ]]; then
    echo "[ukb ref-run] ABORT: ${ENV_MARKER} missing. Run install.sh + fetch_data.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_MARKER}"

: "${REGENIE_BIN:?install.sh did not write REGENIE_BIN}"
: "${LDSC_CONDA_ENV:?install.sh did not write LDSC_CONDA_ENV}"
: "${STAGED_PREFIX:?fetch_data.sh did not write STAGED_PREFIX}"
: "${STAGED_PHENO:?fetch_data.sh did not write STAGED_PHENO}"
: "${GENO_FORMAT:?fetch_data.sh did not write GENO_FORMAT}"
: "${REF_OUT_DIR:?fetch_data.sh did not write REF_OUT_DIR}"
: "${LD_DIR:?fetch_data.sh did not write LD_DIR}"
: "${WT_DIR:?fetch_data.sh did not write WT_DIR}"

UKB_TRAIT_COL="${UKB_TRAIT_COL:-PHENO}"
UKB_COVAR_COLS="${UKB_COVAR_COLS:-AGE,SEX,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10}"

# --- Pre-flight (REGENIE Step 1 is the largest peak; matches install.sh) ----
preflight_check "ukb-ref-run" 100 32

mkdir -p "${REF_OUT_DIR}"
LOWMEM_DIR="${REF_OUT_DIR}/.step1_tmp"
mkdir -p "${LOWMEM_DIR}"

# --- Idempotence cache: skip if all reference outputs exist + cached --------
CACHE_MARKER="${REF_OUT_DIR}/.cache_complete"
if [[ -f "${CACHE_MARKER}" && "${UKB_FORCE_REF:-0}" != "1" ]]; then
    echo "[ukb ref-run] cached reference outputs in ${REF_OUT_DIR}; skipping."
    echo "[ukb ref-run] Force re-run with: UKB_FORCE_REF=1 bash $0"
    ls "${REF_OUT_DIR}"
    exit 0
fi

# --- 1. REGENIE Step 1 (whole-genome ridge LOCO) ----------------------------
# Per Mbatchou 2021 §"Computational cost", Step 1 on UKB-scale (~500K samples)
# uses ~24 GB peak RSS with `--lowmem`. We use the same `--bsize 1000` as the
# UKB application paper.
echo "[ukb ref-run] (1/3) REGENIE Step 1 ridge LOCO predictor"

REGENIE_BED_FLAG=""
case "${GENO_FORMAT}" in
    bed)  REGENIE_BED_FLAG="--bed ${STAGED_PREFIX}" ;;
    pgen) REGENIE_BED_FLAG="--pgen ${STAGED_PREFIX}" ;;
    *)
        echo "[ukb ref-run] ABORT: unknown GENO_FORMAT=${GENO_FORMAT}"
        exit 1
        ;;
esac

# Comma-separated covariate list expected by REGENIE.
STEP1_TIME_LOG="${REF_OUT_DIR}/regenie_step1.time.log"
/usr/bin/time -v -o "${STEP1_TIME_LOG}" \
    "${REGENIE_BIN}" \
        --step 1 \
        ${REGENIE_BED_FLAG} \
        --phenoFile "${STAGED_PHENO}" \
        --phenoCol "${UKB_TRAIT_COL}" \
        --covarFile "${STAGED_PHENO}" \
        --covarColList "${UKB_COVAR_COLS}" \
        --bsize 1000 \
        --qt \
        --lowmem \
        --lowmem-prefix "${LOWMEM_DIR}/qt_" \
        --threads "${UKB_THREADS:-4}" \
        --out "${REF_OUT_DIR}/regenie_step1" \
    || {
        echo "[ukb ref-run] FAIL: REGENIE Step 1 exited non-zero"
        tail -40 "${STEP1_TIME_LOG}" || true
        exit 1
    }

echo "[ukb ref-run]   Step 1 peak RSS:"
grep -E "Maximum resident|Elapsed.*wall" "${STEP1_TIME_LOG}" | sed 's/^/    /'

# --- 2. REGENIE Step 2 (per-SNP score test) ---------------------------------
echo ""
echo "[ukb ref-run] (2/3) REGENIE Step 2 per-SNP association"

STEP2_TIME_LOG="${REF_OUT_DIR}/regenie_step2.time.log"
/usr/bin/time -v -o "${STEP2_TIME_LOG}" \
    "${REGENIE_BIN}" \
        --step 2 \
        ${REGENIE_BED_FLAG} \
        --phenoFile "${STAGED_PHENO}" \
        --phenoCol "${UKB_TRAIT_COL}" \
        --covarFile "${STAGED_PHENO}" \
        --covarColList "${UKB_COVAR_COLS}" \
        --pred "${REF_OUT_DIR}/regenie_step1_pred.list" \
        --qt \
        --bsize 400 \
        --threads "${UKB_THREADS:-4}" \
        --out "${REF_OUT_DIR}/regenie_step2" \
    || {
        echo "[ukb ref-run] FAIL: REGENIE Step 2 exited non-zero"
        tail -40 "${STEP2_TIME_LOG}" || true
        exit 1
    }

REGENIE_OUT="${REF_OUT_DIR}/regenie_step2_${UKB_TRAIT_COL}.regenie"
if [[ ! -f "${REGENIE_OUT}" ]]; then
    echo "[ukb ref-run] ABORT: expected ${REGENIE_OUT} not produced."
    ls "${REF_OUT_DIR}"
    exit 1
fi
echo "[ukb ref-run]   Step 2 output: ${REGENIE_OUT} ($(wc -l < "${REGENIE_OUT}") lines)"
echo "[ukb ref-run]   Step 2 peak RSS:"
grep -E "Maximum resident|Elapsed.*wall" "${STEP2_TIME_LOG}" | sed 's/^/    /'

# --- 3. Munge REGENIE output to LDSC sumstats + run LDSC --h2 ---------------
# REGENIE writes columns: CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE
#                         CHISQ LOG10P
# LDSC sumstats expects: SNP A1 A2 Z N (and optional CHISQ).
echo ""
echo "[ukb ref-run] (3/3) LDSC h² on REGENIE Step 2 output"

MUNGED="${REF_OUT_DIR}/regenie_munged.sumstats.gz"
python3 - "${REGENIE_OUT}" "${MUNGED}" <<'PY'
import sys
import pandas as pd
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, sep=r"\s+", engine="python")
# REGENIE may emit per-trait or `TEST` column. We keep ADD rows only.
if "TEST" in df.columns:
    df = df[df["TEST"] == "ADD"].copy()
df["Z"] = df["BETA"].astype(float) / df["SE"].astype(float)
out = pd.DataFrame({
    "SNP":  df["ID"].astype(str),
    "A1":   df["ALLELE1"].astype(str),
    "A2":   df["ALLELE0"].astype(str),
    "Z":    df["Z"],
    "N":    df["N"].astype(int) if "N" in df.columns else len(df) * [-1],
    "CHISQ": df["CHISQ"].astype(float) if "CHISQ" in df.columns
              else (df["BETA"].astype(float) / df["SE"].astype(float)) ** 2,
})
out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Z", "N"])
out.to_csv(dst, sep="\t", compression="gzip", index=False)
print(f"[munge] wrote {dst} ({len(out)} variants)")
PY

LD_PATH="${LD_DIR}/LDscore.22.l2.ldscore.gz"
WT_PATH="${WT_DIR}/weights.hm3_noMHC.22.l2.ldscore.gz"
M_PATH="${LD_DIR}/LDscore.22.l2.M_5_50"
LDSC_OUT="${REF_OUT_DIR}/ldsc_h2"
LDSC_TIME_LOG="${REF_OUT_DIR}/ldsc_h2.time.log"

for f in "${LD_PATH}" "${WT_PATH}" "${M_PATH}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ukb ref-run] ABORT: ${f} missing. Re-run fetch_data.sh."
        exit 1
    fi
done

# Activate the conda env for LDSC and run --h2 with --two-step 99999 to
# match TG's single-pass IRWLS (mirrors validation/external/ldsc/run_ldsc.sh
# on the validation branch — see compare.py docstring for why).
/usr/bin/time -v -o "${LDSC_TIME_LOG}" bash -c "
    set +eu
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate \"${LDSC_CONDA_ENV}\"
    set -e
    ldsc.py \
        --h2 \"${MUNGED}\" \
        --ref-ld \"${LD_DIR}/LDscore.22\" \
        --w-ld \"${WT_DIR}/weights.hm3_noMHC.22\" \
        --no-check-alleles \
        --two-step 99999 \
        --out \"${LDSC_OUT}\"
" || {
    echo "[ukb ref-run] FAIL: LDSC --h2 exited non-zero"
    tail -40 "${LDSC_TIME_LOG}" || true
    exit 1
}

if [[ ! -f "${LDSC_OUT}.log" ]]; then
    echo "[ukb ref-run] ABORT: LDSC log ${LDSC_OUT}.log not produced."
    exit 1
fi
echo "[ukb ref-run]   LDSC log: ${LDSC_OUT}.log"
grep -E "Maximum resident|Elapsed.*wall" "${LDSC_TIME_LOG}" | sed 's/^/    /'

# --- Cleanup + cache marker --------------------------------------------------
rm -rf "${LOWMEM_DIR}"
touch "${CACHE_MARKER}"

echo ""
echo "[ukb ref-run] DONE. Reference outputs cached in ${REF_OUT_DIR}."
echo "Next: python3 ${HERE}/compare.py"
