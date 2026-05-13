#!/usr/bin/env bash
# Run TorchGWAS LMM scan + LDSC h² estimate on the UKB chr22 fixture.
#
# Streaming-first invariants (per `feedback_streaming` memory):
#   - lmm-scan reads chunks via `torchgwas.io.PlinkBedReader.iter_chunks` —
#     never materializes the full (n × m) matrix.
#   - GRM built via `grm_vanraden_streaming` (chunked accumulation).
#   - Default chunk size kept at the CLI default (1024 SNPs); the streaming
#     audit (`docs/efficiency/streaming_audit.md` on the validation branch)
#     documents this as the canonical biobank-scale-safe configuration.
#
# Peak-RSS measurement: `/usr/bin/time -v` records `Maximum resident set
# size` for both `lmm-scan` and the LDSC sub-call. We log to a side file the
# `compare.py` parses and ledger.
#
# Outputs:
#   torchgwas_outputs/lmm_chr22.assoc.tsv
#   torchgwas_outputs/lmm_chr22.time.log    (peak RSS + wall time)
#   torchgwas_outputs/h2_chr22.json         (h² point estimate + SE + extras)
#   torchgwas_outputs/h2_chr22.time.log
#
# Pre-flight (per-run): ≥ 100 GB disk free; ≥ 32 GB RAM (matches install.sh
# gate; this is the largest peak in the harness).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

ENV_MARKER="${HERE}/.env_marker"
if [[ ! -f "${ENV_MARKER}" ]]; then
    echo "[ukb tg-run] ABORT: ${ENV_MARKER} missing. Run install.sh + fetch_data.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_MARKER}"

# Required env-var fall-throughs
: "${STAGED_PREFIX:?fetch_data.sh did not write STAGED_PREFIX}"
: "${STAGED_PHENO:?fetch_data.sh did not write STAGED_PHENO}"
: "${GENO_FORMAT:?fetch_data.sh did not write GENO_FORMAT}"
: "${TG_OUT_DIR:?fetch_data.sh did not write TG_OUT_DIR}"
: "${LD_DIR:?fetch_data.sh did not write LD_DIR}"
: "${WT_DIR:?fetch_data.sh did not write WT_DIR}"

# Trait + covariate config: user can override; defaults assume the phenotype
# file's first non-{FID,IID} column is the trait, no covariates beyond age + sex
# + 10 PCs (UKB convention). This is a scaffold; the user adapts to their
# real phenotype-table schema.
UKB_TRAIT_COL="${UKB_TRAIT_COL:-PHENO}"
UKB_COVAR_COLS="${UKB_COVAR_COLS:-AGE,SEX,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10}"

# --- Pre-flight (per-run; matches install.sh gate) --------------------------
preflight_check "ukb-tg-run" 100 32

mkdir -p "${TG_OUT_DIR}"

LMM_OUT="${TG_OUT_DIR}/lmm_chr22"
LMM_TIME_LOG="${TG_OUT_DIR}/lmm_chr22.time.log"
H2_OUT="${TG_OUT_DIR}/h2_chr22.json"
H2_TIME_LOG="${TG_OUT_DIR}/h2_chr22.time.log"

# --- Resolve genotype path for the CLI --------------------------------------
case "${GENO_FORMAT}" in
    bed)  GENO_ARG="${STAGED_PREFIX}.bed" ;;
    pgen) GENO_ARG="${STAGED_PREFIX}.pgen" ;;
    *)
        echo "[ukb tg-run] ABORT: unknown GENO_FORMAT=${GENO_FORMAT}"
        exit 1
        ;;
esac

# --- 1. TorchGWAS LMM scan (streaming) --------------------------------------
# Streaming-first invariants:
#   - PlinkBedReader.iter_chunks default chunk_size (1024 SNPs at default)
#     yields per-chunk peak ≈ n × 1024 × 8 B = 4 GB at n=500K.
#   - K (n × n) is the dominant always-resident allocation: at n=500K,
#     K float64 = 1 TB — out of scope for a 32 GB host. The user is
#     EXPECTED to subsample to n ≤ 50K (≈ 20 GB K) or supply a
#     pre-computed K via --kinship (out-of-scope flag for this scaffold).
#     Document this as a known limitation in README.md.
#   - GRM built via streaming variant (no full G in memory).
echo "[ukb tg-run] (1/2) TorchGWAS lmm-scan on ${GENO_ARG}"
echo "[ukb tg-run]   trait: ${UKB_TRAIT_COL}; covariates: ${UKB_COVAR_COLS}"

# Slice the user's phenotype TSV into a separate covariate TSV that
# lmm-scan can consume via --covariate (the real flag takes a FILE
# path, not column names). UKB_COVAR_COLS is the comma-separated list
# from the user; we extract FID + IID + those columns into a sibling
# file.
COVAR_TSV="${TG_OUT_DIR}/covar_subset.tsv"
UKB_PHENO_PATH="${STAGED_PHENO}" \
UKB_COVAR_COLS="${UKB_COVAR_COLS}" \
COVAR_TSV_OUT="${COVAR_TSV}" \
python3 - <<'PYEOF'
import os
import pandas as pd

pheno_path = os.environ["UKB_PHENO_PATH"]
covar_cols = os.environ.get("UKB_COVAR_COLS", "").strip()
out_path = os.environ["COVAR_TSV_OUT"]

df = pd.read_csv(pheno_path, sep=None, engine="python")

required_id_cols = ["FID", "IID"]
if not all(c in df.columns for c in required_id_cols):
    raise SystemExit(
        f"phenotype TSV must contain FID + IID columns; got {list(df.columns)}"
    )

if not covar_cols:
    # No covariates declared; emit FID + IID only and let --covariate
    # downstream just provide ID-aligned rows
    keep = required_id_cols
else:
    requested = [c.strip() for c in covar_cols.split(",") if c.strip()]
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise SystemExit(
            f"phenotype TSV is missing covariate columns: {missing}; "
            f"available: {list(df.columns)}"
        )
    keep = required_id_cols + requested

df[keep].to_csv(out_path, sep="\t", index=False)
print(f"Wrote covariate subset to {out_path} with columns: {keep}")
PYEOF

# `/usr/bin/time -v` records peak RSS in the time log; we strip it out
# downstream in compare.py.
/usr/bin/time -v -o "${LMM_TIME_LOG}" \
    torchgwas lmm-scan \
        --genotype "${GENO_ARG}" \
        --phenotype "${STAGED_PHENO}" \
        --traits "${UKB_TRAIT_COL}" \
        --covariate "${COVAR_TSV}" \
        --test wald \
        --output "${LMM_OUT}" \
    || {
        echo "[ukb tg-run] FAIL: lmm-scan exited non-zero"
        echo "[ukb tg-run] last 20 lines of time log:"
        tail -20 "${LMM_TIME_LOG}" || true
        exit 1
    }

echo "[ukb tg-run]   wrote ${LMM_OUT}.assoc.tsv ($(wc -l < "${LMM_OUT}.assoc.tsv") lines)"
echo "[ukb tg-run]   peak RSS:"
grep -E "Maximum resident|Elapsed.*wall" "${LMM_TIME_LOG}" | sed 's/^/    /'

# --- 2. TorchGWAS LDSC h² estimate ------------------------------------------
# `torchgwas` ships `compute_ld_scores` and `ldsc_h2`. There isn't currently a
# single CLI subcommand wrapper for "sumstats -> h²" (the LDSC subcommand
# count was 0 in the v0.3.8 CLI matrix), so we drive it from a small inline
# Python script that:
#   1. Reads the lmm-scan output (z = beta / se).
#   2. Loads chr22 LD scores + weights from the public 1000G archive.
#   3. Calls `torchgwas.postgwas.ldsc_h2(chi2, ld_scores, w_ld, n, m_total)`.
#   4. Dumps the LDSCResult fields to JSON.
# This mirrors the `validation/external/ldsc/compare.py` code path on the
# validation branch, scaled up to UKB chr22.
echo ""
echo "[ukb tg-run] (2/2) TorchGWAS h² (chi² + LD-score regression on chr22)"

# We need the chr22 LD score + weight files; the canonical names from the
# 1000G Phase 3 archive used by the validation-branch LDSC harness:
LD_PATH="${LD_DIR}/LDscore.22.l2.ldscore.gz"
WT_PATH="${WT_DIR}/weights.hm3_noMHC.22.l2.ldscore.gz"
M_PATH="${LD_DIR}/LDscore.22.l2.M_5_50"
for f in "${LD_PATH}" "${WT_PATH}" "${M_PATH}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ukb tg-run] ABORT: ${f} not found. Re-run fetch_data.sh."
        exit 1
    fi
done

# Inline driver: small, self-contained, no separate file needed.
TG_PY_DRIVER="${TG_OUT_DIR}/.driver_h2.py"
cat > "${TG_PY_DRIVER}" <<'PY'
"""h² driver: lmm-scan output + 1000G LD scores -> LDSC h² estimate.

Mirrors validation/external/ldsc/compare.py (validation-pillar-A-coverage
branch) input handling. Outputs JSON with h² / h²_SE / intercept / mean χ²
/ n_snps / m_total / wall clock.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--lmm-tsv", required=True)
ap.add_argument("--ld-path", required=True)
ap.add_argument("--w-path", required=True)
ap.add_argument("--m-path", required=True)
ap.add_argument("--n-samples", type=int, required=True,
                help="GWAS sample size (post-QC); REGENIE/TG should agree.")
ap.add_argument("--out-json", required=True)
args = ap.parse_args()

# Parent torchgwas import path:
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from torchgwas.postgwas import ldsc_h2  # noqa: E402

lmm = pd.read_csv(args.lmm_tsv, sep="\t")
# Defensive: trait column scheme varies by CLI version; SNP+CHISQ work in
# every observed case. Fall back to (BETA/SE)² if CHISQ absent.
if "CHISQ" in lmm.columns:
    chi2_np = lmm["CHISQ"].astype(np.float64).to_numpy()
else:
    chi2_np = (lmm["BETA"].astype(np.float64).to_numpy()
               / lmm["SE"].astype(np.float64).to_numpy()) ** 2

# SNP id column: TG writes `SNP` by default.
snps = lmm["SNP"].astype(str).tolist()

ld_df = pd.read_csv(args.ld_path, sep="\t", compression="gzip")
w_df = pd.read_csv(args.w_path, sep="\t", compression="gzip")
m_total = int(open(args.m_path).read().strip().split()[0])

ld_df["SNP"] = ld_df["SNP"].astype(str)
w_df["SNP"] = w_df["SNP"].astype(str)

merged = (
    pd.DataFrame({"SNP": snps, "CHISQ": chi2_np})
    .merge(ld_df[["SNP", "L2"]].rename(columns={"L2": "L2_ref"}), on="SNP", how="inner")
    .merge(w_df[["SNP", "L2"]].rename(columns={"L2": "L2_w"}), on="SNP", how="inner")
)
print(f"[h2-driver] LMM rows={len(lmm)}  LD-merged rows={len(merged)}", flush=True)

if len(merged) == 0:
    raise SystemExit("[h2-driver] FAIL: zero SNP overlap between lmm-scan output and LD-score panel")

chi2 = torch.tensor(merged["CHISQ"].to_numpy(dtype=np.float64), dtype=torch.float64)
ld = torch.tensor(merged["L2_ref"].to_numpy(dtype=np.float64), dtype=torch.float64)
w_ld = torch.tensor(merged["L2_w"].to_numpy(dtype=np.float64), dtype=torch.float64)

result = ldsc_h2(chi2, ld, n=args.n_samples, m_total=m_total, w_ld=w_ld)

payload = {
    "h2": float(result.h2),
    "h2_se": float(result.h2_se),
    "intercept": float(result.intercept),
    "intercept_se": float(result.intercept_se),
    "mean_chi2": float(result.mean_chi2),
    "lambda_gc": float(result.lambda_gc),
    "n_snps_used": int(len(merged)),
    "m_total": m_total,
    "n_samples": args.n_samples,
}
Path(args.out_json).write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY

# Sample size N = number of FID lines in the staged phenotype (drops header).
#
# Assumption: the staged TSV has EXACTLY one header line and no comment
# lines, no trailing blank lines. If your fixture is non-conforming
# (e.g. has '#' comment rows, or trailing newlines), `wc -l` will
# over-count and N will not match what REGENIE / TG actually consumed.
# We do a defensive sanity check below: N must lie between 100 and
# 1_000_000 (UKB-plausible). The harness exits 1 if not.
N_SAMPLES="$(( $(wc -l < "${STAGED_PHENO}") - 1 ))"
if (( N_SAMPLES <= 0 )); then
    echo "[ukb tg-run] ABORT: phenotype file ${STAGED_PHENO} has zero data rows."
    exit 1
fi
if (( N_SAMPLES < 100 )) || (( N_SAMPLES > 1000000 )); then
    echo "[ukb tg-run] ABORT: N_SAMPLES=${N_SAMPLES} from ${STAGED_PHENO} is outside the"
    echo "  UKB-plausible range [100, 1000000]. Check for comment lines, trailing"
    echo "  blank lines, or a missing/extra header row in your phenotype TSV."
    exit 1
fi

/usr/bin/time -v -o "${H2_TIME_LOG}" \
    python3 "${TG_PY_DRIVER}" \
        --lmm-tsv "${LMM_OUT}.assoc.tsv" \
        --ld-path "${LD_PATH}" \
        --w-path "${WT_PATH}" \
        --m-path "${M_PATH}" \
        --n-samples "${N_SAMPLES}" \
        --out-json "${H2_OUT}" \
    || {
        echo "[ukb tg-run] FAIL: TorchGWAS h² driver exited non-zero"
        tail -20 "${H2_TIME_LOG}" || true
        exit 1
    }

echo "[ukb tg-run]   h² JSON: ${H2_OUT}"
grep -E "Maximum resident|Elapsed.*wall" "${H2_TIME_LOG}" | sed 's/^/    /'

echo ""
echo "[ukb tg-run] DONE. Outputs in ${TG_OUT_DIR}."
echo "Next: bash ${HERE}/run_reference.sh"
