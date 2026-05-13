#!/usr/bin/env bash
# Stage simulated MR sumstats for the TwoSampleMR harness.
#
# We do NOT pull from OpenGWAS:
#   - OpenGWAS endpoints (`extract_instruments`, `extract_outcome_data`)
#     require an API token and an internet connection at every CI run,
#     and the underlying datasets shift over time as the IEU pipeline
#     re-harmonizes meta-analyses. This breaks reproducibility.
#   - The MR estimator is *summary-statistic only* — it does not need
#     real biology to test correctness. Simulated data with a planted
#     causal effect tests every code path in TwoSampleMR's `mr()` and
#     MRPRESSO's `mr_presso()`.
#
# Idempotent: if data/sumstats.tsv and data/sim_truth.json already exist,
# exits 0 without touching them.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
SUMSTATS="${DATA_DIR}/sumstats.tsv"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${SUMSTATS}" && -f "${TRUTH_JSON}" ]]; then
    echo "[twosamplemr fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[twosamplemr fetch]   sumstats:  ${SUMSTATS}"
    echo "[twosamplemr fetch]   truth:     ${TRUTH_JSON}"
    exit 0
fi

# --- Pre-flight (~1 KB sumstats, peak RAM trivial) ---------------------------
preflight_check_with_data_size "twosamplemr-fetch" 1 1

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[twosamplemr fetch] ABORT: Rscript not on PATH (needed for simulate_mr.R)."
    exit 1
fi

# --- Simulate ----------------------------------------------------------------
mkdir -p "${DATA_DIR}"

echo "[twosamplemr fetch] simulating MR sumstats with planted theta=0.5, K=30, 3 pleiotropic"
Rscript --vanilla "${HERE}/simulate_mr.R" \
    --output-dir "${DATA_DIR}" \
    --seed 42 \
    --k 30 \
    --theta 0.5 \
    --n-pleiotropic 3 \
    --pleio-sigma 0.15

echo "[twosamplemr fetch] done."
echo "[twosamplemr fetch]   sumstats:  ${SUMSTATS}"
echo "[twosamplemr fetch]   truth:     ${TRUTH_JSON}"
