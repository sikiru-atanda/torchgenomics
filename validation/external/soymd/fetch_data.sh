#!/usr/bin/env bash
# Stage the simulated multi-omics triple for the SoyMD harness.
#
# We do NOT pull from SoyMD's web portal:
#   - SoyMD (Soybean Multi-omics Database) is a published platform that
#     hosts per-cohort genotype + RNA-seq + phenotype tables; full datasets
#     are 100+ MB per cohort and require manual navigation through the
#     web UI to download. This breaks reproducibility under CI.
#   - The mediation estimator we test (`torchgenomics.multiomics.mediate_lmm`
#     vs `mediation::mediate`) is *triple-only* — it does not need real
#     biology to test correctness. Simulated data with a planted causal
#     effect tests every code path.
#   - The README's "Scaling to real SoyMD data" section documents how a
#     future user would swap in real SoyMD-derived TSVs.
#
# Idempotent: if data/triple.tsv, data/K.tsv, and data/sim_truth.json all
# exist, exits 0 without touching them.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
TRIPLE="${DATA_DIR}/triple.tsv"
KMAT="${DATA_DIR}/K.tsv"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${TRIPLE}" && -f "${KMAT}" && -f "${TRUTH_JSON}" ]]; then
    echo "[soymd fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[soymd fetch]   triple: ${TRIPLE}"
    echo "[soymd fetch]   K:      ${KMAT}"
    echo "[soymd fetch]   truth:  ${TRUTH_JSON}"
    exit 0
fi

# --- Pre-flight (~few KB total, peak RAM trivial) ----------------------------
preflight_check_with_data_size "soymd-fetch" 1 1

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[soymd fetch] ABORT: Rscript not on PATH (needed for simulate_multiomics.R)."
    exit 1
fi

# --- Simulate ----------------------------------------------------------------
mkdir -p "${DATA_DIR}"

echo "[soymd fetch] simulating multi-omics triple with planted ACME=0.20, total=0.30"
Rscript --vanilla "${HERE}/simulate_multiomics.R" \
    --output-dir "${DATA_DIR}" \
    --seed 42 \
    --n 200 \
    --a 0.5 \
    --b 0.4 \
    --c-prime 0.1

echo "[soymd fetch] done."
echo "[soymd fetch]   triple: ${TRIPLE}"
echo "[soymd fetch]   K:      ${KMAT}"
echo "[soymd fetch]   truth:  ${TRUTH_JSON}"
