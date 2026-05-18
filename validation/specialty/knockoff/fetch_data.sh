#!/usr/bin/env bash
# Stage simulated genotype + phenotype replicates for the knockoff harness.
#
# We do NOT pull from a public reference panel:
#   - The knockoff filter is a generic FDR-control procedure - it does not
#     need real biology to test correctness. A block-LD simulation with a
#     known causal set tests every code path in both R `knockoff::knockoff.filter`
#     and `torchgwas.models.knockoff_lmm.KnockoffLMM`.
#   - Real-data injection adds confounding sources (population structure,
#     family relatedness, allele-flip orientation) that would mix into the
#     FDR estimate and defeat the comparison.
#
# The same fixture is read by both run_torchgwas.py and run_reference.R, so
# the only source of disagreement is the FDR-control algorithm itself, not
# the data prep.
#
# Idempotent: if data/replicates.npz already exists, exits 0.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"
FIXTURE="${DATA_DIR}/replicates.npz"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${FIXTURE}" && -f "${TRUTH_JSON}" ]]; then
    echo "[knockoff fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[knockoff fetch]   fixture: ${FIXTURE}"
    echo "[knockoff fetch]   truth:   ${TRUTH_JSON}"
    exit 0
fi

# --- Pre-flight --------------------------------------------------------------
# 100 replicates x (n=500, p=200) float64 ~ 80 MB.  Headroom: 1 GB / 1 GB.
preflight_check_with_data_size "knockoff-fetch" 1 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "[knockoff fetch] ABORT: python3 not on PATH (needed for generate.py)."
    exit 1
fi

mkdir -p "${DATA_DIR}"

echo "[knockoff fetch] simulating replicate genotype/phenotype fixtures"
python3 "${HERE}/generate.py" \
    --output-dir "${DATA_DIR}" \
    --n-replicates 100 \
    --n 500 \
    --p 200 \
    --block-size 10 \
    --k-causal 8 \
    --effect 1.0 \
    --noise-sd 0.3 \
    --within-block-rho 0.7 \
    --seed 42

echo "[knockoff fetch] done."
echo "[knockoff fetch]   fixture: ${FIXTURE}"
echo "[knockoff fetch]   truth:   ${TRUTH_JSON}"
