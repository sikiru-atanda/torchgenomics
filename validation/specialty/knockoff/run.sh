#!/usr/bin/env bash
# Run both knockoff filters (TorchGWAS + R knockoff) on the same fixture
# and compute agreement.  Mirrors the pillar B/specialty contract:
#   - Pre-flight gate before any heavy compute.
#   - Idempotent: re-running overwrites previous outputs/.
#
# Output:
#   outputs/torchgwas_per_replicate.tsv
#   outputs/torchgwas_summary.json
#   outputs/reference_per_replicate.tsv
#   outputs/reference_summary.json
#   results/summary.tsv
#   results/agreement.json
#   results/manifest.sha256
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[knockoff run] ABORT: ${INSTALL_MARKER} not found.  Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
RESULTS_DIR="${HERE}/results"

if [[ ! -f "${DATA_DIR}/replicates.npz" || ! -f "${DATA_DIR}/sim_truth.json" ]]; then
    echo "[knockoff run] ABORT: data fixture not staged. Run fetch_data.sh first."
    exit 1
fi

# --- Pre-flight: peak working set per replicate is ~ n*p*8 = 500*200*8 ~ 0.8 MB
# x 100 replicates loaded into Python = 80 MB; R holds at most 1 replicate at a
# time (~1 MB).  Conservative budget: 2 GB disk / 4 GB RAM.
preflight_check_with_data_size "knockoff-run" 1 4

mkdir -p "${OUT_DIR}" "${RESULTS_DIR}"

# --- 1. TorchGWAS scan -------------------------------------------------------
echo "[knockoff run] step 1/3: TorchGWAS KnockoffLMM"
python3 "${HERE}/run_torchgwas.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}" \
    --target-fdr 0.2 \
    --ld-method r2 \
    --ld-window 20 \
    --r2-threshold 0.3 \
    --aggregation max_stat

# --- 2. R knockoff reference -------------------------------------------------
echo "[knockoff run] step 2/3: R knockoff::knockoff.filter"
Rscript --vanilla "${HERE}/run_reference.R" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}" \
    --target-fdr 0.2

# --- 3. Compare --------------------------------------------------------------
echo "[knockoff run] step 3/3: compare + agreement gate"
python3 "${HERE}/compare.py" \
    --tg-summary "${OUT_DIR}/torchgwas_summary.json" \
    --ref-summary "${OUT_DIR}/reference_summary.json" \
    --tg-per-rep "${OUT_DIR}/torchgwas_per_replicate.tsv" \
    --ref-per-rep "${OUT_DIR}/reference_per_replicate.tsv" \
    --truth "${DATA_DIR}/sim_truth.json" \
    --results-dir "${RESULTS_DIR}" \
    --target-fdr 0.2

# --- 4. SHA256 manifest ------------------------------------------------------
echo "[knockoff run] writing SHA256 manifest for results/"
(
    cd "${RESULTS_DIR}"
    sha256sum summary.tsv agreement.json 2>/dev/null > manifest.sha256
)
echo "[knockoff run] manifest ->"
sed 's/^/  /' "${RESULTS_DIR}/manifest.sha256"

echo "[knockoff run] done. results -> ${RESULTS_DIR}"
