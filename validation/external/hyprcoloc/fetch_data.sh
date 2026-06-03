#!/usr/bin/env bash
# Stage simulated 3-trait sumstats for the hyprcoloc harness.
#
# Why simulated (not real GWAS):
#   - The shared-causal-variant hypothesis is best tested with a *known*
#     architecture so disagreement reflects the algorithm, not the data prep.
#   - Real 3-trait fixtures (e.g., UKB lipid triplet + GTEx eQTL) require
#     OpenGWAS / external access, breaking reproducibility.
#   - Per Foley et al. 2021 the simplest informative fixture is 3 correlated
#     traits with one planted shared causal SNP. Cluster {T1,T2,T3} should
#     emerge with PP >= 0.95 in both R and TorchGenomics.
#
# Idempotent: skips if data/sumstats.tsv + data/sim_truth.json exist.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
SUMSTATS="${DATA_DIR}/sumstats.tsv"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${SUMSTATS}" && -f "${TRUTH_JSON}" ]]; then
    echo "[hyprcoloc fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[hyprcoloc fetch]   sumstats:  ${SUMSTATS}"
    echo "[hyprcoloc fetch]   truth:     ${TRUTH_JSON}"
    exit 0
fi

# --- Pre-flight (~1 KB output, peak RAM trivial) -----------------------------
preflight_check_with_data_size "hyprcoloc-fetch" 1 1

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[hyprcoloc fetch] ABORT: Rscript not on PATH (needed for simulate.R)."
    exit 1
fi

# --- Simulate ----------------------------------------------------------------
mkdir -p "${DATA_DIR}"

echo "[hyprcoloc fetch] simulating 3-trait sumstats with planted shared causal SNP"
Rscript --vanilla "${HERE}/simulate.R" \
    --output-dir "${DATA_DIR}" \
    --seed 42 \
    --m 100 \
    --k 3 \
    --causal-idx 50 \
    --causal-beta 0.5 \
    --background-sd 0.05 \
    --se 0.1

echo "[hyprcoloc fetch] done."
echo "[hyprcoloc fetch]   sumstats:  ${SUMSTATS}"
echo "[hyprcoloc fetch]   truth:     ${TRUTH_JSON}"

