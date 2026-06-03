#!/usr/bin/env bash
# OCF C6 harness orchestrator.
#
# Pipeline (paper-faithful, all stages deterministic given the same seed):
#   1. generate.R: simulate 100 replicates of the partially-linear DGP.
#   2. run_reference.R: hand-coded DML2 estimator (Chernozhukov 2018).
#   3. run_torchgenomics.py: TorchGenomics OCFLMM on the same replicates.
#   4. compare.py: empirical coverage + bias comparison + manifest hash.
#
# Output layout:
#   data/
#       fixture_meta.json
#       rep_001.rds .. rep_100.rds
#       rep_001/{Y,G,W,K_id}.csv .. rep_100/{...}.csv
#   outputs/
#       reference_results.tsv
#       reference_summary.json
#       torchgenomics_results.tsv
#       torchgenomics_summary.json
#   results/
#       summary.tsv
#       agreement.json
#       manifest.sha256

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

# Memory pre-flight (fixture is small; comfortable headroom).
preflight_check "ocf-run" 1 2

DATA="${HERE}/data"
OUT="${HERE}/outputs"
RES="${HERE}/results"
mkdir -p "${DATA}" "${OUT}" "${RES}"

# Defaults (overridable by env).
N="${N:-400}"
M="${M:-20}"
REPS="${REPS:-100}"
DW="${DW:-5}"
KFOLD="${KFOLD:-5}"
SEED="${SEED:-42}"
THETA0="${THETA0:-0.30}"
RIDGE="${RIDGE:-1e-2}"

echo "[ocf run] configuration: N=${N} M=${M} REPS=${REPS} DW=${DW} K=${KFOLD} SEED=${SEED} theta0=${THETA0} ridge=${RIDGE}"

# --- Stage 1: simulate fixture --------------------------------------------
echo "[ocf run] stage 1/4: generate fixture"
Rscript --vanilla "${HERE}/generate.R" \
    --n "${N}" --m "${M}" --reps "${REPS}" --dW "${DW}" \
    --theta0 "${THETA0}" --seed "${SEED}" --out-dir "${DATA}"

# --- Stage 2: reference (R hand-coded DML) --------------------------------
echo "[ocf run] stage 2/4: reference DML (R)"
Rscript --vanilla "${HERE}/run_reference.R" \
    --data-dir "${DATA}" --out-dir "${OUT}" \
    --k "${KFOLD}" --ridge "${RIDGE}" --seed "${SEED}"

# --- Stage 3: TorchGenomics OCFLMM --------------------------------------------
echo "[ocf run] stage 3/4: TorchGenomics OCFLMM"
python3 "${HERE}/run_torchgenomics.py" \
    --data-dir "${DATA}" --out-dir "${OUT}" \
    --k "${KFOLD}" --seed "${SEED}"

# --- Stage 4: compare ------------------------------------------------------
echo "[ocf run] stage 4/4: compare + emit results/"
python3 "${HERE}/compare.py" \
    --reference "${OUT}/reference_results.tsv" \
    --reference-summary "${OUT}/reference_summary.json" \
    --torchgenomics "${OUT}/torchgenomics_results.tsv" \
    --torchgenomics-summary "${OUT}/torchgenomics_summary.json" \
    --fixture-meta "${DATA}/fixture_meta.json" \
    --output-dir "${RES}"

echo "[ocf run] DONE.  See ${RES}/agreement.json + ${RES}/summary.tsv."
