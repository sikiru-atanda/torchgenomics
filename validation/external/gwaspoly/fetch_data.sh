#!/usr/bin/env bash
# Fetch + stage the tetraploid potato fixture for the GWASpoly harness.
#
# The GWASpoly fixture (957 genotypes × 9888 markers, 6 environments, 1 trait
# 'vine.maturity') ships in two forms in the canonical repo:
#
#   1. Source data (input to GWASpoly):
#        benchmark/gwaspoly_data/new_potato_geno.csv
#        benchmark/gwaspoly_data/new_potato_pheno.csv
#
#   2. Aligned data + reference outputs (produced by run_gwaspoly.R):
#        benchmark/gwaspoly_results/potato_geno_aligned.csv
#        benchmark/gwaspoly_results/potato_pheno_with_genoid.csv
#        benchmark/gwaspoly_results/potato_map.csv
#        benchmark/gwaspoly_results/gwaspoly_<model>.csv  (5 gene-action models)
#        benchmark/gwaspoly_results/gwaspoly_kinship.csv
#
# This script copies BOTH sets into data/ — the source for re-running
# GWASpoly when available, the aligned set for compare.py.
#
# Idempotent: if data/ already contains the fixture, exit 0 immediately.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Source paths ------------------------------------------------------------
SOURCE_DATA_DIR=""
for c in \
    "${ROOT}/benchmark/gwaspoly_data" \
    "${HOME}/Documents/GWAS_Expert/benchmark/gwaspoly_data"; do
    if [[ -f "${c}/new_potato_geno.csv" ]]; then
        SOURCE_DATA_DIR="${c}"
        break
    fi
done
SOURCE_RESULTS_DIR=""
for c in \
    "${ROOT}/benchmark/gwaspoly_results" \
    "${HOME}/Documents/GWAS_Expert/benchmark/gwaspoly_results"; do
    if [[ -f "${c}/gwaspoly_additive.csv" ]]; then
        SOURCE_RESULTS_DIR="${c}"
        break
    fi
done

if [[ -z "${SOURCE_DATA_DIR}" || -z "${SOURCE_RESULTS_DIR}" ]]; then
    echo "[gwaspoly fetch] ABORT: cannot find canonical GWASpoly fixture."
    echo "[gwaspoly fetch] Looked for:"
    echo "[gwaspoly fetch]   benchmark/gwaspoly_data/new_potato_geno.csv"
    echo "[gwaspoly fetch]   benchmark/gwaspoly_results/gwaspoly_additive.csv"
    echo "[gwaspoly fetch] in:"
    echo "[gwaspoly fetch]   ${ROOT}/"
    echo "[gwaspoly fetch]   ${HOME}/Documents/GWAS_Expert/"
    exit 1
fi

# --- Pre-flight (~50 MB on disk; trivial RAM) --------------------------------
preflight_check_with_data_size "gwaspoly-fetch" 2 2

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

# --- Idempotence -------------------------------------------------------------
required_files=(
    "potato_geno_aligned.csv"
    "potato_pheno_with_genoid.csv"
    "potato_map.csv"
)
all_present=1
for f in "${required_files[@]}"; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        all_present=0
        break
    fi
done
if (( all_present == 1 )); then
    echo "[gwaspoly fetch] data already staged in ${DATA_DIR}; skipping copy"
    exit 0
fi

# --- Copy aligned-results files (the input to compare.py) -------------------
echo "[gwaspoly fetch] copying aligned data from ${SOURCE_RESULTS_DIR}/ → ${DATA_DIR}/"
for f in potato_geno_aligned.csv potato_pheno_with_genoid.csv potato_map.csv; do
    src="${SOURCE_RESULTS_DIR}/${f}"
    dst="${DATA_DIR}/${f}"
    if [[ ! -f "${src}" ]]; then
        echo "[gwaspoly fetch] ABORT: missing source: ${src}"
        exit 1
    fi
    cp -f "${src}" "${dst}"
    actual=$(sha256sum "${dst}" | awk '{print $1}')
    echo "[gwaspoly fetch] FIRST-RUN HASH ${f}: ${actual}"
done

# --- Copy SOURCE GWASpoly inputs (for optional re-runs) ---------------------
echo "[gwaspoly fetch] copying GWASpoly source inputs from ${SOURCE_DATA_DIR}/ → ${DATA_DIR}/"
for f in new_potato_geno.csv new_potato_pheno.csv; do
    src="${SOURCE_DATA_DIR}/${f}"
    dst="${DATA_DIR}/${f}"
    if [[ -f "${src}" ]]; then
        cp -f "${src}" "${dst}"
    fi
done

# --- Summary -----------------------------------------------------------------
n_markers=$(($(wc -l < "${DATA_DIR}/potato_map.csv") - 1))
n_pheno=$(($(wc -l < "${DATA_DIR}/potato_pheno_with_genoid.csv") - 1))
echo "[gwaspoly fetch] done. ${n_pheno} pheno rows, ${n_markers} markers"
ls -la "${DATA_DIR}"
