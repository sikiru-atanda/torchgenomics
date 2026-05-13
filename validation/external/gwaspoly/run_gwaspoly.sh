#!/usr/bin/env bash
# Run GWASpoly against the tetraploid potato fixture to produce reference
# outputs across 5 polyploid gene-action models:
#   1. gwaspoly_additive.csv
#   2. gwaspoly_1_dom_alt.csv  (+ 1_dom_ref.csv)
#   3. gwaspoly_2_dom_alt.csv  (+ 2_dom_ref.csv)
#   4. gwaspoly_diplo_additive.csv  + gwaspoly_diplo_general.csv
#   5. gwaspoly_general.csv
# Plus the kinship matrix and aligned data.
#
# IF GWASpoly is installed in R, we re-run end-to-end.
# IF GWASpoly is NOT available, we fall back to the canonical pre-Pillar-A
# reference outputs at benchmark/gwaspoly_results/ — these are the
# GWASpoly-generated values that the existing golden test (and the §16
# floor) was calibrated against.
#
# Idempotent: if outputs/ already contains the expected CSVs, exit 0.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA="${HERE}/data"
OUT="${HERE}/outputs"
RLIB="${HERE}/bin/Rlib"
VERSION_FILE="${HERE}/bin/gwaspoly_version.txt"

required_inputs=(
    "${DATA}/potato_geno_aligned.csv"
    "${DATA}/potato_pheno_with_genoid.csv"
    "${DATA}/potato_map.csv"
)
for f in "${required_inputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[gwaspoly run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "gwaspoly-run" 2 4

mkdir -p "${OUT}"

expected_outputs=(
    "${OUT}/gwaspoly_additive.csv"
    "${OUT}/gwaspoly_1_dom_alt.csv"
    "${OUT}/gwaspoly_2_dom_alt.csv"
    "${OUT}/gwaspoly_2_dom_ref.csv"
    "${OUT}/gwaspoly_diplo_additive.csv"
)
all_present=1
for f in "${expected_outputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        all_present=0
        break
    fi
done
if (( all_present == 1 )); then
    echo "[gwaspoly run] all outputs already present in ${OUT}; skipping"
    ls -la "${OUT}"
    exit 0
fi

# --- Decide: re-run GWASpoly, or fall back? ---------------------------------
# By default we use the committed pre-Pillar-A reference outputs at
# benchmark/gwaspoly_results/ — GWASpoly's serial per-marker LMM on 9888
# markers across 5 gene-action models takes 10-15 minutes, which is too
# slow for routine harness runs. To force a fresh re-run, set
# GWASPOLY_FORCE_RERUN=1.
gwaspoly_available=0
force_rerun="${GWASPOLY_FORCE_RERUN:-0}"

if [[ "${force_rerun}" == "1" ]]; then
    if [[ -f "${VERSION_FILE}" ]] && grep -q '^FOUND' "${VERSION_FILE}" 2>/dev/null; then
        gwaspoly_available=1
    fi
    if (( gwaspoly_available == 0 )) && command -v Rscript >/dev/null 2>&1; then
        if RLIB="${RLIB}" Rscript -e '
local_lib <- Sys.getenv("RLIB", unset = NA)
if (!is.na(local_lib) && nzchar(local_lib) && dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))
if (requireNamespace("GWASpoly", quietly = TRUE)) { cat("FOUND GWASpoly"); quit(status=0) }
quit(status=1)
' >/dev/null 2>&1; then
            gwaspoly_available=1
        fi
    fi
fi

if (( gwaspoly_available == 1 )); then
    echo "[gwaspoly run] GWASpoly detected — re-running end-to-end"
    RLIB="${RLIB}" Rscript "${HERE}/run_gwaspoly.R" "${DATA}" "${OUT}" 2>&1 | tail -40
else
    REF_DIR=""
    for c in \
        "${ROOT}/benchmark/gwaspoly_results" \
        "${HOME}/Documents/GWAS_Expert/benchmark/gwaspoly_results"; do
        if [[ -f "${c}/gwaspoly_additive.csv" ]]; then
            REF_DIR="${c}"
            break
        fi
    done
    if [[ -z "${REF_DIR}" ]]; then
        echo "[gwaspoly run] ABORT: GWASpoly not installed AND committed reference outputs not found."
        exit 1
    fi

    if [[ "${force_rerun}" == "1" ]]; then
        echo "[gwaspoly run] GWASpoly not installed despite GWASPOLY_FORCE_RERUN=1"
        echo "[gwaspoly run]   falling back to committed pre-Pillar-A reference outputs"
    else
        echo "[gwaspoly run] using committed pre-Pillar-A reference outputs (default)"
        echo "[gwaspoly run]   to force a fresh GWASpoly re-run, set GWASPOLY_FORCE_RERUN=1"
    fi
    echo "[gwaspoly run]   source: ${REF_DIR}"
    for f in \
        gwaspoly_additive.csv \
        gwaspoly_1_dom_alt.csv gwaspoly_1_dom_ref.csv \
        gwaspoly_2_dom_alt.csv gwaspoly_2_dom_ref.csv \
        gwaspoly_diplo_additive.csv gwaspoly_diplo_general.csv \
        gwaspoly_general.csv \
        gwaspoly_kinship.csv; do
        if [[ -f "${REF_DIR}/${f}" ]]; then
            cp -f "${REF_DIR}/${f}" "${OUT}/${f}"
            echo "[gwaspoly run]   copied ${f}"
        fi
    done
fi

# --- Verify outputs ----------------------------------------------------------
for f in "${expected_outputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[gwaspoly run] ABORT: expected output not produced: ${f}"
        exit 1
    fi
done

echo "[gwaspoly run] all outputs in ${OUT}:"
ls -la "${OUT}" | head -15
