#!/usr/bin/env bash
# Run GAPIT3 against the MDP fixture to produce 4 reference outputs:
#   1. GLM_GWAS.csv      (GLM, no kinship)
#   2. MLM_GWAS.csv      (MLM with VanRaden kinship + 3 PCs)
#   3. FarmCPU_GWAS.csv  (multi-locus iterative; default GAPIT params)
#   4. BLINK_GWAS.csv    (multi-locus LD-pruned; default GAPIT params)
#
# IF GAPIT3 is installed in R, we re-run end-to-end.
# IF GAPIT3 is NOT available, we fall back to the canonical pre-Pillar-A
# reference outputs at benchmark/gapit_results/ — these are the
# GAPIT-generated values that the existing golden test (and the §16 floor)
# was calibrated against.
#
# Idempotent: if outputs/ already contains all 4 CSVs, exit 0 immediately.
#
# Pillar B contract:
#   - Memory pre-flight before run.
#   - Outputs go to ${HERE}/outputs (.gitignored).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA="${HERE}/data"
OUT="${HERE}/outputs"
RLIB="${HERE}/bin/Rlib"
VERSION_FILE="${HERE}/bin/gapit_version.txt"

required_inputs=(
    "${DATA}/mdp_numeric.txt"
    "${DATA}/mdp_traits.txt"
    "${DATA}/mdp_SNP_information.txt"
)
for f in "${required_inputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[gapit run] ABORT: ${f} not found. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "gapit-run" 1 4

mkdir -p "${OUT}"

expected_outputs=(
    "${OUT}/GLM_GWAS.csv"
    "${OUT}/MLM_GWAS.csv"
    "${OUT}/FarmCPU_GWAS.csv"
    "${OUT}/BLINK_GWAS.csv"
)
all_present=1
for f in "${expected_outputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        all_present=0
        break
    fi
done
if (( all_present == 1 )); then
    echo "[gapit run] all outputs already present in ${OUT}; skipping"
    ls -la "${OUT}"
    exit 0
fi

# --- Decide: re-run GAPIT, or fall back to committed reference? --------------
gapit_available=0
if [[ -f "${VERSION_FILE}" ]]; then
    if grep -q '^FOUND' "${VERSION_FILE}" 2>/dev/null; then
        gapit_available=1
    fi
fi
# Even without install.sh having been run, probe directly
if (( gapit_available == 0 )) && command -v Rscript >/dev/null 2>&1; then
    PROBE_OUT=$(RLIB="${RLIB}" Rscript -e '
local_lib <- Sys.getenv("RLIB", unset = NA)
if (!is.na(local_lib) && nzchar(local_lib) && dir.exists(local_lib)) .libPaths(c(local_lib, .libPaths()))
for (pkg in c("GAPIT3", "GAPIT")) if (requireNamespace(pkg, quietly = TRUE)) { cat("FOUND ", pkg); quit(status=0) }
quit(status=1)
' 2>/dev/null) && gapit_available=1 || true
fi

if (( gapit_available == 1 )); then
    echo "[gapit run] GAPIT3 detected — re-running end-to-end"
    RLIB="${RLIB}" Rscript "${HERE}/run_gapit.R" "${DATA}" "${OUT}" 2>&1 | tail -30
else
    # Locate the canonical pre-Pillar-A reference outputs.
    REF_DIR=""
    for candidate in \
        "${ROOT}/benchmark/gapit_results" \
        "${HOME}/Documents/GWAS_Expert/benchmark/gapit_results"; do
        if [[ -f "${candidate}/GLM_GWAS.csv" ]]; then
            REF_DIR="${candidate}"
            break
        fi
    done
    if [[ -z "${REF_DIR}" ]]; then
        echo "[gapit run] ABORT: GAPIT3 not installed AND committed reference outputs not found."
        echo "[gapit run] Looked in:"
        echo "[gapit run]   ${ROOT}/benchmark/gapit_results/"
        echo "[gapit run]   ${HOME}/Documents/GWAS_Expert/benchmark/gapit_results/"
        echo "[gapit run] Either install GAPIT3 (see install.sh) or restore the"
        echo "[gapit run] benchmark/gapit_results/ canonical fixture."
        exit 1
    fi

    echo "[gapit run] GAPIT3 not installed — using committed pre-Pillar-A reference outputs"
    echo "[gapit run]   source: ${REF_DIR}"
    for f in GLM_GWAS.csv MLM_GWAS.csv FarmCPU_GWAS.csv BLINK_GWAS.csv; do
        if [[ -f "${REF_DIR}/${f}" ]]; then
            cp -f "${REF_DIR}/${f}" "${OUT}/${f}"
            echo "[gapit run]   copied ${f}"
        fi
    done
fi

# --- Verify outputs ----------------------------------------------------------
for f in "${expected_outputs[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[gapit run] ABORT: expected output not produced: ${f}"
        exit 1
    fi
done

echo "[gapit run] all outputs in ${OUT}:"
ls -la "${OUT}"
