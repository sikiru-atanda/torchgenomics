#!/usr/bin/env bash
# SMR harness reference run --- sources ../_lib/preflight.sh (pre-flight gate).
#
# Two-step invocation:
#  1. SMR --make-besd  -->  builds the binary eQTL summary from probe.esd.
#  2. SMR --bfile --gwas-summary --beqtl-summary  -->  runs SMR + HEIDI.
# HEIDI inclusion knobs (Zhu 2016 Nat Genet 48:481 supplementary):
#   --peqtl-heidi 0.01     SNPs above this eQTL p threshold are excluded.
#   --heidi-min-m 3        require at least 3 SNPs for HEIDI to run.
#   --cis-wind 2000        cis window radius in kb.
# Allele-freq consistency relaxed (--diff-freq 0.9 --diff-freq-prop 0.9) because
# the simulated reference panel + GWAS share realised allele frequencies
# but SMR re-checks regardless.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[smr run] ABORT: ${INSTALL_MARKER} missing. Run install.sh first."
    exit 1
fi
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
REF_PREFIX="${DATA_DIR}/ref"
GWAS_MA="${DATA_DIR}/gwas.ma"
FLIST="${DATA_DIR}/probe.flist"
BESD_PREFIX="${OUT_DIR}/sim_eqtl"
SMR_RESULT_PREFIX="${OUT_DIR}/smr_results"
SMR_RESULT_FILE="${SMR_RESULT_PREFIX}.smr"

for f in "${REF_PREFIX}.bed" "${REF_PREFIX}.bim" "${REF_PREFIX}.fam" "${GWAS_MA}" "${FLIST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[smr run] ABORT: ${f} missing. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "smr-run" 1 1

mkdir -p "${OUT_DIR}"

echo "[smr run] step 1: build BESD from probe.esd"
(
    cd "${DATA_DIR}"
    "${SMR_BIN}" --eqtl-flist "probe.flist" --make-besd --out "${BESD_PREFIX}"
) > "${OUT_DIR}/make_besd.log" 2>&1
echo "[smr run]   wrote ${BESD_PREFIX}.{besd,esi,epi,summary}"

echo "[smr run] step 2: run SMR + HEIDI"
"${SMR_BIN}" \
    --bfile "${REF_PREFIX}" \
    --gwas-summary "${GWAS_MA}" \
    --beqtl-summary "${BESD_PREFIX}" \
    --peqtl-smr 1.0 \
    --peqtl-heidi 0.01 \
    --heidi-min-m 3 \
    --cis-wind 2000 \
    --diff-freq 0.9 \
    --diff-freq-prop 0.9 \
    --out "${SMR_RESULT_PREFIX}" > "${OUT_DIR}/smr_run.log" 2>&1

if [[ ! -f "${SMR_RESULT_FILE}" ]]; then
    echo "[smr run] ABORT: SMR did not produce ${SMR_RESULT_FILE}"
    tail -20 "${OUT_DIR}/smr_run.log"
    exit 1
fi

echo "[smr run] done. Output: ${SMR_RESULT_FILE}"
head -2 "${SMR_RESULT_FILE}"
