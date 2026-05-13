#!/usr/bin/env bash
# Stage the SoyNAM RIL panel TSVs for the harness.
#
# We invoke `extract_data.R` to:
#   - Run `SoyNAM::BLUP(trait="yield", family=c(2,3,4,5), ...)` to get the
#     canonical preprocessed view (BLUPs + imputed genotypes + family vec).
#   - Subset to 4 families × ~140 RILs each (≈ 560 RILs) so the full
#     TG vs rrBLUP comparison runs in <5 minutes wall time and inside
#     the harness's memory pre-flight envelope.
#   - Emit geno.tsv / pheno.tsv / family.tsv / marker_map.tsv +
#     data_manifest.json with sha256 sums.
#
# We do NOT extract the full 40-family / 5590-RIL panel by default — that
# would push rrBLUP::GWAS over its 30-min wall-time budget on this host
# and consume ~20 GB RAM. The 4-family subset preserves the core
# multi-family RIL property (the only thing this harness needs to test
# WithinFamilyLMM dual-scan).
#
# Idempotent: if data/geno.tsv + data/pheno.tsv + data/family.tsv +
# data/marker_map.tsv already exist with the manifest sha256s, skip.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
GENO="${DATA_DIR}/geno.tsv"
PHENO="${DATA_DIR}/pheno.tsv"
FAMILY="${DATA_DIR}/family.tsv"
MAP="${DATA_DIR}/marker_map.tsv"
MANIFEST="${DATA_DIR}/data_manifest.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${GENO}" && -f "${PHENO}" && -f "${FAMILY}" && -f "${MAP}" && -f "${MANIFEST}" ]]; then
    echo "[soynam fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[soynam fetch]   geno:        ${GENO}"
    echo "[soynam fetch]   pheno:       ${PHENO}"
    echo "[soynam fetch]   family:      ${FAMILY}"
    echo "[soynam fetch]   marker_map:  ${MAP}"
    echo "[soynam fetch]   manifest:    ${MANIFEST}"
    exit 0
fi

# --- Pre-flight (BLUP on 4 families ≈ 560 RILs × 4000 SNPs uses ~2 GB RAM) --
preflight_check_with_data_size "soynam-fetch" 2 4

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[soynam fetch] ABORT: Rscript not on PATH (needed for extract_data.R)."
    exit 1
fi

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[soynam fetch] ABORT: ${INSTALL_MARKER} not found. Run install.sh first."
    exit 1
fi

# --- Extract -----------------------------------------------------------------
mkdir -p "${DATA_DIR}"

echo "[soynam fetch] extracting SoyNAM yield BLUP × 4-family panel"
Rscript --vanilla "${HERE}/extract_data.R" \
    --output-dir "${DATA_DIR}" \
    --trait yield \
    --families "2,3,4,5" \
    --maf 0.05 \
    --impute FM

echo "[soynam fetch] done."
echo "[soynam fetch]   geno:       ${GENO}"
echo "[soynam fetch]   pheno:      ${PHENO}"
echo "[soynam fetch]   family:     ${FAMILY}"
echo "[soynam fetch]   marker_map: ${MAP}"
echo "[soynam fetch]   manifest:   ${MANIFEST}"
