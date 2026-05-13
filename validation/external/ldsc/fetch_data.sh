#!/usr/bin/env bash
# Fetch + stage LDSC reference data:
#   1. 1000G Phase 3 EUR LD scores (eur_w_ld_chr equivalent)  (~13 MB)
#   2. 1000G Phase 3 HM3 weights (no MHC)                     (~13 MB)
#
# Both come from the canonical Zenodo mirror published by Steven Gazal
# (DOI 10.5281/zenodo.7768714), which is the supported successor to the
# now-broken alkesgroup.broadinstitute.org S3 paths.
#
# We do NOT download the GIANT 2018 height sumstats — instead, we
# *simulate* sumstats from the real LD scores under a known h² + rg.
# Why: the GIANT 2018 archive is ~50 MB even before munging, and the
# UKB-scale full sumstats archive is 2.2 GB; simulating chi² statistics
# from the real LD scores keeps the harness fast (< 1 min total) while
# still exercising LDSC's full estimation path on realistic LD geometry.
#
# This is idempotent: if `data/ld_scores/` and `data/weights/` and
# `data/sim_trait1.sumstats.gz` already exist, exits 0.
#
# Pillar B contract:
#   - Memory + disk pre-flight before any download.
#   - SHA-256 verification on each upstream archive.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
CACHE_DIR="${HERE}/.cache"
mkdir -p "${DATA_DIR}" "${CACHE_DIR}"

# --- Pinned reference data ---------------------------------------------------
# Zenodo record 7768714 ("S-LDSC reference files", Steven Gazal, 2017-10-01).
# Files are version-frozen on Zenodo so the SHA-256 below is permanent.

LDSCORES_URL="https://zenodo.org/api/records/7768714/files/1000G_Phase3_ldscores.tgz/content"
LDSCORES_TGZ="${CACHE_DIR}/1000G_Phase3_ldscores.tgz"
# SHA-256 captured 2026-05-04 from a verified download.
LDSCORES_SHA256="470fe954080ba164f6b18ca2be8f8a6e2585d983b4a248556b5ffff2c3d0145f"

WEIGHTS_URL="https://zenodo.org/api/records/7768714/files/1000G_Phase3_weights_hm3_no_MHC.tgz/content"
WEIGHTS_TGZ="${CACHE_DIR}/1000G_Phase3_weights_hm3_no_MHC.tgz"
# SHA-256 captured 2026-05-04 from a verified download.
WEIGHTS_SHA256="d3e29112c64766dcee0ad9ec69ebda3d24cbe29ddb559d75443b10bcf8756c91"

LDSCORES_DIR="${DATA_DIR}/ld_scores"      # extracted: ./LDscore.<chr>.l2.ldscore.gz, etc.
WEIGHTS_DIR="${DATA_DIR}/weights"         # extracted: ./weights.hm3_noMHC.<chr>...
SUMSTATS_1="${DATA_DIR}/sim_trait1.sumstats.gz"
SUMSTATS_2="${DATA_DIR}/sim_trait2.sumstats.gz"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

# --- Idempotence -------------------------------------------------------------
if [[ -d "${LDSCORES_DIR}" && -d "${WEIGHTS_DIR}" \
      && -f "${SUMSTATS_1}" && -f "${SUMSTATS_2}" && -f "${TRUTH_JSON}" ]]; then
    echo "[ldsc fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[ldsc fetch]   ld_scores: $(ls "${LDSCORES_DIR}" | wc -l) files"
    echo "[ldsc fetch]   weights:   $(ls "${WEIGHTS_DIR}" | wc -l) files"
    echo "[ldsc fetch]   sumstats:  $(ls "${DATA_DIR}"/sim_*.sumstats.gz | wc -l) files"
    exit 0
fi

# --- Pre-flight (~30 MB downloaded, ~150 MB extracted, ~50 MB working set) ---
preflight_check_with_data_size "ldsc-fetch" 1 1

# --- Download archives -------------------------------------------------------
_download() {
    local url="$1" dest="$2" expected_sha="$3" label="$4"
    if [[ ! -f "${dest}" ]]; then
        echo "[ldsc fetch] downloading ${label} → ${dest}"
        curl --fail --location --show-error --silent \
            --max-time 600 \
            --output "${dest}" \
            "${url}"
    else
        echo "[ldsc fetch] cached: ${dest}"
    fi
    verify_checksum "${dest}" "${expected_sha}"
}

_download "${LDSCORES_URL}" "${LDSCORES_TGZ}" "${LDSCORES_SHA256}" "1000G_Phase3_ldscores.tgz (13 MB)"
_download "${WEIGHTS_URL}"  "${WEIGHTS_TGZ}"  "${WEIGHTS_SHA256}"  "1000G_Phase3_weights_hm3_no_MHC.tgz (13 MB)"

# --- Extract -----------------------------------------------------------------
mkdir -p "${LDSCORES_DIR}" "${WEIGHTS_DIR}"

# 1000G_Phase3_ldscores.tgz extracts to LDscore/LDscore.<chr>.l2.ldscore.gz etc.
# We re-house the inner directory as DATA/ld_scores so paths are stable.
echo "[ldsc fetch] extracting LD scores → ${LDSCORES_DIR}"
TMP_LD="$(mktemp -d)"
tar xzf "${LDSCORES_TGZ}" -C "${TMP_LD}"
# The archive contains a single directory called "LDscore"
mv "${TMP_LD}/LDscore"/* "${LDSCORES_DIR}/"
rmdir "${TMP_LD}/LDscore"
rmdir "${TMP_LD}"

echo "[ldsc fetch] extracting weights → ${WEIGHTS_DIR}"
TMP_W="$(mktemp -d)"
tar xzf "${WEIGHTS_TGZ}" -C "${TMP_W}"
# Archive contains "1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.<chr>..."
mv "${TMP_W}/1000G_Phase3_weights_hm3_no_MHC"/* "${WEIGHTS_DIR}/"
rmdir "${TMP_W}/1000G_Phase3_weights_hm3_no_MHC"
rmdir "${TMP_W}"

# --- Simulate sumstats from the real LD scores -------------------------------
# Two correlated traits (rg = 0.5) with h² = 0.4 each, N1 = 100 000, N2 = 80 000.
# We simulate at the union of HM3 SNPs covered by both the LD scores and
# the weights so LDSC has matching SNPs in all three inputs.
echo "[ldsc fetch] simulating sumstats (chr 22 only — keeps LDSC runtime ~5 s per run)"
python3 "${HERE}/simulate_sumstats.py" \
    --ldscores-dir "${LDSCORES_DIR}" \
    --weights-dir "${WEIGHTS_DIR}" \
    --output-dir "${DATA_DIR}" \
    --chrom 22 \
    --h2 0.4 \
    --rg 0.5 \
    --n1 100000 \
    --n2 80000 \
    --seed 42

echo "[ldsc fetch] done."
echo "[ldsc fetch]   ld_scores: $(ls "${LDSCORES_DIR}" | wc -l) files"
echo "[ldsc fetch]   weights:   $(ls "${WEIGHTS_DIR}" | wc -l) files"
echo "[ldsc fetch]   sumstats:  ${SUMSTATS_1} + ${SUMSTATS_2}"
echo "[ldsc fetch]   truth:     ${TRUTH_JSON}"
