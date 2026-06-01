#!/usr/bin/env bash
# Fetch upstream files for the plant multi-omics worked example (Paper F4 panel C).
#
# Option beta (selected): Arabidopsis 1001 Genomes + 1001 Transcriptomes + Atwell-2010.
# See README.md for option selection rationale + license verdict.
#
# This script:
#   1. Sources validation/external/_lib/preflight.sh and asserts >=10 GB free
#      disk on /home (per the B3 brief).
#   2. Downloads the three canonical upstream files to raw/.
#   3. Is idempotent: re-running with the files already present is a no-op.
#
# Networking note: the actual downloads are large (the SNP matrix is ~1.0 GB).
# If you only need the committed derived fixture, you do not need to run this
# script -- the parquet bundle in fixtures/ is committed to the repository.
# Re-run this only when you want to re-derive aligned.parquet from upstream.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

RAW_DIR="${HERE}/raw"
mkdir -p "${RAW_DIR}"

# --- Pre-flight: B3 brief requires >=10 GB free disk -------------------------
preflight_check "plant-multiomics-fetch" 10 2

# --- Canonical upstream files ------------------------------------------------
# Each entry: <relpath> <url>
# The 1001G consortium pins releases by version (v3.1); SHA256s for the
# committed derived fixture are in fixtures/manifest.sha256. Upstream SHA256s
# are intentionally not pinned here because the 1001G mirror occasionally
# re-bgzips files with slightly different framing, which changes the SHA256
# without changing the per-variant content.

GENO_URL="https://1001genomes.org/data/GMI-MPI/releases/v3.1/intersection_snp_short_indel_vcf/1001genomes_snp-short-indel_only_ACGTN.vcf.gz"
EXPR_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE80nnn/GSE80744/suppl/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz"
PHENO_URL="https://static-content.springer.com/esm/art%3A10.1038%2Fnature08800/MediaObjects/41586_2010_BFnature08800_MOESM286_ESM.xls"

GENO_FILE="${RAW_DIR}/1001genomes_snp-short-indel_only_ACGTN.vcf.gz"
EXPR_FILE="${RAW_DIR}/GSE80744_ath1001_tx_norm.tsv.gz"
PHENO_FILE="${RAW_DIR}/Atwell2010_FT10.xls"

# --- Helper: download with curl, idempotent ----------------------------------
download() {
    local url="$1"
    local out="$2"
    if [[ -f "${out}" && -s "${out}" ]]; then
        echo "[plant fetch] already present: ${out}"
        return 0
    fi
    echo "[plant fetch] downloading ${url}"
    echo "[plant fetch]   -> ${out}"
    curl -fSL -o "${out}.partial" "${url}"
    mv "${out}.partial" "${out}"
}

# --- Verify curl available ---------------------------------------------------
if ! command -v curl >/dev/null 2>&1; then
    echo "[plant fetch] ABORT: curl not on PATH (required for upstream downloads)."
    exit 1
fi

# --- Download all three layers ----------------------------------------------
download "${GENO_URL}"  "${GENO_FILE}"
download "${EXPR_URL}"  "${EXPR_FILE}"
download "${PHENO_URL}" "${PHENO_FILE}"

echo ""
echo "[plant fetch] done. Raw files in ${RAW_DIR}"
echo "[plant fetch] next step:"
echo "  python3 ${HERE}/prepare.py"
echo "[plant fetch] then verify the manifest:"
echo "  cd ${HERE}/fixtures && sha256sum -c manifest.sha256"
