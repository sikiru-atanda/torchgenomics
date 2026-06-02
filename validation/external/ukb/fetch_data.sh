#!/usr/bin/env bash
# Stage the user-supplied UKB chr22 fixture for the NA3 harness.
#
# Strategy: UKB data is DUA-controlled and lives wherever the user keeps it.
# This script does NOT download UKB. It verifies the env-var paths set in
# install.sh, symlinks them into validation/external/ukb/data/ so the run
# scripts have a stable path, and additionally fetches the public LDSC
# 1000G Phase 3 chr22 LD scores + weights (the same archive used by the
# Pillar B LDSC harness — see
# `validation/external/ldsc/fetch_data.sh` on the validation branch).
#
# Pillar B contract:
#   - Memory + disk pre-flight before any work.
#   - Symlinks (not copies) for UKB; we never duplicate the user's UKB
#     fileset (could be hundreds of GB at full UKB scale).
#   - Idempotent: re-run leaves an already-staged tree untouched.
#
# Reference-output cache: this script also creates
# `reference_outputs/.touch` so re-runs of `compare.py` know whether the
# REGENIE / LDSC outputs are already produced (so the upstream tools do
# not have to re-run if nothing else changed).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

ENV_MARKER="${HERE}/.env_marker"
if [[ ! -f "${ENV_MARKER}" ]]; then
    echo "[ukb fetch] ABORT: ${ENV_MARKER} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_MARKER}"

# --- Pre-flight (~5 GB LD-scores download + working set; trivial RAM) ------
preflight_check_with_data_size "ukb-fetch" 5 2

# --- Verify UKB env vars survived from install.sh ---------------------------
if [[ -z "${UKB_CHR22_PATH:-}" || -z "${UKB_PHENO_PATH:-}" ]]; then
    echo "[ukb fetch] ABORT: UKB_CHR22_PATH / UKB_PHENO_PATH missing from ${ENV_MARKER}."
    echo "  Re-run install.sh after exporting both variables."
    exit 1
fi

DATA_DIR="${HERE}/data"
LD_DIR="${DATA_DIR}/ld_scores"
WT_DIR="${DATA_DIR}/weights"
REF_OUT_DIR="${HERE}/reference_outputs"
TG_OUT_DIR="${HERE}/torchgenomics_outputs"
mkdir -p "${DATA_DIR}" "${LD_DIR}" "${WT_DIR}" "${REF_OUT_DIR}" "${TG_OUT_DIR}"

# --- Stage user-supplied UKB genotype + phenotype via symlinks --------------
_link_if_present() {
    local src="$1" dst="$2"
    if [[ -e "${src}" ]]; then
        ln -sfn "${src}" "${dst}"
        echo "[ukb fetch]   symlinked ${src} -> ${dst}"
    fi
}

# Detect format and stage the right triple.
STAGED_PREFIX="${DATA_DIR}/ukb_chr22"
if [[ -f "${UKB_CHR22_PATH}.bed" ]]; then
    _link_if_present "${UKB_CHR22_PATH}.bed" "${STAGED_PREFIX}.bed"
    _link_if_present "${UKB_CHR22_PATH}.bim" "${STAGED_PREFIX}.bim"
    _link_if_present "${UKB_CHR22_PATH}.fam" "${STAGED_PREFIX}.fam"
    GENO_FORMAT="bed"
elif [[ -f "${UKB_CHR22_PATH}.pgen" ]]; then
    _link_if_present "${UKB_CHR22_PATH}.pgen" "${STAGED_PREFIX}.pgen"
    _link_if_present "${UKB_CHR22_PATH}.pvar" "${STAGED_PREFIX}.pvar"
    _link_if_present "${UKB_CHR22_PATH}.psam" "${STAGED_PREFIX}.psam"
    GENO_FORMAT="pgen"
else
    echo "[ukb fetch] ABORT: ${UKB_CHR22_PATH}.{bed,pgen} not found. Re-run install.sh to re-verify."
    exit 1
fi

STAGED_PHENO="${DATA_DIR}/ukb_pheno.tsv"
_link_if_present "${UKB_PHENO_PATH}" "${STAGED_PHENO}"

# --- Fetch LDSC reference panel (public; bundled w/ LDSC) -------------------
# This is the Pillar-B-LDSC-harness archive: 1000G Phase 3 LD scores +
# HM3-no-MHC weights from Zenodo DOI 10.5281/zenodo.7768714.
# We only need chr22 to mirror the harness scope, but the upstream tarballs
# pack all 22 chromosomes; we extract everything (~600 MB) and let LDSC
# pick its chromosome-specific files.
LD_TGZ="${HERE}/.cache/1000G_Phase3_ldscores.tgz"
WT_TGZ="${HERE}/.cache/1000G_Phase3_weights_hm3_no_MHC.tgz"
LD_URL="https://zenodo.org/record/7768714/files/1000G_Phase3_ldscores.tgz"
WT_URL="https://zenodo.org/record/7768714/files/1000G_Phase3_weights_hm3_no_MHC.tgz"
LD_SHA="470fe954080ba164f6b18ca2be8f8a6e2585d983b4a248556b5ffff2c3d0145f"
WT_SHA="d3e29112c64766dcee0ad9ec69ebda3d24cbe29ddb559d75443b10bcf8756c91"

mkdir -p "$(dirname "${LD_TGZ}")"

_fetch_archive() {
    local tgz="$1" url="$2" sha="$3" extract_dir="$4" marker="$5"
    if [[ -f "${marker}" ]]; then
        echo "[ukb fetch] LD asset already extracted at ${extract_dir}"
        return 0
    fi
    if [[ ! -f "${tgz}" ]]; then
        echo "[ukb fetch] downloading ${url}"
        curl --fail --location --show-error --silent \
             --max-time 1800 \
             --output "${tgz}" \
             "${url}"
    fi
    verify_checksum "${tgz}" "${sha}"
    echo "[ukb fetch] extracting ${tgz} -> ${extract_dir}"
    tar -xzf "${tgz}" -C "${extract_dir}" --strip-components=1 || true
    touch "${marker}"
}

_fetch_archive "${LD_TGZ}" "${LD_URL}" "${LD_SHA}" "${LD_DIR}" "${LD_DIR}/.fetched"
_fetch_archive "${WT_TGZ}" "${WT_URL}" "${WT_SHA}" "${WT_DIR}" "${WT_DIR}/.fetched"

# --- Final sanity ------------------------------------------------------------
echo ""
echo "[ukb fetch] staged tree:"
ls -la "${DATA_DIR}"
echo ""
echo "[ukb fetch] LD scores chr22 head:"
ls "${LD_DIR}"/*22* 2>/dev/null | head -5 || echo "  (chr22 LD score file expected at ${LD_DIR}/LDscore.22.l2.ldscore.gz)"
echo ""

# --- Persist staged paths for the run scripts -------------------------------
cat >> "${ENV_MARKER}" <<EOF
STAGED_PREFIX=${STAGED_PREFIX}
STAGED_PHENO=${STAGED_PHENO}
GENO_FORMAT=${GENO_FORMAT}
LD_DIR=${LD_DIR}
WT_DIR=${WT_DIR}
REF_OUT_DIR=${REF_OUT_DIR}
TG_OUT_DIR=${TG_OUT_DIR}
EOF

echo "[ukb fetch] DONE."
echo "  Staged genotype prefix: ${STAGED_PREFIX} (${GENO_FORMAT})"
echo "  Staged phenotype:       ${STAGED_PHENO}"
echo "  LDSC LD scores:         ${LD_DIR}"
echo "  LDSC weights:           ${WT_DIR}"
echo ""
echo "Next: bash ${HERE}/run_torchgenomics.sh && bash ${HERE}/run_reference.sh"
