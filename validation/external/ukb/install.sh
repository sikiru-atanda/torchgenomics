#!/usr/bin/env bash
# UKB-scale validation harness installer (Task NA3).
#
# Prepares the host for a single biobank-scale comparison:
#   - REGENIE v4.x binary (LMM Step 1 + Step 2; quantitative trait MVP).
#   - LDSC 1.0.1 (h² estimation from sumstats; conda env `ldsc_env`).
#   - Pre-flight: ≥ 100 GB free disk, ≥ 32 GB free RAM (per SESSION_HANDOFF
#     "few CPU-hours per run" guidance and the n=500K × m_chr22 working set).
#   - Verifies user-supplied UKB data paths via env vars (DOES NOT download
#     UKB; that data is DUA-controlled and the user supplies the path).
#
# Pillar B harness contract (mirrored from
# validation/external/{regenie,ldsc}/install.sh on the
# `validation/pillar-A-coverage` branch):
#   - Memory + disk pre-flight before any work
#     (`validation/external/_lib/preflight.sh`).
#   - Pinned reference-tool versions; abort on mismatch.
#   - Idempotent: re-running this script with everything in place exits 0
#     without re-downloading or re-installing.
#
# REGENIE pin policy (per Task NA3 spec, Phase 57 modernization):
#   - Try v4.2 first. If the upstream release is missing or unavailable
#     (UNVERIFIED at scaffold time — Anthropic knowledge cutoff Jan 2026),
#     fall back to v4.1. Both v4.x lines support biobank-scale Step 1 +
#     Step 2 quantitative LMM, which is what this harness exercises.
#   - The pin is set via the REGENIE_VERSION env var (override-able for the
#     user to bump as soon as a newer release is verified).
#
# UKB data prerequisites (user-supplied; we verify but never download):
#   - UKB_CHR22_PATH: path to chromosome-22 genotype prefix
#     (BED/BIM/FAM or PGEN/PVAR/PSAM). The script accepts either format.
#   - UKB_PHENO_PATH: path to a tab-separated phenotype table with columns
#     FID  IID  <trait>  [covariates ...]. Trait must be quantitative
#     (Gaussian) for the MVP (per Phase 57 modernization scope).
#
# Output side-effects:
#   - validation/external/ukb/bin/regenie  (downloaded if missing)
#   - conda env `ldsc_env`                  (created if missing)
#   - validation/external/ukb/.env_marker   (records pinned versions)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned versions --------------------------------------------------------
# REGENIE: try v4.2, fall back to v4.1. Override via env if needed.
REGENIE_VERSION_PRIMARY="${REGENIE_VERSION:-v4.2}"
REGENIE_VERSION_FALLBACK="v4.1"
LDSC_VERSION="1.0.1"
LDSC_BIOCONDA_BUILD="pyhdfd78af_2"
LDSC_CONDA_ENV="ldsc_env"

BIN_DIR="${HERE}/bin"
REGENIE_BIN="${BIN_DIR}/regenie"
ENV_MARKER="${HERE}/.env_marker"

# --- Pre-flight (HARD GATE; per `feedback_preflight` memory) ----------------
# Required at install time:
#   - 100 GB free disk on /home: REGENIE binary (~30 MB), LDSC env (~200 MB),
#     UKB chr22 LD scores + weights (~4 GB), Step 1 LOCO predictor cache
#     (~8 GB at n=500K), Step 2 outputs + working space (~10 GB), plus
#     ~50 GB headroom for re-runs and TG outputs.
#   - 32 GB RAM: REGENIE Step 1 ridge predictor on n=500K is the largest
#     peak; documented as ~24-28 GB by Mbatchou 2021 §"Computational cost"
#     for biobank-scale UKB Step 1. We add 4 GB headroom.
# preflight.sh aborts with a clear message if either is missing.
preflight_check "ukb-install" 100 32

# --- Verify user-supplied UKB data paths (DO NOT download) ------------------
# Per the harness scope: UKB is DUA-controlled. We verify the user has
# provided env vars pointing at locally-readable files; we never attempt to
# fetch UKB.
if [[ -z "${UKB_CHR22_PATH:-}" ]]; then
    echo "[ukb install] ABORT: UKB_CHR22_PATH not set."
    echo ""
    echo "  Set UKB_CHR22_PATH to the path prefix of your UKB chr22 genotype"
    echo "  fileset (without .bed/.bim/.fam or .pgen/.pvar/.psam suffix)."
    echo ""
    echo "  Example:"
    echo "    export UKB_CHR22_PATH=/path/to/ukb_imp_chr22_v3"
    echo "    export UKB_PHENO_PATH=/path/to/ukb_pheno.tsv"
    echo ""
    echo "  UKB data is DUA-controlled; this harness does not download it."
    exit 1
fi
if [[ -z "${UKB_PHENO_PATH:-}" ]]; then
    echo "[ukb install] ABORT: UKB_PHENO_PATH not set."
    echo "  Set UKB_PHENO_PATH to a tab-separated phenotype table with columns:"
    echo "    FID  IID  <trait_col>  [covariates ...]"
    exit 1
fi

# Detect format and verify file readability.
_check_genotype_files() {
    local prefix="$1"
    if [[ -f "${prefix}.bed" && -f "${prefix}.bim" && -f "${prefix}.fam" ]]; then
        echo "[ukb install] genotype format: PLINK 1 BED (${prefix}.{bed,bim,fam})"
        return 0
    fi
    if [[ -f "${prefix}.pgen" && -f "${prefix}.pvar" && -f "${prefix}.psam" ]]; then
        echo "[ukb install] genotype format: PLINK 2 PGEN (${prefix}.{pgen,pvar,psam})"
        return 0
    fi
    echo "[ukb install] ABORT: ${prefix}.{bed,bim,fam} or ${prefix}.{pgen,pvar,psam} not found."
    echo "  UKB_CHR22_PATH must point at a PLINK 1 (.bed/.bim/.fam) or PLINK 2 (.pgen/.pvar/.psam) prefix."
    return 1
}

_check_genotype_files "${UKB_CHR22_PATH}"

if [[ ! -r "${UKB_PHENO_PATH}" ]]; then
    echo "[ukb install] ABORT: UKB_PHENO_PATH=${UKB_PHENO_PATH} not readable."
    exit 1
fi
echo "[ukb install] phenotype file: ${UKB_PHENO_PATH} ($(wc -l < "${UKB_PHENO_PATH}") lines)"

# --- REGENIE install (idempotent) -------------------------------------------
mkdir -p "${BIN_DIR}"

_regenie_installed_version() {
    [[ -x "${REGENIE_BIN}" ]] || { echo ""; return; }
    "${REGENIE_BIN}" --version 2>/dev/null | head -1 || echo "unknown"
}

_install_regenie_attempt() {
    local version="$1"
    local url="https://github.com/rgcgithub/regenie/releases/download/${version}/regenie_${version}.gz_x86_64_Linux.zip"
    local zip_path="${HERE}/.cache/regenie_${version}.gz_x86_64_Linux.zip"

    mkdir -p "$(dirname "${zip_path}")"

    echo "[ukb install] attempting REGENIE ${version} from ${url}"
    if [[ ! -f "${zip_path}" ]]; then
        if ! curl --fail --location --show-error --silent \
                  --max-time 600 \
                  --output "${zip_path}" \
                  "${url}"; then
            echo "[ukb install]   download failed for ${version}"
            rm -f "${zip_path}"
            return 1
        fi
    else
        echo "[ukb install]   cached zip: ${zip_path}"
    fi

    local tmp_extract
    tmp_extract="$(mktemp -d)"
    if ! unzip -q -o "${zip_path}" -d "${tmp_extract}"; then
        echo "[ukb install]   unzip failed for ${version}"
        rm -rf "${tmp_extract}"
        return 1
    fi

    local extracted
    extracted="$(find "${tmp_extract}" -maxdepth 2 -type f -name 'regenie*' ! -name '*.zip' | head -1)"
    if [[ -z "${extracted}" || ! -f "${extracted}" ]]; then
        echo "[ukb install]   extracted archive missing regenie binary (${version})"
        rm -rf "${tmp_extract}"
        return 1
    fi

    mv -f "${extracted}" "${REGENIE_BIN}"
    chmod +x "${REGENIE_BIN}"
    rm -rf "${tmp_extract}"

    if ! "${REGENIE_BIN}" --version >/dev/null 2>&1; then
        echo "[ukb install]   ${version} binary failed --version smoke test"
        return 1
    fi
    echo "[ukb install]   REGENIE ${version} installed at ${REGENIE_BIN}"
    return 0
}

INSTALLED_REGENIE_VERSION="$(_regenie_installed_version)"
if [[ "${INSTALLED_REGENIE_VERSION}" == *"${REGENIE_VERSION_PRIMARY}"* ]] \
   || [[ "${INSTALLED_REGENIE_VERSION}" == *"${REGENIE_VERSION_FALLBACK}"* ]]; then
    echo "[ukb install] REGENIE already installed: ${INSTALLED_REGENIE_VERSION}"
    REGENIE_VERSION_USED="${INSTALLED_REGENIE_VERSION}"
else
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ;;
        *)
            echo "[ukb install] ABORT: only Linux x86_64 prebuilt REGENIE binaries are wired."
            echo "[ukb install] Detected: $(uname -s)-$(uname -m)"
            echo "[ukb install] Other platforms: build from source at https://github.com/rgcgithub/regenie"
            exit 1
            ;;
    esac

    REGENIE_VERSION_USED=""
    if _install_regenie_attempt "${REGENIE_VERSION_PRIMARY}"; then
        REGENIE_VERSION_USED="${REGENIE_VERSION_PRIMARY}"
    else
        echo "[ukb install] primary pin ${REGENIE_VERSION_PRIMARY} unavailable; falling back to ${REGENIE_VERSION_FALLBACK}"
        if _install_regenie_attempt "${REGENIE_VERSION_FALLBACK}"; then
            REGENIE_VERSION_USED="${REGENIE_VERSION_FALLBACK}"
        else
            echo "[ukb install] ABORT: neither REGENIE ${REGENIE_VERSION_PRIMARY} nor ${REGENIE_VERSION_FALLBACK} installed."
            echo "[ukb install] Manually drop a regenie binary at ${REGENIE_BIN}, then re-run."
            exit 1
        fi
    fi
fi

"${REGENIE_BIN}" --version || true

# --- LDSC install (idempotent; bioconda) ------------------------------------
_check_ldsc_env() {
    (
        set +eu
        # shellcheck disable=SC1091
        source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null
        conda activate "${LDSC_CONDA_ENV}" 2>/dev/null || exit 1
        command -v ldsc.py >/dev/null 2>&1 || exit 1
    )
}

if _check_ldsc_env; then
    echo "[ukb install] LDSC already installed in conda env '${LDSC_CONDA_ENV}'"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ukb install] ABORT: conda not on PATH."
        echo "[ukb install] LDSC 1.0.1 requires Python 2.7; bioconda is the only practical install path."
        echo "[ukb install] Install miniconda from https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo "[ukb install] creating conda env '${LDSC_CONDA_ENV}' with bioconda::ldsc=${LDSC_VERSION}=${LDSC_BIOCONDA_BUILD}"
    conda create -n "${LDSC_CONDA_ENV}" \
        -c bioconda -c conda-forge \
        "ldsc=${LDSC_VERSION}=${LDSC_BIOCONDA_BUILD}" \
        -y
    (
        set +eu
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${LDSC_CONDA_ENV}"
        which ldsc.py
        ldsc.py --help 2>&1 | head -3 || true
    )
fi

# --- Persist env marker for downstream scripts ------------------------------
cat > "${ENV_MARKER}" <<EOF
REGENIE_VERSION_USED=${REGENIE_VERSION_USED}
REGENIE_BIN=${REGENIE_BIN}
LDSC_CONDA_ENV=${LDSC_CONDA_ENV}
LDSC_VERSION=${LDSC_VERSION}
UKB_CHR22_PATH=${UKB_CHR22_PATH}
UKB_PHENO_PATH=${UKB_PHENO_PATH}
EOF

echo ""
echo "[ukb install] DONE."
echo "  REGENIE: ${REGENIE_VERSION_USED} at ${REGENIE_BIN}"
echo "  LDSC:    ${LDSC_VERSION} in conda env ${LDSC_CONDA_ENV}"
echo "  UKB chr22: ${UKB_CHR22_PATH}"
echo "  UKB pheno: ${UKB_PHENO_PATH}"
echo "  Marker file: ${ENV_MARKER}"
echo ""
echo "Next: bash $(dirname "${BASH_SOURCE[0]}")/fetch_data.sh"
