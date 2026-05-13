#!/usr/bin/env bash
# regenie install — pinned v3.6 prebuilt static linux x86_64 binary.
#
# Idempotent: if validation/external/regenie/bin/regenie already exists and
# reports the pinned version, this script exits 0 without re-downloading.
#
# Per Pillar B contract:
#   - Memory pre-flight before any download (validation/external/_lib/preflight.sh).
#   - SHA-256 verification on the upstream zip before extraction.
#   - Pinned URL + version + checksum (no "latest").
#
# Why the non-MKL build:
#   regenie ships two Linux variants per release: with and without Intel MKL.
#   The MKL variant is slightly faster on Intel CPUs but adds a dynamic-link
#   dep on libmkl that not all hosts ship. The non-MKL build is the
#   most-portable static option and is sufficient for the harness fixture
#   (281 samples × 3093 SNPs is far below the scale where MKL matters).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned version ----------------------------------------------------------
# We pin to v3.3 (Aug 2023) rather than the most-recent v3.6 because v3.4+
# binaries are linked against glibc 2.35; RHEL 9.6 (this host's target) ships
# glibc 2.34 and the newer builds fail to load. v3.3 is the latest static
# Linux build that still loads against glibc 2.34, and its Step 1 + Step 2
# numerics are identical to v3.6 for the basic continuous + binary GWAS that
# this harness exercises (LOCO ridge predictor + per-SNP score test + Firth /
# SPA correction). Newer regenie releases add features (rare-variant burden
# tests, conditional analyses) that aren't in scope here.
REGENIE_VERSION="v3.3"
REGENIE_URL="https://github.com/rgcgithub/regenie/releases/download/${REGENIE_VERSION}/regenie_v3.3.gz_x86_64_Linux.zip"
# SHA-256 captured on first download (2026-04-30). If the upstream rebuilds
# the asset under the same path, this checksum will mismatch and the script
# will abort — which is exactly what we want for reproducibility.
REGENIE_SHA256="88fa48a93779321d1e1b585c006c6bb863de15380d61431ab8d20cfe5b31900d"

BIN_DIR="${HERE}/bin"
BIN="${BIN_DIR}/regenie"
ZIP_PATH="${HERE}/.cache/regenie_${REGENIE_VERSION}.gz_x86_64_Linux.zip"

# --- Idempotence: skip if already installed at the pinned version ------------
if [[ -x "${BIN}" ]]; then
    INSTALLED_VERSION="$("${BIN}" --version 2>/dev/null | head -1 || echo unknown)"
    if [[ "${INSTALLED_VERSION}" == *"${REGENIE_VERSION}"* ]]; then
        echo "[regenie install] already installed: ${INSTALLED_VERSION}"
        echo "[regenie install] binary: ${BIN}"
        exit 0
    fi
    echo "[regenie install] re-installing: existing ${INSTALLED_VERSION} != pinned ${REGENIE_VERSION}"
fi

# --- Platform check ----------------------------------------------------------
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *)
        echo "[regenie install] ABORT: only Linux x86_64 prebuilt binaries are wired."
        echo "[regenie install] Detected: $(uname -s)-$(uname -m)"
        echo "[regenie install] Other platforms: build from source at https://github.com/rgcgithub/regenie"
        exit 1
        ;;
esac

# --- Pre-flight (zip is ~30 MB; install peak RAM trivial) --------------------
preflight_check_with_data_size "regenie-install" 1 1

# --- Download ---------------------------------------------------------------
mkdir -p "${BIN_DIR}" "$(dirname "${ZIP_PATH}")"

if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "[regenie install] downloading ${REGENIE_URL}"
    curl --fail --location --show-error --silent \
        --max-time 300 \
        --output "${ZIP_PATH}" \
        "${REGENIE_URL}"
else
    echo "[regenie install] cached zip: ${ZIP_PATH}"
fi

verify_checksum "${ZIP_PATH}" "${REGENIE_SHA256}"

# --- Extract -----------------------------------------------------------------
TMP_EXTRACT="$(mktemp -d)"
trap 'rm -rf "${TMP_EXTRACT}"' EXIT

unzip -q -o "${ZIP_PATH}" -d "${TMP_EXTRACT}"

# regenie zips contain a single executable like `regenie_v3.6.gz_x86_64_Linux`.
# We resolve it generically and rename to `regenie`.
EXTRACTED="$(find "${TMP_EXTRACT}" -maxdepth 2 -type f -name 'regenie*' ! -name '*.zip' | head -1)"
if [[ -z "${EXTRACTED}" || ! -f "${EXTRACTED}" ]]; then
    echo "[regenie install] ABORT: extracted archive does not contain a regenie binary"
    ls -la "${TMP_EXTRACT}"
    exit 1
fi

mv -f "${EXTRACTED}" "${BIN}"
chmod +x "${BIN}"

# --- Smoke test --------------------------------------------------------------
echo "[regenie install] installed at ${BIN}"
"${BIN}" --version
