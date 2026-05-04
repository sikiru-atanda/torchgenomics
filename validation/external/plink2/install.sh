#!/usr/bin/env bash
# PLINK 2.0 install — pinned alpha-7 build (25 Apr 2026, build 20260425).
#
# Idempotent: if validation/external/plink2/bin/plink2 already exists and
# reports the pinned version, this script exits 0 without re-downloading.
#
# Per Pillar B contract:
#   - Memory pre-flight before any download (validation/external/_lib/preflight.sh).
#   - SHA-256 verification on the upstream zip before extraction.
#   - Pinned URL + version + checksum (no "latest").

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned version ----------------------------------------------------------
PLINK2_VERSION="v2.0.0-a.7.0LM"
PLINK2_BUILD="20260425"
PLINK2_URL="https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_x86_64_${PLINK2_BUILD}.zip"
# SHA-256 captured on first download (2026-05-04). If the upstream rebuilds
# the asset under the same path, this checksum will mismatch and the script
# will abort — which is exactly what we want for reproducibility.
PLINK2_SHA256="e70a283aefe004122fca3e632ae0b24023a24635f98a8e768ea8d542bbc659a9"

BIN_DIR="${HERE}/bin"
BIN="${BIN_DIR}/plink2"
ZIP_PATH="${HERE}/.cache/plink2_linux_x86_64_${PLINK2_BUILD}.zip"

# --- Idempotence: skip if already installed at the pinned version ------------
if [[ -x "${BIN}" ]]; then
    INSTALLED_VERSION="$("${BIN}" --version 2>/dev/null | head -1 || echo unknown)"
    if [[ "${INSTALLED_VERSION}" == *"${PLINK2_VERSION}"* ]]; then
        echo "[plink2 install] already installed: ${INSTALLED_VERSION}"
        echo "[plink2 install] binary: ${BIN}"
        exit 0
    fi
    echo "[plink2 install] re-installing: existing ${INSTALLED_VERSION} != pinned ${PLINK2_VERSION}"
fi

# --- Platform check ----------------------------------------------------------
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *)
        echo "[plink2 install] ABORT: only Linux x86_64 prebuilt binaries are wired."
        echo "[plink2 install] Detected: $(uname -s)-$(uname -m)"
        echo "[plink2 install] Other platforms: install manually from https://www.cog-genomics.org/plink/2.0/"
        exit 1
        ;;
esac

# --- Pre-flight (zip is ~8 MB; install peak RAM trivial) ---------------------
preflight_check_with_data_size "plink2-install" 1 1

# --- Download ---------------------------------------------------------------
mkdir -p "${BIN_DIR}" "$(dirname "${ZIP_PATH}")"

if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "[plink2 install] downloading ${PLINK2_URL}"
    curl --fail --location --show-error --silent \
        --max-time 120 \
        --output "${ZIP_PATH}" \
        "${PLINK2_URL}"
else
    echo "[plink2 install] cached zip: ${ZIP_PATH}"
fi

verify_checksum "${ZIP_PATH}" "${PLINK2_SHA256}"

# --- Extract -----------------------------------------------------------------
TMP_EXTRACT="$(mktemp -d)"
trap 'rm -rf "${TMP_EXTRACT}"' EXIT

unzip -q -o "${ZIP_PATH}" -d "${TMP_EXTRACT}"

if [[ ! -f "${TMP_EXTRACT}/plink2" ]]; then
    echo "[plink2 install] ABORT: extracted archive does not contain a 'plink2' binary"
    ls -la "${TMP_EXTRACT}"
    exit 1
fi

mv -f "${TMP_EXTRACT}/plink2" "${BIN}"
chmod +x "${BIN}"

# --- Smoke test --------------------------------------------------------------
echo "[plink2 install] installed at ${BIN}"
"${BIN}" --version
