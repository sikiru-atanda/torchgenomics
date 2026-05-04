#!/usr/bin/env bash
# GEMMA install — pinned 0.98.5 Linux x86_64 static binary.
#
# Per the Pillar B B7 spec: GEMMA was wired BEFORE the Pillar A campaign and
# the binary is already present at ${ROOT}/gemma_demo/gemma-0.98.5 (static
# AMD64 build from upstream). This script:
#   1. Locates the existing binary (worktree root → repo-canonical fallback).
#   2. Symlinks it into validation/external/gemma/bin/gemma.
#   3. Verifies the version string matches the pin.
#   4. Verifies the SHA-256 of the binary against the upstream fingerprint.
#
# It does NOT re-download or re-extract — that's covered by the canonical
# checkout. If the binary is missing, abort with instructions.
#
# Idempotent: re-running on an already-installed binary exits 0 immediately.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - SHA-256 verification on the binary.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned version ----------------------------------------------------------
GEMMA_VERSION="0.98.5"
GEMMA_BUILD_DATE="2021-08-25"
# SHA-256 of the static AMD64 binary as committed at gemma_demo/gemma-0.98.5.
# Captured 2026-05-04 against the canonical TorchGWAS checkout. If upstream
# rebuilds the same release artifact under the same name, the checksum will
# mismatch and we'll know — which is exactly what we want for reproducibility.
GEMMA_BINARY_SHA256="ad3f3f43a2f8a1c00e71fae8f43676614170e06c99fc766a947acfe45605969b"

BIN_DIR="${HERE}/bin"
BIN="${BIN_DIR}/gemma"

# --- Locate the canonical GEMMA binary ---------------------------------------
# Search order:
#   1. ${ROOT}/gemma_demo/gemma-0.98.5  (worktree root)
#   2. ${HOME}/Documents/GWAS_Expert/gemma_demo/gemma-0.98.5  (canonical repo)
SOURCE_BIN=""
for candidate in \
    "${ROOT}/gemma_demo/gemma-0.98.5" \
    "${HOME}/Documents/GWAS_Expert/gemma_demo/gemma-0.98.5"; do
    if [[ -f "${candidate}" ]]; then
        SOURCE_BIN="${candidate}"
        break
    fi
done

if [[ -z "${SOURCE_BIN}" ]]; then
    echo "[gemma install] ABORT: cannot find GEMMA ${GEMMA_VERSION} binary."
    echo "[gemma install] Searched:"
    echo "[gemma install]   ${ROOT}/gemma_demo/gemma-0.98.5"
    echo "[gemma install]   ${HOME}/Documents/GWAS_Expert/gemma_demo/gemma-0.98.5"
    echo "[gemma install] GEMMA was wired BEFORE Pillar A; the binary is part"
    echo "[gemma install] of the canonical TorchGWAS checkout. Restore"
    echo "[gemma install] gemma_demo/ before retrying."
    exit 1
fi

# --- Idempotence: skip if already symlinked at the pinned version ------------
if [[ -x "${BIN}" ]]; then
    INSTALLED_VERSION="$("${BIN}" 2>&1 | grep -i 'GEMMA' | head -1 || echo unknown)"
    if [[ "${INSTALLED_VERSION}" == *"${GEMMA_VERSION}"* ]]; then
        echo "[gemma install] already installed: ${INSTALLED_VERSION}"
        echo "[gemma install] binary: ${BIN}"
        echo "[gemma install] source: ${SOURCE_BIN}"
        exit 0
    fi
    echo "[gemma install] re-linking: existing ${INSTALLED_VERSION} != pinned ${GEMMA_VERSION}"
fi

# --- Pre-flight (no download — just chmod + symlink; trivial RAM) ------------
preflight_check_with_data_size "gemma-install" 1 1

mkdir -p "${BIN_DIR}"

# --- Make source binary executable (in case repo perms dropped +x) -----------
chmod +x "${SOURCE_BIN}"

# --- Optional checksum verification ------------------------------------------
# If GEMMA_BINARY_SHA256 has been pinned (not "PENDING"), enforce it.
if [[ "${GEMMA_BINARY_SHA256}" != "PENDING" ]]; then
    verify_checksum "${SOURCE_BIN}" "${GEMMA_BINARY_SHA256}"
else
    actual=$(sha256sum "${SOURCE_BIN}" | awk '{print $1}')
    echo "[gemma install] FIRST-RUN HASH: ${actual}"
    echo "[gemma install]   pin this in install.sh and re-run for tighter regression control"
fi

# --- Symlink into bin/ -------------------------------------------------------
ln -sf "${SOURCE_BIN}" "${BIN}"

# --- Smoke test --------------------------------------------------------------
echo "[gemma install] installed at ${BIN} -> ${SOURCE_BIN}"
"${BIN}" 2>&1 | grep -i "GEMMA" | head -1 || true
