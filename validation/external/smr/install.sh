#!/usr/bin/env bash
# SMR v1.3.1 installer --- sources ../_lib/preflight.sh (pre-flight gate).
#
# Yang lab SMR (Zhu et al. 2016, Nature Genetics 48:481, doi:10.1038/ng.3538).
# v1.3.1 Linux x86_64 prebuilt binary (released 2024-03-08, build by GCC 8.3).
#
# Idempotent: if .install_marker exists and the binary smoke-tests, skip
# re-download / re-unpack. Otherwise verify SHA256 + unpack into ./bin/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

SMR_VERSION="1.3.1"
SMR_URL="https://yanglab.westlake.edu.cn/software/smr/download/smr-1.3.1-linux-x86_64.zip"
SMR_ZIP_SHA256="4d779197a0b3399db36c9cdf7b4b4190ea40fa33a47253f5419ad27c3bce251e"
SMR_CACHED_ZIP="/tmp/smr_v1.3.1.zip"
SMR_BIN_NAME="smr"

BIN_DIR="${HERE}/bin"
INSTALL_MARKER="${HERE}/.install_marker"
SMR_BINARY="${BIN_DIR}/${SMR_BIN_NAME}"

# Smoke-test helper. SMR with no args writes its banner and exits 1, and
# `head -3` closes the pipe early --- with `set -o pipefail` this raises 141
# (SIGPIPE) which would abort install.sh under `set -e`. Wrap the test so the
# pipe error is swallowed but the grep verdict survives.
smr_banner_ok() {
    set +e
    local banner
    banner=$( { "${SMR_BINARY}" 2>&1 || true; } | head -3 )
    local rc=$?
    set -e
    if (( rc != 0 )); then return 1; fi
    printf '%s' "${banner}" | grep -qi 'summary-data-based mendelian'
}

if [[ -f "${INSTALL_MARKER}" && -x "${SMR_BINARY}" ]]; then
    # shellcheck disable=SC1090
    source "${INSTALL_MARKER}"
    if [[ "${SMR_INSTALLED_VERSION:-}" == "${SMR_VERSION}" ]] && smr_banner_ok; then
        echo "[smr install] already installed at ${SMR_BINARY}"
        exit 0
    fi
fi

preflight_check_with_data_size "smr-install" 1 1

for tool in sha256sum unzip find file; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[smr install] ABORT: required tool not found on PATH: ${tool}"
        exit 1
    fi
done

mkdir -p "${BIN_DIR}"

# Prefer cached zip if SHA256 matches; otherwise fetch.
ZIP_PATH=""
if [[ -f "${SMR_CACHED_ZIP}" ]]; then
    CACHED_HASH="$(sha256sum "${SMR_CACHED_ZIP}" | awk '{print $1}')"
    if [[ "${CACHED_HASH}" == "${SMR_ZIP_SHA256}" ]]; then
        echo "[smr install] using cached zip at ${SMR_CACHED_ZIP} (SHA256 verified)"
        ZIP_PATH="${SMR_CACHED_ZIP}"
    else
        echo "[smr install] cached zip hash mismatch; refetching"
        echo "[smr install]   expected: ${SMR_ZIP_SHA256}"
        echo "[smr install]   observed: ${CACHED_HASH}"
    fi
fi

if [[ -z "${ZIP_PATH}" ]]; then
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        echo "[smr install] ABORT: need curl or wget to download SMR zip."
        exit 1
    fi
    ZIP_PATH="${HERE}/.cache/smr-${SMR_VERSION}-linux-x86_64.zip"
    mkdir -p "$(dirname "${ZIP_PATH}")"
    echo "[smr install] fetching ${SMR_URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${SMR_URL}" -o "${ZIP_PATH}"
    else
        wget -q "${SMR_URL}" -O "${ZIP_PATH}"
    fi
    OBSERVED_HASH="$(sha256sum "${ZIP_PATH}" | awk '{print $1}')"
    if [[ "${OBSERVED_HASH}" != "${SMR_ZIP_SHA256}" ]]; then
        echo "[smr install] ABORT: SHA256 mismatch after download."
        echo "  expected: ${SMR_ZIP_SHA256}"
        echo "  observed: ${OBSERVED_HASH}"
        exit 1
    fi
fi

# Unpack into a scratch directory and lift out the ELF (the zip ships
# smr-1.3.1-linux-x86_64/smr alongside macOS resource forks under __MACOSX/).
echo "[smr install] unpacking into ${BIN_DIR}"
TMP_UNPACK="${HERE}/.unpack_tmp"
rm -rf "${TMP_UNPACK}"
mkdir -p "${TMP_UNPACK}"
unzip -q "${ZIP_PATH}" -d "${TMP_UNPACK}"

FOUND_BIN=""
while IFS= read -r f; do
    if [[ -f "${f}" ]] && file "${f}" 2>/dev/null | grep -qi "elf.*executable"; then
        FOUND_BIN="${f}"
        break
    fi
done < <(find "${TMP_UNPACK}" -type f -name "smr*" ! -path "*__MACOSX*" | sort)

if [[ -z "${FOUND_BIN}" ]]; then
    echo "[smr install] ABORT: no SMR ELF executable found inside zip."
    find "${TMP_UNPACK}" -maxdepth 3 -type f | head -20
    exit 1
fi

cp "${FOUND_BIN}" "${SMR_BINARY}"
chmod +x "${SMR_BINARY}"
rm -rf "${TMP_UNPACK}"

if ! smr_banner_ok; then
    echo "[smr install] ABORT: smoke test failed (binary did not emit the v1.3.1 banner)."
    set +e
    "${SMR_BINARY}" 2>&1 | head -5
    set -e
    exit 1
fi
echo "[smr install] smoke test passed"

# Marker
{
    printf 'SMR_VERSION="%s"\n' "${SMR_VERSION}"
    printf 'SMR_INSTALLED_VERSION="%s"\n' "${SMR_VERSION}"
    printf 'SMR_ZIP_SHA256="%s"\n' "${SMR_ZIP_SHA256}"
    printf 'SMR_BIN="%s"\n' "${SMR_BINARY}"
    printf 'SMR_URL="%s"\n' "${SMR_URL}"
} > "${INSTALL_MARKER}"

echo "[smr install] done. Marker -> ${INSTALL_MARKER}"
echo "[smr install]   binary: ${SMR_BINARY}"
