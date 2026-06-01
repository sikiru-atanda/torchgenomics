#!/usr/bin/env bash
# MetaXcan v0.8.1 install --- sources ../_lib/preflight.sh (pre-flight gate).
# Pinned commit 964f1fdb5bf9585585690e85bb0eca7b67663ddb.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

METAXCAN_TAG="v0.8.1"
METAXCAN_COMMIT="964f1fdb5bf9585585690e85bb0eca7b67663ddb"
METAXCAN_REPO="https://github.com/hakyimlab/MetaXcan.git"

REPO_DIR="${HERE}/MetaXcan"
INSTALL_MARKER="${HERE}/.install_marker"
SPREDIXCAN="${REPO_DIR}/software/SPrediXcan.py"

if [[ -f "${INSTALL_MARKER}" && -f "${SPREDIXCAN}" ]]; then
    source "${INSTALL_MARKER}"
    if [[ "${METAXCAN_INSTALLED_COMMIT:-}" == "${METAXCAN_COMMIT}" ]]; then
        echo "already installed"
        if PYTHONPATH="${REPO_DIR}/software" python3 "${SPREDIXCAN}" --help >/dev/null 2>&1; then
            exit 0
        fi
    fi
fi

preflight_check_with_data_size "metaxcan-install" 1 1

for tool in python3 git; do
    command -v "${tool}" >/dev/null 2>&1 || exit 1
done

python3 -c "import numpy, pandas, scipy" || exit 1

if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "[metaxcan install] cloning ${METAXCAN_REPO} at ${METAXCAN_TAG}"
    rm -rf "${REPO_DIR}"
    git clone --depth 1 --branch "${METAXCAN_TAG}" --quiet "${METAXCAN_REPO}" "${REPO_DIR}"
fi

ACTUAL_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${METAXCAN_COMMIT}" ]]; then
    git -C "${REPO_DIR}" fetch --depth 1 origin "${METAXCAN_COMMIT}" 2>/dev/null \
        || git -C "${REPO_DIR}" fetch --unshallow --quiet origin
    git -C "${REPO_DIR}" checkout --quiet "${METAXCAN_COMMIT}"
    ACTUAL_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
fi
echo "[metaxcan install] HEAD: ${ACTUAL_COMMIT}"

if [[ ! -f "${SPREDIXCAN}" ]]; then
    echo "[metaxcan install] ABORT: SPrediXcan.py not found"
    exit 1
fi

if ! PYTHONPATH="${REPO_DIR}/software" python3 "${SPREDIXCAN}" --help >/dev/null 2>&1; then
    exit 1
fi
echo "smoke test passed"

PYTHON_VERSION="$(python3 --version 2>&1)"
printf 'METAXCAN_TAG="%s"\n' "${METAXCAN_TAG}" > "${INSTALL_MARKER}"
printf 'METAXCAN_INSTALLED_COMMIT="%s"\n' "${ACTUAL_COMMIT}" >> "${INSTALL_MARKER}"
printf 'METAXCAN_REPO_DIR="%s"\n' "${REPO_DIR}" >> "${INSTALL_MARKER}"
printf 'METAXCAN_SPREDIXCAN="%s"\n' "${SPREDIXCAN}" >> "${INSTALL_MARKER}"
printf 'PYTHON_VERSION="%s"\n' "${PYTHON_VERSION}" >> "${INSTALL_MARKER}"

echo "done. Marker -> ${INSTALL_MARKER}"
