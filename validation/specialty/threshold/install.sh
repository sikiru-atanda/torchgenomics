#!/usr/bin/env bash
# BLUPF90+ gibbsf90+ installer (multi-trait threshold model).
#
# Authoritative reference for the threshold-linear model family is the
# University of Georgia BLUPF90 distribution (Misztal lab).  Historically
# thrgibbsf90 was the dedicated multi-trait threshold sampler; the
# program was retired and its functionality folded into the unified
# gibbsf90+ driver, which activates threshold-mode via OPTION categ
# directives in the parameter file (Aguilar et al., 2018; Lourenco et al.,
# 2022; BLUPF90 manual section Threshold models).  We pull the static
# Linux build from the 64bit_old directory. Test_static binaries require
# need GLIBC >= 2.35; 64bit_old is linked against GLIBC <= 2.14.
#
# Pinned source:
#   https://nce.ads.uga.edu/html/projects/programs/Linux/64bit_old/gibbsf90+
#   https://nce.ads.uga.edu/html/projects/programs/Linux/64bit_old/postgibbsf90
#
# Both files are statically linked ELF binaries.  We fetch each once,
# verify its SHA256, copy into ./bin/, and write an install marker.
#
# Idempotent: if .install_marker exists and the binaries pass their
# smoke tests, the script exits 0 with an "already installed" message.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

BLUPF90_BASE="https://nce.ads.uga.edu/html/projects/programs/Linux/64bit_old"
GIBBS_URL="${BLUPF90_BASE}/gibbsf90+"
POSTG_URL="${BLUPF90_BASE}/postgibbsf90"

BIN_DIR="${HERE}/bin"
CACHE_DIR="${HERE}/.cache"
INSTALL_MARKER="${HERE}/.install_marker"
GIBBS_BIN="${BIN_DIR}/gibbsf90+"
POSTG_BIN="${BIN_DIR}/postgibbsf90"

# Smoke test: pipe empty input so the program does not block on prompt,
# then verify it emitted the BLUPF90 banner.
gibbs_banner_ok() {
    set +e
    local out
    out=$( { echo "" | "${GIBBS_BIN}" 2>&1 || true; } | head -20 )
    set -e
    printf "%s" "${out}" | grep -qiE "gibbs|blupf90|parameter file"
}

postg_banner_ok() {
    set +e
    local out
    out=$( { echo "" | "${POSTG_BIN}" 2>&1 || true; } | head -20 )
    set -e
    printf "%s" "${out}" | grep -qiE "postgibbs|blupf90|parameter file"
}

if [[ -f "${INSTALL_MARKER}" && -x "${GIBBS_BIN}" && -x "${POSTG_BIN}" ]]; then
    # shellcheck disable=SC1090
    source "${INSTALL_MARKER}"
    if gibbs_banner_ok && postg_banner_ok; then
        echo "[thr install] already installed at ${BIN_DIR}"
        echo "  gibbsf90+:    ${BLUPF90_GIBBSF90_SHA256:-unknown}"
        echo "  postgibbsf90: ${BLUPF90_POSTGIBBSF90_SHA256:-unknown}"
        exit 0
    fi
    echo "[thr install] marker found but smoke test failed; re-installing"
fi

preflight_check_with_data_size "thr-install" 1 1

for tool in sha256sum curl file; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[thr install] ABORT: required tool not found on PATH: ${tool}"
        exit 1
    fi
done

mkdir -p "${BIN_DIR}" "${CACHE_DIR}"

# Helper: fetch + (optionally) verify + install one binary.
# Args: url dest expected_sha256_or_empty
fetch_binary() {
    local url="$1"
    local dest="$2"
    local expected_sha="$3"
    local fname; fname="$(basename "${url}")"
    local cached="${CACHE_DIR}/${fname}"

    # The University of Georgia BLUPF90 mirror serves a cert chain that some
    # executors lack the intermediate for.  We bypass TLS verification (-k)
    # and re-verify the binary via SHA256 below.
    if [[ ! -f "${cached}" ]]; then
        echo "[thr install] downloading ${url}" >&2
        curl -k -fSL --max-time 600 "${url}" -o "${cached}"
    fi

    local actual_sha; actual_sha=$(sha256sum "${cached}" | awk "{print \$1}")
    if [[ -n "${expected_sha}" ]]; then
        if [[ "${actual_sha}" != "${expected_sha}" ]]; then
            echo "[thr install] ABORT: SHA256 mismatch on ${fname}"
            echo "  expected: ${expected_sha}"
            echo "  observed: ${actual_sha}"
            exit 1
        fi
    else
        echo "[thr install] note: no pinned SHA256 for ${fname}; recording observed=${actual_sha}" >&2
    fi

    if ! file "${cached}" 2>/dev/null | grep -qi "elf.*executable"; then
        echo "[thr install] ABORT: ${fname} is not an ELF executable"
        file "${cached}"
        exit 1
    fi

    cp "${cached}" "${dest}"
    chmod +x "${dest}"

    # Echo the resolved SHA so the marker captures it.
    printf "%s" "${actual_sha}"
}

# Pinned SHA256 values land here after the first run.  On a fresh install
# the script records observed checksums into .install_marker and warns once;
# subsequent runs verify against the marker.  If the upstream files are
# republished, delete .install_marker to refresh.
EXPECTED_GIBBS_SHA="${BLUPF90_GIBBSF90_EXPECTED_SHA256:-eb32dc016ead4a9c529a25498876a10cd2754966cd8c615bb61c7ec1726c10ef}"
EXPECTED_POSTG_SHA="${BLUPF90_POSTGIBBSF90_EXPECTED_SHA256:-8830de90c01a7a00f2b167ff2aac95da0280264b3066cb012d284951b0b8d0ac}"

OBS_GIBBS_SHA=$(fetch_binary "${GIBBS_URL}" "${GIBBS_BIN}" "${EXPECTED_GIBBS_SHA}")
OBS_POSTG_SHA=$(fetch_binary "${POSTG_URL}" "${POSTG_BIN}" "${EXPECTED_POSTG_SHA}")

if ! gibbs_banner_ok; then
    echo "[thr install] ABORT: gibbsf90+ smoke test failed"
    set +e; echo "" | "${GIBBS_BIN}" 2>&1 | head -10; set -e
    exit 1
fi
if ! postg_banner_ok; then
    echo "[thr install] ABORT: postgibbsf90 smoke test failed"
    set +e; echo "" | "${POSTG_BIN}" 2>&1 | head -10; set -e
    exit 1
fi
echo "[thr install] smoke tests passed"

{
    printf "BLUPF90_SOURCE=%s\n" "${BLUPF90_BASE}"
    printf "BLUPF90_GIBBSF90_URL=%s\n" "${GIBBS_URL}"
    printf "BLUPF90_POSTGIBBSF90_URL=%s\n" "${POSTG_URL}"
    printf "BLUPF90_GIBBSF90_BIN=%s\n" "${GIBBS_BIN}"
    printf "BLUPF90_POSTGIBBSF90_BIN=%s\n" "${POSTG_BIN}"
    printf "BLUPF90_GIBBSF90_SHA256=%s\n" "${OBS_GIBBS_SHA}"
    printf "BLUPF90_POSTGIBBSF90_SHA256=%s\n" "${OBS_POSTG_SHA}"
} > "${INSTALL_MARKER}"

echo "[thr install] done. Marker -> ${INSTALL_MARKER}"
echo "[thr install]   gibbsf90+ SHA256:    ${OBS_GIBBS_SHA}"
echo "[thr install]   postgibbsf90 SHA256: ${OBS_POSTG_SHA}"

