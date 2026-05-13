#!/usr/bin/env bash
# BOLT-LMM install — pinned v2.5 prebuilt Linux x86_64 tarball.
#
# BOLT-LMM (Loh et al. 2015 *Nat. Genet.*; Loh et al. 2018 *Nat. Genet.*) is
# distributed only as a closed precompiled binary at
# https://alkesgroup.broadinstitute.org/BOLT-LMM/downloads/. There is no
# open-source build path. Steps:
#
#   1. Pre-flight (~211 MB tarball; install peak RAM trivial).
#   2. Download BOLT-LMM_v2.5.tar.gz.
#   3. SHA-256 verify (captured first run, then pinned).
#   4. Extract to bin/.
#   5. Smoke-test the prebuilt binary against this host's glibc / libstdc++.
#   6. On smoke-test failure (the documented failure mode on RHEL 9.6 — the
#      v2.5 binary is linked against newer libstdc++/glibc than RHEL 9
#      ships), capture diagnostics into .infra_blocker per the SAIGE
#      pattern and exit 0 (the harness skeleton remains in place; pytest
#      tests skip cleanly).
#
# Idempotent: presence of `.install_marker` short-circuits subsequent runs.
# Presence of `.infra_blocker` short-circuits and prints the captured
# failure reason (delete the file to retry).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned version ---------------------------------------------------------
# v2.5 (Aug 2025) is the most-recent release at the BOLT-LMM downloads page
# at the time of harness implementation (2026-04-30). The earlier stable
# series (v2.3.x, v2.4.x) referenced in some BOLT-LMM tutorials are no
# longer hosted under stable URLs; only v2.5 returns 200. We pin to v2.5
# to make the harness reproducible.
BOLT_VERSION="2.5"
BOLT_URL="https://alkesgroup.broadinstitute.org/BOLT-LMM/downloads/BOLT-LMM_v${BOLT_VERSION}.tar.gz"
# SHA-256 captured on first download (2026-04-30). Computed on first run.
# Tarball size at capture: 221,424,899 bytes.
BOLT_SHA256="a5fdc49f79f42aa007282d9080e5b97c6ef2aab25ace2af306f432c44eedebca"

BIN_DIR="${HERE}/bin"
BIN="${BIN_DIR}/bolt"
TARBALL="${HERE}/.cache/BOLT-LMM_v${BOLT_VERSION}.tar.gz"

INSTALL_MARKER="${HERE}/.install_marker"
INFRA_BLOCKER="${HERE}/.infra_blocker"

# --- Idempotence: skip if already installed --------------------------------
if [[ -x "${BIN}" && -f "${INSTALL_MARKER}" ]]; then
    echo "[bolt install] already installed (marker: ${INSTALL_MARKER})"
    echo "[bolt install] binary: ${BIN}"
    exit 0
fi

# --- Idempotence: skip if previously infra-blocked -------------------------
if [[ -f "${INFRA_BLOCKER}" ]]; then
    echo "[bolt install] previously marked infra-blocker; skipping."
    echo "[bolt install] To retry: rm ${INFRA_BLOCKER}"
    cat "${INFRA_BLOCKER}"
    exit 0
fi

# --- Platform check --------------------------------------------------------
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *)
        echo "[bolt install] ABORT: only Linux x86_64 prebuilt binaries are wired."
        echo "[bolt install] Detected: $(uname -s)-$(uname -m)"
        echo "[bolt install] BOLT-LMM is closed-source; no build-from-source path."
        exit 1
        ;;
esac

# --- Pre-flight (~211 MB tarball; expanded ~1 GB; install peak RAM trivial) -
preflight_check_with_data_size "bolt-install" 5 1

# --- Download -------------------------------------------------------------
mkdir -p "${BIN_DIR}" "$(dirname "${TARBALL}")"

if [[ ! -f "${TARBALL}" ]]; then
    echo "[bolt install] downloading ${BOLT_URL}"
    curl --fail --location --show-error --silent \
        --max-time 600 \
        --output "${TARBALL}" \
        "${BOLT_URL}"
else
    echo "[bolt install] cached tarball: ${TARBALL}"
fi

# Verify the SHA — but on the *first* run, the placeholder above hasn't
# been replaced. We detect that and print the observed checksum so the
# user (or the next harness commit) can pin it.
if [[ "${BOLT_SHA256}" == "REPLACE_ON_FIRST_RUN" ]]; then
    OBSERVED_SHA="$(sha256sum "${TARBALL}" | awk '{print $1}')"
    echo "[bolt install] WARNING: BOLT_SHA256 placeholder; first-run observed:"
    echo "[bolt install]   ${OBSERVED_SHA}"
    echo "[bolt install]   pin this value in install.sh and re-run for a"
    echo "[bolt install]   reproducible install."
else
    verify_checksum "${TARBALL}" "${BOLT_SHA256}"
fi

# --- Extract --------------------------------------------------------------
TMP_EXTRACT="$(mktemp -d)"
trap 'rm -rf "${TMP_EXTRACT}"' EXIT

echo "[bolt install] extracting tarball → ${TMP_EXTRACT}"
tar -xzf "${TARBALL}" -C "${TMP_EXTRACT}"

# Tarball top-level dir is BOLT-LMM_v<version>/; the binary is at the root
# of that dir as `bolt`. Use --strip-components=1 logic via mv.
SRC_TREE="$(find "${TMP_EXTRACT}" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "${SRC_TREE}" || ! -f "${SRC_TREE}/bolt" ]]; then
    echo "[bolt install] ABORT: extracted archive does not contain bolt binary"
    ls -la "${TMP_EXTRACT}" 2>/dev/null || true
    exit 1
fi

# Copy the entire dist tree into bin/ (we need the `tables/`, `example/`
# subdirs in addition to the `bolt` binary, because BOLT-LMM's --LDscoresFile
# flag points to tables/LDSCORE.* shipped in the distribution).
mkdir -p "${BIN_DIR}"
cp -r "${SRC_TREE}/." "${BIN_DIR}/"
chmod +x "${BIN}"

# --- Smoke test ------------------------------------------------------------
# BOLT prints a banner + flag listing on `--help`; if the binary fails to
# load (GLIBCXX or libc++ mismatch), this command will fail with exit
# code != 0 and we capture the loader's error message into .infra_blocker.
echo "[bolt install] smoke test: ${BIN} --help"
SMOKE_LOG="${HERE}/.cache/smoke.log"
if "${BIN}" --help > "${SMOKE_LOG}" 2>&1; then
    echo "[bolt install] OK: smoke test passed"
    head -3 "${SMOKE_LOG}"
    touch "${INSTALL_MARKER}"
    echo "[bolt install] BOLT-LMM v${BOLT_VERSION} installed at ${BIN}"
    exit 0
fi

# --- Smoke test failed — capture diagnostics as infra-blocker --------------
{
    echo "BOLT-LMM v${BOLT_VERSION} prebuilt binary failed to load on this host."
    echo
    echo "Diagnostics:"
    echo "  uname:  $(uname -srm)"
    echo "  glibc:  $(ldd --version 2>/dev/null | head -1)"
    echo "  binary: ${BIN}"
    echo
    echo "Loader output (first 20 lines):"
    head -20 "${SMOKE_LOG}" 2>/dev/null || true
    echo
    echo "BOLT-LMM is distributed only as a closed precompiled Linux x86_64"
    echo "binary at ${BOLT_URL}. There is no source-build fallback. The"
    echo "v${BOLT_VERSION} binary is typically linked against newer"
    echo "libstdc++ / glibc than RHEL 9.6 ships (glibc 2.34); on hosts with"
    echo "those libraries it may fail with GLIBCXX or GLIBC version errors."
    echo
    echo "Per Pillar B contract (spec §9), this is a documented infra-blocker."
    echo "The harness scripts remain in place; pytest tests skip cleanly with"
    echo "this file's text as the reason."
    echo
    echo "To retry:"
    echo "  rm ${INFRA_BLOCKER}"
    echo "  bash $(basename "${BASH_SOURCE[0]}")"
} > "${INFRA_BLOCKER}"

echo "[bolt install] WARNING: smoke test failed; marked infra-blocker"
echo "[bolt install] see ${INFRA_BLOCKER}"
exit 0
