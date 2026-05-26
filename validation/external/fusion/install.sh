#!/usr/bin/env bash
# FUSION (TWAS, measured-expression mode) install --- sources
# ../_lib/preflight.sh.
#
# The measured-expression mode of FUSION (gusevlab/fusion_twas)
# implements the same per-gene OLS Wald that this harness's
# run_reference.R reproduces. We therefore only require base R for
# the comparison; install.sh probes Rscript availability and writes
# the install marker.
#
# An optional FUSION repo clone is performed (pinned to commit
# 6e15bc54a36ff63d70b67d2cf6ae9bbd25b9ee35) only if `git` is on PATH;
# it is the source-of-truth reference for the FUSION.assoc_test.R
# imputed-GReX mode, not exercised by run.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

FUSION_COMMIT="6e15bc54a36ff63d70b67d2cf6ae9bbd25b9ee35"
FUSION_REPO="https://github.com/gusevlab/fusion_twas.git"
REPO_DIR="${HERE}/fusion_twas"
INSTALL_MARKER="${HERE}/.install_marker"

if [[ -f "${INSTALL_MARKER}" ]]; then
    source "${INSTALL_MARKER}"
    if [[ "${FUSION_INSTALLED_COMMIT:-}" == "${FUSION_COMMIT}" ]] || \
       [[ "${FUSION_INSTALLED_COMMIT:-}" == "measured-expression-only" ]]; then
        echo "[fusion install] already installed"
        exit 0
    fi
fi

preflight_check_with_data_size "fusion-install" 1 1

if ! command -v Rscript >/dev/null 2>&1; then
    echo "[fusion install] ABORT: Rscript not on PATH."
    echo "  Install R >= 4.0: https://www.r-project.org/."
    exit 1
fi
Rscript -e 'cat("R OK\n")' >/dev/null 2>&1 || {
    echo "[fusion install] ABORT: Rscript present but fails to execute."
    exit 1
}

REPO_INSTALLED="measured-expression-only"
if command -v git >/dev/null 2>&1; then
    if [[ ! -d "${REPO_DIR}" ]]; then
        git clone "${FUSION_REPO}" "${REPO_DIR}" 2>/dev/null || {
            echo "[fusion install] note: FUSION repo clone failed; "\
                 "continuing with measured-expression-only mode."
        }
    fi
    if [[ -d "${REPO_DIR}" ]]; then
        ( cd "${REPO_DIR}" && git checkout "${FUSION_COMMIT}" 2>/dev/null ) \
            || true
        REPO_INSTALLED="${FUSION_COMMIT}"
    fi
fi

cat > "${INSTALL_MARKER}" <<EOF
FUSION_INSTALLED_COMMIT=${REPO_INSTALLED}
EOF

echo "[fusion install] OK (mode: ${REPO_INSTALLED})"
