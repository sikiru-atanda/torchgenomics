#!/usr/bin/env bash
# TwoSampleMR + MRPRESSO install (R packages from GitHub).
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned-or-recorded tool versions (we record the version actually
#     resolved into ${INSTALL_MARKER} after a successful install).
#   - Idempotent: skip if marker is present and reports the same R + pkg
#     versions we last installed.
#
# Why R + remotes::install_github (not bioconda or CRAN):
#   - TwoSampleMR is the canonical MR-IEU implementation; only on GitHub.
#   - MRPRESSO (Verbanck et al. 2018) is also GitHub-only.
#   - These compile a number of dependencies from source — first install
#     can take 10-15 minutes wall time. Subsequent installs are skipped.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

# --- Idempotence check -------------------------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    # Sanity-check that the recorded packages still load. If they do, skip;
    # otherwise re-install. This handles the case where the user updated R
    # or wiped ~/R/.
    _check_loadable() {
        Rscript --vanilla -e '
            ok <- TRUE
            for (p in c("TwoSampleMR", "MRPRESSO")) {
                if (!requireNamespace(p, quietly = TRUE)) {
                    cat("missing:", p, "\n")
                    ok <- FALSE
                }
            }
            if (!ok) quit(status = 1)
            cat("TwoSampleMR:", as.character(packageVersion("TwoSampleMR")), "\n")
            cat("MRPRESSO:", as.character(packageVersion("MRPRESSO")), "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[twosamplemr install] already installed (marker present, packages load); skipping"
        echo "[twosamplemr install] versions:"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[twosamplemr install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight (~500 MB of R deps + working set) ----------------------------
preflight_check_with_data_size "twosamplemr-install" 2 1

# --- Verify Rscript on PATH --------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[twosamplemr install] ABORT: Rscript not on PATH."
    echo "[twosamplemr install] Install R 4.x: https://cran.r-project.org/"
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[twosamplemr install] R: ${R_VERSION}"

# --- Install via R -----------------------------------------------------------
# We install into the user's default library so subsequent harness runs need
# no environment activation. `upgrade = "never"` keeps existing dependency
# versions stable (avoids pulling in bleeding-edge CRAN updates that change
# the numerics from one run to the next).
echo "[twosamplemr install] installing TwoSampleMR + MRPRESSO from GitHub (this can take ~10 min)"
Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))

# 1. remotes (CRAN) — needed to install from GitHub.
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", quiet = TRUE)
}

# 2. TwoSampleMR (MRC-IEU GitHub).
if (!requireNamespace("TwoSampleMR", quietly = TRUE)) {
    remotes::install_github("MRCIEU/TwoSampleMR", upgrade = "never", quiet = TRUE)
}

# 3. MRPRESSO (rondolab GitHub).
if (!requireNamespace("MRPRESSO", quietly = TRUE)) {
    remotes::install_github("rondolab/MR-PRESSO", upgrade = "never", quiet = TRUE)
}

# 4. jsonlite — used by run_twosamplemr.R for JSON output.
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", quiet = TRUE)
}

cat("TwoSampleMR version:", as.character(packageVersion("TwoSampleMR")), "\n")
cat("MRPRESSO version:", as.character(packageVersion("MRPRESSO")), "\n")
cat("jsonlite version:", as.character(packageVersion("jsonlite")), "\n")
RSCRIPT

# --- Smoke test --------------------------------------------------------------
echo "[twosamplemr install] smoke test: load both packages and report versions"
Rscript --vanilla -e '
    suppressPackageStartupMessages({
        library(TwoSampleMR)
        library(MRPRESSO)
        library(jsonlite)
    })
    cat("TwoSampleMR:", as.character(packageVersion("TwoSampleMR")), "\n")
    cat("MRPRESSO:", as.character(packageVersion("MRPRESSO")), "\n")
    cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        cat(sprintf("TWOSAMPLEMR_VERSION=%s\n", packageVersion("TwoSampleMR")))
        cat(sprintf("MRPRESSO_VERSION=%s\n", packageVersion("MRPRESSO")))
    '
} > "${INSTALL_MARKER}"

echo "[twosamplemr install] done. Marker → ${INSTALL_MARKER}"
cat "${INSTALL_MARKER}" | sed 's/^/  /'
