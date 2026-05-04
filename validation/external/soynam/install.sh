#!/usr/bin/env bash
# SoyNAM + rrBLUP install (R packages from CRAN).
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned-or-recorded tool versions (we record the version actually
#     resolved into ${INSTALL_MARKER} after a successful install).
#   - Idempotent: skip if marker is present and reports the same R + pkg
#     versions we last installed.
#
# Why CRAN (not GitHub):
#   - SoyNAM (Diers et al. 2018) is published on CRAN; the package contains
#     the complete genotype + phenotype tables for the soybean Nested
#     Association Mapping panel (~5000 RILs across 40+ families × ~5000 SNPs).
#   - rrBLUP (Endelman 2011) is the canonical kinship-aware single-trait
#     LMM implementation in agriculture; provides the reference for
#     TorchGWAS' SingleTraitLMM at this scale + ag-trait-noise regime.
#
# SoyNAM compile note: the package contains a small amount of compiled C++
# (Rcpp; for kinship + simulation utilities). On RHEL 9.6 with the
# system R 4.5 + gcc toolchain, install completes in ~3-5 min wall time.
# If the upstream C++ build fails on a different host, document the failure
# and fall back to a pre-extracted fixture committed under `data/` (see
# fetch_data.sh for how that path is wired).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

# --- Idempotence check -------------------------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    _check_loadable() {
        Rscript --vanilla -e '
            ok <- TRUE
            for (p in c("SoyNAM", "rrBLUP")) {
                if (!requireNamespace(p, quietly = TRUE)) {
                    cat("missing:", p, "\n")
                    ok <- FALSE
                }
            }
            if (!ok) quit(status = 1)
            cat("SoyNAM:", as.character(packageVersion("SoyNAM")), "\n")
            cat("rrBLUP:", as.character(packageVersion("rrBLUP")), "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[soynam install] already installed (marker present, packages load); skipping"
        echo "[soynam install] versions:"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[soynam install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight (~50 MB of R deps; SoyNAM ships ~10 MB of data) --------------
preflight_check_with_data_size "soynam-install" 2 1

# --- Verify Rscript on PATH --------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[soynam install] ABORT: Rscript not on PATH."
    echo "[soynam install] Install R 4.x: https://cran.r-project.org/"
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[soynam install] R: ${R_VERSION}"

# --- Install via R -----------------------------------------------------------
# `dependencies = TRUE` is required: SoyNAM's Imports / LinkingTo include
# Rcpp / RcppArmadillo / NAM. We allow CRAN to resolve the dep chain;
# subsequent installs skip via the marker.
echo "[soynam install] installing SoyNAM + rrBLUP from CRAN (this can take ~5 min on first run)"
Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("SoyNAM", quietly = TRUE)) {
    install.packages("SoyNAM", quiet = TRUE)
}
if (!requireNamespace("rrBLUP", quietly = TRUE)) {
    install.packages("rrBLUP", quiet = TRUE)
}

cat("SoyNAM version:", as.character(packageVersion("SoyNAM")), "\n")
cat("rrBLUP version:", as.character(packageVersion("rrBLUP")), "\n")
RSCRIPT

# --- Smoke test --------------------------------------------------------------
echo "[soynam install] smoke test: load both packages and report versions"
Rscript --vanilla -e '
    suppressPackageStartupMessages({
        library(SoyNAM)
        library(rrBLUP)
    })
    cat("SoyNAM:", as.character(packageVersion("SoyNAM")), "\n")
    cat("rrBLUP:", as.character(packageVersion("rrBLUP")), "\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        cat(sprintf("SOYNAM_VERSION=%s\n", packageVersion("SoyNAM")))
        cat(sprintf("RRBLUP_VERSION=%s\n", packageVersion("rrBLUP")))
    '
} > "${INSTALL_MARKER}"

echo "[soynam install] done. Marker -> ${INSTALL_MARKER}"
cat "${INSTALL_MARKER}" | sed 's/^/  /'
