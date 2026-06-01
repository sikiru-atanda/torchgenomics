#!/usr/bin/env bash
# R `coloc` reference-tool install for the two-trait colocalization harness.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned-or-recorded tool versions (we record the version actually
#     resolved into ${INSTALL_MARKER} after a successful install).
#   - Idempotent: skip if marker is present and reports the same R + pkg
#     versions we last installed.
#
# Why CRAN (not GitHub / bioconda):
#   - `coloc` (Wallace lab) is on CRAN at stable releases. The harness pins
#     to >=5.2 (the version that ships the corrected H4 normalization per
#     Wallace 2020 erratum, matching the math TorchGWAS implements).
#   - `jsonlite` is the bridge to the Python comparator.
#
# Two-trait reference: `coloc::coloc.abf(p1=1e-4, p2=1e-4, p12=1e-5)` with
# the default Wakefield prior W = 0.15^2 (Giambartolomei 2014 sec 2.2).

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
            for (p in c("coloc", "jsonlite")) {
                if (!requireNamespace(p, quietly = TRUE)) {
                    cat("missing:", p, "\n")
                    ok <- FALSE
                }
            }
            if (!ok) quit(status = 1)
            cat("coloc:",    as.character(packageVersion("coloc")),    "\n")
            cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[coloc install] already installed (marker present, packages load); skipping"
        echo "[coloc install] versions:"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[coloc install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight (~300 MB of R deps + working set) ----------------------------
preflight_check_with_data_size "coloc-install" 2 1

# --- Verify Rscript on PATH --------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[coloc install] ABORT: Rscript not on PATH."
    echo "[coloc install] Install R 4.x: https://cran.r-project.org/"
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[coloc install] R: ${R_VERSION}"

# --- Install via R -----------------------------------------------------------
# `coloc` 5.x compiles a few Bayesian-VS deps (susieR, Matrix). Keep
# `upgrade = "never"` so existing dep versions remain stable run-to-run.
echo "[coloc install] installing coloc (CRAN) + jsonlite (this can take ~3-5 min)"
Rscript --vanilla - <<RSCRIPT_EOF
options(repos = c(CRAN = "https://cloud.r-project.org"))

# 1. coloc (CRAN; Wallace lab).
if (!requireNamespace("coloc", quietly = TRUE)) {
    install.packages("coloc", quiet = TRUE)
}
cv <- packageVersion("coloc")
if (cv < "5.2.0") {
    stop(sprintf(
        "coloc version %s installed; harness requires >=5.2 (Wallace 2020 erratum H4 correction).",
        as.character(cv)
    ))
}

# 2. jsonlite -- used by run.R for JSON output.
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", quiet = TRUE)
}

cat("coloc version:",    as.character(packageVersion("coloc")),    "\n")
cat("jsonlite version:", as.character(packageVersion("jsonlite")), "\n")
RSCRIPT_EOF

# --- Smoke test --------------------------------------------------------------
echo "[coloc install] smoke test: load coloc + jsonlite and run a trivial coloc.abf"
Rscript --vanilla -e '
    suppressPackageStartupMessages({
        library(coloc)
        library(jsonlite)
    })
    set.seed(1)
    bx <- rnorm(20, 0, 0.05)
    sx <- rep(0.02, 20)
    by <- 0.8 * bx + rnorm(20, 0, 0.005)
    sy <- rep(0.02, 20)
    res <- coloc.abf(
        dataset1 = list(beta = bx, varbeta = sx^2, snp = sprintf("rs%02d", 1:20),
                        type = "quant", sdY = 1.0),
        dataset2 = list(beta = by, varbeta = sy^2, snp = sprintf("rs%02d", 1:20),
                        type = "quant", sdY = 1.0)
    )
    cat("smoke PP.H4.abf:", res$summary["PP.H4.abf"], "\n")
    cat("coloc:",    as.character(packageVersion("coloc")),    "\n")
    cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        cat(sprintf("COLOC_VERSION=%s\n",    packageVersion("coloc")))
        cat(sprintf("JSONLITE_VERSION=%s\n", packageVersion("jsonlite")))
    '
} > "${INSTALL_MARKER}"

echo "[coloc install] done. Marker -> ${INSTALL_MARKER}"
cat "${INSTALL_MARKER}" | sed 's/^/  /'
