#!/usr/bin/env bash
# Knockoff FDR-control harness - install the R `knockoff` package + glmnet.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned-or-recorded tool versions (we record the resolved versions
#     into ${INSTALL_MARKER} after a successful install).
#   - Idempotent: skip if marker present and packages load.
#
# Reference (paper): Sesia, Sabatti & Candes 2020 JASA - KnockoffGWAS;
#                    Candes, Fan, Janson & Lv 2018 JRSSB - Model-X knockoffs.
# Package: R `knockoff` (CRAN; https://cran.r-project.org/package=knockoff).
# Importance: lasso coefficient difference (glmnet) - the package default
#             matches the JRSSB paper.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

# --- Idempotence check -------------------------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    _check_loadable() {
        Rscript --vanilla -e '
            ok <- TRUE
            for (p in c("knockoff", "glmnet", "jsonlite")) {
                if (!requireNamespace(p, quietly = TRUE)) { cat("missing:", p, "\n"); ok <- FALSE }
            }
            if (!ok) quit(status = 1)
            cat("knockoff:", as.character(packageVersion("knockoff")), "\n")
            cat("glmnet:",   as.character(packageVersion("glmnet")), "\n")
            cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[knockoff install] already installed (marker present, packages load); skipping"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[knockoff install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight --------------------------------------------------------------
# knockoff + glmnet + Rdsdp ~150 MB; R ~200 MB. Budget: 1 GB disk / 1 GB RAM.
preflight_check_with_data_size "knockoff-install" 1 1

if ! command -v Rscript >/dev/null 2>&1; then
    echo "[knockoff install] ABORT: Rscript not on PATH."
    echo "[knockoff install] Install R 4.x: https://cran.r-project.org/"
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[knockoff install] R: ${R_VERSION}"

# --- Install via R from CRAN -------------------------------------------------
echo "[knockoff install] installing knockoff (CRAN) + glmnet + jsonlite"
Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))

for (p in c("knockoff", "glmnet", "jsonlite")) {
    if (!requireNamespace(p, quietly = TRUE)) {
        install.packages(p, quiet = TRUE)
    }
}

cat("knockoff version:", as.character(packageVersion("knockoff")), "\n")
cat("glmnet version:",   as.character(packageVersion("glmnet")),   "\n")
cat("jsonlite version:", as.character(packageVersion("jsonlite")), "\n")
RSCRIPT

# --- Smoke test --------------------------------------------------------------
echo "[knockoff install] smoke test: create.fixed on a toy matrix"
Rscript --vanilla -e '
    suppressPackageStartupMessages({ library(knockoff); library(glmnet); library(jsonlite) })
    set.seed(1L)
    n <- 80L; p <- 20L
    X <- matrix(rnorm(n * p), n, p)
    Xk <- create.fixed(X)$Xk
    stopifnot(dim(Xk) == c(n, p))
    cat("smoke OK: create.fixed produced", n, "x", p, "knockoff matrix\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        cat(sprintf("KNOCKOFF_VERSION=%s\n", packageVersion("knockoff")))
        cat(sprintf("GLMNET_VERSION=%s\n",   packageVersion("glmnet")))
        cat(sprintf("JSONLITE_VERSION=%s\n", packageVersion("jsonlite")))
    '
} > "${INSTALL_MARKER}"

echo "[knockoff install] done. Marker -> ${INSTALL_MARKER}"
cat "${INSTALL_MARKER}" | sed 's/^/  /'
