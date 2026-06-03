#!/usr/bin/env bash
# SoyMD reference harness install — `mediation` R package (Imai/Keele/Tingley
# 2010) plus `jsonlite` for the JSON output bridge.
#
# Why this harness exists:
#   The user's spec §12 + §5.4 names SoyMD (Soybean Multi-omics Database) as
#   the only external reference for the entire `torchgenomics.multiomics` module
#   (mediation + multi-kernel h² + eQTL prefilter). SoyMD itself is a *data
#   source*, not a comparison TOOL — the platform doesn't ship a library that
#   re-runs the same `mediate_lmm` analysis. So the actual comparison
#   reference is the canonical causal-mediation R package: `mediation`
#   (Tingley et al. 2014 JSS, "mediation: R Package for Causal Mediation
#   Analysis"). It returns ACME / ADE / total effect with their CIs from a
#   pair of fitted models (mediator ~ X, outcome ~ M + X), which is exactly
#   the comparator we need for `torchgenomics.multiomics.mediate_lmm`.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned-or-recorded tool versions (we record the version actually
#     resolved into ${INSTALL_MARKER} after a successful install).
#   - Idempotent: skip if marker is present and reports the same R + pkg
#     versions we last installed.

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
            for (p in c("mediation", "jsonlite")) {
                if (!requireNamespace(p, quietly = TRUE)) {
                    cat("missing:", p, "\n")
                    ok <- FALSE
                }
            }
            if (!ok) quit(status = 1)
            cat("mediation:", as.character(packageVersion("mediation")), "\n")
            cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[soymd install] already installed (marker present, packages load); skipping"
        echo "[soymd install] versions:"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[soymd install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight (~80 MB R deps + working set) --------------------------------
preflight_check_with_data_size "soymd-install" 2 1

# --- Verify Rscript on PATH --------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[soymd install] ABORT: Rscript not on PATH."
    echo "[soymd install] Install R 4.x: https://cran.r-project.org/"
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[soymd install] R: ${R_VERSION}"

# --- Install via R -----------------------------------------------------------
# `dependencies = TRUE` is required: `mediation` Imports/Suggests Matrix,
# MASS, mvtnorm, sandwich, lpSolve, etc. We allow CRAN to resolve the dep
# chain; subsequent installs skip via the marker.
echo "[soymd install] installing mediation + jsonlite from CRAN (this can take ~5 min on first run)"
Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("mediation", quietly = TRUE)) {
    install.packages("mediation", dependencies = TRUE, quiet = TRUE)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", quiet = TRUE)
}

cat("mediation version:", as.character(packageVersion("mediation")), "\n")
cat("jsonlite version:", as.character(packageVersion("jsonlite")), "\n")
RSCRIPT

# --- Smoke test --------------------------------------------------------------
echo "[soymd install] smoke test: load both packages and report versions"
Rscript --vanilla -e '
    suppressPackageStartupMessages({
        library(mediation)
        library(jsonlite)
    })
    cat("mediation:", as.character(packageVersion("mediation")), "\n")
    cat("jsonlite:", as.character(packageVersion("jsonlite")), "\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        cat(sprintf("MEDIATION_VERSION=%s\n", packageVersion("mediation")))
        cat(sprintf("JSONLITE_VERSION=%s\n", packageVersion("jsonlite")))
    '
} > "${INSTALL_MARKER}"

echo "[soymd install] done. Marker -> ${INSTALL_MARKER}"
cat "${INSTALL_MARKER}" | sed 's/^/  /'
