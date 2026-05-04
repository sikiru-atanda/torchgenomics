#!/usr/bin/env bash
# GAPIT3 install — verifies an R installation with GAPIT3 reachable.
#
# GAPIT3 is an R package shipped via GitHub (jiabowang/GAPIT3). Per the Pillar B
# B7 spec: GAPIT was wired BEFORE the Pillar A campaign, and the canonical
# pre-Pillar-A reference outputs live in benchmark/gapit_results/. Installing
# GAPIT3 fresh has many transitive dependencies (multtest from Bioconductor,
# scatterplot3d, gplots, ape, ...). This script:
#
#   1. Probes whether `library(GAPIT3)` (or `library(GAPIT)`) succeeds in R.
#   2. If yes — records the version into bin/gapit_version.txt and exits 0.
#   3. If no — attempts a non-interactive install via BiocManager + remotes
#      into a per-harness library at bin/Rlib/. If that fails (missing
#      compiler / Bioconductor unreachable / etc.), prints a clear notice and
#      exits 0 — the harness will fall back to the committed reference
#      outputs (run_gapit.sh handles the fallback).
#
# Idempotent: re-running on an already-installed package short-circuits.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Per-harness R library so we don't pollute the user's environment.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned versions ---------------------------------------------------------
GAPIT_VERSION_MIN="3.4"        # spec / docs/validation.md
RLIB="${HERE}/bin/Rlib"
VERSION_FILE="${HERE}/bin/gapit_version.txt"

mkdir -p "${RLIB}"

# --- Pre-flight (R install peak ~ 1 GB if all deps build from source) -------
preflight_check_with_data_size "gapit-install" 2 2

# --- Probe: is Rscript on PATH? ----------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[gapit install] R is not installed (no Rscript on PATH)."
    echo "[gapit install]   the harness will fall back to the committed"
    echo "[gapit install]   pre-Pillar-A reference outputs in run_gapit.sh."
    echo "no-R" > "${VERSION_FILE}"
    exit 0
fi

# --- Probe: is GAPIT or GAPIT3 already loadable? -----------------------------
PROBE_SCRIPT="$(cat <<'RSCRIPT'
local_lib <- Sys.getenv("RLIB", unset = NA)
if (!is.na(local_lib) && nzchar(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

for (pkg in c("GAPIT3", "GAPIT")) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    v <- as.character(packageVersion(pkg))
    cat(sprintf("FOUND %s %s\n", pkg, v))
    quit(save = "no", status = 0)
  }
}
cat("MISSING\n")
quit(save = "no", status = 1)
RSCRIPT
)"

if RLIB="${RLIB}" Rscript -e "${PROBE_SCRIPT}" >"${VERSION_FILE}.tmp" 2>/dev/null; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
    echo "[gapit install] $(cat "${VERSION_FILE}")"
    echo "[gapit install] R library: ${RLIB} (or system)"
    exit 0
fi

# --- Attempt install ---------------------------------------------------------
rm -f "${VERSION_FILE}.tmp"
echo "[gapit install] GAPIT not found; attempting non-interactive install..."

INSTALL_SCRIPT="$(cat <<'RSCRIPT'
local_lib <- Sys.getenv("RLIB")
.libPaths(c(local_lib, .libPaths()))

# Install BiocManager if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org",
                    lib = local_lib, quiet = TRUE)
}

# multtest is a Bioconductor dep of GAPIT3
if (!requireNamespace("multtest", quietly = TRUE)) {
  tryCatch({
    BiocManager::install("multtest", update = FALSE, ask = FALSE,
                         lib = local_lib, quiet = TRUE)
  }, error = function(e) cat("[install] multtest failed:", conditionMessage(e), "\n"))
}

# CRAN deps
for (pkg in c("scatterplot3d", "gplots", "ape", "data.table", "remotes",
              "EMMREML", "lme4", "compiler")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    tryCatch({
      install.packages(pkg, repos = "https://cloud.r-project.org",
                       lib = local_lib, quiet = TRUE)
    }, error = function(e) cat("[install]", pkg, "failed:", conditionMessage(e), "\n"))
  }
}

# GAPIT3 from GitHub
tryCatch({
  remotes::install_github("jiabowang/GAPIT3", upgrade = "never",
                          quiet = TRUE, lib = local_lib)
}, error = function(e) {
  cat("[install] GAPIT3 from GitHub failed:", conditionMessage(e), "\n")
})

# Verify
for (pkg in c("GAPIT3", "GAPIT")) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    v <- as.character(packageVersion(pkg))
    cat(sprintf("FOUND %s %s\n", pkg, v))
    quit(save = "no", status = 0)
  }
}
cat("MISSING\n")
quit(save = "no", status = 1)
RSCRIPT
)"

if RLIB="${RLIB}" Rscript -e "${INSTALL_SCRIPT}" >"${VERSION_FILE}.tmp" 2>&1; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
    echo "[gapit install] $(cat "${VERSION_FILE}")"
    exit 0
fi

# --- Install failed → fall back to committed reference outputs ---------------
echo "[gapit install] install failed (see attempt log above)."
echo "[gapit install]   the harness will fall back to the committed"
echo "[gapit install]   pre-Pillar-A reference outputs at"
echo "[gapit install]   benchmark/gapit_results/ — run_gapit.sh handles this."
echo "missing" > "${VERSION_FILE}"
exit 0
