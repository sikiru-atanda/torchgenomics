#!/usr/bin/env bash
# GWASpoly install — verifies an R installation with GWASpoly reachable.
#
# GWASpoly is an R package shipped via GitHub (jendelman/GWASpoly). Per the
# Pillar B B7 spec: GWASpoly was wired BEFORE the Pillar A campaign, and the
# canonical pre-Pillar-A reference outputs live in benchmark/gwaspoly_results/.
# This script:
#
#   1. Probes whether `library(GWASpoly)` succeeds in R.
#   2. If yes — records the version into bin/gwaspoly_version.txt and exits 0.
#   3. If no — attempts a non-interactive install via remotes into a
#      per-harness library at bin/Rlib/. If that fails, prints a clear
#      notice and exits 0 — the harness will fall back to the committed
#      reference outputs (run_gwaspoly.sh handles the fallback).
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

GWASPOLY_VERSION_MIN="2.12"
RLIB="${HERE}/bin/Rlib"
VERSION_FILE="${HERE}/bin/gwaspoly_version.txt"

mkdir -p "${RLIB}"

preflight_check_with_data_size "gwaspoly-install" 2 2

if ! command -v Rscript >/dev/null 2>&1; then
    echo "[gwaspoly install] R is not installed (no Rscript on PATH)."
    echo "[gwaspoly install]   the harness will fall back to the committed"
    echo "[gwaspoly install]   pre-Pillar-A reference outputs in run_gwaspoly.sh."
    echo "no-R" > "${VERSION_FILE}"
    exit 0
fi

PROBE_SCRIPT="$(cat <<'RSCRIPT'
local_lib <- Sys.getenv("RLIB", unset = NA)
if (!is.na(local_lib) && nzchar(local_lib) && dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}
if (requireNamespace("GWASpoly", quietly = TRUE)) {
  v <- as.character(packageVersion("GWASpoly"))
  cat(sprintf("FOUND GWASpoly %s\n", v))
  quit(save = "no", status = 0)
}
cat("MISSING\n")
quit(save = "no", status = 1)
RSCRIPT
)"

if RLIB="${RLIB}" Rscript -e "${PROBE_SCRIPT}" >"${VERSION_FILE}.tmp" 2>/dev/null; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
    echo "[gwaspoly install] $(cat "${VERSION_FILE}")"
    echo "[gwaspoly install] R library: ${RLIB} (or system)"
    exit 0
fi

rm -f "${VERSION_FILE}.tmp"
echo "[gwaspoly install] GWASpoly not found; attempting non-interactive install..."

INSTALL_SCRIPT="$(cat <<'RSCRIPT'
local_lib <- Sys.getenv("RLIB")
.libPaths(c(local_lib, .libPaths()))

for (pkg in c("remotes", "scam", "rrBLUP", "parallel", "stringr", "ggplot2")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    tryCatch({
      install.packages(pkg, repos = "https://cloud.r-project.org",
                       lib = local_lib, quiet = TRUE)
    }, error = function(e) cat("[install]", pkg, "failed:", conditionMessage(e), "\n"))
  }
}

tryCatch({
  remotes::install_github("jendelman/GWASpoly", upgrade = "never",
                          quiet = TRUE, lib = local_lib)
}, error = function(e) {
  cat("[install] GWASpoly from GitHub failed:", conditionMessage(e), "\n")
})

if (requireNamespace("GWASpoly", quietly = TRUE)) {
  v <- as.character(packageVersion("GWASpoly"))
  cat(sprintf("FOUND GWASpoly %s\n", v))
  quit(save = "no", status = 0)
}
cat("MISSING\n")
quit(save = "no", status = 1)
RSCRIPT
)"

if RLIB="${RLIB}" Rscript -e "${INSTALL_SCRIPT}" >"${VERSION_FILE}.tmp" 2>&1; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
    echo "[gwaspoly install] $(cat "${VERSION_FILE}")"
    exit 0
fi

echo "[gwaspoly install] install failed (see attempt log above)."
echo "[gwaspoly install]   the harness will fall back to the committed"
echo "[gwaspoly install]   pre-Pillar-A reference outputs at"
echo "[gwaspoly install]   benchmark/gwaspoly_results/ — run_gwaspoly.sh handles this."
echo "missing" > "${VERSION_FILE}"
exit 0
