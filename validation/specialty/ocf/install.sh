#!/usr/bin/env bash
# OCF (Orthogonal Cross-Fit LMM) install -- verifies R reference toolchain.
#
# Reference policy: per Plan B C6 brief, preferred reference is R DoubleML
# at pinned version. Fallback (used here): hand-coded DML pipeline matching
# Chernozhukov et al. (2018, Econometrica) eq. (3.1) Algorithm 1.
#
# Why fallback: DoubleML/mlr3 not installed and pulling ~30 dependencies is
# heavyweight; the DML algorithm itself is ~40 lines of R and is encoded
# in run_reference.R directly. This is explicitly documented in README.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

# --- Idempotence -----------------------------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    echo "[ocf install] marker present; verifying packages still load"
    if Rscript --vanilla -e '
        ok <- TRUE
        for (p in c("data.table", "jsonlite", "MASS")) {
            if (!requireNamespace(p, quietly = TRUE)) {
                cat("missing:", p, "\n"); ok <- FALSE
            }
        }
        if (!ok) quit(status = 1)
    '  2>/dev/null; then
        echo "[ocf install] already installed (marker + load OK); skipping"
        sed "s/^/  /" "${INSTALL_MARKER}"
        exit 0
    else
        echo "[ocf install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight: tiny (no downloads) ---------------------------------------
# Simulation generates ~ N=400 x p=20 x 100 reps ~ 64 MB peak.
preflight_check "ocf-install" 1 2

# --- Verify Rscript --------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[ocf install] ABORT: Rscript not on PATH."
    exit 1
fi
R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[ocf install] R: ${R_VERSION}"

# --- Verify required R packages --------------------------------------------
Rscript --vanilla -e '
  needed <- c("data.table", "jsonlite", "MASS")
  missing <- character(0)
  for (p in needed) {
      if (!requireNamespace(p, quietly = TRUE)) missing <- c(missing, p)
  }
  if (length(missing) > 0) {
      cat("[ocf install] installing missing packages:", paste(missing, collapse=", "), "\n")
      install.packages(missing, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  for (p in needed) {
      cat(sprintf("  %s: %s\n", p, packageVersion(p)))
  }
'

# --- Verify Python torchgenomics import ----------------------------------------
echo "[ocf install] verifying Python torchgenomics import"
python3 -c "import sys; sys.path.insert(0, '${HERE}/../../..'); from torchgenomics.models.ocf_lmm import OCFLMM; print('OCFLMM import: OK')"

# --- Persist marker --------------------------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    Rscript --vanilla -e '
        for (p in c("data.table", "jsonlite", "MASS")) {
            cat(sprintf("%s_VERSION=%s\n", toupper(p), packageVersion(p)))
        }
    '
    echo "INSTALL_TIMESTAMP=\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
} > "${INSTALL_MARKER}"

echo "[ocf install] done. Marker -> ${INSTALL_MARKER}"
sed "s/^/  /" "${INSTALL_MARKER}"
