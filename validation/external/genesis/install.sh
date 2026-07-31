#!/usr/bin/env bash
# GENESIS install — Bioconductor R packages SNPRelate + GWASTools + GENESIS.
#
# These three packages form the definitive external reference chain for
# torchgenomics.linalg.kinship_admixed's KING-robust / PC-AiR / PC-Relate
# estimators (Phase 57 Unit A, Task 6):
#   SNPRelate  -- snpgdsBED2GDS, snpgdsIBDKING (Zheng et al. 2012)
#   GWASTools  -- GdsGenotypeReader / GenotypeData (Gogarten et al. 2012),
#                 the object GENESIS's pcair/pcrelate expect
#   GENESIS    -- pcair, pcrelate (Conomos et al. 2015/2016; Gogarten et al.
#                 2019, Bioinformatics 35:5346)
#
# Idempotent: probes whether all three packages are already loadable before
# attempting any install.
#
# Pillar B contract:
#   - Memory pre-flight before any work (validation/external/_lib/preflight.sh).
#   - Per-harness install log recorded to bin/genesis_versions.txt.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

VERSION_FILE="${HERE}/bin/genesis_versions.txt"
mkdir -p "${HERE}/bin"

# --- Pre-flight ---------------------------------------------------------
# Bioconductor source builds (SNPRelate has a small C core; GWASTools and
# GENESIS pull in Biobase/S4Vectors/IRanges and similar Bioc infra) can peak
# a few hundred MB to ~1-2 GB during compilation across all transitive deps.
preflight_check_with_data_size "genesis-install" 3 2

# --- Verify Rscript on PATH -----------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[genesis install] ABORT: Rscript not found on PATH."
    echo "[genesis install]   Install R (>= 4.0) before running this harness."
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[genesis install] R: ${R_VERSION}"

# --- Idempotence: probe if all three packages already load ---------------
PROBE_SCRIPT='
ok <- TRUE
for (p in c("SNPRelate", "GWASTools", "GENESIS")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat("missing:", p, "\n")
    ok <- FALSE
  }
}
if (!ok) quit(save = "no", status = 1)
for (p in c("SNPRelate", "GWASTools", "GENESIS")) {
  cat(sprintf("%s: %s\n", p, as.character(packageVersion(p))))
}
quit(save = "no", status = 0)
'

if Rscript --vanilla -e "${PROBE_SCRIPT}" > "${VERSION_FILE}.tmp" 2>/dev/null; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
    echo "[genesis install] already installed:"
    sed 's/^/  /' "${VERSION_FILE}"
    exit 0
fi
rm -f "${VERSION_FILE}.tmp"

# --- Install via BiocManager ----------------------------------------------
echo "[genesis install] one or more of SNPRelate/GWASTools/GENESIS missing;"
echo "[genesis install] installing via BiocManager..."

INSTALL_SCRIPT='
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org",
                    quiet = TRUE)
}
BiocManager::install(c("SNPRelate", "GWASTools", "GENESIS"),
                     update = FALSE, ask = FALSE)

ok <- TRUE
for (p in c("SNPRelate", "GWASTools", "GENESIS")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat("FAILED to install:", p, "\n")
    ok <- FALSE
  } else {
    cat(sprintf("%s: %s\n", p, as.character(packageVersion(p))))
  }
}
if (!ok) quit(save = "no", status = 1)
'

if Rscript --vanilla -e "${INSTALL_SCRIPT}" | tee "${VERSION_FILE}.tmp"; then
    mv "${VERSION_FILE}.tmp" "${VERSION_FILE}"
else
    rm -f "${VERSION_FILE}.tmp"
    echo "[genesis install] ABORT: BiocManager::install() failed for one or"
    echo "[genesis install]   more of SNPRelate / GWASTools / GENESIS. See"
    echo "[genesis install]   the install log above for the specific failure."
    exit 1
fi

echo "[genesis install] done. Versions:"
sed 's/^/  /' "${VERSION_FILE}"
