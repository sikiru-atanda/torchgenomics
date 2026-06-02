#!/usr/bin/env bash
# hapref install -- haplo.stats reference for TG HaplotypeGWAS.
#
# TOOL SUBSTITUTION RATIONALE:
#   PLINK 1.9 --hap-* family was REMOVED upstream (cog-genomics docs note
#   deprecation since 1.07). PLINK 2 has NO haplotype-association commands.
#   There is therefore no PLINK reference for the EM-based block / window
#   haplotype scans implemented in torchgenomics/models/haplotype_gwas.py.
#
#   The canonical CRAN substitute is R/haplo.stats (Schaid et al. 2002,
#   AJHG). It implements:
#     - haplo.em()  : Excoffier-Slatkin / Hill EM (same algorithm as TG
#                     _enumerate_haplotypes_unphased), and
#     - haplo.glm() : GLM regression of phenotype on inferred haplotype
#                     dosages with user-selectable baseline haplotype
#                     (haplo.glm.control(haplo.base=...)).
#   This is the right reference for TG HaplotypeGWAS HTR F-test path.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned: haplo.stats 1.9.8.7 (CRAN), jsonlite 2.0.0 (CRAN), R 4.5.1.
#   - Idempotent: re-running checks the marker + smoke-loads the package.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
export HAPLO_STATS_VERSION="1.9.8.7"
export JSONLITE_VERSION="2.0.0"

# --- Idempotence check (delegate to R) ---------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    if Rscript --vanilla "${HERE}/_install_steps.R" check 2>/dev/null; then
        echo "[hapref install] already installed (marker + load OK); skipping"
        sed "s/^/  /" "${INSTALL_MARKER}"
        exit 0
    else
        echo "[hapref install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight --------------------------------------------------------------
# haplo.stats is pure R (no C++ deps beyond what base R supplies); ~10 MB.
preflight_check_with_data_size "hapref-install" 1 1

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[hapref install] ABORT: Rscript not on PATH."
    exit 1
fi
export R_VERSION="$(Rscript --vanilla -e "cat(R.version.string)" 2>/dev/null)"
echo "[hapref install] R: ${R_VERSION}"

# --- Install / verify pinned versions ----------------------------------------
echo "[hapref install] resolving pinned versions (haplo.stats=${HAPLO_STATS_VERSION}, jsonlite=${JSONLITE_VERSION})"
Rscript --vanilla "${HERE}/_install_steps.R" install

# --- Smoke test --------------------------------------------------------------
echo "[hapref install] smoke test: load haplo.stats and run a small haplo.em"
Rscript --vanilla "${HERE}/_install_steps.R" smoke

# --- Persist marker ----------------------------------------------------------
Rscript --vanilla "${HERE}/_install_steps.R" marker > "${INSTALL_MARKER}"

echo "[hapref install] done. Marker -> ${INSTALL_MARKER}"
sed "s/^/  /" "${INSTALL_MARKER}"

