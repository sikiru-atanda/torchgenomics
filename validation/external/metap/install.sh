#!/usr/bin/env bash
# p-value combination harness install --- sources ../_lib/preflight.sh.
#
# Probes Rscript availability. R packages required by the reference:
#   metap (CRAN)               — Fisher's, Stouffer's, harmonic mean p, Tippett's.
#   EmpiricalBrownsMethod (BioC) — Empirical Brown.
#
# Both are optional: run_reference.R degrades gracefully and skips the
# methods whose R package is unavailable. The harness's compare.py
# accepts a partial reference output and only compares present rows.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

if [[ -f "${INSTALL_MARKER}" ]]; then
    echo "[metap install] already installed"
    exit 0
fi

preflight_check_with_data_size "metap-install" 1 1

if ! command -v Rscript >/dev/null 2>&1; then
    echo "[metap install] ABORT: Rscript not on PATH."
    echo "  Install R >= 4.0: https://www.r-project.org/."
    exit 1
fi
Rscript -e 'cat("R OK\n")' >/dev/null 2>&1 || {
    echo "[metap install] ABORT: Rscript present but fails to execute."
    exit 1
}

# Probe metap and EmpiricalBrownsMethod; tag the install marker with
# what's actually usable.
HAVE_METAP="false"
HAVE_EBM="false"
Rscript -e 'cat(as.character(requireNamespace("metap", quietly=TRUE)))' \
    2>/dev/null | grep -qx TRUE && HAVE_METAP="true"
Rscript -e 'cat(as.character(requireNamespace("EmpiricalBrownsMethod", quietly=TRUE)))' \
    2>/dev/null | grep -qx TRUE && HAVE_EBM="true"

if [[ "${HAVE_METAP}" == "false" && "${HAVE_EBM}" == "false" ]]; then
    echo "[metap install] warning: neither 'metap' nor 'EmpiricalBrownsMethod' "
    echo "  is installed in R; the reference run will skip all comparisons."
    echo "  Install with:"
    echo "    R -e 'install.packages(\"metap\", repos=\"https://cloud.r-project.org\")'"
    echo "    R -e 'BiocManager::install(\"EmpiricalBrownsMethod\")'"
fi

cat > "${INSTALL_MARKER}" <<EOF
METAP_HAVE_METAP=${HAVE_METAP}
METAP_HAVE_EBM=${HAVE_EBM}
EOF
echo "[metap install] OK (metap=${HAVE_METAP}, EBM=${HAVE_EBM})"
