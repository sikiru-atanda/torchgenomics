#!/usr/bin/env bash
# Family-aware GWAS reference-tool install — within-family / direct-indirect.
#
# Plan B Tier 3 Agent C3 harness for torchgwas.models.within_family_lmm.
# WithinFamilyLMM (Phase 23; Young, Benonisdottir, Przeworski & Kong 2022,
# Nat Genet 54:263-273, "Mendelian imputation of parental genotypes
# improves estimates of direct genetic effects", doi:10.1038/s41588-022-01016-z).
#
# Reference tool selection:
#   1. PREFERRED: snipar (AlexTISYoung/snipar) — canonical Python
#      implementation of Young 2022's direct + non-transmitted-coefficient
#      (NTC) estimator. Installed via pip from upstream git.
#   2. FALLBACK: paper simulator + paper estimator. snipar's pyproject.toml
#      pins legacy numpy==1.21.1 + Cython==0.29.28, which are not buildable
#      under modern Python 3.13 + numpy 2.x. In that case we fall back to a
#      self-contained reference implementation of the OLS direct+indirect
#      estimator from Young 2022 section 2.1 (within-family + between-family
#      decomposition on sib-pairs).
#
# Pillar-B contract:
#   - Pre-flight memory + disk gate (validation/external/_lib/preflight.sh).
#   - Pinned reference-tool version captured at install time.
#   - Idempotent via .install_marker.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

source "${ROOT}/validation/external/_lib/preflight.sh"

MARKER="${HERE}/.install_marker"
REF_TOOL_FILE="${HERE}/.ref_tool"

SNIPAR_REPO="https://github.com/AlexTISYoung/snipar.git"
SNIPAR_PIN="master"

preflight_check_with_data_size "family-install" 2 2

if [[ -f "${MARKER}" ]]; then
    echo "[family install] already installed; marker: ${MARKER}"
    while IFS= read -r line; do echo "  ${line}"; done < "${MARKER}"
    exit 0
fi

# --- Attempt snipar install -------------------------------------------------
echo "[family install] attempting snipar install"
INSTALL_LOG="${HERE}/.cache_install.log"
INSTALL_OK=0

if pip install "snipar @ git+${SNIPAR_REPO}@${SNIPAR_PIN}" > "${INSTALL_LOG}" 2>&1; then
    INSTALL_OK=1
fi

if (( INSTALL_OK == 1 )); then
    SNIPAR_VERSION="snipar_installed"
    SVR="$(pip show snipar 2>/dev/null | awk -F': ' '/^Version/ {print $2}' || echo unknown)"
    echo "ref_tool=snipar" > "${MARKER}"
    echo "snipar_pip_version=${SVR}" >> "${MARKER}"
    echo "snipar_repo=${SNIPAR_REPO}" >> "${MARKER}"
    echo "snipar_pin=${SNIPAR_PIN}" >> "${MARKER}"
    echo "install_date=$(date -Iseconds)" >> "${MARKER}"
    printf 'snipar\n' > "${REF_TOOL_FILE}"
    echo "[family install] snipar installed"
    exit 0
fi

echo "[family install] snipar install FAILED"
echo "[family install] tail of install log:"
tail -20 "${INSTALL_LOG}" || true
echo "[family install] falling back to paper-simulator reference path."
echo "[family install] Reference: Young AI et al. 2022 Nat Genet 54:263-273"
echo "[family install] doi:10.1038/s41588-022-01016-z"

for mod in numpy pandas scipy torch; do
    if ! python3 -c "import ${mod}" 2>/dev/null; then
        echo "[family install] ABORT: ${mod} not importable"
        exit 1
    fi
done

echo "ref_tool=paper_simulator" > "${MARKER}"
echo "fallback_reason=snipar_pip_install_failed" >> "${MARKER}"
echo "paper_citation=Young et al. 2022 Nat Genet 54:263-273" >> "${MARKER}"
echo "paper_doi=10.1038/s41588-022-01016-z" >> "${MARKER}"
echo "install_date=$(date -Iseconds)" >> "${MARKER}"
echo "python_version=$(python3 --version 2>&1)" >> "${MARKER}"
NUMPY_VER="$(python3 -c 'import numpy; print(numpy.__version__)')"
echo "numpy_version=${NUMPY_VER}" >> "${MARKER}"
SCIPY_VER="$(python3 -c 'import scipy; print(scipy.__version__)')"
echo "scipy_version=${SCIPY_VER}" >> "${MARKER}"
TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "torch_version=${TORCH_VER}" >> "${MARKER}"

printf 'paper_simulator\n' > "${REF_TOOL_FILE}"
echo "[family install] fallback path ready; marker written at ${MARKER}"
