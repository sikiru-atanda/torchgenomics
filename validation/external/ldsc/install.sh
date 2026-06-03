#!/usr/bin/env bash
# LDSC install — bioconda pinned 1.0.1 build (Bulik-Sullivan et al. 2015).
#
# Idempotent: if the conda env `ldsc_env` already exists with `ldsc.py` on
# the path at the pinned version, this script exits 0 without re-creating it.
#
# Per Pillar B contract:
#   - Memory pre-flight before any conda work (validation/external/_lib/preflight.sh).
#   - Pinned tool version (LDSC 1.0.1 from bioconda).
#   - Conda is the dependency manager: LDSC 1.0.1 is upstream Python 2.7 with
#     pinned numpy/scipy/pandas — bioconda has the exact build wired.
#
# Why bioconda not pip:
#   - Upstream LDSC (https://github.com/bulik/ldsc) is Python 2.7 with
#     numpy<1.17 / scipy<0.19 / pandas<0.21 pinned. Modern pip + venv cannot
#     install these against a current Python.
#   - The maintained Python-3 fork (belowlab/ldsc) is more recent but its
#     numerical output drifts from the canonical reference; the bioconda
#     1.0.1 build is the closest match to the published method and is what
#     S-LDSC papers continue to cite.
#
# Why we use a separate `ldsc_env` rather than the project venv:
#   - LDSC 1.0.1 requires Python 2.7. TorchGenomics requires Python ≥ 3.10.
#   - Process-isolation (subprocess from `compare.py`) is the cleanest way to
#     bridge the two.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned version ----------------------------------------------------------
LDSC_VERSION="1.0.1"
LDSC_BIOCONDA_BUILD="pyhdfd78af_2"
CONDA_ENV_NAME="ldsc_env"

ENV_MARKER="${HERE}/.env_marker"

# --- Idempotence: skip if env already exists with pinned LDSC ----------------
# We run the check in a subshell with `set +e` so a partial-state failure
# (env missing, broken activation) propagates as "needs install" rather than
# aborting the script via the outer `set -euo pipefail`.
_check_env() {
    (
        set +eu
        # shellcheck disable=SC1091
        source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null
        conda activate "${CONDA_ENV_NAME}" 2>/dev/null || exit 1
        command -v ldsc.py >/dev/null 2>&1 || exit 1
        installed_version=$(conda list ldsc --json 2>/dev/null \
            | python -c "import json,sys; data=json.load(sys.stdin); print(data[0]['version'] if data else '')" 2>/dev/null \
            || echo "")
        [[ "${installed_version}" == "${LDSC_VERSION}" ]]
    )
}

if _check_env; then
    echo "[ldsc install] already installed: LDSC ${LDSC_VERSION} in conda env '${CONDA_ENV_NAME}'"
    echo "[ldsc install] activate with: conda activate ${CONDA_ENV_NAME}"
    # Refresh the marker so downstream scripts know the env is wired.
    cat > "${ENV_MARKER}" <<EOF
CONDA_ENV_NAME=${CONDA_ENV_NAME}
LDSC_VERSION=${LDSC_VERSION}
EOF
    exit 0
fi

# --- Pre-flight (~200 MB conda env, peak RAM trivial) ------------------------
preflight_check_with_data_size "ldsc-install" 1 1

# --- Conda detection --------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ldsc install] ABORT: conda is not on PATH."
    echo "[ldsc install] LDSC 1.0.1 requires Python 2.7 with pinned numpy/scipy."
    echo "[ldsc install] Bioconda is the only practical install path on a modern host."
    echo "[ldsc install] Install miniconda from https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Source conda's shell hooks so `conda activate` works inside this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Create env (no --dry-run; real install) ---------------------------------
echo "[ldsc install] creating conda env '${CONDA_ENV_NAME}' with bioconda::ldsc=${LDSC_VERSION}=${LDSC_BIOCONDA_BUILD}"
echo "[ldsc install] (this pins both the LDSC version and the bioconda build hash for reproducibility)"

# We pin ldsc=1.0.1=pyhdfd78af_2 (the latest 1.0.1 build on bioconda as of
# 2026-05-04). conda will solve the rest of the env per LDSC's metadata pins.
conda create -n "${CONDA_ENV_NAME}" \
    -c bioconda -c conda-forge \
    "ldsc=${LDSC_VERSION}=${LDSC_BIOCONDA_BUILD}" \
    -y

# --- Smoke test --------------------------------------------------------------
# We run the smoke test in a subshell so that conda's activate/deactivate
# scripts (which can trip `set -u` on certain compiler-toolchain hooks) do
# not pollute the parent shell.
(
    set +eu
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    echo "[ldsc install] LDSC binaries:"
    which ldsc.py
    which munge_sumstats.py
    echo "[ldsc install] LDSC --help (first 3 lines):"
    ldsc.py --help 2>&1 | head -3 || true
)

# --- Persist env marker for downstream scripts -------------------------------
cat > "${ENV_MARKER}" <<EOF
CONDA_ENV_NAME=${CONDA_ENV_NAME}
LDSC_VERSION=${LDSC_VERSION}
EOF

echo "[ldsc install] done. Env: ${CONDA_ENV_NAME}; LDSC ${LDSC_VERSION} ready."
