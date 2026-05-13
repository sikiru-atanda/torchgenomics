#!/usr/bin/env bash
# Install susieR via CRAN. Idempotent: detects existing install and skips.
# Per NA1 design spec section 5.2; pre-flight per feedback_preflight memory.
#
# Pillar B contract:
#   - Memory + disk pre-flight before any install work (validation/external/_lib/preflight.sh).
#   - Idempotent: detect installed susieR and exit 0.
#   - Explicit fall-back guidance to a containerized R if the host R toolchain
#     can't compile susieR's deps (R-NA1-2 risk mitigation).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pre-flight (susieR + Matrix + dependencies; ~2 GB disk + 4 GB RAM) ------
# Per NA1 design spec section 5.2. Uses the shared preflight_check entry point
# exposed by validation/external/_lib/preflight.sh (positional args:
# <tool_name> <required_disk_gb> <required_ram_gb>).
preflight_check "susieR-install" 2 4

# --- R toolchain ------------------------------------------------------------
if ! command -v R >/dev/null 2>&1; then
    echo "[susieR install] FATAL: R is not installed."
    echo "[susieR install] Install R first (e.g., conda install -c conda-forge r-base) and retry."
    exit 2
fi

# --- Idempotence: skip if already installed ---------------------------------
if R -e 'suppressMessages(library(susieR))' >/dev/null 2>&1; then
    echo "[susieR install] susieR already installed; skipping."
    exit 0
fi

echo "[susieR install] installing susieR via CRAN..."
R -e 'install.packages("susieR", repos="https://cloud.r-project.org")'

# --- Verify -----------------------------------------------------------------
if ! R -e 'suppressMessages(library(susieR))' >/dev/null 2>&1; then
    echo "[susieR install] FATAL: susieR install failed."
    echo "[susieR install] Check the R toolchain (Matrix / BH / RcppArmadillo build) or fall back"
    echo "[susieR install] to containerized R per the R-NA1-2 mitigation:"
    echo "    docker run --rm rocker/r-ver:latest R -e 'install.packages(\"susieR\")'"
    exit 3
fi

echo "[susieR install] susieR install succeeded."
