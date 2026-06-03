#!/usr/bin/env bash
# Wrapper around generate_and_compare.py for the LRO internal-consistency
# simulator. No external tool to install; no network needed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

# Small fixture; 1 GB disk + 2 GB RAM is plenty.
preflight_check "specialty-lro" 1 2

TORCHGENOMICS_DISABLE_NATIVE="${TORCHGENOMICS_DISABLE_NATIVE:-1}" \
    python3 "${HERE}/generate_and_compare.py"
