#!/usr/bin/env bash
# Generates the synthetic fixture for the metap harness.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

preflight_check_with_data_size "metap-fetch" 1 1
mkdir -p "${HERE}/data"
python3 "${HERE}/simulate_fixture.py" --output-dir "${HERE}/data" --seed 42
echo "[metap fetch] fixture ready at ${HERE}/data"
