#!/usr/bin/env bash
# fetch_data.sh -- regenerate the random-regression fixture deterministically.
#
# This harness does not download external data: the comparison is on a
# deterministic simulated longitudinal panel (n=300, T=5, b=3) authored
# in simulate_rr_fixture.py. The simulator is bit-deterministic given
# the pinned seed (=42), so re-running fetch_data.sh always produces the
# same data/ files (verifiable via data/fixture.sha256).
#
# Why simulate rather than reuse a real longitudinal dataset:
#   - SoyNAM (Diers et al. 2018) is a balanced longitudinal phenotype
#     panel but only ships per-trait BLUPs (one row per line), not the
#     raw per-time-point observations required for a random-regression
#     reference comparison. The longitudinal raw data is held by the
#     individual breeding programs and is not redistributable.
#   - Public animal-breeding longitudinal datasets (e.g. lambsT, dairy
#     test-day) ship under restrictive licenses or via dataset-broker
#     services (Iowa BLUPF90 group, EAAP).
#   - The mathematical reduction beta(t) = phi(t) dot beta_j is purely
#     algebraic; numerical agreement requires a well-formed (Y, T, G)
#     fixture, which the simulator provides deterministically. Same
#     strategy used by validation/external/smr (simulated SMR fixture).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

preflight_check "rr-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

echo "[rr fetch] regenerating fixture in ${DATA_DIR}"
python3 "${HERE}/simulate_rr_fixture.py" --out-dir "${DATA_DIR}"

echo "[rr fetch] done."
echo "[rr fetch] fixture SHA256:"
cat "${DATA_DIR}/fixture.sha256"
