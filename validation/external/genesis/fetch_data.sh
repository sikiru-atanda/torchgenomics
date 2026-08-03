#!/usr/bin/env bash
# GENESIS harness data staging — no download needed.
#
# The fixture is entirely synthetic (tests/fixtures/admixed/make_admixed.py:
# a 2-population Balding-Nichols admixed cohort with injected
# parent-offspring pairs), so "fetching data" here means generating it
# deterministically and exporting it to the two formats GENESIS and
# torchgenomics each need: PLINK BED/BIM/FAM (for SNPRelate) and G.npy (for
# torchgenomics) — see export_fixture.py for exactly how the two are kept
# byte-identical.
#
# Idempotent: skips regeneration if out/G.npy and out/fixture.bed already
# exist. Delete validation/external/genesis/out/ to force a fresh export.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

OUT_DIR="${HERE}/out"

# --- Idempotence -----------------------------------------------------------
if [[ -f "${OUT_DIR}/G.npy" && -f "${OUT_DIR}/fixture.bed" ]]; then
    echo "[genesis fetch] fixture already exported to ${OUT_DIR}; skipping"
    echo "[genesis fetch]   (delete ${OUT_DIR} to force regeneration)"
    exit 0
fi

# --- Pre-flight --------------------------------------------------------
# n=120, m=2000 fixture: BED ~60 KB, G.npy ~2 MB — trivially small; a fixed
# floor headroom check is still asserted per the hard rule.
preflight_check_with_data_size "genesis-fetch" 1 1

# --- Verify python3 on PATH --------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[genesis fetch] ABORT: python3 not found on PATH."
    exit 1
fi

echo "[genesis fetch] generating synthetic admixed fixture + PLINK export..."
python3 "${HERE}/export_fixture.py"

echo "[genesis fetch] done. Staged under ${OUT_DIR}:"
ls -la "${OUT_DIR}"
