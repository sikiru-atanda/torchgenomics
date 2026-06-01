#!/usr/bin/env bash
# Stage 01: install every reference tool harness.
#
# Each harness is idempotent (skips if its `.install_marker` matches the
# pinned version), so re-running this stage is cheap once the first install
# has completed.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Tier 1 reference-tool harnesses (5 new + 10 pre-existing on the paper branch).
TIER1_TOOLS=(
    bolt_lmm
    gapit
    gemma
    gwaspoly
    ldsc
    plink2
    regenie
    saige
    susieR
    twosamplemr
    metaxcan
    smr
    coloc
    hyprcoloc
    hapref
)

for tool in "${TIER1_TOOLS[@]}"; do
    harness="$REPO_ROOT/validation/external/$tool"
    if [[ ! -d "$harness" ]]; then
        echo "[stage 01] WARN: $harness not present; skipping"
        continue
    fi
    if [[ -x "$harness/install.sh" ]]; then
        echo "[stage 01] ==> install $tool"
        bash "$harness/install.sh"
    else
        echo "[stage 01] WARN: $harness/install.sh missing or not executable; skipping"
    fi
done

echo "[stage 01] complete."
