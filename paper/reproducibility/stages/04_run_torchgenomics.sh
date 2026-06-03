#!/usr/bin/env bash
# Stage 04: run TorchGenomics against each staged fixture and produce the
# results/ directories that the figure renderers consume.
#
# The actual head-to-head computation is implemented inside each
# harness's compare.py — this stage just invokes it.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Tier 1 head-to-head compare scripts.
TIER1_TOOLS=(
    bolt_lmm gapit gemma gwaspoly ldsc plink2 regenie saige
    susieR twosamplemr metaxcan smr coloc hyprcoloc hapref
)
for tool in "${TIER1_TOOLS[@]}"; do
    harness="$REPO_ROOT/validation/external/$tool"
    if [[ -f "$harness/compare.py" ]]; then
        echo "[stage 04] ==> compare $tool"
        TORCHGENOMICS_DISABLE_NATIVE="${TORCHGENOMICS_DISABLE_NATIVE:-1}" \
            python3 "$harness/compare.py"
    fi
done

# Tier 3 specialty compare scripts.
for model in survival rr family threshold knockoff ocf; do
    harness="$REPO_ROOT/validation/specialty/$model"
    if [[ -f "$harness/compare.py" ]]; then
        echo "[stage 04] ==> compare specialty/$model"
        TORCHGENOMICS_DISABLE_NATIVE="${TORCHGENOMICS_DISABLE_NATIVE:-1}" \
            python3 "$harness/compare.py"
    fi
done

echo "[stage 04] complete."
