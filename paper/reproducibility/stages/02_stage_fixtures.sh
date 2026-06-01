#!/usr/bin/env bash
# Stage 02: stage all fixtures (reference-tool + multi-omics + specialty).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Tier 1 reference-tool fixtures.
TIER1_TOOLS=(
    bolt_lmm gapit gemma gwaspoly ldsc plink2 regenie saige
    susieR twosamplemr metaxcan smr coloc hyprcoloc hapref
)
for tool in "${TIER1_TOOLS[@]}"; do
    harness="$REPO_ROOT/validation/external/$tool"
    if [[ -x "$harness/fetch_data.sh" ]]; then
        echo "[stage 02] ==> fetch $tool"
        bash "$harness/fetch_data.sh"
    fi
done

# Tier 2 multi-omics fixtures.
for fixture in sim gtex_ukb plant; do
    dir="$REPO_ROOT/validation/multiomics/$fixture"
    if [[ ! -d "$dir" ]]; then
        echo "[stage 02] WARN: $dir not present; skipping"
        continue
    fi
    if [[ -x "$dir/fetch.sh" ]]; then
        echo "[stage 02] ==> stage multiomics/$fixture (via fetch.sh)"
        bash "$dir/fetch.sh"
    elif [[ -f "$dir/generate.py" ]]; then
        echo "[stage 02] ==> stage multiomics/$fixture (via generate.py)"
        (cd "$dir" && python3 generate.py)
    else
        echo "[stage 02] WARN: $dir has no fetch.sh or generate.py; skipping"
    fi
done

# Tier 3 specialty fixtures.
for model in survival rr family threshold knockoff ocf; do
    dir="$REPO_ROOT/validation/specialty/$model"
    if [[ -x "$dir/fetch_data.sh" ]]; then
        echo "[stage 02] ==> stage specialty/$model"
        bash "$dir/fetch_data.sh"
    fi
done

echo "[stage 02] complete."
