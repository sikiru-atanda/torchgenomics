#!/usr/bin/env bash
# Stage 03: run every reference tool against its staged fixture.
#
# Each harness's run.sh invokes the upstream binary / R package (the
# reference side of the head-to-head). Output is written under
# validation/external/<tool>/outputs/ or .../results/ per the harness's
# own convention.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

TIER1_TOOLS=(
    bolt_lmm gapit gemma gwaspoly ldsc plink2 regenie saige
    susieR twosamplemr metaxcan smr coloc hyprcoloc hapref
)

for tool in "${TIER1_TOOLS[@]}"; do
    harness="$REPO_ROOT/validation/external/$tool"
    if [[ -x "$harness/run.sh" ]]; then
        echo "[stage 03] ==> run $tool"
        bash "$harness/run.sh"
    elif [[ -f "$harness/run.R" ]]; then
        echo "[stage 03] ==> run $tool (run.R)"
        (cd "$harness" && Rscript run.R)
    else
        echo "[stage 03] WARN: $harness has no run.sh / run.R; skipping"
    fi
done

echo "[stage 03] complete."
