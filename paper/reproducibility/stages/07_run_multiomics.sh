#!/usr/bin/env bash
# Stage 07: multi-omics integration (F4 panels A / B / C).
#
# Tier 2 fixtures (sim, gtex_ukb, plant) only stage the data — the actual
# head-to-head against TG twas_sumstat / smr_test / coloc_pairwise /
# mediate_lmm is exercised inside each multi-omics fixture's eventual
# compare.py (Tier 4 D2.4 authors these renderers; they consume the
# fixtures committed in Tier 2).
#
# For now, this stage is a passthrough: the data is already staged by
# stage 02, and the figure renderer in 08_render_figures.py reads
# directly from the validation/multiomics/<fixture>/fixtures/*
# artifacts.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

for fixture in sim gtex_ukb plant; do
    dir="$REPO_ROOT/validation/multiomics/$fixture"
    if [[ ! -d "$dir" ]]; then
        echo "[stage 07] WARN: $dir not present; skipping"
        continue
    fi
    if [[ -f "$dir/compare.py" ]]; then
        echo "[stage 07] ==> compare multiomics/$fixture"
        TORCHGENOMICS_DISABLE_NATIVE="${TORCHGENOMICS_DISABLE_NATIVE:-1}" \
            python3 "$dir/compare.py"
    else
        echo "[stage 07] (multiomics/$fixture: no compare.py; data-only stage)"
    fi
done

echo "[stage 07] complete."
