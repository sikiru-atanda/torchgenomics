#!/usr/bin/env bash
# Pre-flight gate for the paper reproducibility pipeline.
#
# Sources the project-wide pre-flight library and asserts the cumulative
# headroom needed for the full pipeline (install 11 reference tools +
# stage 3 multi-omics fixtures + run all 14 harnesses + render 7 figures).
#
# Budget rationale:
#   - 11 reference-tool installs (R packages, MetaXcan clone, SMR binary,
#     coxme, lme4, knockoff, gibbsf90+, etc.):  ~3 GB on disk.
#   - Multi-omics fixtures: GTEx v8 cis-eQTL (~1.5 GB) + GLGC LDL (~1.4 GB)
#     + Arabidopsis 1001G + 1001T (~3-4 GB).
#   - Specialty fixtures + cached reference outputs:  ~2 GB.
#   - Peak working set during scan + figure rendering:  ~8 GB.
#   Total disk floor:  50 GB (with 4x headroom multiplier per spec §5.3).
#   Total RAM floor:   16 GB (large enough for the streaming bench p-sweep
#                            + figure rendering with torch + matplotlib).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
# shellcheck source=../../validation/external/_lib/preflight.sh
source "$REPO_ROOT/validation/external/_lib/preflight.sh"

preflight_check "paper-reproducibility" 50 16

echo "[paper reproducibility] Pre-flight OK."
