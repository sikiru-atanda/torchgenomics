#!/usr/bin/env bash
# TorchGenomics Genome Biology Methods paper — one-command reproducibility.
#
# Regenerates every number in every figure / table from the validation
# harnesses + bench infrastructure committed under
#   - validation/external/<tool>/        (reference-tool harnesses)
#   - validation/multiomics/<fixture>/   (multi-omics fixtures)
#   - validation/specialty/<model>/      (specialty model fixtures)
#   - bench/native_speedups.py           (native kernel speedup distribution)
#   - tests/test_streaming_memory.py     (streaming-memory regression net)
#
# All stages source the project pre-flight gate
#   validation/external/_lib/preflight.sh
# which asserts disk + RAM headroom before any download / install / run.
#
# Usage (from the repo root OR from this directory):
#   bash paper/reproducibility/reproduce_paper.sh
#
# Output:
#   paper/reproducibility/output/F{1..7}.pdf
#   paper/reproducibility/manifest.json   (every numeric result + commit SHA)
#
# Optional environment overrides:
#   REPRODUCE_SKIP_INSTALL=1  — skip stage 01 (use installed tools as-is)
#   REPRODUCE_SKIP_FETCH=1    — skip stage 02 (use staged fixtures as-is)
#   REPRODUCE_QUICK=1         — pass --quick to bench scripts when supported

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

bash preflight.sh

if [[ "${REPRODUCE_SKIP_INSTALL:-0}" != "1" ]]; then
    bash stages/01_install_references.sh
else
    echo "[reproduce] SKIP stage 01 (REPRODUCE_SKIP_INSTALL=1)"
fi

if [[ "${REPRODUCE_SKIP_FETCH:-0}" != "1" ]]; then
    bash stages/02_stage_fixtures.sh
else
    echo "[reproduce] SKIP stage 02 (REPRODUCE_SKIP_FETCH=1)"
fi

bash stages/03_run_references.sh
bash stages/04_run_torchgenomics.sh
bash stages/05_run_streaming_bench.sh
bash stages/06_run_native_bench.sh
bash stages/07_run_multiomics.sh

python3 stages/08_render_figures.py

echo ""
echo "Reproducibility pipeline complete."
echo "  Figures:  $HERE/output/F{1..7}.pdf"
echo "  Manifest: $HERE/manifest.json"
