#!/usr/bin/env bash
# Run the haplo.stats reference (haplo.em + haplo.glm) on the staged
# 5-SNP chr1 window.  Idempotent: skips if outputs/haplo_results.json exists.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
OUT_JSON="${OUT_DIR}/haplo_results.json"

mkdir -p "${OUT_DIR}"

# Pre-flight: 281 taxa x 10 columns trivial; haplo.glm uses <50 MB RAM.
preflight_check_with_data_size "hapref-run" 1 1

if [[ -f "${OUT_JSON}" ]]; then
    echo "[hapref run] ${OUT_JSON} exists; re-running to refresh"
fi

if ! command -v Rscript >/dev/null 2>&1; then
    echo "[hapref run] ABORT: Rscript not on PATH."
    exit 1
fi

echo "[hapref run] launching run.R"
Rscript --vanilla "${HERE}/run.R" --data-dir "${DATA_DIR}" --output-dir "${OUT_DIR}"

echo "[hapref run] done. -> ${OUT_JSON}"

