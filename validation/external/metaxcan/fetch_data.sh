#!/usr/bin/env bash
# MetaXcan harness data fetch --- sources ../_lib/preflight.sh (pre-flight gate).
# Generates a deterministic simulated PrediXcan-compatible fixture under
# validation/external/metaxcan/data/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
MODEL_DB="${DATA_DIR}/model.db"
COV_GZ="${DATA_DIR}/model.txt.gz"
SUMSTATS_GZ="${DATA_DIR}/gwas_sumstats.txt.gz"
TRUTH_JSON="${DATA_DIR}/sim_truth.json"

if [[ -f "${MODEL_DB}" && -f "${COV_GZ}" && -f "${SUMSTATS_GZ}" && -f "${TRUTH_JSON}" ]]; then
    echo "[metaxcan fetch] data already staged at ${DATA_DIR}; skipping"
    exit 0
fi

preflight_check_with_data_size "metaxcan-fetch" 1 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "[metaxcan fetch] ABORT: python3 not on PATH."
    exit 1
fi

mkdir -p "${DATA_DIR}"

echo "[metaxcan fetch] generating simulated PrediXcan fixture (seed=42)"
python3 "${HERE}/simulate_fixture.py" --output-dir "${DATA_DIR}" --seed 42

echo "[metaxcan fetch] done."
ls -la "${DATA_DIR}"
