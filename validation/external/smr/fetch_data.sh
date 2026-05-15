#!/usr/bin/env bash
# SMR harness data fetch --- sources ../_lib/preflight.sh (pre-flight gate).
# Generates a deterministic simulated SMR-compatible fixture under
# validation/external/smr/data/. The fixture is independent-SNP by
# construction so the LD-weighted HEIDI test inside SMR reduces to the
# diagonal (delta-method) variance form that torchgwas/postgwas/_smr.py
# computes. See README for the rationale (Zhu 2016 simulation framework).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
REF_BED="${DATA_DIR}/ref.bed"
REF_BIM="${DATA_DIR}/ref.bim"
REF_FAM="${DATA_DIR}/ref.fam"
GWAS_MA="${DATA_DIR}/gwas.ma"
ESD="${DATA_DIR}/probe.esd"
FLIST="${DATA_DIR}/probe.flist"
TRUTH="${DATA_DIR}/sim_truth.json"

NEED_REGEN=0
for f in "${REF_BED}" "${REF_BIM}" "${REF_FAM}" "${GWAS_MA}" "${ESD}" "${FLIST}" "${TRUTH}"; do
    if [[ ! -f "${f}" ]]; then
        NEED_REGEN=1
        break
    fi
done

if (( NEED_REGEN == 0 )); then
    echo "[smr fetch] data already staged at ${DATA_DIR}; skipping"
    exit 0
fi

preflight_check_with_data_size "smr-fetch" 1 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "[smr fetch] ABORT: python3 not on PATH."
    exit 1
fi
python3 -c "import numpy, scipy" >/dev/null 2>&1 || {
    echo "[smr fetch] ABORT: python deps numpy+scipy required."
    exit 1
}

mkdir -p "${DATA_DIR}"
echo "[smr fetch] generating simulated SMR fixture (seed=42)"
python3 "${HERE}/simulate_smr_fixture.py" --output-dir "${DATA_DIR}" --seed 42
echo "[smr fetch] done."
ls -la "${DATA_DIR}"
