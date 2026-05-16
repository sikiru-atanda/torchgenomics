#!/usr/bin/env bash
# Survival GWAS fixture generation (Tier 3 C1).
#
# Deterministic Weibull Cox PH frailty simulation:
#   n=500, m=20 test SNPs + m_grm=200 background SNPs, h2=0.30,
#   3 causal SNPs with log_hr=0.6 each, ~15% censoring (in-range for
#   the brief 10-20% target), seed=42.
#
# Outputs: data/pheno.csv, data/geno.csv, data/kinship.csv, data/truth.json
# See simulate.py for the algorithmic spec.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"

# Pre-flight: simulate runs entirely in RAM. n=500, m=200 -> ~3-4 MB on disk,
# < 200 MB RSS. We ask for 1 GB disk, 2 GB RAM headroom (generous margin).
preflight_check "survival-fetch" 1 2

mkdir -p "${DATA_DIR}"

# Idempotent: only regenerate if data files are missing.
if [[ -f "${DATA_DIR}/pheno.csv" && -f "${DATA_DIR}/geno.csv" && -f "${DATA_DIR}/kinship.csv" && -f "${DATA_DIR}/truth.json" ]]; then
    echo "[survival fetch] data already present; skipping (rm ${DATA_DIR}/*.csv to regenerate)"
    exit 0
fi

echo "[survival fetch] simulating Weibull Cox PH frailty fixture (seed=42)"
python3 "${HERE}/simulate.py" \
    --out-dir "${DATA_DIR}" \
    --n 500 --m 20 --m-grm 200 \
    --n-causal 3 --log-hr 0.6 --h2 0.30 \
    --censor-rate 0.15 --weibull-shape 1.5 --seed 42

echo "[survival fetch] done. Files in ${DATA_DIR}:"
ls -la "${DATA_DIR}"
