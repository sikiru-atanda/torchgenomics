#!/usr/bin/env bash
# Stage MDP fixture + pick a high-LD 5-SNP window on chr 1 for hapref.
#
# Window selection (offline scoring on 540 chr1 SNPs):
#   win@343 of chr 1 -- SNPs id1.6, id1.3, id1.2, PZB02144.4, PZB02144.5
#   positions 238902012, 238902078, 238902091, 238902135, 238902252
#   MAF 0.22, 0.22, 0.22, 0.24, 0.15; mean |r| = 0.867
#   This is a tight-LD block at the distal end of chr 1 (B73 AGPv1).
#
# Phenotype: EarHT from mdp_traits.txt (281 taxa intersection -> 279 with EarHT).
#
# Idempotent: skips if data/mdp_*.txt + data/window.json exist with the right
# SHA256s.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
SOURCE_DIR="/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data"

WINDOW_JSON="${DATA_DIR}/window.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${WINDOW_JSON}" \
      && -f "${DATA_DIR}/mdp_numeric.txt" \
      && -f "${DATA_DIR}/mdp_SNP_information.txt" \
      && -f "${DATA_DIR}/mdp_traits.txt" ]]; then
    echo "[hapref fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[hapref fetch]   window:    ${WINDOW_JSON}"
    exit 0
fi

# --- Pre-flight (~3 MB MDP files; trivial RAM) -------------------------------
preflight_check_with_data_size "hapref-fetch" 1 1

mkdir -p "${DATA_DIR}"

echo "[hapref fetch] copying MDP fixture from ${SOURCE_DIR}"
cp "${SOURCE_DIR}/mdp_numeric.txt"          "${DATA_DIR}/"
cp "${SOURCE_DIR}/mdp_SNP_information.txt"  "${DATA_DIR}/"
cp "${SOURCE_DIR}/mdp_traits.txt"           "${DATA_DIR}/"

echo "[hapref fetch] emitting window.json (5-SNP chr1 block + SHA256s)"
Rscript --vanilla "${HERE}/_fetch_window.R" "${DATA_DIR}"

echo "[hapref fetch] done."
echo "[hapref fetch]   window:    ${WINDOW_JSON}"

