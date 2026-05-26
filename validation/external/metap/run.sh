#!/usr/bin/env bash
# Invoke run_reference.R against the synthetic fixture.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[metap run] ABORT: ${INSTALL_MARKER} missing. Run install.sh first."
    exit 1
fi

P_TSV="${HERE}/data/pvalues.tsv"
D_TSV="${HERE}/data/data_matrix.tsv"
OUT="${HERE}/outputs/reference.tsv"

for f in "${P_TSV}" "${D_TSV}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[metap run] ABORT: ${f} missing. Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "metap-run" 1 1
mkdir -p "${HERE}/outputs"

Rscript "${HERE}/run_reference.R" \
    --p-tsv "${P_TSV}" \
    --data-tsv "${D_TSV}" \
    --out "${OUT}"

if [[ ! -f "${OUT}" ]]; then
    echo "[metap run] ABORT: reference did not produce ${OUT}"
    exit 1
fi
echo "[metap run] OK: $(wc -l < ${OUT}) lines in ${OUT}"
