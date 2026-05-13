#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the GAPIT harness.
#
# GAPIT's run_gapit.R consumes the canonical MDP fixture (mdp_traits.txt,
# mdp_numeric.txt, mdp_SNP_information.txt) from benchmark/data/. We copy
# them into our local data/ directory so the harness is self-contained.
#
# Idempotent: if data/ already contains the fixture, exit 0 immediately.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Checksum verification on the source files.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Source fixture paths (committed; not downloaded) ------------------------
SOURCE_DIR=""
for candidate in \
    "${ROOT}/benchmark/data" \
    "${HOME}/Documents/GWAS_Expert/benchmark/data"; do
    if [[ -f "${candidate}/mdp_numeric.txt" ]]; then
        SOURCE_DIR="${candidate}"
        break
    fi
done

if [[ -z "${SOURCE_DIR}" ]]; then
    echo "[gapit fetch] ABORT: cannot find MDP source fixture."
    echo "[gapit fetch] Looked for mdp_numeric.txt in:"
    echo "[gapit fetch]   ${ROOT}/benchmark/data/"
    echo "[gapit fetch]   ${HOME}/Documents/GWAS_Expert/benchmark/data/"
    echo "[gapit fetch] These are committed fixtures from the canonical"
    echo "[gapit fetch] TorchGWAS checkout. Restore benchmark/data/ before retrying."
    exit 1
fi

# --- Pre-flight (~3 MB total; trivial RAM) -----------------------------------
preflight_check_with_data_size "gapit-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

# --- Idempotence check -------------------------------------------------------
required_files=(
    "mdp_numeric.txt"
    "mdp_traits.txt"
    "mdp_SNP_information.txt"
)

all_present=1
for f in "${required_files[@]}"; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        all_present=0
        break
    fi
done

if (( all_present == 1 )); then
    echo "[gapit fetch] data already staged in ${DATA_DIR}; skipping copy"
    exit 0
fi

# --- Pinned source checksums (committed MDP fixture) -------------------------
declare -A FIXTURE_SHA256=(
    [mdp_numeric.txt]="d7bdcbc66a6dfa40cdcbd4bef6aa229673a7e08e4b033044df45f552acb51f2e"
    [mdp_traits.txt]="644ead8d53f237d2576e30cf686c434c04cb3687c3379fb7468074a16d2b8ebd"
    [mdp_SNP_information.txt]="c64d2971650b964c396ab3561cdb576b337570c83cb70d734d15f4e7488adcb9"
)

# --- Copy fixtures + verify checksums ----------------------------------------
echo "[gapit fetch] copying fixtures from ${SOURCE_DIR}/ → ${DATA_DIR}/"
for f in "${required_files[@]}"; do
    src="${SOURCE_DIR}/${f}"
    dst="${DATA_DIR}/${f}"
    if [[ ! -f "${src}" ]]; then
        echo "[gapit fetch] ABORT: missing source fixture: ${src}"
        exit 1
    fi
    cp -f "${src}" "${dst}"
    expected="${FIXTURE_SHA256[$f]:-PENDING}"
    actual=$(sha256sum "${dst}" | awk '{print $1}')
    if [[ "${expected}" == "PENDING" ]]; then
        echo "[gapit fetch] FIRST-RUN HASH ${f}: ${actual}"
    elif [[ "${actual}" != "${expected}" ]]; then
        echo "[gapit fetch] WARN: ${f} sha256 mismatch"
        echo "[gapit fetch]   expected: ${expected}"
        echo "[gapit fetch]   actual:   ${actual}"
    else
        echo "[gapit fetch] [checksum OK] ${f}"
    fi
done

# --- Summary -----------------------------------------------------------------
n_snps=$(($(wc -l < "${DATA_DIR}/mdp_numeric.txt") - 1))
n_samples=$(($(wc -l < "${DATA_DIR}/mdp_traits.txt") - 1))
echo "[gapit fetch] done. ${n_samples} samples, ${n_snps} SNPs"
ls -la "${DATA_DIR}"
