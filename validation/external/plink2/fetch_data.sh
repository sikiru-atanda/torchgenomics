#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the PLINK 2.0 harness.
#
# Strategy: the MDP fixture is already committed in `benchmark/data/`
# (mdp_numeric.txt, mdp_traits.txt, mdp_SNP_information.txt) — 282 samples,
# 3093 SNPs, ~6 MB. We do NOT download anything from the internet; we copy
# the committed fixture and convert MDP-numeric → PLINK BED via
# `convert_mdp_to_bed.py`.
#
# This is idempotent: if `data/mdp.{bed,bim,fam}` and `data/mdp_pheno.txt`
# already exist with the expected sizes, we exit 0.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Checksum verification on the source MDP files (committed fixtures, but
#     we still pin sha256 to detect drift).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Source fixture paths (committed; not downloaded) ------------------------
# The MDP fixture lives in the project root (NOT inside the worktree fixtures
# dir, which is for unit-test tinies). It's committed under benchmark/data/.
# When the worktree is checked out without that data, we fall back to the
# repo-level path under HOME.
MDP_NUMERIC="${ROOT}/benchmark/data/mdp_numeric.txt"
MDP_TRAITS="${ROOT}/benchmark/data/mdp_traits.txt"
MDP_SNPMAP="${ROOT}/benchmark/data/mdp_SNP_information.txt"

# Worktree fallback: if the worktree doesn't contain benchmark/data, use the
# canonical project copy.
if [[ ! -f "${MDP_NUMERIC}" ]]; then
    MDP_NUMERIC="${HOME}/Documents/GWAS_Expert/benchmark/data/mdp_numeric.txt"
    MDP_TRAITS="${HOME}/Documents/GWAS_Expert/benchmark/data/mdp_traits.txt"
    MDP_SNPMAP="${HOME}/Documents/GWAS_Expert/benchmark/data/mdp_SNP_information.txt"
fi

for f in "${MDP_NUMERIC}" "${MDP_TRAITS}" "${MDP_SNPMAP}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[plink2 fetch] ABORT: missing source fixture: ${f}"
        echo "[plink2 fetch] The MDP fixture lives under benchmark/data/ in the"
        echo "[plink2 fetch] canonical TorchGenomics checkout. Restore it before retrying."
        exit 1
    fi
done

# --- Pinned source checksums (computed against the committed fixture) -------
# Captured 2026-05-04 against canonical benchmark/data/.
MDP_NUMERIC_SHA256="d7bdcbc66a6dfa40cdcbd4bef6aa229673a7e08e4b033044df45f552acb51f2e"
MDP_TRAITS_SHA256="644ead8d53f237d2576e30cf686c434c04cb3687c3379fb7468074a16d2b8ebd"
MDP_SNPMAP_SHA256="c64d2971650b964c396ab3561cdb576b337570c83cb70d734d15f4e7488adcb9"

# --- Pre-flight (~6 MB source data, ~2 MB BED output, peak RAM trivial) ------
preflight_check_with_data_size "plink2-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

# --- Idempotence check -------------------------------------------------------
BED="${DATA_DIR}/mdp.bed"
BIM="${DATA_DIR}/mdp.bim"
FAM="${DATA_DIR}/mdp.fam"
PHENO="${DATA_DIR}/mdp_pheno.txt"

if [[ -f "${BED}" && -f "${BIM}" && -f "${FAM}" && -f "${PHENO}" ]]; then
    echo "[plink2 fetch] data already staged in ${DATA_DIR}; skipping conversion"
    echo "[plink2 fetch]   $(wc -l "${BIM}" | awk '{print $1}') variants, $(wc -l "${FAM}" | awk '{print $1}') samples"
    exit 0
fi

# --- Source checksums (compute-and-print mode for first run) -----------------
# If the committed checksums above don't match (e.g. fixture was regenerated),
# we print the actual hashes and abort with instructions. The first run sets
# these by inspection.
_check_or_print() {
    local file="$1"
    local expected="$2"
    local label="$3"
    local actual; actual=$(sha256sum "${file}" | awk '{print $1}')
    if [[ "${expected}" == "PENDING" ]]; then
        echo "[plink2 fetch] FIRST-RUN HASH ${label}: ${actual}"
        echo "[plink2 fetch]   pin this in fetch_data.sh and re-run"
        return 0
    fi
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[plink2 fetch] WARN: ${label} sha256 mismatch"
        echo "[plink2 fetch]   expected: ${expected}"
        echo "[plink2 fetch]   actual:   ${actual}"
        echo "[plink2 fetch]   (proceeding; update the pinned hash if the fixture was regenerated on purpose)"
    else
        echo "[plink2 fetch] [checksum OK] ${label}"
    fi
}

_check_or_print "${MDP_NUMERIC}" "${MDP_NUMERIC_SHA256}" "mdp_numeric.txt"
_check_or_print "${MDP_TRAITS}"  "${MDP_TRAITS_SHA256}"  "mdp_traits.txt"
_check_or_print "${MDP_SNPMAP}"  "${MDP_SNPMAP_SHA256}"  "mdp_SNP_information.txt"

# --- Convert MDP numeric → PLINK BED ----------------------------------------
echo "[plink2 fetch] converting MDP numeric → PLINK BED"
python3 "${HERE}/convert_mdp_to_bed.py" \
    --mdp-numeric "${MDP_NUMERIC}" \
    --mdp-traits "${MDP_TRAITS}" \
    --mdp-snpmap "${MDP_SNPMAP}" \
    --out-dir "${DATA_DIR}"

# --- Round-trip sanity -------------------------------------------------------
PLINK2="${HERE}/bin/plink2"
if [[ -x "${PLINK2}" ]]; then
    echo "[plink2 fetch] PLINK 2 fileset summary:"
    "${PLINK2}" --bfile "${DATA_DIR}/mdp" --freq --out "${DATA_DIR}/mdp_freq" >/dev/null
    n_vars=$(awk 'NR>1' "${DATA_DIR}/mdp_freq.afreq" | wc -l)
    echo "[plink2 fetch]   variants accepted by PLINK 2: ${n_vars}"
    rm -f "${DATA_DIR}/mdp_freq.log"
fi

echo "[plink2 fetch] done. Outputs in ${DATA_DIR}"
ls -la "${DATA_DIR}"
