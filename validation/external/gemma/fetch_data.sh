#!/usr/bin/env bash
# Fetch + stage the MDP maize fixture for the GEMMA harness.
#
# Strategy: GEMMA's run_gemma_mdp.sh consumes BIMBAM-format genotype +
# 1-trait/2-trait phenotype + SNP annotation files that already live in
# the canonical checkout at ${ROOT}/gemma_demo/. We copy them into our
# local data/ directory so the harness is self-contained and idempotent.
#
# Idempotent: if data/ already contains the fixture, exit 0 immediately.
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Checksum verification on the source files (committed fixtures, but we
#     still pin sha256 to detect drift).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Source fixture paths (committed; not downloaded) ------------------------
# Search order (worktree root → canonical repo).
SOURCE_DIR=""
for candidate in \
    "${ROOT}/gemma_demo" \
    "${HOME}/Documents/GWAS_Expert/gemma_demo"; do
    if [[ -f "${candidate}/mdp_geno_bimbam.txt" ]]; then
        SOURCE_DIR="${candidate}"
        break
    fi
done

if [[ -z "${SOURCE_DIR}" ]]; then
    echo "[gemma fetch] ABORT: cannot find GEMMA MDP fixture."
    echo "[gemma fetch] Looked for mdp_geno_bimbam.txt in:"
    echo "[gemma fetch]   ${ROOT}/gemma_demo/"
    echo "[gemma fetch]   ${HOME}/Documents/GWAS_Expert/gemma_demo/"
    echo "[gemma fetch] These are committed fixtures from the canonical"
    echo "[gemma fetch] TorchGWAS checkout. Restore gemma_demo/ before retrying."
    exit 1
fi

# --- Pre-flight (~3 MB total; trivial RAM) -----------------------------------
preflight_check_with_data_size "gemma-fetch" 1 1

DATA_DIR="${HERE}/data"
mkdir -p "${DATA_DIR}"

# --- Idempotence check -------------------------------------------------------
required_files=(
    "mdp_geno_bimbam.txt"
    "mdp_pheno1.txt"
    "mdp_pheno2.txt"
    "mdp_snps_bimbam.txt"
)

all_present=1
for f in "${required_files[@]}"; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        all_present=0
        break
    fi
done

if (( all_present == 1 )); then
    echo "[gemma fetch] data already staged in ${DATA_DIR}; skipping copy"
    n_snps=$(wc -l < "${DATA_DIR}/mdp_snps_bimbam.txt")
    n_samples=$(wc -l < "${DATA_DIR}/mdp_pheno1.txt")
    echo "[gemma fetch]   ${n_samples} samples, ${n_snps} SNPs"
    exit 0
fi

# --- Pinned source checksums (committed BIMBAM fixtures) ---------------------
# Captured 2026-05-04 against the canonical TorchGWAS checkout.
declare -A FIXTURE_SHA256=(
    [mdp_geno_bimbam.txt]="9a33dad98be41ea9d6059588b1dc855a9ddf6dc553e282148c75eee9c7541635"
    [mdp_pheno1.txt]="bf5cf095db55e2fd37e3731c312f8911cb08763c824a433e7be28bb4b5d1ece7"
    [mdp_pheno2.txt]="936e25faa19e343caa5bdb0056403867e06589406a42e5d1853d08fb977601f6"
    [mdp_snps_bimbam.txt]="c1f17fb3a810bffa47126cc00b32a3a366d95d84fa1300346792b184a0ce7f29"
)

# --- Copy fixtures + verify checksums ----------------------------------------
echo "[gemma fetch] copying fixtures from ${SOURCE_DIR}/ → ${DATA_DIR}/"
for f in "${required_files[@]}"; do
    src="${SOURCE_DIR}/${f}"
    dst="${DATA_DIR}/${f}"
    if [[ ! -f "${src}" ]]; then
        echo "[gemma fetch] ABORT: missing source fixture: ${src}"
        exit 1
    fi
    cp -f "${src}" "${dst}"
    expected="${FIXTURE_SHA256[$f]:-PENDING}"
    actual=$(sha256sum "${dst}" | awk '{print $1}')
    if [[ "${expected}" == "PENDING" ]]; then
        echo "[gemma fetch] FIRST-RUN HASH ${f}: ${actual}"
    elif [[ "${actual}" != "${expected}" ]]; then
        echo "[gemma fetch] WARN: ${f} sha256 mismatch"
        echo "[gemma fetch]   expected: ${expected}"
        echo "[gemma fetch]   actual:   ${actual}"
    else
        echo "[gemma fetch] [checksum OK] ${f}"
    fi
done

# --- Also copy the source MDP map files for cross-reference ------------------
# These are needed by compare.py to align GEMMA output with TorchGWAS
# (same trait values, but TorchGWAS reads from benchmark/data/mdp_traits.txt).
# We copy them so the harness is self-contained.
for src_dir in \
    "${ROOT}/benchmark/data" \
    "${HOME}/Documents/GWAS_Expert/benchmark/data"; do
    if [[ -f "${src_dir}/mdp_numeric.txt" ]]; then
        for f in mdp_numeric.txt mdp_traits.txt mdp_SNP_information.txt; do
            if [[ -f "${src_dir}/${f}" ]]; then
                cp -f "${src_dir}/${f}" "${DATA_DIR}/${f}"
            fi
        done
        break
    fi
done

# --- Summary -----------------------------------------------------------------
n_snps=$(wc -l < "${DATA_DIR}/mdp_snps_bimbam.txt")
n_samples=$(wc -l < "${DATA_DIR}/mdp_pheno1.txt")
echo "[gemma fetch] done. ${n_samples} samples, ${n_snps} SNPs"
ls -la "${DATA_DIR}"
