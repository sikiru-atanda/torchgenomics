#!/usr/bin/env bash
# Provision the MDP-derived per-locus fixture for SuSiE-RSS validation.
# Per NA1 design spec section 5.2.
#
# Strategy: run lmm-scan on MDP genotype + phenotype, take a window of
# p=500 SNPs around the strongest hit, compute_pairwise_ld for R, save
# (z, R, n) triple as torch tensors.
#
# Tier A scope: this script wires the pre-flight + directory layout +
# fixture-existence check. The full Python helper (run lmm-scan + extract
# per-locus z/R/n) is a Tier B deliverable. For Tier A, this script emits
# a placeholder note and exits 0 so downstream run_*.sh scripts SKIP
# gracefully when the fixture isn't present (the test_external_susieR.py
# test is marked external/golden so CI skips it cleanly when data/ is empty).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HERE}/data"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pre-flight (per-locus tensors are <100 MB; runtime peak modest) --------
# Use the shared preflight_check (positional: tool / disk_gb / ram_gb).
preflight_check "susieR-fetch_data" 1 2

mkdir -p "${DATA_DIR}"

# --- Source MDP fixture ------------------------------------------------------
MDP_DIR="${HOME}/Documents/GWAS_Expert/benchmark/data"
if [ ! -f "${MDP_DIR}/mdp_genotype_test.hmp.txt" ]; then
    echo "[susieR fetch_data] FATAL: MDP fixture not found at ${MDP_DIR}/mdp_genotype_test.hmp.txt"
    exit 2
fi

# --- Idempotence: skip if locus tensors already provisioned ------------------
if [ -f "${DATA_DIR}/locus_z.pt" ] && [ -f "${DATA_DIR}/locus_R.pt" ]; then
    echo "[susieR fetch_data] fixture already provisioned at ${DATA_DIR}/; skipping."
    exit 0
fi

# --- Tier A placeholder -----------------------------------------------------
echo "[susieR fetch_data] NOTE: Tier A placeholder."
echo "[susieR fetch_data] Full Tier B implementation runs:"
echo "    torchgenomics lmm-scan --genotype ${MDP_DIR}/mdp_genotype_test.hmp.txt \\"
echo "                       --phenotype ${MDP_DIR}/mdp_traits.txt \\"
echo "                       --output ${DATA_DIR}/mdp_lmm"
echo "    plus extract_locus.py to slice a p=500 window around the top hit and"
echo "    save data/locus_{z,R,meta,n}.pt for SuSiE-RSS comparison."
echo "[susieR fetch_data] Tier A: test_external_susieR.py SKIPs cleanly when data/ is empty."
exit 0
