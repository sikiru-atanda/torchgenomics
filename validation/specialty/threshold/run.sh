#!/usr/bin/env bash
# Threshold-linear harness reference run -- sources ../../external/_lib/preflight.sh.
#
# BLUPF90+ parses DATAFILE / FILE with a fixed-width buffer, so we stage the
# data + pedigree into the OUT_DIR via symlink and call gibbsf90+ with
# relative paths.  Order of operations:
#
#   1. Stage pheno.txt + pedigree.dat (symlink) into outputs/.
#   2. Write renf90.par using relative DATAFILE / FILE paths.
#   3. Run gibbsf90+: prompts answered via stdin
#      (parameter file, number of samples, burn-in, thin).
#   4. Run postgibbsf90 to produce posterior summaries.
#
# Variance components are clamped to the simulator truth via OPTION
# fix_residual_var + fix_var_genetic so the test isolates the threshold
# sampling / fixed-effects solve.  Chain settings: 5000 samples, burn-in
# 1000, thin 5 (~3 min on 1 core for n=300, c=3 threshold).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
if [[ ! -f "${INSTALL_MARKER}" ]]; then
    echo "[thr run] ABORT: ${INSTALL_MARKER} missing.  Run install.sh first."
    exit 1
fi
source "${INSTALL_MARKER}"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
PARFILE="${OUT_DIR}/renf90.par"

for f in "${DATA_DIR}/pheno.txt" "${DATA_DIR}/pedigree.dat" \
         "${DATA_DIR}/sim_truth.json"; do
    if [[ ! -f "${f}" ]]; then
        echo "[thr run] ABORT: ${f} missing.  Run fetch_data.sh first."
        exit 1
    fi
done

preflight_check_with_data_size "thr-run" 1 2

mkdir -p "${OUT_DIR}"

# Stage data + pedigree as symlinks so BLUPF90 fixed-width buffer is happy
for src_name in pheno.txt pedigree.dat; do
    src="${DATA_DIR}/${src_name}"
    dst="${OUT_DIR}/${src_name}"
    if [[ ! -e "${dst}" ]]; then
        ln -sf "${src}" "${dst}"
    fi
done

echo "[thr run] step 1: write renf90.par"
Rscript "${HERE}/make_parfile.R" \
    --truth "${DATA_DIR}/sim_truth.json" \
    --pheno "pheno.txt" \
    --pheno-data "${DATA_DIR}/pheno.txt" \
    --pedigree "pedigree.dat" \
    --out "${PARFILE}"

# Step 2: launch gibbsf90+.  Prompt order:
#   name of parameter file?
#   number of samples?
#   length of burn-in?
#   thinning?
echo "[thr run] step 2: gibbsf90+ (5000 samples, 1000 burn-in, thin 5)"
cd "${OUT_DIR}"
set +e
printf "%s\n%s\n%s\n%s\n" "renf90.par" "5000" "1000" "5" \
    | "${BLUPF90_GIBBSF90_BIN}" > "${OUT_DIR}/gibbs.log" 2>&1
GIBBS_RC=$?
set -e
if (( GIBBS_RC != 0 )); then
    echo "[thr run] ABORT: gibbsf90+ exited rc=${GIBBS_RC}"
    tail -30 "${OUT_DIR}/gibbs.log"
    exit 1
fi

echo "[thr run] step 3: postgibbsf90 (posterior summary)"
set +e
# postgibbsf90 prompts: parameter file, burn-in, thinning, (a final \n).
printf "%s\n%s\n%s\n%s\n%s\n" "renf90.par" "1000" "5" "0" "" \
    | "${BLUPF90_POSTGIBBSF90_BIN}" >> "${OUT_DIR}/gibbs.log" 2>&1
POSTG_RC=$?
set -e
if (( POSTG_RC != 0 )); then
    echo "[thr run] WARNING: postgibbsf90 rc=${POSTG_RC} (solutions may still be valid)"
    tail -20 "${OUT_DIR}/gibbs.log"
fi
cd "${HERE}"

# gibbsf90+ writes the converged point estimate to outputs/solutions.
# postgibbsf90 (if it ran cleanly) emits posterior summaries to outputs/postmean*.
if [[ ! -f "${OUT_DIR}/last_solutions" && ! -f "${OUT_DIR}/final_solutions" ]]; then
    echo "[thr run] ABORT: neither solutions nor final_solutions produced."
    ls -la "${OUT_DIR}"
    tail -40 "${OUT_DIR}/gibbs.log"
    exit 1
fi

echo "[thr run] done. Outputs in ${OUT_DIR}"
ls -la "${OUT_DIR}"

