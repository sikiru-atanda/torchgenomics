#!/usr/bin/env bash
# Run the reference estimator for direct/indirect effects on the sib-pair fixture.
#
# Reference estimator (Young et al. 2022, section 2.1, "Sib-pair OLS"):
#   For each SNP j, fit OLS:
#       Y = mu + beta_direct * (g_i - g_bar_f) + beta_between * g_bar_f + e
#   where g_bar_f = mean of sib genotypes within family f.
#   This recovers:
#       beta_direct  ≈ planted beta_d   (within-family slope, attenuation-free)
#       beta_between ≈ planted (beta_d + 2*beta_i)  (between-family slope)
#       beta_indirect = (beta_between - beta_direct) / 2
#   The factor of 2 follows from g_bar_f = (g_M + g_F)/2 under random mating
#   and 2 sibs per family (Young 2022 eq. 2-4).
#
#   Attenuation factor = beta_direct_OLS / beta_OLS_marginal
#       where beta_OLS_marginal is the standard one-SNP regression of Y on g
#       (no family-mean covariate). Captures how much population-structure /
#       indirect confounding inflates the marginal estimate.
#
# This is a self-contained Python implementation of the OLS estimator; no
# external tool dependency. It is the FALLBACK path (paper simulator + paper
# estimator) when snipar is not installable on the host. If snipar is
# installed, the snipar-backed branch is selected instead.
#
# Outputs:
#   outputs/reference.tsv  : per-SNP [SNP, beta_d_ref, se_d_ref, beta_i_ref, se_i_ref,
#                                     beta_marginal_ref, attenuation_ref]
#   outputs/run.log        : run metadata (ref_tool, n, m, wall_time, version)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"

source "${ROOT}/validation/external/_lib/preflight.sh"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
REF_TOOL_FILE="${HERE}/.ref_tool"

preflight_check_with_data_size "family-run" 1 2

if [[ ! -f "${REF_TOOL_FILE}" ]]; then
    echo "[family run] ABORT: ${REF_TOOL_FILE} missing. Run install.sh first."
    exit 1
fi
REF_TOOL="$(< "${REF_TOOL_FILE}")"
echo "[family run] reference tool: ${REF_TOOL}"

for f in pheno.tsv geno.tsv parent_geno.tsv family.tsv truth.tsv meta.json; do
    if [[ ! -f "${DATA_DIR}/${f}" ]]; then
        echo "[family run] ABORT: ${DATA_DIR}/${f} missing. Run fetch_data.sh first."
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

case "${REF_TOOL}" in
    snipar)
        echo "[family run] running snipar-backed reference estimator"
        REF_SCRIPT="${HERE}/reference_snipar.py"
        if [[ ! -f "${REF_SCRIPT}" ]]; then
            echo "[family run] ABORT: ${REF_SCRIPT} missing (expected for snipar path)"
            exit 1
        fi
        python3 "${REF_SCRIPT}" --data-dir "${DATA_DIR}" --output-dir "${OUT_DIR}"
        ;;
    paper_simulator)
        echo "[family run] running paper-simulator reference (OLS direct+indirect)"
        REF_SCRIPT="${HERE}/reference_paper.py"
        if [[ ! -f "${REF_SCRIPT}" ]]; then
            echo "[family run] ABORT: ${REF_SCRIPT} missing (expected for paper path)"
            exit 1
        fi
        python3 "${REF_SCRIPT}" --data-dir "${DATA_DIR}" --output-dir "${OUT_DIR}"
        ;;
    *)
        echo "[family run] ABORT: unknown reference tool '${REF_TOOL}'"
        exit 1
        ;;
esac

echo "[family run] outputs:"
ls -la "${OUT_DIR}"
echo
echo "[family run] reference.tsv head:"
head -5 "${OUT_DIR}/reference.tsv"
