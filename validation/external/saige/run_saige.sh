#!/usr/bin/env bash
# Run SAIGE Step 1 (null GLMM fit) + Step 2 (per-SNP SPA test) on the
# MDP fixture for the simulated binary phenotype.
#
# SAIGE pipeline overview:
#   Step 1: fit_NULLGLMM
#     - Estimates the null GLMM via PCG (preconditioned conjugate gradient)
#       using a full GRM derived from the same PLINK fileset that Step 2
#       will scan.
#     - Estimates the per-MAC variance-ratio (Zhou 2018 §"variance ratio")
#       which makes Step 2 scalable to millions of SNPs.
#     - Outputs: <prefix>.rda (null model + GRM artifacts) and
#       <prefix>.varianceRatio.txt.
#
#   Step 2: SPAtests
#     - Per-SNP score test against the Step 1 null.
#     - SPA tail correction is automatic for chi-square > spaCutoff (default 2)
#       and for SNPs with case/control imbalance below threshold.
#     - Outputs: text file with BETA, SE, p-value (Tstat, var, AF, MAC,
#       N, AC_Allele2, etc.).
#
# Idempotent: rerun overwrites previous outputs/.
#
# We run inside the wzhou88/saige:1.4.4 container by mounting the harness
# directory at /work and invoking step1_fitNULLGLMM.R + step2_SPAtests.R.
#
# Pillar B contract:
#   - Memory pre-flight before run (Step 1's PCG can spike RAM; we ask 6 GB
#     headroom even though N=281 will use < 200 MB).
#   - Outputs go to ${HERE}/outputs (.gitignored).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
OUT_DIR="${HERE}/outputs"
INSTALL_METHOD_FILE="${HERE}/.install_method"
INFRA_BLOCKER="${HERE}/.infra_blocker"

if [[ -f "${INFRA_BLOCKER}" ]]; then
    echo "[saige run] ABORT: SAIGE install was marked infra-blocker."
    cat "${INFRA_BLOCKER}"
    echo "[saige run] To proceed, rerun install.sh after fixing the underlying issue."
    exit 1
fi
if [[ ! -f "${INSTALL_METHOD_FILE}" ]]; then
    echo "[saige run] ABORT: ${INSTALL_METHOD_FILE} not found. Run install.sh first."
    exit 1
fi
# shellcheck disable=SC1090
source "${INSTALL_METHOD_FILE}"

if [[ ! -f "${DATA_DIR}/sample.bed" || ! -f "${DATA_DIR}/saige_pheno.tsv" ]]; then
    echo "[saige run] ABORT: required staging files missing in ${DATA_DIR}."
    echo "[saige run] Run fetch_data.sh first."
    exit 1
fi

# Pre-flight (~1 GB working set, 6 GB peak RAM).
preflight_check_with_data_size "saige-run" 1 6

mkdir -p "${OUT_DIR}"

# --- Container helper -------------------------------------------------------
# When USE_DOCKER=1, we run inside the SAIGE image, mounting the harness
# directory at /work. Within the container, /work is the harness root, so
# data lives at /work/data, outputs at /work/outputs.
#
# When USE_R=1 (Bioconductor fallback), we'd Rscript directly — but the
# install.sh path always tries docker first and only falls through if it
# fails, and we currently observe docker working on RHEL 9.6. The R path
# would need additional shell wiring; we leave it documented in install.sh
# and short-circuit here if the user hits it.

if [[ "${USE_DOCKER:-}" == "1" ]]; then
    # NOTE: ":Z" tells podman/docker to relabel the SELinux context on the
    # mount so the container can write to it. On RHEL 9 with rootless podman
    # (this host) the container's root uid is uid-mapped back to the host
    # user, so created files end up owned by the host user. This is the
    # cleanest cross-engine way to make the harness writable; on real
    # Docker the ":Z" suffix is also accepted (silently ignored on
    # systems without SELinux).
    SAIGE_RUN=(
        docker run --rm
        -v "${HERE}:/work:Z"
        --workdir /work
        "${DOCKER_IMAGE}"
    )
elif [[ "${USE_R:-}" == "1" ]]; then
    echo "[saige run] ABORT: R-fallback runner is not currently wired."
    echo "[saige run] Install via docker (preferred) or extend run_saige.sh"
    echo "[saige run] to invoke step1_fitNULLGLMM.R directly via Rscript."
    exit 1
else
    echo "[saige run] ABORT: install method indicator is malformed."
    cat "${INSTALL_METHOD_FILE}"
    exit 1
fi

# --- Step 1: null GLMM fit (PCG + variance ratio) ---------------------------
# Notes on flags:
#   --plinkFile=/work/data/sample
#     SAIGE will resolve <prefix>.{bed,bim,fam}.
#   --phenoFile=/work/data/saige_pheno.tsv
#   --phenoCol=ybin
#     The simulated 0/1 case-control trait.
#   --traitType=binary
#   --covarColList=age,sex,PC1,PC2
#     Match the regenie harness (PC1+PC2) plus age + sex covariates we
#     synthesized to exercise SAIGE's covariate plumbing.
#   --qCovarColList=sex
#     sex is treated as categorical (binary).
#   --sampleIDColinphenoFile=IID
#     Column whose values match plinkFile FAM IIDs.
#   --nThreads=2
#     SAIGE PCG parallelizes over markers; 2 threads is plenty on 281×~3k.
#   --LOCO=FALSE
#     LOCO Step 1 produces per-chrom predictors; we disable since the
#     fixture has only 10 chroms × ~300 SNPs each, which is below LOCO's
#     useful regime. Step 2 then uses the single full-GRM null model.
#   --IsOverwriteVarianceRatioFile=TRUE
#     Lets the script overwrite an existing varianceRatio.txt for idempotent
#     reruns.
echo "[saige run] (1/2) Step 1: fitting null GLMM (binary, PCG)"
"${SAIGE_RUN[@]}" step1_fitNULLGLMM.R \
    --plinkFile=/work/data/sample \
    --phenoFile=/work/data/saige_pheno.tsv \
    --phenoCol=ybin \
    --covarColList=age,sex,PC1,PC2 \
    --qCovarColList=sex \
    --sampleIDColinphenoFile=IID \
    --traitType=binary \
    --outputPrefix=/work/outputs/step1 \
    --nThreads=2 \
    --LOCO=FALSE \
    --IsOverwriteVarianceRatioFile=TRUE

# --- Step 2: per-SNP SPA tests ---------------------------------------------
# Notes on flags:
#   --bedFile=/work/data/sample.bed (and .bim, .fam)
#   --GMMATmodelFile=/work/outputs/step1.rda
#     The Step 1 null model artifact.
#   --varianceRatioFile=/work/outputs/step1.varianceRatio.txt
#   --SAIGEOutputFile=/work/outputs/step2.txt
#     Whitespace-separated table with one row per SNP.
#   --LOCO=FALSE matches Step 1.
#   --is_Firth_beta=TRUE / --pCutoffforFirth=0.01
#     Firth-corrected β estimates for low-count SNPs (matches regenie's
#     --firth --pThresh 0.01). Without this, SAIGE returns the SPA
#     p-value but a non-Firth β/SE that diverges noticeably from
#     TG's BinaryGLMM β.
#   --minMAF=0.01 / --minMAC=5
#     Drop very-rare variants where SAIGE's SPA is unstable on N=281.
echo "[saige run] (2/2) Step 2: SPA tests"
"${SAIGE_RUN[@]}" step2_SPAtests.R \
    --bedFile=/work/data/sample.bed \
    --bimFile=/work/data/sample.bim \
    --famFile=/work/data/sample.fam \
    --GMMATmodelFile=/work/outputs/step1.rda \
    --varianceRatioFile=/work/outputs/step1.varianceRatio.txt \
    --SAIGEOutputFile=/work/outputs/step2.txt \
    --LOCO=FALSE \
    --is_Firth_beta=TRUE \
    --pCutoffforFirth=0.01 \
    --minMAF=0.01 \
    --minMAC=5

echo "[saige run] outputs:"
ls -la "${OUT_DIR}"
echo
echo "[saige run] step2.txt head:"
head -3 "${OUT_DIR}/step2.txt"
