#!/usr/bin/env bash
# Survival GWAS reference-tool install.
#
# Pillar (Tier 3 C1) reference: Therneau coxme (R) -- the canonical Cox PH
# mixed-model implementation. coxme exposes Surv(time, event) ~ X + (1|id)
# with arbitrary positive-semidefinite kernels (varlist = coxmeFull(K))
# and fits the penalized partial likelihood via Laplace-approximated REML
# (Therneau 2003; coxme reference manual section 2.3).
#
# Rationale for coxme over coxKM (the brief's preferred):
#   - coxKM (Cai et al. 2011, Biometrics 67:975) is a kernel-machine score
#     test designed for SNP-set association, not a per-SNP Wald test. Our
#     SurvivalGLMM scans per-SNP via martingale residual score test, so
#     coxKM is not the direct algorithmic counterpart.
#   - coxme provides a Wald / LRT per-SNP fit with frailty exactly like
#     SurvivalGLMM, and is already exercised by tests/test_survival_reference.py.
#   - SAIGE-COX (Bi et al. 2020 SPACox; fallback per brief) is the secondary
#     reference; its install lives under validation/external/saige/.
#
# Install path: distro R + CRAN coxme + survival. RHEL 9.6 + R 4.5.1 has
# both available out of the box on this host (verified via dnf + Rscript).
#
# Idempotent: presence of .install_marker short-circuits.
# Failure modes: writes .infra_blocker if R or coxme cannot be brought up.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
INFRA_BLOCKER="${HERE}/.infra_blocker"

if [[ -f "${INSTALL_MARKER}" ]]; then
    echo "[survival install] already installed (marker: ${INSTALL_MARKER})"
    cat "${INSTALL_MARKER}"
    exit 0
fi

if [[ -f "${INFRA_BLOCKER}" ]]; then
    echo "[survival install] previously marked infra-blocker; skipping."
    cat "${INFRA_BLOCKER}"
    exit 0
fi

# Pre-flight: install is essentially a verification (no large downloads).
# Reserve disk for the optional coxme/survival source compile if the package
# binary cache is empty.
preflight_check "survival-install" 2 2

# Verify R is on PATH.
if ! command -v Rscript >/dev/null 2>&1; then
    cat > "${INFRA_BLOCKER}" <<EOF
R / Rscript not found on PATH. coxme requires R >= 3.0.
On RHEL 9: sudo dnf install -y R-core
On Debian/Ubuntu: sudo apt-get install -y r-base-core
EOF
    echo "[survival install] WARNING: marked infra-blocker (no Rscript)"
    exit 0
fi

R_VER=$(Rscript -e 'cat(R.version$major, R.version$minor, sep=".")' 2>/dev/null || echo unknown)
echo "[survival install] R version: ${R_VER}"

# Try to load coxme + survival; install from CRAN if missing.
echo "[survival install] checking coxme + survival ..."
if ! Rscript -e '
  ok <- requireNamespace("survival", quietly=TRUE) &&
        requireNamespace("coxme", quietly=TRUE)
  if (ok) {
    cat("survival:", as.character(packageVersion("survival")), "\n")
    cat("coxme:", as.character(packageVersion("coxme")), "\n")
    quit(status=0)
  } else {
    quit(status=1)
  }
'; then
    echo "[survival install] coxme not found; attempting CRAN install"
    if ! Rscript -e '
      repo <- "https://cloud.r-project.org"
      if (!requireNamespace("survival", quietly=TRUE))
        install.packages("survival", repos=repo)
      if (!requireNamespace("coxme", quietly=TRUE))
        install.packages("coxme", repos=repo)
      stopifnot(requireNamespace("survival", quietly=TRUE))
      stopifnot(requireNamespace("coxme", quietly=TRUE))
      cat("survival:", as.character(packageVersion("survival")), "\n")
      cat("coxme:", as.character(packageVersion("coxme")), "\n")
    '; then
        cat > "${INFRA_BLOCKER}" <<EOF
coxme install failed. coxme depends on bdsmatrix + Matrix + survival; the
source build requires a C/Fortran toolchain (gcc-gfortran on RHEL/Debian).

To retry:
  rm ${INFRA_BLOCKER}
  bash $(basename "${BASH_SOURCE[0]}")
EOF
        echo "[survival install] WARNING: marked infra-blocker (coxme install failed)"
        exit 0
    fi
fi

COXME_VER=$(Rscript -e 'cat(as.character(packageVersion("coxme")))' 2>/dev/null)
SURVIVAL_VER=$(Rscript -e 'cat(as.character(packageVersion("survival")))' 2>/dev/null)

cat > "${INSTALL_MARKER}" <<EOF
REFERENCE_TOOL=coxme
R_VERSION=${R_VER}
COXME_VERSION=${COXME_VER}
SURVIVAL_VERSION=${SURVIVAL_VER}
INSTALL_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "[survival install] OK: coxme ${COXME_VER}, survival ${SURVIVAL_VER}"
