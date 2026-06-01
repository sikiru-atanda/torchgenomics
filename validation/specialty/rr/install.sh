#!/usr/bin/env bash
# Random Regression LMM reference-tool harness installer (validation/specialty/rr/).
#
# Reference tool: lme4 (R) -- open-source Henderson-style LMM.
#
# Rationale for lme4 over ASReml-R / BLUPF90:
#   - ASReml-R is commercial and requires a per-machine license file;
#     the paper harness must be reproducible on any open Linux machine.
#   - BLUPF90 family (gibbs1f90 / blupf90+) is free-of-charge but
#     binary-only, requires manual click-through registration with the
#     Iowa State BLUPF90 group, and has no stable URL; it cannot be
#     bootstrapped from CI.
#   - lme4 (Bates et al. 2015, J Stat Softw 67(1)) is the standard
#     open-source reference for Henderson mixed-model equations in the
#     breeding / longitudinal-genetics literature when ASReml is
#     unavailable. Installable from CRAN with no license check.
#
# Modelling alignment: lme4 fits the same Henderson-style LMM that
# RandomRegressionLMM reduces to under K = I (independent subjects)
# with (1 + t + t^2 | subject) as the random-regression term. Both
# engines therefore see the same statistical model on the same
# fixture; any divergence in per-time beta or variance ratios is a
# numerical / fitter discrepancy, not a model-class mismatch.
#
# Pre-flight: tiny harness (~5 MB fixture). Assert 1 GB disk + 1 GB
# RAM headroom for safety margin.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"

# Idempotency: skip if marker exists and lme4 loads.
if [[ -f "${INSTALL_MARKER}" ]]; then
    if Rscript -e 'suppressWarnings(suppressMessages(library(lme4))); cat(as.character(packageVersion("lme4")), "\n")' >/dev/null 2>&1; then
        # shellcheck disable=SC1090
        source "${INSTALL_MARKER}"
        echo "[rr install] already installed; lme4 ${LME4_VERSION}"
        exit 0
    fi
fi

preflight_check "rr-install" 1 1

for tool in Rscript R sha256sum; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "[rr install] ABORT: required tool not found on PATH: ${tool}"
        exit 1
    fi
done

# Verify lme4 is loadable; if not, attempt CRAN install.
if ! Rscript -e 'suppressWarnings(suppressMessages(library(lme4)))' >/dev/null 2>&1; then
    echo "[rr install] lme4 not loadable -- attempting CRAN install"
    Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org")'
fi

LME4_VERSION="$(Rscript -e 'suppressWarnings(suppressMessages(library(lme4))); cat(as.character(packageVersion("lme4")))' 2>/dev/null)"
LMERTEST_VERSION="$(Rscript -e 'suppressWarnings(suppressMessages(library(lmerTest))); cat(as.character(packageVersion("lmerTest")))' 2>/dev/null || true)"

if [[ -z "${LME4_VERSION}" ]]; then
    echo "[rr install] ABORT: lme4 install / load failed"
    exit 1
fi

echo "[rr install] lme4 ${LME4_VERSION} loaded; smoke-testing random-regression fit"

# Smoke: fit a tiny random-regression (intercept+slope) and confirm convergence.
SMOKE_OK="$(Rscript - <<'SMOKE_EOF'
suppressWarnings(suppressMessages({
    library(lme4)
}))
set.seed(1)
n <- 20; T <- 5
df <- expand.grid(sid=1:n, t=1:T)
df$y <- rnorm(nrow(df)) + 0.3 * df$t
fit <- suppressWarnings(suppressMessages(lmer(y ~ t + (1 + t | sid), data=df)))
cat(if (isSingular(fit)) "singular" else "ok")
SMOKE_EOF
)" || true

if [[ -z "${SMOKE_OK}" ]]; then
    echo "[rr install] ABORT: lme4 smoke test failed (no output)"
    exit 1
fi

echo "[rr install] lme4 smoke test verdict: ${SMOKE_OK}"

{
    printf 'LME4_VERSION="%s"\n' "${LME4_VERSION}"
    printf 'LMERTEST_VERSION="%s"\n' "${LMERTEST_VERSION}"
    printf "R_VERSION=%q\n" "$(R --version | head -1)"
} > "${INSTALL_MARKER}"

echo "[rr install] done. Marker -> ${INSTALL_MARKER}"
