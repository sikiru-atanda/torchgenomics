#!/usr/bin/env bash
# SAIGE install — pinned wzhou88/saige:1.4.4 docker (or podman) image.
#
# SAIGE (Zhou et al. 2018, Nat. Genet.) is the gold-standard biobank-scale
# GLMM for binary / ordinal / survival traits with kinship correction +
# saddlepoint approximation. It's also notoriously hard to install:
#   1. Bioconductor distribution requires bgenix + savvy + htslib + a long
#      C++ toolchain bring-up.
#   2. Source builds at https://github.com/saigegit/SAIGE share that pain.
#   3. Docker image is the only path that "just works" reproducibly.
#
# Strategy:
#   - Try (1) docker pull wzhou88/saige:1.4.4 first.
#   - Fall back to (2) Bioconductor R install only if no docker is present.
#   - Mark .infra_blocker if both paths fail (per spec §9), and exit 0 so
#     downstream tests skip cleanly with a documented reason.
#
# Idempotent: presence of `.install_marker` (and a usable image / package)
# short-circuits subsequent runs.
#
# Pillar B contract:
#   - Memory pre-flight before any download (image is ~3 GB).
#   - Pinned tag wzhou88/saige:1.4.4 (no "latest").
#   - Failure modes documented as infra-blocker artifacts, not silent skips.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

# --- Pinned image ----------------------------------------------------------
# 1.4.4 is the most recent release at time of harness implementation
# (2026-04-30). It pins SAIGE R-package version 1.4.4 atop R 4.2.3 +
# bgenix + savvy. Sub-version drift (1.4.x → 1.5.x) introduces minor-but-
# observable differences in the saddlepoint cutoff defaults; pinning here
# is what reproducibility requires.
SAIGE_IMAGE_REPO="docker.io/wzhou88/saige"
SAIGE_IMAGE_TAG="1.4.4"
SAIGE_IMAGE="${SAIGE_IMAGE_REPO}:${SAIGE_IMAGE_TAG}"

INSTALL_MARKER="${HERE}/.install_marker"
INFRA_BLOCKER_MARKER="${HERE}/.infra_blocker"
INSTALL_METHOD="${HERE}/.install_method"

# --- Idempotence: skip if already installed --------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    echo "[saige install] already installed (marker: ${INSTALL_MARKER})"
    if [[ -f "${INSTALL_METHOD}" ]]; then
        cat "${INSTALL_METHOD}"
    fi
    exit 0
fi

# --- If a previous attempt was an infra-blocker, don't retry automatically -
if [[ -f "${INFRA_BLOCKER_MARKER}" ]]; then
    echo "[saige install] previously marked infra-blocker; skipping."
    echo "[saige install] To retry: rm ${INFRA_BLOCKER_MARKER}"
    cat "${INFRA_BLOCKER_MARKER}"
    exit 0
fi

# Pre-flight (~3 GB image; peak RAM small for pull, ~4 GB for actual run).
preflight_check_with_data_size "saige-install" 10 4

# --- Try 1: docker / podman pull -------------------------------------------
if command -v docker >/dev/null 2>&1; then
    echo "[saige install] (1/2) trying docker pull ${SAIGE_IMAGE}"
    # Probe daemon first; bail to next path on failure.
    if docker info >/dev/null 2>&1; then
        # NOTE: on RHEL 9 hosts `docker` is a podman alias; podman enforces
        # short-name resolution and fails on bare `wzhou88/saige:tag`. We
        # therefore always pass the fully-qualified `docker.io/...` form,
        # which works for both real Docker and podman.
        if docker pull "${SAIGE_IMAGE}" 2>&1; then
            # Smoke test: verify the image entrypoint resolves to an
            # executable script we expect.
            if docker run --rm "${SAIGE_IMAGE}" step1_fitNULLGLMM.R --help \
                   >/dev/null 2>&1; then
                cat > "${INSTALL_METHOD}" <<EOF
USE_DOCKER=1
DOCKER_IMAGE=${SAIGE_IMAGE}
SAIGE_IMAGE_REPO=${SAIGE_IMAGE_REPO}
SAIGE_IMAGE_TAG=${SAIGE_IMAGE_TAG}
EOF
                touch "${INSTALL_MARKER}"
                echo "[saige install] OK via docker/podman: ${SAIGE_IMAGE}"
                exit 0
            else
                echo "[saige install] docker pull succeeded but smoke test failed; trying R fallback"
            fi
        else
            echo "[saige install] docker pull failed; trying R fallback"
        fi
    else
        echo "[saige install] docker daemon not reachable; trying R fallback"
    fi
fi

# --- Try 2: Bioconductor R install (slow; ~30 min cold) --------------------
if command -v Rscript >/dev/null 2>&1; then
    echo "[saige install] (2/2) trying Bioconductor R install of SAIGE"
    if Rscript -e '
        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager", repos="https://cloud.r-project.org")
        if (!requireNamespace("SAIGE", quietly = TRUE)) {
            BiocManager::install("SAIGE", update = FALSE, ask = FALSE)
        }
        cat("SAIGE installed:", as.character(packageVersion("SAIGE")), "\n")
    ' 2>&1; then
        cat > "${INSTALL_METHOD}" <<EOF
USE_R=1
DOCKER_IMAGE=
EOF
        touch "${INSTALL_MARKER}"
        echo "[saige install] OK via Bioconductor R"
        exit 0
    fi
fi

# --- Fallthrough: mark infra-blocker (do not fail the harness) -------------
cat > "${INFRA_BLOCKER_MARKER}" <<EOF
SAIGE install failed: neither docker pull nor Bioconductor R install succeeded.

Diagnostics at the time of failure:
  - docker present: $(command -v docker >/dev/null 2>&1 && echo yes || echo no)
  - Rscript present: $(command -v Rscript >/dev/null 2>&1 && echo yes || echo no)

Per spec §9, this is a documented infra-blocker outcome. The harness scripts
remain in place; pytest tests skip cleanly (see tests/test_external_saige.py).

To retry:
  rm ${INFRA_BLOCKER_MARKER}
  bash $(basename "${BASH_SOURCE[0]}")
EOF

echo "[saige install] WARNING: marked infra-blocker (see ${INFRA_BLOCKER_MARKER})"
exit 0
