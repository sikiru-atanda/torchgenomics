#!/usr/bin/env bash
# hyprcoloc install (R packages from GitHub, via conda gmp/mpfr).
#
# Pillar B contract:
#   - Memory pre-flight before any work.
#   - Pinned tool versions (hyprcoloc @ 0348bbd; HEAD of jrs95/hyprcoloc as of
#     2024-04-08; verify in .install_marker).
#   - Idempotent: re-running checks the marker + smoke-loads packages.
#
# Why conda gmp/mpfr (not system yum):
#   - Rmpfr and gmp R packages need gmp.h / mpfr.h headers + libgmp / libmpfr
#     at compile and link time.
#   - System install would need root (yum install gmp-devel mpfr-devel).
#   - Conda-forge supplies both headers and shared libs into the user's base
#     conda env without sudo.
#
# Why RcppEigen 0.3.3.9.4 (not the CRAN HEAD 0.3.4.0.2):
#   - hyprcoloc src uses the legacy Eigen-3.3 operator()(int,int) scalar
#     overload. RcppEigen 0.3.4 ships Eigen-3.4 which routes
#     operator()(double,double) to the IndexedView overload, emitting
#     compile errors like "cannot convert IndexedView<...> to double".
#   - We pin RcppEigen to 0.3.3.9.4 (CRAN archive) for this harness.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

INSTALL_MARKER="${HERE}/.install_marker"
CONDA_BIN="/home/sikiru.atanda/miniconda3/bin/conda"
CONDA_PREFIX_LIB="/home/sikiru.atanda/miniconda3/lib"
CONDA_PREFIX_INC="/home/sikiru.atanda/miniconda3/include"

# Pinned versions (record-of-record).
HYPRCOLOC_SHA="0348bbd"
RCPPEIGEN_VERSION="0.3.3.9.4"



# --- Idempotence check -------------------------------------------------------
if [[ -f "${INSTALL_MARKER}" ]]; then
    _check_loadable() {
        Rscript --vanilla -e '
            Sys.setenv(LD_LIBRARY_PATH = paste("/home/sikiru.atanda/miniconda3/lib",
                                               Sys.getenv("LD_LIBRARY_PATH"),
                                               sep = ":"))
            ok <- TRUE
            for (p in c("gmp", "Rmpfr", "iterpc", "arrangements", "hyprcoloc")) {
                if (!requireNamespace(p, quietly = TRUE)) {
                    cat("missing:", p, "\n")
                    ok <- FALSE
                }
            }
            if (!ok) quit(status = 1)
            cat("hyprcoloc:", as.character(packageVersion("hyprcoloc")), "\n")
            cat("Rmpfr:",     as.character(packageVersion("Rmpfr")),     "\n")
            cat("gmp:",       as.character(packageVersion("gmp")),       "\n")
        ' 2>/dev/null
    }
    if _check_loadable; then
        echo "[hyprcoloc install] already installed (marker + load OK); skipping"
        _check_loadable | sed 's/^/  /'
        exit 0
    else
        echo "[hyprcoloc install] marker present but packages no longer load; re-installing"
        rm -f "${INSTALL_MARKER}"
    fi
fi

# --- Pre-flight --------------------------------------------------------------
# Conda gmp/mpfr (~5 MB), R-package compile (~500 MB peak; RcppEigen + Rmpfr).
preflight_check_with_data_size "hyprcoloc-install" 2 1

# --- Verify Rscript + conda --------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[hyprcoloc install] ABORT: Rscript not on PATH."
    exit 1
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "[hyprcoloc install] ABORT: conda binary not found at ${CONDA_BIN}."
    exit 1
fi

R_VERSION="$(Rscript --vanilla -e 'cat(R.version.string)' 2>/dev/null)"
echo "[hyprcoloc install] R: ${R_VERSION}"

# --- Step 1: conda gmp + mpfr ------------------------------------------------
echo "[hyprcoloc install] step 1: conda-forge gmp + mpfr + pkg-config (base env)"
"${CONDA_BIN}" install -n base -c conda-forge gmp mpfr pkg-config -y 2>&1 | tail -3

# Sanity-check: headers and libs end up where we expect.
for f in gmp.h mpfr.h; do
    if [[ ! -f "${CONDA_PREFIX_INC}/${f}" ]]; then
        echo "[hyprcoloc install] ABORT: ${CONDA_PREFIX_INC}/${f} not present after conda install"
        exit 1
    fi
done
for f in libgmp.so libmpfr.so; do
    if [[ ! -f "${CONDA_PREFIX_LIB}/${f}" ]]; then
        echo "[hyprcoloc install] ABORT: ${CONDA_PREFIX_LIB}/${f} not present after conda install"
        exit 1
    fi
done

GMP_VERSION="$("${CONDA_BIN}" list -n base 2>&1 | awk '/^gmp / {print $2}')"
MPFR_VERSION="$("${CONDA_BIN}" list -n base 2>&1 | awk '/^mpfr / {print $2}')"
echo "[hyprcoloc install] conda gmp=${GMP_VERSION} mpfr=${MPFR_VERSION}"


# --- Step 2: patch ~/.R/Makevars so R sees conda include/lib -----------------
# We do NOT rely on PKG_CONFIG_PATH. Instead we append the conda include dir
# to CPPFLAGS and the conda lib dir + rpath to LDFLAGS. Existing CC/CXX lines
# are preserved; the conda flags are appended (+=) so other R-package builds
# keep working. Old Makevars is backed up to ~/.R/Makevars.bak.hyprcoloc.
Rscript --vanilla -e '
    home <- Sys.getenv("HOME")
    mk_dir <- file.path(home, ".R")
    if (!dir.exists(mk_dir)) dir.create(mk_dir, showWarnings = FALSE)
    mk <- file.path(mk_dir, "Makevars")
    bak <- file.path(mk_dir, "Makevars.bak.hyprcoloc")
    if (file.exists(mk) && !file.exists(bak)) file.copy(mk, bak)
    cur <- if (file.exists(mk)) readLines(mk, warn = FALSE) else character(0)
    keep <- cur[!grepl("hyprcoloc-conda-marker", cur, fixed = TRUE)]
    add <- c(
        "# hyprcoloc-conda-marker (appended by validation/external/hyprcoloc/install.sh)",
        "CPPFLAGS+=-I/home/sikiru.atanda/miniconda3/include",
        "LDFLAGS+=-L/home/sikiru.atanda/miniconda3/lib -Wl,-rpath,/home/sikiru.atanda/miniconda3/lib"
    )
    writeLines(c(keep, add), mk)
    cat("[Makevars] wrote ", mk, "\n")
'

# --- Step 3: install R deps via R --------------------------------------------
echo "[hyprcoloc install] step 3: install R packages (this can take 10-15 min)"
HYPRCOLOC_SHA="${HYPRCOLOC_SHA}" RCPPEIGEN_VERSION="${RCPPEIGEN_VERSION}" \
CONDA_PREFIX_LIB="${CONDA_PREFIX_LIB}" \
Rscript --vanilla -e '
options(repos = c(CRAN = "https://cloud.r-project.org"))
Sys.setenv(LD_LIBRARY_PATH = paste(Sys.getenv("CONDA_PREFIX_LIB"),
                                   Sys.getenv("LD_LIBRARY_PATH"),
                                   sep = ":"))

HYPRCOLOC_SHA <- Sys.getenv("HYPRCOLOC_SHA")
RCPPEIGEN_VERSION <- Sys.getenv("RCPPEIGEN_VERSION")

# (a) remotes
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", quiet = TRUE)
}

# (b) RcppEigen pinned to RCPPEIGEN_VERSION.
cur <- if (requireNamespace("RcppEigen", quietly = TRUE)) {
    as.character(packageVersion("RcppEigen"))
} else NA
if (is.na(cur) || cur != RCPPEIGEN_VERSION) {
    cat(sprintf("[install] RcppEigen %s -> %s\n", cur, RCPPEIGEN_VERSION))
    remotes::install_version("RcppEigen", version = RCPPEIGEN_VERSION,
                             repos = "https://cloud.r-project.org",
                             upgrade = "never", quiet = TRUE)
} else {
    cat(sprintf("[install] RcppEigen already at %s\n", cur))
}

# (c) gmp
if (!requireNamespace("gmp", quietly = TRUE)) {
    install.packages("gmp", quiet = TRUE)
}

# (d) Rmpfr (needs gmp + conda mpfr).
if (!requireNamespace("Rmpfr", quietly = TRUE)) {
    install.packages("Rmpfr", quiet = TRUE)
}

# (e) iterpc + arrangements (hyprcoloc cluster enumeration deps).
for (p in c("iterpc", "arrangements")) {
    if (!requireNamespace(p, quietly = TRUE)) {
        install.packages(p, quiet = TRUE)
    }
}

# (f) hyprcoloc at pinned SHA.
if (!requireNamespace("hyprcoloc", quietly = TRUE)) {
    remotes::install_github("jrs95/hyprcoloc", ref = HYPRCOLOC_SHA,
                            upgrade = "never", quiet = TRUE)
}

# (g) jsonlite
if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", quiet = TRUE)
}

cat("\n[install] resolved versions:\n")
for (p in c("RcppEigen", "gmp", "Rmpfr", "iterpc", "arrangements",
            "hyprcoloc", "jsonlite")) {
    cat(sprintf("  %s: %s\n", p, as.character(packageVersion(p))))
}
'


# --- Step 4: smoke test ------------------------------------------------------
echo "[hyprcoloc install] smoke test: load hyprcoloc and run a 3-trait toy"
Rscript --vanilla -e '
    Sys.setenv(LD_LIBRARY_PATH = paste("/home/sikiru.atanda/miniconda3/lib",
                                       Sys.getenv("LD_LIBRARY_PATH"),
                                       sep = ":"))
    suppressPackageStartupMessages({
        library(hyprcoloc)
        library(jsonlite)
    })
    set.seed(0)
    m <- 20; K <- 3
    be <- matrix(rnorm(m * K, sd = 0.05), m, K)
    se <- matrix(0.1, m, K)
    be[10, ] <- c(0.5, 0.5, 0.5)
    r <- hyprcoloc(be, se,
                   trait.names = c("T1", "T2", "T3"),
                   snp.id = paste0("rs", seq_len(m)))
    stopifnot(r$results$candidate_snp == "rs10")
    cat("[smoke] PASS: rs10 recovered; PP =", r$results$posterior_prob, "\n")
'

# --- Persist marker with version info ----------------------------------------
{
    echo "R_VERSION=\"${R_VERSION}\""
    echo "HYPRCOLOC_SHA=\"${HYPRCOLOC_SHA}\""
    echo "RCPPEIGEN_VERSION=\"${RCPPEIGEN_VERSION}\""
    echo "CONDA_GMP_VERSION=\"${GMP_VERSION}\""
    echo "CONDA_MPFR_VERSION=\"${MPFR_VERSION}\""
    Rscript --vanilla -e '
        for (p in c("hyprcoloc", "Rmpfr", "gmp", "RcppEigen",
                    "iterpc", "arrangements", "jsonlite")) {
            cat(sprintf("%s_VERSION=%s\n", toupper(p), packageVersion(p)))
        }
    '
} > "${INSTALL_MARKER}"

echo "[hyprcoloc install] done. Marker -> ${INSTALL_MARKER}"
sed 's/^/  /' "${INSTALL_MARKER}"
