#!/usr/bin/env bash
# Stage simulated two-trait sumstats for the coloc.abf reference harness.
#
# Why simulated (not real GTEx/eQTL):
#   - Real eQTL + GWAS sumstats require dbGaP / external access tokens and
#     pin to specific protected releases; that breaks reproducibility.
#   - coloc.abf is summary-statistic only and does not need real biology
#     to test correctness. Three scenarios (shared / distinct / null) exercise
#     every branch of the H0..H4 decomposition.
#
# Scenarios (50 SNPs each, planted causal at idx 25 (1-based) where applicable):
#   1. shared    -- both traits causal at idx 25 (theta_y = 0.9 * beta_x);
#                   expect H4 dominant.
#   2. distinct  -- trait1 causal at idx 25, trait2 causal at idx 40;
#                   expect H3 dominant.
#   3. null      -- both traits null (no signal); expect H0 dominant.
#
# Idempotent: if data/scenarios.json exists, exits 0 without touching it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../_lib/preflight.sh
source "${HERE}/../_lib/preflight.sh"

DATA_DIR="${HERE}/data"
SCENARIOS_JSON="${DATA_DIR}/scenarios.json"

# --- Idempotence -------------------------------------------------------------
if [[ -f "${SCENARIOS_JSON}" ]]; then
    echo "[coloc fetch] data already staged in ${DATA_DIR}; skipping"
    echo "[coloc fetch]   scenarios:  ${SCENARIOS_JSON}"
    exit 0
fi

# --- Pre-flight (~10 KB sumstats, peak RAM trivial) --------------------------
preflight_check_with_data_size "coloc-fetch" 1 1

# --- Verify Rscript ----------------------------------------------------------
if ! command -v Rscript >/dev/null 2>&1; then
    echo "[coloc fetch] ABORT: Rscript not on PATH (needed for inline simulator)."
    exit 1
fi

mkdir -p "${DATA_DIR}"

# --- Inline R simulator ------------------------------------------------------
# All three scenarios share the same seed = 42 and the same MAF / SE schedule.
# We emit one tidy TSV per scenario (sumstats_<name>.tsv) plus a manifest
# scenarios.json listing them. The TorchGWAS comparator (compare.py) and the
# R reference (run.R) both read from these TSVs -- single source of truth.
echo "[coloc fetch] simulating 3 two-trait coloc scenarios (seed=42, m=50, causal_idx=25)"
Rscript --vanilla - <<RSCRIPT_EOF
suppressPackageStartupMessages({ library(jsonlite) })

set.seed(42L)
M <- 50L                  # SNPs per scenario
CAUSAL_IDX1 <- 25L        # 1-based; trait1 causal SNP under shared/distinct
CAUSAL_IDX2_DIST <- 40L   # 1-based; trait2 causal SNP under distinct
DATA_DIR <- "${DATA_DIR}"

# Common per-SNP SEs (reflect 1000-Genomes-scale eQTL/GWAS sumstats).
# se ~ 0.04 on the effect-size scale -> z of ~5 for a beta of 0.2.
SE1 <- rep(0.04, M)
SE2 <- rep(0.04, M)

snp_ids <- sprintf("rs%05d", seq_len(M))
chr     <- rep("1", M)
pos     <- as.integer(seq(1000L, by = 1000L, length.out = M))

# Helper: draw N(0, se^2) noise vector with a fresh seed-aware call.
draw_noise <- function(se) rnorm(length(se), mean = 0, sd = se)

# ---- Scenario 1: shared causal ---------------------------------------------
# trait1 has causal beta_x at idx 25; trait2 has beta_y = 0.9 * beta_x at the
# same SNP. Background SNPs are zero-mean noise.
beta1_s <- rep(0.0, M)
beta2_s <- rep(0.0, M)
beta1_s[CAUSAL_IDX1] <- 0.30
beta2_s[CAUSAL_IDX1] <- 0.9 * 0.30
beta1_s <- beta1_s + draw_noise(SE1)
beta2_s <- beta2_s + draw_noise(SE2)

# ---- Scenario 2: distinct causal --------------------------------------------
beta1_d <- rep(0.0, M)
beta2_d <- rep(0.0, M)
beta1_d[CAUSAL_IDX1]      <- 0.30   # trait1 causal at idx 25
beta2_d[CAUSAL_IDX2_DIST] <- 0.30   # trait2 causal at idx 40
beta1_d <- beta1_d + draw_noise(SE1)
beta2_d <- beta2_d + draw_noise(SE2)

# ---- Scenario 3: null (no signal) ------------------------------------------
beta1_n <- draw_noise(SE1)
beta2_n <- draw_noise(SE2)

# ---- Emit per-scenario TSVs ------------------------------------------------
write_scenario <- function(name, b1, b2) {
    df <- data.frame(
        SNP   = snp_ids,
        chr   = chr,
        pos   = pos,
        a1    = "A",
        a2    = "G",
        beta1 = b1,
        se1   = SE1,
        beta2 = b2,
        se2   = SE2,
        stringsAsFactors = FALSE
    )
    out <- file.path(DATA_DIR, paste0("sumstats_", name, ".tsv"))
    write.table(df, file = out, sep = "\t", quote = FALSE, row.names = FALSE)
    cat(sprintf("[simulate_coloc] wrote %s (M=%d, scenario=%s)\n", out, nrow(df), name))
    out
}

path_shared   <- write_scenario("shared",   beta1_s, beta2_s)
path_distinct <- write_scenario("distinct", beta1_d, beta2_d)
path_null     <- write_scenario("null",     beta1_n, beta2_n)

# ---- Manifest --------------------------------------------------------------
# The 1-based causal indices are R-native; we also emit 0-based for Python.
manifest <- list(
    seed           = 42L,
    n_snps         = M,
    causal_idx_one_based  = CAUSAL_IDX1,
    causal_idx_zero_based = CAUSAL_IDX1 - 1L,
    scenarios = list(
        shared = list(
            file = basename(path_shared),
            expected_pp = "H4",
            description = "trait1 + trait2 both causal at idx 25; beta_y = 0.9 * beta_x",
            causal_idx_zero_based = list(trait1 = CAUSAL_IDX1 - 1L,
                                          trait2 = CAUSAL_IDX1 - 1L)
        ),
        distinct = list(
            file = basename(path_distinct),
            expected_pp = "H3",
            description = "trait1 causal at idx 25; trait2 causal at idx 40 (different SNPs)",
            causal_idx_zero_based = list(trait1 = CAUSAL_IDX1 - 1L,
                                          trait2 = CAUSAL_IDX2_DIST - 1L)
        ),
        null = list(
            file = basename(path_null),
            expected_pp = "H0",
            description = "both traits null (no planted signal)",
            causal_idx_zero_based = list(trait1 = NA_integer_, trait2 = NA_integer_)
        )
    )
)
write_json(
    manifest,
    path = file.path(DATA_DIR, "scenarios.json"),
    pretty = TRUE, auto_unbox = TRUE, na = "null"
)

cat(sprintf("[simulate_coloc] manifest -> %s\n", file.path(DATA_DIR, "scenarios.json")))
RSCRIPT_EOF

echo "[coloc fetch] done."
echo "[coloc fetch]   scenarios:  ${SCENARIOS_JSON}"
ls -1 "${DATA_DIR}" | sed 's/^/  /'
