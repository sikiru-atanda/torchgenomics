#!/usr/bin/env Rscript
# Run TwoSampleMR + MRPRESSO on the simulated sumstats and emit a JSON
# blob the Python harness (compare.py) parses.
#
# Methods exercised:
#   1. mr_ivw                 -- inverse-variance weighted
#   2. mr_egger_regression    -- MR-Egger (intercept + slope)
#   3. mr_weighted_median     -- weighted median (bootstrap SE)
#   4. mr_presso              -- pleiotropy global test + outlier-corrected estimate
#
# Output layout (outputs/twosamplemr_results.json):
#   {
#     "ivw":             {"b": ..., "se": ..., "pval": ..., "nsnp": ...},
#     "egger":           {"b": ..., "se": ..., "pval": ..., "nsnp": ...,
#                         "intercept": ..., "intercept_se": ..., "intercept_pval": ...},
#     "weighted_median": {"b": ..., "se": ..., "pval": ..., "nsnp": ...},
#     "mr_presso": {
#         "global_p": ...,
#         "global_rss": ...,
#         "outlier_indices_one_based": [...],
#         "n_outliers": ...,
#         "raw_b": ..., "raw_se": ..., "raw_pval": ...,
#         "corrected_b": ..., "corrected_se": ..., "corrected_pval": ...
#     },
#     "package_versions": {"TwoSampleMR": ..., "MRPRESSO": ..., "R": ...}
#   }

suppressPackageStartupMessages({
    library(TwoSampleMR)
    library(MRPRESSO)
    library(jsonlite)
})

# --- CLI args ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA_DIR <- parse_arg("--data-dir", "data")
OUT_DIR <- parse_arg("--output-dir", "outputs")
N_PERMS <- as.integer(parse_arg("--n-perms", "1000"))
SEED <- as.integer(parse_arg("--seed", "42"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf(
    "[run_twosamplemr] data_dir=%s out_dir=%s n_perms=%d seed=%d\n",
    DATA_DIR, OUT_DIR, N_PERMS, SEED
))

# --- Load simulated sumstats -------------------------------------------------
sumstats <- read.delim(file.path(DATA_DIR, "sumstats.tsv"), stringsAsFactors = FALSE)
stopifnot(nrow(sumstats) >= 3)

# --- Build a TwoSampleMR-compatible harmonized data frame --------------------
# TwoSampleMR's `mr()` expects the columns laid out by `harmonise_data`:
#   id.exposure, id.outcome, exposure, outcome, beta.exposure, se.exposure,
#   beta.outcome, se.outcome, mr_keep, ...
# Since our SNPs are pre-aligned (same A1 in both), we build the structure
# directly without invoking harmonise_data.
harmonized <- data.frame(
    SNP            = sumstats$SNP,
    beta.exposure  = sumstats$beta_exposure,
    se.exposure    = sumstats$se_exposure,
    pval.exposure  = 2 * pnorm(-abs(sumstats$beta_exposure / sumstats$se_exposure)),
    effect_allele.exposure = "A",
    other_allele.exposure  = "G",
    beta.outcome   = sumstats$beta_outcome,
    se.outcome     = sumstats$se_outcome,
    pval.outcome   = 2 * pnorm(-abs(sumstats$beta_outcome / sumstats$se_outcome)),
    effect_allele.outcome  = "A",
    other_allele.outcome   = "G",
    mr_keep        = TRUE,
    id.exposure    = "EXP",
    id.outcome     = "OUT",
    exposure       = "exposure",
    outcome        = "outcome",
    stringsAsFactors = FALSE
)

# --- Method 1+2+3: mr() with three methods -----------------------------------
mr_methods <- c("mr_ivw", "mr_egger_regression", "mr_weighted_median")

# TwoSampleMR's mr_weighted_median uses set.seed internally. We seed the
# global RNG just before so the bootstrap SE is reproducible.
set.seed(SEED)
mr_results <- mr(harmonized, method_list = mr_methods)

# Pull each row by method label.
get_row <- function(df, method_label) {
    row <- df[df$method == method_label, ]
    if (nrow(row) != 1) {
        stop(sprintf("expected exactly 1 row for method '%s', got %d",
                     method_label, nrow(row)))
    }
    list(
        b      = row$b,
        se     = row$se,
        pval   = row$pval,
        nsnp   = row$nsnp
    )
}

ivw <- get_row(mr_results, "Inverse variance weighted")
egger <- get_row(mr_results, "MR Egger")
median_res <- get_row(mr_results, "Weighted median")

# Egger intercept (separate function in TwoSampleMR).
egger_int <- mr_pleiotropy_test(harmonized)
stopifnot(nrow(egger_int) == 1)
egger$intercept      <- egger_int$egger_intercept
egger$intercept_se   <- egger_int$se
egger$intercept_pval <- egger_int$pval

# --- Method 4: MR-PRESSO -----------------------------------------------------
# Note: MRPRESSO's `mr_presso` uses sample(...) for permutations. We seed
# explicitly so the global p and outlier set are reproducible. n_perms must
# be > nrow(data) for the global test (Verbanck 2018), enforced by the
# package itself.
set.seed(SEED)
presso <- mr_presso(
    BetaOutcome = "beta.outcome", BetaExposure = "beta.exposure",
    SdOutcome  = "se.outcome",   SdExposure   = "se.exposure",
    OUTLIERtest = TRUE, DISTORTIONtest = TRUE,
    data = harmonized, NbDistribution = N_PERMS, SignifThreshold = 0.05
)

# MRPRESSO returns a list-of-lists. The global RSS test sits in
# `presso$`MR-PRESSO results`$`Global Test``, the corrected/raw IVW slopes
# sit in `presso$`Main MR results``, and outlier indices come from
# `presso$`MR-PRESSO results`$`Outlier Test``.
main_mr <- presso[["Main MR results"]]
mr_results_block <- presso[["MR-PRESSO results"]]
global_test <- mr_results_block[["Global Test"]]
outlier_test <- mr_results_block[["Outlier Test"]]

# main_mr always contains 2 rows: row 1 = "Raw" IVW, row 2 = "Outlier-
# corrected" IVW. If there were no outliers, the corrected row's b/se/p are
# NA. We emit them as JSON nulls.
raw <- main_mr[main_mr$"MR Analysis" == "Raw", ]
corrected <- main_mr[main_mr$"MR Analysis" == "Outlier-corrected", ]
stopifnot(nrow(raw) == 1)

# Outlier indices: the per-SNP outlier test has nrow(harmonized) rows; the
# significant ones are flagged with `Pvalue < SignifThreshold`. Indices are
# 1-based in R; we expose both 1-based (matches R) and 0-based (for Python).
outlier_pvals <- outlier_test$Pvalue
# `Pvalue` may be the literal string "<1e-04" if the perm-tail truncates;
# force-convert to numeric (NA for the literal — treat as significant).
out_p_num <- suppressWarnings(as.numeric(outlier_pvals))
out_p_num[is.na(out_p_num)] <- 0.0  # treat "<1e-04" as 0 → outlier
outlier_idx_1b <- which(out_p_num < 0.05)
n_outliers <- length(outlier_idx_1b)

mr_presso_block <- list(
    global_p   = unname(global_test$Pvalue),
    global_rss = unname(global_test$RSSobs),
    outlier_indices_one_based = as.integer(outlier_idx_1b),
    outlier_indices_zero_based = as.integer(outlier_idx_1b - 1L),
    n_outliers = n_outliers,
    n_perms    = N_PERMS,
    raw_b      = unname(raw$"Causal Estimate"),
    raw_se     = unname(raw$Sd),
    raw_pval   = unname(raw$"P-value"),
    corrected_b   = if (nrow(corrected) == 1) unname(corrected$"Causal Estimate") else NA_real_,
    corrected_se  = if (nrow(corrected) == 1) unname(corrected$Sd) else NA_real_,
    corrected_pval = if (nrow(corrected) == 1) unname(corrected$"P-value") else NA_real_
)

# --- Compose JSON output -----------------------------------------------------
out <- list(
    ivw             = ivw,
    egger           = egger,
    weighted_median = median_res,
    mr_presso       = mr_presso_block,
    package_versions = list(
        TwoSampleMR = as.character(packageVersion("TwoSampleMR")),
        MRPRESSO    = as.character(packageVersion("MRPRESSO")),
        R           = R.version.string,
        seed        = SEED,
        n_perms     = N_PERMS
    ),
    n_instruments = nrow(harmonized)
)

out_path <- file.path(OUT_DIR, "twosamplemr_results.json")
write_json(out, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf("[run_twosamplemr] wrote %s\n", out_path))

# Echo a one-line summary for quick eyeballing.
cat(sprintf(
    "[run_twosamplemr] IVW b=%.4f se=%.4f p=%.3e | Egger b=%.4f int=%.4f | WMed b=%.4f | PRESSO global_p=%.3e n_outliers=%d\n",
    ivw$b, ivw$se, ivw$pval,
    egger$b, egger$intercept,
    median_res$b,
    mr_presso_block$global_p, mr_presso_block$n_outliers
))
