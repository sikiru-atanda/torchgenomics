#!/usr/bin/env Rscript
# Simulate 3-trait sumstats for the hyprcoloc reference harness.
#
# Design (per Foley et al. 2021 §Simulations):
#   - m markers, K = 3 traits.
#   - One "causal" SNP at index `causal-idx` (1-based) with effect
#     `causal-beta` on ALL three traits (shared causal architecture).
#   - All other markers have small N(0, background-sd^2) noise effects on
#     each trait (no horizontal pleiotropy).
#   - SEs are fixed at `se` for all traits and markers (matches a balanced-
#     sample-size design; ploidy-invariant assumption).
#   - Truth: cluster {T1, T2, T3} colocalizes; candidate SNP = `causal-idx`.
#
# Outputs:
#   data/sumstats.tsv     # long format: SNP, trait, beta, se
#   data/sim_truth.json   # ground truth manifest
#
# The TorchGWAS side reads the long-format sumstats and reshapes to per-trait
# SumStats objects. The R side reshapes to (m x K) matrices.

suppressPackageStartupMessages({
    library(jsonlite)
})

# --- CLI args ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

OUTPUT_DIR    <- parse_arg("--output-dir", "data")
SEED          <- as.integer(parse_arg("--seed", "42"))
M             <- as.integer(parse_arg("--m", "100"))
K             <- as.integer(parse_arg("--k", "3"))
CAUSAL_IDX    <- as.integer(parse_arg("--causal-idx", "50"))
CAUSAL_BETA   <- as.numeric(parse_arg("--causal-beta", "0.5"))
BACKGROUND_SD <- as.numeric(parse_arg("--background-sd", "0.05"))
SE            <- as.numeric(parse_arg("--se", "0.1"))

if (CAUSAL_IDX < 1 || CAUSAL_IDX > M) {
    stop("--causal-idx must be in [1, m]")
}
if (K < 2) {
    stop("--k must be >= 2 for hyprcoloc")
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
set.seed(SEED)

# --- Simulate effects --------------------------------------------------------
# Background noise effects (m x K).
beta_mat <- matrix(rnorm(M * K, mean = 0, sd = BACKGROUND_SD), nrow = M, ncol = K)
se_mat   <- matrix(SE, nrow = M, ncol = K)

# Plant the shared causal effect across all K traits.
beta_mat[CAUSAL_IDX, ] <- CAUSAL_BETA

# --- Compose long-format sumstats -------------------------------------------
trait_names <- paste0("T", seq_len(K))
snp_ids <- sprintf("rs%05d", seq_len(M))

rows <- do.call(rbind, lapply(seq_len(K), function(k) {
    data.frame(
        SNP   = snp_ids,
        trait = trait_names[k],
        beta  = beta_mat[, k],
        se    = se_mat[, k],
        stringsAsFactors = FALSE
    )
}))

write.table(
    rows,
    file = file.path(OUTPUT_DIR, "sumstats.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
)

# --- Truth manifest ----------------------------------------------------------
truth <- list(
    seed = SEED,
    m = M,
    k = K,
    trait_names = trait_names,
    causal_index_one_based = CAUSAL_IDX,
    causal_index_zero_based = CAUSAL_IDX - 1L,
    causal_snp = snp_ids[CAUSAL_IDX],
    causal_beta = CAUSAL_BETA,
    background_sd = BACKGROUND_SD,
    se = SE,
    expected_cluster_one_based = seq_len(K),
    expected_cluster_zero_based = seq_len(K) - 1L,
    expected_cluster_traits = trait_names
)
write_json(
    truth,
    path = file.path(OUTPUT_DIR, "sim_truth.json"),
    pretty = TRUE, auto_unbox = TRUE
)

cat(sprintf(
    "[simulate] wrote %d markers x %d traits to %s; causal=%s; beta=%.3f\n",
    M, K, file.path(OUTPUT_DIR, "sumstats.tsv"), snp_ids[CAUSAL_IDX], CAUSAL_BETA
))

