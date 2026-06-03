#!/usr/bin/env Rscript
# Simulate two-sample MR data with a known causal effect theta_true.
#
# Setup (Bowden et al. 2016 §Simulations, simplified):
#   - K instruments, each with effect beta_x_j on the exposure.
#   - True causal effect of exposure on outcome: theta = theta_true.
#   - Outcome effect for SNP j (no horizontal pleiotropy):
#         beta_y_j = theta * beta_x_j + eps_j,    eps_j ~ N(0, sigma_y^2)
#   - We add a small fraction of pleiotropic SNPs whose outcome effect
#     deviates from the causal direction, so MR-PRESSO has something to
#     detect.
#   - SEs are fixed to plausible values (se_x ~ 0.02, se_y ~ 0.03) so the
#     Wald ratios have realistic precision.
#
# Outputs:
#   data/sumstats.tsv     # SNP, beta_exposure, se_exposure, beta_outcome, se_outcome
#   data/sim_truth.json   # {theta_true, K, n_pleiotropic, ...}
#
# Why R (not Python):
#   - This script feeds run_twosamplemr.R, which lives in R already.
#   - Single source of truth for the simulated data; both the R reference
#     run and the Python TorchGenomics run read from data/sumstats.tsv.

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

OUTPUT_DIR <- parse_arg("--output-dir", "data")
SEED <- as.integer(parse_arg("--seed", "42"))
K <- as.integer(parse_arg("--k", "30"))
THETA_TRUE <- as.numeric(parse_arg("--theta", "0.5"))
N_PLEIO <- as.integer(parse_arg("--n-pleiotropic", "3"))
SIGMA_PLEIO <- as.numeric(parse_arg("--pleio-sigma", "0.15"))

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(SEED)

# --- Simulate instruments ----------------------------------------------------
# Each SNP has a random exposure effect drawn N(0, 0.1^2). We sign-flip half
# of them so the Wald ratios are not all positive (mirrors a real GWAS).
beta_exposure <- rnorm(K, mean = 0, sd = 0.1)
# Force at least some sign diversity so the median isn't degenerate.
beta_exposure[1:floor(K / 3)] <- abs(beta_exposure[1:floor(K / 3)])
beta_exposure[(floor(K / 3) + 1):floor(2 * K / 3)] <- -abs(
    beta_exposure[(floor(K / 3) + 1):floor(2 * K / 3)]
)

se_exposure <- rep(0.02, K)
se_outcome <- rep(0.03, K)

# Outcome effect under the IVW model: by_j = theta * bx_j + eps_j
beta_outcome_clean <- THETA_TRUE * beta_exposure + rnorm(K, mean = 0, sd = 0.005)

# Inject pleiotropic SNPs: outcome effect drifts away from theta * bx by
# extra noise. These should be detected by MR-PRESSO as outliers.
pleio_idx <- sample(seq_len(K), size = N_PLEIO, replace = FALSE)
beta_outcome <- beta_outcome_clean
beta_outcome[pleio_idx] <- beta_outcome_clean[pleio_idx] + rnorm(
    N_PLEIO, mean = 0, sd = SIGMA_PLEIO
)

# --- Compose data frame ------------------------------------------------------
SNP <- sprintf("rs%05d", seq_len(K))
df <- data.frame(
    SNP = SNP,
    beta_exposure = beta_exposure,
    se_exposure = se_exposure,
    beta_outcome = beta_outcome,
    se_outcome = se_outcome,
    pleiotropic = seq_len(K) %in% pleio_idx,
    stringsAsFactors = FALSE
)

write.table(
    df,
    file = file.path(OUTPUT_DIR, "sumstats.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
)

# --- Truth manifest ----------------------------------------------------------
truth <- list(
    theta_true = THETA_TRUE,
    K = K,
    n_pleiotropic = N_PLEIO,
    pleio_sigma = SIGMA_PLEIO,
    pleio_indices_zero_based = sort(pleio_idx - 1L),  # zero-based for Python
    pleio_indices_one_based = sort(pleio_idx),
    seed = SEED,
    se_exposure = unique(se_exposure)[1],
    se_outcome = unique(se_outcome)[1]
)
write_json(
    truth,
    path = file.path(OUTPUT_DIR, "sim_truth.json"),
    pretty = TRUE, auto_unbox = TRUE
)

cat(sprintf(
    "[simulate_mr] wrote %d instruments to %s; theta_true=%.3f; %d pleiotropic\n",
    K, file.path(OUTPUT_DIR, "sumstats.tsv"), THETA_TRUE, N_PLEIO
))
