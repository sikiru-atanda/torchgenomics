#!/usr/bin/env Rscript
# Simulate a known-truth multi-omics triple (SNP, mediator, outcome) with a
# planted causal mediation effect under a kinship-correlated random effect.
#
# Setup (single-mediator causal-mediation model; Imai-Keele-Tingley 2010):
#   - n individuals with a *kinship-structured* polygenic random effect g_y
#     (mimicking SoyMD's elite soybean breeding panels: pedigree + SNP-
#     based kinship is non-negligible).
#   - 1 causal SNP (additive 0/1/2 dosage).
#   - 1 mediator M (gene expression for one cis gene):
#         M = a · SNP + e_m
#   - 1 outcome Y (a quantitative phenotype):
#         Y = c' · SNP + b · M + g_y + e_y
#
#   The polygenic random effect g_y enters **only the outcome**, not the
#   mediator. This corresponds to the standard sequential-ignorability
#   regime (Imai-Keele-Tingley §3.1): g_y is uncorrelated with M after
#   conditioning on SNP, so OLS for `Y ~ M + SNP` is unbiased on b. The
#   LMM correction in TG matters because g_y inflates the residual SE on Y;
#   the kinship-aware reweighting tightens the per-coefficient SE without
#   biasing the point estimate.
#
#   This is the regime where TorchGenomics' `mediate_lmm` and R's
#   `mediation::mediate` should agree on point estimates of a, b, c',
#   ACME, and total — both are unbiased; only the SEs differ.
#
#   True parameters (planted):
#     a (SNP -> M)        = 0.5
#     b (M -> Y | SNP)    = 0.4
#     c' (direct SNP->Y)  = 0.1
#     ACME = a · b        = 0.20
#     Total = a · b + c'  = 0.30
#
# We simulate kinship-correlated g_y via K = AA'/p where A is a small n × p
# IID-Gaussian "ancestor effects" matrix (p = n / 2). This produces a real
# PSD kinship matrix that is non-trivial (not the identity) so the LMM
# correction in TG actually has work to do — TG handles the kinship in the
# null fit while R `mediation::mediate` ignores it (fits two OLS models),
# leading to larger SEs on R's coefficients but the same point estimates.
#
# Outputs (under --output-dir):
#   triple.tsv     # id, snp, mediator, outcome  (the analyzed table)
#   K.tsv          # n × n GRM (used by TG mediate_lmm; mediation::mediate
#                  # ignores it, which is part of the documented divergence)
#   sim_truth.json # planted parameters + sim metadata
#
# Why R (not Python):
#   - Single source of truth for the simulated data; both the R reference
#     run and the Python TorchGenomics run read from data/triple.tsv +
#     data/K.tsv.
#   - The simulator is small (~30 LOC of math) and `mediation` already
#     requires R; keeping the simulator in R avoids a Python-only
#     dependency for the data-staging step.

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
N <- as.integer(parse_arg("--n", "200"))
A_TRUE <- as.numeric(parse_arg("--a", "0.5"))
B_TRUE <- as.numeric(parse_arg("--b", "0.4"))
CPRIME_TRUE <- as.numeric(parse_arg("--c-prime", "0.1"))
SIGMA_G <- as.numeric(parse_arg("--sigma-g", "0.55"))   # sqrt(0.3)
SIGMA_M <- as.numeric(parse_arg("--sigma-m", "0.71"))   # sqrt(0.5)
SIGMA_Y <- as.numeric(parse_arg("--sigma-y", "0.71"))   # sqrt(0.5)
P_ANCESTOR <- as.integer(parse_arg("--p-ancestor", "100"))

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(SEED)

# --- Simulate kinship-structured polygenic effect ---------------------------
# A: n × p IID Gaussian "ancestor effects" matrix.
# K = A A^T / p   (positive semidefinite, tr(K) ~= n).
# g_y = A z / sqrt(p), z ~ N(0, sigma_g^2 I_p)  =>   g_y ~ N(0, sigma_g^2 K).
# g_y enters Y only (not M) — see header for why.
A <- matrix(rnorm(N * P_ANCESTOR), nrow = N, ncol = P_ANCESTOR)
K <- A %*% t(A) / P_ANCESTOR
# Symmetrize numerically (defensive).
K <- 0.5 * (K + t(K))

z <- rnorm(P_ANCESTOR, mean = 0, sd = SIGMA_G)
g_y <- as.numeric(A %*% z) / sqrt(P_ANCESTOR)

# --- Simulate SNP, mediator, outcome -----------------------------------------
snp <- rbinom(N, size = 2, prob = 0.3)            # diploid additive dosage
mediator <- A_TRUE * snp + rnorm(N, 0, SIGMA_M)
outcome <- CPRIME_TRUE * snp + B_TRUE * mediator + g_y + rnorm(N, 0, SIGMA_Y)

# --- Compose data frame ------------------------------------------------------
df <- data.frame(
    id       = sprintf("s%04d", seq_len(N)),
    snp      = as.numeric(snp),
    mediator = mediator,
    outcome  = outcome,
    stringsAsFactors = FALSE
)
write.table(df, file = file.path(OUTPUT_DIR, "triple.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- Save kinship ------------------------------------------------------------
# Plain TSV (n rows × n cols, no header) for trivial Python ingestion.
write.table(K, file = file.path(OUTPUT_DIR, "K.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

# --- Truth manifest ----------------------------------------------------------
truth <- list(
    a            = A_TRUE,
    b            = B_TRUE,
    c_prime      = CPRIME_TRUE,
    acme         = A_TRUE * B_TRUE,
    total        = A_TRUE * B_TRUE + CPRIME_TRUE,
    proportion_mediated = (A_TRUE * B_TRUE) / (A_TRUE * B_TRUE + CPRIME_TRUE),
    n            = N,
    sigma_g      = SIGMA_G,
    sigma_m      = SIGMA_M,
    sigma_y      = SIGMA_Y,
    p_ancestor   = P_ANCESTOR,
    seed         = SEED,
    kinship_trace = sum(diag(K))
)
write_json(truth, path = file.path(OUTPUT_DIR, "sim_truth.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat(sprintf(
    "[simulate_multiomics] wrote n=%d (a=%.2f, b=%.2f, c'=%.2f); ACME=%.3f total=%.3f to %s\n",
    N, A_TRUE, B_TRUE, CPRIME_TRUE,
    truth$acme, truth$total, OUTPUT_DIR
))
