#!/usr/bin/env Rscript
# OCF fixture generator -- simulates G + nuisance covariates with a known SNP
# effect under confounding, per Plan B C6 brief (Chernozhukov et al. 2018).
#
# Data-generating process (partially-linear model, eq. 1.1 in the paper):
#
#   y_i = theta_0 * g_i + l(W_i) + sigma_y * eps_i
#   g_i = m(W_i) + sigma_g * v_i
#
# where:
#   * theta_0  = 0.30 is the SNP effect we are trying to estimate (fixed across reps).
#   * W_i in R^d_W is a vector of nuisance covariates (confounders).
#   * l(W), m(W) are non-linear functions of W (confounding through nuisance).
#   * eps_i, v_i are independent standard normals.
#
# By construction the SNP g_i is correlated with the confounders W_i through
# m(W_i). A naive OLS y ~ g + W with linear W will be biased because the
# true m and l are quadratic + interaction. A DML estimator that learns
# m, l on K-1 folds and tests on the held-out fold is consistent with
# asymptotic chi^2(1) null + Normal sampling distribution for theta_hat.
#
# Output layout (one file per rep, plus shared meta):
#   data/
#       fixture_meta.json     -- {n, m, reps, theta0, seed, dW, sigma_y, sigma_g}
#       rep_001.rds           -- list(Y=numeric(n), G=matrix(n,m), W=matrix(n,dW), K=matrix(n,n))
#       rep_002.rds
#       ...
#
# Why .rds + parquet alternatives: we keep the reference (R) and the
# torchgwas (Python) sides on a common .rds that Python loads via
# pyreadr in compare.py, OR Python parses an .npz mirror written alongside.
# To keep the dep surface minimal we also write rep_NNN.npz from R using
# data.table::fwrite per matrix (compare.py reads the .npz).

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
    library(MASS)
})

# --- CLI args --------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

N      <- as.integer(parse_arg("--n",   "400"))
M      <- as.integer(parse_arg("--m",   "20"))
REPS   <- as.integer(parse_arg("--reps", "100"))
SEED   <- as.integer(parse_arg("--seed", "42"))
DW     <- as.integer(parse_arg("--dW",   "5"))
SIG_Y  <- as.numeric(parse_arg("--sigma-y", "1.0"))
SIG_G  <- as.numeric(parse_arg("--sigma-g", "1.0"))
THETA0 <- as.numeric(parse_arg("--theta0",  "0.30"))
CAUSAL <- as.integer(parse_arg("--causal-snp", "1"))  # 1-based index
OUT    <- parse_arg("--out-dir", "data")

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[generate] n=%d m=%d reps=%d dW=%d theta0=%.4f seed=%d out=%s\n",
            N, M, REPS, DW, THETA0, SEED, OUT))

# --- DGP functions (paper eq. 1.1) ----------------------------------------
# m(W) = 1 + W1 + 0.25 * W2^2 + 0.5 * W3 * W4 -- non-linear confounder -> SNP
# l(W) = -0.5 + 0.5 * W1 + 0.25 * W2 + 0.5 * W3^2 + 0.25 * sin(W5)
#
# Both have linear AND non-linear pieces in W. The fixture is engineered so
# that:
#   * a naive linear adjustment (OLS y ~ g + W) is biased (E[m(W) - lin(W)] != 0).
#   * a DML estimator with a quadratic-feature ridge learner recovers theta0
#     to within Monte Carlo noise + Op(n^{-1/2}).

m_W <- function(W) {
    1.0 + W[, 1] + 0.25 * W[, 2]^2 + 0.5 * W[, 3] * W[, 4]
}

l_W <- function(W) {
    -0.5 + 0.5 * W[, 1] + 0.25 * W[, 2] + 0.5 * W[, 3]^2 + 0.25 * sin(W[, 5])
}

# --- Replicate loop --------------------------------------------------------
set.seed(SEED)
# Build an independent stream per rep for reproducibility.
rep_seeds <- sample.int(.Machine$integer.max, REPS)


for (r in seq_len(REPS)) {
    set.seed(rep_seeds[r])
    # Draw confounders W (correlated to make the problem realistic).
    Sigma_W <- diag(DW)
    for (i in 1:DW) for (j in 1:DW) if (i != j) Sigma_W[i, j] <- 0.3^abs(i-j)
    W <- MASS::mvrnorm(N, mu = rep(0, DW), Sigma = Sigma_W)

    # SNP-like genotype matrix.  Column CAUSAL is the test SNP we are inferring.
    # We engineer it as g_causal = m(W) + sigma_g * v  (confounded with W).
    # The remaining (M-1) SNPs are independent N(0,1) draws -- they are
    # bystanders, used to test that the harness handles a multi-variant scan.
    v <- rnorm(N)
    g_causal <- m_W(W) + SIG_G * v
    # Centre + scale to mean 0, sd ~1 to match a standardised dosage column.
    g_causal <- (g_causal - mean(g_causal)) / sd(g_causal)

    G <- matrix(rnorm(N * M), N, M)
    G[, CAUSAL] <- g_causal

    # Outcome.
    eps <- rnorm(N)
    Y <- THETA0 * g_causal + l_W(W) + SIG_Y * eps

    # Identity kinship (OCFLMM still requires a kinship for its REML step;
    # for this DGP we use I so the LMM portion reduces to the OLS limit and
    # any residual divergence reflects DML mechanics, not REML noise).
    K <- diag(N)

    saveRDS(list(Y = Y, G = G, W = W, K = K, rep_seed = rep_seeds[r]),
            file = file.path(OUT, sprintf("rep_%03d.rds", r)))

    # Mirror as .npz-style: one .csv per matrix in a per-rep folder.
    rep_dir <- file.path(OUT, sprintf("rep_%03d", r))
    dir.create(rep_dir, showWarnings = FALSE)
    data.table::fwrite(data.table::data.table(Y = Y), file.path(rep_dir, "Y.csv"))
    data.table::fwrite(as.data.table(G), file.path(rep_dir, "G.csv"))
    data.table::fwrite(as.data.table(W), file.path(rep_dir, "W.csv"))
    # K is identity; we just record its size so Python can rebuild without IO.
    data.table::fwrite(data.table::data.table(n = N), file.path(rep_dir, "K_id.csv"))

    if (r %% 10 == 1) cat(sprintf("  rep %03d/%03d generated\n", r, REPS))
}


# --- Persist meta ----------------------------------------------------------
meta <- list(
    n = N, m = M, reps = REPS, dW = DW,
    theta0 = THETA0, causal_snp = CAUSAL,
    sigma_y = SIG_Y, sigma_g = SIG_G,
    seed = SEED, rep_seeds = rep_seeds,
    dgp = list(
        m_W_formula = "1 + W1 + 0.25 W2^2 + 0.5 W3 W4",
        l_W_formula = "-0.5 + 0.5 W1 + 0.25 W2 + 0.5 W3^2 + 0.25 sin(W5)",
        notes = "partially-linear, Chernozhukov et al. 2018 eq. 1.1"
    ),
    R_version = R.version.string,
    package_versions = list(
        data.table = as.character(packageVersion("data.table")),
        jsonlite   = as.character(packageVersion("jsonlite")),
        MASS       = as.character(packageVersion("MASS"))
    ),
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
)
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
           file.path(OUT, "fixture_meta.json"))
cat(sprintf("[generate] wrote %d reps + fixture_meta.json under %s\n", REPS, OUT))
