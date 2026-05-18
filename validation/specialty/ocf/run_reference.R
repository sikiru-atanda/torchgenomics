#!/usr/bin/env Rscript
# OCF reference: hand-coded DML for partially-linear regression
#
#     y_i = theta * g_i + l(W_i) + eps_i
#     g_i = m(W_i) + v_i
#
# implementing Chernozhukov et al. (2018, Econometrica) Algorithm 2 and the
# Neyman-orthogonal score eq. (3.1):
#
#     psi(W, theta, eta) = (Y - l(W) - theta (G - m(W))) (G - m(W))
#
# with K-fold cross-fitting (DML2, eq. 3.3): per fold k,
#   * train l_hat^(-k), m_hat^(-k) on the OTHER K-1 folds,
#   * compute residuals on fold k: y_tilde_k = Y_k - l_hat^(-k)(W_k);
#                                  g_tilde_k = G_k - m_hat^(-k)(W_k).
# Stack across folds and solve the moment equation:
#   theta_hat = sum_i (g_tilde_i * y_tilde_i) / sum_i (g_tilde_i^2)
#
# Variance is the sandwich form (eq. 3.10):
#   var(theta_hat) = (1/n) sum_i (psi_i^2) / ( (1/n) sum_i (g_tilde_i^2) )^2
#
# Nuisance learner: quadratic-feature ridge.  This is intentionally a
# specific, transparent, paper-faithful choice -- not a black-box ML model
# -- so that re-runs are deterministic and any divergence with TorchGWAS
# OCFLMM is attributable to algorithmic differences, not learner stochasticity.

suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
})

# --- CLI args --------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA   <- parse_arg("--data-dir", "data")
OUT    <- parse_arg("--out-dir",  "outputs")
KFOLD  <- as.integer(parse_arg("--k", "5"))
LAMBDA <- as.numeric(parse_arg("--ridge", "1e-2"))
SEED   <- as.integer(parse_arg("--seed", "42"))

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

meta <- jsonlite::fromJSON(file.path(DATA, "fixture_meta.json"))
cat(sprintf("[reference] data=%s reps=%d n=%d m=%d theta0=%.4f K=%d ridge=%.4g\n",
            DATA, meta$reps, meta$n, meta$m, meta$theta0, KFOLD, LAMBDA))

# --- Quadratic-feature ridge learner ---------------------------------------
# Features: intercept + linear W + squared W + all pairwise products of W.
# Closed-form ridge: beta_hat = (X^T X + lambda I)^{-1} X^T y

build_features <- function(W) {
    n <- nrow(W); d <- ncol(W)
    # linear
    Phi <- cbind(1, W)
    # squared
    Phi <- cbind(Phi, W^2)
    # pairwise products W_i * W_j (i<j)
    if (d >= 2) {
        for (i in 1:(d-1)) for (j in (i+1):d) {
            Phi <- cbind(Phi, W[, i] * W[, j])
        }
    }
    Phi
}

ridge_fit <- function(X, y, lambda) {
    p <- ncol(X)
    A <- crossprod(X) + lambda * diag(p)
    b <- crossprod(X, y)
    drop(solve(A, b))
}

ridge_predict <- function(X, beta) drop(X %*% beta)

# --- DML2 estimator (Chernozhukov et al. 2018 Algorithm 2) ----------------
# Note: this is the DML estimator for partially-linear regression with a
# single target g. We adapt it to the GWAS case by always taking g_i = G[, causal_snp].
#
# In the GWAS setting the bystander SNPs are not part of the DML moment;
# they are present in the fixture only to mirror a multi-variant scan and
# verify the harness handles the matrix shape end-to-end. Reference inference
# is for the causal SNP only.

dml_pl_estimate <- function(Y, g, W, k_folds, lambda, fold_seed) {
    n <- length(Y)
    set.seed(fold_seed)
    folds <- sample(rep(seq_len(k_folds), length.out = n))

    y_tilde <- numeric(n)
    g_tilde <- numeric(n)

    Phi <- build_features(W)

    for (k in seq_len(k_folds)) {
        te <- which(folds == k)
        tr <- which(folds != k)
        # Train l on W_tr -> Y_tr, m on W_tr -> g_tr.
        bl <- ridge_fit(Phi[tr, , drop = FALSE], Y[tr], lambda)
        bm <- ridge_fit(Phi[tr, , drop = FALSE], g[tr], lambda)
        # Predict on the held-out fold.
        y_tilde[te] <- Y[te] - ridge_predict(Phi[te, , drop = FALSE], bl)
        g_tilde[te] <- g[te] - ridge_predict(Phi[te, , drop = FALSE], bm)
    }

    # DML2 closed-form (eq. 3.3 with Neyman score eq. 3.1).
    gg <- sum(g_tilde^2)
    gy <- sum(g_tilde * y_tilde)
    theta_hat <- gy / gg

    # Sandwich variance (eq. 3.10): n^{-1} sum psi^2 / (n^{-1} sum g_tilde^2)^2
    psi <- (y_tilde - theta_hat * g_tilde) * g_tilde
    var_theta <- (sum(psi^2) / n) / ((gg / n)^2) / n  # divides by n to get var of mean-score estimator

    se   <- sqrt(var_theta)
    zval <- theta_hat / se
    pval <- 2 * pnorm(-abs(zval))
    ci_lo <- theta_hat - 1.96 * se
    ci_hi <- theta_hat + 1.96 * se

    list(theta_hat = theta_hat, se = se, z = zval, p = pval,
         ci_lo = ci_lo, ci_hi = ci_hi)
}

# --- Per-rep loop ----------------------------------------------------------
results <- data.table::data.table(
    rep      = integer(0),
    theta0   = numeric(0),
    theta_hat= numeric(0),
    se       = numeric(0),
    z        = numeric(0),
    p        = numeric(0),
    ci_lo    = numeric(0),
    ci_hi    = numeric(0),
    covered  = logical(0)
)

fold_seeds <- {
    set.seed(SEED + 1)
    sample.int(.Machine$integer.max, meta$reps)
}

for (r in seq_len(meta$reps)) {
    rd <- readRDS(file.path(DATA, sprintf("rep_%03d.rds", r)))
    g_idx <- meta$causal_snp
    res <- dml_pl_estimate(
        Y = rd$Y, g = rd$G[, g_idx], W = rd$W,
        k_folds = KFOLD, lambda = LAMBDA, fold_seed = fold_seeds[r]
    )
    covered <- (res$ci_lo <= meta$theta0) && (meta$theta0 <= res$ci_hi)
    results <- rbind(results, data.table::data.table(
        rep       = r,
        theta0    = meta$theta0,
        theta_hat = res$theta_hat,
        se        = res$se,
        z         = res$z,
        p         = res$p,
        ci_lo     = res$ci_lo,
        ci_hi     = res$ci_hi,
        covered   = covered
    ))
    if (r %% 10 == 1)
        cat(sprintf("  rep %03d/%03d: theta_hat=%.4f se=%.4f covered=%s\n",
                    r, meta$reps, res$theta_hat, res$se, covered))
}

# --- Aggregate + persist ---------------------------------------------------
emp_cov <- mean(results$covered)
mean_bias <- mean(results$theta_hat) - meta$theta0
data.table::fwrite(results, file.path(OUT, "reference_results.tsv"), sep = "\t")

summary_json <- list(
    n_reps         = nrow(results),
    theta0         = meta$theta0,
    mean_theta_hat = mean(results$theta_hat),
    mean_bias      = mean_bias,
    mean_se        = mean(results$se),
    sd_theta_hat   = sd(results$theta_hat),
    empirical_coverage_95 = emp_cov,
    median_p       = median(results$p),
    k_folds        = KFOLD,
    ridge_lambda   = LAMBDA,
    seed           = SEED,
    reference_kind = "hand-coded DML2 (Chernozhukov 2018 Algorithm 2 / eq. 3.1, 3.3, 3.10)",
    nuisance_learner = "quadratic-feature ridge",
    R_version = R.version.string,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
)
writeLines(jsonlite::toJSON(summary_json, auto_unbox = TRUE, pretty = TRUE),
           file.path(OUT, "reference_summary.json"))

cat(sprintf("[reference] DONE.  empirical_coverage=%.3f mean_theta_hat=%.4f mean_bias=%.4f mean_se=%.4f\n",
            emp_cov, mean(results$theta_hat), mean_bias, mean(results$se)))
