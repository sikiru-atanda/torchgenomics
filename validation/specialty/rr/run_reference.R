#!/usr/bin/env Rscript
# run_reference.R -- lme4 random-regression reference fit.
#
# Model fit on the same fixture run_torchgenomics.py consumes:
#
#   y_{i,t} = mu + beta1 P1(t) + beta2 P2(t)
#           + ( u0_i + u1_i P1(t) + u2_i P2(t) ) + e_{i,t}
#
# where P_k are normalized Legendre polynomials in
# t_std = 2 (t - t_min) / (t_max - t_min) - 1.  This matches
# torchgenomics.linalg.basis.legendre_basis exactly.  Random effects are
# correlated across the b=3 basis coefficients within subject (the
# (0 + P0 + P1 + P2 | sid) term).  Residuals are iid Normal.
#
# For the planted causal SNP we fit a marker-augmented model:
#   y ~ 1 + P1 + P2 + g + g:P1 + g:P2 + (0 + P0 + P1 + P2 | sid)
# This is the lme4 analogue of TGs full b-vector SNP effect: the per-SNP
# beta vector (b_g, b_gP1, b_gP2) is interpreted as the b=3 basis-
# coefficient SNP effect.  Per-time-point reconstruction is
#   beta_g(t) = b_g P0(t) + b_gP1 P1(t) + b_gP2 P2(t),
# which agrees term-for-term with TGs beta_at_t output under the same
# Phi evaluation.
#
# Outputs (validation/specialty/rr/outputs/):
#   reference_null.tsv   -- null-model variance components + logLik.
#   reference_causal.tsv -- causal SNP fixed effects + per-time beta(t).

suppressWarnings(suppressMessages({
    library(lme4)
    library(lmerTest)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("usage: run_reference.R <data_dir> <out_dir>")
}
DATA_DIR <- args[1]
OUT_DIR  <- args[2]
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

pheno <- read.table(file.path(DATA_DIR, "long_pheno.tsv"),
                    header = TRUE, sep = "\t")
geno  <- read.table(file.path(DATA_DIR, "geno.tsv"),
                    header = TRUE, sep = "\t", check.names = FALSE)
truth <- jsonlite::fromJSON(file.path(DATA_DIR, "truth.json"))

t_min <- truth$t_min; t_max <- truth$t_max
causal_name <- truth$causal_snp_name

legendre_norm <- function(t_std, order) {
    Tn <- length(t_std); b <- order + 1
    P <- matrix(0, Tn, b)
    P[, 1] <- 1.0
    if (order >= 1) P[, 2] <- t_std
    if (order >= 2) {
        for (k in 1:(order - 1)) {
            P[, k + 2] <- ((2 * k + 1) * t_std * P[, k + 1] -
                          k * P[, k]) / (k + 1)
        }
    }
    norms <- sqrt((2 * (0:order) + 1) / 2)
    sweep(P, 2, norms, `*`)
}

t_std_full <- 2 * (pheno$t - t_min) / (t_max - t_min) - 1
Phi_full <- legendre_norm(t_std_full, order = truth$basis_order)
pheno$P0 <- Phi_full[, 1]
pheno$P1 <- Phi_full[, 2]
pheno$P2 <- Phi_full[, 3]

pheno$g <- geno[match(pheno$sid, geno$sid), causal_name]

# ---- null model: no SNP -------------------------------------------
cat("[reference] fitting lme4 null model...\n")
null_fit <- lmer(
    y ~ 1 + P1 + P2 + (0 + P0 + P1 + P2 | sid),
    data = pheno,
    REML = TRUE,
    control = lmerControl(optimizer = "bobyqa",
                          optCtrl = list(maxfun = 50000)),
)

sigma2_e_ref <- attr(VarCorr(null_fit), "sc")^2
K_coef_ref <- as.matrix(VarCorr(null_fit)$sid)
# Reorder rows/cols if lme4 returns them in a different order than P0,P1,P2.
want_order <- c("P0", "P1", "P2")
K_coef_ref <- K_coef_ref[want_order, want_order]
ll_null <- as.numeric(logLik(null_fit))

null_rows <- c()
for (i in 1:3) {
    for (j in 1:3) {
        null_rows <- c(null_rows,
            sprintf("K_coef[%d,%d]\t%.10g",
                    i - 1, j - 1, K_coef_ref[i, j]))
    }
}
# Variance ratios across the basis -- the brief gates on these
# (1e-2 absolute).
diag_K <- diag(K_coef_ref)
total_basis_var <- sum(diag_K)
for (i in 1:3) {
    null_rows <- c(null_rows,
        sprintf("var_ratio_basis[%d]\t%.10g",
                i - 1, diag_K[i] / total_basis_var))
}
null_rows <- c(null_rows,
    sprintf("sigma2_e\t%.10g", sigma2_e_ref),
    sprintf("logLik_REML\t%.10g", ll_null))
writeLines(c("name\tvalue", null_rows),
           file.path(OUT_DIR, "reference_null.tsv"))
cat("[reference] wrote reference_null.tsv\n")

# ---- causal SNP model: fit g + g:P1 + g:P2 ------------------------
cat("[reference] fitting causal SNP model (g + g:P1 + g:P2)...\n")
causal_fit <- lmer(
    y ~ 1 + P1 + P2 + g + g:P1 + g:P2 + (0 + P0 + P1 + P2 | sid),
    data = pheno,
    REML = TRUE,
    control = lmerControl(optimizer = "bobyqa",
                          optCtrl = list(maxfun = 50000)),
)
fe <- summary(causal_fit, ddf = "Satterthwaite")$coefficients
# lme4 may swap interaction label order; detect at runtime.
label_gp1 <- if ("g:P1" %in% rownames(fe)) "g:P1" else "P1:g"
label_gp2 <- if ("g:P2" %in% rownames(fe)) "g:P2" else "P2:g"

row_g    <- fe["g",       ]
row_g_p1 <- fe[label_gp1, ]
row_g_p2 <- fe[label_gp2, ]

b_g    <- row_g["Estimate"]
b_g_p1 <- row_g_p1["Estimate"]
b_g_p2 <- row_g_p2["Estimate"]

# Full 3x3 vcov sub-block for (g, gP1, gP2)
V <- as.matrix(vcov(causal_fit))
idx_sel <- c("g", label_gp1, label_gp2)
V_sub <- V[idx_sel, idx_sel]
b_vec <- c(b_g, b_g_p1, b_g_p2)

# Per-time-point reconstruction beta(t) = sum_k Phi_grid[t, k+1] * b_vec[k+1]
unique_times <- sort(unique(pheno$t))
t_std_grid <- 2 * (unique_times - t_min) / (t_max - t_min) - 1
Phi_grid <- legendre_norm(t_std_grid, order = truth$basis_order)
beta_t <- as.numeric(Phi_grid %*% b_vec)
# Var(beta(t)) = Phi_grid[t,] V_sub Phi_grid[t,]^T
var_beta_t <- as.numeric(rowSums((Phi_grid %*% V_sub) * Phi_grid))
se_beta_t <- sqrt(pmax(var_beta_t, 0))

# Joint chi2(3) Wald test: b_vec V_sub^{-1} b_vec
chi2_joint <- as.numeric(t(b_vec) %*% solve(V_sub, b_vec))
p_joint    <- pchisq(chi2_joint, df = 3, lower.tail = FALSE)

causal_rows <- c(
    sprintf("beta_g\t%.10g",  b_g),
    sprintf("se_g\t%.10g",    row_g["Std. Error"]),
    sprintf("p_g\t%.10g",     row_g["Pr(>|t|)"]),
    sprintf("beta_g_P1\t%.10g", b_g_p1),
    sprintf("se_g_P1\t%.10g",   row_g_p1["Std. Error"]),
    sprintf("p_g_P1\t%.10g",    row_g_p1["Pr(>|t|)"]),
    sprintf("beta_g_P2\t%.10g", b_g_p2),
    sprintf("se_g_P2\t%.10g",   row_g_p2["Std. Error"]),
    sprintf("p_g_P2\t%.10g",    row_g_p2["Pr(>|t|)"]),
    sprintf("chi2_joint_b\t%.10g", chi2_joint),
    sprintf("p_joint_b\t%.10g",    p_joint)
)
for (k in seq_along(unique_times)) {
    causal_rows <- c(causal_rows,
        sprintf("beta_at_t_%g\t%.10g", unique_times[k], beta_t[k]),
        sprintf("se_at_t_%g\t%.10g",   unique_times[k], se_beta_t[k]))
}
causal_rows <- c(causal_rows,
    sprintf("causal_snp\t%s", causal_name),
    sprintf("logLik_REML_causal\t%.10g", as.numeric(logLik(causal_fit))))

writeLines(c("name\tvalue", causal_rows),
           file.path(OUT_DIR, "reference_causal.tsv"))
cat("[reference] wrote reference_causal.tsv\n")
cat("[reference] lme4 reference fit complete.\n")
