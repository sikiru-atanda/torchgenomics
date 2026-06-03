#!/usr/bin/env Rscript
# FUSION measured-expression reference run.
#
# Mirrors what FUSION's measured-expression mode does in the
# gusevlab/fusion_twas FAQ ("running TWAS with measured expression"):
# for each gene column g, regress phenotype y on the design matrix
# [intercept | covariates | expression_g] and emit the Wald statistic
# for the expression coefficient. This is the same OLS Wald that
# torchgenomics.postgwas.twas_observed_expression reuses from
# torchgenomics.models.glm.GLM — agreement should be at FP precision.
#
# We do NOT shell out to FUSION.assoc_test.R for this fixture because
# that path is built around imputed GReX from cis-eQTL weight files;
# the measured-expression mode (no weights) is a thin wrapper around
# the same OLS Wald implemented directly here. This script is the
# operational reference for that mode and is the file the FUSION
# README points at when the user has measured expression.

argv <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag) {
    i <- which(argv == flag)
    if (length(i) == 1 && i + 1 <= length(argv)) return(argv[i + 1])
    NULL
}
opt <- list(
    expression = parse_arg("--expression"),
    phenotype  = parse_arg("--phenotype"),
    covariates = parse_arg("--covariates"),
    out        = parse_arg("--out")
)
stopifnot(!is.null(opt$expression), !is.null(opt$phenotype), !is.null(opt$out))

expr <- read.table(opt$expression, header=TRUE, sep="\t",
                   check.names=FALSE, stringsAsFactors=FALSE)
sample_col <- colnames(expr)[1]
gene_cols <- colnames(expr)[-1]
sample_ids <- as.character(expr[[sample_col]])
expr_mat <- as.matrix(expr[, gene_cols, drop=FALSE])

pheno <- read.table(opt$phenotype, header=TRUE, sep="\t",
                    stringsAsFactors=FALSE)
stopifnot(c("IID", "TRAIT") %in% colnames(pheno))
rownames(pheno) <- as.character(pheno$IID)
y <- pheno[sample_ids, "TRAIT"]
stopifnot(!any(is.na(y)))

if (!is.null(opt$covariates)) {
    cov_df <- read.table(opt$covariates, header=TRUE, sep="\t",
                         stringsAsFactors=FALSE)
    stopifnot("IID" %in% colnames(cov_df))
    rownames(cov_df) <- as.character(cov_df$IID)
    cov_mat <- as.matrix(cov_df[sample_ids,
                                setdiff(colnames(cov_df),
                                        c("FID", "IID")), drop=FALSE])
    storage.mode(cov_mat) <- "numeric"
} else {
    cov_mat <- matrix(0, nrow=length(sample_ids), ncol=0)
}

# Build the OLS design once per gene.
intercept <- rep(1.0, length(sample_ids))
res <- data.frame(
    gene_id   = gene_cols,
    beta      = numeric(length(gene_cols)),
    se        = numeric(length(gene_cols)),
    z_twas    = numeric(length(gene_cols)),
    p_twas    = numeric(length(gene_cols)),
    n_samples = integer(length(gene_cols)),
    stringsAsFactors = FALSE
)

for (j in seq_along(gene_cols)) {
    X <- cbind(intercept, cov_mat, expression = expr_mat[, j])
    fit <- lm.fit(x = X, y = y)
    n <- length(y)
    p <- ncol(X)
    resid <- fit$residuals
    sigma2 <- sum(resid^2) / (n - p)
    XtX_inv <- solve(crossprod(X))
    coef_idx <- ncol(X)  # 'expression' is the last column
    beta_hat <- fit$coefficients[coef_idx]
    var_beta <- sigma2 * XtX_inv[coef_idx, coef_idx]
    se_beta  <- sqrt(var_beta)
    z_val    <- beta_hat / se_beta
    # Two-sided F(1, n-p): p = pf(z^2, 1, n-p, lower.tail = FALSE)
    p_val    <- pf(z_val^2, 1, n - p, lower.tail = FALSE)
    res$beta[j]      <- beta_hat
    res$se[j]        <- se_beta
    res$z_twas[j]    <- z_val
    res$p_twas[j]    <- p_val
    res$n_samples[j] <- n
}

write.table(res, file=opt$out, sep="\t", quote=FALSE, row.names=FALSE)
cat(sprintf("[fusion reference] wrote %s with %d genes\n",
            opt$out, nrow(res)))
