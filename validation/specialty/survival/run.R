# Survival GWAS reference: per-SNP Cox PH frailty fits via coxme.
#
# For each test SNP s_j, fit:
#   Surv(time, event) ~ g + (1 | id_factor),  varlist = coxmeFull(K)
# where (1 | id_factor) attaches a Gaussian random-effect frailty whose
# covariance is shaped by the GRM K.  coxme uses Therneau-Grambsch-Pankratz
# (2003) Laplace-approximated REML for the random-effect variance; the
# fixed-effect (SNP) beta is read from fixef() and its SE from vcov().
#
# References
# ----------
# - Therneau & Grambsch (2000). Modeling Survival Data. Springer.
# - Therneau, Grambsch & Pankratz (2003). Penalized survival models and
#   frailty. J Comp Graph Stat 12:156-175.
# - Therneau (2024). The coxme package, R package vignette.
#
# Usage:
#   Rscript run.R <pheno_csv> <geno_csv> <kinship_csv> <out_csv>
#
# Output CSV columns: snp, beta, se, z, p_wald

suppressPackageStartupMessages({
    library(coxme)
    library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
pheno_file <- args[1]
geno_file  <- args[2]
kin_file   <- args[3]
out_file   <- args[4]

pheno <- read.csv(pheno_file)
geno  <- read.csv(geno_file)
K     <- as.matrix(read.csv(kin_file, header = FALSE))

n <- nrow(pheno)
stopifnot(nrow(K) == n, ncol(K) == n)
rownames(K) <- colnames(K) <- as.character(pheno$id)

# Merge phenotype with genotype on integer id; ensure stable row order.
dat <- merge(pheno, geno, by = "id")
dat <- dat[order(dat$id), ]
dat$id_factor <- factor(as.character(dat$id), levels = as.character(pheno$id))

snp_cols <- grep("^snp", names(dat), value = TRUE)
m <- length(snp_cols)

results <- data.frame(
    snp = character(m),
    beta = numeric(m),
    se = numeric(m),
    z = numeric(m),
    p_wald = numeric(m),
    converged = logical(m),
    stringsAsFactors = FALSE
)

# coxme warns about singular fits frequently in small-n regimes; we keep
# them as informational messages, not errors, since the brief's tolerance
# gates are observed-then-floored on the per-SNP beta.
options(warn = 1)

wall_start <- Sys.time()
for (j in seq_along(snp_cols)) {
    snp_name <- snp_cols[j]
    dat$g <- dat[[snp_name]]

    if (sd(dat$g) < 1e-8) {
        results[j, ] <- list(snp_name, 0, Inf, 0, 1, FALSE)
        next
    }

    fit_ok <- TRUE
    coef_val <- NA_real_
    se_val   <- NA_real_
    z_val    <- NA_real_
    p_val    <- NA_real_
    tryCatch({
        fit <- coxme(
            Surv(time, event) ~ g + (1 | id_factor),
            data = dat,
            varlist = coxmeFull(K)
        )
        coef_val <- as.numeric(fixef(fit)["g"])
        vcov_mat <- as.matrix(vcov(fit))
        se_val <- sqrt(vcov_mat["g", "g"])
        z_val <- coef_val / se_val
        p_val <- 2 * pnorm(-abs(z_val))
    }, error = function(e) {
        message(sprintf("[coxme] %s: %s", snp_name, conditionMessage(e)))
        fit_ok <<- FALSE
    })

    results[j, ] <- list(snp_name, coef_val, se_val, z_val, p_val, fit_ok)
}
wall_secs <- as.numeric(difftime(Sys.time(), wall_start, units = "secs"))

write.csv(results, out_file, row.names = FALSE)
cat(sprintf("[coxme] wrote %s (%d SNPs, %.1f s)\n", out_file, m, wall_secs))
