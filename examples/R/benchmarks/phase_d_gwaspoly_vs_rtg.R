## =============================================================================
## phase_d_gwaspoly_vs_rtg.R -- rTorchGWAS::gwas_lmm vs GWASpoly on tetraploid
## =============================================================================
##
## Replicates the multi-environment data prep used by tests/test_golden_gwaspoly.py
## but exercises the rTorchGWAS *R* wrapper end-to-end:
##
##   * 957 unique potato genotypes (tetraploid, 9888 markers)
##   * 1249 phenotype rows (vine.maturity across 6 envs: Hancock15-20)
##   * Z incidence matrix maps phenotype rows -> unique genotypes
##   * G_obs   = Z %*% G_geno          (n_obs x m)
##   * K_obs   = Z %*% K_geno %*% t(Z) (n_obs x n_obs)
##   * X0     ~ intercept + (E - 1) env dummies as fixed effects
##
## We compare rTorchGWAS gwas_lmm against GWASpoly's `additive` reference output.
## The R wrapper currently exposes only the additive gene action (gwas_polyploid
## ignores its `gene_action` arg and recode_gene_action is not exported), so
## 1-dom / 2-dom / 3-dom equivalence is left to the existing Python golden test
## (tests/test_golden_gwaspoly.py).
##
## Pass criterion: cor(-log10 p) > 0.99 vs gwaspoly_additive.csv on the inner
## join.  GWASpoly uses Q+K with the same env-fixed-effect setup, so the only
## sources of disagreement are GWASpoly's per-marker REML refit (vs P3D) and
## numerical implementation differences.

`%||%` <- function(a, b) if (!is.null(a)) a else b

configure_python_for_torchgwas <- function() {
  if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) return(invisible())
  candidates <- c(
    "C:/Program Files/Python311/python.exe",
    "C:/Program Files/Python312/python.exe",
    "C:/Program Files/Python310/python.exe",
    Sys.which("python"), Sys.which("python3"))
  candidates <- unique(candidates[nzchar(candidates) & file.exists(candidates)])
  for (py in candidates) {
    chk <- suppressWarnings(system2(py,
      args = c("-c", shQuote("import torchgwas; print(torchgwas.__version__)")),
      stdout = TRUE, stderr = TRUE))
    if (length(chk) >= 1 && !grepl("Error|Traceback", chk[1])) {
      Sys.setenv(RETICULATE_PYTHON = py); return(invisible())
    }
  }
  stop("torchgwas not importable", call. = FALSE)
}
configure_python_for_torchgwas()
suppressPackageStartupMessages(library(rTorchGWAS))

results_dir <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/gwaspoly_results"
out_dir     <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

PLOIDY <- 4L

## ---- Load data --------------------------------------------------------------
pheno <- read.csv(file.path(results_dir, "potato_pheno_with_genoid.csv"),
                  stringsAsFactors = FALSE)
geno  <- read.csv(file.path(results_dir, "potato_geno_aligned.csv"),
                  row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
mmap  <- read.csv(file.path(results_dir, "potato_map.csv"),
                  stringsAsFactors = FALSE)

rownames(geno) <- as.character(rownames(geno))
pheno$geno_id  <- as.character(pheno$geno_id)
pheno <- pheno[!is.na(pheno$vine.maturity), , drop = FALSE]
pheno <- pheno[pheno$geno_id %in% rownames(geno), , drop = FALSE]
pheno <- pheno[order(pheno$env, pheno$geno_id), , drop = FALSE]
rownames(pheno) <- NULL

unique_geno <- sort(unique(pheno$geno_id))
n_obs  <- nrow(pheno)
n_geno <- length(unique_geno)
cat(sprintf("Loaded: %d phenotype rows across %d unique genotypes (%d markers)\n",
            n_obs, n_geno, ncol(geno)))

geno_sub <- as.matrix(geno[unique_geno, , drop = FALSE])
storage.mode(geno_sub) <- "double"

## Mean-impute residual NA dosages on the n_geno x m matrix.
G_geno <- impute_genotypes(geno_sub, method = "mean")
rownames(G_geno) <- unique_geno
colnames(G_geno) <- colnames(geno_sub)
stopifnot(!any(is.na(G_geno)))

## Z incidence: n_obs x n_geno
geno_idx <- setNames(seq_len(n_geno), unique_geno)
Z <- matrix(0, nrow = n_obs, ncol = n_geno)
for (i in seq_len(n_obs)) Z[i, geno_idx[pheno$geno_id[i]]] <- 1

## Expand to phenotype scale
G_obs <- Z %*% G_geno  # n_obs x m
rownames(G_obs) <- NULL
colnames(G_obs) <- colnames(G_geno)
cat(sprintf("G_obs : %d x %d (Z %%*%% G_geno)\n", nrow(G_obs), ncol(G_obs)))

## ---- Polyploid VanRaden GRM at the genotype scale, then expand --------------
K_geno <- compute_grm(G_geno, method = "vanraden", ploidy = PLOIDY)
stopifnot(nrow(K_geno) == n_geno)
K_obs <- Z %*% K_geno %*% t(Z)
cat(sprintf("K_obs : %d x %d (Z %%*%% K_geno %%*%% Z')\n", nrow(K_obs), ncol(K_obs)))

## ---- Fixed effects: env factor (drop_first to avoid collinearity) -----------
env_dummies <- model.matrix(~ env, data = pheno)[, -1, drop = FALSE]  # drops intercept col
cat(sprintf("env dummies: %d x %d\n", nrow(env_dummies), ncol(env_dummies)))

y <- pheno$vine.maturity
stopifnot(!any(is.na(y)))

## ---- Run rTorchGWAS gwas_lmm ------------------------------------------------
cat("\nRunning rTorchGWAS::gwas_lmm (additive, K=Z K_geno Z') ...\n")
t0 <- proc.time()
res <- gwas_lmm(G_obs, y, covar = env_dummies, K = K_obs,
                test = "wald", n_pcs = 0L)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done: %.1fs, %d SNPs scanned\n", elapsed, length(res$pvalues)))

R_df <- data.frame(snp = colnames(G_obs),
                   p_R = as.numeric(res$pvalues),
                   stringsAsFactors = FALSE)

## ---- Load GWASpoly additive reference ---------------------------------------
gp <- read.csv(file.path(results_dir, "gwaspoly_additive.csv"),
               stringsAsFactors = FALSE)
gp_df <- data.frame(snp     = as.character(gp$marker),
                    p_gwaspoly = as.numeric(gp$pvalue),
                    stringsAsFactors = FALSE)
cat(sprintf("GWASpoly additive: %d markers\n", nrow(gp_df)))

## ---- Compare ----------------------------------------------------------------
m <- merge(R_df, gp_df, by = "snp")
m <- m[is.finite(m$p_R) & is.finite(m$p_gwaspoly) &
       m$p_R > 0 & m$p_gwaspoly > 0, , drop = FALSE]
cat(sprintf("Inner join (positive finite p): %d markers\n", nrow(m)))

logp_R  <- -log10(m$p_R)
logp_gp <- -log10(m$p_gwaspoly)
pearson  <- cor(logp_R, logp_gp)
spearman <- cor(logp_R, logp_gp, method = "spearman")
med_dlogp <- median(abs(logp_R - logp_gp))
max_dlogp <- max(abs(logp_R - logp_gp))

top_k   <- 20
top_R   <- m$snp[order(m$p_R)[seq_len(top_k)]]
top_GP  <- m$snp[order(m$p_gwaspoly)[seq_len(top_k)]]
jaccard <- length(intersect(top_R, top_GP)) / length(union(top_R, top_GP))

cat("\n=== Phase D: rTorchGWAS::gwas_lmm vs GWASpoly additive ===\n")
cat(sprintf("  n_markers_compared    : %d\n", nrow(m)))
cat(sprintf("  cor(-log10 p)         : %.6f\n", pearson))
cat(sprintf("  spearman              : %.6f\n", spearman))
cat(sprintf("  median |d -log10 p|   : %.3e\n", med_dlogp))
cat(sprintf("  max    |d -log10 p|   : %.3e\n", max_dlogp))
cat(sprintf("  top%d Jaccard          : %.3f\n", top_k, jaccard))
cat(sprintf("  rTG min p             : %.3g\n", min(m$p_R)))
cat(sprintf("  GWASpoly min p        : %.3g\n", min(m$p_gwaspoly)))

verdict <- if (pearson > 0.99 && jaccard > 0.50) "PASS" else "REVIEW"
cat(sprintf("\n>>> VERDICT: %s\n", verdict))

summary_row <- data.frame(
  phase                = "D",
  reference_tool       = "GWASpoly additive (Q+K, env fixed)",
  rtorchgwas_verb      = "gwas_lmm (additive, K=Z K_geno Z')",
  dataset              = "Potato vine.maturity tetraploid",
  ploidy               = PLOIDY,
  n_obs                = n_obs,
  n_unique_geno        = n_geno,
  n_markers_compared   = nrow(m),
  cor_neg_log10_p      = pearson,
  spearman             = spearman,
  median_abs_d_log10p  = med_dlogp,
  max_abs_d_log10p     = max_dlogp,
  top_k                = top_k,
  topk_jaccard         = jaccard,
  rtorchgwas_elapsed_s = unname(elapsed),
  verdict              = verdict,
  stringsAsFactors     = FALSE
)
write.csv(m, file.path(out_dir, "phase_d_gwaspoly_per_snp.csv"), row.names = FALSE)
write.csv(summary_row, file.path(out_dir, "phase_d_gwaspoly_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_d_gwaspoly_summary.csv")))
