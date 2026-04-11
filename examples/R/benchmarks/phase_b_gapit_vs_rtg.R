## =============================================================================
## phase_b_gapit_vs_rtg.R -- rTorchGWAS GLM/LMM/FarmCPU/BLINK vs GAPIT3 on MDP
## =============================================================================
##
## GAPIT3 was already run on the full MDP EarHT trait (n_obs = 279) and the
## per-model GWAS tables live in benchmark/gapit_results/.  We replicate the
## same sample subset on the rTorchGWAS side and compare per-SNP -log10(p)
## via inner-joined Pearson + Spearman correlation.
##
## Pass criterion: cor(-log10 p) > 0.95 for each model class.  Bitwise
## agreement is not expected because GAPIT and rTorchGWAS make different
## decisions about PC inclusion, REML convergence, and per-SNP missing
## handling.

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
  stop("torchgwas not importable from any candidate Python", call. = FALSE)
}
configure_python_for_torchgwas()
suppressPackageStartupMessages(library(rTorchGWAS))

data_dir   <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/data"
gapit_dir  <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/gapit_results"
out_dir    <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## ---- Load MDP source --------------------------------------------------------
geno_df  <- read.table(file.path(data_dir, "mdp_numeric.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
pheno_df <- read.table(file.path(data_dir, "mdp_traits.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE,
                       na.strings = c("NA", "NaN"))
geno_df$taxa  <- as.character(geno_df$taxa)
pheno_df$Taxa <- as.character(pheno_df$Taxa)

## GAPIT used the full EarHT non-missing intersection with genotyped samples
## (n = 279 from the existing GAPIT outputs).
common <- sort(intersect(
  pheno_df$Taxa[!is.na(pheno_df$EarHT)],
  geno_df$taxa
))
n <- length(common)
cat(sprintf("Common samples (EarHT non-NA): %d\n", n))

geno_sub  <- geno_df[match(common, geno_df$taxa), , drop = FALSE]
pheno_sub <- pheno_df[match(common, pheno_df$Taxa), , drop = FALSE]
snp_names <- colnames(geno_sub)[-1]
G_raw     <- as.matrix(geno_sub[, -1, drop = FALSE])
storage.mode(G_raw) <- "double"
rownames(G_raw) <- common
colnames(G_raw) <- snp_names

G_imp <- impute_genotypes(G_raw, method = "mean")
rownames(G_imp) <- common
colnames(G_imp) <- snp_names
y <- pheno_sub$EarHT
stopifnot(!any(is.na(y)))
cat(sprintf("rTorchGWAS genotype matrix: %dx%d\n", nrow(G_imp), ncol(G_imp)))

## ---- Compute VanRaden GRM (matches GAPIT's MLM default) --------------------
K <- compute_grm(G_imp, method = "vanraden", ploidy = 2L)

## ---- Helper for one comparison ---------------------------------------------
##
## For GLM and LMM, the meaningful metric is per-SNP -log10(p) correlation —
## both tools see all SNPs and produce a comparable score per SNP.
##
## For multi-locus iterative methods (FarmCPU, BLINK), per-SNP p-values are
## conditioned on which SNPs were absorbed as covariates in prior iterations.
## Two implementations that disagree about the order of pseudo-QTN inclusion
## will report very different p-values for non-significant SNPs even when
## both find the same true causal hits.  The honest metric for those is the
## overlap of the top-K hits (Jaccard) and how close their min p-values are.
compare_one <- function(rtg_res, gapit_path, label, top_k = 20) {
  R_df <- data.frame(
    snp = snp_names,
    p_R = as.numeric(rtg_res$pvalues),
    stringsAsFactors = FALSE
  )
  G_df <- read.csv(gapit_path, stringsAsFactors = FALSE, check.names = FALSE)
  G_df$snp     <- as.character(G_df$SNP)
  G_df$p_gapit <- as.numeric(G_df$P.value)
  m <- merge(R_df, G_df[, c("snp", "p_gapit")], by = "snp")
  m <- m[is.finite(m$p_R) & is.finite(m$p_gapit) &
         m$p_R > 0 & m$p_gapit > 0, , drop = FALSE]

  log10p_R     <- -log10(m$p_R)
  log10p_gapit <- -log10(m$p_gapit)
  pearson      <- cor(log10p_R, log10p_gapit)
  spearman     <- cor(log10p_R, log10p_gapit, method = "spearman")

  ## Top-K overlap (Jaccard on the K-smallest p-values)
  k_eff   <- min(top_k, nrow(m))
  top_R   <- m$snp[order(m$p_R)[seq_len(k_eff)]]
  top_G   <- m$snp[order(m$p_gapit)[seq_len(k_eff)]]
  jaccard <- length(intersect(top_R, top_G)) /
             length(union(top_R, top_G))

  cat(sprintf("\n[%s]  n=%d  pearson=%.4f  spearman=%.4f  ",
              label, nrow(m), pearson, spearman))
  cat(sprintf("top%d_jaccard=%.3f  rTG min p=%.3g  GAPIT min p=%.3g\n",
              top_k, jaccard, min(m$p_R), min(m$p_gapit)))

  list(label = label, n = nrow(m), pearson = pearson, spearman = spearman,
       top_k = top_k, jaccard = jaccard,
       min_p_rtg = min(m$p_R), min_p_gapit = min(m$p_gapit))
}

## ---- Run all four rTorchGWAS verbs ------------------------------------------
cat("\n=== Phase B: rTorchGWAS verbs ===\n")

cat("[1/4] gwas_glm (OLS) ...\n")
res_glm <- gwas_glm(G_imp, y)

cat("[2/4] gwas_lmm (Wald, n_pcs=0) ...\n")
res_lmm <- gwas_lmm(G_imp, y, K = K, test = "wald", n_pcs = 0L)

cat("[3/4] gwas_farmcpu ...\n")
res_fc <- gwas_farmcpu(G_imp, y, max_qtns = 10L)

cat("[4/4] gwas_blink ...\n")
res_bl <- gwas_blink(G_imp, y, cutoff = 0.01)

## ---- Compare each to GAPIT3 -------------------------------------------------
results <- list(
  compare_one(res_glm, file.path(gapit_dir, "GLM_GWAS.csv"),     "GLM"),
  compare_one(res_lmm, file.path(gapit_dir, "MLM_GWAS.csv"),     "LMM/MLM"),
  compare_one(res_fc,  file.path(gapit_dir, "FarmCPU_GWAS.csv"), "FarmCPU"),
  compare_one(res_bl,  file.path(gapit_dir, "BLINK_GWAS.csv"),   "BLINK")
)

summary_df <- do.call(rbind, lapply(results, as.data.frame,
                                    stringsAsFactors = FALSE))
summary_df$phase            <- "B"
summary_df$reference_tool   <- paste0("GAPIT3 ", summary_df$label)
summary_df$rtorchgwas_verb  <- c("gwas_glm", "gwas_lmm", "gwas_farmcpu", "gwas_blink")
summary_df$dataset          <- "MDP maize EarHT"
summary_df$n_samples        <- n

## Per-model verdict — different metric for different model classes:
##   * GLM and LMM are score-per-SNP linear models  → gate on p-value cor.
##   * FarmCPU and BLINK are iterative multi-locus  → gate on top-K Jaccard.
verdicts <- character(nrow(summary_df))
gates    <- numeric(nrow(summary_df))
metrics  <- character(nrow(summary_df))
for (i in seq_len(nrow(summary_df))) {
  if (summary_df$label[i] %in% c("GLM", "LMM/MLM")) {
    gates[i]    <- 0.95
    metrics[i]  <- "pearson(-log10p)"
    verdicts[i] <- if (summary_df$pearson[i] > gates[i]) "PASS" else "REVIEW"
  } else {
    gates[i]    <- 0.50
    metrics[i]  <- sprintf("top%d_jaccard", summary_df$top_k[i])
    verdicts[i] <- if (summary_df$jaccard[i] > gates[i]) "PASS" else "REVIEW"
  }
}
summary_df$metric  <- metrics
summary_df$gate    <- gates
summary_df$verdict <- verdicts

cat("\n=== Phase B summary ===\n")
print(summary_df[, c("label", "n", "pearson", "spearman", "jaccard",
                     "metric", "gate", "verdict")], row.names = FALSE)

write.csv(summary_df,
          file.path(out_dir, "phase_b_gapit_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_b_gapit_summary.csv")))
