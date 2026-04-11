## =============================================================================
## phase_c_mvlmm_vs_gemma.R -- rTorchGWAS::gwas_multi_trait vs GEMMA -lmm 1
## =============================================================================
##
## Two-trait joint test (EarHT + dpoll) on MDP maize, n = 276 samples.
## We pass GEMMA's own centered kinship into rTorchGWAS so both stacks see
## the same K, then compare per-SNP joint Wald p-values via inner-join.

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

data_dir  <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/data"
gemma_dir <- "C:/Users/Sikiru/Documents/GWAS_Expert/gemma_demo"
out_dir   <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## ---- Load data --------------------------------------------------------------
geno_df  <- read.table(file.path(data_dir, "mdp_numeric.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
pheno_df <- read.table(file.path(data_dir, "mdp_traits.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE,
                       na.strings = c("NA", "NaN"))
geno_df$taxa  <- as.character(geno_df$taxa)
pheno_df$Taxa <- as.character(pheno_df$Taxa)

## Two-trait subset to match GEMMA -n 1 2
common <- sort(intersect(
  pheno_df$Taxa[!is.na(pheno_df$EarHT) & !is.na(pheno_df$dpoll)],
  geno_df$taxa))
n <- length(common)
cat(sprintf("Common samples (EarHT+dpoll non-NA): %d\n", n))
stopifnot(n == 276L)

geno_sub  <- geno_df[match(common, geno_df$taxa), , drop = FALSE]
pheno_sub <- pheno_df[match(common, pheno_df$Taxa), , drop = FALSE]
snp_names <- colnames(geno_sub)[-1]
G_raw     <- as.matrix(geno_sub[, -1, drop = FALSE])
storage.mode(G_raw) <- "double"

G_imp <- impute_genotypes(G_raw, method = "mean")
rownames(G_imp) <- common
colnames(G_imp) <- snp_names

K_gemma <- as.matrix(read.table(file.path(gemma_dir, "output/mdp_kinship.cXX.txt")))
stopifnot(nrow(K_gemma) == n)

Y <- as.matrix(pheno_sub[, c("EarHT", "dpoll")])
stopifnot(!any(is.na(Y)))

## ---- Run rTorchGWAS multi-trait --------------------------------------------
cat("\nRunning rTorchGWAS::gwas_multi_trait ...\n")
t0 <- proc.time()
res_mt <- gwas_multi_trait(G_imp, Y, K = K_gemma)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done: %.1fs, %d SNPs scanned\n", elapsed, length(res_mt$pvalues)))

R_df <- data.frame(snp = snp_names,
                   p_R = as.numeric(res_mt$pvalues),
                   stringsAsFactors = FALSE)

## ---- Load GEMMA mvLMM -------------------------------------------------------
gemma <- read.table(file.path(gemma_dir, "output/mdp_mvlmm_wald.assoc.txt"),
                    header = TRUE, sep = "\t", stringsAsFactors = FALSE)
G_df <- data.frame(snp     = gemma$rs,
                   n_miss  = gemma$n_miss,
                   p_gemma = gemma$p_wald,
                   stringsAsFactors = FALSE)
cat(sprintf("GEMMA mvLMM assoc: %d SNPs (%d n_miss=0)\n",
            nrow(G_df), sum(G_df$n_miss == 0)))

## ---- Compare ----------------------------------------------------------------
m <- merge(R_df, G_df, by = "snp")
m <- m[is.finite(m$p_R) & is.finite(m$p_gemma) &
       m$p_R > 0 & m$p_gemma > 0, , drop = FALSE]
cat(sprintf("Inner join (positive finite p): %d SNPs\n", nrow(m)))

log10p_R     <- -log10(m$p_R)
log10p_gemma <- -log10(m$p_gemma)
pearson      <- cor(log10p_R, log10p_gemma)
spearman     <- cor(log10p_R, log10p_gemma, method = "spearman")
med_dlogp    <- median(abs(log10p_R - log10p_gemma))
max_dlogp    <- max(abs(log10p_R - log10p_gemma))

## Top-K hit overlap
top_k <- 20
top_R <- m$snp[order(m$p_R)[seq_len(top_k)]]
top_G <- m$snp[order(m$p_gemma)[seq_len(top_k)]]
jaccard <- length(intersect(top_R, top_G)) / length(union(top_R, top_G))

cat("\n=== Phase C: rTorchGWAS::gwas_multi_trait vs GEMMA -lmm 1 ===\n")
cat(sprintf("  n_snps_compared       : %d\n", nrow(m)))
cat(sprintf("  cor(-log10 p_wald)    : %.6f\n", pearson))
cat(sprintf("  spearman              : %.6f\n", spearman))
cat(sprintf("  median |d -log10 p|   : %.3e\n", med_dlogp))
cat(sprintf("  max    |d -log10 p|   : %.3e\n", max_dlogp))
cat(sprintf("  top%d Jaccard          : %.3f\n", top_k, jaccard))
cat(sprintf("  rTG min p             : %.3g\n", min(m$p_R)))
cat(sprintf("  GEMMA min p           : %.3g\n", min(m$p_gemma)))

verdict <- if (pearson > 0.99 && jaccard > 0.50) "PASS" else "REVIEW"
cat(sprintf("\n>>> VERDICT: %s\n", verdict))

summary_row <- data.frame(
  phase                = "C",
  reference_tool       = "GEMMA 0.98.5 -lmm 1 (mvLMM Wald)",
  rtorchgwas_verb      = "gwas_multi_trait",
  dataset              = "MDP maize EarHT+dpoll",
  n_samples            = n,
  n_snps_compared      = nrow(m),
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
write.csv(m, file.path(out_dir, "phase_c_mvlmm_per_snp.csv"), row.names = FALSE)
write.csv(summary_row, file.path(out_dir, "phase_c_mvlmm_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_c_mvlmm_summary.csv")))
