## =============================================================================
## phase_f_plink2_vs_rtg.R -- rTorchGWAS::gwas_binary vs PLINK2 --glm logistic
## =============================================================================
##
## MDP has no native binary trait, so we dichotomize EarHT at the median
## (top half = case = 1, bottom half = control = 0) to get a balanced
## case/control phenotype on n=279 samples.
##
## We export MDP to PLINK 1.x dosage format, run plink2 --glm with logistic
## regression, then compare per-SNP -log10(p) against rTorchGWAS gwas_binary
## (which uses BinaryGLM with the score test by default).  Score and Wald
## tests differ by O(1/n) for finite samples, so we expect r > 0.95 on
## -log10(p), not the >0.999 gate used for matched-test comparisons.

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

data_dir   <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/data"
plink_bin  <- "C:/Users/Sikiru/Documents/GWAS_Expert/tools/plink2/plink2.exe"
work_dir   <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/plink2_work"
out_dir    <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"
dir.create(work_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir,  showWarnings = FALSE, recursive = TRUE)

## ---- Load MDP ---------------------------------------------------------------
geno_df  <- read.table(file.path(data_dir, "mdp_numeric.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
pheno_df <- read.table(file.path(data_dir, "mdp_traits.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE,
                       na.strings = c("NA", "NaN"))
snp_df   <- read.table(file.path(data_dir, "mdp_SNP_information.txt"),
                       header = TRUE, sep = "\t", stringsAsFactors = FALSE)
geno_df$taxa  <- as.character(geno_df$taxa)
pheno_df$Taxa <- as.character(pheno_df$Taxa)

common <- sort(intersect(pheno_df$Taxa[!is.na(pheno_df$EarHT)], geno_df$taxa))
n <- length(common)
cat(sprintf("Common samples (EarHT non-NA): %d\n", n))

geno_sub  <- geno_df[match(common, geno_df$taxa), , drop = FALSE]
pheno_sub <- pheno_df[match(common, pheno_df$Taxa), , drop = FALSE]
snp_names <- colnames(geno_sub)[-1]
G_raw     <- as.matrix(geno_sub[, -1, drop = FALSE])
storage.mode(G_raw) <- "double"

G_imp <- impute_genotypes(G_raw, method = "mean")
rownames(G_imp) <- common
colnames(G_imp) <- snp_names

## ---- Dichotomize phenotype --------------------------------------------------
y_cont <- pheno_sub$EarHT
med    <- median(y_cont)
y_bin  <- as.integer(y_cont > med)  # 0 / 1
cat(sprintf("Binary phenotype: %d cases (1) / %d controls (0)\n",
            sum(y_bin == 1), sum(y_bin == 0)))

## ---- Write PLINK 1.x dosage format ------------------------------------------
##
## File layout (chr/pos embedded as extra columns, see --import-dosage skip1):
##   header: SNP CHR POS A1 A2 FID1 IID1 FID2 IID2 ...
##   rows:   snp1 1 12345 A B  d11  d21  ...
## skip0=0, skip1=2 (CHR + POS columns between SNP and A1), skip2=0
## Then chr-col-num=2 and pos-col-num=3 tell PLINK2 where chr/pos live.
dose_path <- file.path(work_dir, "mdp.dose")
psam_path <- file.path(work_dir, "mdp.psam")

cat("Writing PLINK dosage file ...\n")

mp <- snp_df[match(snp_names, snp_df$SNP), c("Chromosome", "Position")]
stopifnot(!any(is.na(mp$Chromosome)), !any(is.na(mp$Position)))

## Header: SNP CHR POS A1 A2 then alternating FID IID per sample
hdr <- c("SNP", "CHR", "POS", "A1", "A2",
         as.vector(rbind(common, common)))
hdr_line <- paste(hdr, collapse = " ")

## Body rows
body <- character(length(snp_names))
G_t  <- t(G_imp)  # snps x samples
for (i in seq_along(snp_names)) {
  vals <- formatC(G_t[i, ], digits = 6, format = "f")
  body[i] <- paste(c(snp_names[i],
                     mp$Chromosome[i], mp$Position[i],
                     "A", "B", vals),
                   collapse = " ")
}
writeLines(c(hdr_line, body), dose_path)

## .psam with FID and IID columns (matches the FID/IID pairs in the dosage
## header).  PLINK2 expects 1=control, 2=case for binary phenotypes.
psam_lines <- c("#FID IID PHENO1",
                paste(common, common, y_bin + 1L))
writeLines(psam_lines, psam_path)

## ---- Step 1: import dosage to PGEN ------------------------------------------
cat("PLINK2 --import-dosage -> .pgen ...\n")
imp_log <- file.path(work_dir, "mdp_import")
imp_cmd <- c("--import-dosage", dose_path,
             "skip1=2", "chr-col-num=2", "pos-col-num=3",
             "--psam", psam_path,
             "--make-pgen",
             "--out",  imp_log)
imp_out <- system2(plink_bin, imp_cmd, stdout = TRUE, stderr = TRUE)
cat(tail(imp_out, 5), sep = "\n")
stopifnot(file.exists(paste0(imp_log, ".pgen")))

## ---- Step 2: --glm logistic --------------------------------------------------
cat("\nPLINK2 --glm logistic ...\n")
glm_log <- file.path(work_dir, "mdp_glm")
glm_cmd <- c("--pfile", imp_log,
             "--glm", "allow-no-covars", "hide-covar", "omit-ref",
             "--out", glm_log)
glm_out <- system2(plink_bin, glm_cmd, stdout = TRUE, stderr = TRUE)
cat(tail(glm_out, 5), sep = "\n")

## PLINK2 names the output PHENO1.glm.logistic.hybrid for binary
glm_file <- list.files(work_dir,
                       pattern = "mdp_glm\\.PHENO1\\.glm\\.(logistic|firth)",
                       full.names = TRUE)
stopifnot(length(glm_file) >= 1)
cat(sprintf("PLINK2 GLM output: %s\n", basename(glm_file[1])))

plink_glm <- read.table(glm_file[1], header = TRUE, sep = "\t",
                        comment.char = "", check.names = FALSE,
                        stringsAsFactors = FALSE)
## Column name is "#CHROM" so the header line begins with #
colnames(plink_glm) <- gsub("^#", "", colnames(plink_glm))
cat(sprintf("PLINK2 GLM rows: %d  cols: %s\n",
            nrow(plink_glm), paste(colnames(plink_glm), collapse = ", ")))

## ---- Step 3: rTorchGWAS gwas_binary -----------------------------------------
cat("\nRunning rTorchGWAS::gwas_binary (score test) ...\n")
t0 <- proc.time()
res_bin <- gwas_binary(G_imp, y_bin)  # default: BinaryGLM, score test
t_rtg <- (proc.time() - t0)["elapsed"]
cat(sprintf("Done: %.1fs, %d SNPs scanned\n", t_rtg, length(res_bin$pvalues)))

R_df <- data.frame(snp = colnames(G_imp),
                   p_R = as.numeric(res_bin$pvalues),
                   stringsAsFactors = FALSE)

## ---- Step 4: compare ---------------------------------------------------------
P_df <- data.frame(snp = as.character(plink_glm$ID),
                   p_plink = as.numeric(plink_glm$P),
                   stringsAsFactors = FALSE)
m <- merge(R_df, P_df, by = "snp")
m <- m[is.finite(m$p_R) & is.finite(m$p_plink) &
       m$p_R > 0 & m$p_plink > 0, , drop = FALSE]
cat(sprintf("Inner join (positive finite p): %d SNPs\n", nrow(m)))

logp_R     <- -log10(m$p_R)
logp_plink <- -log10(m$p_plink)
pearson    <- cor(logp_R, logp_plink)
spearman   <- cor(logp_R, logp_plink, method = "spearman")
med_dlogp  <- median(abs(logp_R - logp_plink))
max_dlogp  <- max(abs(logp_R - logp_plink))

top_k <- 20
top_R <- m$snp[order(m$p_R)[seq_len(top_k)]]
top_P <- m$snp[order(m$p_plink)[seq_len(top_k)]]
jaccard <- length(intersect(top_R, top_P)) / length(union(top_R, top_P))

cat("\n=== Phase F: rTorchGWAS::gwas_binary vs PLINK2 --glm logistic ===\n")
cat(sprintf("  n_snps_compared       : %d\n", nrow(m)))
cat(sprintf("  cor(-log10 p)         : %.6f\n", pearson))
cat(sprintf("  spearman              : %.6f\n", spearman))
cat(sprintf("  median |d -log10 p|   : %.3e\n", med_dlogp))
cat(sprintf("  max    |d -log10 p|   : %.3e\n", max_dlogp))
cat(sprintf("  top%d Jaccard          : %.3f\n", top_k, jaccard))
cat(sprintf("  rTG min p             : %.3g\n", min(m$p_R)))
cat(sprintf("  PLINK2 min p          : %.3g\n", min(m$p_plink)))

## Score test (rTorchGWAS) vs Wald test (PLINK2 logistic) -- expect agreement
## on rank but not bit-identical p-values, especially in the tails.
verdict <- if (pearson > 0.95 && jaccard >= 0.50) "PASS" else "REVIEW"
cat(sprintf("\n>>> VERDICT: %s\n", verdict))

summary_row <- data.frame(
  phase                = "F",
  reference_tool       = "PLINK2 v2.0.0-a.6.1 --glm logistic (Wald)",
  rtorchgwas_verb      = "gwas_binary (BinaryGLM, score)",
  dataset              = "MDP maize EarHT-dichotomized",
  n_samples            = n,
  n_cases              = sum(y_bin == 1),
  n_controls           = sum(y_bin == 0),
  n_snps_compared      = nrow(m),
  cor_neg_log10_p      = pearson,
  spearman             = spearman,
  median_abs_d_log10p  = med_dlogp,
  max_abs_d_log10p     = max_dlogp,
  top_k                = top_k,
  topk_jaccard         = jaccard,
  rtorchgwas_elapsed_s = unname(t_rtg),
  verdict              = verdict,
  stringsAsFactors     = FALSE
)
write.csv(m, file.path(out_dir, "phase_f_plink2_per_snp.csv"), row.names = FALSE)
write.csv(summary_row, file.path(out_dir, "phase_f_plink2_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_f_plink2_summary.csv")))
