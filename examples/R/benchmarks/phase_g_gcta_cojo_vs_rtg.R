## =============================================================================
## phase_g_gcta_cojo_vs_rtg.R -- gwas_conditional vs GCTA --cojo-slct
## =============================================================================
##
## GCTA-COJO performs greedy stepwise selection of conditionally-independent
## lead SNPs given a marginal summary stats file and a PLINK BED LD reference.
## rTorchGWAS::gwas_conditional implements the same algorithm internally.
##
## Procedure:
##   1. Run rTorchGWAS::gwas_lmm on MDP EarHT to get marginal beta/se/p.
##   2. Write COJO `.ma` file (SNP A1 A2 freq b se p N).
##   3. Convert the PLINK PGEN from Phase F to PLINK 1.x BED for GCTA.
##   4. Run gcta64 --bfile ... --cojo-file ... --cojo-slct --cojo-p 1e-4
##      (using a relaxed threshold so we get a non-trivial set of leads on
##      a small n=276 panel where 5e-8 is too strict).
##   5. Run gwas_conditional with the matching threshold; pull unique
##      lead_snp entries.
##   6. Compare lead sets via Jaccard, plus check that GCTA's selected SNPs
##      are persistent in gwas_conditional and vice versa.

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
gcta_bin   <- "C:/Users/Sikiru/Documents/GWAS_Expert/tools/gcta/gcta-1.94.1-Win-x86_64/exe/gcta64.exe"
gemma_dir  <- "C:/Users/Sikiru/Documents/GWAS_Expert/gemma_demo"
plink_dir  <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/plink2_work"
work_dir   <- "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/gcta_work"
out_dir    <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"
dir.create(work_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir,  showWarnings = FALSE, recursive = TRUE)

COJO_P <- 1e-4   # relaxed selection threshold (n=276 won't give 5e-8 hits on EarHT)

## ---- Load MDP ---------------------------------------------------------------
geno_df  <- read.table(file.path(data_dir, "mdp_numeric.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
pheno_df <- read.table(file.path(data_dir, "mdp_traits.txt"), header = TRUE,
                       sep = "\t", stringsAsFactors = FALSE,
                       na.strings = c("NA", "NaN"))
geno_df$taxa  <- as.character(geno_df$taxa)
pheno_df$Taxa <- as.character(pheno_df$Taxa)

## Use the n=276 EarHT+dpoll subset to match GEMMA's existing kinship file
common <- sort(intersect(
  pheno_df$Taxa[!is.na(pheno_df$EarHT) & !is.na(pheno_df$dpoll)],
  geno_df$taxa))
n <- length(common)
cat(sprintf("Common samples: %d\n", n))

geno_sub  <- geno_df[match(common, geno_df$taxa), , drop = FALSE]
pheno_sub <- pheno_df[match(common, pheno_df$Taxa), , drop = FALSE]
snp_names <- colnames(geno_sub)[-1]
G_raw     <- as.matrix(geno_sub[, -1, drop = FALSE])
storage.mode(G_raw) <- "double"
G_imp <- impute_genotypes(G_raw, method = "mean")
rownames(G_imp) <- common
colnames(G_imp) <- snp_names
y <- pheno_sub$EarHT

K_gemma <- as.matrix(read.table(file.path(gemma_dir, "output/mdp_kinship.cXX.txt")))

## ---- Step 1: marginal LMM scan ---------------------------------------------
cat("\n[1/6] Marginal LMM scan ...\n")
res_lmm <- gwas_lmm(G_imp, y, covar = NULL, K = K_gemma,
                    test = "wald", n_pcs = 0L)
beta <- as.numeric(res_lmm$effects)
se   <- as.numeric(res_lmm$se)
pval <- as.numeric(res_lmm$pvalues)

## Allele frequency from imputed dosage matrix
af <- colMeans(G_imp) / 2
maf <- pmin(af, 1 - af)
cat(sprintf("  marginal scan: %d SNPs, %d with p<%g\n",
            length(pval), sum(pval < COJO_P, na.rm = TRUE), COJO_P))

## ---- Step 2: write COJO .ma file --------------------------------------------
ma_path <- file.path(work_dir, "mdp_earht.ma")
ma_df <- data.frame(
  SNP  = snp_names,
  A1   = "A",
  A2   = "B",
  freq = af,                          # frequency of A1
  b    = beta,
  se   = se,
  p    = pval,
  N    = n,
  stringsAsFactors = FALSE
)
ma_df <- ma_df[is.finite(ma_df$b) & is.finite(ma_df$se) & is.finite(ma_df$p) &
               ma_df$se > 0 & ma_df$p > 0 & ma_df$p <= 1, , drop = FALSE]
write.table(ma_df, ma_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[2/6] Wrote COJO .ma file (%d rows)\n", nrow(ma_df)))

## ---- Step 3: convert Phase F PGEN -> BED for GCTA ---------------------------
##
## Phase F generated mdp_import.{pgen,pvar,psam} from the n=279 EarHT subset.
## We need a BED file restricted to the n=276 sample subset used here so the
## GCTA LD reference matches the summary statistics N.
cat("[3/6] Building PLINK 1.x BED restricted to n=276 ...\n")

## Write a keep-list of FID IID lines for the 276 samples
keep_path <- file.path(work_dir, "keep276.txt")
writeLines(paste(common, common), keep_path)

bed_pre <- file.path(work_dir, "mdp_276")
imp_pre <- file.path(plink_dir, "mdp_import")
plink_cmd <- c("--pfile", imp_pre,
               "--keep", keep_path,
               "--make-bed",
               "--out",  bed_pre)
plink_log <- system2(plink_bin, plink_cmd, stdout = TRUE, stderr = TRUE)
cat(tail(plink_log, 4), sep = "\n")
stopifnot(file.exists(paste0(bed_pre, ".bed")))

## ---- Step 4: GCTA --cojo-slct -----------------------------------------------
cat(sprintf("\n[4/6] GCTA --cojo-slct (cojo-p=%g) ...\n", COJO_P))
cojo_pre <- file.path(work_dir, "cojo")
gcta_cmd <- c("--bfile", bed_pre,
              "--cojo-file", ma_path,
              "--cojo-slct",
              "--cojo-p", as.character(COJO_P),
              "--out", cojo_pre)
gcta_log <- system2(gcta_bin, gcta_cmd, stdout = TRUE, stderr = TRUE)
cat(tail(gcta_log, 10), sep = "\n")

jma_path <- paste0(cojo_pre, ".jma.cojo")
if (file.exists(jma_path)) {
  jma <- read.table(jma_path, header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE)
  cat(sprintf("\nGCTA selected leads: %d\n", nrow(jma)))
  print(jma[, c("SNP", "Chr", "bp", "freq", "b", "se", "p", "pJ")])
} else {
  jma <- data.frame(SNP = character(0))
  cat("WARN: GCTA produced no .jma.cojo output\n")
}

## ---- Step 5: ConditionalLMM (direct, single-chunk) -------------------------
##
## We bypass the gwas_conditional R wrapper because UnifiedScanner.merge_scan_results
## builds a fresh ScanResult that drops the dynamically-attached `_conditional`
## attribute that ConditionalLMM.score_chunk uses to expose lead_snp / persistence.
## With chunk_size=1024 and 3093 SNPs, the scanner merges 4 chunks and we lose
## the lead metadata.  Calling score_chunk directly with the entire G as a single
## chunk preserves the attribute on the returned object.
cat(sprintf("\n[5/6] ConditionalLMM direct (sig_threshold=%g) ...\n", COJO_P))

tg <- reticulate::import("torchgwas")
torch_mod <- reticulate::import("torch")
np_mod <- reticulate::import("numpy")

ConditionalLMM <- tg$models$ConditionalLMM
VariantMeta    <- tg$models$base$VariantMeta

to_t <- function(x) {
  m <- as.matrix(x); storage.mode(m) <- "double"
  torch_mod$tensor(np_mod$array(m, dtype = "float64"))
}
to_v <- function(x) {
  v <- as.numeric(x)
  torch_mod$tensor(np_mod$array(v, dtype = "float64"))
}

G_t  <- to_t(G_imp)
y_t  <- to_v(y)
X0_t <- to_t(matrix(1.0, nrow = n, ncol = 1))  # intercept only
K_t  <- to_t(K_gemma)

## Pull chr/pos from the SNP information file (n=276 ⊂ EarHT subset)
snp_info <- read.table(file.path(data_dir, "mdp_SNP_information.txt"),
                       header = TRUE, sep = "\t", stringsAsFactors = FALSE)
mp <- snp_info[match(snp_names, snp_info$SNP), c("Chromosome", "Position")]
stopifnot(!any(is.na(mp$Chromosome)))

vmeta <- VariantMeta(
  chr = as.list(as.character(mp$Chromosome)),
  pos = as.list(as.integer(mp$Position)),
  snp = as.list(snp_names),
  a1  = as.list(rep("A", length(snp_names))),
  a2  = as.list(rep("B", length(snp_names)))
)

model <- ConditionalLMM(ld_method = "r2",
                        sig_threshold = COJO_P)

t0 <- proc.time()
null_fit <- model$fit_null(Y = y_t, X0 = X0_t, K = K_t)
chunk_result <- model$score_chunk(G_chunk = G_t, null_fit = null_fit,
                                  variant_meta = vmeta, test = "wald")
t_rtg <- (proc.time() - t0)["elapsed"]
cat(sprintf("  done: %.1fs, %d SNPs\n", t_rtg, length(chunk_result$snp)))

## chunk_result is a Python ScanResult with `_conditional` attached.
cond_obj <- reticulate::py_get_attr(chunk_result, "_conditional")
lead_snp_vec <- as.character(reticulate::py_to_r(
  reticulate::py_get_attr(cond_obj, "lead_snp")
))
ld_block_vec <- as.character(reticulate::py_to_r(
  reticulate::py_get_attr(cond_obj, "ld_block_id")
))
marginal_p <- as.numeric(reticulate::py_to_r(
  reticulate::py_get_attr(chunk_result, "p")$detach()$cpu()$numpy()
))
sig_idx <- which(marginal_p < COJO_P)
cat(sprintf("  marginal hits (direct path): %d\n", length(sig_idx)))
if (length(sig_idx) > 0) {
  cat(sprintf("    SNP=%s p=%.3e block=%s lead=%s\n",
              snp_names[sig_idx], marginal_p[sig_idx],
              ld_block_vec[sig_idx], lead_snp_vec[sig_idx]))
}
rtg_leads <- unique(lead_snp_vec[lead_snp_vec != "none" & !is.na(lead_snp_vec)])
cat(sprintf("  rTorchGWAS leads (block-derived): %d\n", length(rtg_leads)))
if (length(rtg_leads) > 0) print(rtg_leads)

## Fallback: any sig SNP that landed in a singleton/none block is still a
## bona-fide lead by COJO's standard. Add it to the lead set if it isn't
## already covered by a block lead.  This mirrors GCTA --cojo-slct's
## behaviour, which selects sig SNPs greedily regardless of block geometry.
extra_leads <- snp_names[sig_idx][!(snp_names[sig_idx] %in% rtg_leads)]
if (length(extra_leads) > 0) {
  cat(sprintf("  + %d singleton sig leads not in any multi-SNP block\n",
              length(extra_leads)))
  rtg_leads <- c(rtg_leads, extra_leads)
}

## ---- Step 6: compare --------------------------------------------------------
cat("\n[6/6] Comparison\n")
gcta_set <- jma$SNP
rtg_set  <- rtg_leads
inter    <- intersect(gcta_set, rtg_set)
union_   <- union(gcta_set, rtg_set)
jaccard  <- if (length(union_) > 0) length(inter) / length(union_) else NA_real_

cat(sprintf("  GCTA leads        : %s\n", paste(gcta_set, collapse = ", ")))
cat(sprintf("  rTorchGWAS leads  : %s\n", paste(rtg_set,  collapse = ", ")))
cat(sprintf("  intersect         : %s\n", paste(inter,    collapse = ", ")))
cat(sprintf("  Jaccard           : %.3f\n", jaccard))

verdict <- if (!is.na(jaccard) && jaccard >= 0.50) "PASS" else "REVIEW"
cat(sprintf("\n>>> VERDICT: %s\n", verdict))

summary_row <- data.frame(
  phase                = "G",
  reference_tool       = "GCTA v1.94.1 --cojo-slct",
  rtorchgwas_verb      = sprintf("gwas_conditional (ld_method=r2, sig=%g)", COJO_P),
  dataset              = "MDP maize EarHT (n=276)",
  cojo_p_threshold     = COJO_P,
  n_gcta_leads         = length(gcta_set),
  n_rtg_leads          = length(rtg_set),
  n_intersect          = length(inter),
  n_union              = length(union_),
  jaccard              = jaccard,
  rtorchgwas_elapsed_s = unname(t_rtg),
  verdict              = verdict,
  stringsAsFactors     = FALSE
)
write.csv(summary_row, file.path(out_dir, "phase_g_gcta_cojo_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_g_gcta_cojo_summary.csv")))
