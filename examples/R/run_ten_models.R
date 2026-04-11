## =============================================================================
## run_ten_models.R -- Twelve GWAS models via rTorchGWAS on CSV data
## =============================================================================
##
## Uses the native rTorchGWAS R verbs (gwas_glm, gwas_lmm, gwas_multi_trait,
## gwas_multi_kernel, gwas_gxe, gwas_farmcpu, gwas_blink, gwas_binary,
## gwas_ordinal, gwas_conditional, gwas_bayesian, gwas_polyploid).
##
## Installation (one-time)
## -----------------------
##     # --- Python backend (skip install_torchgwas(); it pulls from PyPI) ---
##     # From a shell, in the TorchGWAS repo root:
##     #     pip install -e ".[dev]"
##     # This makes the `torchgwas` module importable from whichever Python
##     # you ran pip in. Note the full path to that python.exe:
##     #     python -c "import sys; print(sys.executable)"
##
##     # --- R side ---
##     devtools::install_github("sikiru-atanda/rTorchGWAS")
##
## Data (one-time)
## ---------------
##     Rscript examples/R/00_simulate_csv_data.R
##
## Run
## ---
##     Rscript examples/R/run_ten_models.R
##
## Output
## ------
##     examples/R/results/<NN_model>.csv        -- per-model association table
##     (pvalue, effect, se, statistic, log10p) via as.data.frame(<result>)
## =============================================================================

## ---- Point reticulate at the Python where `torchgwas` is already installed --
## IMPORTANT: this must happen BEFORE `library(rTorchGWAS)`, because the
## package imports torchgwas via reticulate at .onLoad.
##
## The rTorchGWAS helper `install_torchgwas()` pip-installs from PyPI, but
## torchgwas v0.1.0 is not yet published there. Use your existing editable
## install (`pip install -e .` from the repo root) instead.

`%||%` <- function(a, b) if (!is.null(a)) a else b

## 1. respect an existing RETICULATE_PYTHON if the user has already set one;
## 2. otherwise probe a short list of candidates for a working torchgwas import.
configure_python_for_torchgwas <- function() {
  if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) return(invisible())

  candidates <- c(
    "C:/Program Files/Python311/python.exe",
    "C:/Program Files/Python312/python.exe",
    "C:/Program Files/Python310/python.exe",
    "C:/Python/python.exe",
    Sys.which("python"),
    Sys.which("python3")
  )
  candidates <- unique(candidates[nzchar(candidates) & file.exists(candidates)])

  for (py in candidates) {
    chk <- suppressWarnings(system2(py,
      args = c("-c", shQuote("import torchgwas; print(torchgwas.__version__)")),
      stdout = TRUE, stderr = TRUE))
    if (length(chk) >= 1 && !grepl("Error|Traceback", chk[1])) {
      Sys.setenv(RETICULATE_PYTHON = py)
      message(sprintf("Using Python: %s  (torchgwas %s)", py, chk[1]))
      return(invisible())
    }
  }
  stop(
    "Could not find a Python with `torchgwas` importable.\n",
    "Run from a shell in the TorchGWAS repo root:\n",
    "    pip install -e \".[dev]\"\n",
    "then set RETICULATE_PYTHON to that python.exe and retry.",
    call. = FALSE
  )
}
configure_python_for_torchgwas()

suppressPackageStartupMessages(library(rTorchGWAS))

## Sanity check — prints the python + torchgwas version rTorchGWAS sees.
torchgwas_config()

`%||%` <- function(a, b) if (!is.null(a)) a else b

## ---- Paths ------------------------------------------------------------------
.this_file <- tryCatch(
  {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
    if (length(file_arg) > 0) file_arg[1] else sys.frame(1)$ofile
  },
  error = function(e) NULL
)
here     <- normalizePath(dirname(.this_file %||% "."))
data_dir <- file.path(here, "data")
out_dir  <- file.path(here, "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

geno_csv      <- file.path(data_dir, "geno.csv")
poly_geno_csv <- file.path(data_dir, "poly_geno.csv")
pheno_csv     <- file.path(data_dir, "pheno.csv")
covar_csv     <- file.path(data_dir, "covariates.csv")

stopifnot(file.exists(geno_csv), file.exists(pheno_csv), file.exists(covar_csv))

## ---- Step 1: load CSVs into plain R matrices --------------------------------
## rTorchGWAS's gwas_* verbs all take plain (n x m) R matrices, so we read
## the CSV with base R, strip the SNP / Chr / Pos columns, and transpose.
read_geno_csv <- function(path) {
  df   <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  snp  <- as.character(df[[1]])
  chr  <- as.integer(df[[2]])
  pos  <- as.integer(df[[3]])
  sid  <- colnames(df)[-(1:3)]
  G    <- t(as.matrix(df[, -(1:3), drop = FALSE]))   # transpose -> (n x m)
  rownames(G) <- sid
  colnames(G) <- snp
  list(G = G, snp = snp, chr = chr, pos = pos, sample_ids = sid)
}

geno      <- read_geno_csv(geno_csv)
poly_geno <- read_geno_csv(poly_geno_csv)

pheno <- read.csv(pheno_csv, stringsAsFactors = FALSE)
pheno <- pheno[match(geno$sample_ids, pheno$IID), ]            # align to genotype order

covar_df <- read.csv(covar_csv, stringsAsFactors = FALSE)
covar_df <- covar_df[match(geno$sample_ids, covar_df$IID), ]
covar    <- as.matrix(covar_df[, c("sex", "age")])

cat(sprintf("Loaded %d samples x %d variants from %s\n",
            nrow(geno$G), ncol(geno$G), basename(geno_csv)))

## ---- Step 2: imputation + QC (rTorchGWAS verbs) -----------------------------
G <- impute_genotypes(geno$G, method = "mean")

qc <- compute_variant_qc(G, ploidy = 2L)
cat(sprintf("Variant QC: median MAF = %.3f, median call rate = %.3f\n",
            median(qc$maf, na.rm = TRUE),
            1 - median(qc$missingness, na.rm = TRUE)))

filt <- apply_qc_filters(G, maf_threshold = 0.05,
                         hwe_threshold = 1e-6,
                         call_rate_threshold = 0.95,
                         ploidy = 2L)
G    <- filt$genotypes
snp  <- geno$snp[filt$kept_indices]
chr  <- geno$chr[filt$kept_indices]
pos  <- geno$pos[filt$kept_indices]
cat(sprintf("After QC: %d x %d  (%d variants retained)\n",
            nrow(G), ncol(G), length(filt$kept_indices)))

## ---- Step 3: precompute the GRM once, reuse across models -------------------
K <- compute_grm(G, method = "vanraden", ploidy = 2L)

## ---- Step 4: pull trait vectors / matrices from the phenotype frame ---------
y1     <- pheno$Y1
y2     <- pheno$Y2
ybin   <- pheno$Ybin
yord   <- pheno$Yord
E1     <- pheno$E1
fam_id <- pheno$FID
Y_mt   <- as.matrix(pheno[, c("Y1", "Y2")])

## ---- Helper: save a tgwas_result -> CSV with SNP metadata -------------------
save_res <- function(res, name) {
  df <- as.data.frame(res)                              # pvalue, effect, se, ...
  if (nrow(df) == length(snp)) {
    df <- cbind(snp = snp, chr = chr, pos = pos, df)
  }
  out <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(df, out, row.names = FALSE)
  cat(sprintf("  -> %-28s  min p = %.3g   sig@5e-8 = %d\n",
              paste0(name, ".csv"),
              min(df$pvalue, na.rm = TRUE),
              sum(df$pvalue < 5e-8, na.rm = TRUE)))
  invisible(df)
}

## =============================================================================
## Step 5: Twelve GWAS models, each one call
## =============================================================================
cat("\n=== Running 12 GWAS models via rTorchGWAS ===\n")

## 1. GLM ---------------------------------------------------------------------
cat("\n[1/12] gwas_glm         (OLS baseline)\n")
res_glm <- gwas_glm(G, y1, covar = covar)
save_res(res_glm, "01_glm")

## 2. Single-trait LMM (GEMMA-equivalent) -------------------------------------
cat("\n[2/12] gwas_lmm         (single-trait LMM, Wald)\n")
res_lmm <- gwas_lmm(G, y1, covar = covar, K = K, test = "wald", n_pcs = 3L)
save_res(res_lmm, "02_single_trait_lmm")

## 3. Multi-trait LMM (Y1 + Y2 joint) -----------------------------------------
cat("\n[3/12] gwas_multi_trait (Y1 + Y2 joint test)\n")
res_mt <- gwas_multi_trait(G, Y_mt, covar = covar, K = K)
save_res(res_mt, "03_multi_trait_lmm")

## 4. Multi-kernel LMM (additive + dominance-style) ---------------------------
cat("\n[4/12] gwas_multi_kernel (additive + second kernel)\n")
K_add <- compute_grm(G, method = "vanraden", ploidy = 2L)
K_alt <- compute_grm(G, method = "zhang",    ploidy = 2L)
res_mk <- gwas_multi_kernel(G, y1, kernels = list(K_add, K_alt), covar = covar)
save_res(res_mk, "04_multi_kernel_lmm")

## 5. GxE (HetLMM) ------------------------------------------------------------
cat("\n[5/12] gwas_gxe         (HetLMM GxE with E1)\n")
res_gxe <- gwas_gxe(G, y1, env = E1, covar = covar, K = K, model = "het")
save_res(res_gxe, "05_gxe")

## 6. FarmCPU (multi-locus iterative) -----------------------------------------
cat("\n[6/12] gwas_farmcpu     (FEM/REM iterative)\n")
res_fc <- gwas_farmcpu(G, y1, covar = covar, max_qtns = 10L)
save_res(res_fc, "06_farmcpu")

## 7. BLINK -------------------------------------------------------------------
cat("\n[7/12] gwas_blink       (LD-aware multi-locus)\n")
res_bk <- gwas_blink(G, y1, covar = covar, cutoff = 0.01)
save_res(res_bk, "07_blink")

## 8. Binary GWAS (logistic + SPA) --------------------------------------------
cat("\n[8/12] gwas_binary      (case/control, SPA)\n")
res_bin <- gwas_binary(G, ybin, covar = covar, use_glmm = FALSE, firth = FALSE)
save_res(res_bin, "08_binary_glm")

## 9. Ordinal GWAS (proportional odds) ----------------------------------------
cat("\n[9/12] gwas_ordinal     (3-level proportional odds)\n")
res_ord <- gwas_ordinal(G, yord, n_categories = 3L, covar = covar, use_glmm = FALSE)
save_res(res_ord, "09_ordinal_glm")

## 10. LD-conditional (GCTA-COJO stepwise) ------------------------------------
cat("\n[10/12] gwas_conditional (COJO-style stepwise)\n")
res_cond <- gwas_conditional(G, y1, covar = covar, K = K,
                             ld_method = "r2", sig_threshold = 5e-8)
save_res(res_cond, "10_conditional_lmm")

## 11. Bayesian variable selection (SuSiE) ------------------------------------
cat("\n[11/12] gwas_bayesian    (SuSiE fine-mapping)\n")
res_bvs <- gwas_bayesian(G, y1, method = "susie", n_signals = 5L,
                         covar = covar, K = K)
save_res(res_bvs, "11_bayesian_susie")

## 12. Polyploid GWAS (tetraploid) --------------------------------------------
cat("\n[12/12] gwas_polyploid   (tetraploid, additive)\n")
G4   <- impute_genotypes(poly_geno$G, method = "mean")
y1_4 <- pheno$Y1[match(poly_geno$sample_ids, pheno$IID)]
res_poly <- gwas_polyploid(G4, y1_4, ploidy = 4L, gene_action = "additive")
save_res(res_poly, "12_polyploid")

## =============================================================================
## Step 6: multiple-testing correction demo (BH) on the LMM result
## =============================================================================
cat("\n=== Multiple testing correction (BH) on LMM result ===\n")
adj <- multiple_testing(res_lmm$pvalues, method = "bh", alpha = 0.05)
cat(sprintf("  BH: %d / %d variants significant at alpha = 0.05\n",
            sum(adj$significant), nrow(adj)))

## Quick summaries via the package's S3 methods
cat("\n=== summary() dispatch examples ===\n")
print(summary(res_lmm))
print(summary(res_mt))
print(summary(res_fc))

cat("\nAll 12 models complete. Results in: ", out_dir, "\n", sep = "")
