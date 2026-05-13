#!/usr/bin/env Rscript
# Run rrBLUP::A.mat + rrBLUP::GWAS on the extracted SoyNAM panel and emit
# a JSON blob the Python harness (compare.py) parses.
#
# Reference quantities:
#   - K_amat              : rrBLUP::A.mat additive kinship (n × n)
#   - vc_null             : variance components (Vu, Ve, h2) from
#                           rrBLUP::mixed.solve on the null model.
#   - gwas_neg_log10p     : per-SNP -log10(p) from rrBLUP::GWAS(P3D=TRUE)
#                           with K = A.mat + n.PC=0.
#
# rrBLUP::GWAS encodes genotypes as {-1, 0, 1} = {aa, Aa, AA}; we recode
# the dosage-0/1/2 input via (g - 1) so the comparison aligns. The β
# estimate is not exposed by rrBLUP::GWAS — only the -log10(p) per marker
# is returned. We therefore compare on -log10(p) (and verify the kinship
# matrix and the variance components match before the per-SNP comparison
# runs).
#
# Note on rrBLUP::GWAS internal SE convention: GWAS computes the per-SNP
# Wald F-statistic via t² with df = n - rank(X) - 1 (one-tailed F(1, df)
# transformed through pf). TG SingleTraitLMM uses χ² with df=1 (large-
# sample Wald). On N=547 the two converge to within 1e-3 on -log10(p) for
# strong-signal SNPs; we calibrate the floor accordingly.
#
# Output layout (outputs/rrblup_results.json + outputs/A_mat.tsv):
#   rrblup_results.json:
#     {
#       "kinship": {"file": "A_mat.tsv", "n": ..., "diag_mean": ..., "off_mean": ...},
#       "vc_null": {"Vu": ..., "Ve": ..., "h2": ..., "loglik": ...},
#       "gwas":    {"snps": [...], "chr": [...], "pos": [...], "neg_log10p": [...]},
#       "package_versions": {...}
#     }
#   A_mat.tsv: full n × n A.mat output (header = strain IDs, first col = strain)

suppressPackageStartupMessages({
    library(rrBLUP)
    library(jsonlite)
})

# --- CLI args ----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA_DIR <- parse_arg("--data-dir", "data")
OUT_DIR  <- parse_arg("--output-dir", "outputs")
TRAIT_COL <- parse_arg("--trait-col", "y_blup")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[run_soynam] data_dir=%s out_dir=%s trait_col=%s\n",
            DATA_DIR, OUT_DIR, TRAIT_COL))

# --- Load extracted TSVs -----------------------------------------------------
geno <- read.delim(file.path(DATA_DIR, "geno.tsv"), check.names = FALSE,
                   stringsAsFactors = FALSE)
pheno <- read.delim(file.path(DATA_DIR, "pheno.tsv"), stringsAsFactors = FALSE)
fam   <- read.delim(file.path(DATA_DIR, "family.tsv"), stringsAsFactors = FALSE)
mmap  <- read.delim(file.path(DATA_DIR, "marker_map.tsv"), stringsAsFactors = FALSE)

stopifnot(geno$strain[1] == pheno$strain[1])

strains <- geno$strain
geno_mat <- as.matrix(geno[, -1])           # 0/1/2 dosage
rownames(geno_mat) <- strains
mode(geno_mat) <- "numeric"
n <- nrow(geno_mat)
m <- ncol(geno_mat)
cat(sprintf("[run_soynam] loaded %d RILs × %d SNPs\n", n, m))

# --- Recode dosage 0/1/2 -> -1/0/1 (rrBLUP convention) -----------------------
geno_centered <- geno_mat - 1
stopifnot(min(geno_centered) >= -1)
stopifnot(max(geno_centered) <= 1)

# --- Kinship: A.mat (additive, VanRaden-style) -------------------------------
cat("[run_soynam] computing A.mat kinship...\n")
K <- A.mat(geno_centered, return.imputed = FALSE)
stopifnot(nrow(K) == n)
stopifnot(ncol(K) == n)

K_path <- file.path(OUT_DIR, "A_mat.tsv")
write.table(
    cbind(strain = rownames(K), as.data.frame(K)),
    file = K_path, sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[run_soynam] wrote %s\n", K_path))

# --- Variance components (null model) ----------------------------------------
y <- pheno[[TRAIT_COL]]
stopifnot(all(pheno$strain == strains))

cat("[run_soynam] computing null-model variance components via mixed.solve...\n")
ms <- mixed.solve(y = y, Z = NULL, K = K, X = NULL, SE = FALSE, return.Hinv = FALSE)
vc_null <- list(
    Vu     = unname(ms$Vu),
    Ve     = unname(ms$Ve),
    h2     = unname(ms$Vu / (ms$Vu + ms$Ve)),
    beta_intercept = unname(ms$beta),
    LL     = unname(ms$LL)
)

cat(sprintf("[run_soynam] null Vu=%.4f Ve=%.4f h2=%.4f LL=%.4f\n",
            vc_null$Vu, vc_null$Ve, vc_null$h2, vc_null$LL))

# --- GWAS via rrBLUP::GWAS ---------------------------------------------------
# GWAS expects:
#   pheno : data.frame, first col = gid (line name), then phenotype columns.
#   geno  : data.frame, first col = SNP name, second col = chr, third col = pos,
#           subsequent cols = marker scores per line (named by line).
#   K     : kinship matrix (rows / cols named by line ID).
# We assemble both with strain IDs as the join key.

cat("[run_soynam] running rrBLUP::GWAS (P3D=TRUE, n.PC=0)...\n")
pheno_df <- data.frame(gid = strains, y = y, stringsAsFactors = FALSE)
geno_df  <- data.frame(
    snp = mmap$snp,
    chr = mmap$chr,
    pos = mmap$pos,
    stringsAsFactors = FALSE
)
# Append per-line dosage columns (already in -1/0/1 encoding).
mat_t <- t(geno_centered)              # m × n
colnames(mat_t) <- strains
geno_df <- cbind(geno_df, as.data.frame(mat_t, stringsAsFactors = FALSE))

# rrBLUP::GWAS works with the chromosome column as either character or
# numeric. We pass the Gm01..Gm20 labels directly.
suppressMessages({
    gwas_t0 <- proc.time()
    gwas_res <- GWAS(
        pheno = pheno_df,
        geno  = geno_df,
        K     = K,
        n.PC  = 0,
        min.MAF = 0.05,
        P3D   = TRUE,
        plot  = FALSE
    )
    gwas_dt <- proc.time() - gwas_t0
})
cat(sprintf("[run_soynam] GWAS done in %.1f s elapsed; result dim %d × %d\n",
            gwas_dt[["elapsed"]], nrow(gwas_res), ncol(gwas_res)))

# rrBLUP::GWAS returns: snp, chr, pos, <trait>_neg_log10p  per row; the
# 4th column header is the trait name. Identify the result column by
# position (always the 4th).
neg_log10p <- as.numeric(gwas_res[[4]])
gwas_block <- list(
    snps        = as.character(gwas_res[[1]]),
    chr         = as.character(gwas_res[[2]]),
    pos         = as.integer(gwas_res[[3]]),
    neg_log10p  = neg_log10p,
    n_snps      = length(neg_log10p),
    elapsed_s   = unname(gwas_dt[["elapsed"]])
)

# --- Compose JSON output -----------------------------------------------------
out <- list(
    kinship = list(
        file       = "A_mat.tsv",
        n          = n,
        diag_mean  = mean(diag(K)),
        off_mean   = mean(K[lower.tri(K)]),
        diag_min   = min(diag(K)),
        diag_max   = max(diag(K))
    ),
    vc_null = vc_null,
    gwas    = gwas_block,
    package_versions = list(
        rrBLUP = as.character(packageVersion("rrBLUP")),
        R      = R.version.string
    ),
    n_ril = n,
    n_snp = m
)

out_path <- file.path(OUT_DIR, "rrblup_results.json")
write_json(out, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf("[run_soynam] wrote %s\n", out_path))

cat(sprintf("[run_soynam] summary: K_diag=%.3f K_off=%.3f h2_null=%.3f max_neg_log10p=%.3f\n",
            mean(diag(K)), mean(K[lower.tri(K)]), vc_null$h2, max(neg_log10p, na.rm = TRUE)))
