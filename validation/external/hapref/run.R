#!/usr/bin/env Rscript
# Run haplo.em + haplo.glm on the MDP 5-SNP chr1 block; emit JSON for
# compare.py.  Output: outputs/haplo_results.json
#
# Window: chr1:238902012-238902252  (id1.6, id1.3, id1.2, PZB02144.4, PZB02144.5)
# Phenotype: EarHT (continuous, mm) from mdp_traits.txt
#
# Reference-haplotype pinning:
#   TG HaplotypeGWAS drops the *most frequent* haplotype (argmax(frequencies)).
#   We pin haplo.glm to the same baseline via haplo.glm.control(haplo.base=...)
#   AFTER running haplo.em so both tools regress on the same K-1 design.
#
# Allele encoding:
#   mdp_numeric.txt: value 2 = minor-allele homozygote count (GAPIT convention).
#   For each SNP X/Y in HapMap, X = major (codes value 0), Y = minor (codes
#   value 2).  This script splits each dosage into two pseudo-alleles for
#   haplo.em (a1 = first allele copy, a2 = second copy) using:
#     value 0 -> (X, X)
#     value 2 -> (Y, Y)
#     value 1 -> (X, Y)   # order arbitrary; haplo.em is phase-agnostic

suppressPackageStartupMessages({
    library(haplo.stats)
    library(jsonlite)
})

# --- CLI args --------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

HERE     <- normalizePath(dirname(sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), mustWork = FALSE)
if (is.na(HERE) || HERE == "") HERE <- "."
DATA_DIR <- parse_arg("--data-dir", file.path(HERE, "data"))
OUT_DIR  <- parse_arg("--output-dir", file.path(HERE, "outputs"))
SEED     <- as.integer(parse_arg("--seed", "42"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("[run] data_dir=%s out_dir=%s seed=%d\n", DATA_DIR, OUT_DIR, SEED))

# --- Load fixture ----------------------------------------------------------
win <- fromJSON(file.path(DATA_DIR, "window.json"))
snp_ids <- win$snp_ids
snp_a1  <- win$snp_a1   # major allele (codes value 0)
snp_a2  <- win$snp_a2   # minor allele (codes value 2)
M <- length(snp_ids)
stopifnot(M == 5)

geno <- read.table(file.path(DATA_DIR, "mdp_numeric.txt"), header = TRUE,
                   sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
pheno <- read.table(file.path(DATA_DIR, "mdp_traits.txt"), header = TRUE,
                    sep = "\t", stringsAsFactors = FALSE)

geno$taxa <- as.character(geno$taxa)
pheno$Taxa <- as.character(pheno$Taxa)

# Restrict to common taxa with non-missing EarHT
pheno_sub <- pheno[!is.na(pheno$EarHT), c("Taxa", "EarHT"), drop = FALSE]
common <- intersect(geno$taxa, pheno_sub$Taxa)
common <- sort(common)
cat(sprintf("[run] n_common_taxa_with_EarHT=%d\n", length(common)))

geno_sub  <- geno[match(common, geno$taxa), ]
pheno_sub <- pheno_sub[match(common, pheno_sub$Taxa), ]
stopifnot(all(geno_sub$taxa == pheno_sub$Taxa))

# --- Build the (n x 2M) genotype matrix for haplo.em -----------------------
# For each SNP and each individual:
#   numeric 0 -> (a1, a1)
#   numeric 2 -> (a2, a2)
#   numeric 1 -> (a1, a2)   (order is irrelevant for haplo.em)
#   NA        -> (NA, NA)
n <- nrow(geno_sub)
geno_mat <- matrix(NA_character_, nrow = n, ncol = 2 * M)
colnames(geno_mat) <- as.vector(rbind(paste0(snp_ids, ".a"), paste0(snp_ids, ".b")))

for (j in seq_len(M)) {
    snp <- snp_ids[j]
    v <- geno_sub[[snp]]
    a1 <- snp_a1[j]
    a2 <- snp_a2[j]
    col_a <- 2 * j - 1
    col_b <- 2 * j
    for (i in seq_len(n)) {
        vi <- v[i]
        if (is.na(vi)) {
            geno_mat[i, col_a] <- NA
            geno_mat[i, col_b] <- NA
        } else if (vi == 0) {
            geno_mat[i, col_a] <- a1
            geno_mat[i, col_b] <- a1
        } else if (vi == 2) {
            geno_mat[i, col_a] <- a2
            geno_mat[i, col_b] <- a2
        } else if (vi == 1) {
            geno_mat[i, col_a] <- a1
            geno_mat[i, col_b] <- a2
        } else {
            stop(sprintf("unexpected numeric value %s at i=%d j=%d", vi, i, j))
        }
    }
}

y <- pheno_sub$EarHT

cat(sprintf("[run] genotype matrix: n=%d, 2M=%d, missing=%d\n",
            n, 2 * M, sum(is.na(geno_mat))))

# --- Run haplo.em ----------------------------------------------------------
set.seed(SEED)
em <- haplo.em(geno = geno_mat, locus.label = snp_ids,
               control = haplo.em.control(min.posterior = 1e-4))

# em$haplotype is an (H x M) matrix of allele labels; em$hap.prob is the
# vector of EM frequencies.
hap_letters <- em$haplotype                # (H, M) character
hap_freq    <- as.numeric(em$hap.prob)     # (H,)
H <- length(hap_freq)

# Build the TG-style 0/1 label per haplotype.
# In the TG world, position j is 0 if it equals snp_a1[j], 1 if snp_a2[j].
tg_label <- character(H)
hap_str  <- character(H)   # space-delimited letters for human display
for (h in seq_len(H)) {
    letters <- as.character(hap_letters[h, ])
    bits <- ifelse(letters == snp_a1, "0",
             ifelse(letters == snp_a2, "1", NA))
    if (any(is.na(bits))) stop(sprintf("haplotype %d has letter outside {a1,a2}", h))
    tg_label[h] <- paste(bits, collapse = "")
    hap_str[h]  <- paste(letters, collapse = "")
}

cat(sprintf("[run] haplo.em returned %d haplotypes; sum(freq)=%.6f\n",
            H, sum(hap_freq)))
for (h in order(-hap_freq)) {
    cat(sprintf("  %s  (tg=%s)  freq=%.4f\n", hap_str[h], tg_label[h], hap_freq[h]))
}

# --- Pick reference haplotype (most frequent) ------------------------------
# TG drops argmax(frequencies); pin haplo.glm to the same baseline.
ref_idx <- which.max(hap_freq)
ref_tg <- tg_label[ref_idx]
ref_letters <- hap_str[ref_idx]
cat(sprintf("[run] reference haplotype (most frequent): tg=%s  letters=%s  freq=%.4f\n",
            ref_tg, ref_letters, hap_freq[ref_idx]))

# --- haplo.glm: phenotype ~ haplotypes -------------------------------------
# haplo.glm internally re-runs haplo.em; we pass control with our reference
# pinned via haplo.base (the row index in em$haplotype).  Family = gaussian.
#
# Note: haplo.glm builds its own design from `geno`; passing the row index
# of the most-frequent haplotype keeps the dropped baseline aligned with TG.

glm_data <- data.frame(y = y, geno_mat, stringsAsFactors = FALSE,
                       check.names = FALSE)
hgcontrol <- haplo.glm.control(haplo.base = ref_idx,
                               haplo.min.count = 5,
                               em.c = haplo.em.control(min.posterior = 1e-4))
set.seed(SEED)
# Build the formula explicitly so haplo.glm sees all 2M genotype columns.
loc_form <- paste(colnames(geno_mat), collapse = ", ")
# haplo.glm needs the haplo.score formula syntax: y ~ haplo(loc1.a, loc1.b, ...)
# Use the geno.glm convenience function with all locus columns.
# The recommended pattern (see ?haplo.glm) is:
#   fit <- haplo.glm(y ~ snp1 + snp2 + ... where each snpJ collapses 2 cols)
# but the cleaner path is via setupGeno().
geno_setup <- setupGeno(geno_mat, miss.val = c(NA))
glm_data2  <- data.frame(y = y, geno_setup)
# colnames after setupGeno: first col "geno_setup" (an S4-ish factor) holds
# the multi-locus genotype.  haplo.glm expects: y ~ geno_setup
fit <- haplo.glm(
    formula = y ~ geno_setup,
    family  = gaussian(),
    data    = glm_data2,
    na.action = "na.geno.keep",
    locus.label = snp_ids,
    control = hgcontrol
)

# --- Extract per-haplotype coefficients ------------------------------------
# fit$haplo.unique  : (H x M) data.frame of allele labels per haplotype
# fit$haplo.base    : index in haplo.unique used as the dropped baseline
# fit$haplo.common  : indices of haplotypes that entered the model
# fit$haplo.rare    : indices of haplotypes pooled into a single "rare" bin
# fit$haplo.freq    : EM frequencies for haplo.unique rows
# coef(fit)         : intercept + geno_setup.<hap_idx> per common non-base hap
# vcov(fit)         : covariance of the coefficients (use sqrt(diag) for SE)
# summary(fit)      : returns coef table with t-statistic + pvalue

smry <- summary(fit)
coef_table <- smry$coefficients
cat("\n[run] haplo.glm coefficient table:\n")
print(coef_table)

# Extract intercept + per-haplotype rows.
# Row names look like "geno_setup.5" where 5 is the index into haplo.unique.
rn <- rownames(coef_table)
int_row_idx <- which(rn == "(Intercept)")
hap_row_idx <- grep("^geno_setup\\.", rn)

intercept_beta <- if (length(int_row_idx) == 1) coef_table[int_row_idx, 1] else NA_real_
intercept_se   <- if (length(int_row_idx) == 1) coef_table[int_row_idx, 2] else NA_real_
intercept_p    <- if (length(int_row_idx) == 1) coef_table[int_row_idx, 4] else NA_real_

hap_records <- list()
if (length(hap_row_idx) > 0) {
    for (i in hap_row_idx) {
        rn_i <- rn[i]
        # Parse the index out of "geno_setup.<idx>"
        idx_str <- sub("^geno_setup\\.", "", rn_i)
        if (idx_str == "rare") {
            tg <- "rare"
            letters_s <- "rare"
            freq <- NA_real_
        } else {
            hap_idx <- as.integer(idx_str)
            letters <- as.character(fit$haplo.unique[hap_idx, ])
            bits <- ifelse(letters == snp_a1, "0",
                     ifelse(letters == snp_a2, "1", NA))
            if (any(is.na(bits))) stop(sprintf("haplo.unique row %d has out-of-set letter", hap_idx))
            tg <- paste(bits, collapse = "")
            letters_s <- paste(letters, collapse = "")
            freq <- as.numeric(fit$haplo.freq[hap_idx])
        }
        hap_records[[length(hap_records) + 1]] <- list(
            term      = rn_i,
            haplo_idx = idx_str,
            tg_label  = tg,
            letters   = letters_s,
            freq      = freq,
            beta      = unname(coef_table[i, 1]),
            se        = unname(coef_table[i, 2]),
            tstat     = unname(coef_table[i, 3]),
            pval      = unname(coef_table[i, 4])
        )
    }
}

# --- Global LRT for haplotype effect ---------------------------------------
# haplo.glm reports the LRT of the full model vs null directly:
#   fit$lrt    : 2 * (lnlike_full - lnlike_null)
#   fit$lnlike / fit$lnlike.null are the underlying log-likelihoods.
# df = number of haplotype coefficients (length(hap_row_idx)).
df_diff <- length(hap_row_idx)
lrt_list <- fit$lrt
lrt_stat <- if (is.list(lrt_list) && !is.null(lrt_list$lrt)) as.numeric(lrt_list$lrt)[1] else as.numeric(lrt_list)[1]
lrt_df_pkg <- if (is.list(lrt_list) && !is.null(lrt_list$df)) as.numeric(lrt_list$df)[1] else NA_real_
if (is.null(lrt_stat) || length(lrt_stat) == 0) lrt_stat <- NA_real_
# Prefer haplo.glm reported df (fit$lrt$df); fall back to count of haplotype terms.
lrt_df <- if (!is.na(lrt_df_pkg) && lrt_df_pkg > 0) as.integer(lrt_df_pkg) else as.integer(df_diff)
lrt_p <- if (!is.na(lrt_stat) && lrt_df > 0) pchisq(lrt_stat, df = lrt_df, lower.tail = FALSE) else NA_real_
cat(sprintf("[run] global LRT (haplo.glm fit$lrt): stat=%.4f  df=%d  p=%.4e\n",
            lrt_stat, df_diff, lrt_p))

# Also compute the F-statistic from the haplo.glm residual sums of squares.
# For a Gaussian GLM, F = ((dev_null - dev_full) / df_diff) / (dev_full / df_resid).
dev_full <- as.numeric(fit$deviance)
dev_null <- as.numeric(fit$null.deviance)
df_resid <- as.numeric(fit$df.residual)
if (df_diff > 0 && !is.na(df_resid) && df_resid > 0 && dev_full > 0) {
    f_stat <- ((dev_null - dev_full) / df_diff) / (dev_full / df_resid)
    f_p    <- pf(f_stat, df1 = df_diff, df2 = df_resid, lower.tail = FALSE)
} else {
    f_stat <- NA_real_
    f_p    <- NA_real_
}
cat(sprintf("[run] global F-test (from haplo.glm deviances): F=%.4f  p=%.4e\n",
            f_stat, f_p))

# --- Compose JSON ----------------------------------------------------------
out <- list(
    window_id  = win$window_id,
    snp_ids    = snp_ids,
    snp_a1     = snp_a1,
    snp_a2     = snp_a2,
    snp_pos    = win$snp_pos,
    n_obs      = as.integer(n),
    phenotype  = win$phenotype,
    em_haplotypes = lapply(seq_len(H), function(h) list(
        tg_label = tg_label[h],
        letters  = hap_str[h],
        freq     = unname(hap_freq[h])
    )),
    em_n_haplotypes = H,
    em_sum_freq     = sum(hap_freq),
    reference_haplotype = list(
        tg_label    = ref_tg,
        letters     = ref_letters,
        haplo_idx   = ref_idx,
        freq        = unname(hap_freq[ref_idx])
    ),
    glm_intercept = list(
        beta = intercept_beta,
        se   = intercept_se,
        pval = intercept_p
    ),
    glm_haplotypes = hap_records,
    global_lrt = list(
        stat = unname(lrt_stat),
        df   = df_diff,
        pval = unname(lrt_p)
    ),
    global_anova_f = list(
        stat = unname(f_stat),
        pval = unname(f_p)
    ),
    package_versions = list(
        haplo.stats = as.character(packageVersion("haplo.stats")),
        jsonlite    = as.character(packageVersion("jsonlite")),
        R           = R.version.string,
        seed        = SEED
    )
)

out_path <- file.path(OUT_DIR, "haplo_results.json")
write_json(out, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 10)
cat(sprintf("\n[run] wrote %s\n", out_path))

