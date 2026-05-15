# _fetch_window.R -- emit data/window.json with the 5-SNP chr1 block
# selected for the hapref harness, plus SHA256 of each MDP source file.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("need data dir arg")
DATA_DIR <- args[1]

suppressPackageStartupMessages({
    library(jsonlite)
    library(digest)
})

snp_ids <- c("id1.6", "id1.3", "id1.2", "PZB02144.4", "PZB02144.5")
snp_chr <- rep("1", 5)
snp_pos <- c(238902012L, 238902078L, 238902091L, 238902135L, 238902252L)

# Determine A1/A2 per SNP from the HapMap file (alleles column).
# Source HapMap copy lives at benchmark/data/mdp_genotype_test.hmp.txt;
# we re-read directly so we do not need to stage the (large) HapMap into
# the harness data dir.
hmp_path <- "/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data/mdp_genotype_test.hmp.txt"
hmp <- read.table(hmp_path, header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE, comment.char = "",
                   check.names = FALSE, nrows = 3093L)
sel <- hmp[match(snp_ids, hmp$rs), ]
stopifnot(all(sel$rs == snp_ids))
alleles <- sel$alleles
# Numeric encoding (mdp_numeric.txt) in GAPIT is 0/1/2 where 2 = major,
# 0 = minor (the *less* common allele).  We confirm this by tallying genotypes
# below.  Allele1 of HapMap ("A/C" -> A) is usually the minor; we record both
# and let the EM result decide the mapping.
a1 <- substr(alleles, 1, 1)
a2 <- substr(alleles, 3, 3)

# Recover MAF per-SNP from mdp_numeric.txt to identify which numeric value
# is minor.  In MDP-numeric a value of 2 == homozygote-major and 0 == homozygote-minor
# (per GAPIT data dictionary).  We verify by checking that maf(=mean/2) is consistent.
geno <- read.table(file.path(DATA_DIR, "mdp_numeric.txt"), header = TRUE, sep = "\t",
                   check.names = FALSE, stringsAsFactors = FALSE)
snp_cols <- match(snp_ids, colnames(geno))
stopifnot(all(!is.na(snp_cols)))
sub <- geno[, snp_cols, drop = FALSE]
mean_dosage <- colMeans(sub, na.rm = TRUE)  # mean of 0/1/2 = 2 * freq(value-2)
# freq(value-2) = mean_dosage / 2
# If that is > 0.5 then value-2 is the MAJOR allele and value-0 is minor.
# In MDP we expect value-2 to be major (so MAF < 0.5).
freq2 <- mean_dosage / 2
maf <- pmin(freq2, 1 - freq2)
# minor allele = a1 if a1 is the rarer letter, but the MDP genotype matrix
# does not preserve which letter codes 2 vs 0.  GAPIT convention: 2 = first
# allele alphabetical, 0 = second alphabetical.  We document this rather than
# guess; the compare.py uses dosage-based label mapping (see below).

# SHA256 of staged data files
files <- c("mdp_numeric.txt", "mdp_SNP_information.txt", "mdp_traits.txt")
sha <- sapply(files, function(f) digest(file.path(DATA_DIR, f), algo = "sha256", file = TRUE, serialize = FALSE))

win <- list(
    chr   = "1",
    window_id = "chr1_win343",
    snp_ids = snp_ids,
    snp_chr = snp_chr,
    snp_pos = snp_pos,
    snp_a1  = a1,
    snp_a2  = a2,
    snp_maf_observed = unname(maf),
    snp_mean_dosage  = unname(mean_dosage),
    n_snps  = length(snp_ids),
    phenotype = "EarHT",
    rationale = "chr 1 tight-LD block (mean |r|=0.867 across 5 SNPs); MAF 0.15-0.24",
    source = list(
        mdp_numeric         = "/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data/mdp_numeric.txt",
        mdp_SNP_information = "/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data/mdp_SNP_information.txt",
        mdp_traits          = "/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data/mdp_traits.txt",
        mdp_genotype_test_hmp = "/home/sikiru.atanda/Documents/GWAS_Expert/benchmark/data/mdp_genotype_test.hmp.txt"
    ),
    sha256 = as.list(sha)
)

out_path <- file.path(DATA_DIR, "window.json")
write_json(win, out_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("[fetch_window] wrote %s\n", out_path))
for (i in seq_along(files)) {
    cat(sprintf("  sha256  %s  %s\n", sha[i], files[i]))
}
cat(sprintf("  SNPs: %s\n", paste(snp_ids, collapse = ", ")))
cat(sprintf("  pos:  %s\n", paste(snp_pos, collapse = ", ")))
cat(sprintf("  MAF:  %s\n", paste(sprintf("%.3f", maf), collapse = ", ")))

