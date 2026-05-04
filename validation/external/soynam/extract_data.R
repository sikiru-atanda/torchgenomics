#!/usr/bin/env Rscript
# Extract SoyNAM genotype + phenotype + family TSVs for the harness.
#
# Strategy: use SoyNAM::BLUP() to get the canonical preprocessed view of
# the panel — phenotype BLUPs, imputed genotypes, family vector, and
# chromosome layout. We subset to a manageable family set (default 4
# families × ~140 RILs each ≈ 560 RILs) so the full TG vs rrBLUP
# comparison runs in <5 minutes wall time. Larger family sets are still
# tractable but eat more memory in `rrBLUP::A.mat` + `rrBLUP::GWAS`.
#
# We do NOT use the full 40-family / 5590-RIL panel by default; the
# rrBLUP::GWAS reference takes >30 min on that scale and saturates the
# user's RAM at ~20 GB. The 4-family subset preserves the core property
# (multi-family RIL design with a common parent) while staying inside
# the harness's memory pre-flight envelope.
#
# Output layout (under data/):
#   geno.tsv         — n_RIL × n_SNP, rows = strain ID, cols = SNP, dosage 0/1/2 (no NA after imputation)
#   pheno.tsv        — strain, family, yield_blup
#   family.tsv       — strain, family
#   marker_map.tsv   — snp, chr, idx (BLUP() exposes 1-based chrom break indices, which we expand to a per-SNP chr label)
#   data_manifest.json — summary + sha256 of each TSV

suppressPackageStartupMessages({
    library(SoyNAM)
    library(jsonlite)
})

# --- CLI args ----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

OUT_DIR     <- parse_arg("--output-dir", "data")
TRAIT       <- parse_arg("--trait", "yield")
FAMILY_LIST <- parse_arg("--families", "2,3,4,5")  # comma-sep family codes
MAF         <- as.numeric(parse_arg("--maf", "0.05"))
IMPUTE      <- parse_arg("--impute", "FM")          # forward-marker (default rrBLUP-friendly)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

fams <- as.integer(strsplit(FAMILY_LIST, ",")[[1]])
cat(sprintf("[extract_data] trait=%s families=%s maf=%.2f impute=%s\n",
            TRAIT, paste(fams, collapse=","), MAF, IMPUTE))

# --- Build canonical view via BLUP() -----------------------------------------
# BLUP() does:
#   1. Subsets data.line to the requested families.
#   2. Computes BLUPs for the requested trait (random RIL effect within
#      check-corrected linear model; broad-sense H2 reported).
#   3. Imputes the genotype matrix and removes redundant + low-MAF SNPs.
#   4. Returns Phen (n,), Gen (n × m), Chrom (chromosome break indices),
#      Fam (n,), r2 (per-SNP rep count), nReps (per-strain count).
res <- BLUP(
    trait     = TRAIT,
    family    = fams,
    MAF       = MAF,
    use.check = TRUE,
    impute    = IMPUTE,
    rm.rep    = TRUE
)

stopifnot(is.list(res))
stopifnot(all(c("Phen", "Gen", "Chrom", "Fam") %in% names(res)))

n_ril <- length(res$Phen)
n_snp <- ncol(res$Gen)
cat(sprintf("[extract_data] BLUP() returned %d RILs × %d SNPs across %d families\n",
            n_ril, n_snp, length(unique(res$Fam))))

# Sanity: rownames(Gen) should be strain IDs
strains <- rownames(res$Gen)
stopifnot(length(strains) == n_ril)
stopifnot(!any(is.na(res$Gen)))

# --- Build per-SNP chromosome labels ----------------------------------------
# `Chrom` is a length-20 integer vector with the per-chromosome SNP count
# (Soybean has 20 chromosomes / Gm01..Gm20). We convert this to a per-SNP
# chromosome label by replicating Gm01..Gm20 by the corresponding count.
stopifnot(length(res$Chrom) == 20)
stopifnot(sum(res$Chrom) == n_snp)

chr_labels <- unlist(mapply(
    function(k, n) rep(sprintf("Gm%02d", k), n),
    seq_len(20), as.integer(res$Chrom),
    SIMPLIFY = FALSE
), use.names = FALSE)
stopifnot(length(chr_labels) == n_snp)
stopifnot(!any(chr_labels == ""))

# Try to recover physical positions from the SNP names (format Gm01_<bp>_REF_ALT).
snp_names <- colnames(res$Gen)
parsed <- strsplit(snp_names, "_")
positions <- vapply(parsed, function(p) {
    if (length(p) >= 2) suppressWarnings(as.integer(p[2])) else NA_integer_
}, integer(1))
positions[is.na(positions)] <- seq_along(positions)[is.na(positions)]

# --- Write geno.tsv ----------------------------------------------------------
# Format: strain, then one column per SNP (dosage 0/1/2). Header includes
# strain + SNP IDs.
geno_path <- file.path(OUT_DIR, "geno.tsv")
write.table(
    cbind(strain = strains, as.data.frame(res$Gen, stringsAsFactors = FALSE)),
    file = geno_path, sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[extract_data] wrote %s (%d × %d)\n", geno_path, n_ril, n_snp + 1))

# --- Write pheno.tsv ---------------------------------------------------------
pheno_path <- file.path(OUT_DIR, "pheno.tsv")
write.table(
    data.frame(
        strain = strains,
        family = sprintf("F%02d", as.integer(res$Fam)),
        y_blup = as.numeric(res$Phen),
        stringsAsFactors = FALSE
    ),
    file = pheno_path, sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[extract_data] wrote %s (%d rows)\n", pheno_path, n_ril))

# --- Write family.tsv (clean two-col view) ----------------------------------
family_path <- file.path(OUT_DIR, "family.tsv")
write.table(
    data.frame(
        strain = strains,
        family = sprintf("F%02d", as.integer(res$Fam)),
        stringsAsFactors = FALSE
    ),
    file = family_path, sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[extract_data] wrote %s (%d rows)\n", family_path, n_ril))

# --- Write marker_map.tsv ----------------------------------------------------
map_path <- file.path(OUT_DIR, "marker_map.tsv")
write.table(
    data.frame(
        snp = snp_names,
        chr = chr_labels,
        pos = positions,
        idx = seq_len(n_snp),
        stringsAsFactors = FALSE
    ),
    file = map_path, sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[extract_data] wrote %s (%d rows)\n", map_path, n_snp))

# --- Manifest with sha256 + summary ------------------------------------------
sha256_of <- function(path) {
    if (!file.exists(path)) return(NA_character_)
    proc <- system2("sha256sum", path, stdout = TRUE)
    strsplit(proc, "\\s+")[[1]][1]
}

manifest <- list(
    trait      = TRAIT,
    families   = fams,
    n_ril      = n_ril,
    n_snp      = n_snp,
    n_chr      = length(unique(chr_labels)),
    maf_filter = MAF,
    impute     = IMPUTE,
    files = list(
        geno      = list(path = "geno.tsv",       sha256 = sha256_of(geno_path)),
        pheno     = list(path = "pheno.tsv",      sha256 = sha256_of(pheno_path)),
        family    = list(path = "family.tsv",     sha256 = sha256_of(family_path)),
        marker_map= list(path = "marker_map.tsv", sha256 = sha256_of(map_path))
    ),
    package_versions = list(
        SoyNAM = as.character(packageVersion("SoyNAM")),
        R      = R.version.string
    )
)
manifest_path <- file.path(OUT_DIR, "data_manifest.json")
write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("[extract_data] wrote %s\n", manifest_path))

cat("[extract_data] done.\n")
