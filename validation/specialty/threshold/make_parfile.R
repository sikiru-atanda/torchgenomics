#!/usr/bin/env Rscript
## make_parfile.R -- emit a BLUPF90+ parameter file for the 3-trait
## threshold-linear model from validation/specialty/threshold/.
##
## BLUPF90+ parameter file layout (Misztal et al. manual section 4.1):
##   DATAFILE
##     <path>
##   NUMBER_OF_TRAITS
##     <c>
##   NUMBER_OF_EFFECTS
##     <k>
##   OBSERVATION(S)
##     <cols ...>
##   WEIGHT(S)
##     <zeros>
##   EFFECTS:
##     <ncol1> <levels1> <type1> [traits...]
##     ...
##   RANDOM_RESIDUAL VALUES
##     <R as c-by-c lower-triangle on c lines>
##   RANDOM_GROUP
##     <which effect>
##   RANDOM_TYPE
##     add_animal
##   FILE
##     <pedigree path>
##   (CO)VARIANCES
##     <G as c-by-c>
##   OPTION cat <n1> <n2> ... <nc>
##   OPTION fix_residual_VAR
##   OPTION fix_var_genetic   (BLUPF90 calls it fix_genetic, manual section 8.7)
##
## Reference: BLUPF90+ User Manual (Misztal, Lourenco, Aguilar, Tsuruta;
## Lourenco et al. 2022).  Threshold-mode is activated by OPTION cat with
## any integer >0 in any trait slot.  Continuous traits use 0.

suppressPackageStartupMessages({
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
        stop("jsonlite not installed (CRAN); install with R -e install.packages(\"jsonlite\")")
    }
})
library(jsonlite)

## --- CLI parsing (no external optparse dependency) ---
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, args) {
    idx <- match(flag, args)
    if (is.na(idx) || idx == length(args)) {
        stop(sprintf("missing value for %s", flag))
    }
    args[idx + 1]
}
truth_path <- get_arg("--truth", args)
pheno_path <- get_arg("--pheno", args)
ped_path   <- get_arg("--pedigree", args)
out_path   <- get_arg("--out", args)
## --pheno is the path that goes into the parameter file (BLUPF90 sees it
## with this string as DATAFILE).  --pheno-data is the actual on-disk
## path we read here to count effect levels.  They differ when we stage
## phenotypes into OUT_DIR via symlink and want the par-file to be portable.
pheno_data <- tryCatch(get_arg("--pheno-data", args), error = function(e) pheno_path)

truth <- fromJSON(truth_path, simplifyVector = TRUE)
R     <- matrix(unlist(truth$R), nrow = truth$c, byrow = TRUE)
G_cov <- matrix(unlist(truth$G_cov), nrow = truth$c, byrow = TRUE)
ncats <- truth$n_categories

## Format a c-by-c covariance matrix on c lines, lower-triangle conformant
## (BLUPF90 accepts square form too -- safer for reproducibility).
fmt_cov <- function(M) {
    apply(M, 1, function(row) {
        paste(formatC(row, format = "f", digits = 6, width = 12), collapse = " ")
    })
}

## EFFECTS block:
##   col1: column 2 in pheno -- intercept (single class)  -> cross 1 each trait
##   col2: column 3 in pheno -- sex (class)               -> cross 2 each trait
##   col3: column 1 in pheno -- animal ID (random)        -> n_animals each trait
## All three effects apply to all three traits.
##
## We need the number of distinct levels for each class effect.  Read
## pheno.txt to count.
pheno <- read.table(pheno_data, header = FALSE)
colnames(pheno) <- c("id", "intercept", "sex", "y1", "y2", "y3")
n_animals <- length(unique(pheno$id))
n_sex     <- length(unique(pheno$sex))

obs_cols <- c(4, 5, 6)
## Multi-trait EFFECTS layout: <pos_trait1> <pos_trait2> ... <pos_traitC> <nlevels> <type>
## All three effects (intercept, sex, animal) share the same column across traits.
pos_intercept <- paste(rep(2L, truth$c), collapse = " ")
pos_sex       <- paste(rep(3L, truth$c), collapse = " ")
pos_animal    <- paste(rep(1L, truth$c), collapse = " ")
effects_block <- c(
    sprintf("%s %d cross", pos_intercept, 1L),
    sprintf("%s %d cross", pos_sex,       as.integer(n_sex)),
    sprintf("%s %d cross", pos_animal,    as.integer(n_animals))
)

par_lines <- c(
    "DATAFILE",
    pheno_path,
    "NUMBER_OF_TRAITS",
    sprintf("%d", truth$c),
    "NUMBER_OF_EFFECTS",
    "3",
    "OBSERVATION(S)",
    paste(obs_cols, collapse = " "),
    "WEIGHT(S)",
    "",
    "EFFECTS: POSITIONS_IN_DATAFILE NUMBER_OF_LEVELS TYPE_OF_EFFECT [EFFECT NESTED]",
    effects_block,
    "RANDOM_RESIDUAL VALUES",
    fmt_cov(R),
    "RANDOM_GROUP",
    "3",
    "RANDOM_TYPE",
    "add_animal",
    "FILE",
    ped_path,
    "(CO)VARIANCES",
    fmt_cov(G_cov),
    sprintf("OPTION cat %s", paste(ncats, collapse = " ")),
    "OPTION fix_threshold",
    "OPTION fix_residual_var",
    "OPTION fix_var_genetic",
    "OPTION sol mean",
    "OPTION blocksize_per_effect 1"
)
writeLines(par_lines, out_path)
cat(sprintf("[make_parfile] wrote %s (%d lines)\n", out_path, length(par_lines)))

