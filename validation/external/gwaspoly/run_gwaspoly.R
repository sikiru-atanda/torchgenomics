#!/usr/bin/env Rscript
# Run GWASpoly on the built-in tetraploid potato dataset to produce 8 reference
# outputs:
#   - gwaspoly_additive.csv
#   - gwaspoly_1_dom_alt.csv  + gwaspoly_1_dom_ref.csv
#   - gwaspoly_2_dom_alt.csv  + gwaspoly_2_dom_ref.csv
#   - gwaspoly_diplo_additive.csv
#   - gwaspoly_diplo_general.csv
#   - gwaspoly_general.csv
# Plus the kinship matrix and aligned data.
#
# Mirrors benchmark/run_gwaspoly.R but parameterized by data_dir + out_dir.

suppressPackageStartupMessages({
  local_lib <- Sys.getenv("RLIB", unset = NA)
  if (!is.na(local_lib) && nzchar(local_lib) && dir.exists(local_lib)) {
    .libPaths(c(local_lib, .libPaths()))
    # Propagate to forked parallel workers via R_LIBS_USER
    Sys.setenv(R_LIBS_USER = local_lib)
  }
  library(GWASpoly)
  library(rrBLUP)  # ensure mixed.solve loadable in parallel workers
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
if (is.na(data_dir) || is.na(out_dir)) {
  stop("usage: Rscript run_gwaspoly.R <data_dir> <out_dir>")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

genofile  <- file.path(data_dir, "new_potato_geno.csv")
phenofile <- file.path(data_dir, "new_potato_pheno.csv")
if (!file.exists(genofile) || !file.exists(phenofile)) {
  # Fallback: built-in package data
  genofile  <- system.file("extdata", "new_potato_geno.csv",  package = "GWASpoly")
  phenofile <- system.file("extdata", "new_potato_pheno.csv", package = "GWASpoly")
}
cat("[gwaspoly] geno:", genofile, "\n")
cat("[gwaspoly] pheno:", phenofile, "\n")

data <- read.GWASpoly(ploidy = 4, pheno.file = phenofile, geno.file = genofile,
                      format = "numeric", n.traits = 1, delim = ",")

N <- nrow(data@pheno)
cat(sprintf("[gwaspoly] %d samples, %d markers\n", N, nrow(data@map)))

cat("[gwaspoly] computing kinship (LOCO=FALSE)\n")
data.K <- set.K(data, LOCO = FALSE)

params <- set.params(geno.freq = 1 - 5/N, fixed = "env", fixed.type = "factor")
models <- c("additive", "1-dom", "2-dom", "diplo-additive", "diplo-general", "general")

cat(sprintf("[gwaspoly] running models: %s\n", paste(models, collapse = ", ")))
n_core <- as.integer(Sys.getenv("GWASPOLY_NCORES", unset = "1"))
result <- GWASpoly(data = data.K, models = models,
                   traits = c("vine.maturity"), params = params, n.core = n_core)

trait_scores <- result@scores[["vine.maturity"]]
all_models <- names(trait_scores)
map <- result@map

for (model_name in all_models) {
  scores_vec <- trait_scores[[model_name]]
  if (is.null(scores_vec)) next
  df <- data.frame(
    marker = map$Marker,
    chrom  = map$Chrom,
    pos    = map$Position,
    score  = scores_vec,
    stringsAsFactors = FALSE
  )
  df$pvalue <- 10^(-df$score)
  df <- df[!is.na(df$score), ]

  fname <- gsub("-", "_", model_name)
  out_path <- file.path(out_dir, sprintf("gwaspoly_%s.csv", fname))
  write.csv(df, out_path, row.names = FALSE)
  cat(sprintf("[gwaspoly] wrote %s (%d markers)\n", out_path, nrow(df)))
}

K <- data.K@K$all
write.csv(K, file.path(out_dir, "gwaspoly_kinship.csv"), row.names = TRUE)

# Aligned data exports (for compare.py)
geno_df <- data@geno
map_df <- data@map
pheno_df <- data@pheno
fixed_df <- data@fixed

pheno_out <- data.frame(
  id = rownames(pheno_df),
  vine.maturity = pheno_df[, "vine.maturity"],
  stringsAsFactors = FALSE
)
if (!is.null(fixed_df)) pheno_out$env <- fixed_df[, "env"]
write.csv(pheno_out, file.path(out_dir, "potato_pheno_aligned.csv"), row.names = FALSE)
write.csv(geno_df, file.path(out_dir, "potato_geno_aligned.csv"), row.names = TRUE)
map_out <- data.frame(
  marker = map_df$Marker,
  chrom  = map_df$Chrom,
  pos    = map_df$Position,
  stringsAsFactors = FALSE
)
write.csv(map_out, file.path(out_dir, "potato_map.csv"), row.names = FALSE)

cat("[gwaspoly] all reference outputs written\n")
