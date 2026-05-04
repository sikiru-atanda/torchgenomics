#!/usr/bin/env Rscript
# Run GAPIT3 on the MDP fixture to produce reference outputs:
#
#   - GLM_GWAS.csv      via model="GLM"
#   - MLM_GWAS.csv      via model="MLM"
#   - FarmCPU_GWAS.csv  via model="FarmCPU"
#   - BLINK_GWAS.csv    via model="BLINK"
#
# Phenotype: EarHT (fewest missing values on the MDP fixture).
#
# Invoked by run_gapit.sh; see validation/external/gapit/README.md.

suppressPackageStartupMessages({
  local_lib <- Sys.getenv("RLIB", unset = NA)
  if (!is.na(local_lib) && nzchar(local_lib) && dir.exists(local_lib)) {
    .libPaths(c(local_lib, .libPaths()))
  }
  if (requireNamespace("GAPIT3", quietly = TRUE)) {
    library(GAPIT3)
  } else {
    library(GAPIT)
  }
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
if (is.na(data_dir) || is.na(out_dir)) {
  stop("usage: Rscript run_gapit.R <data_dir> <out_dir>")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
setwd(out_dir)

cat("[gapit] data_dir:", data_dir, "\n")
cat("[gapit] out_dir:", out_dir, "\n")

myY  <- read.table(file.path(data_dir, "mdp_traits.txt"), head = TRUE)
myGD <- read.table(file.path(data_dir, "mdp_numeric.txt"), head = TRUE)
myGM <- read.table(file.path(data_dir, "mdp_SNP_information.txt"), head = TRUE)

cat("[gapit] phenotype:", dim(myY), "\n")
cat("[gapit] genotype:",  dim(myGD), "\n")
cat("[gapit] map:",       dim(myGM), "\n")

myY_single <- myY[, c("Taxa", "EarHT")]

run_one <- function(model_name) {
  cat(sprintf("[gapit] running %s...\n", model_name))
  res <- tryCatch({
    GAPIT(Y = myY_single, GD = myGD, GM = myGM,
          model = model_name, file.output = FALSE)
  }, error = function(e) {
    cat(sprintf("[gapit] %s ERROR: %s\n", model_name, conditionMessage(e)))
    NULL
  })
  if (!is.null(res) && !is.null(res$GWAS)) {
    out_path <- file.path(out_dir, sprintf("%s_GWAS.csv", model_name))
    write.csv(res$GWAS, out_path, row.names = FALSE)
    cat(sprintf("[gapit] wrote %s\n", out_path))
  }
}

for (m in c("GLM", "MLM", "FarmCPU", "BLINK")) {
  run_one(m)
}

cat("[gapit] all runs complete\n")
