# Regenerate GAPIT3 / GWASpoly reference outputs on the MDP + potato fixtures.
#
# Companion to scripts/generate_golden_data.py. Runs the R-based reference
# tools (GAPIT3, GWASpoly) against the same MDP / potato data used by the
# TorchGWAS test suite and writes outputs into the locations the golden tests
# look for:
#
#   * gapit_demo/output/mdp_farmcpu.GWAS.Results.csv
#   * gapit_demo/output/mdp_blink.GWAS.Results.csv
#   * benchmark/gwaspoly_results/gwaspoly_*.csv  (already present; regen only)
#
# Usage:
#   Rscript scripts/generate_golden_data.R --gapit
#   Rscript scripts/generate_golden_data.R --gwaspoly
#   Rscript scripts/generate_golden_data.R --all
#
# Prerequisites:
#   * R >= 4.2
#   * GAPIT3 installed (see https://zzlab.net/GAPIT/)
#   * GWASpoly installed (see https://github.com/jendelman/GWASpoly)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  cat("Usage: Rscript scripts/generate_golden_data.R [--gapit|--gwaspoly|--all]\n")
  quit(status = 0)
}

run_gapit <- "--gapit" %in% args || "--all" %in% args
run_gwaspoly <- "--gwaspoly" %in% args || "--all" %in% args

repo_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), ".."))
data_dir <- file.path(repo_root, "benchmark", "data")

if (run_gapit) {
  cat("[GAPIT3] Regenerating FarmCPU + BLINK outputs on MDP...\n")
  suppressWarnings(dir.create(file.path(repo_root, "gapit_demo", "output"),
                              recursive = TRUE, showWarnings = FALSE))
  setwd(file.path(repo_root, "gapit_demo", "output"))

  if (!requireNamespace("GAPIT", quietly = TRUE)) {
    stop("GAPIT3 not installed. See https://zzlab.net/GAPIT/")
  }
  library(GAPIT)

  pheno <- read.table(file.path(data_dir, "mdp_traits.txt"),
                      header = TRUE, sep = "\t")
  geno <- read.table(file.path(data_dir, "mdp_numeric.txt"),
                     header = TRUE, sep = "\t")
  snpmap <- read.table(file.path(data_dir, "mdp_SNP_information.txt"),
                       header = TRUE, sep = "\t")

  pheno_sub <- pheno[, c("Taxa", "EarHT")]
  pheno_sub <- pheno_sub[!is.na(pheno_sub$EarHT), ]

  # FarmCPU
  GAPIT(Y = pheno_sub, GD = geno, GM = snpmap, model = "FarmCPU",
        file.output = TRUE, output.numerical = FALSE)
  # BLINK
  GAPIT(Y = pheno_sub, GD = geno, GM = snpmap, model = "BLINK",
        file.output = TRUE, output.numerical = FALSE)

  # Normalize output filenames to what the golden tests expect.
  farmcpu_src <- Sys.glob("GAPIT.FarmCPU.*.GWAS.Results.csv")
  blink_src <- Sys.glob("GAPIT.BLINK.*.GWAS.Results.csv")
  if (length(farmcpu_src) > 0) {
    file.copy(farmcpu_src[1], "mdp_farmcpu.GWAS.Results.csv", overwrite = TRUE)
  }
  if (length(blink_src) > 0) {
    file.copy(blink_src[1], "mdp_blink.GWAS.Results.csv", overwrite = TRUE)
  }
  cat("[GAPIT3] Done.\n")
}

if (run_gwaspoly) {
  cat("[GWASpoly] Regenerating tetraploid potato outputs...\n")
  if (!requireNamespace("GWASpoly", quietly = TRUE)) {
    stop("GWASpoly not installed. See https://github.com/jendelman/GWASpoly")
  }
  stop(
    "GWASpoly regeneration is dataset-specific and already committed at ",
    "benchmark/gwaspoly_results/. Edit this block only when the upstream ",
    "data or GWASpoly version changes. See docs/validation_protocol.md."
  )
}
