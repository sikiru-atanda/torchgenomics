#' Integration test: real Python venv + real torchgenomics.
#'
#' Opt-in via `Sys.setenv(RTORCHGENOMICS_INTEGRATION = "1")`. Assumes
#' the user has already run `tg_install()` (or the CI runner has the
#' r-rtorchgenomics venv prepared).

skip_if_not_integration <- function() {
  if (Sys.getenv("RTORCHGENOMICS_INTEGRATION", unset = "") != "1") {
    skip("Integration test — set RTORCHGENOMICS_INTEGRATION=1 to enable.")
  }
  if (!reticulate::virtualenv_exists("r-rtorchgenomics")) {
    skip("r-rtorchgenomics venv not found; run tg_install() first.")
  }
}

test_that("[integration] tg_validate on the tiny fixture", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")
  r <- tg_validate(genotype = bed, phenotype = pheno)
  expect_s4_class(r, "ValidateRun")
  expect_equal(r@format_name, "bed")
  expect_equal(r@n_samples_genotype, 10L)
  expect_equal(r@n_variants_total, 20L)
})

test_that("[integration] tg_lmm_scan returns a populated ScanRun", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")
  tmp <- tempfile("tg-lmm-")
  dir.create(tmp)

  r <- tg_lmm_scan(
    genotype = bed, phenotype = pheno,
    output = tmp, chunk_size = 10L, maf_min = 0.0
  )
  expect_s4_class(r, "ScanRun")
  expect_equal(r@model, "SingleTraitLMM")
  expect_true(r@n_variants > 0L)
  expect_true(file.exists(r@output_files$tsv) ||
              file.exists(r@output_files$parquet))
})

test_that("[integration] tg_manhattan builds from a real ScanRun", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")
  tmp <- tempfile("tg-lmm-")
  dir.create(tmp)
  r <- tg_lmm_scan(genotype = bed, phenotype = pheno,
                   output = tmp, chunk_size = 10L, maf_min = 0.0)
  p <- tg_manhattan(r)
  expect_s3_class(p, "ggplot")
})
