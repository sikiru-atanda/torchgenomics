test_that("ScanRun can be constructed from a dict-like list", {
  dict <- list(
    runtime_s = 12.5,
    output_files = list(tsv = "/tmp/run/results.tsv"),
    log_excerpt = list(),
    model = "SingleTraitLMM",
    test = "wald",
    correction = "bh",
    n_variants = 500L,
    n_significant = 3L,
    significance_threshold = 5e-8,
    lambda_gc = 1.02,
    sigma2_g = 0.42,
    sigma2_e = 0.58,
    h2 = 0.42,
    n_samples = 250L,
    top_hits = list(
      list(CHR = "1", POS = 100, SNP = "rs1", P = 1e-9),
      list(CHR = "2", POS = 200, SNP = "rs2", P = 1e-7)
    )
  )
  r <- ScanRun$new_from_dict(dict)

  expect_s4_class(r, "ScanRun")
  expect_equal(r@n_variants, 500L)
  expect_equal(r@lambda_gc, 1.02)
  expect_s3_class(r@top_hits, "tbl_df")
  expect_equal(nrow(r@top_hits), 2L)
  expect_equal(r@top_hits$SNP, c("rs1", "rs2"))
})

test_that("show.ScanRun prints a multi-line summary", {
  r <- new("ScanRun",
           runtime_s = 1.0, output_files = list(),
           log_excerpt = character(0),
           model = "SingleTraitLMM", test = "wald",
           correction = "bh",
           n_variants = 100L, n_significant = 0L,
           significance_threshold = 5e-8,
           lambda_gc = NA_real_, sigma2_g = NA_real_,
           sigma2_e = NA_real_, h2 = NA_real_, n_samples = 0L,
           top_hits = tibble::tibble())
  out <- capture.output(show(r))
  expect_true(length(out) > 3L)
  expect_true(any(grepl("SingleTraitLMM", out)))
  expect_true(any(grepl("Variants tested: 100", out)))
})

test_that("ValidateRun construction + accessor", {
  dict <- list(
    runtime_s = 0.5,
    output_files = list(),
    log_excerpt = list(),
    ok = TRUE,
    format_name = "bed",
    n_samples_genotype = 10L,
    n_samples_phenotype = 12L,
    n_samples_aligned = 10L,
    n_variants_total = 20L,
    n_traits = 2L,
    ploidy = 2L,
    genotype_missingness_pct = 0.5,
    phenotype_missingness_pct = 4.0,
    warnings = list(),
    errors = list()
  )
  r <- ValidateRun$new_from_dict(dict)
  expect_s4_class(r, "ValidateRun")
  expect_true(r@ok)
  expect_equal(r@n_variants_total, 20L)
})

test_that("GwasResult (tier-2 generic) holds arbitrary fields", {
  dict <- list(
    runtime_s = 7.5,
    output_files = list(tsv = "/tmp/x.tsv"),
    log_excerpt = list(),
    summary = "MVLMM scan: 100 variants, 0 significant",
    n_variants = 100L,
    top_hits = list()
  )
  r <- GwasResult$new_from_dict(dict, command = "mvlmm_scan")
  expect_s4_class(r, "GwasResult")
  expect_equal(r@command, "mvlmm_scan")
  expect_equal(r@n_variants, 100L)
  expect_match(r@summary_text, "MVLMM scan")
})
