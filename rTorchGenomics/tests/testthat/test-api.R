# Tests for tier-1 api wrappers (Task 5).
#
# Note: this task ships 11 wrappers. The remaining two plot-based
# wrappers (`tg_manhattan`, `tg_qq`) land in Task 7 (R/plots.R) since
# they read result files from disk and build ggplots. The exports
# sentinel below tracks only the 11 names defined here; extend it
# when Task 7 adds the plotting helpers.

test_that("all 13 tier-1 functions are exported", {
  expected <- c(
    "tg_validate", "tg_convert", "tg_impute",
    "tg_lmm_scan", "tg_glm_scan",
    "tg_ld_blocks", "tg_clump", "tg_meta",
    "tg_pgs_fit", "tg_pgs_score",
    "tg_annotate_hits",
    "tg_lgebv", "tg_iclass"
  )
  ns <- getNamespace("rTorchGenomics")
  for (fn in expected) {
    expect_true(exists(fn, envir = ns, inherits = FALSE), info = fn)
  }
})

test_that("tg_validate dispatches via bridge_call and wraps result", {
  mockery::stub(tg_validate, "bridge_call", function(fn, args) {
    expect_equal(fn, "validate")
    expect_equal(args$genotype, "g.bed")
    expect_equal(args$phenotype, "p.tsv")
    list(
      runtime_s = 0.5, output_files = list(), log_excerpt = list(),
      ok = TRUE, format_name = "bed",
      n_samples_genotype = 10L, n_samples_phenotype = 12L,
      n_samples_aligned = 10L, n_variants_total = 20L,
      n_traits = 2L, ploidy = 2L,
      genotype_missingness_pct = 0.5, phenotype_missingness_pct = 4.0,
      warnings = list(), errors = list()
    )
  })

  r <- tg_validate(genotype = "g.bed", phenotype = "p.tsv")
  expect_s4_class(r, "ValidateRun")
  expect_true(r@ok)
  expect_equal(r@n_variants_total, 20L)
})

test_that("tg_lmm_scan passes typed defaults through to bridge_call", {
  mockery::stub(tg_lmm_scan, "bridge_call", function(fn, args) {
    expect_equal(fn, "lmm_scan")
    expect_equal(args$test, "wald")
    expect_equal(args$correction, "bh")
    expect_equal(args$chunk_size, 10000L)
    list(
      runtime_s = 1.0, output_files = list(),
      log_excerpt = list(), model = "SingleTraitLMM", test = "wald",
      correction = "bh", n_variants = 100L, n_significant = 0L,
      significance_threshold = 5e-8, lambda_gc = NA_real_,
      sigma2_g = NA_real_, sigma2_e = NA_real_, h2 = NA_real_,
      n_samples = 0L, top_hits = list()
    )
  })

  r <- tg_lmm_scan(genotype = "g.bed", phenotype = "p.tsv")
  expect_s4_class(r, "ScanRun")
  expect_equal(r@model, "SingleTraitLMM")
})

test_that("tg_ld_blocks passes tolerance through to the bridge", {
  captured <- new.env()
  mockery::stub(tg_ld_blocks, "bridge_call", function(fn, args) {
    captured$fn <- fn
    captured$args <- args
    list(
      runtime_s = 0.2, output_files = list(), log_excerpt = list(),
      method = "r2", n_blocks = 4L, n_variants_in_blocks = 30L,
      median_block_size_bp = 1000, median_block_n_snps = 7,
      blocks = list()
    )
  })
  r <- tg_ld_blocks(genotype = "g.bed", method = "r2",
                    r2_threshold = 0.7, tolerance = 2)
  expect_s4_class(r, "LDBlocksRun")
  expect_equal(captured$fn, "ld_blocks")
  expect_equal(captured$args$method, "r2")
  expect_equal(captured$args$tolerance, 2L)
})

test_that("tg_ld_blocks tolerance default is 0L", {
  captured <- new.env()
  mockery::stub(tg_ld_blocks, "bridge_call", function(fn, args) {
    captured$args <- args
    list(
      runtime_s = 0.2, output_files = list(), log_excerpt = list(),
      method = "r2", n_blocks = 1L, n_variants_in_blocks = 5L,
      median_block_size_bp = 500, median_block_n_snps = 5,
      blocks = list()
    )
  })
  r <- tg_ld_blocks(genotype = "g.bed", method = "r2")
  expect_equal(captured$args$tolerance, 0L)
})

test_that("tg_ld_blocks rejects negative tolerance", {
  expect_error(
    tg_ld_blocks(genotype = "g.bed", method = "r2", tolerance = -1L),
    "non-negative integer"
  )
})

test_that("tg_pgs_fit forwards method + n_iter", {
  mockery::stub(tg_pgs_fit, "bridge_call", function(fn, args) {
    expect_equal(fn, "pgs_fit")
    expect_equal(args$method, "ldpred2-auto")
    list(
      runtime_s = 5.0, output_files = list(),
      log_excerpt = list(), method = "ldpred2-auto",
      n_variants_input = 1000L, n_variants_used = 950L,
      n_variants_with_weights = 950L, h2 = 0.3,
      converged = TRUE, diagnostics = list()
    )
  })

  r <- tg_pgs_fit(
    sumstats = "ss.tsv", ld_ref = "ld.pt",
    output = "out/weights.tsv", method = "ldpred2-auto"
  )
  expect_s4_class(r, "PgsFitRun")
  expect_true(r@converged)
})
