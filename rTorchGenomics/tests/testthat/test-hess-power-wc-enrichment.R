# Tests for tg_power / tg_winners_curse / tg_gene_set_enrichment / tg_hess
# wrappers + PowerRun / WinnersCurseRun / EnrichmentRun / HessRun S4 classes
# (Task 6).

test_that("tg_power forwards args and wraps into PowerRun", {
  captured <- NULL
  mockery::stub(tg_power, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(alpha = 5e-8, n = 5000, target_power = 0.8, n_variants = 3L,
         n_powered = 1L, results = list(), curve = list())
  })
  r <- tg_power("gwas.tsv", n = 5000, power_curve = TRUE)
  expect_equal(captured$fn, "power")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$n, 5000)
  expect_true(isTRUE(captured$args$power_curve))
  expect_s4_class(r, "PowerRun")
  expect_equal(r@n_variants, 3L)
  expect_equal(r@n_powered, 1L)
})

test_that("tg_winners_curse forwards args and wraps into WinnersCurseRun", {
  captured <- NULL
  mockery::stub(tg_winners_curse, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "conditional_likelihood", n_corrected = 2L, n_variants = 10L,
         results = list())
  })
  r <- tg_winners_curse("gwas.tsv", method = "conditional_likelihood")
  expect_equal(captured$fn, "winners_curse")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$method, "conditional_likelihood")
  expect_s4_class(r, "WinnersCurseRun")
  expect_equal(r@n_corrected, 2L)
})

test_that("tg_winners_curse forwards n_boot/seed only for bootstrap", {
  captured <- NULL
  mockery::stub(tg_winners_curse, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "bootstrap", n_corrected = 4L, n_variants = 10L, results = list())
  })
  r <- tg_winners_curse("gwas.tsv", method = "bootstrap", n_boot = 200L, seed = 1L)
  expect_equal(captured$args$n_boot, 200L)
  expect_equal(captured$args$seed, 1L)
  expect_s4_class(r, "WinnersCurseRun")
})

test_that("tg_gene_set_enrichment forwards args and wraps into EnrichmentRun", {
  captured <- NULL
  mockery::stub(tg_gene_set_enrichment, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(n_genes_total = 100L, n_gene_sets = 5L, n_significant = 1L,
         results = list(), genes = list())
  })
  r <- tg_gene_set_enrichment("gwas.tsv", "genes.tsv", "sets.gmt")
  expect_equal(captured$fn, "gene_set_enrichment")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$gene_annotation, "genes.tsv")
  expect_equal(captured$args$gene_sets, "sets.gmt")
  expect_s4_class(r, "EnrichmentRun")
  expect_equal(r@n_gene_sets, 5L)
})

test_that("tg_hess forwards args and wraps into HessRun", {
  captured <- NULL
  mockery::stub(tg_hess, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(mode = "h2", h2_total = 0.3, h2_total_se = 0.05, n_regions = 4L,
         n_snps_total = 400L, results = list())
  })
  r <- tg_hess("gwas.tsv", "ld.npy", "regions.tsv", n = 50000)
  expect_equal(captured$fn, "hess")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$ld_matrix, "ld.npy")
  expect_equal(captured$args$regions, "regions.tsv")
  expect_equal(captured$args$n, 50000)
  expect_s4_class(r, "HessRun")
  expect_equal(r@n_regions, 4L)
})

test_that("tg_hess forwards gwas2/n2 for local rg", {
  captured <- NULL
  mockery::stub(tg_hess, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(mode = "rg", h2_total = 0.1, h2_total_se = 0.02, n_regions = 4L,
         n_snps_total = 400L, results = list())
  })
  r <- tg_hess("gwas.tsv", "ld.npy", "regions.tsv", n = 50000, n2 = 40000,
              gwas2 = "gwas2.tsv")
  expect_equal(captured$args$gwas2, "gwas2.tsv")
  expect_equal(captured$args$n2, 40000)
  expect_equal(r@mode, "rg")
})
