# Tests for tg_smr / tg_mr_mega wrappers + SMRRun / MRMegaRun S4 classes (Task 5).

test_that("tg_smr forwards args and wraps into SMRRun", {
  captured <- NULL
  mockery::stub(tg_smr, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(n_genes_tested = 5L, n_significant_smr = 2L, n_pass_heidi = 1L, results = list())
  })
  r <- tg_smr("gwas.tsv", "eqtl.tsv", "map.tsv")
  expect_equal(captured$fn, "smr")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$gene_map, "map.tsv")
  expect_s4_class(r, "SMRRun")
})

test_that("tg_mr_mega forwards a sumstats vector and wraps into MRMegaRun", {
  captured <- NULL
  mockery::stub(tg_mr_mega, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "mr_mega", n_axes = 1L, n_populations = 3L, n_snps = 40L,
         min_p_meta = 1e-8, results = list())
  })
  r <- tg_mr_mega(c("p1.tsv", "p2.tsv", "p3.tsv"), n_axes = 1L)
  expect_equal(captured$fn, "mr_mega")
  expect_equal(captured$args$sumstats, c("p1.tsv", "p2.tsv", "p3.tsv"))
  expect_s4_class(r, "MRMegaRun")
})
