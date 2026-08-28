# Tests for tg_finemap wrapper + FineMapRun S4 class (Task 4).

test_that("tg_finemap forwards args and wraps into FineMapRun", {
  captured <- NULL
  mockery::stub(tg_finemap, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(n_variants = 5L, n_credible_sets = 1L, n_variants_in_credible_sets = 2L,
         results = list(), credible_sets = list())
  })
  r <- tg_finemap("ss.tsv", ld_ref = "ld.pt", max_num_causal = 2L)
  expect_equal(captured$fn, "finemap")
  expect_equal(captured$args$sumstats, "ss.tsv")
  expect_equal(captured$args$ld_ref, "ld.pt")
  expect_s4_class(r, "FineMapRun")
})
