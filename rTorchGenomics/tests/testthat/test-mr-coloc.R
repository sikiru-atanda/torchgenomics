# Tests for tg_mr / tg_coloc wrappers + MRRun / ColocRun S4 classes (Task 5).

test_that("tg_mr forwards args and wraps the result", {
  captured <- NULL
  mockery::stub(tg_mr, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "all", n_instruments = 25L, primary_beta = 0.42,
         primary_p = 1e-6, results = list())
  })
  r <- tg_mr("exp.tsv", "out.tsv", method = "all")
  expect_equal(captured$fn, "mr")
  expect_equal(captured$args$exposure, "exp.tsv")
  expect_equal(captured$args$method, "all")
  expect_s4_class(r, "MRRun")
  expect_equal(r@n_instruments, 25L)
  expect_equal(r@primary_beta, 0.42)
})

test_that("tg_coloc forwards args and wraps the result", {
  captured <- NULL
  mockery::stub(tg_coloc, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "pairwise", pp = list(h0 = 0.01, h4 = 0.92),
         candidate_snp = 137L, headline = 0.92, table = list())
  })
  r <- tg_coloc("a.tsv", "b.tsv", method = "pairwise")
  expect_equal(captured$fn, "coloc")
  expect_equal(captured$args$sumstats2, "b.tsv")
  expect_s4_class(r, "ColocRun")
  expect_equal(r@candidate_snp, 137L)
  expect_equal(r@pp$h4, 0.92)
})

test_that("tg_coloc forwards a character vector of sumstats for hyprcoloc", {
  captured <- NULL
  mockery::stub(tg_coloc, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "hyprcoloc", pp = list(), candidate_snp = 42L,
         headline = 0.75, table = list())
  })
  r <- tg_coloc(c("a.tsv", "b.tsv", "c.tsv"), method = "hyprcoloc")
  expect_equal(captured$args$sumstats, c("a.tsv", "b.tsv", "c.tsv"))
  expect_null(captured$args$sumstats2)
  expect_s4_class(r, "ColocRun")
  expect_equal(r@method, "hyprcoloc")
})
