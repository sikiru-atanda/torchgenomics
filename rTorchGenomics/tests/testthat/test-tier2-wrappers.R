test_that("regenerated api_auto.R wraps into CliRun and excludes hand-crafted commands", {
  src <- readLines(system.file("..", "R", "api_auto.R", package = "rTorchGenomics",
                               mustWork = FALSE))
  if (length(src) == 0) src <- readLines("R/api_auto.R")
  joined <- paste(src, collapse = "\n")
  # no auto wrapper fakes a GwasResult anymore
  expect_false(grepl("GwasResult\\$new_from_dict", joined))
  # every auto wrapper wraps into CliRun
  expect_true(grepl("CliRun\\$new_from_dict", joined))
  # hand-crafted commands are NOT auto-generated (no clash)
  for (fn in c("tg_gwas <- function", "tg_mr <- function", "tg_coloc <- function",
               "tg_recommend <- function", "tg_models <- function")) {
    expect_false(grepl(fn, joined, fixed = TRUE))
  }
})

test_that("tg_bayes_scan forwards to the bridge and wraps into CliRun", {
  captured <- NULL
  mockery::stub(tg_bayes_scan, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(command = "bayes-scan", exit_code = 0L, output_files = list(), args = list())
  })
  r <- tg_bayes_scan(genotype = "g.bed", phenotype = "p.txt")
  expect_equal(captured$fn, "bayes_scan")
  expect_s4_class(r, "CliRun")
})
