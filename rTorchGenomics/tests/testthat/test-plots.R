.make_scan_run_with_tsv <- function(tmp) {
  tsv <- file.path(tmp, "results.tsv")
  utils::write.table(
    data.frame(
      CHR = c("1", "1", "2", "2"),
      POS = c(100, 200, 300, 400),
      SNP = c("rs1", "rs2", "rs3", "rs4"),
      P = c(1e-9, 1e-3, 1e-8, 0.5)
    ),
    tsv, sep = "\t", quote = FALSE, row.names = FALSE
  )
  new("ScanRun",
      runtime_s = 1.0,
      output_files = list(tsv = tsv),
      log_excerpt = character(0),
      model = "SingleTraitLMM", test = "wald", correction = "bh",
      n_variants = 4L, n_significant = 2L,
      significance_threshold = 5e-8,
      lambda_gc = 1.0, sigma2_g = NA_real_, sigma2_e = NA_real_,
      h2 = NA_real_, n_samples = 0L,
      top_hits = tibble::tibble(
        CHR = c("1", "2"), POS = c(100, 300),
        SNP = c("rs1", "rs3"), P = c(1e-9, 1e-8)
      )
  )
}

test_that("tg_manhattan returns a ggplot object", {
  tmp <- tempdir()
  r <- .make_scan_run_with_tsv(tmp)
  p <- tg_manhattan(r)
  expect_s3_class(p, "ggplot")
  layer_classes <- vapply(p$layers, function(L) class(L$geom)[1L], character(1L))
  expect_true(any(grepl("GeomPoint", layer_classes)))
  expect_true(any(grepl("GeomHline", layer_classes)))   # significance line
})

test_that("tg_manhattan falls back to top_hits when TSV missing", {
  r <- new("ScanRun",
           runtime_s = 1.0,
           output_files = list(),
           log_excerpt = character(0),
           model = "SingleTraitLMM", test = "wald", correction = "bh",
           n_variants = 2L, n_significant = 1L,
           significance_threshold = 5e-8,
           lambda_gc = NA_real_, sigma2_g = NA_real_,
           sigma2_e = NA_real_, h2 = NA_real_, n_samples = 0L,
           top_hits = tibble::tibble(
             CHR = c("1", "2"), POS = c(100, 200),
             SNP = c("rs1", "rs2"), P = c(1e-9, 0.5)
           )
  )
  p <- tg_manhattan(r)
  expect_s3_class(p, "ggplot")
})

test_that("tg_qq returns a ggplot object with the diagonal reference", {
  tmp <- tempdir()
  r <- .make_scan_run_with_tsv(tmp)
  p <- tg_qq(r)
  expect_s3_class(p, "ggplot")
  layer_classes <- vapply(p$layers, function(L) class(L$geom)[1L], character(1L))
  expect_true(any(grepl("GeomAbline|GeomLine", layer_classes)))
})
