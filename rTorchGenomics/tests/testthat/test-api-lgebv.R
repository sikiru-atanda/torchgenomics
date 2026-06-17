# Tests for tg_lgebv (Local Genomic Estimated Breeding Values).
#
# Hits the wrapper at the bridge boundary via a mocked tg_py().
# Integration tests in test-integration.R cover the real round-trip.

.fake_lgebv_payload <- function(favorable = c(TRUE, FALSE, TRUE)) {
  list(
    runtime_s = 0.1,
    output_files = list(),
    log_excerpt = list(),
    block_id = c("block_0", "block_1", "block_2"),
    chrom = c("1", "1", "2"),
    start = list(100L, 500L, 100L),
    end = list(300L, 900L, 400L),
    n_variants = list(5L, 6L, 4L),
    lgebv = list(-0.42, 0.31, -0.18),
    block_variance = list(0.21, 0.15, 0.08),
    favorable = as.list(favorable),
    marker_effects = NULL,
    h2_used = 0.4,
    method = "rrblup"
  )
}

# --- helper that mocks tg_py() so the bridge layer is not exercised ----
.with_mock_lgebv <- function(favorable, .body) {
  fake_py <- new.env()
  fake_api <- new.env()
  fake_api$lgebv <- function(...) {
    args <- list(...)
    # The mocked Python function returns a list whose $to_dict() yields
    # the fake payload. We make it look like a dict directly to skip
    # the py_has_attr/to_dict branch.
    .fake_lgebv_payload(favorable)
  }
  fake_py$api <- fake_api

  with_mocked_bindings <- function(env, names, vals, expr) {
    olds <- mget(names, envir = env, inherits = FALSE)
    on.exit({
      for (n in names) {
        assign(n, olds[[n]], envir = env)
      }
    })
    for (i in seq_along(names)) {
      assign(names[[i]], vals[[i]], envir = env)
    }
    force(expr)
  }

  ns <- getNamespace("rTorchGenomics")
  # Save originals
  old_tg_py <- get("tg_py", envir = ns, inherits = FALSE)
  old_py_has_attr <- reticulate::py_has_attr
  old_py_to_r <- reticulate::py_to_r

  # tg_py() returns our fake module
  mock_tg_py <- function() fake_py
  # Patch reticulate to bypass py_to_r / py_has_attr on R lists
  assignInNamespace("tg_py", mock_tg_py, ns = "rTorchGenomics")
  on.exit({
    assignInNamespace("tg_py", old_tg_py, ns = "rTorchGenomics")
  })

  .body()
}

# Build a tiny n=30 x m=15 fixture (no genotype file needed because
# the mocked Python side doesn't read it).
.fake_inputs <- function(seed = 42L) {
  set.seed(seed)
  n <- 30L; m <- 15L
  list(
    y = rnorm(n),
    G = matrix(sample(0:2, n * m, replace = TRUE), nrow = n, ncol = m),
    blocks = list(
      list(variant_indices = 1:5),
      list(variant_indices = 6:11),
      list(variant_indices = 12:15)
    )
  )
}

test_that("tg_lgebv runs on tiny fixture and wraps as LGEBVResult", {
  fav <- c(TRUE, FALSE, TRUE)
  .with_mock_lgebv(fav, function() {
    inp <- .fake_inputs()
    r <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks)
    expect_s4_class(r, "LGEBVResult")
    expect_type(r@favorable, "logical")
    expect_equal(length(r@favorable), length(r@block_id))
    expect_equal(r@favorable, fav)
    expect_equal(r@method, "rrblup")
    expect_equal(length(r@lgebv), 3L)
  })
})

test_that("tg_lgebv populates n_variants and chrom slots", {
  .with_mock_lgebv(c(TRUE, FALSE, TRUE), function() {
    inp <- .fake_inputs()
    r <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks)
    expect_equal(r@n_variants, c(5L, 6L, 4L))
    expect_equal(r@chrom, c("1", "1", "2"))
    expect_equal(r@h2_used, 0.4)
  })
})

test_that("tg_lgebv favorable_direction inverts the per-block flag", {
  # Mock the Python side so that "positive" returns the inversion of
  # the "negative" payload, matching the real lgebv() semantics.
  fav_neg <- c(TRUE, FALSE, TRUE)
  fav_pos <- !fav_neg

  ns <- getNamespace("rTorchGenomics")
  old_tg_py <- get("tg_py", envir = ns, inherits = FALSE)
  on.exit(assignInNamespace("tg_py", old_tg_py, ns = "rTorchGenomics"))

  fake_py <- new.env()
  fake_api <- new.env()
  fake_api$lgebv <- function(...) {
    args <- list(...)
    direction <- if (is.null(args$favorable_direction)) "negative" else args$favorable_direction
    if (direction == "positive") .fake_lgebv_payload(fav_pos)
    else .fake_lgebv_payload(fav_neg)
  }
  fake_py$api <- fake_api
  assignInNamespace("tg_py", function() fake_py, ns = "rTorchGenomics")

  inp <- .fake_inputs()
  r_neg <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks,
                   favorable_direction = "negative")
  r_pos <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks,
                   favorable_direction = "positive")

  expect_equal(r_neg@favorable, fav_neg)
  expect_equal(r_pos@favorable, fav_pos)
  expect_equal(r_neg@favorable, !r_pos@favorable)
})

test_that("tg_lgebv returns a real LGEBVResult S4 instance (not virtual)", {
  expect_false(methods::isVirtualClass("LGEBVResult"))
  .with_mock_lgebv(c(TRUE, FALSE, TRUE), function() {
    inp <- .fake_inputs()
    r <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks)
    expect_true(methods::is(r, "LGEBVResult"))
  })
})

test_that("to_dataframe(LGEBVResult) returns a per-block tibble", {
  .with_mock_lgebv(c(TRUE, FALSE, TRUE), function() {
    inp <- .fake_inputs()
    r <- tg_lgebv(y = inp$y, genotype = inp$G, blocks = inp$blocks)
    df <- to_dataframe(r)
    expect_s3_class(df, "tbl_df")
    expect_equal(nrow(df), 3L)
    expect_true(all(c("block_id", "chrom", "lgebv", "favorable") %in% names(df)))
  })
})
