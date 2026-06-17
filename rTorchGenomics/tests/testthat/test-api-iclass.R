# Tests for tg_iclass (G x E classification on FA loadings).
#
# Like tg_lgebv, we mock the Python side by intercepting tg_py() and
# reticulate::import(), so these tests don't need the r-rtorchgenomics
# venv. Integration tests (test-integration.R) cover the real path.

# Build a fake IClassResult Python attribute lookup. The returned object
# mimics how reticulate exposes a Python dataclass: each attribute is
# directly accessible via `$`.
.fake_iclass_pyresult <- function(cluster_labels, env_ids, loadings, threshold = 0.0) {
  n_envs <- nrow(loadings)
  n_factors <- ncol(loadings)
  pol <- matrix("Z", nrow = n_envs, ncol = n_factors)
  for (i in seq_len(n_envs)) {
    pol[i, ] <- strsplit(cluster_labels[i], "")[[1]]
  }
  cm <- list()
  for (i in seq_along(env_ids)) {
    lbl <- cluster_labels[i]
    cm[[lbl]] <- c(cm[[lbl]], env_ids[i])
  }
  list(
    cluster_labels = cluster_labels,
    cluster_membership = cm,
    loadings = loadings,
    polarity_matrix = pol,
    n_factors = n_factors,
    n_envs = n_envs,
    env_ids = env_ids,
    threshold = threshold
  )
}

.with_mock_iclass <- function(make_payload, .body) {
  ns <- getNamespace("rTorchGenomics")
  old_tg_py <- get("tg_py", envir = ns, inherits = FALSE)
  on.exit(assignInNamespace("tg_py", old_tg_py, ns = "rTorchGenomics"))
  assignInNamespace("tg_py", function() new.env(), ns = "rTorchGenomics")

  # Patch reticulate::import + py_to_r so the wrapper sees the fake module.
  fake_module <- new.env()
  fake_module$iclass <- function(...) {
    args <- list(...)
    make_payload(args)
  }

  mockery::stub(tg_iclass, "reticulate::import", fake_module)
  mockery::stub(tg_iclass, "reticulate::py_to_r", function(x) x)

  .body()
}

test_that("tg_iclass returns expected cluster count (3 envs -> 2 clusters)", {
  L <- matrix(c(0.6, 0.5, -0.4,
                0.4, 0.3, -0.3),
              nrow = 3, ncol = 2,
              dimnames = list(c("E1", "E2", "E3"), c("F1", "F2")))
  cluster_labels <- c("PP", "PP", "NN")

  mockery::stub(tg_iclass, "tg_py", function() new.env())
  mockery::stub(
    tg_iclass, "reticulate::import",
    list(iclass = function(...) {
      args <- list(...)
      env_ids <- if (is.null(args$env_ids)) c("E1", "E2", "E3") else as.character(unlist(args$env_ids))
      .fake_iclass_pyresult(cluster_labels, env_ids, L)
    })
  )
  mockery::stub(tg_iclass, "reticulate::py_to_r", function(x) x)

  r <- tg_iclass(L)
  expect_s4_class(r, "IClassResult")
  expect_equal(length(r@cluster_membership), 2L)
  expect_setequal(names(r@cluster_membership), c("PP", "NN"))
  expect_equal(r@cluster_membership$PP, c("E1", "E2"))
  expect_equal(r@cluster_membership$NN, "E3")
})

test_that("tg_iclass S4 fields are populated", {
  L <- matrix(c(0.6, -0.2,
                0.5, -0.3,
                0.7,  0.1),
              nrow = 3, ncol = 2, byrow = TRUE)
  cluster_labels <- c("PN", "PN", "PP")
  env_ids <- c("Env1", "Env2", "Env3")

  mockery::stub(tg_iclass, "tg_py", function() new.env())
  mockery::stub(
    tg_iclass, "reticulate::import",
    list(iclass = function(...) {
      .fake_iclass_pyresult(cluster_labels, env_ids, L)
    })
  )
  mockery::stub(tg_iclass, "reticulate::py_to_r", function(x) x)

  r <- tg_iclass(L, env_ids = env_ids)
  expect_equal(r@n_envs, 3L)
  expect_equal(r@n_factors, 2L)
  expect_equal(r@env_ids, env_ids)
  expect_equal(r@cluster_labels, cluster_labels)
  expect_equal(dim(r@loadings), c(3L, 2L))
  expect_equal(dim(r@polarity_matrix), c(3L, 2L))
})

test_that("tg_iclass env_ids preserved from matrix rownames", {
  L <- matrix(c(0.6, 0.5, 0.4,
                0.3, 0.2, 0.1),
              nrow = 3, ncol = 2,
              dimnames = list(c("EnvA", "EnvB", "EnvC"), NULL))
  cluster_labels <- c("PP", "PP", "PP")

  mockery::stub(tg_iclass, "tg_py", function() new.env())
  mockery::stub(
    tg_iclass, "reticulate::import",
    list(iclass = function(...) {
      args <- list(...)
      env_ids <- as.character(unlist(args$env_ids))
      .fake_iclass_pyresult(cluster_labels, env_ids, L)
    })
  )
  mockery::stub(tg_iclass, "reticulate::py_to_r", function(x) x)

  r <- tg_iclass(L)   # no env_ids supplied
  expect_equal(r@env_ids, c("EnvA", "EnvB", "EnvC"))
})

test_that("tg_iclass rejects non-numeric loadings", {
  expect_error(
    tg_iclass(matrix("a", 2, 2)),
    "numeric matrix"
  )
})

test_that("tg_iclass rejects negative threshold", {
  L <- matrix(rnorm(6), 3, 2)
  expect_error(tg_iclass(L, threshold = -0.1), "non-negative")
})

test_that("tg_iclass IClassResult is not virtual", {
  expect_false(methods::isVirtualClass("IClassResult"))
})

test_that("to_dataframe(IClassResult) returns env_id + cluster_label tibble", {
  L <- matrix(c(0.6, 0.5, -0.4,
                0.4, 0.3, -0.3),
              nrow = 3, ncol = 2,
              dimnames = list(c("E1", "E2", "E3"), NULL))
  cluster_labels <- c("PP", "PP", "NN")

  mockery::stub(tg_iclass, "tg_py", function() new.env())
  mockery::stub(
    tg_iclass, "reticulate::import",
    list(iclass = function(...) {
      .fake_iclass_pyresult(cluster_labels, c("E1", "E2", "E3"), L)
    })
  )
  mockery::stub(tg_iclass, "reticulate::py_to_r", function(x) x)

  r <- tg_iclass(L)
  df <- to_dataframe(r)
  expect_s3_class(df, "tbl_df")
  expect_equal(nrow(df), 3L)
  expect_equal(df$env_id, c("E1", "E2", "E3"))
  expect_equal(df$cluster_label, cluster_labels)
})
