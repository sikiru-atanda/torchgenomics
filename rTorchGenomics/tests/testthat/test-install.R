test_that("tg_install builds the correct package spec from the R pkg version", {
  install_calls <- list()
  mockery::stub(tg_install, "reticulate::virtualenv_exists", function(envname) TRUE)
  mockery::stub(tg_install, "reticulate::py_install",
                function(packages, envname, pip, method) {
                  install_calls[[length(install_calls) + 1L]] <<- list(
                    packages = packages, envname = envname
                  )
                  invisible(TRUE)
                })

  tg_install()

  expect_length(install_calls, 1L)
  expect_match(install_calls[[1]]$packages, "^torchgenomics\\[mcp\\]==")
  expect_equal(install_calls[[1]]$envname, "r-rtorchgenomics")
})

test_that("tg_install with force=TRUE removes existing venv first", {
  removed <- FALSE
  mockery::stub(tg_install, "reticulate::virtualenv_exists", function(envname) TRUE)
  mockery::stub(tg_install, "reticulate::virtualenv_remove",
                function(envname, confirm) {
                  removed <<- TRUE
                })
  mockery::stub(tg_install, "reticulate::virtualenv_create",
                function(envname, python) invisible(TRUE))
  mockery::stub(tg_install, "reticulate::py_install",
                function(...) invisible(TRUE))
  mockery::stub(tg_install, "reticulate::virtualenv_python",
                function() "/usr/bin/python3")

  tg_install(force = TRUE)

  expect_true(removed)
})
