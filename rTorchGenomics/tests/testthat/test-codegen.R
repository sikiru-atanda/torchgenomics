test_that("rbridge_manifest.json is present and parses", {
  path <- system.file("rbridge_manifest.json", package = "rTorchGenomics")
  expect_true(nzchar(path), info = "manifest file shipped")
  manifest <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  expect_true(length(manifest$commands) >= 30L)
  expect_match(manifest$torchgenomics_version, "^[0-9]+\\.[0-9]+\\.[0-9]+$")
})

test_that("api_auto.R defines tg_* for every tier-2 manifest command", {
  path <- system.file("rbridge_manifest.json", package = "rTorchGenomics")
  manifest <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  ns <- getNamespace("rTorchGenomics")
  for (cmd in manifest$commands) {
    fn_name <- paste0("tg_", cmd$name)
    expect_true(exists(fn_name, envir = ns, inherits = FALSE), info = fn_name)
  }
})

test_that("generate_api_auto produces syntactically valid R", {
  src <- system.file("rbridge_manifest.json", package = "rTorchGenomics")
  tmp <- tempfile(fileext = ".R")
  on.exit(unlink(tmp), add = TRUE)
  generate_api_auto(manifest_path = src, output_path = tmp)
  parsed <- tryCatch(parse(tmp), error = function(e) NULL)
  expect_false(is.null(parsed))
  expect_true(length(parsed) >= 30L)
})

test_that("an auto-generated wrapper dispatches via bridge_call", {
  path <- system.file("rbridge_manifest.json", package = "rTorchGenomics")
  manifest <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  cmd <- manifest$commands[[1L]]
  fn_name <- paste0("tg_", cmd$name)
  fn <- get(fn_name, envir = asNamespace("rTorchGenomics"))

  mockery::stub(fn, "bridge_call", function(name, args) {
    expect_equal(name, cmd$name)
    list(
      runtime_s = 1.0, output_files = list(), log_excerpt = list(),
      summary = paste(cmd$name, "ran"), n_variants = 0L,
      top_hits = list()
    )
  })

  args <- list()
  for (a in cmd$args) {
    if (a$required) args[[a$name]] <- "dummy"
  }
  r <- do.call(fn, args)
  expect_s4_class(r, "GwasResult")
})
