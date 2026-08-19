# Tests for the friendly-API R wrappers `tg_gwas` / `tg_recommend` /
# `tg_models` (Task 9).
#
# Two layers, mirroring the rest of the test suite:
#  - Mocked unit tests (mockery::stub over bridge_call / tg_py, as in
#    test-api.R) -- always run, no Python required, exercise the R-side
#    argument-building and single-vs-comparison branching logic exactly.
#  - A real, opt-in integration block (RTORCHGENOMICS_INTEGRATION=1;
#    the same gate test-integration.R uses) against the tiny bundled
#    fixture, run through the actual r-rtorchgenomics venv.

# --- Mocked unit tests --------------------------------------------------

test_that("tg_gwas/tg_recommend/tg_models are exported", {
  ns <- getNamespace("rTorchGenomics")
  for (fn in c("tg_gwas", "tg_recommend", "tg_models")) {
    expect_true(exists(fn, envir = ns, inherits = FALSE), info = fn)
  }
})

.fake_scan_dict <- function(model = "SingleTraitLMM", n_significant = 0L) {
  list(
    runtime_s = 1.2, output_files = list(tsv = "out.tsv"), log_excerpt = list(),
    model = model, test = "wald", correction = "bh",
    n_variants = 20L, n_significant = n_significant,
    significance_threshold = 5e-8, lambda_gc = 0.93,
    sigma2_g = 1.1, sigma2_e = 0.5, h2 = 0.4,
    n_samples = 12L, top_hits = list(),
    trait_type = "continuous", warnings = list()
  )
}

test_that("tg_gwas: models='auto' (default) calls bridge_call('gwas', ...) with a scalar and returns a single ScanRun", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$fn <- fn
    captured$args <- args
    .fake_scan_dict()
  })

  r <- tg_gwas(phenotype = "p.tsv", genotype = "g.bed")

  expect_equal(captured$fn, "gwas")
  expect_equal(captured$args$phenotype, "p.tsv")
  expect_equal(captured$args$genotype, "g.bed")
  expect_equal(captured$args$models, "auto")
  expect_true(is.character(captured$args$models) && length(captured$args$models) == 1L)
  expect_s4_class(r, "ScanRun")
  expect_equal(r@model, "SingleTraitLMM")
  expect_true(is.data.frame(r@top_hits))
})

test_that("tg_gwas: a single alias string ('lmm') stays a scalar and returns a single ScanRun", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$args <- args
    .fake_scan_dict(model = "SingleTraitLMM")
  })

  r <- tg_gwas(phenotype = "p.tsv", genotype = "g.bed", models = "lmm")

  expect_equal(captured$args$models, "lmm")
  expect_true(is.character(captured$args$models) && length(captured$args$models) == 1L)
  expect_s4_class(r, "ScanRun")
})

test_that("tg_gwas: a length>1 character vector of models is sent as a list and returns a named list of ScanRun", {
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    expect_true(is.list(args$models))
    expect_equal(unlist(args$models, use.names = FALSE), c("lmm", "blink"))
    list(
      runtime_s = 2.4, output_files = list(), log_excerpt = list(),
      trait_name = "y", trait_type = "continuous",
      results = list(
        lmm = .fake_scan_dict(model = "SingleTraitLMM", n_significant = 0L),
        blink = .fake_scan_dict(model = "BLINK", n_significant = 1L)
      )
    )
  })

  r <- tg_gwas(phenotype = "p.tsv", genotype = "g.bed", models = c("lmm", "blink"))

  expect_type(r, "list")
  expect_named(r, c("lmm", "blink"))
  expect_s4_class(r$lmm, "ScanRun")
  expect_s4_class(r$blink, "ScanRun")
  expect_equal(r$lmm@model, "SingleTraitLMM")
  expect_equal(r$blink@model, "BLINK")
  expect_equal(r$blink@n_significant, 1L)
})

test_that("tg_gwas: a one-element list of models (Fix-2 style) also returns a comparison list, not a bare ScanRun", {
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    expect_true(is.list(args$models))
    expect_equal(unlist(args$models, use.names = FALSE), "lmm")
    list(
      runtime_s = 1.0, output_files = list(), log_excerpt = list(),
      trait_name = "y", trait_type = "continuous",
      results = list(lmm = .fake_scan_dict())
    )
  })

  r <- tg_gwas(phenotype = "p.tsv", genotype = "g.bed", models = list("lmm"))

  expect_type(r, "list")
  expect_named(r, "lmm")
  expect_s4_class(r$lmm, "ScanRun")
})

test_that("tg_gwas: kinship=FALSE is passed through unchanged (not dropped like NULL would be)", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$args <- args
    .fake_scan_dict(model = "GLM")
  })

  tg_gwas(phenotype = "p.tsv", genotype = "g.bed", kinship = FALSE, models = "glm")

  expect_identical(captured$args$kinship, FALSE)
})

test_that("tg_gwas: numeric pcs is coerced to integer", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$args <- args
    .fake_scan_dict()
  })

  tg_gwas(phenotype = "p.tsv", genotype = "g.bed", pcs = 3)

  expect_identical(captured$args$pcs, 3L)
})

test_that("tg_gwas: device defaults to 'auto' and match.arg()s an explicit valid value", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$args <- args
    .fake_scan_dict()
  })

  tg_gwas(phenotype = "p.tsv", genotype = "g.bed")
  expect_identical(captured$args$device, "auto")

  tg_gwas(phenotype = "p.tsv", genotype = "g.bed", device = "cuda")
  expect_identical(captured$args$device, "cuda")
})

test_that("tg_gwas: an invalid device is a clear match.arg() error, not a bridge_call round-trip", {
  expect_error(
    tg_gwas(phenotype = "p.tsv", genotype = "g.bed", device = "tpu"),
    "'arg' should be one of"
  )
})

test_that("tg_gwas: models of the wrong type is a clear R-level error, not a bridge_call crash", {
  expect_error(
    tg_gwas(phenotype = "p.tsv", genotype = "g.bed", models = 1L),
    "models must be"
  )
})

test_that("tg_gwas: an empty 'results' in a comparison response is a clear runtime error", {
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    list(runtime_s = 0, output_files = list(), log_excerpt = list(),
         trait_name = "y", trait_type = "continuous")
  })
  expect_error(
    tg_gwas(phenotype = "p.tsv", genotype = "g.bed", models = c("lmm", "blink")),
    "comparison"
  )
})

test_that("tg_gwas forwards model_options to the bridge", {
  captured <- new.env()
  mockery::stub(tg_gwas, "bridge_call", function(fn, args) {
    captured$args <- args
    .fake_scan_dict(model = "GLMM")
  })

  tg_gwas("p.tsv", "g.bed", models = "glmm",
          model_options = list(glmm = list(family = "ordinal")))

  expect_equal(captured$args$model_options$glmm$family, "ordinal")
})

test_that("tg_gwas forwards env and regions paths to the bridge", {
  captured <- NULL
  mockery::stub(tg_gwas, "bridge_call", function(fn, args = list()) {
    captured <<- args
    .fake_scan_dict()          # existing helper in this file
  })
  tg_gwas("pheno.tsv", "data.bed", models = "gxe", env = "env.tsv")
  expect_equal(captured$env, "env.tsv")
  tg_gwas("pheno.tsv", "data.bed", models = "set", regions = "regions.bed")
  expect_equal(captured$regions, "regions.bed")
})

test_that("tg_recommend dispatches via bridge_call('recommend', ...) and returns a printable plan", {
  captured <- new.env()
  mockery::stub(tg_recommend, "bridge_call", function(fn, args) {
    captured$fn <- fn
    captured$args <- args
    list(
      trait_name = "y", trait_type = "continuous", n_samples = 12L,
      suggested_model = "lmm",
      rationale = "continuous trait + kinship on -> lmm",
      n_pcs = 0L,
      qc_summary = list(maf_min = 0.01, miss_max = 0.1, hwe_p_min = 1e-6),
      plan = "Plan for trait 'y' (continuous), n=12 samples:\n  model: 'lmm' / SingleTraitLMM"
    )
  })

  rec <- tg_recommend(phenotype = "p.tsv", genotype = "g.bed")

  expect_equal(captured$fn, "recommend")
  expect_equal(captured$args$phenotype, "p.tsv")
  expect_s3_class(rec, "tg_recommendation")
  expect_equal(rec$suggested_model, "lmm")
  expect_equal(rec$qc_summary$maf_min, 0.01)
  out <- capture.output(print(rec))
  expect_true(any(grepl("Plan for trait", out)))
})

test_that("tg_models calls torchgenomics.api.list_models() directly (not bridge_call) and returns a data.frame", {
  fake_df <- data.frame(
    alias = c("lmm", "glm", "farmcpu"),
    label = c("GRM mixed model", "Fixed-effects GLM", "FarmCPU"),
    trait_types = c("continuous", "continuous,binary,categorical", "continuous"),
    description = c("d1", "d2", "d3"),
    stringsAsFactors = FALSE
  )
  fake_api <- list(list_models = function() fake_df)
  fake_py <- list(api = fake_api)
  mockery::stub(tg_models, "tg_py", function() fake_py)

  df <- tg_models()

  expect_true(is.data.frame(df))
  expect_true("lmm" %in% df$alias)
  expect_setequal(names(df), c("alias", "label", "trait_types", "description"))
})

test_that(".na_preserving_unlist keeps a NULL element as NA instead of dropping it", {
  # A pandas None/NaN cell, once reticulate hands back a per-column list,
  # shows up as a NULL (length-0) element. A bare unlist() would drop it
  # and shorten the vector; .na_preserving_unlist() must not.
  col_with_null <- list(1, NULL, 3)
  out <- .na_preserving_unlist(col_with_null)
  expect_length(out, 3L)
  expect_equal(out, c(1, NA, 3))

  # Character column, NULL in the middle.
  char_col <- list("a", NULL, "c")
  out_chr <- .na_preserving_unlist(char_col)
  expect_length(out_chr, 3L)
  expect_equal(out_chr, c("a", NA, "c"))

  # An already-atomic column (reticulate sometimes simplifies a fully
  # populated column) passes through unchanged.
  atomic_col <- c(1, 2, 3)
  expect_identical(.na_preserving_unlist(atomic_col), atomic_col)
})

test_that(".pandas_df_to_r fallback path preserves row count and NA position when a column has a NULL cell", {
  # Simulate the reticulate-style shape .pandas_df_to_r's fallback branch
  # consumes: df$columns$tolist() -> a list of column names,
  # df$to_dict(orient = "list") -> a named list of per-column lists, one
  # of which contains a NULL (a pandas None/NaN cell). We stand in for
  # the "is this a real Python object" check and for reticulate::py_to_r
  # (which reticulate would normally apply recursively) with an identity
  # pass-through, matching the pattern already used for tg_iclass in
  # test-api-iclass.R -- no real Python/venv required.
  fake_df <- list(
    columns = list(tolist = function() list("id", "value")),
    to_dict = function(orient) {
      list(
        id = list(1, 2, 3),
        value = list(10, NULL, 30)  # row 2's `value` is missing
      )
    }
  )

  mockery::stub(.pandas_df_to_r, "inherits", function(x, what) TRUE)
  mockery::stub(.pandas_df_to_r, "reticulate::py_to_r", function(x) x)

  out <- .pandas_df_to_r(fake_df)

  expect_true(is.data.frame(out))
  expect_equal(nrow(out), 3L)
  expect_equal(names(out), c("id", "value"))
  expect_equal(out$id, c(1, 2, 3))
  expect_equal(out$value, c(10, NA, 30))
  expect_true(is.na(out$value[2]))
})

# --- Real, opt-in integration tests -------------------------------------
#
# Same gate as test-integration.R: `Sys.setenv(RTORCHGENOMICS_INTEGRATION =
# "1")` plus a working `r-rtorchgenomics` reticulate venv (created by
# tg_install(), or already present from a prior integration run). Skips
# gracefully — including when the installed torchgenomics predates
# `gwas`/`recommend`/`models()` on `torchgenomics.api` — rather than
# failing CI runs that don't have the managed venv set up.

skip_if_not_integration <- function() {
  if (Sys.getenv("RTORCHGENOMICS_INTEGRATION", unset = "") != "1") {
    skip("Integration test — set RTORCHGENOMICS_INTEGRATION=1 to enable.")
  }
  if (!reticulate::virtualenv_exists("r-rtorchgenomics")) {
    skip("r-rtorchgenomics venv not found; run tg_install() first.")
  }
  py <- tryCatch(tg_py(), error = function(e) NULL)
  if (is.null(py) || !reticulate::py_has_attr(py$api, "gwas")) {
    skip("torchgenomics.api in the managed venv has no 'gwas' (older install); run tg_install(force = TRUE).")
  }
}

test_that("[integration] tg_models() lists 'lmm' among the real registry aliases", {
  skip_if_not_integration()
  df <- tg_models()
  expect_true(is.data.frame(df))
  expect_true("lmm" %in% df$alias)
})

test_that("[integration] tg_gwas() on the tiny fixture (single model) returns a populated ScanRun", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")

  r <- tg_gwas(phenotype = pheno, genotype = bed, models = "lmm", verbose = FALSE)

  expect_s4_class(r, "ScanRun")
  expect_true(is.data.frame(r@top_hits))
  expect_equal(r@model, "SingleTraitLMM")
  # ScanRun's show() method (invoked by print()) names the fitted model;
  # ScanRun has no `trait` slot of its own (it mirrors Python's
  # ScanRun/GwasResult identity, which stores trait_type but not a
  # trait *name*), so we check the model name is what's visibly reported.
  out <- capture.output(print(r))
  expect_true(any(grepl("SingleTraitLMM", out)))
})

test_that("[integration] tg_gwas() on the tiny fixture (comparison) returns a named list of ScanRun", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")

  r <- tg_gwas(phenotype = pheno, genotype = bed, models = c("lmm", "glm"), verbose = FALSE)

  expect_type(r, "list")
  expect_named(r, c("lmm", "glm"))
  expect_s4_class(r$lmm, "ScanRun")
  expect_s4_class(r$glm, "ScanRun")
})

test_that("[integration] tg_recommend() on the tiny fixture returns a real plan", {
  skip_if_not_integration()
  bed <- system.file("extdata", "tiny.bed", package = "rTorchGenomics")
  pheno <- system.file("extdata", "tiny_pheno.tsv", package = "rTorchGenomics")

  rec <- tg_recommend(phenotype = pheno, genotype = bed)

  expect_s3_class(rec, "tg_recommendation")
  expect_equal(rec$trait_type, "continuous")
  expect_true(nzchar(rec$suggested_model))
  out <- capture.output(print(rec))
  expect_true(any(grepl("Plan for trait", out)))
})
