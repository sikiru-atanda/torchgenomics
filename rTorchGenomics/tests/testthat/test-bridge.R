test_that("bridge_call invokes the Python function with the right args", {
  fake_api <- list(
    lmm_scan = function(genotype, phenotype) {
      list(
        to_dict = function() list(
          n_variants = 100L, n_significant = 5L,
          model = "SingleTraitLMM", top_hits = list()
        )
      )
    }
  )
  mockery::stub(bridge_call, "tg_py", function() list(api = fake_api))

  result <- bridge_call("lmm_scan", list(genotype = "g.bed", phenotype = "p.tsv"))

  expect_type(result, "list")
  expect_equal(result$n_variants, 100L)
  expect_equal(result$model, "SingleTraitLMM")
})

test_that("bridge_call maps Python ValueError to tg_input_error", {
  fake_api <- list(
    lmm_scan = function(...) stop(
      structure(
        class = c("python.builtin.ValueError", "python.builtin.Exception",
                  "error", "condition"),
        list(message = "phenotype not found", call = NULL)
      )
    )
  )
  mockery::stub(bridge_call, "tg_py", function() list(api = fake_api))

  expect_error(
    bridge_call("lmm_scan", list(genotype = "g.bed", phenotype = "missing.tsv")),
    class = "tg_input_error"
  )
})

test_that("bridge_call maps Python RuntimeError to tg_runtime_error", {
  fake_api <- list(
    lmm_scan = function(...) stop(
      structure(
        class = c("python.builtin.RuntimeError", "python.builtin.Exception",
                  "error", "condition"),
        list(message = "REML did not converge", call = NULL)
      )
    )
  )
  mockery::stub(bridge_call, "tg_py", function() list(api = fake_api))

  expect_error(
    bridge_call("lmm_scan", list(genotype = "g.bed", phenotype = "p.tsv")),
    class = "tg_runtime_error"
  )
})

test_that("bridge_call signals tg_no_python when reticulate import fails", {
  mockery::stub(bridge_call, "tg_py",
                function() stop(structure(
                  class = c("tg_no_python", "error", "condition"),
                  list(message = "no python interpreter found")
                )))

  expect_error(
    bridge_call("lmm_scan", list()),
    class = "tg_no_python"
  )
})
