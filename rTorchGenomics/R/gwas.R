# rTorchGenomics/R/gwas.R
#
# R wrappers for the friendly-API decision core (torchgenomics.api.gwas /
# .recommend / .models -- Task 5/6 on the Python side of the friendly-API
# plan). Unlike the hand-crafted wrappers in R/api.R, these three are
# reached through `bridge_call()` almost unchanged: `bridge_call()` already
# resolves `torchgenomics.api[["gwas"]]` / `[["recommend"]]` by name, calls
# it, and auto-detects + calls `.to_dict()` on the result when present
# (see R/bridge.R). `tg_models()` is the one exception -- `models()`
# returns a pandas DataFrame, not a `to_dict()`-able result object, so it
# is called directly (see its own docs below).

#' Run a GWAS scan end-to-end from whatever inputs are on hand.
#'
#' The R entry point to the friendly API's single call:
#' `torchgenomics.gwas(phenotype, genotype, ...)` on the Python side. It
#' resolves and sample-aligns `phenotype` / `genotype` / `covariates`,
#' picks a model (or validates the caller's choice), runs it, and returns
#' a `ScanRun` -- the same S4 class built by [tg_lmm_scan()] / [tg_glm_scan()]
#' (Python's `GwasResult` *is* `ScanRun`; see the "Value" section of
#' `torchgenomics.api._results.ScanRun`'s docstring). No new statistics are
#' computed here or on the Python side of this call -- it is pure
#' orchestration over already-tested model paths
#' (`torchgenomics.api._dispatch.run_model`).
#'
#' @param phenotype Path to a phenotype file (TSV/CSV; first non-ID column
#'   is used as the trait unless `trait` is given).
#' @param genotype Path to a genotype file (any supported format: BED,
#'   VCF/BCF, BGEN, HapMap, Zarr, CSV dosage -- auto-detected).
#' @param covariates Optional path to a covariates file, sample-aligned to
#'   `phenotype`.
#' @param kinship GRM / kinship policy. `"auto"` (default) computes a
#'   streaming VanRaden GRM when a mixed model is used; a path loads a
#'   pre-computed GRM; `FALSE` disables kinship correction for models that
#'   support running without it (steers `models = "auto"` toward `"glm"`
#'   instead of `"lmm"` for a continuous trait). **Use `FALSE`, not `NULL`,
#'   to disable kinship** -- `NULL` is dropped from the call entirely (see
#'   [`.compact()`]) and the Python-side default `"auto"` is used instead,
#'   which would silently re-enable kinship correction.
#' @param pcs Number of genotype principal components to add as
#'   covariates. `"auto"` (default; currently resolves to 0 -- see
#'   `torchgenomics.api._dispatch._n_pcs_from_opts`), or an integer count.
#'   As with `kinship`, pass a concrete value rather than `NULL` if you
#'   need to override the default.
#' @param trait Optional trait column name, when `phenotype` has more than
#'   one candidate trait column.
#' @param trait_type Optional forced `"continuous"` / `"binary"` /
#'   `"categorical"`, instead of letting the Python side auto-detect it.
#'   Also feeds the `models = "auto"` decision.
#' @param models `"auto"` (default) picks exactly one model and returns a
#'   single [ScanRun-class] object. A single alias (character scalar,
#'   e.g. `"lmm"`) likewise runs one model and returns a single
#'   [ScanRun-class]. **A character vector of length >= 2** (e.g.
#'   `c("lmm", "blink")`) or an R `list` of aliases (even a one-element
#'   list, e.g. `list("lmm")`, mirroring the Python side's "list means
#'   comparison" rule -- Fix 2 in `torchgenomics.api.gwas`) runs every
#'   alias and returns a **named `list` of [ScanRun-class] objects**, one
#'   per alias, in request order -- see "Value" below. Call [tg_models()]
#'   for the full registry of valid aliases.
#' @param qc Logical; per-variant QC (MAF / missingness / Hardy-Weinberg).
#'   Default `TRUE`.
#' @param correction Multiple-testing correction: `"bh"` (default),
#'   `"bonferroni"`, `"holm"`, `"by"`, `"storey"`, `"none"`, ...
#' @param output Optional output directory. For a single model, the
#'   association table and a `summary.txt` + best-effort plots are written
#'   there. For a model comparison, each model's own report is written to
#'   a same-named subfolder, plus a top-level `comparison_summary.txt`.
#' @param device `"cpu"` / `"cuda"` / `"auto"` / `NULL` (resolves to
#'   `"auto"`).
#' @param verbose Print the Python-side decision log (trait name + type,
#'   sample count, the model(s) chosen and why, #PCs, correction) before
#'   running. Default `TRUE`; pass `FALSE` for silent/scripted use.
#'
#' @return
#' - When `models` is a character **scalar** (`"auto"` or a single alias
#'   like `"lmm"`): a single [ScanRun-class] object.
#' - When `models` is a character **vector of length >= 2**, or an R
#'   `list` of any length: a named `list` of [ScanRun-class] objects, one
#'   per requested alias, in request order. This mirrors the Python
#'   `GwasComparison.results` dict but is deliberately returned as a
#'   *plain* named list rather than a new R S4/S3 comparison class --
#'   comparison-level conveniences (`GwasComparison$summary()` /
#'   `$report()`) are not yet mirrored on the R side; call
#'   `lapply(result, function(r) r@model)` etc., or `show()` /
#'   `top_hits()` on each element directly. The comparison-level
#'   `trait_name` / `trait_type` / total `runtime_s` from the Python
#'   `GwasComparison` are also not attached to the returned list (each
#'   per-model `ScanRun` already carries its own `runtime_s`); use
#'   [tg_recommend()] beforehand if you need the trait name/type without
#'   running a scan.
#'
#' @examples
#' \dontrun{
#' r <- tg_gwas("pheno.tsv", "data.bed")
#' print(r); top_hits(r)
#'
#' # Explicit single model, kinship correction off:
#' r2 <- tg_gwas("pheno.tsv", "data.bed", models = "glm", kinship = FALSE)
#'
#' # Multi-model comparison -- returns a named list of ScanRun objects:
#' cmp <- tg_gwas("pheno.tsv", "data.bed", models = c("lmm", "blink"))
#' cmp$lmm@model; cmp$blink@n_significant
#' }
#' @export
tg_gwas <- function(phenotype, genotype,
                    covariates = NULL,
                    kinship = "auto",
                    pcs = "auto",
                    trait = NULL,
                    trait_type = NULL,
                    models = "auto",
                    qc = TRUE,
                    correction = "bh",
                    output = NULL,
                    device = NULL,
                    verbose = TRUE) {
  if (!is.character(models) && !is.list(models)) {
    stop("models must be a character alias (or vector of aliases), ",
         "'auto', or a list of aliases; see tg_models() for valid names.")
  }
  # Fix-2-mirroring rule (see the `models` @param above): a bare character
  # scalar means "single result"; a character vector of length > 1, or any
  # R list (even one-element), means "comparison" -- matches the Python
  # side's isinstance(models, str) vs list/tuple branch in
  # torchgenomics.api.gwas._resolve_models().
  models_is_comparison <- is.list(models) || (is.character(models) && length(models) > 1L)
  models_py <- if (models_is_comparison) {
    as.list(as.character(unlist(models, use.names = FALSE)))
  } else {
    as.character(models)[[1]]
  }

  pcs_py <- if (is.numeric(pcs)) as.integer(pcs) else pcs

  args <- .compact(list(
    phenotype = .as_path(phenotype),
    genotype = .as_path(genotype),
    covariates = .as_path(covariates),
    kinship = kinship,
    pcs = pcs_py,
    trait = trait,
    trait_type = trait_type,
    models = models_py,
    qc = qc,
    correction = correction,
    output = .as_path(output),
    device = device,
    verbose = verbose
  ))

  d <- bridge_call("gwas", args)

  if (models_is_comparison) {
    results <- d$results
    if (is.null(results) || length(results) == 0L) {
      stop(structure(
        class = c("tg_runtime_error", "error", "condition"),
        list(message = paste0(
          "tg_gwas(): expected a multi-model comparison result (models ",
          "had length > 1) but the bridge returned no non-empty 'results' ",
          "field. torchgenomics.api.gwas may be older than the version ",
          "this wrapper targets (GwasComparison.to_dict() was added ",
          "alongside this R wrapper); try tg_install(force = TRUE)."
        ))
      ))
    }
    lapply(results, ScanRun$new_from_dict)
  } else {
    ScanRun$new_from_dict(d)
  }
}

#' Preview what `tg_gwas(...)` would do, without running a scan.
#'
#' A pure planning step, mirroring `torchgenomics.recommend(...)` on the
#' Python side: it loads and sample-aligns `phenotype` / `genotype` /
#' `covariates` exactly as [tg_gwas()] does, then reuses the very same
#' auto-model decision tree and default QC/PC policy that [tg_gwas()]
#' would use for `models = "auto"` -- but stops before running anything:
#' no null model is fit, no GRM is computed, no scan runs, and nothing is
#' written to disk.
#'
#' @param phenotype Path to a phenotype file.
#' @param genotype Path to a genotype file.
#' @param covariates Optional path to a covariates file.
#'
#' @return A named `list` (S3 class `"tg_recommendation"`) with elements:
#'   \describe{
#'     \item{trait_name}{Name of the resolved trait.}
#'     \item{trait_type}{`"continuous"` / `"binary"` / `"categorical"`.}
#'     \item{n_samples}{Sample count after alignment.}
#'     \item{suggested_model}{The registry alias `tg_gwas(models = "auto")`
#'       would pick.}
#'     \item{rationale}{One-line explanation for `suggested_model`.}
#'     \item{n_pcs}{Number of genotype PCs `tg_gwas()` would add under the
#'       default `pcs = "auto"` policy.}
#'     \item{qc_summary}{Named list of planned per-variant QC thresholds
#'       (`maf_min`, `miss_max`, `hwe_p_min`) under the default `qc = TRUE`
#'       policy.}
#'     \item{plan}{Human-readable rendering of the whole plan (the same
#'       text `torchgenomics`'s `Recommendation.explain()` returns on the
#'       Python side); printed by `print()` / `cat()` on the returned
#'       object.}
#'   }
#'
#' @examples
#' \dontrun{
#' rec <- tg_recommend("pheno.tsv", "data.bed")
#' print(rec)                                    # prints rec$plan
#' tg_gwas("pheno.tsv", "data.bed", models = rec$suggested_model)
#' }
#' @export
tg_recommend <- function(phenotype, genotype, covariates = NULL) {
  d <- bridge_call("recommend", .compact(list(
    phenotype = .as_path(phenotype),
    genotype = .as_path(genotype),
    covariates = .as_path(covariates)
  )))
  structure(d, class = c("tg_recommendation", "list"))
}

#' @export
print.tg_recommendation <- function(x, ...) {
  cat(x$plan, "\n", sep = "")
  invisible(x)
}

#' List every model available to `tg_gwas(models = ...)`.
#'
#' Thin wrapper around `torchgenomics.models()` -- the friendly-API model
#' registry, distinct from the `torchgenomics.models` *subpackage*.
#' Unlike the other wrappers in this file, `tg_models()` does **not** go
#' through `bridge_call()`: the Python function returns a `pandas.DataFrame`
#' with no `.to_dict()`, and `DataFrame.to_dict()` (which `bridge_call()`
#' would call if the object had one) produces a nested
#' `{column: {row: value}}` shape that is the wrong orientation for an R
#' `data.frame`. Calling the Python function directly and converting via
#' `reticulate::py_to_r()` instead uses `reticulate`'s built-in
#' pandas-DataFrame-to-`data.frame` conversion (row-oriented, one R
#' `data.frame` row per pandas row), which is what callers actually want.
#'
#' `reticulate::py_to_r()`'s pandas dispatch is itself version-sensitive
#' (it keys off the Python class's reported module path, which has moved
#' across pandas releases); see [.pandas_df_to_r()] for the defensive
#' fallback this function uses when that fast path silently returns the
#' raw Python object instead of converting it.
#'
#' @return A `data.frame` with columns `alias`, `label`, `trait_types`,
#'   `description` -- one row per model [tg_gwas()] can run.
#'
#' @examples
#' \dontrun{
#' m <- tg_models()
#' "lmm" %in% m$alias
#' }
#' @export
tg_models <- function() {
  py <- tryCatch(tg_py(), tg_no_python = function(e) stop(e))
  py_fn <- py$api$models
  if (is.null(py_fn)) {
    stop(structure(
      class = c("tg_runtime_error", "error", "condition"),
      list(message = "torchgenomics.api has no function 'models'")
    ))
  }
  df <- tryCatch(
    py_fn(),
    python.builtin.Exception = function(e) {
      stop(structure(
        class = c("tg_runtime_error", "error", "condition"),
        list(message = paste0("models(): ", conditionMessage(e)))
      ))
    }
  )
  .pandas_df_to_r(df)
}

#' Defensively convert a pandas DataFrame to an R data.frame.
#'
#' `reticulate::py_to_r()` dispatches pandas-DataFrame conversion by S3
#' method name (`py_to_r.pandas.core.frame.DataFrame`), derived from the
#' Python object's reported module path at the time reticulate's method
#' table was built. That module path is not perfectly stable across
#' pandas releases (observed in the wild: pandas re-exporting `DataFrame`
#' such that its `__module__` resolves to `"pandas"` rather than
#' `"pandas.core.frame"`), which makes the registered method silently
#' fail to match and `py_to_r()` fall through to its identity default --
#' returning the raw Python `DataFrame` object instead of an R
#' `data.frame`, with **no error raised**. [tg_models()] hit exactly this
#' during real (non-mocked) integration testing against the managed
#' `r-rtorchgenomics` venv.
#'
#' This helper tries the normal `reticulate::py_to_r()` fast path first
#' (so nothing changes for callers on a reticulate/pandas combination
#' where it already works), and falls back to a manual, version-agnostic
#' conversion via `DataFrame.to_dict(orient = "list")` -- a plain Python
#' dict of `{column: [values...]}}`, whose dict/list conversion does not
#' depend on the pandas-version-sensitive S3 dispatch above -- only when
#' the fast path detectably didn't produce an R `data.frame`.
#'
#' @param df A `pandas.DataFrame` (or an already-converted R object, in
#'   which case it is returned via `as.data.frame()`, unchanged in
#'   substance).
#' @return An R `data.frame`, preserving `df`'s column order.
#' @keywords internal
.pandas_df_to_r <- function(df) {
  if (!inherits(df, "python.builtin.object")) {
    return(as.data.frame(df, stringsAsFactors = FALSE))
  }

  fast <- reticulate::py_to_r(df)
  if (is.data.frame(fast)) {
    return(fast)
  }

  cols <- as.character(unlist(
    reticulate::py_to_r(df$columns$tolist()), use.names = FALSE
  ))
  as_list <- reticulate::py_to_r(df$to_dict(orient = "list"))
  out <- lapply(cols, function(cn) unlist(as_list[[cn]], use.names = FALSE))
  names(out) <- cols
  as.data.frame(out, stringsAsFactors = FALSE)
}
