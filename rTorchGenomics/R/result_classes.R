# rTorchGenomics/R/result_classes.R

# .pkg_env is defined in zzz.R (Task 2 scaffold). %||% is in utils.R.

#' Shared virtual base class for all rTorchGenomics result objects.
#'
#' Internal virtual class; not instantiated directly. Concrete subclasses
#' (e.g., [ScanRun], [ValidateRun]) inherit its shared slots.
#'
#' @slot runtime_s Wall-clock runtime in seconds.
#' @slot output_files Named list of output file paths.
#' @slot log_excerpt Character vector of relevant log lines.
#' @keywords internal
#' @noRd
setClass("_BaseRun",
  representation(
    runtime_s = "numeric",
    output_files = "list",
    log_excerpt = "character",
    "VIRTUAL"
  )
)

#' Validate-run result (preflight checks).
#' @slot ok TRUE iff no errors.
#' @slot format_name Detected genotype format.
#' @slot n_samples_genotype Sample count from genotype file.
#' @slot n_samples_phenotype Sample count from phenotype file.
#' @slot n_samples_aligned Aligned sample count after intersection.
#' @slot n_variants_total Total variants in genotype file.
#' @slot n_traits Trait count.
#' @slot ploidy Detected ploidy.
#' @slot genotype_missingness_pct Genotype missingness percentage.
#' @slot phenotype_missingness_pct Phenotype missingness percentage.
#' @slot warnings Character vector of non-fatal warnings.
#' @slot errors Character vector of fatal errors.
#' @export
setClass("ValidateRun",
  contains = "_BaseRun",
  representation(
    ok = "logical",
    format_name = "character",
    n_samples_genotype = "integer",
    n_samples_phenotype = "integer",
    n_samples_aligned = "integer",
    n_variants_total = "integer",
    n_traits = "integer",
    ploidy = "integer",
    genotype_missingness_pct = "numeric",
    phenotype_missingness_pct = "numeric",
    warnings = "character",
    errors = "character"
  )
)

#' Scan-run result (lmm_scan, glm_scan).
#' @slot model Model name (e.g., SingleTraitLMM).
#' @slot test Test statistic name (wald, score, lrt).
#' @slot correction Multiple-testing correction (bh, by, bonferroni, ...).
#' @slot n_variants Variant count tested.
#' @slot n_significant Variant count exceeding significance threshold.
#' @slot significance_threshold P-value threshold.
#' @slot lambda_gc Genomic-control inflation factor.
#' @slot sigma2_g Additive genetic variance estimate.
#' @slot sigma2_e Residual variance estimate.
#' @slot h2 Narrow-sense heritability estimate.
#' @slot n_samples Sample count used.
#' @slot top_hits Tibble of top variants by p-value.
#' @export
setClass("ScanRun",
  contains = "_BaseRun",
  representation(
    model = "character",
    test = "character",
    correction = "character",
    n_variants = "integer",
    n_significant = "integer",
    significance_threshold = "numeric",
    lambda_gc = "numeric",
    sigma2_g = "numeric",
    sigma2_e = "numeric",
    h2 = "numeric",
    n_samples = "integer",
    top_hits = "data.frame"
  )
)

#' LD-blocks result.
#' @slot method LD block detection method.
#' @slot n_blocks Detected block count.
#' @slot n_variants_in_blocks Variants assigned to a block.
#' @slot median_block_size_bp Median block width in base pairs.
#' @slot median_block_n_snps Median variants per block.
#' @slot blocks Tibble of detected blocks (CHR / start / end / n_snps).
#' @export
setClass("LDBlocksRun",
  contains = "_BaseRun",
  representation(
    method = "character",
    n_blocks = "integer",
    n_variants_in_blocks = "integer",
    median_block_size_bp = "numeric",
    median_block_n_snps = "numeric",
    blocks = "data.frame"
  )
)

#' LD-clumping result.
#' @slot n_input_variants Variants in input sumstats.
#' @slot n_clumps Clumps formed.
#' @slot n_index_variants Index-variant count.
#' @slot p_threshold P-value threshold used.
#' @slot r2_threshold r^2 threshold used.
#' @slot clumps Tibble of clumps.
#' @export
setClass("ClumpRun",
  contains = "_BaseRun",
  representation(
    n_input_variants = "integer",
    n_clumps = "integer",
    n_index_variants = "integer",
    p_threshold = "numeric",
    r2_threshold = "numeric",
    clumps = "data.frame"
  )
)

#' Meta-analysis result.
#' @slot method Meta method (ivw / dl / stouffer / re2).
#' @slot n_studies Studies meta-analyzed.
#' @slot n_variants Variant count in combined sumstats.
#' @slot n_significant Variants meeting the significance threshold.
#' @slot results Tibble of per-variant meta results.
#' @slot heterogeneity_i2_median Median I^2 across variants.
#' @export
setClass("MetaRun",
  contains = "_BaseRun",
  representation(
    method = "character",
    n_studies = "integer",
    n_variants = "integer",
    n_significant = "integer",
    results = "data.frame",
    heterogeneity_i2_median = "numeric"
  )
)

#' PGS fit result.
#' @slot method PGS method (ct / ldpred2-* / prscs).
#' @slot n_variants_input Input variants.
#' @slot n_variants_used Variants used in fitting.
#' @slot n_variants_with_weights Variants assigned non-zero weight.
#' @slot h2 SNP heritability estimate.
#' @slot converged Convergence flag.
#' @slot diagnostics Method-specific diagnostics list.
#' @export
setClass("PgsFitRun",
  contains = "_BaseRun",
  representation(
    method = "character",
    n_variants_input = "integer",
    n_variants_used = "integer",
    n_variants_with_weights = "integer",
    h2 = "numeric",
    converged = "logical",
    diagnostics = "list"
  )
)

#' PGS scoring result.
#' @slot n_samples Scored individual count.
#' @slot n_variants_used Variants applied to scoring.
#' @slot score_mean Mean score across individuals.
#' @slot score_sd Score standard deviation.
#' @slot score_min Minimum score.
#' @slot score_max Maximum score.
#' @slot standardized TRUE iff scores were standardized.
#' @export
setClass("PgsScoreRun",
  contains = "_BaseRun",
  representation(
    n_samples = "integer",
    n_variants_used = "integer",
    score_mean = "numeric",
    score_sd = "numeric",
    score_min = "numeric",
    score_max = "numeric",
    standardized = "logical"
  )
)

#' Annotate-hits result.
#' @slot n_input_hits Input hit count.
#' @slot n_genes Unique annotated genes.
#' @slot crop Crop name (when used).
#' @slot assembly Assembly accession (when used).
#' @slot window_bp Annotation window in base pairs.
#' @slot hits Tibble of annotated hits.
#' @slot genes Tibble of unique genes.
#' @export
setClass("AnnotateRun",
  contains = "_BaseRun",
  representation(
    n_input_hits = "integer",
    n_genes = "integer",
    crop = "character",
    assembly = "character",
    window_bp = "integer",
    hits = "data.frame",
    genes = "data.frame"
  )
)

#' Convert result.
#' @slot input_format Source format.
#' @slot output_format Destination format.
#' @slot n_samples Sample count.
#' @slot n_variants Variant count.
#' @export
setClass("ConvertRun",
  contains = "_BaseRun",
  representation(
    input_format = "character",
    output_format = "character",
    n_samples = "integer",
    n_variants = "integer"
  )
)

#' Impute result.
#' @slot method Imputation method.
#' @slot n_samples Sample count.
#' @slot n_variants Variant count.
#' @slot n_imputed Imputed-genotype count.
#' @slot imputation_quality Aggregate quality metric (e.g., r^2).
#' @export
setClass("ImputeRun",
  contains = "_BaseRun",
  representation(
    method = "character",
    n_samples = "integer",
    n_variants = "integer",
    n_imputed = "integer",
    imputation_quality = "numeric"
  )
)

#' Plot result.
#' @slot figure Optional in-memory figure (matplotlib Figure or ggplot).
#' @slot png_path Path to a saved PNG, when written.
#' @slot png_base64 Base64-encoded PNG, when in-memory.
#' @slot width_px Pixel width.
#' @slot height_px Pixel height.
#' @export
setClass("PlotResult",
  contains = "_BaseRun",
  representation(
    figure = "ANY",
    png_path = "character",
    png_base64 = "character",
    width_px = "integer",
    height_px = "integer"
  )
)

#' Generic tier-2 result (auto-generated wrappers).
#' @slot command Name of the CLI subcommand.
#' @slot summary_text Pretty-printed summary string from Python.
#' @slot n_variants Variant count when reported.
#' @slot top_hits Top-hits tibble when reported.
#' @slot raw Full result dict (for accessors not covered by the slots).
#' @export
setClass("GwasResult",
  contains = "_BaseRun",
  representation(
    command = "character",
    summary_text = "character",
    n_variants = "integer",
    top_hits = "data.frame",
    raw = "list"
  )
)

# --- Helpers -----------------------------------------------------------

.empty_tibble <- function() tibble::tibble()

.tibble_from_list_of_dicts <- function(x) {
  if (is.null(x) || length(x) == 0L) return(.empty_tibble())
  tibble::as_tibble(do.call(rbind.data.frame, lapply(x, as.data.frame, stringsAsFactors = FALSE)))
}

.as_int <- function(x) if (is.null(x)) 0L else as.integer(x)
.as_num <- function(x) if (is.null(x)) NA_real_ else as.numeric(x)
.as_chr <- function(x) if (is.null(x)) NA_character_ else as.character(x)
.as_lgl <- function(x) if (is.null(x)) FALSE else as.logical(x)
.as_list <- function(x) if (is.null(x)) list() else as.list(x)

# Flatten a list-of-strings to a character vector. Always returns
# character(0) for NULL / empty input - `unlist(list(), use.names=FALSE)`
# returns NULL, which fails S4 slot type checks.
.as_chr_vec <- function(x) {
  if (is.null(x) || length(x) == 0L) return(character(0))
  as.character(unlist(x, use.names = FALSE))
}

# --- Constructors ------------------------------------------------------

#' Constructor for the ScanRun S4 class.
#'
#' Wrapper list exposing `$ScanRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name ScanRun
#' @keywords internal
#' @export
ScanRun <- list(
  new_from_dict = function(d) {
    new("ScanRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      model = .as_chr(d$model),
      test = .as_chr(d$test),
      correction = .as_chr(d$correction),
      n_variants = .as_int(d$n_variants),
      n_significant = .as_int(d$n_significant),
      significance_threshold = .as_num(d$significance_threshold),
      lambda_gc = .as_num(d$lambda_gc),
      sigma2_g = .as_num(d$sigma2_g),
      sigma2_e = .as_num(d$sigma2_e),
      h2 = .as_num(d$h2),
      n_samples = .as_int(d$n_samples),
      top_hits = .tibble_from_list_of_dicts(d$top_hits)
    )
  }
)

#' Constructor for the ValidateRun S4 class.
#'
#' Wrapper list exposing `$ValidateRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name ValidateRun
#' @keywords internal
#' @export
ValidateRun <- list(
  new_from_dict = function(d) {
    new("ValidateRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      ok = .as_lgl(d$ok),
      format_name = .as_chr(d$format_name),
      n_samples_genotype = .as_int(d$n_samples_genotype),
      n_samples_phenotype = .as_int(d$n_samples_phenotype),
      n_samples_aligned = .as_int(d$n_samples_aligned),
      n_variants_total = .as_int(d$n_variants_total),
      n_traits = .as_int(d$n_traits),
      ploidy = .as_int(d$ploidy),
      genotype_missingness_pct = .as_num(d$genotype_missingness_pct),
      phenotype_missingness_pct = .as_num(d$phenotype_missingness_pct),
      warnings = .as_chr_vec(d$warnings),
      errors = .as_chr_vec(d$errors)
    )
  }
)

#' Constructor for the LDBlocksRun S4 class.
#'
#' Wrapper list exposing `$LDBlocksRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name LDBlocksRun
#' @keywords internal
#' @export
LDBlocksRun <- list(
  new_from_dict = function(d) {
    new("LDBlocksRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      method = .as_chr(d$method),
      n_blocks = .as_int(d$n_blocks),
      n_variants_in_blocks = .as_int(d$n_variants_in_blocks),
      median_block_size_bp = .as_num(d$median_block_size_bp),
      median_block_n_snps = .as_num(d$median_block_n_snps),
      blocks = .tibble_from_list_of_dicts(d$blocks)
    )
  }
)

#' Constructor for the ClumpRun S4 class.
#'
#' Wrapper list exposing `$ClumpRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name ClumpRun
#' @keywords internal
#' @export
ClumpRun <- list(
  new_from_dict = function(d) {
    new("ClumpRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      n_input_variants = .as_int(d$n_input_variants),
      n_clumps = .as_int(d$n_clumps),
      n_index_variants = .as_int(d$n_index_variants),
      p_threshold = .as_num(d$p_threshold),
      r2_threshold = .as_num(d$r2_threshold),
      clumps = .tibble_from_list_of_dicts(d$clumps)
    )
  }
)

#' Constructor for the MetaRun S4 class.
#'
#' Wrapper list exposing `$MetaRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name MetaRun
#' @keywords internal
#' @export
MetaRun <- list(
  new_from_dict = function(d) {
    new("MetaRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      method = .as_chr(d$method),
      n_studies = .as_int(d$n_studies),
      n_variants = .as_int(d$n_variants),
      n_significant = .as_int(d$n_significant),
      results = .tibble_from_list_of_dicts(d$results),
      heterogeneity_i2_median = .as_num(d$heterogeneity_i2_median)
    )
  }
)

#' Constructor for the PgsFitRun S4 class.
#'
#' Wrapper list exposing `$PgsFitRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name PgsFitRun
#' @keywords internal
#' @export
PgsFitRun <- list(
  new_from_dict = function(d) {
    new("PgsFitRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      method = .as_chr(d$method),
      n_variants_input = .as_int(d$n_variants_input),
      n_variants_used = .as_int(d$n_variants_used),
      n_variants_with_weights = .as_int(d$n_variants_with_weights),
      h2 = .as_num(d$h2),
      converged = .as_lgl(d$converged),
      diagnostics = .as_list(d$diagnostics)
    )
  }
)

#' Constructor for the PgsScoreRun S4 class.
#'
#' Wrapper list exposing `$PgsScoreRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name PgsScoreRun
#' @keywords internal
#' @export
PgsScoreRun <- list(
  new_from_dict = function(d) {
    new("PgsScoreRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      n_samples = .as_int(d$n_samples),
      n_variants_used = .as_int(d$n_variants_used),
      score_mean = .as_num(d$score_mean),
      score_sd = .as_num(d$score_sd),
      score_min = .as_num(d$score_min),
      score_max = .as_num(d$score_max),
      standardized = .as_lgl(d$standardized)
    )
  }
)

#' Constructor for the AnnotateRun S4 class.
#'
#' Wrapper list exposing `$AnnotateRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name AnnotateRun
#' @keywords internal
#' @export
AnnotateRun <- list(
  new_from_dict = function(d) {
    new("AnnotateRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      n_input_hits = .as_int(d$n_input_hits),
      n_genes = .as_int(d$n_genes),
      crop = .as_chr(d$crop),
      assembly = .as_chr(d$assembly),
      window_bp = .as_int(d$window_bp),
      hits = .tibble_from_list_of_dicts(d$hits),
      genes = .tibble_from_list_of_dicts(d$genes)
    )
  }
)

#' Constructor for the ConvertRun S4 class.
#'
#' Wrapper list exposing `$ConvertRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name ConvertRun
#' @keywords internal
#' @export
ConvertRun <- list(
  new_from_dict = function(d) {
    new("ConvertRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      input_format = .as_chr(d$input_format),
      output_format = .as_chr(d$output_format),
      n_samples = .as_int(d$n_samples),
      n_variants = .as_int(d$n_variants)
    )
  }
)

#' Constructor for the ImputeRun S4 class.
#'
#' Wrapper list exposing `$ImputeRun$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name ImputeRun
#' @keywords internal
#' @export
ImputeRun <- list(
  new_from_dict = function(d) {
    new("ImputeRun",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      method = .as_chr(d$method),
      n_samples = .as_int(d$n_samples),
      n_variants = .as_int(d$n_variants),
      n_imputed = .as_int(d$n_imputed),
      imputation_quality = .as_num(d$imputation_quality)
    )
  }
)

#' Constructor for the PlotResult S4 class.
#'
#' Wrapper list exposing `$PlotResult$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name PlotResult
#' @keywords internal
#' @export
PlotResult <- list(
  new_from_dict = function(d, figure = NULL) {
    new("PlotResult",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      figure = figure,
      png_path = .as_chr(d$png_path),
      png_base64 = .as_chr(d$png_base64),
      width_px = .as_int(d$width_px),
      height_px = .as_int(d$height_px)
    )
  }
)

#' Constructor for the GwasResult S4 class.
#'
#' Wrapper list exposing `$GwasResult$new_from_dict(d)` for building an instance
#' from a JSON-shaped list (the dict returned by `py_to_r(to_dict())`).
#' @name GwasResult
#' @keywords internal
#' @export
GwasResult <- list(
  new_from_dict = function(d, command) {
    new("GwasResult",
      runtime_s = .as_num(d$runtime_s),
      output_files = .as_list(d$output_files),
      log_excerpt = .as_chr_vec(d$log_excerpt),
      command = command,
      summary_text = .as_chr(d$summary %||% d$summary_text),
      n_variants = .as_int(d$n_variants),
      top_hits = .tibble_from_list_of_dicts(d$top_hits),
      raw = d
    )
  }
)

# --- show methods ------------------------------------------------------

setMethod("show", "ScanRun", function(object) {
  cat(sprintf("ScanRun (%s, test=%s, correction=%s)\n",
              object@model, object@test, object@correction))
  cat(sprintf("  Samples: %d   Variants tested: %d\n",
              object@n_samples, object@n_variants))
  cat(sprintf("  Significant @ \u03b1=%.2e: %d\n",
              object@significance_threshold, object@n_significant))
  if (!is.na(object@lambda_gc)) {
    cat(sprintf("  \u03bb_GC: %.4f\n", object@lambda_gc))
  }
  if (!is.na(object@h2)) {
    cat(sprintf("  Variance components: \u03c3\u00b2_g=%.4f, \u03c3\u00b2_e=%.4f, h\u00b2=%.4f\n",
                object@sigma2_g, object@sigma2_e, object@h2))
  }
  if (nrow(object@top_hits) > 0L) {
    cat(sprintf("  Top hits (%d shown):\n", min(10L, nrow(object@top_hits))))
    print(utils::head(object@top_hits, 10L))
  }
  cat(sprintf("  Runtime: %.1fs\n", object@runtime_s))
  invisible(object)
})

setMethod("show", "ValidateRun", function(object) {
  cat(sprintf("ValidateRun (ok=%s)\n", object@ok))
  cat(sprintf("  Format: %s\n", object@format_name))
  cat(sprintf("  Samples: %d genotype, %d phenotype, %d aligned\n",
              object@n_samples_genotype, object@n_samples_phenotype,
              object@n_samples_aligned))
  cat(sprintf("  Variants: %d   Traits: %d\n",
              object@n_variants_total, object@n_traits))
  if (length(object@warnings) > 0L) {
    cat(sprintf("  Warnings (%d):\n", length(object@warnings)))
    for (w in object@warnings) cat("    -", w, "\n")
  }
  if (length(object@errors) > 0L) {
    cat(sprintf("  ERRORS (%d):\n", length(object@errors)))
    for (e in object@errors) cat("    -", e, "\n")
  }
  cat(sprintf("  Runtime: %.1fs\n", object@runtime_s))
  invisible(object)
})

setMethod("show", "GwasResult", function(object) {
  cat(sprintf("GwasResult (command=%s)\n", object@command))
  if (!is.na(object@summary_text)) cat("  ", object@summary_text, "\n", sep = "")
  if (object@n_variants > 0L) {
    cat(sprintf("  Variants: %d\n", object@n_variants))
  }
  if (nrow(object@top_hits) > 0L) {
    cat(sprintf("  Top hits (%d shown):\n", min(10L, nrow(object@top_hits))))
    print(utils::head(object@top_hits, 10L))
  }
  cat(sprintf("  Runtime: %.1fs\n", object@runtime_s))
  invisible(object)
})

# One-line show methods for the remaining 9 classes.
setMethod("show", "LDBlocksRun", function(object) {
  cat(sprintf("LDBlocksRun (method=%s, n_blocks=%d, median_size=%.1f kb, runtime=%.1fs)\n",
              object@method, object@n_blocks,
              object@median_block_size_bp / 1000, object@runtime_s))
  invisible(object)
})

setMethod("show", "ClumpRun", function(object) {
  cat(sprintf("ClumpRun (input=%d \u2192 %d clumps; p<%.2e, r\u00b2<%.2f; runtime=%.1fs)\n",
              object@n_input_variants, object@n_clumps,
              object@p_threshold, object@r2_threshold, object@runtime_s))
  invisible(object)
})

setMethod("show", "MetaRun", function(object) {
  cat(sprintf("MetaRun (method=%s, K=%d, n_variants=%d, n_sig=%d, runtime=%.1fs)\n",
              object@method, object@n_studies, object@n_variants,
              object@n_significant, object@runtime_s))
  invisible(object)
})

setMethod("show", "PgsFitRun", function(object) {
  cat(sprintf("PgsFitRun (method=%s, n_with_weights=%d, h\u00b2=%s, converged=%s, runtime=%.1fs)\n",
              object@method, object@n_variants_with_weights,
              ifelse(is.na(object@h2), "NA", sprintf("%.4f", object@h2)),
              object@converged, object@runtime_s))
  invisible(object)
})

setMethod("show", "PgsScoreRun", function(object) {
  cat(sprintf("PgsScoreRun (n_samples=%d, n_variants=%d, mean=%.4f, sd=%.4f, runtime=%.1fs)\n",
              object@n_samples, object@n_variants_used,
              object@score_mean, object@score_sd, object@runtime_s))
  invisible(object)
})

setMethod("show", "AnnotateRun", function(object) {
  target <- if (!is.na(object@crop)) object@crop else if (!is.na(object@assembly)) object@assembly else "(unknown)"
  cat(sprintf("AnnotateRun (n_hits=%d, n_genes=%d, target=%s, runtime=%.1fs)\n",
              object@n_input_hits, object@n_genes, target, object@runtime_s))
  invisible(object)
})

setMethod("show", "ConvertRun", function(object) {
  cat(sprintf("ConvertRun (%s \u2192 %s, n=%d \u00d7 %d, runtime=%.1fs)\n",
              object@input_format, object@output_format,
              object@n_samples, object@n_variants, object@runtime_s))
  invisible(object)
})

setMethod("show", "ImputeRun", function(object) {
  cat(sprintf("ImputeRun (method=%s, n=%d \u00d7 %d, imputed=%d, runtime=%.1fs)\n",
              object@method, object@n_samples, object@n_variants,
              object@n_imputed, object@runtime_s))
  invisible(object)
})

setMethod("show", "PlotResult", function(object) {
  path <- if (!is.na(object@png_path) && nzchar(object@png_path)) object@png_path else "(memory)"
  cat(sprintf("PlotResult (%dx%dpx, file=%s, runtime=%.1fs)\n",
              object@width_px, object@height_px, path, object@runtime_s))
  invisible(object)
})

# --- Accessors ---------------------------------------------------------

#' Extract the top-hits tibble from a result object.
#' @param x A ScanRun or GwasResult.
#' @return A tibble.
#' @export
#' @rdname top_hits
setGeneric("top_hits", function(x) standardGeneric("top_hits"))

#' @rdname top_hits
setMethod("top_hits", "ScanRun", function(x) x@top_hits)

#' @rdname top_hits
setMethod("top_hits", "GwasResult", function(x) x@top_hits)

#' Extract output file paths from a result object.
#' @param x A result object.
#' @return A named list of paths.
#' @export
#' @rdname output_files
setGeneric("output_files", function(x) standardGeneric("output_files"))

#' @rdname output_files
setMethod("output_files", "_BaseRun", function(x) x@output_files)
