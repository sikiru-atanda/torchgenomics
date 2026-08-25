# Tier-1 api wrappers (hand-crafted).
#
# Each function:
#   - has a typed R signature with match.arg() / as.integer() coercion
#   - validates required args (e.g. tg_meta needs >=2 inputs;
#     tg_annotate_hits needs one of crop/taxid/assembly)
#   - builds an arg list via .compact(list(...)) to drop NULL options
#   - calls bridge_call("<fn>", args)
#   - wraps the dict result in the matching S4 class via
#     <ClassName>$new_from_dict(d)
#
# The two plot-based wrappers (`tg_manhattan`, `tg_qq`) ship in Task 7
# (R/plots.R) — they read result files from disk and build ggplots.

#' Validate a TorchGenomics dataset.
#'
#' Pre-flight check: format detection, sample alignment across files,
#' duplicate IDs, missingness, warnings, errors.
#'
#' @param genotype Path to genotype file (any supported format).
#' @param phenotype Path to phenotype TSV.
#' @param covariate Optional path to covariate TSV.
#' @param ploidy Expected ploidy. Default 2.
#' @param output Optional path to write a JSON report.
#'
#' @return A `ValidateRun` S4 object. `ok` slot is TRUE iff no errors.
#'
#' @examples
#' \dontrun{
#' r <- tg_validate("data.bed", "pheno.tsv")
#' stopifnot(r@ok)
#' }
#' @export
tg_validate <- function(genotype, phenotype,
                        covariate = NULL, ploidy = 2L,
                        output = NULL) {
  d <- bridge_call("validate", .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    ploidy = as.integer(ploidy),
    output = .as_path(output)
  )))
  ValidateRun$new_from_dict(d)
}

#' Convert a genotype file between formats.
#'
#' @param input Source file (any supported format).
#' @param output Destination path.
#' @param output_format `"bed"`, `"zarr"`, or `"vcf"`.
#' @param sample Optional sample metadata file.
#' @param map Optional map file with positions.
#'
#' @return A `ConvertRun` S4 object.
#' @examples
#' \dontrun{
#' tg_convert("data.vcf.gz", "data", output_format = "bed")
#' }
#' @export
tg_convert <- function(input, output, output_format = c("bed", "zarr", "vcf"),
                       sample = NULL, map = NULL) {
  output_format <- match.arg(output_format)
  d <- bridge_call("convert", .compact(list(
    input = .as_path(input), output = .as_path(output),
    output_format = output_format,
    sample = .as_path(sample), map = .as_path(map)
  )))
  ConvertRun$new_from_dict(d)
}

#' Impute missing genotypes.
#'
#' @param genotype Path to genotype file with missing dosages.
#' @param output Destination path for imputed dosages.
#' @param method One of `"mean"`, `"mode"`, `"knn"`, `"ld"`,
#'   `"li-stephens"`, `"deep-learning"`.
#' @param ploidy Ploidy. Default 2.
#' @param chunk_size Variants per streaming chunk. Default 1024.
#' @param window_size Sliding-window size for `"ld"`. Default 50.
#'
#' @return An `ImputeRun` S4 object.
#' @examples
#' \dontrun{
#' tg_impute("data.bed", "imputed.pt", method = "mean")
#' }
#' @export
tg_impute <- function(genotype, output,
                      method = c("mean", "mode", "knn", "ld",
                                 "li-stephens", "deep-learning"),
                      ploidy = 2L, chunk_size = 1024L, window_size = 50L) {
  method <- match.arg(method)
  d <- bridge_call("impute", .compact(list(
    genotype = .as_path(genotype), output = .as_path(output),
    method = method,
    ploidy = as.integer(ploidy),
    chunk_size = as.integer(chunk_size),
    window_size = as.integer(window_size)
  )))
  ImputeRun$new_from_dict(d)
}

#' Single-trait LMM GWAS scan.
#'
#' @param genotype Path to genotype file.
#' @param phenotype Path to phenotype TSV.
#' @param trait Optional trait column name (auto-picks first if multiple).
#' @param covariate Optional covariate TSV.
#' @param output Output directory or prefix.
#' @param test `"wald"`, `"score"`, or `"lrt"`.
#' @param correction Multiple-testing correction. Default `"bh"`.
#' @param chunk_size Variants per streaming chunk.
#' @param maf_min Per-variant MAF lower bound.
#' @param miss_max Per-variant missingness upper bound.
#' @param device `"cpu"`, `"cuda"`, or `"auto"`.
#' @param grm Optional pre-computed GRM file.
#' @param grm_method `"vanraden"` (default) or `"zhang"`.
#' @param n_pcs Number of PCs from GRM to add as covariates.
#' @param p3d Plug-in covariance (re-use null variance per variant).
#' @param significance_threshold P-value cutoff for n_significant.
#' @param top_k Number of top hits returned inline.
#'
#' @return A `ScanRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_lmm_scan("data.bed", "pheno.tsv")
#' print(r); tg_manhattan(r)
#' }
#' @export
tg_lmm_scan <- function(genotype, phenotype,
                        trait = NULL, covariate = NULL, output = NULL,
                        test = c("wald", "score", "lrt"),
                        correction = "bh",
                        chunk_size = 10000L,
                        maf_min = 0.01, miss_max = 0.1,
                        device = c("auto", "cpu", "cuda"),
                        grm = NULL, grm_method = c("vanraden", "zhang"),
                        n_pcs = 0L, p3d = TRUE,
                        significance_threshold = 5e-8,
                        top_k = 50L) {
  test <- match.arg(test)
  device <- match.arg(device)
  grm_method <- match.arg(grm_method)
  d <- bridge_call("lmm_scan", .compact(list(
    genotype = .as_path(genotype), phenotype = .as_path(phenotype),
    trait = trait, covariate = .as_path(covariate),
    output = .as_path(output),
    test = test, correction = correction,
    chunk_size = as.integer(chunk_size),
    maf_min = maf_min, miss_max = miss_max,
    device = device,
    grm = .as_path(grm), grm_method = grm_method,
    n_pcs = as.integer(n_pcs), p3d = p3d,
    significance_threshold = significance_threshold,
    top_k = as.integer(top_k)
  )))
  ScanRun$new_from_dict(d)
}

#' GLM-family GWAS scan.
#'
#' Gaussian / binary / ordinal / multinomial.
#'
#' @inheritParams tg_lmm_scan
#' @param family `"gaussian"`, `"binary"`, `"ordinal"`, or `"multinomial"`.
#' @param n_categories Required for ordinal / multinomial.
#' @param firth Use Firth penalization (binary/ordinal only).
#'
#' @return A `ScanRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_glm_scan("data.bed", "pheno.tsv", family = "binary", firth = TRUE)
#' }
#' @export
tg_glm_scan <- function(genotype, phenotype,
                        trait = NULL, covariate = NULL, output = NULL,
                        family = c("gaussian", "binary", "ordinal", "multinomial"),
                        n_categories = NULL, firth = FALSE,
                        test = c("wald", "score", "lrt"),
                        correction = "bh",
                        chunk_size = 10000L,
                        maf_min = 0.01, miss_max = 0.1,
                        device = c("auto", "cpu", "cuda"),
                        significance_threshold = 5e-8,
                        top_k = 50L) {
  family <- match.arg(family)
  test <- match.arg(test)
  device <- match.arg(device)
  d <- bridge_call("glm_scan", .compact(list(
    genotype = .as_path(genotype), phenotype = .as_path(phenotype),
    trait = trait, covariate = .as_path(covariate),
    output = .as_path(output),
    family = family,
    n_categories = if (is.null(n_categories)) NULL else as.integer(n_categories),
    firth = firth,
    test = test, correction = correction,
    chunk_size = as.integer(chunk_size),
    maf_min = maf_min, miss_max = miss_max,
    device = device,
    significance_threshold = significance_threshold,
    top_k = as.integer(top_k)
  )))
  ScanRun$new_from_dict(d)
}

#' Detect LD haplotype blocks.
#'
#' @param genotype Path to genotype file.
#' @param output Output directory or prefix.
#' @param method One of 13 block-detection algorithms (see Python docs).
#' @param max_kb Maximum block size in kb. Default 200.
#' @param r2_threshold r-squared threshold (used by `r2`, `big_ld`, `cc_graph`).
#' @param ci_low,ci_high CI bounds (used by `gabriel`).
#' @param tolerance Integer >= 0. For \code{method = "r2"} only:
#'   number of consecutive low-LD markers tolerated within a sliding
#'   block before terminating it (SelectionTools-style; Wittenburg et al.
#'   2024). \code{tolerance = 0} (the default) reproduces the original
#'   strict r^2-greedy partition; \code{tolerance >= 1} permits
#'   short "gaps" of weakly linked markers inside an otherwise tight
#'   block, which often yields larger, more biologically meaningful
#'   haplo-blocks for breeding programs.
#' @param ... Additional method-specific knobs.
#'
#' @return An `LDBlocksRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_ld_blocks("data.bed", method = "gabriel")
#' # Sliding-window r^2 with tolerance for short low-LD gaps:
#' r2 <- tg_ld_blocks("data.bed", method = "r2",
#'                    r2_threshold = 0.7, tolerance = 2)
#' }
#' @export
tg_ld_blocks <- function(genotype, output = NULL,
                         method = c("gabriel", "four_gamete", "spine", "r2",
                                    "gwas_aligned", "uncertainty", "cross_pop",
                                    "graphical", "changepoint", "big_ld",
                                    "cc_graph", "dp_optimize", "wall_pritchard"),
                         max_kb = 200L,
                         r2_threshold = 0.5,
                         ci_low = 0.7, ci_high = 0.98,
                         tolerance = 0L,
                         ...) {
  method <- match.arg(method)
  if (!is.null(tolerance)) {
    tolerance <- as.integer(tolerance)
    if (length(tolerance) != 1L || is.na(tolerance) || tolerance < 0L) {
      stop("tolerance must be a non-negative integer scalar.")
    }
  }
  extras <- list(...)
  d <- bridge_call("ld_blocks", .compact(c(list(
    genotype = .as_path(genotype),
    output = .as_path(output),
    method = method,
    max_kb = as.integer(max_kb),
    r2_threshold = r2_threshold,
    ci_low = ci_low, ci_high = ci_high,
    tolerance = tolerance
  ), extras)))
  LDBlocksRun$new_from_dict(d)
}

#' LD clumping.
#'
#' @param sumstats Path to summary statistics TSV.
#' @param genotype Path to LD-reference genotype panel.
#' @param output Output directory or prefix.
#' @param p_threshold P-value cutoff for index variants.
#' @param r2 r-squared ceiling for clumping.
#' @param window_kb Clumping window in kb.
#'
#' @return A `ClumpRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_clump("gwas.tsv", "ld.bed", p_threshold = 5e-8, r2 = 0.1)
#' }
#' @export
tg_clump <- function(sumstats, genotype, output = NULL,
                     p_threshold = 5e-8, r2 = 0.1, window_kb = 250) {
  d <- bridge_call("clump", .compact(list(
    sumstats = .as_path(sumstats),
    genotype = .as_path(genotype),
    output = .as_path(output),
    p_threshold = p_threshold,
    r2 = r2, window_kb = window_kb
  )))
  ClumpRun$new_from_dict(d)
}

#' Meta-analysis across GWAS sumstats.
#'
#' @param inputs Character vector of sumstats paths (length >= 2).
#' @param output Output directory or prefix.
#' @param method `"fixed"`, `"random"`, `"han_eskin"`, or `"stouffer"`.
#' @param significance_threshold P-value cutoff.
#'
#' @return A `MetaRun` S4 object.
#' @examples
#' \dontrun{
#' tg_meta(c("study1.tsv", "study2.tsv"), method = "fixed")
#' }
#' @export
tg_meta <- function(inputs, output = NULL,
                    method = c("fixed", "random", "han_eskin", "stouffer"),
                    significance_threshold = 5e-8) {
  method <- match.arg(method)
  if (length(inputs) < 2L) {
    stop("tg_meta requires at least 2 input sumstats files.")
  }
  d <- bridge_call("meta", .compact(list(
    inputs = as.character(inputs),
    output = .as_path(output),
    method = method,
    significance_threshold = significance_threshold
  )))
  MetaRun$new_from_dict(d)
}

#' Two-sample Mendelian randomization.
#'
#' @param exposure Path to instrument sumstats TSV for the exposure trait
#'   (columns chr/pos/snp/a1/a2/beta/se/p, harmonized to the same effect
#'   allele).
#' @param outcome Path to instrument sumstats TSV for the outcome trait,
#'   on the same SNP panel as `exposure`.
#' @param method `"ivw"`, `"egger"`, `"weighted_median"`, `"presso"`, or
#'   `"all"` (runs the full sensitivity panel, one row per method).
#' @param output Optional path to write the results TSV.
#' @param n_boot,n_perm Bootstrap / permutation controls for
#'   weighted-median and MR-PRESSO.
#' @param seed RNG seed.
#'
#' @return An `MRRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_mr("exposure.tsv", "outcome.tsv", method = "all")
#' }
#' @export
tg_mr <- function(exposure, outcome,
                  method = c("ivw", "egger", "weighted_median", "presso", "all"),
                  output = NULL, n_boot = 1000L, n_perm = 1000L, seed = 42L) {
  method <- match.arg(method)
  d <- bridge_call("mr", .compact(list(
    exposure = .as_path(exposure), outcome = .as_path(outcome),
    method = method, output = .as_path(output),
    n_boot = as.integer(n_boot), n_perm = as.integer(n_perm),
    seed = as.integer(seed))))
  MRRun$new_from_dict(d)
}

#' Colocalization (pairwise Giambartolomei or N-trait hyprcoloc).
#'
#' @param sumstats For `method = "pairwise"`: a path to the first sumstats
#'   TSV. For `method = "hyprcoloc"`: a character vector of >= 2 sumstats
#'   paths.
#' @param sumstats2 The second sumstats path for `method = "pairwise"`
#'   (required there; must be `NULL` for hyprcoloc).
#' @param method `"pairwise"` (Giambartolomei 2-trait, PP.H0-H4) or
#'   `"hyprcoloc"` (N-trait).
#' @param output Optional path to write a posteriors TSV.
#' @param prior_1,prior_2,prior_12 Coloc priors, forwarded to the pairwise
#'   method. For `method = "hyprcoloc"` only `prior_1` is forwarded
#'   (hyprcoloc's own defaults apply to the rest); `prior_12` has no
#'   hyprcoloc analogue and is ignored there.
#' @param trait_names Optional trait labels, forwarded to hyprcoloc only.
#'
#' @return A `ColocRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_coloc("a.tsv", "b.tsv", method = "pairwise")
#' r2 <- tg_coloc(c("a.tsv", "b.tsv", "c.tsv"), method = "hyprcoloc")
#' }
#' @export
tg_coloc <- function(sumstats, sumstats2 = NULL,
                     method = c("pairwise", "hyprcoloc"), output = NULL,
                     prior_1 = 1e-4, prior_2 = 1e-4, prior_12 = 1e-5,
                     trait_names = NULL) {
  method <- match.arg(method)
  d <- bridge_call("coloc", .compact(list(
    sumstats = if (length(sumstats) > 1) as.character(sumstats) else .as_path(sumstats),
    sumstats2 = .as_path(sumstats2), method = method, output = .as_path(output),
    prior_1 = prior_1, prior_2 = prior_2, prior_12 = prior_12,
    trait_names = trait_names)))
  ColocRun$new_from_dict(d)
}

#' Fit polygenic-score weights.
#'
#' @param sumstats Path to GWAS sumstats.
#' @param ld_ref Path to LD reference panel.
#' @param output Path to write fitted weights.
#' @param method `"ct"`, `"ldpred2-inf"`, `"ldpred2-grid"`,
#'   `"ldpred2-auto"`, or `"prscs"`.
#' @param h2 Heritability prior / fixed value.
#' @param p_causal Proportion of causal SNPs prior.
#' @param n_iter,n_burnin,n_chains MCMC controls.
#' @param device `"cpu"`, `"cuda"`, or `"auto"`.
#' @param seed Optional integer RNG seed.
#' @param ... Method-specific args (see Python docs).
#'
#' @return A `PgsFitRun` S4 object.
#' @examples
#' \dontrun{
#' tg_pgs_fit("ss.tsv", "ld.pt", "weights.tsv", method = "ldpred2-auto")
#' }
#' @export
tg_pgs_fit <- function(sumstats, ld_ref, output,
                       method = c("ldpred2-auto", "ct", "ldpred2-inf",
                                  "ldpred2-grid", "prscs"),
                       h2 = NULL, p_causal = NULL,
                       n_iter = 1000L, n_burnin = 500L, n_chains = 3L,
                       device = c("auto", "cpu", "cuda"),
                       seed = NULL, ...) {
  method <- match.arg(method)
  device <- match.arg(device)
  extras <- list(...)
  d <- bridge_call("pgs_fit", .compact(c(list(
    sumstats = .as_path(sumstats),
    ld_ref = .as_path(ld_ref),
    output = .as_path(output),
    method = method, h2 = h2, p_causal = p_causal,
    n_iter = as.integer(n_iter),
    n_burnin = as.integer(n_burnin),
    n_chains = as.integer(n_chains),
    device = device,
    seed = if (is.null(seed)) NULL else as.integer(seed)
  ), extras)))
  PgsFitRun$new_from_dict(d)
}

#' Apply PGS weights to target genotypes.
#'
#' @param genotype Path to target genotype file.
#' @param weights Path to fitted weights (from `tg_pgs_fit`).
#' @param output Path to per-individual scores TSV.
#' @param standardize Standardize the scores.
#' @param handle_missing `"mean"`, `"zero"`, or `"drop"`.
#' @param chunk_size Variants per streaming chunk.
#' @param device `"cpu"`, `"cuda"`, or `"auto"`.
#'
#' @return A `PgsScoreRun` S4 object.
#' @examples
#' \dontrun{
#' tg_pgs_score("target.bed", "weights.tsv", "scores.tsv")
#' }
#' @export
tg_pgs_score <- function(genotype, weights, output,
                         standardize = FALSE,
                         handle_missing = c("mean", "zero", "drop"),
                         chunk_size = 10000L,
                         device = c("auto", "cpu", "cuda")) {
  handle_missing <- match.arg(handle_missing)
  device <- match.arg(device)
  d <- bridge_call("pgs_score", .compact(list(
    genotype = .as_path(genotype),
    weights = .as_path(weights),
    output = .as_path(output),
    standardize = standardize,
    handle_missing = handle_missing,
    chunk_size = as.integer(chunk_size),
    device = device
  )))
  PgsScoreRun$new_from_dict(d)
}

#' Annotate GWAS hits with nearby genes via NCBI.
#'
#' @param sumstats Path to sumstats.
#' @param crop Crop name (e.g., `"maize"`). Provide one of crop / taxid / assembly.
#' @param taxid NCBI taxonomy ID.
#' @param assembly NCBI assembly accession.
#' @param output Output directory or prefix.
#' @param p_threshold P-value cutoff for hits to annotate.
#' @param window_up,window_down Flanking window in bp.
#' @param include_go Include GO terms.
#' @param include_orthologs Include ortholog mappings.
#' @param ortholog_taxa Character vector of taxon IDs for orthologs.
#' @param api_key Optional NCBI API key.
#'
#' @return An `AnnotateRun` S4 object.
#' @examples
#' \dontrun{
#' tg_annotate_hits("hits.tsv", crop = "maize")
#' }
#' @export
tg_annotate_hits <- function(sumstats,
                             crop = NULL, taxid = NULL, assembly = NULL,
                             output = NULL,
                             p_threshold = 5e-8,
                             window_up = 50000L, window_down = 50000L,
                             include_go = TRUE, include_orthologs = FALSE,
                             ortholog_taxa = NULL, api_key = NULL) {
  if (is.null(crop) && is.null(taxid) && is.null(assembly)) {
    stop("tg_annotate_hits requires one of: crop, taxid, assembly.")
  }
  d <- bridge_call("annotate_hits", .compact(list(
    sumstats = .as_path(sumstats),
    crop = crop,
    taxid = if (is.null(taxid)) NULL else as.integer(taxid),
    assembly = assembly,
    output = .as_path(output),
    p_threshold = p_threshold,
    window_up = as.integer(window_up),
    window_down = as.integer(window_down),
    include_go = include_go, include_orthologs = include_orthologs,
    ortholog_taxa = if (is.null(ortholog_taxa)) NULL else as.list(as.integer(ortholog_taxa)),
    api_key = api_key
  )))
  AnnotateRun$new_from_dict(d)
}

#' Local Genomic Estimated Breeding Values per haplo-block.
#'
#' Fits ridge-regression BLUP (rrBLUP; Endelman 2011 *Plant Genome*) on
#' a pre-computed BLUE vector, sums per-marker effects within each
#' haplo-block, and classifies blocks as favorable / unfavorable based
#' on effect sign. Applied to breeding-program multi-environment
#' workflows (Pandit et al. 2026, *Theoretical and Applied Genetics*).
#'
#' BLUEs are produced upstream (e.g. ASReml-R's FA(k) MET fit with
#' genotype as fixed). This wrapper does NOT compute them.
#'
#' @param y Numeric vector of BLUEs (length n_genotypes).
#' @param genotype Path to genotype file OR an n_genotypes x n_SNPs
#'   numeric matrix.
#' @param blocks A list of haplo-block specifications (each with
#'   `variant_indices`) OR an `LDBlocksRun` returned by
#'   `tg_ld_blocks()` (in which case its on-disk TSV path is used).
#' @param h2 Heritability prior (0 < h2 < 1); if NULL, REML-estimated.
#' @param favorable_direction One of `"negative"` (resistance traits;
#'   default) or `"positive"` (yield-style traits).
#' @param standardize_G If TRUE, center and ploidy-aware-scale G before fitting.
#' @param method One of `"rrblup"` (default) or `"gblup"`.
#' @param device `"cpu"`, `"cuda"`, or `"auto"`.
#' @param return_marker_effects If TRUE, also return per-marker effects.
#'
#' @return An `LGEBVResult` S4 object with slots `block_id`, `chrom`,
#'   `start`, `end`, `n_variants`, `lgebv`, `block_variance`,
#'   `favorable`, `marker_effects`, `h2_used`, `method`.
#'
#' @examples
#' \dontrun{
#' blocks <- tg_ld_blocks("data.bed", method = "r2",
#'                        r2_threshold = 0.7, tolerance = 2)
#' result <- tg_lgebv(y = blues, genotype = "data.bed",
#'                    blocks = blocks, h2 = NULL)
#' result@favorable
#' }
#' @export
tg_lgebv <- function(y, genotype, blocks,
                     h2 = NULL,
                     favorable_direction = c("negative", "positive"),
                     standardize_G = TRUE,
                     method = c("rrblup", "gblup"),
                     device = NULL,
                     return_marker_effects = FALSE) {
  favorable_direction <- match.arg(favorable_direction)
  method <- match.arg(method)

  py <- tryCatch(tg_py(), tg_no_python = function(e) stop(e))
  py_api <- py$api
  py_fn <- py_api$lgebv
  if (is.null(py_fn)) {
    stop(structure(
      class = c("tg_runtime_error", "error", "condition"),
      list(message = "torchgenomics.api has no function 'lgebv'")
    ))
  }

  # Coerce inputs. Reticulate auto-converts numeric vectors to numpy 1-D
  # and matrices to numpy 2-D. Strings stay as paths for the Python side
  # to resolve via torchgenomics.io.
  y_py <- as.numeric(y)
  if (is.character(genotype)) {
    G_py <- .as_path(genotype)
  } else if (is.matrix(genotype)) {
    storage.mode(genotype) <- "double"
    G_py <- genotype
  } else if (is.data.frame(genotype)) {
    G_py <- as.matrix(genotype)
    storage.mode(G_py) <- "double"
  } else {
    G_py <- genotype  # let reticulate handle (e.g. torch tensor)
  }

  # blocks: either a path (string), an LDBlocksRun S4, or a list-of-blocks.
  if (is.character(blocks)) {
    blocks_py <- .as_path(blocks)
  } else if (methods::is(blocks, "LDBlocksRun")) {
    bp <- blocks@output_files[["tsv"]]
    if (is.null(bp)) bp <- blocks@output_files[["blocks"]]
    if (is.null(bp)) {
      stop("LDBlocksRun has no `output_files$tsv` or $blocks path; ",
           "rerun tg_ld_blocks() with an `output =` argument.")
    }
    blocks_py <- .as_path(bp)
  } else {
    blocks_py <- blocks
  }

  device_py <- if (is.null(device) || identical(device, "auto")) NULL else as.character(device)

  py_args <- list(
    y = y_py, G = G_py, haplo_blocks = blocks_py,
    h2 = if (is.null(h2)) NULL else as.numeric(h2),
    favorable_direction = favorable_direction,
    standardize_G = as.logical(standardize_G),
    method = method,
    device = device_py,
    return_marker_effects = as.logical(return_marker_effects)
  )
  py_args <- .compact(py_args)

  d <- tryCatch(
    {
      py_result <- do.call(py_fn, py_args)
      if (inherits(py_result, "python.builtin.object") &&
          reticulate::py_has_attr(py_result, "to_dict")) {
        reticulate::py_to_r(py_result$to_dict())
      } else {
        reticulate::py_to_r(py_result)
      }
    },
    python.builtin.ValueError = function(e) stop(structure(
      class = c("tg_input_error", "error", "condition"),
      list(message = paste0("lgebv(): ", conditionMessage(e)))
    )),
    python.builtin.TypeError = function(e) stop(structure(
      class = c("tg_input_error", "error", "condition"),
      list(message = paste0("lgebv(): ", conditionMessage(e)))
    )),
    python.builtin.Exception = function(e) {
      tb <- tryCatch(reticulate::py_last_error()$traceback, error = function(...) "")
      if (is.null(tb)) tb <- ""
      tb_str <- paste(as.character(tb), collapse = "\n")
      stop(structure(
        class = c("tg_runtime_error", "error", "condition"),
        list(message = paste0(
          "lgebv(): ", conditionMessage(e),
          if (nzchar(tb_str)) paste0("\n", tb_str) else ""
        ))
      ))
    }
  )

  LGEBVResult$new_from_dict(d)
}

#' G x E classification on factor-analytic loadings (Smith et al. 2015 / 2021).
#'
#' Classifies multi-environment-trial environments by the polarity pattern
#' of their factor loadings. Each environment gets a length-k polarity
#' label (e.g. `"PNN"` for 3 factors with positive/negative/negative
#' loadings); environments sharing the same label form one iClass.
#'
#' @param loadings Numeric `E x k` matrix of rotated FA loadings (E
#'   environments x k factors). Typically obtained from a `met-scan` run
#'   with `vg_structure = "fa(k)"` via the Python side's
#'   `MultiEnvLMM.fa_loadings(null_fit)`.
#' @param env_ids Optional character vector of length E giving
#'   environment IDs. If NULL, uses rownames of `loadings` or generic
#'   `"env_0"`-`"env_{E-1}"`.
#' @param threshold Absolute-loading threshold below which an entry is
#'   classified as `"Z"` (zero); default 0 = strict sign.
#'
#' @return An `IClassResult` S4 object with slots `cluster_labels`,
#'   `cluster_membership`, `loadings`, `polarity_matrix`, `n_factors`,
#'   `n_envs`, `env_ids`, `threshold`.
#'
#' @examples
#' \dontrun{
#' lambda <- matrix(c(0.6, -0.2, 0.5, -0.3, 0.7, 0.1), ncol = 2)
#' iclass_result <- tg_iclass(lambda, env_ids = c("Env1", "Env2", "Env3"))
#' iclass_result@cluster_labels
#' }
#' @export
tg_iclass <- function(loadings, env_ids = NULL, threshold = 0.0) {
  if (!is.numeric(threshold) || length(threshold) != 1L || is.na(threshold) ||
      threshold < 0) {
    stop("threshold must be a non-negative numeric scalar.")
  }
  if (is.data.frame(loadings)) {
    loadings <- as.matrix(loadings)
  }
  if (!is.matrix(loadings) || !is.numeric(loadings)) {
    stop("loadings must be a numeric matrix (E x k).")
  }
  # Recover env_ids from rownames if user didn't pass them.
  if (is.null(env_ids) && !is.null(rownames(loadings))) {
    env_ids <- rownames(loadings)
  }
  if (!is.null(env_ids)) {
    env_ids <- as.character(env_ids)
    if (length(env_ids) != nrow(loadings)) {
      stop(sprintf(
        "env_ids has length %d; expected %d to match nrow(loadings).",
        length(env_ids), nrow(loadings)
      ))
    }
  }

  tryCatch(tg_py(), tg_no_python = function(e) stop(e))
  py_postgwas <- tryCatch(
    reticulate::import("torchgenomics.postgwas"),
    error = function(e) stop(structure(
      class = c("tg_no_python", "error", "condition"),
      list(message = paste0(
        "torchgenomics.postgwas not importable: ",
        conditionMessage(e), ". Try tg_install(force = TRUE)."
      ))
    ))
  )
  py_fn <- py_postgwas$iclass
  if (is.null(py_fn)) {
    stop(structure(
      class = c("tg_runtime_error", "error", "condition"),
      list(message = "torchgenomics.postgwas has no function 'iclass'")
    ))
  }

  storage.mode(loadings) <- "double"
  py_args <- .compact(list(
    loadings = loadings,
    env_ids = if (is.null(env_ids)) NULL else as.list(env_ids),
    threshold = as.numeric(threshold)
  ))

  py_result <- tryCatch(
    do.call(py_fn, py_args),
    python.builtin.ValueError = function(e) stop(structure(
      class = c("tg_input_error", "error", "condition"),
      list(message = paste0("iclass(): ", conditionMessage(e)))
    )),
    python.builtin.TypeError = function(e) stop(structure(
      class = c("tg_input_error", "error", "condition"),
      list(message = paste0("iclass(): ", conditionMessage(e)))
    )),
    python.builtin.Exception = function(e) {
      tb <- tryCatch(reticulate::py_last_error()$traceback, error = function(...) "")
      if (is.null(tb)) tb <- ""
      tb_str <- paste(as.character(tb), collapse = "\n")
      stop(structure(
        class = c("tg_runtime_error", "error", "condition"),
        list(message = paste0(
          "iclass(): ", conditionMessage(e),
          if (nzchar(tb_str)) paste0("\n", tb_str) else ""
        ))
      ))
    }
  )

  # IClassResult is a plain @dataclass without to_dict(); reticulate
  # exposes its attributes via $.
  d <- list(
    cluster_labels = reticulate::py_to_r(py_result$cluster_labels),
    cluster_membership = reticulate::py_to_r(py_result$cluster_membership),
    loadings = reticulate::py_to_r(py_result$loadings),
    polarity_matrix = reticulate::py_to_r(py_result$polarity_matrix),
    n_factors = reticulate::py_to_r(py_result$n_factors),
    n_envs = reticulate::py_to_r(py_result$n_envs),
    env_ids = reticulate::py_to_r(py_result$env_ids),
    threshold = reticulate::py_to_r(py_result$threshold)
  )
  IClassResult$new_from_dict(d)
}
