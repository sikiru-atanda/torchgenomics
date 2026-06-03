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
#' @param ... Additional method-specific knobs.
#'
#' @return An `LDBlocksRun` S4 object.
#' @examples
#' \dontrun{
#' r <- tg_ld_blocks("data.bed", method = "gabriel")
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
                         ...) {
  method <- match.arg(method)
  extras <- list(...)
  d <- bridge_call("ld_blocks", .compact(c(list(
    genotype = .as_path(genotype),
    output = .as_path(output),
    method = method,
    max_kb = as.integer(max_kb),
    r2_threshold = r2_threshold,
    ci_low = ci_low, ci_high = ci_high
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
