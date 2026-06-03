# rTorchGenomics/R/api_auto.R
#
# AUTO-GENERATED FROM inst/rbridge_manifest.json (torchgenomics 0.4.0)
# DO NOT EDIT BY HAND. Re-run generate_api_auto() to regenerate.


#' Bayesian variable selection (spike-and-slab) GWAS
#'
#' Auto-generated from torchgenomics `bayes-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param prior_pi Prior inclusion probability (default 0.01).
#' @param prior_sig2_beta Prior slab variance (default 0.1).
#' @param method Inference method: 'susie' (default) or 'cavi'.
#' @param n_signals Number of single-effect layers for SuSiE (default 10).
#' @param learn_hyperparams Learn hyperparameters via empirical Bayes (default True).
#' @param credible_set_coverage Target coverage for credible sets (default 0.95).
#' @param ld_threshold r^2 threshold for LD grouping in credible sets (CAVI only, default 0.5).
#' @param ploidy Ploidy level (default 2 for diploid).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_bayes_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              prior_pi =    0.01,
                              prior_sig2_beta =     0.1,
                              method = "susie",
                              n_signals = 10L,
                              learn_hyperparams = TRUE,
                              credible_set_coverage =    0.95,
                              ld_threshold =     0.5,
                              ploidy = 2L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    prior_pi = if (is.null(prior_pi)) NULL else as.numeric(prior_pi),
    prior_sig2_beta = if (is.null(prior_sig2_beta)) NULL else as.numeric(prior_sig2_beta),
    method = if (is.null(method)) NULL else match.arg(method, c("susie", "cavi")),
    n_signals = if (is.null(n_signals)) NULL else as.integer(n_signals),
    learn_hyperparams = if (is.null(learn_hyperparams)) NULL else as.logical(learn_hyperparams),
    credible_set_coverage = if (is.null(credible_set_coverage)) NULL else as.numeric(credible_set_coverage),
    ld_threshold = if (is.null(ld_threshold)) NULL else as.numeric(ld_threshold),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy)
  ))
  d <- bridge_call("bayes_scan", args)
  GwasResult$new_from_dict(d, command = "bayes_scan")
}

#' SuSiE-RSS fine-mapping on summary statistics + LD reference
#'
#' Auto-generated from torchgenomics `bayes-scan-rss` CLI subcommand.
#'
#' @param sumstats Sumstats TSV.
#' @param ld_ref Pre-built LD reference (.pt or .npz).
#' @param geno Genotype panel for in-sample LD computation (Tier B; not yet wired).
#' @param regions Block regions TSV with start, stop columns (overrides auto block decomposition).
#' @param max_num_causal Maximum number of causal variants per locus (L; default 10).
#' @param coverage Credible-set coverage threshold (default 0.95).
#' @param purity Minimum |R_jk| within a credible set (default 0.5).
#' @param prior_pi Scalar prior inclusion probability OR path to per-SNP prior file (PRIOR_PI column).
#' @param block_size_threshold Maximum p per block before block decomposition kicks in (default 5000).
#' @param output Output TSV path.
#' @param threads Number of CPU threads for torch ops (default 4).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_bayes_scan_rss <- function(sumstats,
                              ld_ref = NULL,
                              geno = NULL,
                              regions = NULL,
                              max_num_causal = 10L,
                              coverage =    0.95,
                              purity =     0.5,
                              prior_pi = NULL,
                              block_size_threshold = 5000L,
                              output,
                              threads = 4L) {
  args <- .compact(list(
    sumstats = .as_path(sumstats),
    ld_ref = .as_path(ld_ref),
    geno = .as_path(geno),
    regions = .as_path(regions),
    max_num_causal = if (is.null(max_num_causal)) NULL else as.integer(max_num_causal),
    coverage = if (is.null(coverage)) NULL else as.numeric(coverage),
    purity = if (is.null(purity)) NULL else as.numeric(purity),
    prior_pi = .as_path(prior_pi),
    block_size_threshold = if (is.null(block_size_threshold)) NULL else as.integer(block_size_threshold),
    output = .as_path(output),
    threads = if (is.null(threads)) NULL else as.integer(threads)
  ))
  d <- bridge_call("bayes_scan_rss", args)
  GwasResult$new_from_dict(d, command = "bayes_scan_rss")
}

#' BLINK multi-locus GWAS via LD-clustering and BIC selection
#'
#' Auto-generated from torchgenomics `blink-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param cutoff Significance cutoff (default: 0.01).
#' @param max_qtns Maximum pseudo-QTNs (0 = unlimited, default: 0).
#' @param ld_threshold LD correlation threshold for clustering (default: 0.7).
#' @param ld_max_samples Max samples for LD computation (default: 200).
#' @param method_sub P-value substitution method (default: reward).
#' @param maf_threshold Internal MAF threshold for QTN candidacy (default: 0.0).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_blink_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              cutoff =    0.01,
                              max_qtns = 0L,
                              ld_threshold =     0.7,
                              ld_max_samples = 200L,
                              method_sub = "reward",
                              maf_threshold =       0) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    cutoff = if (is.null(cutoff)) NULL else as.numeric(cutoff),
    max_qtns = if (is.null(max_qtns)) NULL else as.integer(max_qtns),
    ld_threshold = if (is.null(ld_threshold)) NULL else as.numeric(ld_threshold),
    ld_max_samples = if (is.null(ld_max_samples)) NULL else as.integer(ld_max_samples),
    method_sub = if (is.null(method_sub)) NULL else match.arg(method_sub, c("reward", "mean", "median", "penalty")),
    maf_threshold = if (is.null(maf_threshold)) NULL else as.numeric(maf_threshold)
  ))
  d <- bridge_call("blink_scan", args)
  GwasResult$new_from_dict(d, command = "blink_scan")
}

#' Combine gene-level GWAS evidence with gene-level TWAS evidence using a chosen p-value combination method.
#'
#' Auto-generated from torchgenomics `combine-gwas-twas` CLI subcommand.
#'
#' @param gwas_sumstats GWAS sumstats TSV with columns chr, pos, snp, a1, a2, beta, se, p, n (and optionally af).
#' @param twas_results TWAS results TSV (output of `torchgenomics twas-scan` or a compatible per-gene table with gene_id / chr / start / end / beta / se / z_twas / p_twas / r2_model).
#' @param method Combination method (default: fisher).
#' @param cis_window_bp Symmetric cis window around each gene (bp). Default 100k.
#' @param weights Optional weighting: 'r2_model' weights TWAS by sqrt(CV-R²).
#' @param correction Gene-wise multiple-testing correction (default: bh).
#' @param p_threshold Significance threshold after correction.
#' @param output Output TSV path.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_combine_gwas_twas <- function(gwas_sumstats,
                              twas_results,
                              method = "fisher",
                              cis_window_bp = 100000L,
                              weights = NULL,
                              correction = "bh",
                              p_threshold =    0.05,
                              output) {
  args <- .compact(list(
    gwas_sumstats = .as_path(gwas_sumstats),
    twas_results = .as_path(twas_results),
    method = if (is.null(method)) NULL else match.arg(method, c("fisher", "stouffer", "cauchy", "brown", "empirical_brown", "hmp", "truncated_product", "min_p")),
    cis_window_bp = if (is.null(cis_window_bp)) NULL else as.integer(cis_window_bp),
    weights = if (is.null(weights)) NULL else match.arg(weights, c("r2_model")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("none", "bonferroni", "bh", "by", "storey")),
    p_threshold = if (is.null(p_threshold)) NULL else as.numeric(p_threshold),
    output = .as_path(output)
  ))
  d <- bridge_call("combine_gwas_twas", args)
  GwasResult$new_from_dict(d, command = "combine_gwas_twas")
}

#' LD-conditional GWAS with block-aware conditioning and persistence metrics
#'
#' Auto-generated from torchgenomics `conditional-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param ld_method LD block detection method (default: r2).
#' @param max_kb Max distance in kb for LD blocks (default: 200).
#' @param persistence_ratio Signal retention ratio for persistence metric (default: 0.5).
#' @param sig_threshold Significance threshold for persistence evaluation (default: 5e-8).
#' @param max_conditioning Max lead SNPs to condition on per block (default: 5).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_conditional_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              ld_method = "r2",
                              max_kb =     200,
                              persistence_ratio =     0.5,
                              sig_threshold =   5e-08,
                              max_conditioning = 5L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    ld_method = if (is.null(ld_method)) NULL else match.arg(ld_method, c("gabriel", "four_gamete", "spine", "r2", "big_ld", "cc_graph", "dp_optimize")),
    max_kb = if (is.null(max_kb)) NULL else as.numeric(max_kb),
    persistence_ratio = if (is.null(persistence_ratio)) NULL else as.numeric(persistence_ratio),
    sig_threshold = if (is.null(sig_threshold)) NULL else as.numeric(sig_threshold),
    max_conditioning = if (is.null(max_conditioning)) NULL else as.integer(max_conditioning)
  ))
  d <- bridge_call("conditional_scan", args)
  GwasResult$new_from_dict(d, command = "conditional_scan")
}

#' Call polyploid allele dosages from VCF read counts via updog
#'
#' Auto-generated from torchgenomics `dosage-call` CLI subcommand.
#'
#' @param vcf Input VCF with AD format field (biallelic).
#' @param output Output prefix for .probs.pt / .meta.json / .snp_diag.tsv.
#' @param ploidy Organism ploidy (2..8).
#' @param model updog flexdog model name (default: norm).
#' @param rscript Path to Rscript (default: PATH lookup).
#' @param bias Estimate allele bias (default).
#' @param od Estimate overdispersion (default).
#' @param seq_error Fix sequencing error rate (default: estimate).
#' @param n_cores Parallelism passed to updog::multidog (default: 1).
#' @param keep_tmpdir Skip tempdir cleanup (debug aid).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_dosage_call <- function(vcf,
                              output,
                              ploidy,
                              model = "norm",
                              rscript = NULL,
                              bias = TRUE,
                              od = TRUE,
                              seq_error = NULL,
                              n_cores = 1L,
                              keep_tmpdir = FALSE) {
  args <- .compact(list(
    vcf = .as_path(vcf),
    output = .as_path(output),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    model = .as_path(model),
    rscript = .as_path(rscript),
    bias = if (is.null(bias)) NULL else as.logical(bias),
    od = if (is.null(od)) NULL else as.logical(od),
    seq_error = if (is.null(seq_error)) NULL else as.numeric(seq_error),
    n_cores = if (is.null(n_cores)) NULL else as.integer(n_cores),
    keep_tmpdir = if (is.null(keep_tmpdir)) NULL else as.logical(keep_tmpdir)
  ))
  d <- bridge_call("dosage_call", args)
  GwasResult$new_from_dict(d, command = "dosage_call")
}

#' Within-family GWAS with confounding diagnostics (Young et al. 2022)
#'
#' Auto-generated from torchgenomics `family-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param family_col Column name in phenotype file containing family IDs (default: FID).
#' @param min_family_size Minimum members per family for within-family scan (default: 2).
#' @param confound_threshold Attenuation deviation threshold for confounding flag (default: 0.5).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_family_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              family_col = "FID",
                              min_family_size = 2L,
                              confound_threshold =     0.5) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    family_col = .as_path(family_col),
    min_family_size = if (is.null(min_family_size)) NULL else as.integer(min_family_size),
    confound_threshold = if (is.null(confound_threshold)) NULL else as.numeric(confound_threshold)
  ))
  d <- bridge_call("family_scan", args)
  GwasResult$new_from_dict(d, command = "family_scan")
}

#' FarmCPU multi-locus GWAS via iterative FEM/REM
#'
#' Auto-generated from torchgenomics `farmcpu-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param p_threshold P-value threshold for QTN selection (default: 0.01).
#' @param max_qtns Maximum pseudo-QTNs per iteration (default: 20).
#' @param bin_sizes Comma-separated bin sizes in bp (default: 500000,5000000,50000000).
#' @param method_bin Binning method (default: static).
#' @param method_sub P-value substitution method (default: reward).
#' @param maf_threshold Internal MAF threshold for QTN candidacy (default: 0.0).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_farmcpu_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              p_threshold =    0.01,
                              max_qtns = 20L,
                              bin_sizes = "500000,5000000,50000000",
                              method_bin = "static",
                              method_sub = "reward",
                              maf_threshold =       0) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    p_threshold = if (is.null(p_threshold)) NULL else as.numeric(p_threshold),
    max_qtns = if (is.null(max_qtns)) NULL else as.integer(max_qtns),
    bin_sizes = .as_path(bin_sizes),
    method_bin = if (is.null(method_bin)) NULL else match.arg(method_bin, c("static", "optimum")),
    method_sub = if (is.null(method_sub)) NULL else match.arg(method_sub, c("reward", "mean", "median", "penalty")),
    maf_threshold = if (is.null(maf_threshold)) NULL else as.numeric(maf_threshold)
  ))
  d <- bridge_call("farmcpu_scan", args)
  GwasResult$new_from_dict(d, command = "farmcpu_scan")
}

#' GLMM association scan (binary/ordinal with random effects, SAIGE-style PQL)
#'
#' Auto-generated from torchgenomics `glmm-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param family Phenotype family (default: binary).
#' @param n_categories Number of ordinal categories (default: 3, ignored for binary).
#' @param firth Use Firth correction for GLM initialisation (binary only).
#' @param no_spa Disable SPA tail correction (binary only).
#' @param pql_max_iter Max PQL outer iterations (default: 30).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_glmm_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              family = "binary",
                              n_categories = 3L,
                              firth = FALSE,
                              no_spa = FALSE,
                              pql_max_iter = 30L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    family = if (is.null(family)) NULL else match.arg(family, c("binary", "ordinal", "multinomial")),
    n_categories = if (is.null(n_categories)) NULL else as.integer(n_categories),
    firth = if (is.null(firth)) NULL else as.logical(firth),
    no_spa = if (is.null(no_spa)) NULL else as.logical(no_spa),
    pql_max_iter = if (is.null(pql_max_iter)) NULL else as.integer(pql_max_iter)
  ))
  d <- bridge_call("glmm_scan", args)
  GwasResult$new_from_dict(d, command = "glmm_scan")
}

#' Genotype-Uncertainty LMM scan (dosage-variance corrected)
#'
#' Auto-generated from torchgenomics `gu-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str parameter.
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param dosage_var Path to dosage variance file (.pt or .npy, shape n×m).
#' @param probs Path to a Phase 55 <prefix>.probs.pt from `torchgenomics dosage-call`. When given, --genotype and --dosage-var are derived internally via expected_dosage / dosage_variance (ploidy read from the sibling <prefix>.meta.json).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_gu_scan <- function(genotype = NULL,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              dosage_var = NULL,
                              probs = NULL) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    dosage_var = .as_path(dosage_var),
    probs = .as_path(probs)
  ))
  d <- bridge_call("gu_scan", args)
  GwasResult$new_from_dict(d, command = "gu_scan")
}

#' Gene-environment interaction LMM scan
#'
#' Auto-generated from torchgenomics `gxe-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param env Environment variable file (TSV: SAMPLE, ENV).
#' @param gxe_model GxE model: 'het' (single-trait HetLMM, default) or 'multi' (multi-trait GxELMM).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_gxe_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              env,
                              gxe_model = "het") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    env = .as_path(env),
    gxe_model = if (is.null(gxe_model)) NULL else match.arg(gxe_model, c("het", "multi"))
  ))
  d <- bridge_call("gxe_scan", args)
  GwasResult$new_from_dict(d, command = "gxe_scan")
}

#' Knockoff FDR-controlled GWAS (Sesia et al. 2020)
#'
#' Auto-generated from torchgenomics `knockoff-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param fdr_level Target FDR level (default: 0.05).
#' @param ld_method LD block detection method (default: gabriel).
#' @param knockoff_method Knockoff generation method (default: equicorrelated).
#' @param aggregation Block importance aggregation (default: max_stat).
#' @param seed Random seed for knockoff generation.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_knockoff_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              fdr_level =    0.05,
                              ld_method = "gabriel",
                              knockoff_method = "equicorrelated",
                              aggregation = "max_stat",
                              seed = NULL) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    fdr_level = if (is.null(fdr_level)) NULL else as.numeric(fdr_level),
    ld_method = if (is.null(ld_method)) NULL else match.arg(ld_method, c("gabriel", "four_gamete", "spine", "r2", "gwas_aligned", "uncertainty", "cross_pop", "graphical", "changepoint", "big_ld", "cc_graph", "dp_optimize")),
    knockoff_method = if (is.null(knockoff_method)) NULL else match.arg(knockoff_method, c("equicorrelated")),
    aggregation = if (is.null(aggregation)) NULL else match.arg(aggregation, c("max_stat", "sum_sq")),
    seed = if (is.null(seed)) NULL else as.integer(seed)
  ))
  d <- bridge_call("knockoff_scan", args)
  GwasResult$new_from_dict(d, command = "knockoff_scan")
}

#' LDSC SNP heritability estimation
#'
#' Auto-generated from torchgenomics `ldsc` CLI subcommand.
#'
#' @param sumstats Summary statistics TSV.
#' @param genotype Genotype file for LD score computation.
#' @param window_kb LD window in kb (default 1000).
#' @param n Sample size (if not in sumstats).
#' @param output argparse-derived str parameter.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_ldsc <- function(sumstats,
                              genotype,
                              window_kb =    1000,
                              n = NULL,
                              output = "torchgenomics_results") {
  args <- .compact(list(
    sumstats = .as_path(sumstats),
    genotype = .as_path(genotype),
    window_kb = if (is.null(window_kb)) NULL else as.numeric(window_kb),
    n = if (is.null(n)) NULL else as.integer(n),
    output = .as_path(output)
  ))
  d <- bridge_call("ldsc", args)
  GwasResult$new_from_dict(d, command = "ldsc")
}

#' LDSC genetic correlation
#'
#' Auto-generated from torchgenomics `ldsc-rg` CLI subcommand.
#'
#' @param sumstats1 Summary stats for trait 1.
#' @param sumstats2 Summary stats for trait 2.
#' @param genotype Genotype file for LD scores.
#' @param window_kb LD window in kb (default 1000).
#' @param n1 Sample size trait 1.
#' @param n2 Sample size trait 2.
#' @param output argparse-derived str parameter.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_ldsc_rg <- function(sumstats1,
                              sumstats2,
                              genotype,
                              window_kb =    1000,
                              n1 = NULL,
                              n2 = NULL,
                              output = "torchgenomics_results") {
  args <- .compact(list(
    sumstats1 = .as_path(sumstats1),
    sumstats2 = .as_path(sumstats2),
    genotype = .as_path(genotype),
    window_kb = if (is.null(window_kb)) NULL else as.numeric(window_kb),
    n1 = if (is.null(n1)) NULL else as.integer(n1),
    n2 = if (is.null(n2)) NULL else as.integer(n2),
    output = .as_path(output)
  ))
  d <- bridge_call("ldsc_rg", args)
  GwasResult$new_from_dict(d, command = "ldsc_rg")
}

#' Leave-Region-Out LMM scan (block-level LOCO)
#'
#' Auto-generated from torchgenomics `lro-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param ld_method LD block detection method (default: r2).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_lro_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              ld_method = "r2") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    ld_method = if (is.null(ld_method)) NULL else match.arg(ld_method, c("gabriel", "four_gamete", "spine", "r2", "gwas_aligned", "graphical", "changepoint", "big_ld", "cc_graph", "dp_optimize"))
  ))
  d <- bridge_call("lro_scan", args)
  GwasResult$new_from_dict(d, command = "lro_scan")
}

#' Multi-environment GLMM scan (binary/ordinal across environments)
#'
#' Auto-generated from torchgenomics `me-glmm-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param family Phenotype family (default: binary).
#' @param n_categories Number of ordinal categories (required for ordinal).
#' @param env_cols Comma-separated environment column names.
#' @param parameterization Test parameterization (default: per_env).
#' @param firth Use Firth correction for per-env GLM init.
#' @param no_spa Disable SPA for per-env binary marginals.
#' @param pql_max_iter Max PQL outer iterations (default: 30).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_me_glmm_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              family = "binary",
                              n_categories = NULL,
                              env_cols = NULL,
                              parameterization = "per_env",
                              firth = FALSE,
                              no_spa = FALSE,
                              pql_max_iter = 30L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    family = if (is.null(family)) NULL else match.arg(family, c("binary", "ordinal")),
    n_categories = if (is.null(n_categories)) NULL else as.integer(n_categories),
    env_cols = .as_path(env_cols),
    parameterization = if (is.null(parameterization)) NULL else match.arg(parameterization, c("per_env", "reaction_norm")),
    firth = if (is.null(firth)) NULL else as.logical(firth),
    no_spa = if (is.null(no_spa)) NULL else as.logical(no_spa),
    pql_max_iter = if (is.null(pql_max_iter)) NULL else as.integer(pql_max_iter)
  ))
  d <- bridge_call("me_glmm_scan", args)
  GwasResult$new_from_dict(d, command = "me_glmm_scan")
}

#' GRM-corrected single-triple causal mediation (SNP -> M -> Y)
#'
#' Auto-generated from torchgenomics `mediate` CLI subcommand.
#'
#' @param y Phenotype vector (.npy/.pt/.tsv).
#' @param snp SNP dosage vector (.npy/.pt/.tsv).
#' @param mediator Mediator vector (.npy/.pt/.tsv).
#' @param kinship GRM (n,n) (.npy/.pt).
#' @param covariates Covariates (n,c) (.npy/.pt/.tsv).
#' @param se argparse-derived str parameter.
#' @param n_mc argparse-derived int parameter.
#' @param n_boot argparse-derived int parameter.
#' @param seed argparse-derived int parameter.
#' @param no_sensitivity Skip Imai rho.
#' @param output Output JSON path.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_mediate <- function(y,
                              snp,
                              mediator,
                              kinship,
                              covariates = NULL,
                              se = "monte-carlo",
                              n_mc = 10000L,
                              n_boot = 1000L,
                              seed = NULL,
                              no_sensitivity = FALSE,
                              output = "mediation.json") {
  args <- .compact(list(
    y = .as_path(y),
    snp = .as_path(snp),
    mediator = .as_path(mediator),
    kinship = .as_path(kinship),
    covariates = .as_path(covariates),
    se = if (is.null(se)) NULL else match.arg(se, c("sobel", "monte-carlo", "bootstrap")),
    n_mc = if (is.null(n_mc)) NULL else as.integer(n_mc),
    n_boot = if (is.null(n_boot)) NULL else as.integer(n_boot),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    no_sensitivity = if (is.null(no_sensitivity)) NULL else as.logical(no_sensitivity),
    output = .as_path(output)
  ))
  d <- bridge_call("mediate", args)
  GwasResult$new_from_dict(d, command = "mediate")
}

#' Genome x molecular-feature mediation scan with cis-window filtering
#'
#' Auto-generated from torchgenomics `mediate-scan` CLI subcommand.
#'
#' @param y Phenotype vector (.npy/.pt/.tsv).
#' @param genotype Genotype matrix (n,s) (.npy/.pt).
#' @param mediator_matrix Mediator matrix (n,f) (.npy/.pt).
#' @param kinship GRM (n,n) (.npy/.pt).
#' @param snp_meta TSV with columns: id,chrom,pos (one row per SNP).
#' @param feature_meta TSV with columns: id,chrom,pos (one row per feature).
#' @param covariates Covariates (.npy/.pt/.tsv).
#' @param cis_window Cis window in bp; -1 means no filter (all pairs).
#' @param se argparse-derived str parameter.
#' @param n_mc argparse-derived int parameter.
#' @param fdr argparse-derived str parameter.
#' @param seed argparse-derived int parameter.
#' @param output Output prefix (.tsv + .top.tsv).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_mediate_scan <- function(y,
                              genotype,
                              mediator_matrix,
                              kinship,
                              snp_meta = NULL,
                              feature_meta = NULL,
                              covariates = NULL,
                              cis_window = 1000000L,
                              se = "monte-carlo",
                              n_mc = 10000L,
                              fdr = "bh",
                              seed = NULL,
                              output = "mediate_scan") {
  args <- .compact(list(
    y = .as_path(y),
    genotype = .as_path(genotype),
    mediator_matrix = .as_path(mediator_matrix),
    kinship = .as_path(kinship),
    snp_meta = .as_path(snp_meta),
    feature_meta = .as_path(feature_meta),
    covariates = .as_path(covariates),
    cis_window = if (is.null(cis_window)) NULL else as.integer(cis_window),
    se = if (is.null(se)) NULL else match.arg(se, c("sobel", "monte-carlo", "bootstrap")),
    n_mc = if (is.null(n_mc)) NULL else as.integer(n_mc),
    fdr = if (is.null(fdr)) NULL else match.arg(fdr, c("bh", "by", "storey")),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    output = .as_path(output)
  ))
  d <- bridge_call("mediate_scan", args)
  GwasResult$new_from_dict(d, command = "mediate_scan")
}

#' Multi-environment trial (MET) GWAS: stable vs environment-contingent effects
#'
#' Auto-generated from torchgenomics `met-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param env_cols Comma-separated environment column names in phenotype file (default: all non-ID columns).
#' @param ploidy Ploidy level (default 2 for diploid).
#' @param parameterization SNP effect decomposition (default: per_env).
#' @param vg_structure Genetic covariance structure: 'unstructured' or 'fa(k)' for factor analytic with k factors (default: unstructured).
#' @param kernel_files Additional kernel matrix files (.npy or .pt). Used alongside the computed additive GRM for multi-kernel MET.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_met_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              env_cols = NULL,
                              ploidy = 2L,
                              parameterization = "per_env",
                              vg_structure = "unstructured",
                              kernel_files = NULL) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    env_cols = .as_path(env_cols),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    parameterization = if (is.null(parameterization)) NULL else match.arg(parameterization, c("per_env", "reaction_norm")),
    vg_structure = .as_path(vg_structure),
    kernel_files = if (is.null(kernel_files)) NULL else as.character(kernel_files)
  ))
  d <- bridge_call("met_scan", args)
  GwasResult$new_from_dict(d, command = "met_scan")
}

#' Multi-kernel LMM scan (additive + dominance + epistatic)
#'
#' Auto-generated from torchgenomics `mklmm-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param kernels Comma-separated kernel types: additive,dominance,epistatic (default: all three).
#' @param ploidy Ploidy level (default 2).
#' @param grm_method GRM method for additive kernel.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_mklmm_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              kernels = "additive,dominance,epistatic",
                              ploidy = 2L,
                              grm_method = "vanraden") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    kernels = .as_path(kernels),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    grm_method = if (is.null(grm_method)) NULL else match.arg(grm_method, c("vanraden", "zhang"))
  ))
  d <- bridge_call("mklmm_scan", args)
  GwasResult$new_from_dict(d, command = "mklmm_scan")
}

#' Multi-trait multi-environment GWAS with separable Kronecker covariance
#'
#' Auto-generated from torchgenomics `mtmet-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param env_cols Comma-separated environment names (e.g., E1,E2,E3).
#' @param vg_structure Vg structure: 'separable' (default), 'unstructured', or 'fa(k)'.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_mtmet_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              env_cols,
                              vg_structure = "separable") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    env_cols = .as_path(env_cols),
    vg_structure = .as_path(vg_structure)
  ))
  d <- bridge_call("mtmet_scan", args)
  GwasResult$new_from_dict(d, command = "mtmet_scan")
}

#' Multi-trait mvLMM association scan
#'
#' Auto-generated from torchgenomics `mvlmm-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param grm Path to pre-computed GRM matrix (NumPy .npy, .npz, or space/tab-delimited text). If provided, --grm-method is ignored.
#' @param grm_method GRM method: 'vanraden' (streaming, scalable) or 'zhang' (matches GAPIT, requires full materialization).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_mvlmm_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              grm = NULL,
                              grm_method = "vanraden") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    grm = .as_path(grm),
    grm_method = if (is.null(grm_method)) NULL else match.arg(grm_method, c("vanraden", "zhang"))
  ))
  d <- bridge_call("mvlmm_scan", args)
  GwasResult$new_from_dict(d, command = "mvlmm_scan")
}

#' Orthogonal Cross-Fit LMM scan (DML debiased inference)
#'
#' Auto-generated from torchgenomics `ocf-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param n_folds Number of cross-fitting folds (default: 5).
#' @param seed Random seed for fold assignment.
#' @param variance_type Variance estimator: HC (sandwich) or homoskedastic (default: HC).
#' @param no_genotype_projection Skip projecting genotypes onto covariate space.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_ocf_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              n_folds = 5L,
                              seed = NULL,
                              variance_type = "HC",
                              no_genotype_projection = FALSE) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    n_folds = if (is.null(n_folds)) NULL else as.integer(n_folds),
    seed = if (is.null(seed)) NULL else as.integer(seed),
    variance_type = if (is.null(variance_type)) NULL else match.arg(variance_type, c("HC", "homoskedastic")),
    no_genotype_projection = if (is.null(no_genotype_projection)) NULL else as.logical(no_genotype_projection)
  ))
  d <- bridge_call("ocf_scan", args)
  GwasResult$new_from_dict(d, command = "ocf_scan")
}

#' Polyploid F1 phasing via PolyOrigin (Phase 56)
#'
#' Auto-generated from torchgenomics `phase-poly` CLI subcommand.
#'
#' @param probs Path to .probs.pt from 'dosage-call' (raw (n,m,k+1) tensor or dict). If a raw tensor, the sibling <prefix>.meta.json is read for sample/variant IDs.
#' @param pedigree TSV with columns: offspring, parent1, parent2 (optional: ploidy).
#' @param map TSV with columns: marker, chrom, pos_bp (optional: cm).
#' @param output Output prefix; writes <prefix>.haplotypes.pt etc.
#' @param ploidy Organism ploidy (2, 4, or 6).
#' @param parent_phased Optional CSV of pre-phased parent genotypes (escape hatch).
#' @param no_refinemap Disable PolyOrigin's map refinement (default: enabled).
#' @param recomrate cM/Mb to synthesize genetic positions when --map lacks a cm column (default: 1.0).
#' @param julia_path Path to an existing Julia binary; else discovered or auto-installed.
#' @param auto_install Auto-install Julia if not found (non-interactive; sets consent=True).
#' @param keep_workdir Do not delete the temp work directory after success (debug aid).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_phase_poly <- function(probs,
                              pedigree,
                              map,
                              output,
                              ploidy,
                              parent_phased = NULL,
                              no_refinemap = TRUE,
                              recomrate =       1,
                              julia_path = NULL,
                              auto_install = NULL,
                              keep_workdir = FALSE) {
  args <- .compact(list(
    probs = .as_path(probs),
    pedigree = .as_path(pedigree),
    map = .as_path(map),
    output = .as_path(output),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    parent_phased = .as_path(parent_phased),
    no_refinemap = if (is.null(no_refinemap)) NULL else as.logical(no_refinemap),
    recomrate = if (is.null(recomrate)) NULL else as.numeric(recomrate),
    julia_path = .as_path(julia_path),
    auto_install = if (is.null(auto_install)) NULL else as.logical(auto_install),
    keep_workdir = if (is.null(keep_workdir)) NULL else as.logical(keep_workdir)
  ))
  d <- bridge_call("phase_poly", args)
  GwasResult$new_from_dict(d, command = "phase_poly")
}

#' Full pipeline: impute -> scan -> correct
#'
#' Auto-generated from torchgenomics `pipeline` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param impute argparse-derived str parameter.
#' @param model argparse-derived str parameter.
#' @param ploidy Ploidy level (default 2 for diploid).
#' @param env_cols Comma-separated environment columns (for --model met).
#' @param parameterization SNP effect decomposition for MET (default: per_env).
#' @param vg_structure Genetic covariance structure for MET: 'unstructured' or 'fa(k)'.
#' @param env Environment variable file (TSV: SAMPLE/IID, ENV) for --model gxe.
#' @param gxe_model GxE model for --model gxe: 'het' (default) or 'multi'.
#' @param approx_method Approximate eigendecomposition method (default: exact).
#' @param approx_components Number of components for randomized SVD / LOBPCG (default: 100).
#' @param approx_landmarks Number of landmark samples for Nystrom (default: 500).
#' @param sparse_threshold Relatedness threshold for sparse GRM (default: 0.05).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_pipeline <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              impute = NULL,
                              model = "lmm",
                              ploidy = 2L,
                              env_cols = NULL,
                              parameterization = "per_env",
                              vg_structure = "unstructured",
                              env = NULL,
                              gxe_model = "het",
                              approx_method = NULL,
                              approx_components = 100L,
                              approx_landmarks = 500L,
                              sparse_threshold =    0.05) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    impute = if (is.null(impute)) NULL else match.arg(impute, c("mean")),
    model = if (is.null(model)) NULL else match.arg(model, c("glm", "lmm", "mvlmm", "farmcpu", "blink", "mklmm", "gxe", "met")),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    env_cols = .as_path(env_cols),
    parameterization = if (is.null(parameterization)) NULL else match.arg(parameterization, c("per_env", "reaction_norm")),
    vg_structure = .as_path(vg_structure),
    env = .as_path(env),
    gxe_model = if (is.null(gxe_model)) NULL else match.arg(gxe_model, c("het", "multi")),
    approx_method = if (is.null(approx_method)) NULL else match.arg(approx_method, c("randomized_svd", "nystrom", "lobpcg", "sparse")),
    approx_components = if (is.null(approx_components)) NULL else as.integer(approx_components),
    approx_landmarks = if (is.null(approx_landmarks)) NULL else as.integer(approx_landmarks),
    sparse_threshold = if (is.null(sparse_threshold)) NULL else as.numeric(sparse_threshold)
  ))
  d <- bridge_call("pipeline", args)
  GwasResult$new_from_dict(d, command = "pipeline")
}

#' Polyploid GWAS scan
#'
#' Auto-generated from torchgenomics `poly-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param ploidy argparse-derived int (required).
#' @param gene_action argparse-derived str parameter.
#' @param p3d P3D=TRUE: estimate variance components once (default).
#' @param max_geno_freq Max genotype class frequency filter (e.g. 0.95). Applied after gene-action encoding.
#' @param bp_window Base-pair window for LD peak pruning (default: 1000000).
#' @param peak_threshold P-value threshold for peak detection (default: 1e-4).
#' @param joint_qtl Fit multi-QTL joint model on detected peaks.
#' @param meff_method Effective test count method (default: gao).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_poly_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              ploidy,
                              gene_action = "additive",
                              p3d = TRUE,
                              max_geno_freq = NULL,
                              bp_window = 1000000L,
                              peak_threshold =  0.0001,
                              joint_qtl = FALSE,
                              meff_method = "gao") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy),
    gene_action = if (is.null(gene_action)) NULL else match.arg(gene_action, c("additive", "general", "all", "1-dom", "diplo-additive", "overdominant")),
    p3d = if (is.null(p3d)) NULL else as.logical(p3d),
    max_geno_freq = if (is.null(max_geno_freq)) NULL else as.numeric(max_geno_freq),
    bp_window = if (is.null(bp_window)) NULL else as.integer(bp_window),
    peak_threshold = if (is.null(peak_threshold)) NULL else as.numeric(peak_threshold),
    joint_qtl = if (is.null(joint_qtl)) NULL else as.logical(joint_qtl),
    meff_method = if (is.null(meff_method)) NULL else match.arg(meff_method, c("gao", "moskvina"))
  ))
  d <- bridge_call("poly_scan", args)
  GwasResult$new_from_dict(d, command = "poly_scan")
}

#' Random Regression LMM × Multi-Environment GWAS for longitudinal MET data
#'
#' Auto-generated from torchgenomics `rr-met-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param id_col Phenotype column with sample identifier (default: IID).
#' @param env_col Phenotype column with environment label (default: ENV).
#' @param time_col Phenotype column with raw time/age values (default: TIME).
#' @param pheno_col Phenotype column with the long-format response (default: Y).
#' @param basis Temporal basis (default: legendre). Pooled across all envs.
#' @param order Legendre polynomial order; b = order + 1 (default: 2).
#' @param n_knots B-spline interior knot count (default: 4).
#' @param degree B-spline degree (default: 3).
#' @param vg_structure Genetic covariance structure for the (b·E) pseudo-traits (default: separable Vg = K_coef ⊗ Vg_env).
#' @param eval_times Comma-separated raw time values for per-(time, env) χ²(1) reconstruction. Produces an extra .rr_met_at_t.tsv output.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_rr_met_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              id_col = "IID",
                              env_col = "ENV",
                              time_col = "TIME",
                              pheno_col = "Y",
                              basis = "legendre",
                              order = 2L,
                              n_knots = 4L,
                              degree = 3L,
                              vg_structure = "separable",
                              eval_times = NULL) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    id_col = .as_path(id_col),
    env_col = .as_path(env_col),
    time_col = .as_path(time_col),
    pheno_col = .as_path(pheno_col),
    basis = if (is.null(basis)) NULL else match.arg(basis, c("legendre", "bspline")),
    order = if (is.null(order)) NULL else as.integer(order),
    n_knots = if (is.null(n_knots)) NULL else as.integer(n_knots),
    degree = if (is.null(degree)) NULL else as.integer(degree),
    vg_structure = if (is.null(vg_structure)) NULL else match.arg(vg_structure, c("separable", "unstructured", "fa(1)", "fa(2)", "fa(3)")),
    eval_times = .as_path(eval_times)
  ))
  d <- bridge_call("rr_met_scan", args)
  GwasResult$new_from_dict(d, command = "rr_met_scan")
}

#' Random Regression LMM GWAS for longitudinal / spatio-temporal data
#'
#' Auto-generated from torchgenomics `rr-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param id_col Phenotype column with sample identifier (default: IID).
#' @param time_col Phenotype column with raw time/age values (default: TIME).
#' @param pheno_col Phenotype column with the long-format response (default: Y).
#' @param basis Temporal basis (default: legendre).
#' @param order Legendre polynomial order; b = order + 1 (default: 2).
#' @param n_knots B-spline interior knot count (default: 4).
#' @param degree B-spline degree (default: 3).
#' @param k_coef_structure K_coef structure: 'unstructured', 'diagonal', or 'fa(k)'.
#' @param include_pe Decompose Ve into a permanent-environment block.
#' @param mode Reduction mode (default: projection).
#' @param eval_times Comma-separated raw time values for per-time-point χ²(1) reconstruction. Produces an extra .rr_at_t.tsv output.
#' @param row_col Phenotype column with field-trial row coordinate (enables 2D P-spline spatial smoother).
#' @param col_col Phenotype column with field-trial column coordinate.
#' @param spatial_knots_row argparse-derived int parameter.
#' @param spatial_knots_col argparse-derived int parameter.
#' @param lambda_row argparse-derived float parameter.
#' @param lambda_col argparse-derived float parameter.
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_rr_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              id_col = "IID",
                              time_col = "TIME",
                              pheno_col = "Y",
                              basis = "legendre",
                              order = 2L,
                              n_knots = 4L,
                              degree = 3L,
                              k_coef_structure = "unstructured",
                              include_pe = FALSE,
                              mode = "projection",
                              eval_times = NULL,
                              row_col = NULL,
                              col_col = NULL,
                              spatial_knots_row = 8L,
                              spatial_knots_col = 8L,
                              lambda_row =       1,
                              lambda_col =       1) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    id_col = .as_path(id_col),
    time_col = .as_path(time_col),
    pheno_col = .as_path(pheno_col),
    basis = if (is.null(basis)) NULL else match.arg(basis, c("legendre", "bspline")),
    order = if (is.null(order)) NULL else as.integer(order),
    n_knots = if (is.null(n_knots)) NULL else as.integer(n_knots),
    degree = if (is.null(degree)) NULL else as.integer(degree),
    k_coef_structure = .as_path(k_coef_structure),
    include_pe = if (is.null(include_pe)) NULL else as.logical(include_pe),
    mode = if (is.null(mode)) NULL else match.arg(mode, c("projection", "stacked")),
    eval_times = .as_path(eval_times),
    row_col = .as_path(row_col),
    col_col = .as_path(col_col),
    spatial_knots_row = if (is.null(spatial_knots_row)) NULL else as.integer(spatial_knots_row),
    spatial_knots_col = if (is.null(spatial_knots_col)) NULL else as.integer(spatial_knots_col),
    lambda_row = if (is.null(lambda_row)) NULL else as.numeric(lambda_row),
    lambda_col = if (is.null(lambda_col)) NULL else as.numeric(lambda_col)
  ))
  d <- bridge_call("rr_scan", args)
  GwasResult$new_from_dict(d, command = "rr_scan")
}

#' Set-based association tests (SKAT/Burden/SKAT-O)
#'
#' Auto-generated from torchgenomics `set-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param regions BED file defining gene/region boundaries (chr, start, end, (name)).
#' @param set_test Set-based test: skat (default), burden, or skat_o.
#' @param weight_a1 Beta(MAF; a1, a2) weight parameter a1 (default 1.0).
#' @param weight_a2 Beta(MAF; a1, a2) weight parameter a2 (default 25.0).
#' @param ploidy Ploidy level (default 2 for diploid).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_set_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              regions,
                              set_test = "skat",
                              weight_a1 =       1,
                              weight_a2 =      25,
                              ploidy = 2L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    regions = .as_path(regions),
    set_test = if (is.null(set_test)) NULL else match.arg(set_test, c("skat", "burden", "skat_o")),
    weight_a1 = if (is.null(weight_a1)) NULL else as.numeric(weight_a1),
    weight_a2 = if (is.null(weight_a2)) NULL else as.numeric(weight_a2),
    ploidy = if (is.null(ploidy)) NULL else as.integer(ploidy)
  ))
  d <- bridge_call("set_scan", args)
  GwasResult$new_from_dict(d, command = "set_scan")
}

#' Survival GWAS scan (Cox PH frailty model with martingale residual score test)
#'
#' Auto-generated from torchgenomics `survival-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param time_col Column name for follow-up time (default: first phenotype column).
#' @param event_col Column name for event indicator 0/1 (default: second phenotype column).
#' @param no_spa Disable SPA for tail p-values.
#' @param spa_threshold Chi-square threshold for SPA activation (default: 2.0).
#' @param pql_max_iter Max PQL outer iterations (default: 30).
#' @param pql_tol PQL convergence tolerance (default: 1e-4).
#' @param ties Tie-handling method (default: breslow).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_survival_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              time_col = NULL,
                              event_col = NULL,
                              no_spa = FALSE,
                              spa_threshold =       2,
                              pql_max_iter = 30L,
                              pql_tol =  0.0001,
                              ties = "breslow") {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    time_col = .as_path(time_col),
    event_col = .as_path(event_col),
    no_spa = if (is.null(no_spa)) NULL else as.logical(no_spa),
    spa_threshold = if (is.null(spa_threshold)) NULL else as.numeric(spa_threshold),
    pql_max_iter = if (is.null(pql_max_iter)) NULL else as.integer(pql_max_iter),
    pql_tol = if (is.null(pql_tol)) NULL else as.numeric(pql_tol),
    ties = if (is.null(ties)) NULL else match.arg(ties, c("breslow"))
  ))
  d <- bridge_call("survival_scan", args)
  GwasResult$new_from_dict(d, command = "survival_scan")
}

#' Threshold-linear GWAS for ordinal + continuous traits (Bermann et al. 2026)
#'
#' Auto-generated from torchgenomics `threshold-scan` CLI subcommand.
#'
#' @param genotype argparse-derived str (required).
#' @param phenotype argparse-derived str (required).
#' @param covariate argparse-derived str parameter.
#' @param test argparse-derived str parameter.
#' @param correction argparse-derived str parameter.
#' @param maf_min argparse-derived float parameter.
#' @param miss_max argparse-derived float parameter.
#' @param output argparse-derived str parameter.
#' @param device argparse-derived str parameter.
#' @param chunk_size argparse-derived int parameter.
#' @param max_iter Maximum REML optimizer iterations (default: from config).
#' @param weights_file TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT).
#' @param gene_map TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE).
#' @param fdr_covariate TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). If omitted, MAF from the scan result is used.
#' @param n_pcs Number of principal components from GRM eigendecomposition to include as covariates for population structure correction (default: 0, meaning no PCs added).
#' @param id_column Column name in phenotype/covariate file containing sample IDs (default: auto-detect from column names like IID, Sample, Taxa).
#' @param traits Comma-separated trait column names from phenotype file. For univariate models, selects the specified trait(s). For multivariate models, all traits are analyzed jointly.
#' @param trait_types Comma-separated trait types: 'ordinal' or 'continuous' (e.g., 'ordinal,ordinal,continuous').
#' @param n_categories Comma-separated number of categories per trait (use 0 for continuous, e.g., '3,2,0').
#' @param R_matrix Path to residual covariance matrix (c x c, CSV/NPY). If omitted, identity is used.
#' @param G_matrix Path to genetic covariance matrix (c x c, CSV/NPY). If omitted, identity is used.
#' @param solver Solver: 'nr' (T-EM warm-start then Newton-Raphson) or 'em' (SQUAREM-accelerated EM).
#' @param em_warmup Number of EM warm-up iterations before NR (default: 5).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_threshold_scan <- function(genotype,
                              phenotype,
                              covariate = NULL,
                              test = "wald",
                              correction = "bh",
                              maf_min =    0.01,
                              miss_max =     0.1,
                              output = "torchgenomics_results",
                              device = NULL,
                              chunk_size = 1024L,
                              max_iter = NULL,
                              weights_file = NULL,
                              gene_map = NULL,
                              fdr_covariate = NULL,
                              n_pcs = 0L,
                              id_column = NULL,
                              traits = NULL,
                              trait_types,
                              n_categories,
                              R_matrix = NULL,
                              G_matrix = NULL,
                              solver = "nr",
                              em_warmup = 5L) {
  args <- .compact(list(
    genotype = .as_path(genotype),
    phenotype = .as_path(phenotype),
    covariate = .as_path(covariate),
    test = if (is.null(test)) NULL else match.arg(test, c("wald", "score", "lrt")),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("bonferroni", "holm", "bh", "by", "storey", "simplem", "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt", "none")),
    maf_min = if (is.null(maf_min)) NULL else as.numeric(maf_min),
    miss_max = if (is.null(miss_max)) NULL else as.numeric(miss_max),
    output = .as_path(output),
    device = .as_path(device),
    chunk_size = if (is.null(chunk_size)) NULL else as.integer(chunk_size),
    max_iter = if (is.null(max_iter)) NULL else as.integer(max_iter),
    weights_file = .as_path(weights_file),
    gene_map = .as_path(gene_map),
    fdr_covariate = .as_path(fdr_covariate),
    n_pcs = if (is.null(n_pcs)) NULL else as.integer(n_pcs),
    id_column = .as_path(id_column),
    traits = .as_path(traits),
    trait_types = .as_path(trait_types),
    n_categories = .as_path(n_categories),
    R_matrix = .as_path(R_matrix),
    G_matrix = .as_path(G_matrix),
    solver = if (is.null(solver)) NULL else match.arg(solver, c("nr", "em")),
    em_warmup = if (is.null(em_warmup)) NULL else as.integer(em_warmup)
  ))
  d <- bridge_call("threshold_scan", args)
  GwasResult$new_from_dict(d, command = "threshold_scan")
}

#' Observed-expression TWAS: gene-trait association on already-normalized RNA-seq expression (no eQTL weights required).
#'
#' Auto-generated from torchgenomics `twas-scan` CLI subcommand.
#'
#' @param expression Path to expression TSV: header row of 'sample_id/tgene1/tgene2/t...'.
#' @param phenotype Phenotype TSV: PLINK-style 'FID/tIID/tTRAIT' or 'sample_id/tTRAIT'.
#' @param trait Trait column name. If omitted, the rightmost non-ID column in the phenotype file is used.
#' @param covariates Optional covariates TSV (PCs, PEER factors, sex, age, batch). Same ID convention as --phenotype.
#' @param kinship Optional (n, n) kinship / GRM matrix (.npy or tab-separated text). If supplied, switches OLS to LMM for population-stratification correction.
#' @param gene_annotation Optional gene-annotation BED: 'chr/tstart/tend/tgene_id(/tgene_name)'.
#' @param output Output TSV path.
#' @param correction Multiple-testing correction (default: bonferroni).
#' @param p_threshold Significance threshold (default: 0.05).
#' @param standardize Per-gene z-score the expression matrix before testing.
#' @param rank_int Apply Blom rank-INT per gene before testing.
#' @param quantile_norm Quantile-normalize columns before testing.
#' @param peer_factors Number of PEER factors to regress out before testing (requires R + peer).
#'
#' @return A `GwasResult` S4 object.
#' @export
tg_twas_scan <- function(expression,
                              phenotype,
                              trait = NULL,
                              covariates = NULL,
                              kinship = NULL,
                              gene_annotation = NULL,
                              output,
                              correction = "bonferroni",
                              p_threshold =    0.05,
                              standardize = FALSE,
                              rank_int = FALSE,
                              quantile_norm = FALSE,
                              peer_factors = NULL) {
  args <- .compact(list(
    expression = .as_path(expression),
    phenotype = .as_path(phenotype),
    trait = .as_path(trait),
    covariates = .as_path(covariates),
    kinship = .as_path(kinship),
    gene_annotation = .as_path(gene_annotation),
    output = .as_path(output),
    correction = if (is.null(correction)) NULL else match.arg(correction, c("none", "bonferroni", "holm", "bh", "by", "storey")),
    p_threshold = if (is.null(p_threshold)) NULL else as.numeric(p_threshold),
    standardize = if (is.null(standardize)) NULL else as.logical(standardize),
    rank_int = if (is.null(rank_int)) NULL else as.logical(rank_int),
    quantile_norm = if (is.null(quantile_norm)) NULL else as.logical(quantile_norm),
    peer_factors = if (is.null(peer_factors)) NULL else as.integer(peer_factors)
  ))
  d <- bridge_call("twas_scan", args)
  GwasResult$new_from_dict(d, command = "twas_scan")
}

