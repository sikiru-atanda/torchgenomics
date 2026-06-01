#!/usr/bin/env Rscript
# Run coloc::coloc.abf on each scenario in data/ and emit
# outputs/coloc_results.json keyed by scenario.
#
# Method exercised:
#   coloc::coloc.abf(dataset1, dataset2,
#                    p1 = 1e-4, p2 = 1e-4, p12 = 1e-5)
#
# Defaults match the Giambartolomei 2014 paper exactly:
#   prior_w = 0.15^2 (Wakefield prior variance on beta) -- this is the
#                    default for `type = "quant"` with `sdY = 1.0`.
#   p1 = p2 = 1e-4   (per-SNP prior of association for each trait)
#   p12 = 1e-5       (per-SNP prior of shared causal variant)
#
# Output layout (outputs/coloc_results.json):
#   {
#     "shared":   {nsnps, PP.H0..H4, candidate_snp, priors{...}},
#     "distinct": {...},
#     "null":     {...},
#     "package_versions": {coloc, jsonlite, R},
#     "defaults": {p1, p2, p12, prior_w}
#   }

suppressPackageStartupMessages({
    library(coloc)
    library(jsonlite)
})

# --- CLI args ----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA_DIR <- parse_arg("--data-dir", "data")
OUT_DIR  <- parse_arg("--output-dir", "outputs")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[run_coloc] data_dir=%s out_dir=%s\n", DATA_DIR, OUT_DIR))

# --- Load manifest ----------------------------------------------------------
manifest_path <- file.path(DATA_DIR, "scenarios.json")
if (!file.exists(manifest_path)) {
    stop(sprintf("manifest not found: %s -- run fetch_data.sh first.", manifest_path))
}
manifest <- fromJSON(manifest_path, simplifyVector = FALSE)

# --- Defaults (per Giambartolomei 2014; coloc.abf documented defaults) -------
P1  <- 1e-4
P2  <- 1e-4
P12 <- 1e-5
PRIOR_W <- 0.15^2

# --- Run one scenario through coloc.abf -------------------------------------
run_one <- function(scenario_name, scenario_meta) {
    sumstats_path <- file.path(DATA_DIR, scenario_meta$file)
    if (!file.exists(sumstats_path)) {
        stop(sprintf("sumstats not found: %s", sumstats_path))
    }
    df <- read.delim(sumstats_path, stringsAsFactors = FALSE)
    nsnps <- nrow(df)

    # coloc.abf expects beta + varbeta + snp + type for each dataset.
    # type = "quant" with sdY = 1.0 makes coloc.abf use the Wakefield ABF
    # with W = sdY^2 * 0.15^2 = 0.15^2 -- matches TG default prior_w.
    d1 <- list(
        beta    = df$beta1,
        varbeta = df$se1^2,
        snp     = df$SNP,
        type    = "quant",
        sdY     = 1.0
    )
    d2 <- list(
        beta    = df$beta2,
        varbeta = df$se2^2,
        snp     = df$SNP,
        type    = "quant",
        sdY     = 1.0
    )

    # coloc.abf emits stdout via message(); suppress for clean JSON harness.
    res <- suppressMessages(
        coloc.abf(d1, d2, p1 = P1, p2 = P2, p12 = P12)
    )

    # Summary: PP.H0..H4.abf are the posterior probs. Candidate SNP is the
    # SNP with the highest per-SNP H4 posterior in res$results.
    pp <- as.list(res$summary)   # named PP.H0.abf .. PP.H4.abf + nsnps

    # Per-SNP results: res$results has columns snp, position, SNP.PP.H4 etc.
    # Candidate SNP under H4 is argmax of SNP.PP.H4.
    perSNP <- res$results
    cand_idx_one_based <- which.max(perSNP$SNP.PP.H4)
    cand_snp_id <- as.character(perSNP$snp[cand_idx_one_based])

    list(
        scenario = scenario_name,
        nsnps    = unname(pp$nsnps),
        PP.H0.abf = unname(pp$PP.H0.abf),
        PP.H1.abf = unname(pp$PP.H1.abf),
        PP.H2.abf = unname(pp$PP.H2.abf),
        PP.H3.abf = unname(pp$PP.H3.abf),
        PP.H4.abf = unname(pp$PP.H4.abf),
        candidate_snp_id           = cand_snp_id,
        candidate_snp_index_one_based  = as.integer(cand_idx_one_based),
        candidate_snp_index_zero_based = as.integer(cand_idx_one_based - 1L),
        priors   = list(p1 = P1, p2 = P2, p12 = P12, prior_w = PRIOR_W),
        expected_pp = scenario_meta$expected_pp
    )
}

# --- Execute every scenario in the manifest ---------------------------------
scenario_names <- names(manifest$scenarios)
scenario_blocks <- list()
for (nm in scenario_names) {
    cat(sprintf("[run_coloc] running coloc.abf on scenario %s\n", nm))
    block <- run_one(nm, manifest$scenarios[[nm]])
    scenario_blocks[[nm]] <- block
    cat(sprintf(
        "[run_coloc]   PP.H0=%.4e  PP.H1=%.4e  PP.H2=%.4e  PP.H3=%.4e  PP.H4=%.4e  cand=%s\n",
        block$PP.H0.abf, block$PP.H1.abf, block$PP.H2.abf,
        block$PP.H3.abf, block$PP.H4.abf, block$candidate_snp_id
    ))
}

# --- Compose JSON output -----------------------------------------------------
out <- c(
    scenario_blocks,
    list(
        package_versions = list(
            coloc    = as.character(packageVersion("coloc")),
            jsonlite = as.character(packageVersion("jsonlite")),
            R        = R.version.string
        ),
        defaults = list(p1 = P1, p2 = P2, p12 = P12, prior_w = PRIOR_W),
        n_scenarios = length(scenario_names),
        manifest_seed = manifest$seed,
        manifest_causal_idx_zero_based = manifest$causal_idx_zero_based
    )
)

out_path <- file.path(OUT_DIR, "coloc_results.json")
write_json(out, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf("[run_coloc] wrote %s\n", out_path))
