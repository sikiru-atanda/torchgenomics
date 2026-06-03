#!/usr/bin/env Rscript
# Run hyprcoloc on the simulated 3-trait sumstats and emit a JSON blob
# the Python harness (compare.py) parses.
#
# Output layout (outputs/hyprcoloc_results.json):
#   {
#     "results": [
#       {
#         "iteration": 1,
#         "traits_str": "T1, T2, T3",
#         "traits_one_based": [1, 2, 3],
#         "traits_zero_based": [0, 1, 2],
#         "posterior_prob": ...,         # joint (cluster regional PP * per-SNP PP)
#         "regional_prob": ...,          # cluster regional PP (matches TG best_cluster_posterior)
#         "candidate_snp": "rs00050",
#         "posterior_explained_by_snp": ...,  # per-SNP PP within cluster (matches TG candidate_snp_posterior)
#         "dropped_trait": null
#       }
#     ],
#     "snpscores_per_iter": [[...per-SNP scores...]],  # one vector per iteration
#     "snp_ids": ["rs00001", ...],
#     "trait_names": ["T1", "T2", "T3"],
#     "m": 100, "k": 3,
#     "package_versions": {
#       "hyprcoloc": "...", "Rmpfr": "...", "gmp": "...", "RcppEigen": "...",
#       "R": "...", "seed": 42
#     }
#   }
#
# hyprcoloc API note:
#   The package returns r$results as a 1-row data.frame per detected cluster
#   (iterative greedy clustering; each row is one identified cluster).
#   For the shared-causal 3-trait fixture we expect a single row covering
#   {T1, T2, T3}. r$snpscores is a list of per-SNP scores, one entry per
#   iteration; the vector sums to 1.0 across SNPs.

suppressPackageStartupMessages({
    library(hyprcoloc)
    library(jsonlite)
})

# --- CLI args ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA_DIR <- parse_arg("--data-dir", "data")
OUT_DIR  <- parse_arg("--output-dir", "outputs")
SEED     <- as.integer(parse_arg("--seed", "42"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf(
    "[run] data_dir=%s out_dir=%s seed=%d\n",
    DATA_DIR, OUT_DIR, SEED
))

# --- Load simulated sumstats and truth ---------------------------------------
sumstats <- read.delim(file.path(DATA_DIR, "sumstats.tsv"), stringsAsFactors = FALSE)
truth <- fromJSON(file.path(DATA_DIR, "sim_truth.json"))

trait_names <- truth$trait_names
m <- truth$m
K <- truth$k

# Reshape long-format sumstats to (m x K) matrices of beta + se.
snp_ids <- unique(sumstats$SNP)
stopifnot(length(snp_ids) == m)

beta_mat <- matrix(NA_real_, nrow = m, ncol = K, dimnames = list(snp_ids, trait_names))
se_mat   <- matrix(NA_real_, nrow = m, ncol = K, dimnames = list(snp_ids, trait_names))
for (k in seq_len(K)) {
    sub <- sumstats[sumstats$trait == trait_names[k], ]
    stopifnot(nrow(sub) == m)
    # Preserve the canonical snp_ids order; reorder if needed.
    sub <- sub[match(snp_ids, sub$SNP), ]
    beta_mat[, k] <- sub$beta
    se_mat[, k]   <- sub$se
}

stopifnot(!any(is.na(beta_mat)))
stopifnot(!any(is.na(se_mat)))

cat(sprintf(
    "[run] loaded %d SNPs x %d traits; causal=%s (idx %d)\n",
    m, K, truth$causal_snp, truth$causal_index_one_based
))

# --- Run hyprcoloc -----------------------------------------------------------
# Defaults (per Foley 2021):
#   prior.1 = 1e-4  - per-trait associated prior
#   prior.c = 0.02  - conditional prior of shared causal given two associated
#   bb.alg  = TRUE  - branch-and-bound algorithm (default; faster for K large)
#   bb.selection = "regional" - regional PP for stopping (matches paper)
#
# TorchGenomics prior wiring:
#   prior_1 = 1e-4 - matches hyprcoloc::prior.1
#   prior_2 = 0.98 - TG's "prior probability that two associated traits share
#                    a causal variant". hyprcoloc parameterizes the same
#                    quantity as prior.c = 0.02 = 1 - prior_2 (under the
#                    Foley 2021 reparameterization).
#   prior_w = 0.15^2 - matches the Wakefield prior variance default.

set.seed(SEED)
hc <- hyprcoloc(
    effect.est   = beta_mat,
    effect.se    = se_mat,
    trait.names  = trait_names,
    snp.id       = snp_ids,
    snpscores    = TRUE,
    bb.alg       = TRUE,
    bb.selection = "regional",
    prior.1      = 1e-4,
    prior.c      = 0.02
)

cat("[run] hyprcoloc returned", nrow(hc$results), "cluster row(s)\n")
print(hc$results)

# --- Parse cluster traits ---------------------------------------------------
# r$results$traits is comma-separated string like "T1, T2, T3" (or
# "None" if no cluster).
parse_cluster <- function(traits_str) {
    if (is.na(traits_str) || traits_str == "None") {
        return(list(zero_based = integer(0), one_based = integer(0), names = character(0)))
    }
    parts <- trimws(strsplit(traits_str, ",", fixed = TRUE)[[1]])
    idx <- match(parts, trait_names)
    list(
        zero_based = as.integer(idx - 1L),
        one_based  = as.integer(idx),
        names      = parts
    )
}

# --- Compose JSON output -----------------------------------------------------
result_rows <- lapply(seq_len(nrow(hc$results)), function(i) {
    row <- hc$results[i, ]
    cl <- parse_cluster(as.character(row$traits))
    list(
        iteration                  = as.integer(row$iteration),
        traits_str                 = as.character(row$traits),
        traits_one_based           = cl$one_based,
        traits_zero_based          = cl$zero_based,
        trait_names                = cl$names,
        posterior_prob             = unname(row$posterior_prob),
        regional_prob              = unname(row$regional_prob),
        candidate_snp              = as.character(row$candidate_snp),
        posterior_explained_by_snp = unname(row$posterior_explained_by_snp),
        dropped_trait              = if (is.na(row$dropped_trait)) NA else as.character(row$dropped_trait)
    )
})

snpscores_per_iter <- lapply(hc$snpscores, function(v) {
    if (is.null(v) || length(v) == 0) return(numeric(0))
    as.numeric(v)
})

out <- list(
    results            = result_rows,
    snpscores_per_iter = snpscores_per_iter,
    snp_ids            = snp_ids,
    trait_names        = trait_names,
    m                  = m,
    k                  = K,
    causal_snp         = truth$causal_snp,
    causal_index_zero_based = truth$causal_index_zero_based,
    package_versions = list(
        hyprcoloc = as.character(packageVersion("hyprcoloc")),
        Rmpfr     = as.character(packageVersion("Rmpfr")),
        gmp       = as.character(packageVersion("gmp")),
        RcppEigen = as.character(packageVersion("RcppEigen")),
        R         = R.version.string,
        seed      = SEED
    )
)

out_path <- file.path(OUT_DIR, "hyprcoloc_results.json")
write_json(out, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf("[run] wrote %s\n", out_path))

# Echo a one-line summary for quick eyeballing.
if (length(result_rows) >= 1) {
    r1 <- result_rows[[1]]
    cat(sprintf(
        "[run] cluster=%s | regional_pp=%.4f | candidate=%s | pp_explained=%.4f\n",
        r1$traits_str, r1$regional_prob, r1$candidate_snp, r1$posterior_explained_by_snp
    ))
}

