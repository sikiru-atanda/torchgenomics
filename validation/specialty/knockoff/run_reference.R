#!/usr/bin/env Rscript
# Reference: R knockoff package - Sesia/Candes 2020 + Candes/Fan/Janson/Lv 2018.
# For each replicate r in data/replicates.npz, run knockoff::knockoff.filter
# with create.fixed (fixed-design Gaussian knockoffs, valid because n>p here)
# and the default lasso-coef-diff statistic.  Record empirical FDR + power.

suppressPackageStartupMessages({ library(knockoff); library(glmnet); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
data_dir <- "validation/specialty/knockoff/data"
out_dir  <- "validation/specialty/knockoff/outputs"
target_fdr <- 0.2
n_replicates_override <- NA_integer_
i <- 1L
while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--data-dir")        { data_dir   <- args[[i + 1]]; i <- i + 2L }
    else if (a == "--output-dir") { out_dir    <- args[[i + 1]]; i <- i + 2L }
    else if (a == "--target-fdr") { target_fdr <- as.numeric(args[[i + 1]]); i <- i + 2L }
    else if (a == "--n-replicates") { n_replicates_override <- as.integer(args[[i + 1]]); i <- i + 2L }
    else { stop(sprintf("unknown arg: %s", a)) }
}

if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
truth <- fromJSON(file.path(data_dir, "sim_truth.json"))
causal_idx_one  <- as.integer(truth$causal_idx_zero_based) + 1L
n_replicates <- if (!is.na(n_replicates_override)) n_replicates_override else truth$n_replicates

rds_path <- file.path(data_dir, "replicates.rds")
npz_path <- file.path(data_dir, "replicates.npz")
if (!file.exists(rds_path) || file.info(rds_path)$mtime < file.info(npz_path)$mtime) {
    cat("[reference] converting replicates.npz -> replicates.rds via Python helper\n")
    converter <- file.path(dirname(data_dir), "_npz_to_rds.py")
    if (!file.exists(converter)) stop("helper not found: ", converter)
    rc <- system2("python3", c(converter, "--input", npz_path, "--output", rds_path))
    if (rc != 0L) stop("npz->rds conversion failed (rc=", rc, ")")
}
replicates <- readRDS(rds_path)

results <- list()
cat(sprintf("[reference] running %d replicates at target_fdr=%.3f\n", n_replicates, target_fdr))

for (r in seq_len(n_replicates)) {
    rep_key <- sprintf("rep_%d", r - 1L)
    G <- replicates[[rep_key]]$G
    y <- replicates[[rep_key]]$y
    n <- nrow(G); p <- ncol(G)
    set.seed(42L + r)
    G_scaled <- scale(G, center = TRUE, scale = TRUE)
    G_scaled[is.nan(G_scaled)] <- 0
    t0 <- Sys.time()
    kf <- tryCatch(
        knockoff.filter(
            X = G_scaled, y = y,
            knockoffs = function(x) create.fixed(x)$Xk,
            statistic = stat.glmnet_coefdiff,
            fdr = target_fdr, offset = 1L  # knockoff+ filter
        ),
        error = function(e) {
            warning(sprintf("rep %d: knockoff.filter failed: %s", r - 1L, conditionMessage(e)))
            list(selected = integer(0), threshold = Inf)
        }
    )
    elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    sel <- as.integer(kf$selected)
    sel_zero <- if (length(sel) > 0L) sel - 1L else integer(0)
    tp <- length(intersect(sel, causal_idx_one))
    fp <- length(setdiff(sel, causal_idx_one))
    fdr_emp <- if (length(sel) > 0L) fp / length(sel) else 0.0
    power_emp <- tp / length(causal_idx_one)
    results[[r]] <- list(
        rep = r - 1L, n_selected = length(sel),
        selected_zero_based = sel_zero,
        tp = tp, fp = fp, fdr = fdr_emp, power = power_emp,
        threshold = if (is.null(kf$threshold)) Inf else as.numeric(kf$threshold),
        elapsed_s = elapsed
    )
    if (r %% 10L == 0L || r == n_replicates) {
        cat(sprintf("  [%3d/%d] sel=%d tp=%d fp=%d fdr=%.3f pow=%.3f t=%.1fs\n",
                    r, n_replicates, length(sel), tp, fp, fdr_emp, power_emp, elapsed))
    }
}


tsv_path <- file.path(out_dir, "reference_per_replicate.tsv")
rows <- do.call(rbind, lapply(results, function(x) {
    data.frame(rep = x$rep, n_selected = x$n_selected, tp = x$tp, fp = x$fp,
               fdr = x$fdr, power = x$power, threshold = x$threshold,
               elapsed_s = x$elapsed_s)
}))
write.table(rows, tsv_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("[reference] wrote %s\n", tsv_path))

fdrs <- sapply(results, function(x) x$fdr)
pows <- sapply(results, function(x) x$power)
summary_list <- list(
    tool = "R knockoff::knockoff.filter",
    package_version = as.character(packageVersion("knockoff")),
    glmnet_version  = as.character(packageVersion("glmnet")),
    n_replicates = as.integer(n_replicates),
    target_fdr = target_fdr,
    statistic = "stat.glmnet_coefdiff",
    knockoffs = "create.fixed",
    offset = 1L,
    mean_fdr = mean(fdrs),
    sd_fdr   = if (length(fdrs) > 1L) sd(fdrs) else 0.0,
    sem_fdr  = if (length(fdrs) > 1L) sd(fdrs) / sqrt(length(fdrs)) else 0.0,
    mean_power = mean(pows),
    sd_power   = if (length(pows) > 1L) sd(pows) else 0.0,
    sem_power  = if (length(pows) > 1L) sd(pows) / sqrt(length(pows)) else 0.0,
    mean_elapsed_s = mean(sapply(results, function(x) x$elapsed_s))
)
writeLines(toJSON(summary_list, auto_unbox = TRUE, pretty = TRUE),
           file.path(out_dir, "reference_summary.json"))
cat(sprintf("[reference] mean_fdr=%.4f +/- %.4f   mean_power=%.4f +/- %.4f\n",
            summary_list$mean_fdr, summary_list$sem_fdr,
            summary_list$mean_power, summary_list$sem_power))

