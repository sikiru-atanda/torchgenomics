#!/usr/bin/env Rscript
# Reference implementations for p-value combination methods from the
# CRAN `metap` package plus Bioconductor `EmpiricalBrownsMethod`.
#
# Each row of data/pvalues.tsv is combined using:
#   - Fisher's combined test           (metap::sumlog)
#   - Stouffer's Z                     (metap::sumz)
#   - Harmonic mean p-value            (metap::sumlog.harmonic IF present, else manual)
#   - Tippett's min-p                  (metap::minimump)
#   - Empirical Brown                  (EmpiricalBrownsMethod::empiricalBrownsMethod, with data_matrix.tsv)
#
# Output: outputs/reference.tsv with columns scenario, method, p_combined.

argv <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag) {
    i <- which(argv == flag)
    if (length(i) == 1 && i + 1 <= length(argv)) return(argv[i + 1])
    NULL
}
P_TSV  <- parse_arg("--p-tsv")
D_TSV  <- parse_arg("--data-tsv")
OUT    <- parse_arg("--out")
stopifnot(!is.null(P_TSV), !is.null(OUT))

have_metap <- requireNamespace("metap", quietly = TRUE)
have_ebm   <- requireNamespace("EmpiricalBrownsMethod", quietly = TRUE)

p_df <- read.table(P_TSV, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
scenarios <- as.character(p_df$scenario)
p_mat <- as.matrix(p_df[, -1, drop = FALSE])
k <- ncol(p_mat)

# Helper: clamp p-values away from {0, 1} for the metap routines which
# can return NaN at the boundaries.
.clamp <- function(p) pmin(pmax(p, 1e-300), 1 - 1e-16)

rows <- list()
add_row <- function(scen, method, p_comb) {
    rows[[length(rows) + 1L]] <<- data.frame(
        scenario = scen, method = method, p_combined = as.numeric(p_comb),
        stringsAsFactors = FALSE
    )
}

for (i in seq_along(scenarios)) {
    scen <- scenarios[i]
    p_vec <- .clamp(p_mat[i, ])

    if (have_metap) {
        # Fisher
        add_row(scen, "fisher", metap::sumlog(p_vec)$p)
        # Stouffer
        add_row(scen, "stouffer", metap::sumz(p_vec)$p)
        # HMP — older metap exposes `meanp` only; harmonic mean p
        # is in metap >=1.7 under `metap::hmp.stat`. We compute the
        # canonical analytic form when the package routine is missing.
        if ("hmp.stat" %in% getNamespaceExports("metap")) {
            add_row(scen, "hmp", metap::hmp.stat(p_vec))
        } else {
            hmp_val <- length(p_vec) / sum(1 / p_vec)
            L <- log(length(p_vec)) + 0.5772156649015329
            add_row(scen, "hmp", min(L * hmp_val, 1))
        }
        # Min-p (Tippett)
        add_row(scen, "min_p", metap::minimump(p_vec)$p)
    }

    # Empirical Brown requires data_matrix.tsv to estimate covariances.
    if (have_ebm && !is.null(D_TSV)) {
        if (file.exists(D_TSV)) {
            d_mat <- as.matrix(read.table(D_TSV, header = TRUE, sep = "\t"))
            ebm_p <- tryCatch(
                EmpiricalBrownsMethod::empiricalBrownsMethod(
                    data_matrix = t(d_mat),  # EBM expects rows = features
                    p_values    = p_vec,
                    extra_info  = FALSE
                ),
                error = function(e) NA_real_
            )
            add_row(scen, "empirical_brown", ebm_p)
        }
    }
}

if (length(rows) == 0L) {
    cat("[metap reference] warning: no R packages available; output is empty.\n")
    df <- data.frame(
        scenario = character(), method = character(),
        p_combined = numeric()
    )
} else {
    df <- do.call(rbind, rows)
}
write.table(df, file = OUT, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[metap reference] wrote %s with %d rows\n", OUT, nrow(df)))
