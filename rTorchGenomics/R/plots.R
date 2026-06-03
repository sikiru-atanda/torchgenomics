# rTorchGenomics/R/plots.R

#' Manhattan plot of a GWAS scan result.
#'
#' Reads the full result TSV from `output_files$tsv` if present;
#' otherwise plots only the inline `top_hits`.
#'
#' @param x A `ScanRun` (or any object with `output_files` and `top_hits`).
#' @param significance_threshold Horizontal red line. Default 5e-8.
#' @param suggestive_threshold Horizontal grey line. Default 1e-5.
#' @param title Plot title.
#' @param chr_col,pos_col,p_col Column names in the result table.
#'
#' @return A `ggplot` object.
#' @examples
#' \dontrun{
#' r <- tg_lmm_scan("data.bed", "pheno.tsv")
#' tg_manhattan(r)
#' }
#' @importFrom ggplot2 .data
#' @export
tg_manhattan <- function(x,
                         significance_threshold = 5e-8,
                         suggestive_threshold = 1e-5,
                         title = NULL,
                         chr_col = "CHR", pos_col = "POS", p_col = "P") {
  df <- .read_results_or_top_hits(x, chr_col, pos_col, p_col)
  df$.neglog10p <- -log10(df[[p_col]])
  df$.chr_num <- as.integer(factor(df[[chr_col]], levels = unique(df[[chr_col]])))

  # Cumulative position so chromosomes lay out horizontally.
  df <- df[order(df$.chr_num, df[[pos_col]]), ]
  chr_offsets <- tapply(df[[pos_col]], df$.chr_num, max)
  cum_offset <- c(0, cumsum(as.numeric(chr_offsets))[-length(chr_offsets)])
  df$.cum_pos <- df[[pos_col]] + cum_offset[df$.chr_num]

  axis_breaks <- tapply(df$.cum_pos, df$.chr_num, function(v) mean(range(v)))
  axis_labels <- unique(df[[chr_col]])

  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.cum_pos, y = .data$.neglog10p,
                                        colour = as.factor(.data$.chr_num))) +
    ggplot2::geom_point(size = 0.8, alpha = 0.8) +
    ggplot2::scale_colour_manual(values = rep(c("#1f77b4", "#d62728"),
                                              length.out = max(df$.chr_num))) +
    ggplot2::geom_hline(yintercept = -log10(significance_threshold),
                        colour = "red", linetype = "dashed") +
    ggplot2::scale_x_continuous(breaks = axis_breaks, labels = axis_labels) +
    ggplot2::labs(
      title = title %||% paste0("Manhattan - ",
                                if (methods::is(x, "ScanRun")) x@model else "scan"),
      x = "Chromosome",
      y = expression(-log[10](italic(p)))
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position = "none",
                   panel.grid.minor = ggplot2::element_blank())

  if (!is.null(suggestive_threshold)) {
    p <- p + ggplot2::geom_hline(yintercept = -log10(suggestive_threshold),
                                 colour = "grey", linetype = "dotted")
  }
  p
}

#' Q-Q plot of GWAS p-values.
#'
#' @param x A `ScanRun` or compatible object.
#' @param title Plot title.
#' @param p_col Column name for p-values.
#'
#' @return A `ggplot` object.
#' @examples
#' \dontrun{
#' tg_qq(tg_lmm_scan("data.bed", "pheno.tsv"))
#' }
#' @importFrom ggplot2 .data
#' @export
tg_qq <- function(x, title = NULL, p_col = "P") {
  df <- .read_results_or_top_hits(x, "CHR", "POS", p_col)
  pvals <- sort(df[[p_col]])
  n <- length(pvals)
  expected <- -log10((seq_len(n) - 0.5) / n)
  observed <- -log10(pvals)
  d <- data.frame(expected = expected, observed = observed)

  ggplot2::ggplot(d, ggplot2::aes(x = .data$expected, y = .data$observed)) +
    ggplot2::geom_abline(slope = 1, intercept = 0,
                         colour = "grey", linetype = "dashed") +
    ggplot2::geom_point(size = 1, alpha = 0.8, colour = "#1f77b4") +
    ggplot2::labs(
      title = title %||% "Q-Q plot",
      x = expression(Expected ~ -log[10](italic(p))),
      y = expression(Observed ~ -log[10](italic(p)))
    ) +
    ggplot2::theme_minimal()
}

#' Resolve the data source for a plotting helper.
#'
#' Returns a data.frame with at least `chr_col`, `pos_col`, `p_col`. Prefers
#' the on-disk TSV at `output_files$tsv`; falls back to the inline `top_hits`
#' tibble for ScanRun objects.
#'
#' @keywords internal
#' @noRd
.read_results_or_top_hits <- function(x, chr_col, pos_col, p_col) {
  tsv <- NULL
  if (methods::is(x, "_BaseRun") || isS4(x)) {
    of <- try(x@output_files, silent = TRUE)
    if (!inherits(of, "try-error") && is.list(of)) {
      tsv <- of[["tsv"]]
    }
  }
  if (!is.null(tsv) && file.exists(tsv)) {
    df <- readr::read_tsv(tsv, show_col_types = FALSE,
                          progress = FALSE,
                          col_types = readr::cols(.default = readr::col_guess()))
    df <- as.data.frame(df)
  } else if (methods::is(x, "ScanRun") && nrow(x@top_hits) > 0L) {
    df <- as.data.frame(x@top_hits)
  } else {
    stop("No result data found on ", class(x), " (need output_files$tsv or top_hits).")
  }
  required <- c(chr_col, pos_col, p_col)
  missing <- setdiff(required, names(df))
  if (length(missing) > 0L) {
    stop("Result is missing columns: ", paste(missing, collapse = ", "))
  }
  df[df[[p_col]] > 0 & !is.na(df[[p_col]]), , drop = FALSE]
}
