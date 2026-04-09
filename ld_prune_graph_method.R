
#' Title
#'
#' @param i_vec
#' @param j_vec
#'
#' @return
#' @export
#'
#' @examples
ld_pair_vec <- function(i_vec, j_vec) {
  keep <- !(is.na(i_vec) | is.na(j_vec))
  g1 <- i_vec[keep]; g2 <- j_vec[keep]
  n  <- length(g1)
  if (n == 0) return(c(NA_real_, NA_real_))
  pA <- mean(g1) / 2;  pB <- mean(g2) / 2
  pAB <- mean((g1 == 2 & g2 == 2) +
                0.5 * (g1 == 2 & g2 == 1) +
                0.5 * (g1 == 1 & g2 == 2) +
                0.25 * (g1 == 1 & g2 == 1))
  D  <- pAB - pA * pB
  denom <- if (D >= 0)
    min(pA * (1 - pB), (1 - pA) * pB)
  else
    max(-pA * pB, -(1 - pA) * (1 - pB))
  Dprime <- if (denom == 0) NA_real_ else D / denom
  R2     <- if (pA * (1 - pA) * pB * (1 - pB) == 0)
    NA_real_
  else
    D * D / (pA * (1 - pA) * pB * (1 - pB))
  c(R2, Dprime)
}

#' Title
#'
#' @param G
#' @param snp_ids
#' @param window
#' @param progress
#' @param as.data.frame
#'
#' @return
#' @export
#'
#' @examples
ld_window <- function(G, snp_ids = colnames(G), window = 100L,
                      progress = TRUE, as.data.frame = TRUE) {
  M <- length(snp_ids)
  if (progress) pb <- txtProgressBar(min = 0, max = M, style = 3)
  out <- vector("list", M)
  for (i in seq_len(M - 1L)) {
    if (progress) setTxtProgressBar(pb, i)
    j_end <- min(M, i + window)
    if (j_end <= i) next
    js <- seq.int(i + 1L, j_end)
    res <- vapply(js, function(j) ld_pair_vec(G[, i], G[, j]), numeric(2L))
    out[[i]] <- data.table::data.table(
      SNP1   = rep.int(snp_ids[i], length(js)),
      SNP2   = snp_ids[js],
      R2     = res[1L, ],
      Dprime = res[2L, ]
    )
  }
  if (progress) close(pb)
  res <- data.table::rbindlist(out, use.names = TRUE)
  if (as.data.frame) res <- as.data.frame(res)
  res
}

#' Title
#'
#' @param geno_data
#' @param window
#' @param R2_threshold
#' @param maf_thresh
#'
#' @return
#' @export
#'
#' @examples
ld_prune_graph <- function(geno_data, window = 100L, R2_threshold = 0.9, maf_thresh = 0.01) {
  if (!is.matrix(geno_data)) geno_data <- as.matrix(geno_data)

  M <- ncol(geno_data)
  snp_ids <- colnames(geno_data)

  #message("Calculating LD...")
  message("Initialize geno optimization...")
  ld_df <- ld_window(as.matrix(geno_data), snp_ids = snp_ids, window = window, progress = TRUE, as.data.frame = TRUE)
  ld_df <- ld_df[ld_df$R2 > R2_threshold & !is.na(ld_df$R2), ]


  #message("Building LD graph...")
  g <- igraph::graph_from_data_frame(ld_df[, c("SNP1", "SNP2")], directed = FALSE)

  # compute MAF
  maf_vec <- colMeans(geno_data, na.rm = TRUE) / 2
  maf_vec <- pmin(maf_vec, 1 - maf_vec)

  # Remove SNPs with too low MAF
  keep_maf <- which(maf_vec >= maf_thresh)
  geno_data <- geno_data[, keep_maf, drop = FALSE]
  snp_ids <- colnames(geno_data)

  # Subset graph to SNPs that passed MAF threshold
  g <- igraph::induced_subgraph(g, vids = intersect(snp_ids, igraph::V(g)$name))

  #message("Pruning LD graph...")
  comps <- igraph::components(g)

  # One SNP per component
  keep <- c()
  for (i in seq_len(comps$no)) {
    snps <- names(comps$membership[comps$membership == i])
    if (length(snps) == 1) {
      keep <- c(keep, snps)
    } else {
      maf_subset <- maf_vec[snps]
      chosen <- snps[which.max(maf_subset)]
      keep <- c(keep, chosen)
    }
  }

  # Add singleton SNPs that weren't in any LD pair
  all_ld_snps <- unique(c(ld_df$SNP1, ld_df$SNP2))
  singleton_snps <- setdiff(snp_ids, all_ld_snps)
  keep <- unique(c(keep, singleton_snps))

  #message(sprintf("Pruned to %d SNPs from %d", length(keep), ncol(geno_data)))
  message(sprintf("Trimmed to %d SNPs from %d", length(keep), ncol(geno_data)))
  return(keep)
}

# rm(list = ls()); ls()
# gc()
# cat('\014')
# graphics.off()
#
# setwd("D:/PredictProR_no_use_file")
# load("barley_data.Rdata")
# keep_prunned_snp <- ld_prune_graph(geno_data)
#
# geno_data <- as.matrix(geno_data)
# geno_data <- geno_data[, keep_prunned_snp, drop=FALSE]
#
# dim(geno_data)
