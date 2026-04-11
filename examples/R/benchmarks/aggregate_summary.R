## =============================================================================
## aggregate_summary.R -- merge per-phase CSV summaries into one report
## =============================================================================

out_dir <- "C:/Users/Sikiru/Documents/GWAS_Expert/examples/R/benchmark_results"

read_phase <- function(name, label) {
  f <- file.path(out_dir, name)
  if (!file.exists(f)) return(NULL)
  df <- read.csv(f, stringsAsFactors = FALSE)
  df$source_file <- basename(f)
  df
}

A <- read_phase("phase_a_lmm_summary.csv",     "A")
B <- read_phase("phase_b_gapit_summary.csv",   "B")
C <- read_phase("phase_c_mvlmm_summary.csv",   "C")
D <- read_phase("phase_d_gwaspoly_summary.csv","D")
E <- read_phase("phase_e_susie_summary.csv",   "E")
F <- read_phase("phase_f_plink2_summary.csv",  "F")
G <- read_phase("phase_g_gcta_cojo_summary.csv","G")

## Build a unified table with the columns common to all phases.
to_row <- function(df, phase, dataset, ref, verb, n_snps_col, cor_col,
                   verdict_col = "verdict", extra_label = NA_character_) {
  if (is.null(df)) return(NULL)
  data.frame(
    phase            = phase,
    label            = if (!is.na(extra_label)) extra_label else
                       (if ("label" %in% names(df)) df$label else NA_character_),
    reference_tool   = ref,
    rtorchgwas_verb  = verb,
    dataset          = dataset,
    n_snps_compared  = if (n_snps_col %in% names(df)) df[[n_snps_col]] else NA,
    primary_metric   = if (cor_col %in% names(df)) df[[cor_col]] else NA,
    verdict          = df[[verdict_col]],
    stringsAsFactors = FALSE
  )
}

rows <- list()

## Phase A
rows[[length(rows)+1]] <- data.frame(
  phase = "A", label = "LMM",
  reference_tool = A$reference_tool,
  rtorchgwas_verb = A$rtorchgwas_verb,
  dataset = A$dataset,
  n_snps_compared = A$n_snps_compared,
  primary_metric  = A$cor_neg_log10_p,
  verdict = A$verdict,
  stringsAsFactors = FALSE
)

## Phase B (4 sub-rows)
rows[[length(rows)+1]] <- data.frame(
  phase = "B", label = B$label,
  reference_tool = B$reference_tool,
  rtorchgwas_verb = B$rtorchgwas_verb,
  dataset = B$dataset,
  n_snps_compared = B$n,
  primary_metric  = ifelse(B$label %in% c("GLM", "LMM/MLM"), B$pearson, B$jaccard),
  verdict = B$verdict,
  stringsAsFactors = FALSE
)

## Phase C
rows[[length(rows)+1]] <- data.frame(
  phase = "C", label = "mvLMM",
  reference_tool = C$reference_tool,
  rtorchgwas_verb = C$rtorchgwas_verb,
  dataset = C$dataset,
  n_snps_compared = C$n_snps_compared,
  primary_metric  = C$cor_neg_log10_p,
  verdict = C$verdict,
  stringsAsFactors = FALSE
)

## Phase D
rows[[length(rows)+1]] <- data.frame(
  phase = "D", label = "Polyploid additive",
  reference_tool = D$reference_tool,
  rtorchgwas_verb = D$rtorchgwas_verb,
  dataset = D$dataset,
  n_snps_compared = D$n_markers_compared,
  primary_metric  = D$cor_neg_log10_p,
  verdict = D$verdict,
  stringsAsFactors = FALSE
)

## Phase E (one row per region)
rows[[length(rows)+1]] <- data.frame(
  phase = "E", label = paste0("SuSiE-", E$region),
  reference_tool = E$reference_tool,
  rtorchgwas_verb = E$rtorchgwas_verb,
  dataset = E$dataset,
  n_snps_compared = E$n_snps,
  primary_metric  = E$spearman,
  verdict = E$verdict,
  stringsAsFactors = FALSE
)

## Phase F
rows[[length(rows)+1]] <- data.frame(
  phase = "F", label = "Binary GLM",
  reference_tool = F$reference_tool,
  rtorchgwas_verb = F$rtorchgwas_verb,
  dataset = F$dataset,
  n_snps_compared = F$n_snps_compared,
  primary_metric  = F$cor_neg_log10_p,
  verdict = F$verdict,
  stringsAsFactors = FALSE
)

## Phase G
if (!is.null(G)) {
  rows[[length(rows)+1]] <- data.frame(
    phase = "G", label = "Conditional (COJO)",
    reference_tool = G$reference_tool,
    rtorchgwas_verb = G$rtorchgwas_verb,
    dataset = G$dataset,
    n_snps_compared = G$n_union,
    primary_metric  = G$jaccard,
    verdict = G$verdict,
    stringsAsFactors = FALSE
  )
}

agg <- do.call(rbind, rows)
agg <- agg[order(agg$phase, agg$label), , drop = FALSE]

cat("\n=== rTorchGWAS cross-tool benchmark aggregate ===\n\n")
print(agg, row.names = FALSE)

write.csv(agg, file.path(out_dir, "phase_all_summary.csv"), row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "phase_all_summary.csv")))

## Pass / fail tally
n_pass    <- sum(agg$verdict == "PASS",    na.rm = TRUE)
n_review  <- sum(agg$verdict == "REVIEW",  na.rm = TRUE)
n_total   <- nrow(agg)
cat(sprintf("\n%d / %d comparisons PASS  (%d REVIEW)\n",
            n_pass, n_total, n_review))
