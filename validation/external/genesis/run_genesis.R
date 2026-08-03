#!/usr/bin/env Rscript
# GENESIS reference pipeline for Phase 57 Unit A Task 6:
#
#     PLINK BED -> snpgdsBED2GDS -> snpgdsIBDKING(type="KING-robust")
#               -> GENESIS::pcair() -> GENESIS::pcrelate()
#
# This is the *definitive* external reference for
# torchgenomics.linalg.kinship_admixed.{king_robust_kinship, pc_air,
# pc_relate}. Every internal torchgenomics test in
# tests/test_kinship_admixed.py explicitly disclaims itself as
# reference-equivalence evidence; this script (+ compare.py) is what closes
# that gap.
#
# Input:  validation/external/genesis/out/fixture.{bed,bim,fam}
#         validation/external/genesis/out/pruned_snp_ids.txt
#         (both produced by export_fixture.py; run that first)
#
# Output (all TSV, written to validation/external/genesis/out/):
#   genesis_king_kinship.tsv     -- (n, n) KING-robust kinship, full unpruned
#                                    marker panel (m=2000)
#   genesis_pcair_pcs.tsv        -- (n, n_pcs) PC-AiR ancestry PCs, computed
#                                    on the SAME LD-pruned marker subset that
#                                    torchgenomics' ld_prune_independent()
#                                    selected (pruned_snp_ids.txt) -- so both
#                                    tools' PC-AiR input is byte-identical,
#                                    not just independently LD-pruned.
#   genesis_pcrelate_kinship.tsv -- (n, n) PC-Relate ancestry-adjusted
#                                    kinship, same pruned marker subset.
#
# Why KING uses the FULL marker panel but PC-AiR/PC-Relate use the PRUNED
# subset: KING-robust is an IBS-category count (per-marker het/hom-hom
# indicators averaged over markers), not a PCA/regression step, so it is
# computed on whatever marker panel is common to both tools -- here, the
# entire unpruned fixture, exactly as both snpgdsIBDKING and
# torchgenomics.king_robust_kinship(G) (not G_pruned) ingest it in
# compare.py. PC-AiR and PC-Relate, by contrast, are explicitly specified to
# take a *pruned* marker panel (Conomos et al. 2015 recommend r2 < 0.1
# pruning before PCA) -- and because the pruning algorithm itself
# (torchgenomics' greedy left-to-right pass vs. SNPRelate's
# snpgdsLDpruning) is NOT under test here, both sides are pinned to
# torchgenomics' chosen pruned subset (pruned_snp_ids.txt) via
# snp.include= below, so a PC-AiR/PC-Relate disagreement can only be
# attributed to the algorithm itself, never to a marker-panel mismatch.
#
# Usage:
#   Rscript run_genesis.R [out_dir]
#   (out_dir defaults to the directory this script lives in, + "/out")

suppressPackageStartupMessages({
  library(SNPRelate)
  library(GWASTools)
  library(GENESIS)
})

args <- commandArgs(trailingOnly = TRUE)
script_dir <- tryCatch({
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) == 1) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
}, error = function(e) getwd())

out_dir <- if (length(args) >= 1) args[[1]] else file.path(script_dir, "out")
cat(sprintf("[run_genesis] out_dir = %s\n", out_dir))

bed_fn <- file.path(out_dir, "fixture.bed")
bim_fn <- file.path(out_dir, "fixture.bim")
fam_fn <- file.path(out_dir, "fixture.fam")
gds_fn <- file.path(out_dir, "fixture.gds")
pruned_ids_fn <- file.path(out_dir, "pruned_snp_ids.txt")

for (f in c(bed_fn, bim_fn, fam_fn, pruned_ids_fn)) {
  if (!file.exists(f)) {
    stop(sprintf(
      "[run_genesis] ABORT: required input not found: %s\n  Run export_fixture.py first.",
      f
    ))
  }
}

# --- Step 1: BED -> GDS ------------------------------------------------------
if (file.exists(gds_fn)) file.remove(gds_fn)
cat("[run_genesis] snpgdsBED2GDS...\n")
snpgdsBED2GDS(bed.fn = bed_fn, fam.fn = fam_fn, bim.fn = bim_fn,
              out.gdsfn = gds_fn, verbose = TRUE)

genofile <- snpgdsOpen(gds_fn)
sample_id <- read.gdsn(index.gdsn(genofile, "sample.id"))
snp_id    <- read.gdsn(index.gdsn(genofile, "snp.id"))
n <- length(sample_id)
cat(sprintf("[run_genesis] GDS opened: n=%d samples, m=%d SNPs\n", n, length(snp_id)))

# Character SNP names ("snp0000", ...) as written by export_fixture.py's
# .bim column 2, read from the SAME still-open genofile handle (avoids a
# second, separately-managed GDS connection later in this script). Most
# SNPRelate builds copy the .bim column-2 string into the "snp.rs.id" gdsn
# node during snpgdsBED2GDS(); if a future SNPRelate version renames this
# node, the explicit existence check below aborts with a clear message
# instead of silently mis-mapping the pruned marker subset.
if (!("snp.rs.id" %in% ls.gdsn(genofile))) {
  stop("[run_genesis] ABORT: GDS node 'snp.rs.id' not found -- installed ",
       "SNPRelate version may store the character SNP name under a ",
       "different node; run `ls.gdsn(genofile)` to inspect and update this ",
       "script's snp_id_char lookup.")
}
snp_id_char <- read.gdsn(index.gdsn(genofile, "snp.rs.id"))

# --- Step 2: KING-robust on the FULL (unpruned) marker panel ----------------
cat("[run_genesis] snpgdsIBDKING(type='KING-robust')...\n")
king <- snpgdsIBDKING(genofile, type = "KING-robust", verbose = TRUE)

# snpgdsIBDKING returns a list; the kinship matrix field is named `kinship`
# in current SNPRelate releases. Guard against a field-name change so this
# script fails loudly (not silently on a NULL matrix) if the installed
# SNPRelate version differs.
if (is.null(king$kinship)) {
  stop("[run_genesis] ABORT: snpgdsIBDKING() result has no $kinship field -- ",
       "installed SNPRelate version may use a different field name; inspect ",
       "names(king) and update this script.")
}
king_mat <- king$kinship
king_sample_id <- king$sample.id
rownames(king_mat) <- king_sample_id
colnames(king_mat) <- king_sample_id
# Explicit reorder to the GDS sample.id order == fixture.fam row order ==
# G.npy row order (export_fixture.py writes sample IDs "IND0000".."IND{n-1}"
# in that exact sequence to all three of fixture.fam, G.npy, and meta.npz).
king_mat <- king_mat[sample_id, sample_id]

write.table(
  data.frame(sample_id = sample_id, king_mat, check.names = FALSE),
  file = file.path(out_dir, "genesis_king_kinship.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)
cat(sprintf("[run_genesis] wrote %s\n", file.path(out_dir, "genesis_king_kinship.tsv")))

snpgdsClose(genofile)

# --- Step 3: PC-AiR on the LD-pruned marker subset --------------------------
# GWASTools wrapper GENESIS expects: a GdsGenotypeReader + GenotypeData.
gds_reader <- GdsGenotypeReader(gds_fn)
geno_data  <- GenotypeData(gds_reader)

pruned_snp_ids <- readLines(pruned_ids_fn)
cat(sprintf("[run_genesis] restricting PC-AiR/PC-Relate to %d / %d pruned SNPs\n",
            length(pruned_snp_ids), length(snp_id)))
# snp_id (numeric GDS IDs) and snp_id_char (character "snp0000", ... names)
# were both read above from the same genofile handle before it was closed,
# so they are already aligned position-for-position; map the pruned
# character names back to their numeric GDS snp.id for snp.include=.
pruned_snp_gds_id <- snp_id[match(pruned_snp_ids, snp_id_char)]
if (any(is.na(pruned_snp_gds_id))) {
  stop("[run_genesis] ABORT: could not map some pruned_snp_ids.txt entries ",
       "back to GDS snp.id -- inspect the GDS node layout (ls.gdsn) and fix ",
       "the snp_id_char lookup above.")
}

cat("[run_genesis] GENESIS::pcair() (defaults; kinobj=divobj=KING-robust matrix)...\n")
pcair_out <- pcair(
  geno_data,
  kinobj = king_mat,
  divobj = king_mat,
  snp.include = pruned_snp_gds_id
)

pcs_mat <- pcair_out$vectors
# Reorder PC-AiR's output rows to the canonical sample_id order.
pcair_sample_id <- rownames(pcs_mat)
if (is.null(pcair_sample_id)) pcair_sample_id <- pcair_out$sample.id
ord <- match(sample_id, pcair_sample_id)
pcs_mat <- pcs_mat[ord, , drop = FALSE]
colnames(pcs_mat) <- paste0("PC", seq_len(ncol(pcs_mat)))

write.table(
  data.frame(sample_id = sample_id, pcs_mat, check.names = FALSE),
  file = file.path(out_dir, "genesis_pcair_pcs.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)
cat(sprintf("[run_genesis] wrote %s (n_pcs=%d)\n",
            file.path(out_dir, "genesis_pcair_pcs.tsv"), ncol(pcs_mat)))

# --- Step 4: PC-Relate on the same pruned marker subset ---------------------
cat("[run_genesis] GENESIS::pcrelate() using pcair PCs (first 3)...\n")
n_pcs_adjust <- min(3, ncol(pcs_mat))
geno_iterator <- GenotypeBlockIterator(geno_data, snpBlock = 10000,
                                       snpInclude = pruned_snp_gds_id)
pcrel <- pcrelate(
  geno_iterator,
  pcs = pcs_mat[, seq_len(n_pcs_adjust), drop = FALSE],
  training.set = pcair_out$unrels
)

# pcrelate() returns $kinBtwn (long: ID1, ID2, kin, ...) and $kinSelf
# (self/inbreeding). We assemble the (n, n) symmetric matrix explicitly;
# the diagonal is fixed at 0.5 to match torchgenomics' pc_relate() diagonal
# convention (both defer the exact GENESIS self-kinship/inbreeding
# estimator; see kinship_admixed.py's pc_relate docstring).
kin_mat <- matrix(0.0, n, n, dimnames = list(sample_id, sample_id))
diag(kin_mat) <- 0.5
kb <- pcrel$kinBtwn
for (row_i in seq_len(nrow(kb))) {
  id1 <- as.character(kb$ID1[row_i])
  id2 <- as.character(kb$ID2[row_i])
  kin_mat[id1, id2] <- kb$kin[row_i]
  kin_mat[id2, id1] <- kb$kin[row_i]
}

write.table(
  data.frame(sample_id = sample_id, kin_mat, check.names = FALSE),
  file = file.path(out_dir, "genesis_pcrelate_kinship.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)
cat(sprintf("[run_genesis] wrote %s\n",
            file.path(out_dir, "genesis_pcrelate_kinship.tsv")))

cat("[run_genesis] done. Golden outputs in:\n")
cat(sprintf("  %s\n", file.path(out_dir, "genesis_king_kinship.tsv")))
cat(sprintf("  %s\n", file.path(out_dir, "genesis_pcair_pcs.tsv")))
cat(sprintf("  %s\n", file.path(out_dir, "genesis_pcrelate_kinship.tsv")))
