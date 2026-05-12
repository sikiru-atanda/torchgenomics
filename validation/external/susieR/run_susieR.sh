#!/usr/bin/env bash
# Run susieR::susie_rss() on the MDP-derived per-locus fixture.
# Per NA1 design spec section 5.2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

if [ ! -f "${DATA_DIR}/locus_z.pt" ]; then
    echo "FATAL: fixture not provisioned. Run ./fetch_data.sh first."
    exit 2
fi

# Convert torch tensors to TSV for R consumption (since R doesn't read .pt)
python3 - <<PYEOF
import torch
import numpy as np
import os

z = torch.load("${DATA_DIR}/locus_z.pt", weights_only=True).numpy()
R = torch.load("${DATA_DIR}/locus_R.pt", weights_only=True).numpy()
np.savetxt("${DATA_DIR}/locus_z.tsv", z, delimiter="\t")
np.savetxt("${DATA_DIR}/locus_R.tsv", R, delimiter="\t")
print(f"Exported z (p={len(z)}) and R ({R.shape[0]}x{R.shape[1]}) for R")
PYEOF

# Run susieR::susie_rss
START=$(date +%s.%N)
/usr/bin/time -v R --vanilla --no-save <<RSCRIPT 2>"${OUT_DIR}/susieR_time.log"
suppressMessages(library(susieR))
z <- as.numeric(scan("${DATA_DIR}/locus_z.tsv"))
R <- as.matrix(read.table("${DATA_DIR}/locus_R.tsv", sep="\t", header=FALSE))
n <- as.integer(readLines("${DATA_DIR}/locus_n.txt")[1])
fit <- susie_rss(z=z, R=R, n=n, L=10, coverage=0.95)

# Extract PIP, beta_mean, beta_sd, credible sets, ELBO
# susieR helpers return one scalar per variant (length p), not an L-column
# matrix. coef() includes an intercept at position 1 that we drop.
pip <- as.numeric(fit\$pip)
beta_post <- as.numeric(coef(fit)[-1])
beta_sd_per_variant <- as.numeric(susie_get_posterior_sd(fit))
elbo <- as.numeric(fit\$elbo[length(fit\$elbo)])
writeLines(as.character(elbo), "${OUT_DIR}/susieR_elbo.txt")

# Credible sets
cs <- fit\$sets\$cs
cs_indices <- if (is.null(cs)) integer(0) else unlist(cs)

# Write output TSV matching our schema
out <- data.frame(
    SNP_idx = seq_along(pip),
    PIP = pip,
    BETA_MEAN = beta_post,
    BETA_SD = beta_sd_per_variant,
    IN_CS = ifelse(seq_along(pip) %in% cs_indices, 1, 0),
    ELBO_FINAL = elbo
)
write.table(out, "${OUT_DIR}/susieR.tsv", sep="\t", row.names=FALSE, quote=FALSE)
RSCRIPT
END=$(date +%s.%N)
echo "$END $START" | awk '{print $1 - $2}' > "${OUT_DIR}/susieR_walltime_seconds.txt"
echo "susieR run complete: ${OUT_DIR}/susieR.tsv"
