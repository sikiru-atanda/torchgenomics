"""Benchmark 3: chromosome-spanning multi-locus wall-time profile.

Partition all surviving MDP SNPs (~2953 after monomorphic drop) into
non-overlapping p=200 windows. Run TorchGenomics bayes-scan-rss on each window;
record per-locus wall-time, peak β_sd, and convergence. Compare aggregate
runtime against the per-locus susieR equivalent on a sub-sampled set of
loci (full susieR sweep would take many minutes; we sample 5 loci for
head-to-head and extrapolate).

Outputs:
  outputs/multilocus_sweep.tsv: per-locus timing + convergence
  outputs/multilocus_summary.txt: aggregate + susieR head-to-head sample
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from torchgenomics.models.bayesian_vs_rss import BayesianVSRss

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MDP_DIR = Path.home() / "Documents" / "GWAS_Expert" / "benchmark" / "data"

WINDOW_SIZE = 200
SUSIE_R_SAMPLE_LOCI = [0, 4, 8, 12]  # subsample for the head-to-head check

# Reload the full MDP sumstats (replicate _build_fixture_mdp.py preprocessing)
geno_df = pd.read_csv(MDP_DIR / "mdp_numeric.txt", sep="\t")
snp_cols = [c for c in geno_df.columns if c != "taxa"]
snp_info = pd.read_csv(MDP_DIR / "mdp_SNP_information.txt", sep="\t")
snp_info = snp_info.rename(columns={"SNP": "snp"}).set_index("snp")
traits = pd.read_csv(MDP_DIR / "mdp_traits.txt", sep="\t")
traits = traits.rename(columns={"Taxa": "taxa"})
traits["taxa"] = traits["taxa"].astype(str)
geno_df["taxa"] = geno_df["taxa"].astype(str)
merged = geno_df.merge(traits, on="taxa", how="inner")
pheno = merged["EarHT"].replace("NaN", np.nan).astype(float).values
keep = ~np.isnan(pheno)
y = pheno[keep] - pheno[keep].mean()
G = merged[snp_cols].values[keep].astype(np.float64)
n = G.shape[0]
col_means = np.nanmean(G, axis=0)
inds = np.where(np.isnan(G))
G[inds] = col_means[inds[1]]
col_var = G.var(axis=0, ddof=1)
nonzero = col_var > 1e-12
G = G[:, nonzero]
G = (G - G.mean(axis=0)) / G.std(axis=0, ddof=1)
beta_hat = (G.T @ y) / (n - 1)
se = np.sqrt(np.var(y, ddof=1) / (n - 1))
z_full = beta_hat / se
p_total = G.shape[1]

n_loci = p_total // WINDOW_SIZE
print(f"MDP genome sweep: n={n}, total p={p_total}, {n_loci} loci of {WINDOW_SIZE} SNPs each")

# Run TorchGenomics on every locus
model = BayesianVSRss(max_num_causal=10, coverage=0.95, purity=0.5)
results = []
total_t = 0.0
for li in range(n_loci):
    start = li * WINDOW_SIZE
    stop = start + WINDOW_SIZE
    z_loc = torch.from_numpy(z_full[start:stop]).to(torch.float64)
    G_loc = G[:, start:stop]
    R_loc = torch.from_numpy((G_loc.T @ G_loc) / (n - 1)).to(torch.float64)
    t0 = time.time()
    res = model.fit_rss(z=z_loc, R=R_loc, n=n)
    dt = time.time() - t0
    total_t += dt
    max_pip = float(res.pip.max())
    n_cs = len(res.credible_sets)
    results.append({
        "locus": li,
        "start": start,
        "stop": stop,
        "max_abs_z": float(np.abs(z_full[start:stop]).max()),
        "max_pip": max_pip,
        "n_credible_sets": n_cs,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "wall_time_s": dt,
    })

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "multilocus_sweep.tsv", sep="\t", index=False)

print(f"\nTorchGenomics multi-locus sweep: {total_t:.2f}s total")
print(f"  per-locus: mean {df['wall_time_s'].mean()*1000:.1f}ms,"
      f" median {df['wall_time_s'].median()*1000:.1f}ms,"
      f" max {df['wall_time_s'].max()*1000:.1f}ms")
print(f"  loci with PIP > 0.5: {(df['max_pip'] > 0.5).sum()}")
print(f"  loci with credible sets: {(df['n_credible_sets'] > 0).sum()}")
print(f"  all converged: {df['converged'].all()}")
print(f"  max_iter used: {df['n_iter'].max()}")

# Sample susieR head-to-head on 4 loci to confirm parity holds across the chromosome
print(f"\n--- susieR head-to-head on {len(SUSIE_R_SAMPLE_LOCI)} sample loci ---")
parity = []
susieR_total_t = 0.0
import subprocess
import tempfile
for li in SUSIE_R_SAMPLE_LOCI:
    if li >= n_loci:
        continue
    start = li * WINDOW_SIZE
    stop = start + WINDOW_SIZE
    z_loc = z_full[start:stop]
    G_loc = G[:, start:stop]
    R_loc = (G_loc.T @ G_loc) / (n - 1)
    np.savetxt(OUT_DIR / "_z.tsv", z_loc, delimiter="\t")
    np.savetxt(OUT_DIR / "_R.tsv", R_loc, delimiter="\t")
    # Tiny inline R script
    rscript = f"""
suppressMessages(library(susieR))
z <- as.numeric(readLines("{OUT_DIR}/_z.tsv"))
R <- as.matrix(read.table("{OUT_DIR}/_R.tsv"))
t0 <- Sys.time()
fit <- susie_rss(z=z, R=R, n={n}, L=10, coverage=0.95, min_abs_corr=0.5)
dt <- as.numeric(Sys.time() - t0, units="secs")
sd_l <- sqrt(colSums(fit$alpha * fit$mu2 - (fit$alpha * fit$mu)^2))
m_l  <- colSums(fit$alpha * fit$mu)
write.table(data.frame(idx=seq_along(fit$pip), PIP=fit$pip, BMEAN=m_l, BSD=sd_l),
            "{OUT_DIR}/_su.tsv", sep="\t", row.names=FALSE, quote=FALSE)
cat(dt, file="{OUT_DIR}/_dt.txt")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
        f.write(rscript)
        rpath = f.name
    subprocess.run(["Rscript", rpath], capture_output=True, check=True)
    su = pd.read_csv(OUT_DIR / "_su.tsv", sep="\t")
    su_dt = float(open(OUT_DIR / "_dt.txt").read().strip())
    susieR_total_t += su_dt
    # Re-fit ours with timing
    z_t = torch.from_numpy(z_loc).to(torch.float64)
    R_t = torch.from_numpy(R_loc).to(torch.float64)
    t0 = time.time()
    res = model.fit_rss(z=z_t, R=R_t, n=n)
    our_dt = time.time() - t0
    pip_corr = float(pearsonr(su["PIP"].values, res.pip.numpy())[0])
    if np.isnan(pip_corr):
        pip_corr = (
            1.0 if np.abs(su["PIP"].values - res.pip.numpy()).max() < 1e-4 else 0.0
        )
    bm_corr = float(pearsonr(su["BMEAN"].values, res.beta_mean.numpy())[0])
    bs_corr = float(pearsonr(su["BSD"].values, res.beta_sd.numpy())[0])
    parity.append((li, pip_corr, bm_corr, bs_corr, su_dt, our_dt))
    print(f"  locus {li:>2d} (start={start:>4d}, max|z|={np.abs(z_loc).max():.2f}): "
          f"PIP r={pip_corr:.4f}, β_mean r={bm_corr:.4f}, β_sd r={bs_corr:.4f}, "
          f"susieR={su_dt*1000:.0f}ms, ours={our_dt*1000:.0f}ms ({our_dt/su_dt:.2f}×)")

# Aggregate
print(f"\n=== Aggregate genome-sweep (extrapolated) ===")
print(f"  TorchGenomics full {n_loci}-locus sweep: {total_t:.2f}s ({total_t/n_loci*1000:.1f}ms/locus avg)")
mean_su = sum(p[4] for p in parity) / len(parity)
mean_ours = sum(p[5] for p in parity) / len(parity)
print(f"  susieR per-locus avg ({len(parity)} loci sampled): {mean_su*1000:.1f}ms")
print(f"  TorchGenomics per-locus avg (same {len(parity)} loci): {mean_ours*1000:.1f}ms")
print(f"  Per-locus ratio: {mean_ours/mean_su:.2f}× susieR")
print(f"  Extrapolated full susieR sweep: {mean_su*n_loci:.2f}s")
print(f"  Extrapolated full TorchGenomics sweep: {mean_ours*n_loci:.2f}s")
