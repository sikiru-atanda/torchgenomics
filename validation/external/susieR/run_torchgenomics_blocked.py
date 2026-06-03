"""Benchmark 1: fit_rss_blocked vs susieR's dense per-locus fit.

Compare our blocked-path SuSiE-RSS against susieR's dense reference on the
LARGE (p=1000) synthetic fixture, using a 4-block partition (250 SNPs each).
This is the path that enables biobank-scale fine-mapping (per-block memory
O(p_block^2) instead of O(p^2)).

Per the docstring caveat in fit_rss_blocked: blocked != dense even on
strictly block-diagonal R because the per-block softmax recalibrates over a
smaller candidate pool. We expect:
  - PIPs at high-signal variants (PIP > 0.5): match within ~1e-2
  - β_mean: match closely (Pearson > 0.99 on actively-fit variants)
  - β_sd: similar caveat
  - Background-noise variants: divergence O(1/p_block) - O(1/p_total)
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
from torchgenomics.postgwas._ld_ref_loader import BlockSpec

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

z = torch.load(DATA_DIR / "locus_z.pt", weights_only=True).to(torch.float64)
R = torch.load(DATA_DIR / "locus_R.pt", weights_only=True).to(torch.float64)
n = int(open(DATA_DIR / "locus_n.txt").read().strip())
p = z.shape[0]

# 4-block partition (250 SNPs each)
N_BLOCKS = 4
block_size = p // N_BLOCKS
blocks = [
    BlockSpec(start=i * block_size, stop=(i + 1) * block_size if i < N_BLOCKS - 1 else p)
    for i in range(N_BLOCKS)
]
print(f"Blocks: {[(b.start, b.stop) for b in blocks]}")

# Run blocked
model = BayesianVSRss(max_num_causal=10, coverage=0.95, purity=0.5)
t0 = time.time()
result_blocked = model.fit_rss_blocked(z=z, R=R, n=n, blocks=blocks)
dt_blocked = time.time() - t0
print(f"\nBlocked fit: {dt_blocked:.3f}s, max_n_iter={result_blocked.n_iter}, "
      f"converged={result_blocked.converged}")

# Compare against susieR reference (already produced by run_susieR.sh on this fixture)
susieR_out = pd.read_csv(OUT_DIR / "susieR.tsv", sep="\t")
susieR_pip = susieR_out["PIP"].values
susieR_beta_mean = susieR_out["BETA_MEAN"].values
susieR_beta_sd = susieR_out["BETA_SD"].values
susieR_in_cs = (susieR_out["IN_CS"].values == 1).astype(int)

# Our blocked outputs
our_pip = result_blocked.pip.numpy()
our_beta_mean = result_blocked.beta_mean.numpy()
our_beta_sd = result_blocked.beta_sd.numpy()
in_our_cs = np.zeros(p, dtype=int)
for layer_idx, members in result_blocked.credible_sets:
    for m in members:
        in_our_cs[m] = 1

# Metrics matched to compare.py contract
hi_up = set(np.where(susieR_pip > 0.5)[0])
hi_ours = set(np.where(our_pip > 0.5)[0])
hi_jaccard = (
    len(hi_up & hi_ours) / max(len(hi_up | hi_ours), 1)
    if (hi_up or hi_ours) else 1.0
)

cs_up = set(np.where(susieR_in_cs == 1)[0])
cs_ours = set(np.where(in_our_cs == 1)[0])
cs_jaccard = (
    len(cs_up & cs_ours) / max(len(cs_up | cs_ours), 1)
    if (cs_up or cs_ours) else 1.0
)

mask = (susieR_pip > 0.1) | (our_pip > 0.1)
if mask.sum() >= 3:
    pip_corr = float(pearsonr(susieR_pip[mask], our_pip[mask])[0])
    if np.isnan(pip_corr):
        pip_corr = 1.0 if np.abs(susieR_pip[mask] - our_pip[mask]).max() < 1e-4 else 0.0
else:
    pip_corr = float("nan")
beta_mean_corr = float(pearsonr(susieR_beta_mean, our_beta_mean)[0])
beta_sd_corr = float(pearsonr(susieR_beta_sd, our_beta_sd)[0])

# susieR wall-time from prior run
wt_susieR = float(open(OUT_DIR / "susieR_walltime_seconds.txt").read().strip())

print(f"\n=== Blocked vs susieR-dense (LARGE p={p}, {N_BLOCKS} blocks) ===")
print(f"  CS Jaccard:                 {cs_jaccard:.4f}  (threshold 0.95)")
print(f"  HiConf PIP Jaccard:         {hi_jaccard:.4f}  (threshold 0.95)")
print(f"  PIP correlation:            {pip_corr:.6f}  (threshold 0.99)")
print(f"  β_mean Pearson:             {beta_mean_corr:.6f}  (threshold 0.999)")
print(f"  β_sd Pearson:               {beta_sd_corr:.6f}  (threshold 0.993)")
print(f"  Wall-time blocked / susieR: {dt_blocked / wt_susieR:.3f}× ({dt_blocked:.3f}s vs {wt_susieR:.3f}s)")

# Also report per-block breakdown
n_active_blocks = (result_blocked.V > 1e-6).any(dim=1).sum().item() if result_blocked.V.ndim == 2 else "N/A (1D V)"
print(f"\n  n active blocks (V > 0 in any layer): {n_active_blocks}")
print(f"  Per-variant β_sd disagreement at causal idx [127,348,502,716,871]:")
for i in [127, 348, 502, 716, 871]:
    print(f"    idx {i}: susieR β_sd={susieR_beta_sd[i]:.5f}  ours={our_beta_sd[i]:.5f}  "
          f"PIP susieR={susieR_pip[i]:.4f}  ours={our_pip[i]:.4f}")
