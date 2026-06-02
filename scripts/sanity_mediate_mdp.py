"""Real-data sanity check for `torchgenomics.multiomics` on MDP maize (GAPIT demo).

Dataset: MDP (Zhao et al. 2011 maize panel) — 281 samples, 3093 SNPs, 3 traits
(EarHT, dpoll, EarDia). Publicly distributed with GAPIT.

Sanity goals:
- Y=EarHT (ear height), M=dpoll (days to pollination) — plausible mediator
- SNPs: top 5 by marginal LMM p-value on Y
- Expect: finite coefficients, p-values in [0, 1], |rho*| <= 1,
  proportion_mediated finite-or-NaN, mkernel_h2 gives h2 in [0, 1]
- Full scan over the 5 SNPs vs. 2 features (dpoll, EarDia) with cis_window=None
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.single_trait_lmm import SingleTraitLMM
from torchgenomics.multiomics import (
    build_expression_kernel,
    mediate_lmm,
    mkernel_h2,
    scan_mediation,
)

DATA = "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/data"

# ---- Load --------------------------------------------------------------------
geno = pd.read_csv(f"{DATA}/mdp_numeric.txt", sep=r"\s+")
traits = pd.read_csv(f"{DATA}/mdp_traits.txt", sep=r"\s+")

geno = geno.rename(columns={"taxa": "Taxa"})
merged = traits.merge(geno, on="Taxa", how="inner").dropna(
    subset=["EarHT", "dpoll", "EarDia"]
)
print(f"n samples w/ complete phenotypes = {len(merged)}")

Y = torch.as_tensor(merged["EarHT"].to_numpy(), dtype=torch.float64)
M_dpoll = torch.as_tensor(merged["dpoll"].to_numpy(), dtype=torch.float64)
M_EarDia = torch.as_tensor(merged["EarDia"].to_numpy(), dtype=torch.float64)
G_cols = [c for c in merged.columns if c not in ("Taxa", "EarHT", "dpoll", "EarDia")]
G = torch.as_tensor(merged[G_cols].to_numpy().astype(np.float64), dtype=torch.float64)
print(f"G shape = {tuple(G.shape)}")

# ---- GRM ---------------------------------------------------------------------
K, _ = grm_vanraden(G, ploidy=2)
print(f"K shape = {tuple(K.shape)}  diag mean = {K.diagonal().mean().item():.3f}")

# ---- Pick top-5 SNPs by marginal LMM on Y ------------------------------------
model = SingleTraitLMM()
intercept = torch.ones(Y.shape[0], 1, dtype=torch.float64)
nf = model.fit_null(Y, intercept, K)
# quick per-SNP Wald
from torchgenomics.linalg.eigh import rotate
U = nf.eigenvectors
w = 1.0 / (nf.eigenvalues * nf.sig2_g + nf.sig2_e).clamp(min=1e-20)
Y_r = nf.Y_rot.squeeze()
X0_r = nf.X0_rot
pvals = np.ones(G.shape[1])
for j in range(G.shape[1]):
    x = rotate(G[:, j], U).squeeze()
    Xj = torch.cat([x.unsqueeze(1), X0_r], dim=1)
    wXj = w.unsqueeze(1) * Xj
    XtWX = Xj.T @ wXj
    XtWy = wXj.T @ Y_r
    try:
        beta = torch.linalg.solve(XtWX, XtWy)
        cov = torch.linalg.inv(XtWX)
        z = beta[0] / cov[0, 0].sqrt().clamp(min=1e-20)
        pvals[j] = math.erfc(abs(float(z)) / math.sqrt(2))
    except Exception:
        pvals[j] = 1.0
top5 = np.argsort(pvals)[:5]
print(f"\nTop-5 SNPs on EarHT:")
for k, j in enumerate(top5):
    print(f"  {k+1}. {G_cols[j]:<14s}  p = {pvals[j]:.3e}")

# ---- Single-triple mediation on the top hit ----------------------------------
print("\n=== mediate_lmm: Y=EarHT, M=dpoll, SNP=top-1 hit ===")
res = mediate_lmm(Y, G[:, top5[0]], M_dpoll, K, se="monte-carlo",
                  n_mc_draws=10_000, seed=42)
for field in ("a", "a_se", "b", "b_se", "c", "c_prime", "indirect",
              "indirect_se", "indirect_ci_lower", "indirect_ci_upper",
              "indirect_pvalue", "proportion_mediated", "inconsistent",
              "sensitivity_rho"):
    v = getattr(res, field)
    print(f"  {field:<25s} = {v}")

# Sanity assertions
assert math.isfinite(res.a) and math.isfinite(res.b)
assert 0.0 <= res.indirect_pvalue <= 1.0
assert res.sensitivity_rho is None or -1.0 <= res.sensitivity_rho <= 1.0
assert abs(res.c - (res.c_prime + res.indirect)) < 1e-4, \
    f"c = c' + ab identity violated: c={res.c}, c'+ab={res.c_prime + res.indirect}"
print("  [OK] all sanity checks pass")

# ---- Sobel vs MC agreement on the same triple --------------------------------
print("\n=== Sobel vs Monte-Carlo agreement ===")
res_s = mediate_lmm(Y, G[:, top5[0]], M_dpoll, K, se="sobel")
print(f"  Sobel      : indirect = {res_s.indirect:.4g}, SE = {res_s.indirect_se:.4g}, "
      f"p = {res_s.indirect_pvalue:.3e}")
print(f"  Monte-Carlo: indirect = {res.indirect:.4g}, SE = {res.indirect_se:.4g}, "
      f"p = {res.indirect_pvalue:.3e}")
# Indirect estimates must match exactly (point estimates are ML not sampled).
assert abs(res_s.indirect - res.indirect) < 1e-12

# ---- scan over top-5 SNPs x 2 mediators --------------------------------------
print("\n=== scan_mediation: top-5 SNPs x 2 features, no cis filter ===")
G_top = G[:, top5]
M_mat = torch.stack([M_dpoll, M_EarDia], dim=1)
scan = scan_mediation(
    Y, G_top, M_mat, K,
    snp_ids=[G_cols[int(j)] for j in top5],
    feature_ids=["dpoll", "EarDia"],
    cis_window_bp=None,
    se="monte-carlo", n_mc_draws=10_000,
    fdr_method="bh", sensitivity=True, seed=42,
)
df = scan.to_dataframe().sort_values("indirect_pvalue").reset_index(drop=True)
print(df[["snp", "feature", "a", "b", "indirect", "indirect_pvalue",
          "q_indirect", "sensitivity_rho"]].to_string(index=False))
assert (df["indirect_pvalue"].between(0, 1)).all()
assert (df["q_indirect"].between(0, 1)).all()
print("  [OK] scan produced finite p and q in [0,1]")

# ---- mkernel_h2: SNP GRM + expression-like context kernel --------------------
print("\n=== mkernel_h2: SNP GRM + context kernel from (dpoll, EarDia) ===")
K_ctx = build_expression_kernel(M_mat, method="linear")
h2 = mkernel_h2(Y, {"snp": K, "context": K_ctx})
print(f"  sigma2 = {h2.sigma2}")
print(f"  h2     = {h2.h2}")
print(f"  h2_total = {h2.h2_total:.4f}")
assert 0.0 <= h2.h2_total <= 1.0 + 1e-6
for name, v in h2.h2.items():
    assert -1e-6 <= v <= 1.0 + 1e-6, f"h2[{name}] = {v} out of range"
print("  [OK] variance components in [0,1]")

print("\nAll sanity checks passed on MDP maize real data.")
