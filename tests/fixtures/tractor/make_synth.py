"""Synthetic admixed+related cohort: PSD-admixture dosages + gene-drop GRM.

Simplified reproduction of the Tan et al. 2026 simulation recipe, sufficient
for unit/equivalence testing (NOT the full biobank sim). Ancestry-specific
allele frequencies differ (Fst), a pedigree induces relatedness (true GRM),
and a subset of variants carry ancestry-specific effects.
"""
from __future__ import annotations
import numpy as np
import torch


def make_synth(n: int = 200, m: int = 50, K: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    np_rng = np.random.RandomState(seed)
    dtype = torch.float64
    # ancestry-specific allele freqs (Balding-Nicols-ish via Beta), Fst=0.1
    fst = 0.1
    base = 0.1 + 0.8 * torch.rand(m, generator=g, dtype=dtype)
    a = base.numpy()
    b = ((1 - base) * (1 - fst) / fst).numpy()
    a_np = (base * (1 - fst) / fst).numpy()
    af = np.stack([np_rng.beta(a_np, b) for _ in range(K)])  # (K, m)
    af = torch.from_numpy(af).to(dtype)
    # global admixture proportion per sample (80/20 for K=2 default), Dirichlet
    alpha_dir = np.array([8.0, 2.0] + [1.0] * (K - 2), dtype=np.float64)
    adm = np_rng.dirichlet(alpha_dir, size=n)  # (n, K)
    adm = torch.from_numpy(adm).to(dtype)
    
    # For each individual, sample 2 ancestry assignments from admixture proportions
    all_ancestry_idx = torch.multinomial(adm, num_samples=2, replacement=True, generator=g)  # (n, 2)
    
    # Count copies from each ancestry for each individual
    ancestry_copies = torch.zeros(n, K, dtype=dtype)
    for i in range(n):
        for k in range(K):
            ancestry_copies[i, k] = (all_ancestry_idx[i] == k).sum().to(dtype)
    
    # Now sample alleles for each ancestry, SNP, and individual
    dosages = torch.zeros(K, n, m, dtype=dtype)
    for k in range(K):
        for j in range(m):
            p = af[k, j]
            for i in range(n):
                n_c = int(ancestry_copies[i, k].item())
                if n_c > 0:
                    sampled = torch.binomial(
                        torch.tensor(float(n_c), dtype=dtype),
                        p,
                        generator=g
                    )
                    dosages[k, i, j] = sampled
    
    # relatedness: build a block pedigree -> true GRM as 2*kinship
    n_fam = max(1, n // 4)
    fam = torch.arange(n) % n_fam
    kinship = (fam.unsqueeze(0) == fam.unsqueeze(1)).to(dtype) * 0.25
    kinship.fill_diagonal_(0.5)
    K_grm = 2.0 * kinship  # (n,n)
    # phenotype: causal on ancestry 0 dosage for a few variants
    causal_idx = list(range(0, m, max(1, m // 5)))[:5]
    beta_true = torch.zeros(K, dtype=dtype)
    beta_true[0] = 0.5
    signal = torch.zeros(n, dtype=dtype)
    for j in causal_idx:
        signal += beta_true[0] * dosages[0, :, j]
    # random effect from GRM + noise
    L = torch.linalg.cholesky(K_grm + 1e-4 * torch.eye(n, dtype=dtype))
    reff = L @ torch.randn(n, generator=g, dtype=dtype)
    X0 = torch.stack([torch.ones(n, dtype=dtype),
                      torch.randn(n, generator=g, dtype=dtype)], dim=1)  # intercept + 1 cov
    y_cont = 1.0 + 0.3 * X0[:, 1] + signal + reff + torch.randn(n, generator=g, dtype=dtype)
    prob = torch.sigmoid(y_cont - y_cont.mean())
    y_bin = torch.bernoulli(prob, generator=g)
    return {"dosages": dosages, "y_cont": y_cont, "y_bin": y_bin, "X0": X0,
            "K_grm": K_grm, "beta_true": beta_true, "causal_idx": causal_idx,
            "adm": adm, "af": af}
