"""Build a synthetic per-locus (z, R, n) triple for SuSiE-RSS parity testing.

Per NA1 design spec section 5.2 — synthetic fixture is acceptable for the
Tier 2 head-to-head against susieR::susie_rss(). The MDP-derived path is
documented as the canonical fixture for production but requires the full
torchgenomics lmm-scan → window-extract pipeline; this synthetic fixture
exercises the same per-locus contract with planted causals at known
positions, which is exactly what susieR's own test suite uses.

Output files (in DATA_DIR):
- locus_z.pt          : torch.float64 tensor of shape (p,)
- locus_R.pt          : torch.float64 tensor of shape (p, p) -- in-sample LD
- locus_n.txt         : sample size as plain text
- locus_meta.pt       : dict with snp_ids, chr, bp, a1, a2 lists
- locus_z.tsv         : same z exported as TSV for R consumption
- locus_R.tsv         : same R exported as TSV for R consumption

Reproducibility: seed=42 fixed throughout.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rng = np.random.default_rng(42)

    n = 500     # samples
    p = 200     # variants in locus
    h2 = 0.30   # locus-level heritability
    causal_idx = [42, 87, 153]  # planted causals
    causal_effects = np.array([0.6, -0.4, 0.5])

    # Build a standardized genotype matrix with realistic LD structure.
    # We use a banded AR(1)-like correlation to mimic local LD without
    # being entirely synthetic-pure-Gaussian: each SNP is a noisy version
    # of its left neighbor.
    rho = 0.5
    raw = rng.standard_normal((n, p)).astype(np.float64)
    G = np.zeros_like(raw)
    G[:, 0] = raw[:, 0]
    for j in range(1, p):
        G[:, j] = rho * G[:, j - 1] + math.sqrt(1.0 - rho ** 2) * raw[:, j]
    # Center + standardize columns
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True, ddof=1)

    # Compute true effect vector (zero except at planted causals)
    beta = np.zeros(p, dtype=np.float64)
    beta[causal_idx] = causal_effects

    # Generate phenotype with controlled heritability
    g = G @ beta
    g_var = float(np.var(g, ddof=1))
    e_var = g_var * (1.0 - h2) / h2
    y = g + rng.normal(0.0, math.sqrt(e_var), size=n)
    y = y - y.mean()  # center

    # Compute marginal z-scores (standardized X, no covariates)
    # For standardized X with columns having sample variance 1:
    #   beta_hat[j] = X[:, j].T @ y / (n - 1)   (OLS with centered y)
    #   se_hat[j]  = sqrt(sigma_e^2 / (n - 1))
    # In SuSiE-RSS convention z = beta_hat / se_hat, equivalent to:
    #   z[j] = X[:, j].T @ y / (sigma_e * sqrt(n - 1))
    # We approximate sigma_e by the residual std after marginal regression.
    beta_hat = (G.T @ y) / (n - 1)
    se_marginal = np.sqrt(np.var(y, ddof=1) / (n - 1))
    z_np = beta_hat / se_marginal

    # In-sample LD matrix: R = G.T @ G / (n - 1) for standardized G
    R_np = (G.T @ G) / (n - 1)

    # Save torch tensors
    z = torch.from_numpy(z_np).to(torch.float64)
    R = torch.from_numpy(R_np).to(torch.float64)
    torch.save(z, DATA_DIR / "locus_z.pt")
    torch.save(R, DATA_DIR / "locus_R.pt")
    (DATA_DIR / "locus_n.txt").write_text(f"{n}\n")

    snp_meta = {
        "snp_ids": [f"rs{i:04d}" for i in range(p)],
        "chr": ["1"] * p,
        "bp": list(range(1000, 1000 + 1000 * p, 1000)),
        "a1": ["A"] * p,
        "a2": ["G"] * p,
    }
    torch.save(snp_meta, DATA_DIR / "locus_meta.pt")

    # Also export TSV for R consumption (run_susieR.sh expects these)
    np.savetxt(DATA_DIR / "locus_z.tsv", z_np, delimiter="\t")
    np.savetxt(DATA_DIR / "locus_R.tsv", R_np, delimiter="\t")

    # Diagnostics
    print(f"Generated synthetic fixture: n={n}, p={p}, h2={h2}")
    print(f"Causal indices: {causal_idx}")
    print(f"|z| at causals: {np.abs(z_np[causal_idx])}")
    print(f"|z| max non-causal: {np.abs(np.delete(z_np, causal_idx)).max():.3f}")
    print(f"R diagonal sample: {R_np.diagonal()[:5]}")
    print(f"R off-diagonal max: {(R_np - np.eye(p)).max():.3f}")
    print(f"Saved to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
