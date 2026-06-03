"""Build a LARGER synthetic per-locus (z, R, n) triple for stress-testing.

5x scale-up vs the canonical fixture: n=2000 samples, p=1000 variants,
5 planted causals, AR(1) rho=0.6 (stronger LD than 0.5).

Tests:
- Algorithm behavior at biobank-medium scale (1000 variants)
- V-update with multiple causal signals (5 instead of 3)
- Stronger LD structure (ρ=0.6 vs 0.5)
- Wall-time at 5x problem size

Output overwrites the canonical fixture files in DATA_DIR so the
existing run_susieR.sh / run_torchgenomics.sh / compare.py harness picks
them up unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rng = np.random.default_rng(123)  # different seed than canonical fixture

    n = 2000
    p = 1000
    h2 = 0.30
    causal_idx = [127, 348, 502, 716, 871]  # 5 planted causals across the locus
    causal_effects = np.array([0.6, -0.5, 0.4, 0.5, -0.4])

    rho = 0.6  # stronger LD than the canonical fixture
    raw = rng.standard_normal((n, p)).astype(np.float64)
    G = np.zeros_like(raw)
    G[:, 0] = raw[:, 0]
    for j in range(1, p):
        G[:, j] = rho * G[:, j - 1] + math.sqrt(1.0 - rho ** 2) * raw[:, j]
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True, ddof=1)

    beta = np.zeros(p, dtype=np.float64)
    beta[causal_idx] = causal_effects

    g = G @ beta
    g_var = float(np.var(g, ddof=1))
    e_var = g_var * (1.0 - h2) / h2
    y = g + rng.normal(0.0, math.sqrt(e_var), size=n)
    y = y - y.mean()

    beta_hat = (G.T @ y) / (n - 1)
    se_marginal = np.sqrt(np.var(y, ddof=1) / (n - 1))
    z_np = beta_hat / se_marginal
    R_np = (G.T @ G) / (n - 1)

    z = torch.from_numpy(z_np).to(torch.float64)
    R = torch.from_numpy(R_np).to(torch.float64)
    torch.save(z, DATA_DIR / "locus_z.pt")
    torch.save(R, DATA_DIR / "locus_R.pt")
    (DATA_DIR / "locus_n.txt").write_text(f"{n}\n")

    snp_meta = {
        "snp_ids": [f"rs{i:05d}" for i in range(p)],
        "chr": ["1"] * p,
        "bp": list(range(1000, 1000 + 1000 * p, 1000)),
        "a1": ["A"] * p,
        "a2": ["G"] * p,
    }
    torch.save(snp_meta, DATA_DIR / "locus_meta.pt")

    np.savetxt(DATA_DIR / "locus_z.tsv", z_np, delimiter="\t")
    np.savetxt(DATA_DIR / "locus_R.tsv", R_np, delimiter="\t")

    print(f"LARGE fixture: n={n}, p={p}, h2={h2}, rho={rho}")
    print(f"Causal indices: {causal_idx}")
    print(f"|z| at causals: {np.round(np.abs(z_np[causal_idx]), 2)}")
    print(f"|z| max non-causal: {np.abs(np.delete(z_np, causal_idx)).max():.3f}")
    print(f"R off-diagonal max: {(R_np - np.eye(p)).max():.3f}")
    print(f"R total size: {R_np.nbytes / 1e6:.1f} MB")
    print(f"Saved to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
