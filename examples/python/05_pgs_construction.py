"""Polygenic Score construction: C+T and LDpred2-Inf on toy sumstats.

Demonstrates: build an LD reference, fit C+T and LDpred2-Inf weights from the
same summary statistics, and score a held-out target sample.

Phases covered: 40 (PGS construction).

Runtime: < 20 s on CPU (m=30 SNPs, n_discovery=3000, n_target=500).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgwas.config import STAT_DTYPE
from torchgwas.pgs import (
    ClumpingThresholding,
    LDpred2Inf,
    build_ld_reference,
    score_individuals,
)
from torchgwas.postgwas._sumstats import SumStats

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_discovery(m=30, n=3000, h2=0.4, seed=1):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=STAT_DTYPE)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)

    beta_true = torch.zeros(m, dtype=STAT_DTYPE)
    n_causal = 5
    idx = torch.randperm(m, generator=g)[:n_causal]
    beta_true[idx] = torch.randn(n_causal, generator=g, dtype=STAT_DTYPE) * math.sqrt(h2 / n_causal)

    y = G @ beta_true + torch.randn(n, generator=g, dtype=STAT_DTYPE) * math.sqrt(1.0 - h2)
    y = (y - y.mean()) / y.std().clamp(min=1e-12)

    # per-SNP univariate OLS
    beta_hat = (G * y.unsqueeze(1)).sum(0) / n
    res = ((y.unsqueeze(1) - G * beta_hat.unsqueeze(0)) ** 2).mean(0)
    se = torch.sqrt(res / n)
    z = beta_hat / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(z.abs() / math.sqrt(2.0))))

    meta = {
        "snp": [f"rs{i}" for i in range(m)],
        "chr": ["1"] * m,
        "pos": list(range(100, 100 + m)),
        "a1": ["A"] * m,
        "a2": ["G"] * m,
    }
    ss = SumStats(
        chr=meta["chr"], pos=meta["pos"], snp=meta["snp"],
        a1=meta["a1"], a2=meta["a2"],
        beta=beta_hat, se=se, p=p,
        n=torch.full((m,), float(n)),
    )
    return ss, G, beta_true, meta


def main() -> None:
    ss, G_disc, beta_true, meta = simulate_discovery()
    print(f"Simulated discovery: {G_disc.shape[0]} samples, {G_disc.shape[1]} SNPs, "
          f"{int((beta_true != 0).sum())} causal")

    ld = build_ld_reference(G_disc, meta, mode="full")
    print(f"LD reference: {ld.m} SNPs (mode={ld.mode})")

    ct = ClumpingThresholding().fit(ss, ld, p_threshold=5e-3, r2_threshold=0.1)
    print(f"C+T (p<5e-3): {int((ct.weight != 0).sum())} SNPs retained")

    ldp = LDpred2Inf().fit(ss, ld, h2=0.4)
    print(f"LDpred2-Inf: h2={ldp.h2}, ||w||_1={float(ldp.weight.abs().sum()):.4f}, "
          f"converged={ldp.converged}")

    # Score a new target cohort
    torch.manual_seed(99)
    G_target = torch.randn(500, ss.m, dtype=STAT_DTYPE)
    G_target = (G_target - G_target.mean(0, keepdim=True)) / G_target.std(0, keepdim=True).clamp(min=1e-12)

    sr_ct = score_individuals(G_target, meta["snp"], meta["a1"], meta["a2"], ct)
    sr_ldp = score_individuals(G_target, meta["snp"], meta["a1"], meta["a2"], ldp)
    print(f"C+T scores   : mean={float(sr_ct.pgs.mean()):.4f}  std={float(sr_ct.pgs.std()):.4f}")
    print(f"LDpred2 scores: mean={float(sr_ldp.pgs.mean()):.4f}  std={float(sr_ldp.pgs.std()):.4f}")

    df = pd.DataFrame({
        "snp": meta["snp"],
        "beta_true": beta_true.cpu().numpy(),
        "weight_ct": ct.weight.cpu().numpy(),
        "weight_ldpred2": ldp.weight.cpu().numpy(),
    })
    out = OUT_DIR / "05_pgs_weights.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
