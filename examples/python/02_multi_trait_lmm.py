"""Multi-trait LMM (mvLMM) on synthetic 2-trait data.

Demonstrates: MultiTraitLMM null fit (REML on Vg/Ve), per-SNP joint-Wald scan.

Phases covered: 5 (MultiTraitLMM).

Runtime: < 10 s on CPU (150 samples x 300 SNPs, d=2 traits).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models import MultiTraitLMM, VariantMeta

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate(n=150, m=300, seed=42):
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    true_Vg = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=STAT_DTYPE)
    true_Ve = torch.tensor([[0.5, 0.1], [0.1, 0.5]], dtype=STAT_DTYPE)
    Lg = torch.linalg.cholesky(true_Vg)
    Lk = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=STAT_DTYPE))
    U = Lk @ torch.randn(n, 2, dtype=STAT_DTYPE) @ Lg.T
    Le = torch.linalg.cholesky(true_Ve)
    E = torch.randn(n, 2, dtype=STAT_DTYPE) @ Le.T
    B = torch.tensor([[2.0, 3.0]], dtype=STAT_DTYPE)
    Y = X0 @ B + U + E

    # Plant a causal SNP on trait 1
    causal_idx = 42
    Y[:, 0] = Y[:, 0] + G[:, causal_idx].to(STAT_DTYPE) * 0.8

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return Y, X0, K, G, vmeta, causal_idx, true_Vg, true_Ve


def main() -> None:
    Y, X0, K, G, vmeta, causal_idx, true_Vg, true_Ve = simulate()
    print(f"Simulated: n={Y.shape[0]} samples, m={G.shape[1]} SNPs, d={Y.shape[1]} traits")
    print(f"Causal SNP planted on trait 0 at index {causal_idx}")

    model = MultiTraitLMM()
    nf = model.fit_null(Y, X0, K=K)
    print(f"Null Vg =\n{nf.Vg.cpu().numpy()}")
    print(f"Null Ve =\n{nf.Ve.cpu().numpy()}")

    result = model.score_chunk(G, nf, vmeta, test="wald")
    p = result.p.cpu().numpy()
    df = pd.DataFrame({
        "snp": result.snp,
        "chr": vmeta.chr,
        "pos": vmeta.pos,
        "pval_joint": p,
    }).sort_values("pval_joint")

    out = OUT_DIR / "02_mvlmm_results.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")
    print(f"Top 5:\n{df.head(5).to_string(index=False)}")
    top_rank = int(np.where(df["snp"].values == f"rs{causal_idx}")[0][0]) + 1
    print(f"Causal rs{causal_idx} ranked #{top_rank} out of {len(df)}")


if __name__ == "__main__":
    main()
