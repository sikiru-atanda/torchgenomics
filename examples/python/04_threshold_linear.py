"""Threshold-linear GWAS for mixed ordinal + continuous traits (Bermann 2026).

Demonstrates: ThresholdLinearModel with a 2-category ordinal trait and a
continuous trait sharing a residual covariance; EM solver; score-test scan.

Phases covered: 22 (threshold-linear GWAS).

Runtime: < 20 s on CPU (300 samples x 100 SNPs, 2 traits).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgwas.config import STAT_DTYPE
from torchgwas.models import ThresholdLinearModel, VariantMeta

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    torch.manual_seed(42)
    n, m = 300, 100

    G = torch.randn(n, m, dtype=STAT_DTYPE)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    liab = torch.randn(n, dtype=STAT_DTYPE)
    causal_idx = 17
    liab = liab + G[:, causal_idx] * 0.5

    Y = torch.zeros(n, 2, dtype=STAT_DTYPE)
    Y[:, 0] = (liab > 0).float()
    Y[:, 1] = liab + 0.3 * torch.randn(n, dtype=STAT_DTYPE)

    model = ThresholdLinearModel(
        trait_types=["ordinal", "continuous"],
        n_categories=[2, 0],
        R=torch.eye(2, dtype=STAT_DTYPE),
        G_cov=torch.eye(2, dtype=STAT_DTYPE) * 0.5,
        solver="em",
        max_iter=30,
    )
    nf = model.fit_null(Y, X0)

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )
    result = model.score_chunk(G, nf, vmeta)
    df = pd.DataFrame({
        "snp": result.snp,
        "beta": result.beta.cpu().numpy(),
        "se": result.se.cpu().numpy(),
        "pval": result.p.cpu().numpy(),
    }).sort_values("pval")

    out = OUT_DIR / "04_threshold_results.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")
    print(f"Top 5:\n{df.head(5).to_string(index=False)}")
    top_rank = int(np.where(df["snp"].values == f"rs{causal_idx}")[0][0]) + 1
    print(f"Causal rs{causal_idx} (ordinal+continuous) ranked #{top_rank} of {len(df)}")


if __name__ == "__main__":
    main()
