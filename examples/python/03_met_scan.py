"""Multi-environment GWAS (MET) with reaction-norm parameterization.

Demonstrates: MultiEnvLMM with unstructured Vg across 3 environments,
reaction-norm (alpha = mean effect, delta = env-specific deviation) scan.

Phases covered: 19 (MET), 36 (MT-MET scaling).

Runtime: < 10 s on CPU (150 samples x 300 SNPs x 3 envs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models import MultiEnvLMM, VariantMeta

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    torch.manual_seed(42)
    n, m, E = 150, 300, 3
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=STAT_DTYPE))
    Vg_true = torch.tensor(
        [[1.0, 0.7, 0.5], [0.7, 1.0, 0.6], [0.5, 0.6, 1.0]], dtype=STAT_DTYPE,
    )
    L_vg = torch.linalg.cholesky(Vg_true)
    Z = torch.randn(n, E, dtype=STAT_DTYPE)
    U = L @ Z @ L_vg.T
    Y = U + torch.randn(n, E, dtype=STAT_DTYPE)

    # Plant a causal SNP with env-specific effect
    causal_idx = 77
    env_effects = torch.tensor([0.8, -0.2, 0.5], dtype=STAT_DTYPE)
    Y = Y + torch.outer(G[:, causal_idx].to(STAT_DTYPE), env_effects)

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )

    print(f"Simulated MET: n={n}, m={m}, E={E}")
    print(f"Causal SNP at index {causal_idx} with env effects {env_effects.tolist()}")

    model = MultiEnvLMM(parameterization="reaction_norm")
    nf = model.fit_null(Y, X0, K, env_names=[f"E{i+1}" for i in range(E)])
    print(f"Null Vg (3x3) trace={float(torch.diag(nf.Vg).sum()):.3f}")

    result = model.score_chunk(G, nf, vmeta)
    df = pd.DataFrame({
        "snp": result.snp,
        "pval_overall": result.p.cpu().numpy(),
        "alpha": result.alpha.cpu().numpy(),
    })
    for e in range(E):
        df[f"beta_E{e+1}"] = result.beta[:, e].cpu().numpy()
    df = df.sort_values("pval_overall")

    out = OUT_DIR / "03_met_results.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")
    print(f"Top 5:\n{df.head(5).to_string(index=False)}")
    top_rank = int(np.where(df["snp"].values == f"rs{causal_idx}")[0][0]) + 1
    print(f"Causal rs{causal_idx} ranked #{top_rank} out of {len(df)}")


if __name__ == "__main__":
    main()
