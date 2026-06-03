"""Fine-mapping with SuSiE Bayesian variable selection (Iterative Bayesian SS).

Demonstrates: fit a SingleTraitLMM null, then run BayesianVS (SuSiE-style)
to identify credible sets of putatively causal SNPs.

Phases covered: 18 (BayesianVS; SuSiE + CAVI).

Runtime: < 15 s on CPU (200 samples x 500 SNPs, 2 planted causal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models import BayesianVS, SingleTraitLMM, VariantMeta

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    torch.manual_seed(42)
    n, m = 200, 500

    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=STAT_DTYPE)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=STAT_DTYPE))
    Y_null = 5.0 + L_V @ torch.randn(n, dtype=STAT_DTYPE)

    causal = [120, 340]
    for idx, eff in zip(causal, [0.8, -0.6]):
        Y_null = Y_null + G[:, idx] * eff
    Y = Y_null.unsqueeze(1)

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )

    lmm = SingleTraitLMM()
    nf = lmm.fit_null(Y, X0, K=K)
    print(f"Null: vg={float(nf.sig2_g):.3f}, ve={float(nf.sig2_e):.3f}")

    bvs = BayesianVS(nf)
    result = bvs.fit(G, vmeta, method="susie", n_signals=5,
                     max_iter=200, prior_pi=0.01, learn_hyperparams=True)
    print(f"SuSiE converged? ELBO final = {result.elbo_trace[-1]:.3f}, "
          f"iters={len(result.elbo_trace)}")
    print(f"Credible sets: {len(result.credible_sets)} recovered")
    for i, cs in enumerate(result.credible_sets):
        print(f"  CS{i+1}: size={len(cs)}, SNPs={cs[:10]}")

    pip = result.pip.cpu().numpy()
    df = pd.DataFrame({
        "snp": [f"rs{i}" for i in range(m)],
        "pip": pip,
        "beta_mean": result.beta_mean.cpu().numpy(),
        "beta_sd": result.beta_sd.cpu().numpy(),
    }).sort_values("pip", ascending=False)

    out = OUT_DIR / "06_susie_pips.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}")
    print(f"Top 10 PIPs:\n{df.head(10).to_string(index=False)}")
    print(f"Planted causal: {['rs' + str(c) for c in causal]}")


if __name__ == "__main__":
    main()
