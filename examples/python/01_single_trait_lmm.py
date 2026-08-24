"""Single-trait LMM on synthetic data — the friendly API and the low-level path.

Demonstrates two ways to run the same single-trait mixed-model GWAS:

1. ``tg.gwas(...)`` — the one-call friendly API (auto QC / kinship / alignment,
   self-describing result).
2. ``SingleTraitLMM`` — the low-level building blocks (null fit + Wald scan),
   with BH correction, a Manhattan + QQ plot, and a results TSV.

Both run on a small simulated fixture with one planted causal SNP, so the
script is fully self-contained (no external data files required).

Phases covered: 1 (GLM baseline), 4 (SingleTraitLMM), 30 (adaptive FDR).

Runtime: < 10 s on CPU (200 samples x 400 SNPs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.stats import benjamini_hochberg
from torchgenomics.viz import manhattan_plot, qq_plot

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate(n=200, m=400, causal_idx=137, effect=1.2, seed=7):
    """Simulate a single continuous trait with a K-correlated background and
    one planted causal SNP. Returns numpy/pandas objects for the friendly API
    plus the torch tensors the low-level path needs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # HWE-consistent genotypes: per-SNP MAF in [0.1, 0.5], dosage ~ Binomial(2, p)
    p = rng.uniform(0.1, 0.5, size=m)
    G_np = rng.binomial(2, p, size=(n, m)).astype(np.float64)
    G = torch.tensor(G_np, dtype=STAT_DTYPE)

    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    # Polygenic background (K-correlated) + causal effect + noise
    Lk = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=STAT_DTYPE))
    u = Lk @ torch.randn(n, 1, dtype=STAT_DTYPE) * 0.7
    noise = torch.randn(n, 1, dtype=STAT_DTYPE) * 0.5
    y = 3.0 + u + G[:, causal_idx : causal_idx + 1] * effect + noise

    sample_ids = [f"IND{i:04d}" for i in range(n)]
    y_series = pd.Series(y.squeeze(1).numpy(), index=sample_ids, name="height")

    vmeta = VariantMeta(
        snp=[f"rs{i:04d}" for i in range(m)],
        chr=[str(1 + i // (m // 5 + 1)) for i in range(m)],
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return y_series, y, G_np, G, X0, K, vmeta, causal_idx


def run_friendly(y_series, G_np, causal_idx):
    """Path 1: the one-call friendly API on in-memory arrays.

    An in-memory genotype array carries no SNP ids, so ``tg.gwas`` assigns
    synthetic names ``snp0..snp{m-1}`` — the causal column is ``snp{causal_idx}``.
    """
    import torchgenomics as tg

    print("=" * 60)
    print("1. Friendly API — tg.gwas(...)")
    print("=" * 60)
    r = tg.gwas(phenotype=y_series, genotype=G_np, models="lmm", verbose=False)
    print(r.summary())
    causal_name = f"snp{causal_idx}"
    top = list(r.hits["SNP"]) if "SNP" in r.hits.columns else []
    if top and top[0] == causal_name:
        print(f"\nCausal {causal_name} is the top hit.")
    elif causal_name in set(top):
        print(f"\nCausal {causal_name} is among the top hits.")
    print()


def run_lowlevel(Y, G, X0, K, vmeta, causal_idx):
    """Path 2: the low-level SingleTraitLMM building blocks + plots + TSV.

    Runs on the exact same simulated trait ``Y`` the friendly path used.
    """
    print("=" * 60)
    print("2. Low-level — SingleTraitLMM.fit_null + score_chunk")
    print("=" * 60)
    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(Y, X0, K=K)
    print(f"Null fit: vg={float(nf.sig2_g):.4f} ve={float(nf.sig2_e):.4f} "
          f"h2={float(nf.sig2_g) / (float(nf.sig2_g) + float(nf.sig2_e)):.3f}")

    result = model.score_chunk(G, nf, vmeta, test="wald")
    p_t = result.p.cpu()
    p = p_t.numpy()

    valid = torch.isfinite(p_t)
    bh_p = torch.full_like(p_t, float("nan"))
    bh_p[valid] = benjamini_hochberg(p_t[valid])
    bh_p = bh_p.numpy()

    chr_numeric = np.array([int(c) for c in vmeta.chr], dtype=int)
    pos = np.array(vmeta.pos, dtype=int)
    df = pd.DataFrame({
        "snp": result.snp, "chr": chr_numeric, "pos": pos,
        "beta": result.beta.cpu().numpy(), "se": result.se.cpu().numpy(),
        "pval": p, "fdr": bh_p,
    }).sort_values("pval")

    out_tsv = OUT_DIR / "01_single_trait_lmm.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv} ({len(df)} rows)")
    print("Top 5 hits:")
    print(df.head(5).to_string(index=False))
    rank = int(np.where(df["snp"].values == f"rs{causal_idx:04d}")[0][0]) + 1
    print(f"Causal rs{causal_idx:04d} ranked #{rank} of {len(df)}")

    mh = OUT_DIR / "01_single_trait_lmm_manhattan.png"
    manhattan_plot(chrom=chr_numeric, pos=pos, p=p,
                   significance_threshold=5e-8, suggestive_threshold=1e-5,
                   title="Synthetic — single-trait LMM", output_path=str(mh))
    qq = OUT_DIR / "01_single_trait_lmm_qq.png"
    qq_plot(p=p, title="Synthetic — QQ", output_path=str(qq))
    print(f"Wrote {mh} and {qq}")


def main() -> None:
    y_series, Y, G_np, G, X0, K, vmeta, causal_idx = simulate()
    causal_snp = vmeta.snp[causal_idx]
    print(f"Simulated: n={len(y_series)} samples, m={G.shape[1]} SNPs, "
          f"causal SNP {causal_snp}\n")
    run_friendly(y_series, G_np, causal_idx)
    run_lowlevel(Y, G, X0, K, vmeta, causal_idx)


if __name__ == "__main__":
    main()
