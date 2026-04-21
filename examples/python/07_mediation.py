"""Causal mediation: SNP -> M (expression) -> Y (phenotype) with GRM correction.

Demonstrates: mediate_lmm for a single SNP-mediator-outcome triple, and
scan_mediation for a genome-wide SNP x feature mediation scan with BH FDR.

Phases covered: 49 (mediate_lmm / scan_mediation), 49b (batched scan).

Runtime: < 15 s on CPU (n=200; single-SNP fit + 20 SNPs x 10 features scan).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torchgwas.config import STAT_DTYPE
from torchgwas.multiomics import mediate_lmm, scan_mediation

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sim_grm(n, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 200))
    K = (Z @ Z.T) / 200.0
    K = 0.5 * (K + K.T) + 1e-3 * np.eye(n)
    return torch.as_tensor(K, dtype=STAT_DTYPE)


def main() -> None:
    # --- Single triple: planted SNP -> M -> Y chain ---
    n = 400
    rng = np.random.default_rng(1)
    K = sim_grm(n, seed=1).numpy()
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    g = L @ rng.standard_normal(n) * 0.4
    snp = rng.binomial(2, 0.3, size=n).astype(np.float64)
    a, b, c_prime = 0.4, 0.5, 0.1
    M = a * snp + rng.standard_normal(n) * 0.5
    Y = c_prime * snp + b * M + g + rng.standard_normal(n) * 0.5

    Yt = torch.as_tensor(Y, dtype=STAT_DTYPE)
    snp_t = torch.as_tensor(snp, dtype=STAT_DTYPE)
    Mt = torch.as_tensor(M, dtype=STAT_DTYPE)
    Kt = torch.as_tensor(K, dtype=STAT_DTYPE)

    res = mediate_lmm(Yt, snp_t, Mt, Kt, se="monte-carlo", n_mc_draws=5000, seed=42)
    print("Single triple mediation:")
    print(f"  a   (SNP -> M)      = {res.a:.4f}")
    print(f"  b   (M -> Y | SNP)  = {res.b:.4f}")
    print(f"  c   (SNP -> Y)      = {res.c:.4f}")
    print(f"  c'  (direct)        = {res.c_prime:.4f}")
    print(f"  ab  (indirect)      = {res.indirect:.4f}  "
          f"se={res.indirect_se:.4f}  p={res.indirect_pvalue:.4g}")
    print(f"  proportion mediated = {res.proportion_mediated:.3f}")
    print(f"  (true a={a}, b={b}, c'={c_prime}, ab={a*b:.4f})")

    # --- Genome-wide scan: 20 SNPs x 10 expression features ---
    n2, n_snps, n_feats = 200, 20, 10
    K2 = sim_grm(n2, seed=7)
    rng2 = np.random.default_rng(7)
    G = torch.as_tensor(
        rng2.binomial(2, 0.3, size=(n2, n_snps)).astype(np.float64), dtype=STAT_DTYPE,
    )
    # Plant causal: SNP 5 -> feature 2 -> Y
    Mbg = torch.randn(n2, n_feats, dtype=STAT_DTYPE) * 0.5
    Mbg[:, 2] = Mbg[:, 2] + G[:, 5] * 0.6
    Y2 = G[:, 5] * 0.1 + Mbg[:, 2] * 0.5 + torch.randn(n2, dtype=STAT_DTYPE) * 0.5

    scan = scan_mediation(
        Y2, G, Mbg, K2,
        snp_ids=[f"rs{i}" for i in range(n_snps)],
        feature_ids=[f"gene{j}" for j in range(n_feats)],
        cis_window_bp=None,
        se="monte-carlo", n_mc_draws=2000, seed=0,
        fdr_method="bh",
    )
    df = scan.to_dataframe().sort_values("q_indirect")
    out = OUT_DIR / "07_mediation_scan.tsv"
    df.to_csv(out, sep="\t", index=False)
    n_sig = int((df["q_indirect"] <= 0.05).sum())
    print(f"\nScan: {len(df)} SNPxfeature pairs, {n_sig} FDR<=0.05")
    cols = ["snp", "feature", "a", "b", "indirect", "indirect_pvalue", "q_indirect"]
    print(f"Top 5 pairs by q_indirect:\n{df[cols].head(5).to_string(index=False)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
