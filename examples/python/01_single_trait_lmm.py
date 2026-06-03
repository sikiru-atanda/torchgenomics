"""Single-trait LMM on MDP maize EarHT.

Demonstrates: SingleTraitLMM null fit + Wald/LRT/score scan on the MDP maize
fixture, BH correction, Manhattan plot, and a results TSV.

Phases covered: 1 (GLM baseline), 4 (SingleTraitLMM), 30 (adaptive FDR).

Runtime: < 15 s on CPU (276 samples x 3093 SNPs).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.stats import benjamini_hochberg
from torchgenomics.viz import manhattan_plot, qq_plot

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MDP_DATA = REPO_ROOT / "benchmark" / "data"
GEMMA_OUT = REPO_ROOT / "gemma_demo" / "output"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    pheno = pd.read_csv(MDP_DATA / "mdp_traits.txt", sep="\t")
    geno = pd.read_csv(MDP_DATA / "mdp_numeric.txt", sep="\t")
    snpmap = pd.read_csv(MDP_DATA / "mdp_SNP_information.txt", sep="\t")

    pheno["Taxa"] = pheno["Taxa"].astype(str)
    geno["taxa"] = geno["taxa"].astype(str)
    pheno = pheno[["Taxa", "EarHT", "dpoll"]].dropna(subset=["EarHT", "dpoll"])
    common = sorted(set(pheno["Taxa"]) & set(geno["taxa"]))
    pheno = pheno[pheno["Taxa"].isin(common)].set_index("Taxa").loc[common]
    geno = geno[geno["taxa"].isin(common)].set_index("taxa").loc[common]

    snp_names = list(geno.columns)
    n, m = geno.shape

    G = torch.tensor(geno.values, dtype=STAT_DTYPE)
    for j in range(m):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()

    Y = torch.tensor(pheno["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
    K = torch.tensor(
        pd.read_csv(GEMMA_OUT / "mdp_kinship.cXX.txt", sep="\t", header=None).values,
        dtype=STAT_DTYPE,
    )

    snp_to_cp = {
        str(r["SNP"]): (str(int(r["Chromosome"])), int(r["Position"]))
        for _, r in snpmap.iterrows()
    }
    chrs = [snp_to_cp.get(s, ("0", 0))[0] for s in snp_names]
    pos = [snp_to_cp.get(s, ("0", 0))[1] for s in snp_names]
    vmeta = VariantMeta(snp=snp_names, chr=chrs, pos=pos,
                        a1=["A"] * m, a2=["G"] * m)

    print(f"Loaded MDP: {n} samples, {m} SNPs")

    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(Y, X0, K=K)
    print(f"Null fit: vg={float(nf.sig2_g):.4f} ve={float(nf.sig2_e):.4f} "
          f"REML logL={float(nf.log_likelihood):.2f}")

    result = model.score_chunk(G, nf, vmeta, test="wald")
    p_t = result.p.cpu()
    p = p_t.numpy()
    beta = result.beta.cpu().numpy()
    se = result.se.cpu().numpy()

    # Mask NaN p-values (rare-allele or non-variable SNPs) from BH; assign fdr=NaN
    valid = torch.isfinite(p_t)
    bh_valid = benjamini_hochberg(p_t[valid])
    bh_p = torch.full_like(p_t, float("nan"))
    bh_p[valid] = bh_valid
    bh_p = bh_p.numpy()

    chr_numeric = np.array(
        [int(c) if str(c).isdigit() else 0 for c in chrs], dtype=int
    )
    df = pd.DataFrame({
        "snp": snp_names,
        "chr": chr_numeric,
        "pos": pos,
        "beta": beta,
        "se": se,
        "pval": p,
        "fdr": bh_p,
    }).sort_values(["chr", "pos"])
    out_tsv = OUT_DIR / "01_mdp_earht_lmm.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv} ({len(df)} rows)")
    print("Top 5 hits:")
    print(df.nsmallest(5, "pval")[["snp", "chr", "pos", "beta", "se", "pval", "fdr"]]
          .to_string(index=False))

    mh_path = OUT_DIR / "01_mdp_earht_manhattan.png"
    manhattan_plot(
        chrom=chr_numeric,
        pos=np.array(pos, dtype=int),
        p=p,
        significance_threshold=5e-8,
        suggestive_threshold=1e-5,
        title="MDP maize - EarHT - SingleTraitLMM",
        output_path=str(mh_path),
    )
    print(f"Wrote {mh_path}")

    qq_path = OUT_DIR / "01_mdp_earht_qq.png"
    qq_plot(p=p, title="MDP maize - EarHT - QQ", output_path=str(qq_path))
    print(f"Wrote {qq_path}")


if __name__ == "__main__":
    main()
