"""Polyploid GWAS on tetraploid potato (GWASpoly benchmark data).

Demonstrates: load the GWASpoly potato fixture, build a ploidy-aware GRM with
`grm_polyploid_gene_action`, and scan with SingleTraitLMM + gene-action recoding
(additive vs 1-dominant-alt) on the vine.maturity trait.

Phases covered: polyploid GRM + gene-action models (V1 core + Phase 11-12).

Runtime: < 30 s on CPU (~250 samples x 200 SNPs, ploidy=4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.linalg.kinship_polyploid import grm_polyploid_gene_action
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.preprocess.polyploid import recode_gene_action

ROOT = Path(__file__).resolve().parents[2]
GWASPOLY_DATA = ROOT / "benchmark" / "gwaspoly_data"
GWASPOLY_RESULTS = ROOT / "benchmark" / "gwaspoly_results"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_potato():
    if (GWASPOLY_RESULTS / "potato_geno_aligned.csv").is_file():
        geno_df = pd.read_csv(GWASPOLY_RESULTS / "potato_geno_aligned.csv", index_col=0)
        pheno_df = pd.read_csv(GWASPOLY_RESULTS / "potato_pheno_with_genoid.csv")
        map_df = pd.read_csv(GWASPOLY_RESULTS / "potato_map.csv")
    elif (GWASPOLY_DATA / "potato_geno_subset.csv").is_file():
        geno_df = pd.read_csv(GWASPOLY_DATA / "potato_geno_subset.csv", index_col=0)
        pheno_df = pd.read_csv(GWASPOLY_DATA / "potato_pheno_subset.csv")
        map_df = None
    else:
        print("[skip] potato benchmark data not found in benchmark/gwaspoly_*")
        sys.exit(0)

    geno_df.index = geno_df.index.astype(str)
    pheno_df = pheno_df.dropna(subset=["vine.maturity"]).copy()
    pheno_df["geno_id"] = pheno_df["geno_id"].astype(str)
    pheno_agg = pheno_df.groupby("geno_id")["vine.maturity"].mean()

    common = sorted(set(geno_df.index) & set(pheno_agg.index))
    geno = geno_df.loc[common]
    pheno = pheno_agg.loc[common]

    G = torch.tensor(geno.values, dtype=STAT_DTYPE)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()
    Y = torch.tensor(pheno.values, dtype=STAT_DTYPE)

    snps = list(geno.columns)[:200]
    if map_df is not None:
        chrs = map_df["chrom"].astype(str).tolist()[:200]
        pos_list = map_df["pos"].astype(int).tolist()[:200]
    else:
        chrs, pos_list = ["1"] * 200, list(range(200))

    vmeta = VariantMeta(
        snp=snps, chr=chrs, pos=pos_list,
        a1=["A"] * 200, a2=["G"] * 200,
    )
    return G, Y, vmeta


def main() -> None:
    G, Y, vmeta = load_potato()
    n, m_total = G.shape
    G_scan = G[:, :200]
    print(f"Potato (tetraploid): n={n}, m_scan={G_scan.shape[1]} (of {m_total} total)")

    K, _ = grm_polyploid_gene_action(G, model="additive", ploidy=4)
    print(f"GRM (additive, ploidy=4): trace/n = {float(torch.diag(K).mean()):.3f}")

    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
    model = SingleTraitLMM()
    nf = model.fit_null(Y.unsqueeze(1), X0, K=K)
    print(f"Null REML: vg={float(nf.sig2_g):.3f}, ve={float(nf.sig2_e):.3f}, "
          f"h2={float(nf.sig2_g / (nf.sig2_g + nf.sig2_e)):.3f}")

    # Additive scan
    res_add = model.score_chunk(G_scan, nf, vmeta, test="wald")
    df_add = pd.DataFrame({
        "snp": res_add.snp,
        "beta": res_add.beta.cpu().numpy(),
        "se": res_add.se.cpu().numpy(),
        "p": res_add.p.cpu().numpy(),
    }).sort_values("p")
    print(f"\nAdditive top 5:\n{df_add.head(5).to_string(index=False)}")

    # 1-dominant recoding (alt-dominant: genotypes 1..k collapsed to 1)
    G_dom = recode_gene_action(G_scan, model="1-dom", ploidy=4)
    res_dom = model.score_chunk(G_dom, nf, vmeta, test="wald")
    df_dom = pd.DataFrame({
        "snp": res_dom.snp,
        "beta": res_dom.beta.cpu().numpy(),
        "p": res_dom.p.cpu().numpy(),
    }).sort_values("p")
    print(f"\n1-dominant top 5:\n{df_dom.head(5).to_string(index=False)}")

    out = OUT_DIR / "09_polyploid_results.tsv"
    df_add.to_csv(out, sep="\t", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
