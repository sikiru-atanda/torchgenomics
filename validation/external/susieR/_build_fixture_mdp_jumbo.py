"""Build a JUMBO MDP fixture: larger window for benchmark 2.

p=270 (rank-full, just under n=279) and p=500 (rank-deficient, 1.8x n).
Both centered on the same top hit as the standard MDP fixture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MDP_DIR = Path.home() / "Documents" / "GWAS_Expert" / "benchmark" / "data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, required=True, help="Window size in SNPs")
    args = parser.parse_args()
    P_WINDOW = args.p

    geno_df = pd.read_csv(MDP_DIR / "mdp_numeric.txt", sep="\t")
    snp_cols = [c for c in geno_df.columns if c != "taxa"]
    snp_info = pd.read_csv(MDP_DIR / "mdp_SNP_information.txt", sep="\t")
    snp_info = snp_info.rename(columns={"SNP": "snp"}).set_index("snp")
    traits = pd.read_csv(MDP_DIR / "mdp_traits.txt", sep="\t")
    traits = traits.rename(columns={"Taxa": "taxa"})
    traits["taxa"] = traits["taxa"].astype(str)
    geno_df["taxa"] = geno_df["taxa"].astype(str)
    merged = geno_df.merge(traits, on="taxa", how="inner")

    pheno = merged["EarHT"].replace("NaN", np.nan).astype(float).values
    keep = ~np.isnan(pheno)
    y = pheno[keep]
    G = merged[snp_cols].values[keep].astype(np.float64)
    n = G.shape[0]
    col_means = np.nanmean(G, axis=0)
    inds = np.where(np.isnan(G))
    G[inds] = col_means[inds[1]]
    col_var = G.var(axis=0, ddof=1)
    nonzero_var = col_var > 1e-12
    G = G[:, nonzero_var]
    surviving_snps = [snp_cols[i] for i in range(len(snp_cols)) if nonzero_var[i]]
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True, ddof=1)
    y = y - y.mean()

    p_total = G.shape[1]
    beta_hat = (G.T @ y) / (n - 1)
    se_marginal = np.sqrt(np.var(y, ddof=1) / (n - 1))
    z_full = beta_hat / se_marginal

    top_idx = int(np.argmax(np.abs(z_full)))
    half = P_WINDOW // 2
    start = max(0, top_idx - half)
    stop = min(p_total, start + P_WINDOW)
    if stop - start < P_WINDOW:
        start = max(0, stop - P_WINDOW)

    z_win = z_full[start:stop]
    G_win = G[:, start:stop]
    snps_win = surviving_snps[start:stop]
    R_win = (G_win.T @ G_win) / (n - 1)
    p_win = stop - start

    print(f"JUMBO fixture p={p_win}: top_idx={top_idx}, window=[{start},{stop})")
    print(f"  |z| max in window: {np.abs(z_win).max():.3f}, n={n}")
    print(f"  |z| > 2 count: {(np.abs(z_win) > 2).sum()}")
    print(f"  R rank-deficient: {p_win > n - 1}")
    if p_win <= n:
        print(f"  R condition number: {np.linalg.cond(R_win):.2e}")
    eigvals = np.linalg.eigvalsh(R_win)
    print(f"  R min eigval: {eigvals.min():.3e},  max: {eigvals.max():.3e}")

    snp_meta = {
        "snp_ids": snps_win,
        "chr": [str(snp_info.loc[s, "Chromosome"]) if s in snp_info.index else "NA" for s in snps_win],
        "bp": [int(snp_info.loc[s, "Position"]) if s in snp_info.index else 0 for s in snps_win],
        "a1": ["A"] * p_win,
        "a2": ["G"] * p_win,
    }

    torch.save(torch.from_numpy(z_win).to(torch.float64), DATA_DIR / "locus_z.pt")
    torch.save(torch.from_numpy(R_win).to(torch.float64), DATA_DIR / "locus_R.pt")
    (DATA_DIR / "locus_n.txt").write_text(f"{n}\n")
    torch.save(snp_meta, DATA_DIR / "locus_meta.pt")
    np.savetxt(DATA_DIR / "locus_z.tsv", z_win, delimiter="\t")
    np.savetxt(DATA_DIR / "locus_R.tsv", R_win, delimiter="\t")
    print(f"Saved to {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
