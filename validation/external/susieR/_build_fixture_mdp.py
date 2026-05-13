"""Build a real-data per-locus (z, R, n) triple from the MDP fixture.

Per NA1 design spec §5.2 (the originally-prescribed Tier 2 fixture):
  - Load MDP genotype (281 maize lines × 3093 SNPs, additive coding {0,1,2})
  - Load MDP phenotype (Taxa, EarHT, dpoll, EarDia)
  - Use 'EarHT' (ear height) as trait — most-complete column
  - Compute marginal-regression z-scores per SNP
  - Take p=200 SNP window centered on the strongest hit (or first 200 SNPs
    if no strong hit) so n > p (R is rank-full)
  - Compute LD matrix R from genotype (in-sample)
  - Save (z, R, n) triple in the existing fixture format

Real LD structure tests:
  - Genuine population LD (not synthetic AR(1))
  - Mix of low-MAF / high-MAF variants
  - Polyploid-adjacent organism (maize) — different from human GWAS LD

Output overwrites the canonical fixture files in DATA_DIR.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MDP_DIR = Path.home() / "Documents" / "GWAS_Expert" / "benchmark" / "data"


def main() -> int:
    # Load genotype: rows = samples, cols = (Taxa, then 3093 SNPs)
    geno_df = pd.read_csv(MDP_DIR / "mdp_numeric.txt", sep="\t")
    snp_cols = [c for c in geno_df.columns if c != "taxa"]
    print(f"MDP genotype: {len(geno_df)} samples × {len(snp_cols)} SNPs")

    # Load SNP info for chr/pos
    snp_info = pd.read_csv(MDP_DIR / "mdp_SNP_information.txt", sep="\t")
    snp_info = snp_info.rename(columns={"SNP": "snp"})
    snp_info = snp_info.set_index("snp")

    # Load phenotype + join to genotype on Taxa
    traits = pd.read_csv(MDP_DIR / "mdp_traits.txt", sep="\t")
    traits = traits.rename(columns={"Taxa": "taxa"})
    traits["taxa"] = traits["taxa"].astype(str)
    geno_df["taxa"] = geno_df["taxa"].astype(str)
    merged = geno_df.merge(traits, on="taxa", how="inner")
    print(f"After joining traits: {len(merged)} samples")

    # Use EarHT (ear height) — drop NaN rows
    pheno = merged["EarHT"].replace("NaN", np.nan).astype(float).values
    keep = ~np.isnan(pheno)
    y = pheno[keep]
    G = merged[snp_cols].values[keep].astype(np.float64)
    n = G.shape[0]
    print(f"After dropping NaN phenotype: n={n}")

    # Mean-impute any missing genotypes (rare in MDP but be safe)
    col_means = np.nanmean(G, axis=0)
    inds = np.where(np.isnan(G))
    G[inds] = col_means[inds[1]]

    # Drop monomorphic / constant SNPs (would produce NaN z-scores)
    col_var = G.var(axis=0, ddof=1)
    nonzero_var = col_var > 1e-12
    G = G[:, nonzero_var]
    surviving_snps = [snp_cols[i] for i in range(len(snp_cols)) if nonzero_var[i]]
    print(f"After dropping monomorphic: p={G.shape[1]} SNPs surviving")

    # Standardize columns (mean-center, unit-variance)
    G = G - G.mean(axis=0, keepdims=True)
    G = G / G.std(axis=0, keepdims=True, ddof=1)
    y = y - y.mean()

    # Marginal-regression z-scores
    p_total = G.shape[1]
    beta_hat = (G.T @ y) / (n - 1)
    se_marginal = np.sqrt(np.var(y, ddof=1) / (n - 1))
    z_full = beta_hat / se_marginal

    # Find top hit and take a window of p=200 SNPs around it (n=281, p=200 → R rank-full)
    P_WINDOW = 200
    top_idx = int(np.argmax(np.abs(z_full)))
    print(f"Top hit: SNP idx {top_idx} ({surviving_snps[top_idx]}), |z|={abs(z_full[top_idx]):.2f}")

    half = P_WINDOW // 2
    start = max(0, top_idx - half)
    stop = min(p_total, start + P_WINDOW)
    if stop - start < P_WINDOW:
        start = max(0, stop - P_WINDOW)
    print(f"Window: indices {start}..{stop} ({stop - start} SNPs)")

    z_win = z_full[start:stop]
    G_win = G[:, start:stop]
    snps_win = surviving_snps[start:stop]
    R_win = (G_win.T @ G_win) / (n - 1)

    # Diagnostics
    p_win = stop - start
    print(f"\nWindow stats:")
    print(f"  |z| at top: {abs(z_full[top_idx]):.3f}")
    print(f"  |z| max in window: {np.abs(z_win).max():.3f}")
    print(f"  |z| > 2 in window: {(np.abs(z_win) > 2).sum()} variants")
    print(f"  R off-diagonal max: {(R_win - np.eye(p_win)).max():.3f}")
    print(f"  R off-diagonal mean abs: {np.abs(R_win[~np.eye(p_win, dtype=bool)]).mean():.3f}")
    print(f"  R condition number: {np.linalg.cond(R_win):.2e}")

    # Build snp_meta with real chr/bp/allele info
    snp_meta = {
        "snp_ids": snps_win,
        "chr": [str(snp_info.loc[s, "Chromosome"]) if s in snp_info.index else "NA" for s in snps_win],
        "bp": [int(snp_info.loc[s, "Position"]) if s in snp_info.index else 0 for s in snps_win],
        "a1": ["A"] * p_win,
        "a2": ["G"] * p_win,
    }

    # Save tensors
    torch.save(torch.from_numpy(z_win).to(torch.float64), DATA_DIR / "locus_z.pt")
    torch.save(torch.from_numpy(R_win).to(torch.float64), DATA_DIR / "locus_R.pt")
    (DATA_DIR / "locus_n.txt").write_text(f"{n}\n")
    torch.save(snp_meta, DATA_DIR / "locus_meta.pt")
    np.savetxt(DATA_DIR / "locus_z.tsv", z_win, delimiter="\t")
    np.savetxt(DATA_DIR / "locus_R.tsv", R_win, delimiter="\t")

    print(f"\nSaved MDP fixture to {DATA_DIR}")
    print(f"  n={n}, p={p_win}, R_size={R_win.nbytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
