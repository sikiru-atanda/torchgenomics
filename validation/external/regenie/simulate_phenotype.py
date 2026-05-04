"""Generate regenie-format phenotype + covariate files from the MDP fixture.

regenie's Step 1 + Step 2 expect a sample-keyed table. We reuse the MDP
EarHT continuous trait and synthesize a matched binary case/control phenotype
by thresholding EarHT at its sample median (a clean balanced split that
preserves the same kinship signal). For balance with regenie's input
expectations:

  - Output `data/regenie_pheno.tsv` with columns:
        FID  IID  EarHT  EarHT_bin
    EarHT is the original continuous trait. EarHT_bin is 1 if EarHT >= median
    else 0. NaN-valued rows propagate as `NA` in both columns (regenie skips).

  - Output `data/regenie_covar.tsv` with columns:
        FID  IID  PC1  PC2
    PCs are computed once from the MDP genotype matrix (mean-imputed) via
    a deterministic SVD on a fixed seed. Two PCs is enough to absorb the
    coarse population structure on this 281-sample fixture without crossing
    into the n_PC > n_samples / 50 rule of thumb.

The phenotype simulator is deliberately simple (thresholding, not logistic
regression on planted causal SNPs) — we want regenie and TorchGWAS to see
the *same* phenotype data; the harness compares per-SNP statistics, not the
ability to recover a planted truth. A planted-truth design would be useful
for power calibration but is not needed for tool-to-tool equivalence.

Usage:
    python3 simulate_phenotype.py \
        --plink-prefix data/mdp \
        --src-pheno data/mdp_pheno.txt \
        --output-dir data \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.io.plink import PlinkBedReader  # noqa: E402


def _mean_impute(G: np.ndarray) -> np.ndarray:
    """In-place mean imputation along axis 0 (samples)."""
    G = G.astype(np.float64)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = np.isnan(col)
        if mask.any():
            valid = col[~mask]
            G[mask, j] = valid.mean() if valid.size else 0.0
    return G


def _compute_pcs(G: np.ndarray, n_pcs: int = 2, seed: int = 42) -> np.ndarray:
    """Centered + scaled PCA via SVD; deterministic given seed (np.linalg.svd is)."""
    af = G.mean(axis=0) / 2.0
    sd = np.sqrt(2.0 * af * (1.0 - af) + 1e-12)
    Z = (G - 2.0 * af) / sd
    # Use np.linalg.svd on centered/scaled matrix; full_matrices=False gives min(n,m).
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :n_pcs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plink-prefix", required=True, type=Path)
    ap.add_argument("--src-pheno", required=True, type=Path,
                    help="MDP-style pheno file (TSV) with FID IID EarHT dpoll")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load PLINK fileset to get the canonical sample order regenie will see.
    reader = PlinkBedReader(args.plink_prefix)
    fam_samples = reader.sample_ids  # IIDs in BED order

    # Build n×m matrix for PC computation
    chunks = []
    for G_chunk, _ in reader.iter_chunks(chunk_size=4096):
        chunks.append(G_chunk.cpu().numpy())
    G_full = np.concatenate(chunks, axis=1)  # (n, m)
    G_full = _mean_impute(G_full)

    pcs = _compute_pcs(G_full, n_pcs=2, seed=args.seed)
    print(f"[simulate-pheno] computed 2 PCs from {G_full.shape[0]}×{G_full.shape[1]} mean-imputed matrix",
          file=sys.stderr)

    # Load source pheno; reindex to match BED FAM order.
    src = pd.read_csv(args.src_pheno, sep="\t")
    src["IID"] = src["IID"].astype(str)
    src["FID"] = src["FID"].astype(str)
    src = src.set_index("IID").reindex(fam_samples).reset_index().rename(columns={"index": "IID"})
    # Re-fill FID if reindex left it null (no missing samples expected, but
    # be defensive).
    src["FID"] = src["FID"].fillna(src["IID"])

    # Binary trait: threshold EarHT at the sample median (excluding NaN).
    earht = src["EarHT"].to_numpy(dtype=np.float64)
    finite = np.isfinite(earht)
    threshold = float(np.median(earht[finite]))
    earht_bin = np.where(finite, (earht >= threshold).astype(int), -9)
    print(f"[simulate-pheno] EarHT median = {threshold:.4f}; "
          f"cases={int((earht_bin == 1).sum())}, controls={int((earht_bin == 0).sum())}, "
          f"missing={int((earht_bin == -9).sum())}",
          file=sys.stderr)

    # regenie convention: missing values for quantitative = "NA"; for binary
    # the canonical encoding is 1=case, 0=control, NA=missing.
    earht_str = pd.Series(earht).where(np.isfinite(earht), "NA").astype(object)
    earht_bin_str = pd.Series(earht_bin).where(earht_bin != -9, "NA").astype(object)

    pheno = pd.DataFrame({
        "FID": src["FID"].astype(str),
        "IID": src["IID"].astype(str),
        "EarHT": earht_str,
        "EarHT_bin": earht_bin_str,
    })
    out_pheno = args.output_dir / "regenie_pheno.tsv"
    pheno.to_csv(out_pheno, sep="\t", index=False)
    print(f"[simulate-pheno] wrote {out_pheno}", file=sys.stderr)

    covar = pd.DataFrame({
        "FID": src["FID"].astype(str),
        "IID": src["IID"].astype(str),
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
    })
    out_covar = args.output_dir / "regenie_covar.tsv"
    covar.to_csv(out_covar, sep="\t", index=False)
    print(f"[simulate-pheno] wrote {out_covar}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
