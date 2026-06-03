"""Generate BOLT-LMM-format phenotype + covariate files from the MDP fixture.

BOLT-LMM expects whitespace-separated TSVs with `FID IID` as the first two
columns and trait/covariate columns following. Missing values are encoded
as ``NA`` and BOLT skips those rows.

We re-use the MDP EarHT continuous trait and produce two PCs from the
genotype matrix as covariates — same shape as the regenie + SAIGE harnesses
so cross-tool comparison stays apples-to-apples.

Outputs
-------
- ``data/bolt_pheno.tsv``: ``FID  IID  EarHT``
- ``data/bolt_covar.tsv``: ``FID  IID  PC1  PC2``

Usage
-----
    python3 simulate_phenotype.py \\
        --plink-prefix data/mdp \\
        --src-pheno data/mdp_pheno.txt \\
        --output-dir data \\
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

from torchgenomics.io.plink import PlinkBedReader  # noqa: E402


def _mean_impute(G: np.ndarray) -> np.ndarray:
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
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :n_pcs]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--plink-prefix", required=True, type=Path)
    ap.add_argument(
        "--src-pheno",
        required=True,
        type=Path,
        help="MDP-style pheno file (TSV) with FID IID EarHT dpoll",
    )
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load PLINK fileset to get the canonical sample order BOLT will see.
    reader = PlinkBedReader(args.plink_prefix)
    fam_samples = reader.sample_ids  # IIDs in BED order

    # Build n×m matrix for PC computation.
    chunks = []
    for G_chunk, _ in reader.iter_chunks(chunk_size=4096):
        chunks.append(G_chunk.cpu().numpy())
    G_full = np.concatenate(chunks, axis=1)
    G_full = _mean_impute(G_full)

    pcs = _compute_pcs(G_full, n_pcs=2, seed=args.seed)
    print(
        f"[simulate-pheno] computed 2 PCs from {G_full.shape[0]}×{G_full.shape[1]} "
        f"mean-imputed matrix",
        file=sys.stderr,
    )

    # Reindex source pheno to FAM order.
    src = pd.read_csv(args.src_pheno, sep="\t")
    src["IID"] = src["IID"].astype(str)
    src["FID"] = src["FID"].astype(str)
    src = src.set_index("IID").reindex(fam_samples).reset_index().rename(
        columns={"index": "IID"}
    )
    src["FID"] = src["FID"].fillna(src["IID"])

    earht = src["EarHT"].to_numpy(dtype=np.float64)
    earht_str = pd.Series(earht).where(np.isfinite(earht), "NA").astype(object)

    pheno = pd.DataFrame({
        "FID": src["FID"].astype(str),
        "IID": src["IID"].astype(str),
        "EarHT": earht_str,
    })
    out_pheno = args.output_dir / "bolt_pheno.tsv"
    pheno.to_csv(out_pheno, sep="\t", index=False)
    print(f"[simulate-pheno] wrote {out_pheno}", file=sys.stderr)

    covar = pd.DataFrame({
        "FID": src["FID"].astype(str),
        "IID": src["IID"].astype(str),
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
    })
    out_covar = args.output_dir / "bolt_covar.tsv"
    covar.to_csv(out_covar, sep="\t", index=False)
    print(f"[simulate-pheno] wrote {out_covar}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
