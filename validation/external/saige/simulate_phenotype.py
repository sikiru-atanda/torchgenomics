"""Generate SAIGE-format phenotype + covariate file from the MDP fixture.

SAIGE Step 1 takes a single tabular file with one row per sample and the
following columns:

    IID  ybin  age  sex  PC1  PC2

  - ``IID`` matches PLINK FAM IID (--sampleIDColinphenoFile=IID).
  - ``ybin`` is the binary phenotype (0/1; missing rows omitted before
    SAIGE writes the file — SAIGE accepts NA but downstream PCG can hang
    on 100% case rates).
  - ``age``, ``sex`` are synthetic covariates (so we can exercise SAIGE's
    --covarColList path); both are computable deterministically from
    sample order.
  - ``PC1``, ``PC2`` are computed from the genotype matrix via SVD
    (matched to the regenie harness simulator).

Phenotype generation strategy: Bernoulli with logit(p) = β·PC1 + β·causal
where ``causal`` is a planted dosage at 5 random SNPs and β = 0.5 each.
This gives a realistically structured signal: PC1-correlated baseline
prevalence + 5 causal hits we can spot in the SAIGE output. We do NOT
simulate a flat case/control split (which would yield β ≈ 0 everywhere)
because that's a less interesting harness check.

Why we don't reuse regenie's median-thresholded phenotype:
  - The regenie path uses thresholded EarHT (a height trait), which gives
    PC1-collinear cases. SAIGE's saddlepoint cutoff defaults to χ² > 2;
    on PC1-collinear cases the SPA path is rarely triggered, which is
    not what we want to exercise.
  - With a logit-on-causal-SNPs design, the AF tail SNPs trigger SPA
    and we exercise both the standard PQL path AND the saddlepoint tail
    correction, matching SAIGE's design regime.

Usage:
    python3 simulate_phenotype.py \
        --plink-prefix data/sample \
        --output-dir data \
        --seed 42 \
        --n-causal 5 \
        --beta 0.5
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
    """Mean-impute NaN entries column-wise."""
    G = G.astype(np.float64)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = np.isnan(col)
        if mask.any():
            valid = col[~mask]
            G[mask, j] = valid.mean() if valid.size else 0.0
    return G


def _compute_pcs(G: np.ndarray, n_pcs: int = 2) -> np.ndarray:
    """SVD-based PCA (deterministic), centered+scaled by 2af(1-af)^0.5."""
    af = G.mean(axis=0) / 2.0
    sd = np.sqrt(2.0 * af * (1.0 - af) + 1e-12)
    Z = (G - 2.0 * af) / sd
    U, _, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :n_pcs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plink-prefix", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-causal", type=int, default=5,
                    help="number of planted causal SNPs (default 5)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="per-causal-SNP additive log-odds effect (default 0.5)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Load genotype matrix -------------------------------------------------
    reader = PlinkBedReader(args.plink_prefix)
    iids = reader.sample_ids  # IIDs in BED order
    chunks = []
    for G_chunk, _ in reader.iter_chunks(chunk_size=4096):
        chunks.append(G_chunk.cpu().numpy())
    G = np.concatenate(chunks, axis=1)  # (n, m)
    G = _mean_impute(G)
    n, m = G.shape
    print(f"[simulate-pheno] loaded G: n={n}, m={m}", file=sys.stderr)

    # --- PCs (population structure) ------------------------------------------
    pcs = _compute_pcs(G, n_pcs=2)
    pc1 = pcs[:, 0]
    pc2 = pcs[:, 1]

    # --- Synthetic covariates: age + sex -------------------------------------
    # `age` is a uniform draw 30..70; `sex` is Bernoulli(0.5). Both are
    # deterministic given the seed and sample count.
    age = rng.uniform(30, 70, size=n)
    sex = rng.integers(0, 2, size=n).astype(int)

    # --- Logistic phenotype --------------------------------------------------
    # Pick `n_causal` SNPs at random with AF in [0.1, 0.4] (avoiding rare).
    af = G.mean(axis=0) / 2.0
    pool = np.where((af >= 0.1) & (af <= 0.4))[0]
    if len(pool) < args.n_causal:
        # fall back to any non-monomorphic SNP
        pool = np.where((af > 0.01) & (af < 0.99))[0]
    causal_idx = rng.choice(pool, size=args.n_causal, replace=False)
    print(f"[simulate-pheno] planted causal SNPs (idx): {sorted(causal_idx.tolist())}",
          file=sys.stderr)

    G_centered = G - 2.0 * af  # center to log-odds-additive scale
    causal_signal = G_centered[:, causal_idx].sum(axis=1) * args.beta

    # Logit linear predictor: small intercept + structure (PC1) + causal.
    # Coefficients chosen so that prevalence sits near 0.5 (balanced).
    eta = -0.05 + 0.3 * pc1 + causal_signal
    p_case = 1.0 / (1.0 + np.exp(-eta))
    ybin = (rng.random(n) < p_case).astype(int)

    n_case = int((ybin == 1).sum())
    n_ctrl = int((ybin == 0).sum())
    print(f"[simulate-pheno] cases={n_case}, controls={n_ctrl}, "
          f"prevalence={n_case / n:.3f}", file=sys.stderr)

    # --- Write SAIGE-format pheno table --------------------------------------
    # SAIGE accepts either CSV or TSV; --phenoFile auto-detects by header.
    # We write TSV with a single header line: IID ybin age sex PC1 PC2.
    pheno = pd.DataFrame({
        "IID": iids,
        "ybin": ybin,
        "age": age.round(2),
        "sex": sex,
        "PC1": pc1.round(6),
        "PC2": pc2.round(6),
    })
    out_pheno = args.output_dir / "saige_pheno.tsv"
    pheno.to_csv(out_pheno, sep="\t", index=False)
    print(f"[simulate-pheno] wrote {out_pheno}", file=sys.stderr)

    # --- Also save the planted causal SNP IDs for downstream comparison ------
    bim = pd.read_csv(Path(str(args.plink_prefix) + ".bim"),
                      sep=r"\s+", engine="python", header=None,
                      names=["chrom", "ID", "cm", "bp", "A1", "A2"])
    causal_snps = bim.iloc[causal_idx]["ID"].tolist()
    with (args.output_dir / "causal_snps.txt").open("w") as f:
        for s in causal_snps:
            f.write(f"{s}\n")
    print(f"[simulate-pheno] wrote causal_snps.txt ({len(causal_snps)} SNPs)",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
