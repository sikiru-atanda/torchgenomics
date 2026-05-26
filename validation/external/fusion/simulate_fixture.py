"""Simulate a measured-expression TWAS fixture for the FUSION harness.

Writes:
  data/
    expression.tsv      sample_id, gene1, ..., gene_K — already normalized.
    phenotype.tsv       FID, IID, TRAIT
    covariates.tsv      FID, IID, PC1, PC2, PC3, sex
    genes.bed           chr, start, end, gene_id, gene_name
    sim_truth.json      {seed, n_samples, n_genes, causal_idx, effect, ...}

The fixture deliberately uses already-normalised expression so the
comparison runs on the same algorithmic surface as
``torchgwas.postgwas.twas_observed_expression``. FUSION's
measured-expression mode wraps this through the same OLS Wald the TG
function reuses from ``torchgwas.models.glm.GLM``; agreement should be
at floating-point precision under matching covariates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate(output_dir: Path, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_samples = 300
    n_genes = 15
    n_covariates = 4   # 3 PCs + sex
    causal_idx = 7
    effect = 0.45

    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    gene_ids = [f"ENSG_{i:04d}" for i in range(n_genes)]
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]

    expression = rng.standard_normal((n_samples, n_genes)).astype(np.float64)
    covariates = np.column_stack([
        rng.standard_normal(n_samples).astype(np.float64),       # PC1
        rng.standard_normal(n_samples).astype(np.float64),       # PC2
        rng.standard_normal(n_samples).astype(np.float64),       # PC3
        rng.integers(0, 2, n_samples).astype(np.float64),        # sex (0/1)
    ])

    # Phenotype = β * expression[:, causal_idx] + 0.2 * PC1 + 0.1 * sex + noise.
    y = (
        effect * expression[:, causal_idx]
        + 0.2 * covariates[:, 0]
        + 0.1 * covariates[:, 3]
        + 0.25 * rng.standard_normal(n_samples).astype(np.float64)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- expression.tsv -----
    header = "sample_id\t" + "\t".join(gene_ids) + "\n"
    expr_path = output_dir / "expression.tsv"
    with open(expr_path, "w") as f:
        f.write(header)
        for i, sid in enumerate(sample_ids):
            row = [sid] + [f"{v:.10g}" for v in expression[i]]
            f.write("\t".join(row) + "\n")

    # ----- phenotype.tsv (PLINK FID/IID/TRAIT) -----
    pheno_path = output_dir / "phenotype.tsv"
    with open(pheno_path, "w") as f:
        f.write("FID\tIID\tTRAIT\n")
        for sid, val in zip(sample_ids, y):
            f.write(f"{sid}\t{sid}\t{val:.10g}\n")

    # ----- covariates.tsv (PLINK FID/IID + named columns) -----
    cov_path = output_dir / "covariates.tsv"
    with open(cov_path, "w") as f:
        f.write("FID\tIID\tPC1\tPC2\tPC3\tsex\n")
        for i, sid in enumerate(sample_ids):
            f.write(
                f"{sid}\t{sid}\t" +
                "\t".join(f"{v:.10g}" for v in covariates[i]) + "\n"
            )

    # ----- genes.bed -----
    bed_path = output_dir / "genes.bed"
    with open(bed_path, "w") as f:
        for i, (gid, gname) in enumerate(zip(gene_ids, gene_names)):
            chrom = f"chr{1 + (i % 22)}"
            start = (i + 1) * 1_000_000
            end = start + 50_000
            f.write(f"{chrom}\t{start}\t{end}\t{gid}\t{gname}\n")

    truth = {
        "seed": int(seed),
        "n_samples": int(n_samples),
        "n_genes": int(n_genes),
        "n_covariates": int(n_covariates),
        "causal_idx": int(causal_idx),
        "causal_gene_id": gene_ids[causal_idx],
        "true_effect": float(effect),
        "covariate_names": ["PC1", "PC2", "PC3", "sex"],
        "fixture_sha256": {
            "expression.tsv":  _sha256(expr_path),
            "phenotype.tsv":   _sha256(pheno_path),
            "covariates.tsv":  _sha256(cov_path),
            "genes.bed":       _sha256(bed_path),
        },
    }
    (output_dir / "sim_truth.json").write_text(
        json.dumps(truth, indent=2)
    )
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    truth = simulate(Path(args.output_dir), seed=args.seed)
    print(f"[fusion fixture] wrote {args.output_dir}")
    print(f"  n_samples={truth['n_samples']} n_genes={truth['n_genes']}")
    print(f"  causal_idx={truth['causal_idx']} effect={truth['true_effect']}")
    print("  SHA256:")
    for k, v in truth["fixture_sha256"].items():
        print(f"    {v}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
