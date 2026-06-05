"""CLI smoke tests for ``torchgenomics twas-scan`` (observed-expression TWAS).

Both tests build a tiny in-memory fixture, write the inputs to a tmp
directory, invoke the CLI as a subprocess, and parse the output TSV.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(tmp_path: Path, n: int = 100, n_genes: int = 4,
                   causal_idx: int = 1, effect: float = 0.6, seed: int = 17):
    """Write expression.tsv, phenotype.tsv, and gene_ids.txt to tmp_path.
    Returns (expression_path, phenotype_path, gene_annotation_path)."""
    gen = torch.Generator().manual_seed(seed)
    expr = torch.randn(n, n_genes, generator=gen, dtype=torch.float64)
    y = effect * expr[:, causal_idx] + 0.1 * torch.randn(
        n, generator=gen, dtype=torch.float64,
    )

    sample_ids = [f"S{i:04d}" for i in range(n)]
    gene_ids = [f"G{i:03d}" for i in range(n_genes)]

    # expression.tsv
    expr_df = pd.DataFrame(
        expr.numpy(), index=sample_ids, columns=gene_ids,
    )
    expr_df.index.name = "sample_id"
    expr_df.reset_index().to_csv(tmp_path / "expression.tsv", sep="\t",
                                 index=False)

    # phenotype.tsv: PLINK FID/IID style.
    pheno_df = pd.DataFrame({
        "FID": sample_ids, "IID": sample_ids, "TRAIT": y.numpy(),
    })
    pheno_df.to_csv(tmp_path / "phenotype.tsv", sep="\t", index=False)

    # gene_annotation.bed (4-column)
    ann_lines = [
        f"chr{1 + (i % 22)}\t{1000 * (i + 1)}\t{1000 * (i + 1) + 500}"
        f"\t{gene_ids[i]}\tNAME_{gene_ids[i]}"
        for i in range(n_genes)
    ]
    (tmp_path / "genes.bed").write_text("\n".join(ann_lines) + "\n")

    return (
        tmp_path / "expression.tsv",
        tmp_path / "phenotype.tsv",
        tmp_path / "genes.bed",
        causal_idx,
        gene_ids,
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "torchgenomics.cli", *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        cwd=str(REPO_ROOT),
    )


def test_twas_scan_cli_observed_ols(tmp_path):
    """OLS path: no kinship, no covariates."""
    expr_path, pheno_path, _ann_path, causal_idx, gene_ids = _build_fixture(
        tmp_path,
    )
    out_path = tmp_path / "out.tsv"

    r = _run_cli(
        "twas-scan",
        "--expression", str(expr_path),
        "--phenotype", str(pheno_path),
        "--trait", "TRAIT",
        "--output", str(out_path),
        "--correction", "bonferroni",
    )
    assert r.returncode == 0, f"twas-scan failed:\nSTDOUT:\n{r.stdout}\n" \
                              f"STDERR:\n{r.stderr}"

    df = pd.read_csv(out_path, sep="\t")
    expected_cols = [
        "gene_id", "gene_name", "chr", "start", "end", "beta", "se",
        "z_twas", "p_twas", "p_adj", "n_samples_used", "n_genes_tested",
    ]
    assert list(df.columns) == expected_cols, list(df.columns)
    assert len(df) == len(gene_ids)
    # Causal gene should be among the rows with p_adj <= 0.05.
    causal_row = df[df["gene_id"] == gene_ids[causal_idx]].iloc[0]
    assert causal_row["p_adj"] <= 0.05, (
        f"causal row p_adj = {causal_row['p_adj']}"
    )


def test_twas_scan_cli_observed_lmm_with_kinship(tmp_path):
    """LMM path: kinship matrix supplied (here, identity)."""
    expr_path, pheno_path, ann_path, causal_idx, gene_ids = _build_fixture(
        tmp_path, n=80, n_genes=3, causal_idx=0, effect=0.6, seed=31,
    )
    # Write an identity kinship as .npy (matches expression sample count).
    K = np.eye(80, dtype=np.float64)
    kinship_path = tmp_path / "K.npy"
    np.save(kinship_path, K)

    out_path = tmp_path / "out_lmm.tsv"
    r = _run_cli(
        "twas-scan",
        "--expression", str(expr_path),
        "--phenotype", str(pheno_path),
        "--trait", "TRAIT",
        "--kinship", str(kinship_path),
        "--gene-annotation", str(ann_path),
        "--output", str(out_path),
        "--correction", "bh",
    )
    assert r.returncode == 0, f"twas-scan failed:\nSTDOUT:\n{r.stdout}\n" \
                              f"STDERR:\n{r.stderr}"

    df = pd.read_csv(out_path, sep="\t")
    assert len(df) == len(gene_ids)
    # gene_annotation should propagate.
    assert df["gene_name"].iloc[causal_idx] == f"NAME_{gene_ids[causal_idx]}"
    assert df["chr"].iloc[causal_idx].startswith("chr")
    # Causal gene should be significant.
    assert df.iloc[causal_idx]["p_adj"] <= 0.05
