"""CLI smoke tests for ``torchgwas combine-gwas-twas``.

Builds a 3-gene / 6-SNP synthetic fixture in tmp_path, invokes the
CLI as a subprocess, parses the output TSV, and asserts the schema +
that the planted gene (G1, strong GWAS + strong TWAS) clears the
correction threshold.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_fixture(tmp_path: Path):
    """Write gwas.tsv + twas.tsv with 3 genes on chr 1."""
    snps = ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"]
    pos = [1100, 1900, 5300, 5800, 10500, 10900]
    beta = [0.30, 0.25, 0.05, 0.02, -0.20, -0.15]
    se = [0.05] * 6
    z = [b / s for b, s in zip(beta, se)]
    p = [math.erfc(abs(zi) / math.sqrt(2.0)) for zi in z]

    gwas_df = pd.DataFrame({
        "chr": ["1"] * 6,
        "pos": pos,
        "snp": snps,
        "a1": ["A"] * 6,
        "a2": ["G"] * 6,
        "beta": beta,
        "se": se,
        "p": p,
        "n": [5000.0] * 6,
    })
    gwas_path = tmp_path / "gwas.tsv"
    gwas_df.to_csv(gwas_path, sep="\t", index=False)

    twas_df = pd.DataFrame({
        "gene_id": ["G1", "G2", "G3"],
        "gene_name": ["GENE_1", "GENE_2", "GENE_3"],
        "chr": ["1", "1", "1"],
        "start": [1000, 5000, 10000],
        "end": [2000, 6000, 11000],
        "beta": [0.21, 0.04, -0.18],
        "se": [0.05, 0.05, 0.05],
        "z_twas": [4.2, 0.8, -3.5],
        "p_twas": [2.7e-05, 0.42, 4.7e-04],
        "r2_model": [0.25, 0.05, 0.30],
        "n_cis_snps": [0, 0, 0],
    })
    twas_path = tmp_path / "twas.tsv"
    twas_df.to_csv(twas_path, sep="\t", index=False)

    return gwas_path, twas_path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "torchgwas.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False,
                          cwd=str(REPO_ROOT))


def test_combine_gwas_twas_cli_fisher(tmp_path):
    """Classical Fisher's path end-to-end."""
    gwas_path, twas_path = _build_fixture(tmp_path)
    out_path = tmp_path / "combined.tsv"

    r = _run_cli(
        "combine-gwas-twas",
        "--gwas-sumstats", str(gwas_path),
        "--twas-results", str(twas_path),
        "--method", "fisher",
        "--cis-window-bp", "200",
        "--correction", "bh",
        "--output", str(out_path),
    )
    assert r.returncode == 0, (
        f"combine-gwas-twas failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )

    df = pd.read_csv(out_path, sep="\t")
    expected_cols = [
        "gene_id", "gene_name", "chr", "start", "end",
        "p_gwas", "p_twas", "p_combined", "p_adj",
        "method", "n_snps_in_gene", "best_gwas_snp", "best_gwas_p",
        "direction_concordance", "r2_model",
    ]
    assert list(df.columns) == expected_cols, list(df.columns)
    assert len(df) == 3
    # G1 is the planted strong gene (beta_gwas > 0, beta_twas > 0).
    g1 = df[df["gene_id"] == "G1"].iloc[0]
    assert g1["p_adj"] < 0.05
    assert g1["direction_concordance"] == 1
    assert g1["method"] == "fisher"
    assert g1["n_snps_in_gene"] == 2


def test_combine_gwas_twas_cli_stouffer_r2_weighted(tmp_path):
    """Stouffer with --weights r2_model (novel R²-weighted path)."""
    gwas_path, twas_path = _build_fixture(tmp_path)
    out_path = tmp_path / "combined_stouffer_r2.tsv"

    r = _run_cli(
        "combine-gwas-twas",
        "--gwas-sumstats", str(gwas_path),
        "--twas-results", str(twas_path),
        "--method", "stouffer",
        "--cis-window-bp", "200",
        "--weights", "r2_model",
        "--correction", "bh",
        "--output", str(out_path),
    )
    assert r.returncode == 0, (
        f"combine-gwas-twas failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )

    df = pd.read_csv(out_path, sep="\t")
    assert len(df) == 3
    # r2_model column should propagate.
    g2 = df[df["gene_id"] == "G2"].iloc[0]
    assert abs(float(g2["r2_model"]) - 0.05) < 1e-9
    assert g2["method"] == "stouffer"
