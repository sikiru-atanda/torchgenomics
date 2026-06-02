"""Tests for gene-level TWAS visualizations and λ_TWAS."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")  # headless backend before pyplot is imported

import numpy as np
import pytest
import torch

from torchgenomics.viz import (
    genomic_inflation_factor_twas,
    manhattan_twas,
    qq_twas,
)


def _fake_twas_result(n_genes: int = 50, n_strong: int = 3,
                     with_coords: bool = True, seed: int = 1):
    """Build a minimal TWASResult-like object from per-gene attributes."""
    from torchgenomics.postgwas._twas import TWASGeneResult, TWASResult

    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_genes, generator=gen, dtype=torch.float64)
    if n_strong > 0:
        z[:n_strong] = z[:n_strong] + 6.0  # strong signals at the front
    # Convert z to two-sided normal p.
    from scipy.special import erfc  # type: ignore[import-untyped]
    p = (erfc(z.abs().numpy() / math.sqrt(2.0)))
    p = np.clip(p, 1e-300, 1.0)

    genes = []
    for i in range(n_genes):
        chr_val = f"chr{1 + (i % 22)}" if with_coords else None
        start_val = (i + 1) * 1000 if with_coords else None
        genes.append(TWASGeneResult(
            gene_id=f"G{i:04d}",
            z_twas=float(z[i].item()),
            p_twas=float(p[i]),
            n_cis_snps=0,
            r2_model=None,
            top_weight_snp=None,
            beta=float(z[i].item()) * 0.1,
            se=0.1,
            chr=chr_val,
            start=start_val,
            end=(start_val + 500) if start_val is not None else None,
            gene_name=f"NAME_G{i:04d}",
        ))
    return TWASResult(genes=genes, n_genes_tested=n_genes, n_significant=n_strong)


# ===================================================================
# manhattan_twas
# ===================================================================

class TestManhattanTwas:
    def test_runs_with_coordinates(self):
        res = _fake_twas_result(n_genes=40, with_coords=True)
        fig, ax = manhattan_twas(res, label_top=3)
        assert ax is not None
        # X axis tick labels should mention chromosome numbers we used.
        xtick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any(lbl.isdigit() for lbl in xtick_labels)
        # Annotations should be present for the top-3 (strong genes).
        # We can't easily query annotation text in headless mode, but the
        # call should not raise.

    def test_runs_without_coordinates(self):
        res = _fake_twas_result(n_genes=20, with_coords=False)
        fig, ax = manhattan_twas(res, label_top=2)
        assert ax is not None
        assert ax.get_xlabel() == "Gene index"

    def test_accepts_iterable_of_gene_results(self):
        """Pass the .genes list directly (not the wrapper)."""
        res = _fake_twas_result(n_genes=15)
        fig, ax = manhattan_twas(res.genes)
        assert ax is not None

    def test_empty_gene_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            manhattan_twas([])


# ===================================================================
# qq_twas
# ===================================================================

class TestQqTwas:
    def test_runs_with_confidence_band(self):
        res = _fake_twas_result(n_genes=100, n_strong=2)
        fig, ax = qq_twas(res, confidence=True)
        assert ax is not None
        # Confidence band → fill_between → at least one PolyCollection.
        from matplotlib.collections import PolyCollection
        assert any(isinstance(c, PolyCollection) for c in ax.collections)

    def test_runs_without_confidence(self):
        res = _fake_twas_result(n_genes=50, n_strong=0)
        fig, ax = qq_twas(res, confidence=False)
        assert ax is not None

    def test_empty_p_list_raises(self):
        with pytest.raises(ValueError):
            qq_twas([])


# ===================================================================
# genomic_inflation_factor_twas (lambda_TWAS)
# ===================================================================

class TestGenomicInflationFactor:
    def test_lambda_near_one_under_null(self):
        """λ_TWAS should be ~1 when z-scores come from N(0, 1)."""
        gen = torch.Generator().manual_seed(7)
        z = torch.randn(2000, generator=gen, dtype=torch.float64)
        from torchgenomics.postgwas._twas import TWASGeneResult, TWASResult
        genes = [
            TWASGeneResult(
                gene_id=f"G{i}", z_twas=float(z[i].item()),
                p_twas=1.0, n_cis_snps=0, r2_model=None,
                top_weight_snp=None,
            )
            for i in range(2000)
        ]
        res = TWASResult(genes=genes, n_genes_tested=2000, n_significant=0)
        lam = genomic_inflation_factor_twas(res)
        # Allow ±0.10 around 1.0 (finite-sample noise at n=2000).
        assert 0.90 < lam < 1.10, f"λ_TWAS = {lam}"

    def test_lambda_above_one_under_inflation(self):
        """Inflate z-scores by 1.4×; λ_TWAS should rise."""
        gen = torch.Generator().manual_seed(11)
        z = 1.4 * torch.randn(2000, generator=gen, dtype=torch.float64)
        from torchgenomics.postgwas._twas import TWASGeneResult, TWASResult
        genes = [
            TWASGeneResult(
                gene_id=f"G{i}", z_twas=float(z[i].item()),
                p_twas=1.0, n_cis_snps=0, r2_model=None,
                top_weight_snp=None,
            )
            for i in range(2000)
        ]
        res = TWASResult(genes=genes, n_genes_tested=2000, n_significant=0)
        lam = genomic_inflation_factor_twas(res)
        # 1.4² = 1.96; expected λ ≈ 1.96 (with sampling noise).
        assert lam > 1.5

    def test_empty_gene_list_raises(self):
        with pytest.raises(ValueError):
            genomic_inflation_factor_twas([])
