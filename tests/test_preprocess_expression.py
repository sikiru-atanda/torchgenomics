"""Tests for torchgenomics.preprocess.expression: rank-INT, quantile-norm,
PEER residualization helpers.

The three helpers are opt-in preprocessing steps for observed-expression
TWAS. rank-INT and quantile-norm are pure-Python; peer_residualize wraps
the R ``peer`` package via subprocess and skips when R/peer is not
installed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
import torch

from torchgenomics.preprocess import (
    inverse_normal_transform,
    peer_residualize,
    quantile_normalize,
)

# ===================================================================
# inverse_normal_transform
# ===================================================================

class TestInverseNormalTransform:
    """Blom-style rank-based inverse normal transform."""

    def test_recovers_approximate_normality(self):
        """A skewed input (exponential) becomes approximately normal
        after INT. Check that mean is near 0 and SD is near 1."""
        gen = torch.Generator().manual_seed(0)
        # Exponential(1) via inverse-CDF on Uniform(0, 1).
        u = torch.rand(500, generator=gen, dtype=torch.float64)
        x = -torch.log(u)
        rint = inverse_normal_transform(x)
        assert abs(float(rint.mean().item())) < 0.05
        assert abs(float(rint.std().item()) - 1.0) < 0.05

    def test_preserves_rank_order(self):
        """argsort(input) == argsort(output) for unique values."""
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(50, generator=gen, dtype=torch.float64)
        rint = inverse_normal_transform(x)
        assert torch.equal(torch.argsort(x), torch.argsort(rint))

    def test_handles_ties_with_average_rank(self):
        """Tied input entries get the same INT output (average-rank
        tie-breaking)."""
        x = torch.tensor([5.0, 5.0, 2.0, 8.0, 1.0], dtype=torch.float64)
        rint = inverse_normal_transform(x)
        assert abs(rint[0].item() - rint[1].item()) < 1e-12

    def test_applies_per_column_for_matrix_input(self):
        """2-D input: each column transformed independently."""
        gen = torch.Generator().manual_seed(2)
        X = torch.randn(20, 4, generator=gen, dtype=torch.float64)
        rint = inverse_normal_transform(X)
        assert rint.shape == X.shape
        # Each column should have rank order preserved.
        for j in range(X.shape[1]):
            assert torch.equal(torch.argsort(X[:, j]), torch.argsort(rint[:, j]))
        # Per-column mean should be near zero (under Phi^{-1} symmetry).
        assert float(rint.mean(0).abs().max().item()) < 1e-10

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            inverse_normal_transform(torch.randn(2, 3, 4))


# ===================================================================
# quantile_normalize
# ===================================================================

class TestQuantileNormalize:
    """Column-wise quantile normalization to a common reference."""

    def test_column_means_match_after_normalization(self):
        """Every output column has the same mean (within FP precision)."""
        gen = torch.Generator().manual_seed(3)
        # Columns with intentionally different distributions.
        X = torch.cat([
            torch.randn(30, 1, generator=gen, dtype=torch.float64) * 2.0,
            torch.randn(30, 1, generator=gen, dtype=torch.float64) + 5.0,
            torch.randn(30, 1, generator=gen, dtype=torch.float64) * 0.5,
        ], dim=1)
        qn = quantile_normalize(X)
        means = qn.mean(0)
        assert float((means - means[0]).abs().max().item()) < 1e-12

    def test_preserves_within_column_rank(self):
        """Within each column the rank order is preserved."""
        gen = torch.Generator().manual_seed(4)
        X = torch.randn(30, 4, generator=gen, dtype=torch.float64)
        qn = quantile_normalize(X)
        for j in range(X.shape[1]):
            assert torch.equal(torch.argsort(X[:, j]), torch.argsort(qn[:, j]))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            quantile_normalize(torch.randn(10))


# ===================================================================
# peer_residualize (R subprocess wrapper)
# ===================================================================

def _peer_available() -> bool:
    """True iff Rscript is on PATH AND the R 'peer' package is installed."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    probe = subprocess.run(
        [rscript, "-e",
         "suppressPackageStartupMessages(library(peer)); "
         'cat(as.character(packageVersion("peer")))'],
        capture_output=True, text=True, check=False,
    )
    return probe.returncode == 0


class TestPeerResidualize:
    """PEER residualization (R subprocess wrapper)."""

    def test_n_factors_validation(self):
        """n_factors must be positive and at most n_samples."""
        gen = torch.Generator().manual_seed(5)
        expr = torch.randn(50, 20, generator=gen, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be positive"):
            peer_residualize(expr, n_factors=0)
        with pytest.raises(ValueError, match="cannot exceed n_samples"):
            peer_residualize(expr, n_factors=51)

    def test_shape_validation(self):
        """expression must be 2-D; covariates must match n_samples."""
        with pytest.raises(ValueError, match="must be 2-D"):
            peer_residualize(torch.randn(10), n_factors=2)
        gen = torch.Generator().manual_seed(6)
        expr = torch.randn(20, 10, generator=gen, dtype=torch.float64)
        bad_cov = torch.randn(15, 3, generator=gen, dtype=torch.float64)
        with pytest.raises(ValueError, match="covariates must be"):
            peer_residualize(expr, n_factors=2, covariates=bad_cov)

    def test_raises_when_R_unavailable(self):
        """Without R + peer, peer_residualize must raise a clear error
        (rather than silently doing nothing)."""
        if _peer_available():
            pytest.skip("R + peer is installed; skipping unavailability test.")
        gen = torch.Generator().manual_seed(7)
        expr = torch.randn(20, 10, generator=gen, dtype=torch.float64)
        with pytest.raises(RuntimeError, match="Rscript not found|peer"):
            peer_residualize(expr, n_factors=2)

    @pytest.mark.skipif(
        not _peer_available(),
        reason="R + peer not installed; subprocess wrapper untested live."
    )
    def test_residualizes_known_covariate(self):
        """When a covariate is provided, the residual should be ~zero-
        correlated with that covariate."""
        gen = torch.Generator().manual_seed(8)
        n, n_genes = 60, 30
        cov = torch.randn(n, 1, generator=gen, dtype=torch.float64)
        # Build expression with a strong cov-driven component.
        loadings = torch.randn(1, n_genes, generator=gen, dtype=torch.float64)
        noise = torch.randn(n, n_genes, generator=gen, dtype=torch.float64) * 0.1
        expr = cov @ loadings + noise
        residuals, summary = peer_residualize(
            expr, n_factors=5, covariates=cov, max_iterations=200,
        )
        assert residuals.shape == expr.shape
        # After residualisation, per-gene correlation with cov should be
        # markedly smaller than in the raw input.
        def per_gene_abs_corr(X, c):
            c_z = (c - c.mean()) / c.std()
            X_z = (X - X.mean(0)) / X.std(0)
            return (X_z * c_z.reshape(-1, 1)).mean(0).abs()
        raw_corr = per_gene_abs_corr(expr, cov.squeeze())
        res_corr = per_gene_abs_corr(residuals, cov.squeeze())
        assert float(res_corr.mean().item()) < 0.3 * float(raw_corr.mean().item())
        assert summary.get("converged", False) is True
