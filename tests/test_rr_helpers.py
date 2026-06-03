"""Tests for the interpretive helpers on RandomRegressionLMM
(Phase 38, Step 9): variance / heritability curves, genetic correlation
surface, and functional principal components."""

import pytest
import torch

from tests.test_rr_lmm_scan import _simulate_with_planted_signal
from torchgenomics.models.rr_lmm import RandomRegressionLMM


@pytest.fixture(scope="module")
def fitted_null():
    Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
        n=80, T=8, b=3, m=10, h2=0.5, seed=90
    )
    model = RandomRegressionLMM(basis="legendre", order=2)
    nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                        t_min=t_min, t_max=t_max)
    return nf, t_min, t_max


class TestVarianceCurves:
    def test_variance_curves_positive_and_finite(self, fitted_null):
        nf, t_min, t_max = fitted_null
        eval_t = torch.linspace(t_min, t_max, 12, dtype=torch.float64)
        var_g = RandomRegressionLMM.genetic_variance_curve(nf, eval_t)
        var_e = RandomRegressionLMM.residual_variance_curve(nf, eval_t)
        assert var_g.shape == (12,)
        assert var_e.shape == (12,)
        assert torch.isfinite(var_g).all() and torch.isfinite(var_e).all()
        assert (var_g >= 0).all() and (var_e >= 0).all()

    def test_heritability_in_unit_interval(self, fitted_null):
        nf, t_min, t_max = fitted_null
        eval_t = torch.linspace(t_min, t_max, 10, dtype=torch.float64)
        h2 = RandomRegressionLMM.heritability_curve(nf, eval_t)
        assert h2.shape == (10,)
        assert (h2 >= 0).all() and (h2 <= 1).all()


class TestGeneticCorrelationSurface:
    def test_correlation_diag_is_one_and_symmetric(self, fitted_null):
        nf, t_min, t_max = fitted_null
        eval_t = torch.linspace(t_min, t_max, 8, dtype=torch.float64)
        R = RandomRegressionLMM.genetic_correlation_surface(nf, eval_t)
        assert R.shape == (8, 8)
        assert torch.allclose(R, R.T, atol=1e-10)
        diag = torch.diagonal(R)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-8)
        # Off-diagonals bounded in [-1, 1]
        assert (R >= -1.0 - 1e-8).all() and (R <= 1.0 + 1e-8).all()


class TestEigenfunctions:
    def test_fpc_shape_and_descending_eigenvalues(self, fitted_null):
        nf, t_min, t_max = fitted_null
        eval_t = torch.linspace(t_min, t_max, 20, dtype=torch.float64)
        evals, eigenfns = RandomRegressionLMM.eigenfunctions(nf, eval_t)
        b = nf.b
        assert evals.shape == (b,)
        assert eigenfns.shape == (20, b)
        # Descending
        for k in range(b - 1):
            assert evals[k] >= evals[k + 1] - 1e-10
        # All eigenvalues non-negative (K_coef is PSD)
        assert (evals >= -1e-8).all()

    def test_n_components_truncation(self, fitted_null):
        nf, t_min, t_max = fitted_null
        eval_t = torch.linspace(t_min, t_max, 15, dtype=torch.float64)
        evals, eigenfns = RandomRegressionLMM.eigenfunctions(
            nf, eval_t, n_components=2
        )
        assert evals.shape == (2,)
        assert eigenfns.shape == (15, 2)
