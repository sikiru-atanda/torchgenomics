"""Benchmark: torchgwas PGS validation vs scipy / statsmodels reference.

Validates R², AUC, correlations, and Nagelkerke R² against established
reference implementations.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from scipy import stats as sp_stats
from statsmodels.discrete.discrete_model import Logit

from torchgwas.pgs.validation import validate_pgs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_continuous_data(seed: int, n: int = 200):
    """Generate a continuous phenotype and correlated PGS."""
    rng = np.random.RandomState(seed)
    y = rng.randn(n)
    pgs = 0.5 * y + rng.randn(n) * 0.5
    return torch.tensor(y, dtype=torch.float64), torch.tensor(pgs, dtype=torch.float64)


def _numpy_ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R² via numpy normal equations."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBenchPearsonR:
    """Pearson r: torchgwas vs scipy.stats.pearsonr."""

    def test_bench_pearson_r_matches_scipy(self):
        y, pgs = _make_continuous_data(seed=42)
        result = validate_pgs(pgs, y, trait_type="continuous")

        ref_r, _ = sp_stats.pearsonr(pgs.numpy(), y.numpy())
        assert abs(result.pearson_r - ref_r) < 1e-10, (
            f"pearson_r mismatch: ours={result.pearson_r}, scipy={ref_r}"
        )


class TestBenchSpearmanR:
    """Spearman r: torchgwas vs scipy.stats.spearmanr."""

    def test_bench_spearman_r_matches_scipy(self):
        y, pgs = _make_continuous_data(seed=42)
        result = validate_pgs(pgs, y, trait_type="continuous")

        ref_rho, _ = sp_stats.spearmanr(pgs.numpy(), y.numpy())
        assert abs(result.spearman_r - ref_rho) < 1e-6, (
            f"spearman_r mismatch: ours={result.spearman_r}, scipy={ref_rho}"
        )


class TestBenchR2:
    """R² (PGS-only model): torchgwas vs numpy OLS."""

    def test_bench_r2_matches_numpy_ols(self):
        y, pgs = _make_continuous_data(seed=42)
        result = validate_pgs(pgs, y, trait_type="continuous")

        y_np = y.numpy()
        pgs_np = pgs.numpy()
        X_np = np.column_stack([np.ones(len(y_np)), pgs_np])
        ref_r2 = _numpy_ols_r2(y_np, X_np)

        assert abs(result.r2 - ref_r2) < 1e-10, (
            f"R² mismatch: ours={result.r2}, numpy={ref_r2}"
        )


class TestBenchIncrementalR2:
    """Incremental R²: torchgwas vs manual numpy OLS difference."""

    def test_bench_incremental_r2_matches_numpy(self):
        rng = np.random.RandomState(99)
        n = 200
        cov = rng.randn(n)
        noise = rng.randn(n)
        y_np = cov * 0.7 + noise * 0.3
        pgs_np = 0.4 * y_np + rng.randn(n) * 0.4

        ones = np.ones((n, 1))
        X_cov = np.column_stack([ones, cov])
        X_full = np.column_stack([ones, cov, pgs_np])

        r2_cov = _numpy_ols_r2(y_np, X_cov)
        r2_full = _numpy_ols_r2(y_np, X_full)
        ref_incr = max(0.0, r2_full - r2_cov)

        y_t = torch.tensor(y_np, dtype=torch.float64)
        pgs_t = torch.tensor(pgs_np, dtype=torch.float64)
        cov_t = torch.tensor(cov, dtype=torch.float64).unsqueeze(-1)

        result = validate_pgs(pgs_t, y_t, covariates=cov_t, trait_type="continuous")

        assert abs(result.r2_incremental - ref_incr) < 1e-8, (
            f"incremental R² mismatch: ours={result.r2_incremental}, numpy={ref_incr}"
        )


class TestBenchAUC:
    """AUC: torchgwas vs scipy Mann-Whitney U."""

    def test_bench_auc_matches_scipy_mannwhitney(self):
        rng = np.random.RandomState(77)
        n = 200
        y_binary = np.concatenate([np.zeros(100), np.ones(100)])
        pgs_np = rng.randn(n)
        # Make cases have higher PGS on average.
        pgs_np[100:] += 1.0

        y_t = torch.tensor(y_binary, dtype=torch.float64)
        pgs_t = torch.tensor(pgs_np, dtype=torch.float64)

        result = validate_pgs(pgs_t, y_t, trait_type="binary")

        # scipy mannwhitneyu with alternative='two-sided'.
        cases = pgs_np[y_binary == 1]
        controls = pgs_np[y_binary == 0]
        u_stat, _ = sp_stats.mannwhitneyu(cases, controls, alternative="two-sided")
        n1 = len(cases)
        n0 = len(controls)
        ref_auc = u_stat / (n1 * n0)

        assert abs(result.auc - ref_auc) < 1e-10, (
            f"AUC mismatch: ours={result.auc}, scipy={ref_auc}"
        )


class TestBenchNagelkerkeR2:
    """Nagelkerke R²: torchgwas vs statsmodels Logit."""

    def test_bench_nagelkerke_r2_matches_statsmodels(self):
        rng = np.random.RandomState(55)
        n = 200
        latent = rng.randn(n)
        y_binary = (latent > 0).astype(float)
        pgs_np = 0.6 * latent + rng.randn(n) * 0.5

        y_t = torch.tensor(y_binary, dtype=torch.float64)
        pgs_t = torch.tensor(pgs_np, dtype=torch.float64)

        result = validate_pgs(pgs_t, y_t, trait_type="binary")

        # statsmodels reference.
        X_sm = sm.add_constant(pgs_np)
        logit_model = Logit(y_binary, X_sm)
        logit_result = logit_model.fit(disp=0)

        ll_full = logit_result.llf
        ll_null = logit_result.llnull
        cox_snell = 1.0 - math.exp(-2.0 / n * (ll_full - ll_null))
        max_cox_snell = 1.0 - math.exp(2.0 / n * ll_null)
        ref_nagelkerke = cox_snell / max_cox_snell if max_cox_snell > 0 else 0.0
        ref_nagelkerke = max(0.0, min(1.0, ref_nagelkerke))

        assert abs(result.nagelkerke_r2 - ref_nagelkerke) < 1e-4, (
            f"Nagelkerke R² mismatch: ours={result.nagelkerke_r2}, "
            f"statsmodels={ref_nagelkerke}"
        )


class TestBenchMAE:
    """MAE: torchgwas vs manual numpy computation."""

    def test_bench_mae_matches_numpy(self):
        y, pgs = _make_continuous_data(seed=42)
        result = validate_pgs(pgs, y, trait_type="continuous")

        y_np = y.numpy()
        pgs_np = pgs.numpy()
        X_np = np.column_stack([np.ones(len(y_np)), pgs_np])
        beta, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
        y_hat = X_np @ beta
        ref_mae = float(np.mean(np.abs(y_np - y_hat)))

        assert abs(result.mae - ref_mae) < 1e-10, (
            f"MAE mismatch: ours={result.mae}, numpy={ref_mae}"
        )


class TestBenchMultipleSeeds:
    """Cross-seed stability: R² and pearson_r match scipy/numpy on 3 seeds."""

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_bench_r2_multiple_seeds(self, seed):
        y, pgs = _make_continuous_data(seed=seed)
        result = validate_pgs(pgs, y, trait_type="continuous")

        y_np = y.numpy()
        pgs_np = pgs.numpy()

        # R² vs numpy OLS.
        X_np = np.column_stack([np.ones(len(y_np)), pgs_np])
        ref_r2 = _numpy_ols_r2(y_np, X_np)
        assert abs(result.r2 - ref_r2) < 1e-10, (
            f"[seed={seed}] R² mismatch: ours={result.r2}, numpy={ref_r2}"
        )

        # Pearson r vs scipy.
        ref_r, _ = sp_stats.pearsonr(pgs_np, y_np)
        assert abs(result.pearson_r - ref_r) < 1e-10, (
            f"[seed={seed}] pearson_r mismatch: ours={result.pearson_r}, scipy={ref_r}"
        )
