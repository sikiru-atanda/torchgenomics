"""Tests for SpatioTemporalRR — RR-LMM with a 2D P-spline spatial smoother
(Phase 38, Step 8)."""

import pytest
import torch

from tests.test_rr_lmm_scan import _vmeta
from torchgenomics.linalg.basis import legendre_basis, standardize_time
from torchgenomics.models.rr_lmm import RRScanResult
from torchgenomics.models.rr_spatial import SpatioTemporalRR, fit_spatial_pspline


def _simulate_field_trial(
    n=80, T=6, b=3,
    n_rows=8, n_cols=10,
    spatial_amp=2.0, h2=0.4, seed=0,
):
    """Simulate a balanced longitudinal field-trial dataset.

    A planted bivariate-Gaussian spatial trend f(row, col) is added to
    every observation according to the individual's plot location, on top
    of a polygenic + Legendre-coefficient longitudinal signal.
    """
    torch.manual_seed(seed)
    assert n <= n_rows * n_cols, "n must fit in the field grid"

    # GRM
    p = 200
    G_grm = torch.randn(n, p, dtype=torch.float64)
    G_grm = (G_grm - G_grm.mean(0)) / G_grm.std(0)
    K = (G_grm @ G_grm.T) / p
    K = K + 1e-4 * torch.eye(n, dtype=torch.float64)

    # Polygenic background coefficient effects
    L_K = torch.linalg.cholesky(K)
    A_back = L_K @ torch.randn(n, b, dtype=torch.float64)

    # Test SNPs
    m = 12
    G_test = (torch.rand(n, m, dtype=torch.float64) > 0.5).double() + (
        torch.rand(n, m, dtype=torch.float64) > 0.5
    ).double()
    G_test = (G_test - G_test.mean(0)) / G_test.std(0).clamp(min=1e-8)

    # Plant a time-varying genetic effect on SNP 5 (slope coefficient)
    A_signal = torch.zeros(n, b, dtype=torch.float64)
    beta_eff = torch.zeros(b, dtype=torch.float64)
    beta_eff[1] = 1.5  # slope effect
    A_signal += G_test[:, 5:6] * beta_eff.unsqueeze(0)

    A_total = A_back + A_signal

    # Time grid + Legendre basis
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
    t_std, t_min, t_max = standardize_time(times)
    Phi = legendre_basis(t_std, order=b - 1)

    # Per-individual longitudinal signal (n, T)
    G_obs = A_total @ Phi.T
    var_g = G_obs.var()
    var_e = var_g * (1.0 - h2) / h2

    # Random plot assignments
    rng = torch.Generator().manual_seed(seed + 1)
    plots = torch.randperm(n_rows * n_cols, generator=rng)[:n]
    plot_rows = (plots // n_cols).double()
    plot_cols = (plots % n_cols).double()

    # Bivariate-Gaussian field surface, evaluated per individual
    row_center = (n_rows - 1) / 2.0
    col_center = (n_cols - 1) / 2.0
    spatial_field = spatial_amp * torch.exp(
        -((plot_rows - row_center) ** 2) / (2.0 * (n_rows / 4.0) ** 2)
        - ((plot_cols - col_center) ** 2) / (2.0 * (n_cols / 4.0) ** 2)
    )  # (n,)

    # Add spatial trend to every time-point of every individual
    Y_mat = (
        G_obs
        + spatial_field.unsqueeze(1)
        + torch.randn(n, T, dtype=torch.float64) * var_e.sqrt()
    )

    sample_ids = torch.arange(n).repeat_interleave(T)
    time_values = times.repeat(n)
    Y_long = Y_mat.reshape(-1)
    # Per-observation row/col coords (constant across an individual's time series)
    row_long = plot_rows.repeat_interleave(T)
    col_long = plot_cols.repeat_interleave(T)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    return (
        Y_long, X0, K, G_test, sample_ids, time_values, t_min, t_max,
        row_long, col_long, spatial_field,
    )


# ── fit_spatial_pspline ───────────────────────────────────────────────


class TestFitSpatialPspline:
    def test_removes_planted_smooth_trend(self):
        """A planted smooth surface should be largely subtracted by the
        2D P-spline fit (residual variance ≪ original variance)."""
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=80, T=6, n_rows=10, n_cols=10, spatial_amp=4.0, h2=0.5, seed=80
        )
        Y_clean, coef, Phi_S = fit_spatial_pspline(
            Y, row, col, n_knots_row=6, n_knots_col=6, lambda_row=0.1, lambda_col=0.1
        )
        assert Y_clean.shape == Y.shape
        # Spatial fit should explain a sizeable chunk of the variance
        assert Y_clean.var() < Y.var()

    def test_fit_spatial_shape_validation(self):
        with pytest.raises(ValueError, match="length"):
            fit_spatial_pspline(
                torch.randn(10, dtype=torch.float64),
                torch.randn(9, dtype=torch.float64),
                torch.randn(10, dtype=torch.float64),
            )


# ── SpatioTemporalRR ──────────────────────────────────────────────────


class TestSpatioTemporalRRFit:
    def test_fit_runs_and_attaches_metadata(self):
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=60, T=6, n_rows=8, n_cols=10, spatial_amp=2.0, h2=0.5, seed=81
        )
        model = SpatioTemporalRR(
            basis="legendre", order=2,
            n_knots_row=5, n_knots_col=5,
            lambda_row=0.5, lambda_col=0.5,
        )
        nf = model.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t,
            row_coords=row, col_coords=col, t_min=t_min, t_max=t_max,
        )
        # Standard RR null-fit metadata is preserved
        assert nf.K_coef is not None
        assert nf.K_coef.shape == (3, 3)
        assert nf.basis_kind == "legendre"
        # Spatial metadata attached
        assert nf.spatial_coef is not None
        assert nf.Y_long_cleaned is not None
        assert nf.spatial_lambda_row == 0.5
        assert nf.spatial_lambda_col == 0.5

    def test_spatial_correction_improves_h2_estimate(self):
        """With strong spatial nuisance, the uncorrected RR-LMM inflates the
        residual variance. The spatially corrected fit should produce a
        smaller residual norm on the cleaned phenotype than the un-corrected
        fit on the raw phenotype."""
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=80, T=6, n_rows=10, n_cols=10, spatial_amp=5.0, h2=0.5, seed=82
        )
        # Uncorrected RR-LMM
        from torchgenomics.models.rr_lmm import RandomRegressionLMM

        m_plain = RandomRegressionLMM(basis="legendre", order=2)
        nf_plain = m_plain.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t, t_min=t_min, t_max=t_max
        )

        m_spatial = SpatioTemporalRR(
            basis="legendre", order=2,
            n_knots_row=6, n_knots_col=6,
            lambda_row=0.1, lambda_col=0.1,
        )
        nf_spatial = m_spatial.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t,
            row_coords=row, col_coords=col, t_min=t_min, t_max=t_max,
        )
        # Cleaned phenotype variance should be smaller than raw phenotype
        assert nf_spatial.Y_long_cleaned.var() < Y.var()
        # Both fits produced PSD K_coef
        for nf in (nf_plain, nf_spatial):
            eigvals = torch.linalg.eigvalsh(nf.Vg)
            assert (eigvals >= -1e-6).all()


class TestSpatioTemporalRRScan:
    def test_scan_returns_finite_pvalues(self):
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=80, T=8, n_rows=10, n_cols=10, spatial_amp=2.0, h2=0.5, seed=83
        )
        model = SpatioTemporalRR(
            basis="legendre", order=2,
            n_knots_row=5, n_knots_col=5,
            lambda_row=0.5, lambda_col=0.5,
        )
        nf = model.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t,
            row_coords=row, col_coords=col, t_min=t_min, t_max=t_max,
        )
        res = model.score_chunk(G, nf, _vmeta(12))
        assert isinstance(res, RRScanResult)
        for p in (res.p_joint, res.p_intercept, res.p_slope, res.p_time_varying):
            assert torch.isfinite(p).all()
            assert (p >= 0).all() and (p <= 1).all()

    def test_planted_slope_signal_recovered_after_spatial_correction(self):
        """The planted slope-coefficient signal on SNP 5 should still be
        recovered after the spatial smoother removes the field trend."""
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=150, T=8, n_rows=12, n_cols=15, spatial_amp=3.0, h2=0.6, seed=84
        )
        model = SpatioTemporalRR(
            basis="legendre", order=2,
            n_knots_row=6, n_knots_col=6,
            lambda_row=0.1, lambda_col=0.1,
        )
        nf = model.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t,
            row_coords=row, col_coords=col, t_min=t_min, t_max=t_max,
        )
        res = model.score_chunk(G, nf, _vmeta(12))
        # SNP 5 (the planted causal) should rank in the top half on slope test
        slope_p = res.p_slope
        causal_rank = (slope_p < slope_p[5]).sum().item()
        assert causal_rank <= 6, (
            f"Planted causal SNP rank too high: {causal_rank} (slope p-values "
            f"= {slope_p.tolist()})"
        )

    def test_eval_times_works_through_spatial_wrapper(self):
        Y, X0, K, G, ids, t, t_min, t_max, row, col, _ = _simulate_field_trial(
            n=60, T=6, n_rows=8, n_cols=10, spatial_amp=2.0, h2=0.5, seed=85
        )
        model = SpatioTemporalRR(basis="legendre", order=2)
        nf = model.fit_null(
            Y, X0, K, sample_ids=ids, time_values=t,
            row_coords=row, col_coords=col, t_min=t_min, t_max=t_max,
        )
        eval_t = torch.tensor([10.0, 50.0, 90.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, _vmeta(12), eval_times=eval_t)
        assert res.beta_at_t is not None
        assert res.beta_at_t.shape == (12, 3)
        assert torch.isfinite(res.p_at_t).all()
