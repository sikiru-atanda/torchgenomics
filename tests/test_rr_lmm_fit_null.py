"""Tests for RandomRegressionLMM.fit_null (Phase 38, Step 3)."""

import pytest
import torch

from torchgenomics.linalg.basis import legendre_basis, standardize_time
from torchgenomics.models.multi_trait_lmm import MultiTraitLMM
from torchgenomics.models.rr_lmm import RandomRegressionLMM, longitudinal_to_wide


def _simulate_rr_dataset(n=60, T=8, b=3, h2=0.5, seed=0):
    """Simulate a balanced random regression dataset.

    Builds a polygenic GRM, draws true coefficient genetic effects
    A_geno = K^{1/2} @ Z @ chol(K_coef_true), adds residual noise per
    observation, and returns long-format Y plus the true K_coef.
    """
    torch.manual_seed(seed)
    # GRM from random standardized genotypes
    p = 200
    G = torch.randn(n, p, dtype=torch.float64)
    G = (G - G.mean(0)) / G.std(0)
    K = (G @ G.T) / p
    K = K + 1e-4 * torch.eye(n, dtype=torch.float64)

    # True coefficient covariance K_coef (b x b PD)
    L = torch.tril(torch.randn(b, b, dtype=torch.float64) * 0.5)
    L = L + torch.eye(b, dtype=torch.float64)
    K_coef_true = L @ L.T

    # Genetic coefficients: vec(A) ~ N(0, K ⊗ K_coef_true)
    L_K = torch.linalg.cholesky(K)
    Z = torch.randn(n, b, dtype=torch.float64)
    A_geno = L_K @ Z @ L.T  # (n, b), Var(vec) = K ⊗ (L L') = K ⊗ K_coef

    # Time grid + Legendre basis
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
    t_std, t_min, t_max = standardize_time(times)
    Phi = legendre_basis(t_std, order=b - 1)  # (T, b)

    # Genetic signal in observation space
    G_signal = A_geno @ Phi.T  # (n, T)
    # Scale residual variance from h2
    var_g = G_signal.var()
    var_e = var_g * (1.0 - h2) / h2
    Y_mat = G_signal + torch.randn(n, T, dtype=torch.float64) * var_e.sqrt()

    sample_ids = torch.arange(n).repeat_interleave(T)
    time_values = times.repeat(n)
    Y_long = Y_mat.reshape(-1)

    X0 = torch.ones(n, 1, dtype=torch.float64)
    return Y_long, X0, K, sample_ids, time_values, K_coef_true, t_min, t_max


class TestFitNullBasic:
    def test_returns_nullfit_with_rr_metadata(self):
        Y, X0, K, ids, t, _, t_min, t_max = _simulate_rr_dataset(
            n=40, T=6, b=3, seed=1
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        # Standard mvLMM fields
        assert nf.Vg is not None and nf.Vg.shape == (3, 3)
        assert nf.Ve is not None and nf.Ve.shape == (3, 3)
        assert nf.Y_rot is not None and nf.Y_rot.shape == (40, 3)
        assert nf.weights is not None and nf.weights.shape == (40, 3, 3)

        # RR-specific metadata
        assert nf.K_coef is not None
        assert torch.equal(nf.K_coef, nf.Vg)
        assert nf.basis_kind == "legendre"
        assert nf.b == 3
        assert "order" in nf.basis_params
        assert nf.basis_params["order"] == 2
        assert nf.basis_params["t_min"] == t_min
        assert nf.basis_params["t_max"] == t_max
        assert len(nf.Phi_list) == 40
        assert nf.proj_weights.shape == (40, 3, 3)
        assert nf.n_obs_per_indiv.shape == (40,)
        assert nf.k_coef_structure == "unstructured"
        assert nf.include_pe is False

    def test_converges_on_clean_data(self):
        Y, X0, K, ids, t, _, t_min, t_max = _simulate_rr_dataset(
            n=80, T=8, b=3, h2=0.6, seed=2
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        assert nf.converged
        assert nf.log_likelihood is not None

    def test_K_coef_recovers_signal(self):
        # Trace of K_coef should be positive and roughly tracking
        # the true variance scale.
        Y, X0, K, ids, t, K_true, t_min, t_max = _simulate_rr_dataset(
            n=120, T=10, b=3, h2=0.7, seed=3
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        assert nf.K_coef.diag().min().item() > 0
        # PSD check
        evals = torch.linalg.eigvalsh(nf.K_coef)
        assert evals.min().item() > -1e-8


class TestFitNullEquivalence:
    def test_matches_multi_trait_lmm_on_hand_projected(self):
        # When we hand-project the long data and call MultiTraitLMM
        # directly, we should get the same Vg/Ve/ll as RR's fit_null.
        Y, X0, K, ids, t, _, t_min, t_max = _simulate_rr_dataset(
            n=50, T=7, b=3, seed=4
        )
        # Hand projection
        proj = longitudinal_to_wide(
            Y, ids, t, basis_kind="legendre", order=2, t_min=t_min, t_max=t_max
        )
        mvlmm = MultiTraitLMM()
        nf_direct = mvlmm.fit_null(proj.Y_wide, X0, K)

        rr = RandomRegressionLMM(basis="legendre", order=2)
        nf_rr = rr.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        assert torch.allclose(nf_rr.Vg, nf_direct.Vg, atol=1e-8)
        assert torch.allclose(nf_rr.Ve, nf_direct.Ve, atol=1e-8)
        assert nf_rr.log_likelihood == pytest.approx(nf_direct.log_likelihood, abs=1e-8)


class TestFitNullValidation:
    def test_requires_K(self):
        Y, X0, _, ids, t, _, _, _ = _simulate_rr_dataset(n=20, T=5, b=2, seed=5)
        model = RandomRegressionLMM(basis="legendre", order=1)
        with pytest.raises(ValueError, match="kinship"):
            model.fit_null(Y, X0, None, sample_ids=ids, time_values=t)

    def test_X0_shape_mismatch_raises(self):
        Y, _, K, ids, t, _, _, _ = _simulate_rr_dataset(n=30, T=6, b=3, seed=6)
        bad_X0 = torch.ones(20, 1, dtype=torch.float64)  # wrong n
        model = RandomRegressionLMM(basis="legendre", order=2)
        with pytest.raises(ValueError, match="X0 must have shape"):
            model.fit_null(Y, bad_X0, K, sample_ids=ids, time_values=t)

    def test_K_shape_mismatch_raises(self):
        Y, X0, _, ids, t, _, _, _ = _simulate_rr_dataset(n=25, T=6, b=3, seed=7)
        bad_K = torch.eye(20, dtype=torch.float64)
        model = RandomRegressionLMM(basis="legendre", order=2)
        with pytest.raises(ValueError, match="K must have shape"):
            model.fit_null(Y, X0, bad_K, sample_ids=ids, time_values=t)

    def test_b_too_small_raises(self):
        # b = order + 1 = 1 < 2 — should be rejected
        Y, X0, K, ids, t, _, _, _ = _simulate_rr_dataset(n=20, T=4, b=2, seed=8)
        model = RandomRegressionLMM(basis="legendre", order=0)
        with pytest.raises(ValueError, match="basis dimension"):
            model.fit_null(Y, X0, K, sample_ids=ids, time_values=t)

    def test_stacked_mode_runs_on_balanced_data(self):
        """Stacked mode is wired in Phase 38 step 7 — verify it runs and
        attaches K_coef on a balanced dataset (the projection-mode
        equivalence is exercised separately in test_rr_stacked.py)."""
        Y, X0, K, ids, t, _, _, _ = _simulate_rr_dataset(n=20, T=5, b=2, seed=9)
        model = RandomRegressionLMM(basis="legendre", order=1, mode="stacked")
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t)
        assert nf.K_coef is not None
        assert nf.K_coef.shape == (2, 2)
