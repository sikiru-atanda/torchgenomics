"""Tests for longitudinal-to-wide projection layer (Phase 38, Step 2)."""

import pytest
import torch

from torchgenomics.linalg.basis import legendre_basis, standardize_time
from torchgenomics.models.rr_lmm import (
    LongitudinalProjection,
    RandomRegressionLMM,
    longitudinal_to_wide,
)


def _make_balanced_long(n=20, T=8, b_true=3, seed=0):
    """Build a balanced longitudinal dataset where the true generator is a
    rank-b_true Legendre random regression model with no noise."""
    torch.manual_seed(seed)
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
    t_std, t_min, t_max = standardize_time(times)
    Phi = legendre_basis(t_std, order=b_true - 1)  # (T, b_true)
    A = torch.randn(n, b_true, dtype=torch.float64)  # individual coefficients
    Y_mat = A @ Phi.T  # (n, T)

    sample_ids = torch.arange(n).repeat_interleave(T)
    time_values = times.repeat(n)
    Y_long = Y_mat.reshape(-1)
    return Y_long, sample_ids, time_values, A, t_min, t_max, Phi


class TestLongitudinalProjectionBalanced:
    def test_returns_dataclass_with_correct_shapes(self):
        Y, ids, t, _, _, _, _ = _make_balanced_long(n=15, T=6, b_true=3)
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=2)
        assert isinstance(proj, LongitudinalProjection)
        assert proj.Y_wide.shape == (15, 3)
        assert proj.weights.shape == (15, 3, 3)
        assert len(proj.Phi_list) == 15
        assert proj.b == 3
        assert proj.basis_kind == "legendre"

    def test_recovers_known_coefficients_noiseless(self):
        # When data is generated as A @ Phi.T with no noise, the OLS
        # projection should recover A exactly.
        Y, ids, t, A_true, t_min, t_max, _ = _make_balanced_long(n=10, T=8, b_true=3)
        proj = longitudinal_to_wide(
            Y, ids, t, basis_kind="legendre", order=2, t_min=t_min, t_max=t_max
        )
        assert torch.allclose(proj.Y_wide, A_true, atol=1e-10)

    def test_round_trip_reconstruction(self):
        # Phi_i @ a_hat_i should reconstruct y_i for noiseless data.
        Y, ids, t, _, _, _, _ = _make_balanced_long(n=12, T=10, b_true=4, seed=1)
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=3)
        # Reconstruct each individual
        for i in range(int(proj.sample_index.shape[0])):
            mask = ids == proj.sample_index[i]
            y_i = Y[mask]
            recon = proj.Phi_list[i] @ proj.Y_wide[i]
            assert torch.allclose(recon, y_i, atol=1e-10)

    def test_weights_equal_gram(self):
        Y, ids, t, _, _, _, _ = _make_balanced_long(n=8, T=6, b_true=3)
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=2)
        for i, gram in enumerate(proj.gram_list):
            assert torch.allclose(proj.weights[i], gram, atol=1e-12)
            # And gram = Phi_i' Phi_i
            Phi_i = proj.Phi_list[i]
            assert torch.allclose(gram, Phi_i.T @ Phi_i, atol=1e-12)


class TestLongitudinalProjectionUnbalanced:
    def test_per_individual_T_i_recorded(self):
        # 3 individuals with 5, 7, 9 observations
        torch.manual_seed(2)
        ts = []
        ids = []
        Ys = []
        for i, T_i in enumerate([5, 7, 9]):
            t_i = torch.linspace(0.0, 50.0, T_i, dtype=torch.float64)
            ts.append(t_i)
            ids.append(torch.full((T_i,), i, dtype=torch.int64))
            Ys.append(torch.randn(T_i, dtype=torch.float64))
        Y = torch.cat(Ys)
        ids = torch.cat(ids)
        t = torch.cat(ts)
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=2)
        assert proj.n_obs_per_indiv.tolist() == [5, 7, 9]
        assert len(proj.Phi_list) == 3
        assert proj.Phi_list[0].shape == (5, 3)
        assert proj.Phi_list[1].shape == (7, 3)
        assert proj.Phi_list[2].shape == (9, 3)

    def test_rank_deficient_raises(self):
        # b = order + 1 = 4 but individual 0 only has 3 observations
        torch.manual_seed(3)
        ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        t = torch.tensor(
            [10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64
        )
        Y = torch.randn(8, dtype=torch.float64)
        with pytest.raises(ValueError, match="Rank-deficient"):
            longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=3)

    def test_unbalanced_round_trip_when_T_i_equals_b(self):
        # When T_i == b for an individual, the OLS fit must reproduce y_i
        # exactly (interpolation).
        torch.manual_seed(4)
        b = 3
        ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        t = torch.tensor([10.0, 30.0, 60.0, 5.0, 25.0, 55.0, 90.0], dtype=torch.float64)
        Y = torch.randn(7, dtype=torch.float64)
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=b - 1)
        # Individual 0 has T_i == b ⇒ must interpolate exactly
        recon0 = proj.Phi_list[0] @ proj.Y_wide[0]
        assert torch.allclose(recon0, Y[:3], atol=1e-10)


class TestProjectionConfig:
    def test_bspline_basis_supported(self):
        torch.manual_seed(5)
        n, T = 10, 12
        t = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
        ids = torch.arange(n).repeat_interleave(T)
        time_values = t.repeat(n)
        Y = torch.randn(n * T, dtype=torch.float64)
        proj = longitudinal_to_wide(
            Y, ids, time_values,
            basis_kind="bspline", n_interior_knots=3, degree=3,
        )
        # b = (n_interior + 2) + degree - 1 = 5 + 3 - 1 = 7
        assert proj.b == 7
        assert proj.Y_wide.shape == (n, 7)
        assert proj.basis_kind == "bspline"
        assert "knots" in proj.basis_params

    def test_unknown_basis_raises(self):
        Y = torch.randn(6, dtype=torch.float64)
        ids = torch.tensor([0, 0, 0, 1, 1, 1])
        t = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="unknown basis_kind"):
            longitudinal_to_wide(Y, ids, t, basis_kind="monomial")

    def test_length_mismatch_raises(self):
        Y = torch.randn(5, dtype=torch.float64)
        ids = torch.tensor([0, 0, 1, 1, 1])
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)  # wrong length
        with pytest.raises(ValueError, match="share length"):
            longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=1)

    def test_reuse_t_min_t_max(self):
        # Held-out individual whose times must align with the training reference
        torch.manual_seed(6)
        # Train: times in [0, 100]
        Y_tr, ids_tr, t_tr, _, t_min, t_max, _ = _make_balanced_long(n=8, T=6, b_true=3)
        proj_tr = longitudinal_to_wide(
            Y_tr, ids_tr, t_tr, basis_kind="legendre", order=2,
            t_min=t_min, t_max=t_max,
        )
        # Test individual measured at a subset of training times — should
        # share basis_params with training projection.
        assert proj_tr.basis_params["t_min"] == t_min
        assert proj_tr.basis_params["t_max"] == t_max


class TestRandomRegressionLMMSkeleton:
    def test_init_validates_basis(self):
        with pytest.raises(ValueError, match="unknown basis"):
            RandomRegressionLMM(basis="bogus")

    def test_init_validates_structure(self):
        with pytest.raises(ValueError, match="k_coef_structure"):
            RandomRegressionLMM(k_coef_structure="weird")

    def test_init_validates_mode(self):
        with pytest.raises(ValueError, match="mode"):
            RandomRegressionLMM(mode="bogus")

    def test_project_method_round_trip(self):
        Y, ids, t, A_true, t_min, t_max, _ = _make_balanced_long(
            n=10, T=7, b_true=3, seed=10
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        proj = model.project(Y, ids, t, t_min=t_min, t_max=t_max)
        assert proj.b == 3
        assert torch.allclose(proj.Y_wide, A_true, atol=1e-10)
