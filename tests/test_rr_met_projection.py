"""Tests for project_multi_env (Phase 39, Step 1)."""

import pytest
import torch

from torchgwas.models.rr_lmm import longitudinal_to_wide
from torchgwas.models.rr_met import (
    MultiEnvLongitudinalProjection,
    project_multi_env,
)


def _simulate_balanced(n=10, T=6, E=3, seed=0):
    """Balanced (every i × every e × every t) longitudinal MET dataset.

    Time grids per env are slightly offset from each other so that the
    pooled (t_min, t_max) differs from any individual env's range — this is
    what makes the shared-time invariant testable.
    """
    torch.manual_seed(seed)
    rows = []
    for i in range(n):
        for e in range(E):
            t_grid = torch.linspace(0.0 + e * 0.5, 100.0 + e * 0.5, T)
            for tval in t_grid:
                rows.append((i, e, float(tval), torch.randn(()).item()))
    sample_ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    env_ids = torch.tensor([r[1] for r in rows], dtype=torch.long)
    time_values = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    Y = torch.tensor([r[3] for r in rows], dtype=torch.float64)
    return Y, sample_ids, env_ids, time_values


class TestShape:
    def test_legendre_returns_dataclass_with_right_shapes(self):
        Y, ids, envs, t = _simulate_balanced(n=8, T=5, E=3)
        proj = project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=2)
        assert isinstance(proj, MultiEnvLongitudinalProjection)
        assert proj.b == 3
        assert proj.E == 3
        assert proj.Y_proj.shape == (8, 3, 3)
        assert proj.sample_index.shape == (8,)
        assert proj.env_index.shape == (3,)
        assert proj.n_obs_per_indiv_per_env.shape == (8, 3)
        assert len(proj.Phi_list_per_env) == 3
        for plist in proj.Phi_list_per_env:
            assert len(plist) == 8

    def test_bspline_returns_dataclass_with_right_shapes(self):
        Y, ids, envs, t = _simulate_balanced(n=6, T=8, E=2)
        proj = project_multi_env(
            Y, ids, envs, t,
            basis_kind="bspline", n_interior_knots=2, degree=3,
        )
        # b = (n_interior + 2) + degree - 1 = 4 + 2 = 6
        assert proj.b == 6
        assert proj.E == 2
        assert proj.Y_proj.shape == (6, 6, 2)
        assert "knots" in proj.basis_params


class TestSharedTimeFrame:
    def test_legendre_t_min_t_max_pooled(self):
        Y, ids, envs, t = _simulate_balanced(n=4, T=6, E=3)
        proj = project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=2)
        # Pooled range — by construction env e has times [e*0.5, 100+e*0.5]
        assert proj.basis_params["t_min"] == pytest.approx(float(t.min().item()))
        assert proj.basis_params["t_max"] == pytest.approx(float(t.max().item()))
        # And it must NOT equal any single env's range
        assert proj.basis_params["t_max"] > 100.0  # because E>1 shifts the max

    def test_bspline_knots_pooled(self):
        Y, ids, envs, t = _simulate_balanced(n=4, T=8, E=3)
        proj = project_multi_env(
            Y, ids, envs, t,
            basis_kind="bspline", n_interior_knots=3, degree=3,
        )
        knots = proj.basis_params["knots"]
        # Boundary knots match pooled time range
        assert float(knots[0].item()) == pytest.approx(float(t.min().item()))
        assert float(knots[-1].item()) == pytest.approx(float(t.max().item()))


class TestEqualsToSingleEnv:
    def test_E1_matches_longitudinal_to_wide(self):
        """With a single env, project_multi_env should reproduce
        longitudinal_to_wide's coefficients (after the same global
        t_min/t_max standardization)."""
        torch.manual_seed(123)
        n, T = 6, 5
        rows = []
        for i in range(n):
            t_grid = torch.linspace(0.0, 50.0, T)
            for tval in t_grid:
                rows.append((i, 0, float(tval), torch.randn(()).item()))
        ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
        envs = torch.tensor([r[1] for r in rows], dtype=torch.long)
        t = torch.tensor([r[2] for r in rows], dtype=torch.float64)
        Y = torch.tensor([r[3] for r in rows], dtype=torch.float64)

        # Single env raises (we require E ≥ 1 to project but the projection
        # itself works; the model layer handles the E=1 dispatch separately).
        proj = project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=2)
        single = longitudinal_to_wide(
            Y, ids, t, basis_kind="legendre", order=2,
            t_min=proj.basis_params["t_min"], t_max=proj.basis_params["t_max"],
        )
        assert torch.allclose(proj.Y_proj[:, :, 0], single.Y_wide, atol=1e-10)


class TestNaNHandling:
    def test_missing_individual_in_one_env(self):
        Y, ids, envs, t = _simulate_balanced(n=5, T=6, E=2)
        # Drop all observations of individual 2 in env 1
        keep = ~((ids == 2) & (envs == 1))
        Y2, ids2, envs2, t2 = Y[keep], ids[keep], envs[keep], t[keep]
        proj = project_multi_env(Y2, ids2, envs2, t2, basis_kind="legendre", order=2)
        # Individual 2's column for env 1 is NaN
        assert torch.isnan(proj.Y_proj[2, :, 1]).all()
        assert int(proj.n_obs_per_indiv_per_env[2, 1].item()) == 0
        # All other cells are finite
        assert torch.isfinite(proj.Y_proj[2, :, 0]).all()
        assert torch.isfinite(proj.Y_proj[0, :, 1]).all()

    def test_n_obs_per_indiv_per_env_correct(self):
        Y, ids, envs, t = _simulate_balanced(n=4, T=6, E=2)
        proj = project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=2)
        assert (proj.n_obs_per_indiv_per_env == 6).all()


class TestSampleIndexAlignment:
    def test_sample_index_is_global_sorted_unique(self):
        Y, ids, envs, t = _simulate_balanced(n=5, T=5, E=3)
        # Permute the long-format rows
        perm = torch.randperm(Y.shape[0])
        proj = project_multi_env(
            Y[perm], ids[perm], envs[perm], t[perm],
            basis_kind="legendre", order=1,
        )
        assert torch.equal(
            proj.sample_index, torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        )
        assert torch.equal(
            proj.env_index, torch.tensor([0, 1, 2], dtype=torch.long)
        )


class TestValidation:
    def test_length_mismatch_raises(self):
        Y = torch.zeros(10, dtype=torch.float64)
        ids = torch.zeros(8, dtype=torch.long)
        envs = torch.zeros(10, dtype=torch.long)
        t = torch.zeros(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="share length"):
            project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=1)

    def test_unknown_basis_raises(self):
        Y, ids, envs, t = _simulate_balanced(n=4, T=4, E=2)
        with pytest.raises(ValueError, match="unknown basis_kind"):
            project_multi_env(Y, ids, envs, t, basis_kind="cubic", order=2)

    def test_rank_deficient_individual_raises(self):
        # An individual with only 1 observation but b=3 → rank-deficient
        torch.manual_seed(0)
        rows = [
            (0, 0, 0.0, 1.0),
            (0, 0, 1.0, 1.0),
            (0, 0, 2.0, 1.0),
            (1, 0, 0.0, 1.0),  # individual 1 has only 1 obs in env 0
            (0, 1, 0.0, 1.0),
            (0, 1, 1.0, 1.0),
            (0, 1, 2.0, 1.0),
            (1, 1, 0.0, 1.0),
            (1, 1, 1.0, 1.0),
            (1, 1, 2.0, 1.0),
        ]
        ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
        envs = torch.tensor([r[1] for r in rows], dtype=torch.long)
        t = torch.tensor([r[2] for r in rows], dtype=torch.float64)
        Y = torch.tensor([r[3] for r in rows], dtype=torch.float64)
        with pytest.raises(ValueError, match="Rank-deficient"):
            project_multi_env(Y, ids, envs, t, basis_kind="legendre", order=2)
