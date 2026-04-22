"""Tests for basis function module (Legendre, B-spline, 2D P-spline)."""

import pytest
import torch

from torchgwas.linalg.basis import (
    bspline_basis,
    difference_penalty,
    evaluate_basis_at,
    legendre_basis,
    place_knots,
    pspline_2d,
    standardize_time,
)

# ── Standardization ────────────────────────────────────────────────


class TestStandardizeTime:
    def test_maps_to_minus_one_one(self):
        t = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        t_std, t_min, t_max = standardize_time(t)
        assert t_min == 10.0
        assert t_max == 50.0
        assert t_std.min().item() == pytest.approx(-1.0)
        assert t_std.max().item() == pytest.approx(1.0)

    def test_reuse_reference_range(self):
        t = torch.tensor([10.0, 50.0])
        _, t_min, t_max = standardize_time(t)
        # New time at midpoint should map to 0
        t_new, _, _ = standardize_time(torch.tensor([30.0]), t_min, t_max)
        assert t_new.item() == pytest.approx(0.0)

    def test_constant_raises(self):
        with pytest.raises(ValueError):
            standardize_time(torch.tensor([5.0, 5.0, 5.0]))


# ── Legendre polynomials ───────────────────────────────────────────


class TestLegendreBasis:
    def test_orthogonality_via_gauss_quadrature(self):
        # Use 20-point Gauss-Legendre quadrature; integrals of P_i*P_j ~ delta_ij
        try:
            from numpy.polynomial.legendre import leggauss
        except ImportError:
            pytest.skip("numpy.polynomial required")
        nodes_np, weights_np = leggauss(30)
        nodes = torch.tensor(nodes_np, dtype=torch.float64)
        weights = torch.tensor(weights_np, dtype=torch.float64)
        Phi = legendre_basis(nodes, order=4)  # (30, 5)
        gram = Phi.T @ (weights.unsqueeze(1) * Phi)
        I = torch.eye(5, dtype=torch.float64)
        assert torch.allclose(gram, I, atol=1e-10)

    def test_correct_shape(self):
        t = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64)
        Phi = legendre_basis(t, order=3)
        assert Phi.shape == (11, 4)

    def test_p0_constant(self):
        t = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
        Phi = legendre_basis(t, order=2)
        # Normalized P_0 = sqrt(1/2)
        assert torch.allclose(
            Phi[:, 0], torch.full((5,), (1 / 2) ** 0.5, dtype=torch.float64)
        )

    def test_negative_order_raises(self):
        with pytest.raises(ValueError):
            legendre_basis(torch.tensor([0.0]), order=-1)

    def test_order_zero(self):
        Phi = legendre_basis(torch.tensor([0.0, 0.5]), order=0)
        assert Phi.shape == (2, 1)


# ── B-spline basis ─────────────────────────────────────────────────


class TestBSplineBasis:
    def test_partition_of_unity_cubic(self):
        # Cubic B-splines should sum to 1 inside the support
        t = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
        knots = place_knots(t, n_interior=4, kind="uniform")
        Phi = bspline_basis(t, knots, degree=3)
        row_sums = Phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(21, dtype=torch.float64), atol=1e-10)

    def test_correct_basis_count(self):
        t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        knots = place_knots(t, n_interior=3, kind="uniform")  # 5 boundary+interior knots
        Phi = bspline_basis(t, knots, degree=3)
        # n_basis = len(knots) + degree - 1 = 5 + 3 - 1 = 7
        assert Phi.shape == (11, 7)

    def test_nonnegativity(self):
        t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        knots = place_knots(t, n_interior=5, kind="quantile")
        Phi = bspline_basis(t, knots, degree=3)
        assert (Phi >= -1e-12).all()

    def test_local_support(self):
        # Each basis function has compact support: most entries are 0
        t = torch.linspace(0.0, 10.0, 100, dtype=torch.float64)
        knots = place_knots(t, n_interior=8, kind="uniform")
        Phi = bspline_basis(t, knots, degree=3)
        # No row should have all entries nonzero (except very pathological case)
        nonzero_per_row = (Phi.abs() > 1e-10).sum(dim=1)
        assert nonzero_per_row.max().item() <= 4  # cubic ⇒ up to degree+1 nonzero


# ── Knot placement ─────────────────────────────────────────────────


class TestPlaceKnots:
    def test_quantile_includes_boundaries(self):
        t = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        knots = place_knots(t, n_interior=2, kind="quantile")
        assert knots[0].item() == 0.0
        assert knots[-1].item() == 4.0
        assert len(knots) == 4  # boundaries + 2 interior

    def test_uniform_evenly_spaced(self):
        t = torch.tensor([0.0, 10.0])
        knots = place_knots(t, n_interior=3, kind="uniform")
        diffs = knots[1:] - knots[:-1]
        assert torch.allclose(diffs, torch.full_like(diffs, 2.5))

    def test_strictly_monotonic(self):
        t = torch.linspace(0.0, 1.0, 50)
        knots = place_knots(t, n_interior=5, kind="quantile")
        assert (knots[1:] > knots[:-1]).all()

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            place_knots(torch.tensor([0.0, 1.0]), n_interior=1, kind="bogus")


# ── Difference penalty ─────────────────────────────────────────────


class TestDifferencePenalty:
    def test_first_order_correct(self):
        P = difference_penalty(b=4, order=1)
        # D = [[-1,1,0,0],[0,-1,1,0],[0,0,-1,1]]; D'D is tri-diagonal
        expected = torch.tensor(
            [
                [1, -1, 0, 0],
                [-1, 2, -1, 0],
                [0, -1, 2, -1],
                [0, 0, -1, 1],
            ],
            dtype=torch.float64,
        )
        assert torch.allclose(P, expected)

    def test_symmetric_psd(self):
        P = difference_penalty(b=10, order=2)
        assert torch.allclose(P, P.T)
        evals = torch.linalg.eigvalsh(P)
        assert evals.min().item() >= -1e-10  # PSD

    def test_b_too_small_raises(self):
        with pytest.raises(ValueError):
            difference_penalty(b=2, order=2)


# ── 2D P-spline ────────────────────────────────────────────────────


class TestPspline2D:
    def test_shapes(self):
        n = 50
        torch.manual_seed(0)
        row = torch.rand(n, dtype=torch.float64) * 10
        col = torch.rand(n, dtype=torch.float64) * 8
        Phi, P_row, P_col = pspline_2d(
            row, col, n_knots_row=4, n_knots_col=3, degree=3
        )
        b = Phi.shape[1]
        assert P_row.shape == (b, b)
        assert P_col.shape == (b, b)
        assert Phi.shape[0] == n

    def test_partition_of_unity(self):
        # Tensor product of partition-of-unity bases is also partition of unity
        n = 30
        torch.manual_seed(1)
        row = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        col = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        Phi, _, _ = pspline_2d(row, col, n_knots_row=3, n_knots_col=3, degree=3)
        row_sums = Phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n, dtype=torch.float64), atol=1e-10)

    def test_penalties_psd(self):
        torch.manual_seed(2)
        row = torch.rand(40, dtype=torch.float64) * 5
        col = torch.rand(40, dtype=torch.float64) * 5
        _, P_row, P_col = pspline_2d(
            row, col, n_knots_row=4, n_knots_col=4, degree=3
        )
        for P in (P_row, P_col):
            evals = torch.linalg.eigvalsh(P)
            assert evals.min().item() >= -1e-10


# ── Re-evaluation at new time points ───────────────────────────────


class TestEvaluateBasisAt:
    def test_legendre_round_trip(self):
        t_train = torch.linspace(0.0, 100.0, 21, dtype=torch.float64)
        _, t_min, t_max = standardize_time(t_train)
        t_std_train, _, _ = standardize_time(t_train)
        Phi_train_direct = legendre_basis(t_std_train, order=3)
        Phi_train_via = evaluate_basis_at(
            t_train, "legendre", {"order": 3, "t_min": t_min, "t_max": t_max}
        )
        assert torch.allclose(Phi_train_direct, Phi_train_via, atol=1e-12)

    def test_bspline_round_trip(self):
        t_train = torch.linspace(0.0, 50.0, 25, dtype=torch.float64)
        knots = place_knots(t_train, n_interior=4, kind="quantile")
        Phi_direct = bspline_basis(t_train, knots, degree=3)
        Phi_via = evaluate_basis_at(
            t_train, "bspline", {"knots": knots, "degree": 3}
        )
        assert torch.allclose(Phi_direct, Phi_via, atol=1e-12)

    def test_unknown_basis_raises(self):
        with pytest.raises(ValueError):
            evaluate_basis_at(torch.tensor([0.0]), "monomial", {})
