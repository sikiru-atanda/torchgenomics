"""Tests for RandomRegressionLMM.score_chunk and the four test types
(Phase 38, Step 4)."""

import pytest
import torch

from torchgwas.linalg.basis import legendre_basis, standardize_time
from torchgwas.models.base import VariantMeta
from torchgwas.models.multi_trait_lmm import MultiTraitLMM
from torchgwas.models.rr_lmm import RandomRegressionLMM, RRScanResult


def _simulate_with_planted_signal(
    n=80, T=8, b=3, m=30, h2=0.4,
    causal_indices=(5,), effect_pattern="intercept", effect_size=2.0, seed=0,
):
    """Simulate longitudinal data with a planted causal SNP whose effect lives
    on a specific basis coefficient.

    effect_pattern:
        "intercept"   — only β_0 is nonzero (time-stable effect)
        "slope"       — only β_1 is nonzero (linear trend)
        "time_varying"— β_2 is nonzero (curvature, no constant)
        "joint"       — full b-vector with random direction
    """
    torch.manual_seed(seed)

    # GRM
    p = 200
    G_grm = torch.randn(n, p, dtype=torch.float64)
    G_grm = (G_grm - G_grm.mean(0)) / G_grm.std(0)
    K = (G_grm @ G_grm.T) / p
    K = K + 1e-4 * torch.eye(n, dtype=torch.float64)

    # Polygenic background coefficient effects
    L_K = torch.linalg.cholesky(K)
    A_back = L_K @ torch.randn(n, b, dtype=torch.float64)

    # Test SNPs (m of them)
    G_test = (torch.rand(n, m, dtype=torch.float64) > 0.5).double() + (
        torch.rand(n, m, dtype=torch.float64) > 0.5
    ).double()  # 0/1/2 dosage
    # Standardize
    G_test = (G_test - G_test.mean(0)) / G_test.std(0).clamp(min=1e-8)

    # Plant signal: each causal SNP adds a coefficient effect
    A_signal = torch.zeros(n, b, dtype=torch.float64)
    for ci in causal_indices:
        beta_eff = torch.zeros(b, dtype=torch.float64)
        if effect_pattern == "intercept":
            beta_eff[0] = effect_size
        elif effect_pattern == "slope":
            beta_eff[1] = effect_size
        elif effect_pattern == "time_varying":
            beta_eff[2] = effect_size
        elif effect_pattern == "joint":
            beta_eff = torch.randn(b, dtype=torch.float64) * effect_size
        else:
            raise ValueError(effect_pattern)
        A_signal += G_test[:, ci:ci + 1] * beta_eff.unsqueeze(0)

    A_total = A_back + A_signal

    # Time grid + Legendre basis
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
    t_std, t_min, t_max = standardize_time(times)
    Phi = legendre_basis(t_std, order=b - 1)  # (T, b)

    # Observations
    G_obs = A_total @ Phi.T  # (n, T)
    var_g = G_obs.var()
    var_e = var_g * (1.0 - h2) / h2
    Y_mat = G_obs + torch.randn(n, T, dtype=torch.float64) * var_e.sqrt()

    sample_ids = torch.arange(n).repeat_interleave(T)
    time_values = times.repeat(n)
    Y_long = Y_mat.reshape(-1)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Convert standardized G to per-individual layout for scan (n, m)
    return Y_long, X0, K, G_test, sample_ids, time_values, t_min, t_max


def _vmeta(m: int) -> VariantMeta:
    return VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1, m + 1)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


class TestScanShapesAndMetadata:
    def test_returns_rrscanresult(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=40, T=6, b=3, m=10, seed=1
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(10))
        assert isinstance(res, RRScanResult)
        assert len(res) == 10
        assert res.beta.shape == (10, 3)
        assert res.se.shape == (10, 3)
        assert res.Var_beta.shape == (10, 3, 3)
        assert res.stat_joint.shape == (10,)
        assert res.p_joint.shape == (10,)
        assert res.stat_intercept.shape == (10,)
        assert res.stat_slope.shape == (10,)
        assert res.stat_time_varying.shape == (10,)
        # stat/p aliases point to joint
        assert torch.equal(res.stat, res.stat_joint)
        assert torch.equal(res.p, res.p_joint)
        # Metadata
        assert res.basis_kind == "legendre"
        assert res.b == 3

    def test_pvalues_in_range(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=50, T=6, b=3, m=15, seed=2
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(15))
        for p in (res.p_joint, res.p_intercept, res.p_slope, res.p_time_varying):
            assert (p >= 0).all() and (p <= 1).all()

    def test_b_equals_2_slope_well_defined(self):
        # b = 2 means the slope test exists and equals β_1 alone
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=40, T=5, b=2, m=8, seed=3
        )
        model = RandomRegressionLMM(basis="legendre", order=1)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(8))
        # When b=2, time-varying df = 1 = slope; values should be identical
        assert torch.allclose(res.stat_slope, res.stat_time_varying, atol=1e-10)


class TestEquivalenceWithMultiTraitLMM:
    def test_joint_test_matches_mvlmm(self):
        # The joint χ²(b) test in RR should equal the joint Wald test in
        # MultiTraitLMM applied to the hand-projected coefficients.
        from torchgwas.models.rr_lmm import longitudinal_to_wide

        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=50, T=7, b=3, m=12, seed=4
        )
        proj = longitudinal_to_wide(
            Y, ids, t, basis_kind="legendre", order=2, t_min=t_min, t_max=t_max
        )
        mvlmm = MultiTraitLMM()
        nf_mv = mvlmm.fit_null(proj.Y_wide, X0, K)
        res_mv = mvlmm.score_chunk(G, nf_mv, _vmeta(12))

        rr = RandomRegressionLMM(basis="legendre", order=2)
        nf_rr = rr.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res_rr = rr.score_chunk(G, nf_rr, _vmeta(12))

        assert torch.allclose(res_rr.beta, res_mv.beta, atol=1e-8)
        assert torch.allclose(res_rr.stat_joint, res_mv.stat, atol=1e-8)
        assert torch.allclose(res_rr.p_joint, res_mv.p, atol=1e-10)


class TestPlantedSignalRecovery:
    def test_intercept_pattern_detected_by_intercept_test(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=150, T=10, b=3, m=40, h2=0.5,
            causal_indices=(7,), effect_pattern="intercept",
            effect_size=2.5, seed=10,
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(40))
        # Causal SNP should have small intercept p-value
        assert res.p_intercept[7].item() < 1e-4
        # And small joint p-value
        assert res.p_joint[7].item() < 1e-3

    def test_slope_pattern_detected_by_slope_test(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=150, T=10, b=3, m=40, h2=0.5,
            causal_indices=(11,), effect_pattern="slope",
            effect_size=2.5, seed=11,
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(40))
        # Causal SNP detected by slope test
        assert res.p_slope[11].item() < 1e-4
        # Time-varying test also detects (since slope is part of time-varying)
        assert res.p_time_varying[11].item() < 1e-3
        # Intercept p should be larger (slope is orthogonal to intercept under
        # Legendre normalization)
        assert res.p_intercept[11].item() > res.p_slope[11].item()

    def test_time_varying_pattern_detected_by_tv_test(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=200, T=12, b=3, m=40, h2=0.6,
            causal_indices=(15,), effect_pattern="time_varying",
            effect_size=2.5, seed=12,
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(40))
        assert res.p_time_varying[15].item() < 1e-3
        # Pure quadratic effect — intercept and slope should not be significant
        assert res.p_intercept[15].item() > 0.01

    def test_null_calibration_joint(self):
        # Null SNPs (no causal): joint test should be approximately uniform
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=100, T=8, b=3, m=200, h2=0.4,
            causal_indices=(), effect_size=0.0, seed=20,
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(200))
        # Approx 5% should fall below 0.05
        fpr = (res.p_joint < 0.05).float().mean().item()
        assert 0.01 < fpr < 0.12, f"FPR={fpr}, expected ~0.05"


class TestScanValidation:
    def test_non_rr_nullfit_raises(self):
        # mvLMM NullFit lacks K_coef → score_chunk should refuse
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=30, T=5, b=2, m=5, causal_indices=(), seed=30
        )
        from torchgwas.models.rr_lmm import longitudinal_to_wide
        proj = longitudinal_to_wide(Y, ids, t, basis_kind="legendre", order=1)
        mvlmm = MultiTraitLMM()
        nf_mv = mvlmm.fit_null(proj.Y_wide, X0, K)

        rr = RandomRegressionLMM(basis="legendre", order=1)
        with pytest.raises(ValueError, match="random regression"):
            rr.score_chunk(G, nf_mv, _vmeta(5))

    def test_unsupported_test_raises(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=30, T=5, b=2, m=5, causal_indices=(), seed=31
        )
        rr = RandomRegressionLMM(basis="legendre", order=1)
        nf = rr.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                         t_min=t_min, t_max=t_max)
        with pytest.raises(ValueError, match="wald"):
            rr.score_chunk(G, nf, _vmeta(5), test="lrt")
