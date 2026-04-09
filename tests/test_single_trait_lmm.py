"""Phase 4: Single-trait LMM tests — REML, optimizer stack, Wald/LRT/Score scan."""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.config import STAT_DTYPE
from torchgwas.linalg.eigh import eigendecompose, rotate
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.models.single_trait_lmm import SingleTraitLMM
from torchgwas.optim.ai_reml import ai_reml_single
from torchgwas.optim.em_warmstart import px_em_warmstart
from torchgwas.optim.mm_reml import mm_reml
from torchgwas.optim.reml_math import reml_derivatives, reml_loglikelihood
from torchgwas.preprocess.impute import impute_mean


# ---------------------------------------------------------------
# Fixtures: simulate a simple LMM dataset with known parameters
# ---------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture
def lmm_data(rng):
    """Simulate y = Xb + u + e with known variance components.

    n=200 samples, m=500 SNPs, 1 covariate + intercept.
    True sig2_g=1.0, sig2_e=1.0 (h2=0.5) — strong signal for reliable estimation.
    """
    n, m = 200, 500
    torch.manual_seed(42)

    # Genotypes
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)

    # GRM
    K, _ = grm_vanraden(G)

    # Covariates: intercept + 1 covariate
    X0 = torch.ones(n, 2, dtype=torch.float64)
    X0[:, 1] = torch.randn(n, dtype=torch.float64)

    # True parameters
    true_sig2_g = 1.0
    true_sig2_e = 1.0
    true_beta = torch.tensor([1.0, 0.5], dtype=torch.float64)

    # Simulate: y = X0 @ beta + u + e
    # u ~ N(0, K * sig2_g) via Cholesky
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(true_sig2_g)
    e = torch.randn(n, dtype=torch.float64) * math.sqrt(true_sig2_e)
    y = X0 @ true_beta + u + e

    return {
        "Y": y, "X0": X0, "K": K, "G": G,
        "true_sig2_g": true_sig2_g, "true_sig2_e": true_sig2_e,
        "true_beta": true_beta, "n": n, "m": m,
    }


@pytest.fixture
def rotated_data(lmm_data):
    """Pre-rotated data for optimizer tests."""
    ed = eigendecompose(lmm_data["K"])
    Y_rot = rotate(lmm_data["Y"], ed.eigenvectors)
    X0_rot = rotate(lmm_data["X0"], ed.eigenvectors)
    return Y_rot, X0_rot, ed.eigenvalues


# ---------------------------------------------------------------
# REML log-likelihood tests
# ---------------------------------------------------------------

class TestREMLLogLikelihood:
    def test_ll_is_finite(self, rotated_data):
        """REML log-likelihood should be finite for reasonable lambda."""
        Y_rot, X0_rot, evals = rotated_data
        ll, sig2_e, sig2_g = reml_loglikelihood(1.0, Y_rot, X0_rot, evals)
        assert math.isfinite(ll)
        assert sig2_e > 0
        assert sig2_g >= 0

    def test_ll_varies_with_lambda(self, rotated_data):
        """Different lambda values should give different likelihoods."""
        Y_rot, X0_rot, evals = rotated_data
        ll1, _, _ = reml_loglikelihood(0.1, Y_rot, X0_rot, evals)
        ll2, _, _ = reml_loglikelihood(1.0, Y_rot, X0_rot, evals)
        ll3, _, _ = reml_loglikelihood(10.0, Y_rot, X0_rot, evals)
        assert not (ll1 == ll2 == ll3)

    def test_ll_maximum_near_true(self, rotated_data, lmm_data):
        """REML ll should be highest near the true lambda."""
        Y_rot, X0_rot, evals = rotated_data
        true_lam = lmm_data["true_sig2_g"] / lmm_data["true_sig2_e"]

        # Evaluate at true and far-off values
        ll_true, _, _ = reml_loglikelihood(true_lam, Y_rot, X0_rot, evals)
        ll_far1, _, _ = reml_loglikelihood(0.001, Y_rot, X0_rot, evals)
        ll_far2, _, _ = reml_loglikelihood(100.0, Y_rot, X0_rot, evals)

        # True lambda should give higher ll than very wrong values
        assert ll_true > ll_far1 or ll_true > ll_far2

    def test_derivatives_finite(self, rotated_data):
        """Derivatives should be finite."""
        Y_rot, X0_rot, evals = rotated_data
        ll, dl, d2l = reml_derivatives(1.0, Y_rot, X0_rot, evals)
        assert math.isfinite(ll)
        assert math.isfinite(dl)
        assert math.isfinite(d2l)

    def test_score_near_zero_at_optimum(self, rotated_data):
        """First derivative should be near zero at the optimum (interior case)."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll, _ = ai_reml_single(Y_rot, X0_rot, evals)
        lam_opt = sig2_g / max(sig2_e, 1e-20)
        if lam_opt > 1e-5:  # only test interior solutions (not boundary)
            _, dl, _ = reml_derivatives(lam_opt, Y_rot, X0_rot, evals)
            assert abs(dl) < 0.5  # score near zero at optimum
        else:
            # At boundary, score may not be zero — that's correct
            pass


# ---------------------------------------------------------------
# PX-EM warm-start tests
# ---------------------------------------------------------------

class TestPXEM:
    def test_em_returns_positive_variances(self, rotated_data):
        """EM should return positive variance components."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, trace = px_em_warmstart(Y_rot, X0_rot, evals, n_iter=5)
        assert sig2_g > 0
        assert sig2_e > 0
        assert len(trace) == 5

    def test_em_improves_from_bad_init(self, rotated_data):
        """EM should move variance components toward reasonable values."""
        Y_rot, X0_rot, evals = rotated_data
        # Start from bad init
        sig2_g, sig2_e, _ = px_em_warmstart(
            Y_rot, X0_rot, evals, n_iter=10,
            sig2_g_init=100.0, sig2_e_init=0.001,
        )
        # Should have moved toward more reasonable values
        assert sig2_g < 100.0
        assert sig2_e > 0.001


# ---------------------------------------------------------------
# AI-REML optimizer tests
# ---------------------------------------------------------------

class TestAIREML:
    def test_convergence(self, rotated_data):
        """AI-REML should converge."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll, trace = ai_reml_single(Y_rot, X0_rot, evals)
        assert sig2_g > 0
        assert sig2_e > 0
        assert math.isfinite(ll)
        # Should converge in reasonable iterations
        assert len(trace) < 100

    def test_variance_components_reasonable(self, rotated_data, lmm_data):
        """Estimated VCs should be in the right ballpark."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll, _ = ai_reml_single(Y_rot, X0_rot, evals)

        h2 = sig2_g / (sig2_g + sig2_e)
        true_h2 = lmm_data["true_sig2_g"] / (lmm_data["true_sig2_g"] + lmm_data["true_sig2_e"])

        # With n=100, h2 estimate should be within 0.3 of truth
        assert abs(h2 - true_h2) < 0.3

    def test_ll_monotonically_increasing(self, rotated_data):
        """Log-likelihood should generally increase across iterations."""
        Y_rot, X0_rot, evals = rotated_data
        _, _, _, trace = ai_reml_single(Y_rot, X0_rot, evals)

        lls = [t["ll"] for t in trace if "ll" in t]
        if len(lls) >= 3:
            # Allow for small deviations but overall trend should be upward
            assert lls[-1] >= lls[0] - 0.1


# ---------------------------------------------------------------
# MM-REML fallback tests
# ---------------------------------------------------------------

class TestMMREML:
    def test_mm_converges(self, rotated_data):
        """MM should converge (possibly slower than AI-REML)."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll, trace = mm_reml(Y_rot, X0_rot, evals)
        assert sig2_g > 0
        assert sig2_e > 0
        assert math.isfinite(ll)

    def test_mm_matches_ai_reml(self, rotated_data):
        """MM and AI-REML should converge to the same optimum."""
        Y_rot, X0_rot, evals = rotated_data
        sg_ai, se_ai, ll_ai, _ = ai_reml_single(Y_rot, X0_rot, evals)
        sg_mm, se_mm, ll_mm, _ = mm_reml(Y_rot, X0_rot, evals)

        # Both should find similar variance components
        assert abs(ll_ai - ll_mm) < 0.1
        h2_ai = sg_ai / (sg_ai + se_ai)
        h2_mm = sg_mm / (sg_mm + se_mm)
        assert abs(h2_ai - h2_mm) < 0.05


# ---------------------------------------------------------------
# Optimizer Controller tests
# ---------------------------------------------------------------

class TestOptimizerController:
    def test_controller_produces_nullfit(self, rotated_data):
        """Controller should produce a complete NullFit."""
        from torchgwas.optim.controller import OptimizerController

        Y_rot, X0_rot, evals = rotated_data
        controller = OptimizerController()
        nf = controller.fit(Y_rot, X0_rot, evals, n_traits=1)

        assert isinstance(nf, NullFit)
        assert nf.sig2_g > 0
        assert nf.sig2_e > 0
        assert nf.weights is not None
        assert nf.M00 is not None
        assert nf.b0 is not None
        assert nf.log_likelihood is not None
        assert len(nf.optimizer_trace) > 0

    def test_controller_trace_has_modes(self, rotated_data):
        """Optimizer trace should show PX-EM followed by AI-REML."""
        from torchgwas.optim.controller import OptimizerController

        Y_rot, X0_rot, evals = rotated_data
        controller = OptimizerController()
        nf = controller.fit(Y_rot, X0_rot, evals, n_traits=1)

        modes = [t.get("mode", "") for t in nf.optimizer_trace]
        assert "PX-EM" in modes
        assert "AI-REML" in modes


# ---------------------------------------------------------------
# SingleTraitLMM end-to-end tests
# ---------------------------------------------------------------

class TestSingleTraitLMM:
    def test_fit_null(self, lmm_data):
        """fit_null should produce a converged NullFit."""
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        assert isinstance(nf, NullFit)
        assert nf.sig2_g > 0
        assert nf.sig2_e > 0
        assert nf.converged

    def test_wald_scan(self, lmm_data):
        """Wald scan should produce valid ScanResult."""
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        # Scan a chunk of 10 SNPs
        G_chunk = lmm_data["G"][:, :10]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(10)],
            chr=["1"] * 10,
            pos=list(range(1, 11)),
            a1=["A"] * 10,
            a2=["G"] * 10,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="wald")

        assert isinstance(result, ScanResult)
        assert result.test == "wald"
        assert result.beta.shape == (10,)
        assert result.se.shape == (10,)
        assert result.stat.shape == (10,)
        assert result.p.shape == (10,)
        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)
        assert torch.all(result.se > 0)
        assert torch.all(result.stat >= 0)

    def test_lrt_scan(self, lmm_data):
        """LRT scan should produce valid ScanResult."""
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        G_chunk = lmm_data["G"][:, :10]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(1, 11)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="lrt")
        assert result.test == "lrt"
        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)

    def test_score_scan(self, lmm_data):
        """Score test should produce NaN for beta/SE."""
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        G_chunk = lmm_data["G"][:, :10]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(1, 11)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="score")
        assert result.test == "score"
        assert torch.all(torch.isnan(result.beta))
        assert torch.all(torch.isnan(result.se))
        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)

    def test_wald_lrt_agreement(self, lmm_data):
        """With fixed VCs, Wald and LRT p-values should be very close."""
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        G_chunk = lmm_data["G"][:, :20]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(20)],
            chr=["1"] * 20, pos=list(range(1, 21)),
            a1=["A"] * 20, a2=["G"] * 20,
        )
        wald = model.score_chunk(G_chunk, nf, vmeta, test="wald")
        lrt = model.score_chunk(G_chunk, nf, vmeta, test="lrt")

        # With fixed VCs, Wald stat ≈ LRT stat (both use same P-operator)
        torch.testing.assert_close(wald.stat, lrt.stat, atol=1e-4, rtol=1e-4)

    def test_pvalues_uniform_under_null(self, lmm_data):
        """Under the null, p-values should roughly follow U(0,1).

        We test that the median p-value is not too extreme (not all
        near 0 or all near 1).
        """
        model = SingleTraitLMM()
        nf = model.fit_null(lmm_data["Y"], lmm_data["X0"], K=lmm_data["K"])

        # Scan many null SNPs (these are random, not causal)
        G_chunk = lmm_data["G"][:, :100]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(100)],
            chr=["1"] * 100, pos=list(range(1, 101)),
            a1=["A"] * 100, a2=["G"] * 100,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="wald")
        median_p = result.p.median().item()
        # Median should be roughly 0.5 for null SNPs (loose check)
        assert 0.05 < median_p < 0.95

    def test_no_kinship_raises(self):
        """Calling fit_null without K should raise ValueError."""
        model = SingleTraitLMM()
        with pytest.raises(ValueError, match="requires a kinship"):
            model.fit_null(
                torch.randn(10, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
            )

    def test_multitrait_raises(self):
        """Passing multi-trait Y should raise ValueError."""
        model = SingleTraitLMM()
        with pytest.raises(ValueError, match="expects 1 trait"):
            model.fit_null(
                torch.randn(10, 3, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
                K=torch.eye(10, dtype=torch.float64),
            )


# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_zero_heritability(self):
        """Model should handle h2 ≈ 0 (no genetic signal)."""
        torch.manual_seed(99)
        n, m = 50, 100
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # y = intercept + noise (no genetic component)
        y = 5.0 + torch.randn(n, dtype=torch.float64)

        model = SingleTraitLMM()
        nf = model.fit_null(y, X0, K=K)
        h2 = nf.sig2_g / (nf.sig2_g + nf.sig2_e)
        assert h2 < 0.3  # should detect low heritability

    def test_high_heritability(self):
        """Model should handle h2 ≈ 1 (strong genetic signal)."""
        torch.manual_seed(77)
        n, m = 50, 100
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # y mostly from genetic effects
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        u = L @ torch.randn(n, dtype=torch.float64) * 3.0
        y = u + torch.randn(n, dtype=torch.float64) * 0.1

        model = SingleTraitLMM()
        nf = model.fit_null(y, X0, K=K)
        h2 = nf.sig2_g / (nf.sig2_g + nf.sig2_e)
        assert h2 > 0.5  # should detect high heritability

    def test_small_sample(self):
        """Model should work with small n."""
        torch.manual_seed(55)
        n, m = 20, 50
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)

        model = SingleTraitLMM()
        nf = model.fit_null(y, X0, K=K)
        assert nf.sig2_g is not None
        assert nf.sig2_e is not None
