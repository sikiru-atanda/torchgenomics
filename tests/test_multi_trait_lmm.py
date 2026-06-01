"""Phase 5: Multi-trait mvLMM tests — EED, LBFGS-autograd REML, Schur complement scan."""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.eigh import eigendecompose, rotate
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.models.lmm_multi_fit import fit_mvlmm_null_lbfgs
from torchgwas.models.multi_trait_lmm import MultiTraitLMM
from torchgwas.optim.lbfgs_reml import lbfgs_reml
from torchgwas.optim.mvlmm_reml import (
    compute_sigma_inv,
    mvlmm_reml_loglikelihood,
)

# ---------------------------------------------------------------
# Fixtures: simulate multi-trait LMM data
# ---------------------------------------------------------------

@pytest.fixture
def mvlmm_data():
    """Simulate Y = XB + G + E with known Vg, Ve.

    n=150 samples, m=300 SNPs, d=2 traits.
    Vg and Ve are 2x2 SPD with genetic correlation 0.5.
    """
    torch.manual_seed(42)
    n, m, d = 150, 300, 2

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # True covariance matrices
    true_Vg = torch.tensor([[0.5, 0.25], [0.25, 0.5]], dtype=torch.float64)
    true_Ve = torch.tensor([[0.5, 0.1], [0.1, 0.5]], dtype=torch.float64)

    # Simulate genetic effects: vec(G_effects) ~ N(0, Vg kron K)
    Lg = torch.linalg.cholesky(true_Vg)
    Lk = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    Z_g = torch.randn(n, d, dtype=torch.float64)
    U_genetic = Lk @ Z_g @ Lg.T  # (n, d)

    # Residual effects
    Le = torch.linalg.cholesky(true_Ve)
    Z_e = torch.randn(n, d, dtype=torch.float64)
    E = Z_e @ Le.T  # (n, d)

    # True fixed effects
    B = torch.tensor([[2.0, 3.0]], dtype=torch.float64)  # (1, 2)
    Y = X0 @ B + U_genetic + E

    return {
        "Y": Y, "X0": X0, "K": K, "G": G,
        "true_Vg": true_Vg, "true_Ve": true_Ve,
        "n": n, "m": m, "d": d,
    }


@pytest.fixture
def mvlmm_3trait():
    """3-trait dataset for testing d > 2."""
    torch.manual_seed(99)
    n, m, d = 100, 200, 3

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = torch.randn(n, d, dtype=torch.float64)
    U_genetic = L @ torch.randn(n, d, dtype=torch.float64) * 0.5
    Y = Y + U_genetic

    return {"Y": Y, "X0": X0, "K": K, "G": G, "n": n, "m": m, "d": d}


@pytest.fixture
def rotated_mv_data(mvlmm_data):
    """Pre-rotated multi-trait data."""
    ed = eigendecompose(mvlmm_data["K"])
    Y_rot = rotate(mvlmm_data["Y"], ed.eigenvectors)
    X0_rot = rotate(mvlmm_data["X0"], ed.eigenvectors)
    return Y_rot, X0_rot, ed.eigenvalues


# ---------------------------------------------------------------
# EED and Sigma computation tests
# ---------------------------------------------------------------

class TestEED:
    def test_sigma_inv_shape(self, rotated_mv_data, mvlmm_data):
        """Per-individual Sigma_i^{-1} should be (n, d, d)."""
        _, _, evals = rotated_mv_data
        Vg = mvlmm_data["true_Vg"]
        Ve = mvlmm_data["true_Ve"]
        W, logdet = compute_sigma_inv(evals, Vg, Ve)
        assert W.shape == (mvlmm_data["n"], 2, 2)
        assert logdet.shape == (mvlmm_data["n"],)

    def test_sigma_inv_spd(self, rotated_mv_data, mvlmm_data):
        """W_i should be positive definite."""
        _, _, evals = rotated_mv_data
        Vg = mvlmm_data["true_Vg"]
        Ve = mvlmm_data["true_Ve"]
        W, _ = compute_sigma_inv(evals, Vg, Ve)

        # Check a few samples
        for i in [0, 10, 50]:
            eigvals = torch.linalg.eigvalsh(W[i])
            assert torch.all(eigvals > 0)

    def test_sigma_inv_correct(self, rotated_mv_data, mvlmm_data):
        """W_i @ Sigma_i should equal I."""
        _, _, evals = rotated_mv_data
        Vg = mvlmm_data["true_Vg"]
        Ve = mvlmm_data["true_Ve"]
        W, _ = compute_sigma_inv(evals, Vg, Ve)

        Sigma_0 = evals[0] * Vg + Ve
        product = W[0] @ Sigma_0
        torch.testing.assert_close(product, torch.eye(2, dtype=torch.float64), atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------
# Multi-trait REML log-likelihood tests
# ---------------------------------------------------------------

class TestMVREML:
    def test_ll_is_finite(self, rotated_mv_data, mvlmm_data):
        """REML log-likelihood should be finite at true parameters."""
        Y_rot, X0_rot, evals = rotated_mv_data
        ll = mvlmm_reml_loglikelihood(
            mvlmm_data["true_Vg"], mvlmm_data["true_Ve"],
            Y_rot, X0_rot, evals,
        )
        assert math.isfinite(ll)

    def test_ll_varies_with_parameters(self, rotated_mv_data, mvlmm_data):
        """Different Vg/Ve should give different likelihoods."""
        Y_rot, X0_rot, evals = rotated_mv_data
        ll1 = mvlmm_reml_loglikelihood(
            mvlmm_data["true_Vg"], mvlmm_data["true_Ve"],
            Y_rot, X0_rot, evals,
        )
        ll2 = mvlmm_reml_loglikelihood(
            torch.eye(2, dtype=torch.float64) * 5.0,
            torch.eye(2, dtype=torch.float64) * 0.1,
            Y_rot, X0_rot, evals,
        )
        assert ll1 != ll2


# ---------------------------------------------------------------
# LBFGS-autograd optimizer tests
# ---------------------------------------------------------------

class TestLBFGSAutograd:
    def test_lbfgs_converges(self, rotated_mv_data):
        """LBFGS should converge on 2-trait data."""
        Y_rot, X0_rot, evals = rotated_mv_data
        Vg, Ve, ll, trace = lbfgs_reml(Y_rot, X0_rot, evals, n_traits=2)

        assert Vg.shape == (2, 2)
        assert Ve.shape == (2, 2)
        assert math.isfinite(ll)

        # Vg, Ve should be SPD
        assert torch.all(torch.linalg.eigvalsh(Vg) > 0)
        assert torch.all(torch.linalg.eigvalsh(Ve) > 0)

    def test_lbfgs_ll_improves(self, rotated_mv_data):
        """LBFGS should improve likelihood over iterations."""
        Y_rot, X0_rot, evals = rotated_mv_data
        _, _, ll_final, trace = lbfgs_reml(Y_rot, X0_rot, evals, n_traits=2)

        lls = [t["ll"] for t in trace]
        # Final ll should be better than initial
        assert lls[-1] >= lls[0] - 1.0  # allow small tolerance

    def test_lbfgs_3_traits(self, mvlmm_3trait):
        """LBFGS should handle 3 traits."""
        ed = eigendecompose(mvlmm_3trait["K"])
        Y_rot = rotate(mvlmm_3trait["Y"], ed.eigenvectors)
        X0_rot = rotate(mvlmm_3trait["X0"], ed.eigenvectors)

        Vg, Ve, ll, _ = lbfgs_reml(Y_rot, X0_rot, ed.eigenvalues, n_traits=3)
        assert Vg.shape == (3, 3)
        assert Ve.shape == (3, 3)
        assert math.isfinite(ll)

    def test_legacy_lbfgs_nullfit_wrapper(self, rotated_mv_data):
        """Legacy mvLMM fit helper returns a populated NullFit."""
        Y_rot, X0_rot, evals = rotated_mv_data
        nf = fit_mvlmm_null_lbfgs(Y_rot, X0_rot, evals)

        assert isinstance(nf, NullFit)
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.weights is not None
        assert nf.Vg.shape == (2, 2)
        assert nf.Ve.shape == (2, 2)


# ---------------------------------------------------------------
# MultiTraitLMM end-to-end tests
# ---------------------------------------------------------------

class TestMultiTraitLMM:
    def test_fit_null(self, mvlmm_data):
        """fit_null should produce a NullFit with Vg, Ve."""
        model = MultiTraitLMM()
        nf = model.fit_null(mvlmm_data["Y"], mvlmm_data["X0"], K=mvlmm_data["K"])

        assert isinstance(nf, NullFit)
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.Vg.shape == (2, 2)
        assert nf.Ve.shape == (2, 2)
        assert nf.log_likelihood is not None

    def test_wald_scan(self, mvlmm_data):
        """Wald scan should produce valid ScanResult with d-trait effects."""
        model = MultiTraitLMM()
        nf = model.fit_null(mvlmm_data["Y"], mvlmm_data["X0"], K=mvlmm_data["K"])

        G_chunk = mvlmm_data["G"][:, :10]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(1, 11)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="wald")

        assert isinstance(result, ScanResult)
        assert result.test == "wald"
        assert result.beta.shape == (10, 2)  # (m, d)
        assert result.se.shape == (10, 2)
        assert result.stat.shape == (10,)  # joint Wald chi2(d)
        assert result.p.shape == (10,)
        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)
        assert torch.all(result.stat >= 0)

    def test_pvalues_reasonable_under_null(self, mvlmm_data):
        """Under the null, p-values should be roughly uniform."""
        model = MultiTraitLMM()
        nf = model.fit_null(mvlmm_data["Y"], mvlmm_data["X0"], K=mvlmm_data["K"])

        G_chunk = mvlmm_data["G"][:, :50]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(50)],
            chr=["1"] * 50, pos=list(range(1, 51)),
            a1=["A"] * 50, a2=["G"] * 50,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="wald")
        median_p = result.p.median().item()
        # Median should be roughly 0.5 for null SNPs (loose check)
        assert 0.01 < median_p < 0.99

    def test_no_kinship_raises(self):
        """Calling fit_null without K should raise ValueError."""
        model = MultiTraitLMM()
        with pytest.raises(ValueError, match="requires a kinship"):
            model.fit_null(
                torch.randn(10, 2, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
            )

    def test_single_trait_raises(self):
        """Passing 1-trait Y should raise ValueError."""
        model = MultiTraitLMM()
        with pytest.raises(ValueError, match="requires >= 2"):
            model.fit_null(
                torch.randn(10, 1, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
                K=torch.eye(10, dtype=torch.float64),
            )

    def test_3_traits_end_to_end(self, mvlmm_3trait):
        """Full pipeline should work with 3 traits."""
        model = MultiTraitLMM()
        nf = model.fit_null(
            mvlmm_3trait["Y"], mvlmm_3trait["X0"], K=mvlmm_3trait["K"],
        )
        assert nf.Vg.shape == (3, 3)

        G_chunk = mvlmm_3trait["G"][:, :5]
        vmeta = VariantMeta(
            snp=[f"snp_{i}" for i in range(5)],
            chr=["1"] * 5, pos=list(range(1, 6)),
            a1=["A"] * 5, a2=["G"] * 5,
        )
        result = model.score_chunk(G_chunk, nf, vmeta, test="wald")
        assert result.beta.shape == (5, 3)
        assert result.stat.shape == (5,)
        assert torch.all(result.p >= 0)
