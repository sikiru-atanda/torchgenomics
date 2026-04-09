"""Tests for Phase 28: Genotype-Uncertainty LMM (GU-LMM).

Covers:
- diag(P) computation (shape, non-negativity, trace)
- Fallback to standard LMM when no uncertainty
- Correction direction (increases denominator, decreases stat)
- Null calibration and power
- End-to-end run() with GP tensor
- Edge cases (single variant, monomorphic, ploidy 4)
- BaseModel protocol conformance
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.models.gu_lmm import GULM, GUResult, _compute_diag_P
from torchgwas.models.single_trait_lmm import SingleTraitLMM


# ===================================================================
# Helper: simulate data
# ===================================================================

def _simulate_gu_data(
    n: int = 200,
    m: int = 100,
    h2: float = 0.5,
    seed: int = 42,
):
    """Simulate LMM data with genotype probabilities.

    Returns G, Y, X0, K, vmeta, gp_probs, dosage_var
    """
    torch.manual_seed(seed)

    # Genotype probabilities: simulate diploid GP
    # probs shape (n, m, 3) for P(0), P(1), P(2)
    raw = torch.randn(n, m, 3, dtype=torch.float64).abs()
    gp_probs = raw / raw.sum(dim=-1, keepdim=True)  # normalize to sum=1

    # Expected dosage
    d_vals = torch.arange(3, dtype=torch.float64)
    G = (gp_probs * d_vals).sum(dim=-1)  # (n, m)

    # Dosage variance
    e_d2 = (gp_probs * d_vals ** 2).sum(dim=-1)
    dosage_var = e_d2 - G ** 2  # (n, m)

    # GRM
    K, _ = grm_vanraden(G)

    # Covariates
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Phenotype: y = X0 @ b + u + e
    sig2_g = h2
    sig2_e = 1.0 - h2
    nK = K + 1e-4 * torch.eye(n, dtype=torch.float64)
    LK = torch.linalg.cholesky(nK)
    u = math.sqrt(sig2_g) * LK @ torch.randn(n, dtype=torch.float64)
    e = math.sqrt(sig2_e) * torch.randn(n, dtype=torch.float64)
    Y = X0.squeeze() * 0.5 + u + e

    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m,
        pos=[i * 1000 for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
    )

    return G, Y, X0, K, vmeta, gp_probs, dosage_var


# ===================================================================
# Test class 1: diag(P) computation
# ===================================================================

class TestDiagP:
    """Tests for _compute_diag_P."""

    @pytest.fixture(autouse=True)
    def setup(self):
        G, Y, X0, K, _, _, _ = _simulate_gu_data(n=100, m=50, seed=1)
        model = GULM()
        self.nf = model.fit_null(Y, X0, K=K)

    def test_shape(self):
        """diag_P should have shape (n,)."""
        assert self.nf.diag_P.shape == (100,)

    def test_nonnegative(self):
        """P is PSD so diagonal entries must be non-negative."""
        # Allow tiny numerical error
        assert (self.nf.diag_P >= -1e-10).all()

    def test_trace_bounded(self):
        """trace(P) should be positive and less than n."""
        n = 100
        trace_P = self.nf.diag_P.sum().item()
        assert trace_P > 0
        assert trace_P < n


# ===================================================================
# Test class 2: Fallback to standard LMM
# ===================================================================

class TestNoUncertaintyFallback:
    """When dosage_var is None or zero, should match standard LMM."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.G, self.Y, self.X0, self.K, self.vmeta, _, _ = _simulate_gu_data(
            n=100, m=50, seed=2
        )

    def test_none_dosage_var_matches_standard(self):
        """With dosage_var=None, score test matches SingleTraitLMM."""
        gu = GULM()
        nf_gu = gu.fit_null(self.Y, self.X0, K=self.K)
        res_gu = gu.score_chunk(self.G, nf_gu, self.vmeta, test="score",
                                dosage_var=None)

        lmm = SingleTraitLMM()
        nf_lmm = lmm.fit_null(self.Y, self.X0, K=self.K)
        res_lmm = lmm.score_chunk(self.G, nf_lmm, self.vmeta, test="score")

        assert torch.allclose(res_gu.stat, res_lmm.stat, rtol=1e-6)
        assert torch.allclose(res_gu.p, res_lmm.p, rtol=1e-6)

    def test_zero_dosage_var_matches_standard(self):
        """With dosage_var=0, should match standard LMM."""
        gu = GULM()
        nf_gu = gu.fit_null(self.Y, self.X0, K=self.K)
        zero_var = torch.zeros_like(self.G)
        res_gu = gu.score_chunk(self.G, nf_gu, self.vmeta, test="score",
                                dosage_var=zero_var)

        lmm = SingleTraitLMM()
        nf_lmm = lmm.fit_null(self.Y, self.X0, K=self.K)
        res_lmm = lmm.score_chunk(self.G, nf_lmm, self.vmeta, test="score")

        assert torch.allclose(res_gu.stat, res_lmm.stat, rtol=1e-6)


# ===================================================================
# Test class 3: Correction direction
# ===================================================================

class TestCorrectionDirection:
    """Uncertainty correction should increase denominator & decrease stat."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.G, self.Y, self.X0, self.K, self.vmeta, _, self.dvar = \
            _simulate_gu_data(n=100, m=50, seed=3)

    def test_correction_decreases_stat(self):
        """Corrected stat should be <= standard stat."""
        gu = GULM()
        nf = gu.fit_null(self.Y, self.X0, K=self.K)

        res_uncorr = gu.score_chunk(self.G, nf, self.vmeta, test="score",
                                    dosage_var=None)
        res_corr = gu.score_chunk(self.G, nf, self.vmeta, test="score",
                                  dosage_var=self.dvar)

        # Corrected stat <= uncorrected (larger denominator)
        assert (res_corr.stat <= res_uncorr.stat + 1e-10).all()

    def test_pvalues_valid_range(self):
        """All p-values should be in [0, 1]."""
        gu = GULM()
        nf = gu.fit_null(self.Y, self.X0, K=self.K)
        res = gu.score_chunk(self.G, nf, self.vmeta, test="score",
                             dosage_var=self.dvar)

        assert (res.p >= 0).all()
        assert (res.p <= 1).all()

    def test_high_uncertainty_large_correction(self):
        """With very high dosage variance, correction should be substantial."""
        gu = GULM()
        nf = gu.fit_null(self.Y, self.X0, K=self.K)

        # Inflate dosage variance 10x
        big_var = self.dvar * 10.0
        res_std = gu.score_chunk(self.G, nf, self.vmeta, test="score",
                                 dosage_var=None)
        res_big = gu.score_chunk(self.G, nf, self.vmeta, test="score",
                                 dosage_var=big_var)

        # Large uncertainty should substantially reduce statistics
        ratio = res_big.stat / res_std.stat.clamp(min=1e-20)
        assert ratio.mean().item() < 0.9  # at least 10% reduction on average


# ===================================================================
# Test class 4: Calibration
# ===================================================================

class TestCalibration:
    """Tests for null calibration and power."""

    def test_null_calibration(self):
        """Under null (no causal), p-values should be approximately uniform.

        Uses small dosage variance (high-quality imputation) for realistic test.
        """
        from scipy.stats import kstest

        torch.manual_seed(10)
        n, m = 200, 500

        # Realistic genotypes with small imputation uncertainty
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        # Small dosage variance (high-quality imputation, ~Rsq > 0.9)
        dvar = torch.rand(n, m, dtype=torch.float64) * 0.05

        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # Null phenotype (no causal SNP)
        nK = K + 1e-4 * torch.eye(n, dtype=torch.float64)
        LK = torch.linalg.cholesky(nK)
        u = 0.5 * LK @ torch.randn(n, dtype=torch.float64)
        Y = 0.5 + u + 0.5 * torch.randn(n, dtype=torch.float64)

        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        gu = GULM()
        nf = gu.fit_null(Y, X0, K=K)
        res = gu.score_chunk(G, nf, vmeta, test="score", dosage_var=dvar)

        # KS test for uniformity
        p_vals = res.p.cpu().numpy()
        valid = p_vals[~(p_vals != p_vals)]  # remove NaN
        ks_stat, ks_p = kstest(valid, "uniform")
        assert ks_p > 0.001

    def test_power_with_signal(self):
        """With planted signal, should detect it (using realistic genotypes)."""
        torch.manual_seed(20)
        n, m = 300, 200

        # Realistic discrete genotypes with small uncertainty
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        dosage_var = torch.rand(n, m, dtype=torch.float64) * 0.05

        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # Plant causal effect at SNP 0
        g0 = G[:, 0] - G[:, 0].mean()
        std = g0.std().clamp(min=1e-6)
        g0 = g0 / std
        Y = X0.squeeze() * 0.5 + g0 * 0.5 + 0.5 * torch.randn(n, dtype=torch.float64)

        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        gu = GULM()
        nf = gu.fit_null(Y, X0, K=K)
        res = gu.score_chunk(G, nf, vmeta, test="score", dosage_var=dosage_var)

        # Causal SNP should have small p-value
        assert res.p[0].item() < 0.05


# ===================================================================
# Test class 5: run() method
# ===================================================================

class TestRunMethod:
    """Tests for the convenience run() method."""

    def test_run_returns_gu_result(self):
        """run() should return a GUResult."""
        G, Y, X0, K, vmeta, gp_probs, _ = _simulate_gu_data(
            n=100, m=50, seed=30,
        )
        gu = GULM()
        result = gu.run(Y, X0, K, gp_probs, vmeta, ploidy=2)

        assert isinstance(result, GUResult)
        assert isinstance(result.scan, ScanResult)
        assert result.mean_correction.shape == (50,)
        assert result.dosage_rsq.shape == (50,)

    def test_run_diagnostics_reasonable(self):
        """Diagnostics should be in valid ranges."""
        G, Y, X0, K, vmeta, gp_probs, _ = _simulate_gu_data(
            n=100, m=50, seed=31,
        )
        gu = GULM()
        result = gu.run(Y, X0, K, gp_probs, vmeta, ploidy=2)

        # Correction fraction should be in [0, 1]
        assert (result.mean_correction >= -0.01).all()
        assert (result.mean_correction <= 1.01).all()

        # Dosage R² should be in [0, 1]
        assert (result.dosage_rsq >= 0).all()
        assert (result.dosage_rsq <= 1).all()


# ===================================================================
# Test class 6: Edge cases
# ===================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_variant(self):
        """Should work with a single variant."""
        G, Y, X0, K, _, gp_probs, dvar = _simulate_gu_data(n=100, m=50, seed=40)
        gu = GULM()
        nf = gu.fit_null(Y, X0, K=K)

        vmeta1 = VariantMeta(
            snp=["SNP0"], chr=["1"], pos=[0], a1=["A"], a2=["G"],
        )
        res = gu.score_chunk(
            G[:, :1], nf, vmeta1, test="score", dosage_var=dvar[:, :1],
        )
        assert res.stat.shape == (1,)

    def test_monomorphic_safe(self):
        """Monomorphic variant should not crash."""
        torch.manual_seed(41)
        n, m = 100, 5
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        G[:, 0] = 1.0  # monomorphic
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)
        dvar = torch.rand(n, m, dtype=torch.float64) * 0.1

        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        gu = GULM()
        nf = gu.fit_null(Y, X0, K=K)
        res = gu.score_chunk(G, nf, vmeta, test="score", dosage_var=dvar)
        assert res.stat.shape == (m,)
        assert not torch.isnan(res.p).any()

    def test_wald_test_delegates(self):
        """Wald test should delegate to standard LMM (no correction)."""
        G, Y, X0, K, vmeta, _, _ = _simulate_gu_data(n=100, m=30, seed=42)
        gu = GULM()
        nf = gu.fit_null(Y, X0, K=K)
        res = gu.score_chunk(G, nf, vmeta, test="wald")

        assert res.test == "wald"
        assert not torch.isnan(res.beta).any()


# ===================================================================
# Test class 7: Protocol
# ===================================================================

class TestProtocol:
    """Tests for BaseModel conformance."""

    def test_has_fit_null_and_score_chunk(self):
        """GULM should have fit_null and score_chunk methods."""
        model = GULM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
        assert callable(model.fit_null)
        assert callable(model.score_chunk)


# ===================================================================
# Test class 8: Determinism
# ===================================================================

class TestDeterminism:
    """Tests for reproducibility."""

    def test_deterministic(self):
        """Same input should give same output."""
        G, Y, X0, K, vmeta, _, dvar = _simulate_gu_data(n=100, m=30, seed=50)

        gu = GULM()
        nf1 = gu.fit_null(Y, X0, K=K)
        res1 = gu.score_chunk(G, nf1, vmeta, test="score", dosage_var=dvar)

        nf2 = gu.fit_null(Y, X0, K=K)
        res2 = gu.score_chunk(G, nf2, vmeta, test="score", dosage_var=dvar)

        assert torch.allclose(res1.stat, res2.stat, rtol=1e-10)
