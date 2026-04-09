"""Tests for Phase 25: Multi-Trait Multi-Environment (MT-MET) GWAS.

Covers:
- Data reshaping (3D → 2D, NaN filtering)
- Unstructured and separable Kronecker Vg fitting
- GLS scan with joint, per-trait, per-env, and GxE tests
- Interpretive methods (genetic correlations, heritabilities)
- Edge cases and BaseModel protocol conformance
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import BaseModel, NullFit, ScanResult, VariantMeta
from torchgwas.models.multi_trait_multi_env_lmm import (
    MultiTraitMultiEnvLMM,
    MTMETScanResult,
)


# ===================================================================
# Helper: simulate MT-MET data
# ===================================================================

def _simulate_mt_met_data(
    n: int = 150,
    n_snps: int = 50,
    d: int = 2,
    E: int = 3,
    h2: float = 0.3,
    causal_beta: float = 0.5,
    seed: int = 42,
):
    """Simulate multi-trait multi-environment data.

    Generates genotypes, GRM, phenotype Y (n, d, E) with known separable
    genetic covariance and a planted causal SNP (index 0).
    """
    torch.manual_seed(seed)
    dE = d * E

    # Genotypes
    G = torch.randint(0, 3, (n, n_snps), dtype=torch.float64)

    # GRM
    K, _ = grm_vanraden(G)

    # Covariates: intercept
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # True genetic covariance: separable Vg = Vg_t kron Vg_e
    Vg_trait = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)[:d, :d]
    Vg_env = torch.eye(E, dtype=torch.float64) * 0.8
    Vg_env[0, 1] = 0.3
    Vg_env[1, 0] = 0.3
    if E > 2:
        Vg_env[0, 2] = 0.2
        Vg_env[2, 0] = 0.2
        Vg_env[1, 2] = 0.4
        Vg_env[2, 1] = 0.4

    Vg_full = torch.kron(Vg_trait, Vg_env)  # (dE, dE)

    # Polygenic effect: vec(U) ~ N(0, Vg kron K)
    nK = K + 1e-4 * torch.eye(n, dtype=torch.float64)
    LK = torch.linalg.cholesky(nK)
    LVg = torch.linalg.cholesky(Vg_full + 1e-6 * torch.eye(dE, dtype=torch.float64))

    # U = LK @ Z @ LVg^T where Z ~ N(0, I)
    Z = torch.randn(n, dE, dtype=torch.float64)
    U = LK @ Z @ LVg.T * math.sqrt(h2)

    # Residual: vec(E) ~ N(0, Ve kron I)
    Ve_full = torch.eye(dE, dtype=torch.float64) * (1 - h2)
    LVe = torch.linalg.cholesky(Ve_full)
    E_noise = torch.randn(n, dE, dtype=torch.float64) @ LVe.T

    # Causal SNP effect (trait 0, all envs — heterogeneous across envs)
    g_causal = G[:, 0] - G[:, 0].mean()
    std = g_causal.std()
    if std > 0:
        g_causal = g_causal / std

    # Beta for causal: (d, E) — strong for trait 0, weak for others
    beta_true = torch.zeros(d, E, dtype=torch.float64)
    beta_true[0, :] = causal_beta  # trait 0 has signal across all envs
    if E > 2:
        beta_true[0, 2] = causal_beta * 0.3  # weaker in env 2 → GxE

    # Add causal effect
    causal_effect = g_causal.unsqueeze(1) * beta_true.reshape(1, dE)

    # Phenotype
    Y_wide = X0 @ torch.ones(1, dE, dtype=torch.float64) * 0.5 + causal_effect + U + E_noise

    # Reshape to (n, d, E)
    Y_3d = Y_wide.reshape(n, d, E)

    # VariantMeta
    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(n_snps)],
        chr=["1"] * n_snps,
        pos=list(range(0, n_snps * 1000, 1000)),
        a1=["A"] * n_snps,
        a2=["G"] * n_snps,
    )

    return {
        "Y_3d": Y_3d,
        "Y_wide": Y_wide,
        "X0": X0,
        "K": K,
        "G": G,
        "vmeta": vmeta,
        "d": d,
        "E": E,
        "Vg_trait": Vg_trait,
        "Vg_env": Vg_env,
        "beta_true": beta_true,
    }


# ===================================================================
# Data reshaping
# ===================================================================

class TestDataReshaping:
    """Input data validation and reshaping."""

    def test_3d_input_accepted(self):
        """Y as (n, d, E) should be reshaped to (n, dE) internally."""
        data = _simulate_mt_met_data(n=80, n_snps=10, d=2, E=3, seed=10)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        assert nf.n_traits == 2
        assert nf.n_envs == 3

    def test_2d_input_with_dims(self):
        """Y as (n, dE) with explicit n_traits/n_envs."""
        data = _simulate_mt_met_data(n=80, n_snps=10, d=2, E=3, seed=11)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(
            data["Y_wide"], data["X0"], K=data["K"],
            n_traits=2, n_envs=3,
        )
        assert nf.n_traits == 2
        assert nf.n_envs == 3

    def test_nan_filter(self):
        """Rows with NaN should be dropped via complete-case filter."""
        data = _simulate_mt_met_data(n=100, n_snps=10, d=2, E=3, seed=12)
        Y = data["Y_3d"].clone()
        Y[5, 0, 1] = float("nan")
        Y[10, 1, 0] = float("nan")

        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(Y, data["X0"], K=data["K"])
        # Should have lost 2 samples
        assert nf.Y_rot.shape[0] == 98


# ===================================================================
# Unstructured fit
# ===================================================================

class TestUnstructuredFit:
    """Unstructured Vg fitting."""

    def test_fit_null_converges(self):
        """Unstructured null model should converge."""
        data = _simulate_mt_met_data(n=100, n_snps=20, d=2, E=2, seed=20)
        model = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.log_likelihood is not None

    def test_vg_ve_spd(self):
        """Vg and Ve should be symmetric positive semi-definite."""
        data = _simulate_mt_met_data(n=100, n_snps=20, d=2, E=2, seed=21)
        model = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        # Check eigenvalues are non-negative
        eig_vg = torch.linalg.eigvalsh(nf.Vg)
        eig_ve = torch.linalg.eigvalsh(nf.Ve)
        assert (eig_vg >= -1e-6).all(), f"Vg has negative eigenvalue: {eig_vg}"
        assert (eig_ve >= -1e-6).all(), f"Ve has negative eigenvalue: {eig_ve}"

    def test_heritability_reasonable(self):
        """Per-trait-per-env h2 should be in [0, 1]."""
        data = _simulate_mt_met_data(n=100, n_snps=20, d=2, E=2, seed=22)
        model = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        h2 = MultiTraitMultiEnvLMM.per_trait_per_env_heritability(nf)
        assert h2.shape == (2, 2)
        assert (h2 >= 0).all()
        assert (h2 <= 1).all()

    def test_unstructured_matches_multi_trait(self):
        """Unstructured MT-MET should match MultiTraitLMM with dE pseudo-traits."""
        data = _simulate_mt_met_data(n=80, n_snps=10, d=2, E=2, seed=23)
        from torchgwas.models.multi_trait_lmm import MultiTraitLMM

        # MT-MET unstructured
        model_mt = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf_mt = model_mt.fit_null(data["Y_wide"], data["X0"], K=data["K"],
                                   n_traits=2, n_envs=2)

        # MultiTraitLMM directly
        model_mv = MultiTraitLMM()
        nf_mv = model_mv.fit_null(data["Y_wide"], data["X0"], K=data["K"])

        # Log-likelihoods should be very close
        assert abs(nf_mt.log_likelihood - nf_mv.log_likelihood) < 1.0, (
            f"MT-MET ll={nf_mt.log_likelihood:.4f} vs mvLMM ll={nf_mv.log_likelihood:.4f}"
        )


# ===================================================================
# Separable Kronecker fit
# ===================================================================

class TestSeparableFit:
    """Separable Kronecker Vg fitting."""

    def test_separable_shapes(self):
        """Vg_trait (d,d) and Vg_env (E,E) should be stored on NullFit."""
        data = _simulate_mt_met_data(n=100, n_snps=15, d=2, E=3, seed=30)
        model = MultiTraitMultiEnvLMM(vg_structure="separable")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        assert hasattr(nf, "Vg_trait")
        assert hasattr(nf, "Vg_env")
        assert nf.Vg_trait.shape == (2, 2)
        assert nf.Vg_env.shape == (3, 3)

    def test_separable_fewer_params(self):
        """Separable should have fewer unique parameters than unstructured."""
        d, E = 2, 3
        dE = d * E
        # Unstructured: dE*(dE+1)/2 = 21 Vg params
        n_unstructured = dE * (dE + 1) // 2
        # Separable: d*(d+1)/2 + E*(E+1)/2 = 3 + 6 = 9 Vg params
        n_separable = d * (d + 1) // 2 + E * (E + 1) // 2
        assert n_separable < n_unstructured

    def test_separable_kronecker_product(self):
        """Vg should equal kron(Vg_trait, Vg_env)."""
        data = _simulate_mt_met_data(n=100, n_snps=15, d=2, E=3, seed=31)
        model = MultiTraitMultiEnvLMM(vg_structure="separable")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        Vg_expected = torch.kron(nf.Vg_trait, nf.Vg_env)
        assert torch.allclose(nf.Vg, Vg_expected, atol=1e-8)

    def test_separable_ll_vs_unstructured(self):
        """Unstructured LL should be >= separable LL (less constrained)."""
        data = _simulate_mt_met_data(n=100, n_snps=15, d=2, E=2, seed=32)

        model_sep = MultiTraitMultiEnvLMM(vg_structure="separable")
        nf_sep = model_sep.fit_null(data["Y_3d"], data["X0"], K=data["K"])

        model_uns = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf_uns = model_uns.fit_null(data["Y_3d"], data["X0"], K=data["K"])

        # Unstructured should fit at least as well (with tolerance for optimizer noise)
        assert nf_uns.log_likelihood >= nf_sep.log_likelihood - 5.0, (
            f"Unstructured ll={nf_uns.log_likelihood:.4f} < "
            f"Separable ll={nf_sep.log_likelihood:.4f}"
        )

    def test_separable_spd(self):
        """Vg_trait and Vg_env should be SPD."""
        data = _simulate_mt_met_data(n=100, n_snps=15, d=2, E=3, seed=33)
        model = MultiTraitMultiEnvLMM(vg_structure="separable")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        eig_t = torch.linalg.eigvalsh(nf.Vg_trait)
        eig_e = torch.linalg.eigvalsh(nf.Vg_env)
        assert (eig_t > -1e-6).all()
        assert (eig_e > -1e-6).all()


# ===================================================================
# Scan
# ===================================================================

class TestScan:
    """GLS scan output shapes and validity."""

    def test_scan_returns_mtmet_result(self):
        """score_chunk should return MTMETScanResult."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=40)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert isinstance(result, MTMETScanResult)

    def test_scan_beta_shape(self):
        """beta should be (m, d, E)."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, n_snps=30, seed=41)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert result.beta.shape == (30, 2, 3)
        assert result.se.shape == (30, 2, 3)

    def test_joint_pvalues_valid(self):
        """Joint p-values should be in (0, 1]."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=42)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert (result.p > 0).all()
        assert (result.p <= 1).all()

    def test_per_trait_stat_shape(self):
        """Per-trait stats should be (m, d)."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=43)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert result.stat_per_trait.shape == (50, 2)
        assert result.p_per_trait.shape == (50, 2)
        assert (result.p_per_trait > 0).all()
        assert (result.p_per_trait <= 1).all()

    def test_planted_signal_detected(self):
        """Causal SNP (index 0) should have lowest joint p-value (approximately)."""
        data = _simulate_mt_met_data(
            n=200, d=2, E=3, causal_beta=1.0, h2=0.2, seed=44,
        )
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        # Causal SNP should be in top 5 by p-value
        top5 = result.p.argsort()[:5].tolist()
        assert 0 in top5, f"Causal SNP not in top 5: top5={top5}"


# ===================================================================
# GxE tests
# ===================================================================

class TestGxETests:
    """Per-trait GxE interaction tests."""

    def test_gxe_per_trait_shape(self):
        """stat_gxe_per_trait should be (m, d)."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=50)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert result.stat_gxe_per_trait.shape == (50, 2)
        assert result.p_gxe_per_trait.shape == (50, 2)
        assert (result.p_gxe_per_trait > 0).all()
        assert (result.p_gxe_per_trait <= 1).all()

    def test_homogeneous_effects_high_gxe_p(self):
        """When effects are uniform across envs, GxE p should be high."""
        torch.manual_seed(51)
        n, m = 150, 20
        d, E = 2, 3
        dE = d * E
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        # Null phenotype (no GxE)
        Y = torch.randn(n, dE, dtype=torch.float64)

        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(Y, X0, K=K, n_traits=d, n_envs=E)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)
        # Median GxE p-value should be > 0.05 under null
        median_gxe_p = result.p_gxe_per_trait.median().item()
        assert median_gxe_p > 0.01, f"GxE p-value too low under null: {median_gxe_p:.4f}"

    def test_per_env_stat_shape(self):
        """Per-env stats should be (m, E)."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=52)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert result.stat_per_env.shape == (50, 3)
        assert result.p_per_env.shape == (50, 3)
        assert (result.p_per_env > 0).all()
        assert (result.p_per_env <= 1).all()


# ===================================================================
# Interpretive methods
# ===================================================================

class TestInterpretive:
    """Genetic correlations and heritabilities."""

    def test_genetic_correlation_traits(self):
        """rg_traits should be (d, d) with diagonal = 1."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=60)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        rg_t = MultiTraitMultiEnvLMM.genetic_correlation_traits(nf)
        assert rg_t.shape == (2, 2)
        assert torch.allclose(rg_t.diag(), torch.ones(2, dtype=torch.float64), atol=1e-6)
        # Off-diagonal should be in [-1, 1]
        assert (rg_t >= -1.01).all()
        assert (rg_t <= 1.01).all()

    def test_genetic_correlation_envs(self):
        """rg_envs should be (E, E) with diagonal = 1."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=61)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        rg_e = MultiTraitMultiEnvLMM.genetic_correlation_envs(nf)
        assert rg_e.shape == (3, 3)
        assert torch.allclose(rg_e.diag(), torch.ones(3, dtype=torch.float64), atol=1e-6)

    def test_per_trait_per_env_heritability(self):
        """h2 matrix should be (d, E) with values in [0, 1]."""
        data = _simulate_mt_met_data(n=100, d=2, E=3, seed=62)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        h2 = MultiTraitMultiEnvLMM.per_trait_per_env_heritability(nf)
        assert h2.shape == (2, 3)
        assert (h2 >= 0).all()
        assert (h2 <= 1).all()


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Error handling and edge cases."""

    def test_single_trait_raises(self):
        """d=1 should raise ValueError."""
        torch.manual_seed(70)
        n = 50
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, 1, 3, dtype=torch.float64)

        model = MultiTraitMultiEnvLMM()
        with pytest.raises(ValueError, match="d >= 2"):
            model.fit_null(Y, X0, K=K)

    def test_single_env_raises(self):
        """E=1 should raise ValueError."""
        torch.manual_seed(71)
        n = 50
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, 2, 1, dtype=torch.float64)

        model = MultiTraitMultiEnvLMM()
        with pytest.raises(ValueError, match="E >= 2"):
            model.fit_null(Y, X0, K=K)

    def test_empty_chunk(self):
        """Empty genotype chunk should return empty MTMETScanResult."""
        data = _simulate_mt_met_data(n=80, n_snps=10, d=2, E=3, seed=72)
        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        G_empty = torch.zeros(80, 0, dtype=torch.float64)
        vmeta = VariantMeta(snp=[], chr=[], pos=[], a1=[], a2=[])
        result = model.score_chunk(G_empty, nf, vmeta)
        assert len(result) == 0
        assert result.beta.shape == (0, 2, 3)


# ===================================================================
# BaseModel protocol
# ===================================================================

class TestFAFit:
    """FA(k) Vg structure fitting and scanning."""

    def test_fa_fit_converges(self):
        """FA(1) null model should converge on dE=6 space."""
        data = _simulate_mt_met_data(n=120, n_snps=15, d=2, E=3, seed=80)
        model = MultiTraitMultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.log_likelihood is not None
        assert hasattr(nf, "fa_Lambda")
        assert hasattr(nf, "fa_psi")

    def test_fa_scan_shapes(self):
        """FA(1) scan should produce correct output shapes."""
        data = _simulate_mt_met_data(n=120, n_snps=20, d=2, E=3, seed=81)
        model = MultiTraitMultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(data["Y_3d"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert isinstance(result, MTMETScanResult)
        assert result.beta.shape == (20, 2, 3)
        assert result.p.shape == (20,)
        assert (result.p > 0).all()
        assert (result.p <= 1).all()

    def test_fa_ll_vs_unstructured(self):
        """Unstructured LL should be >= FA(1) LL (FA is more constrained)."""
        data = _simulate_mt_met_data(n=100, n_snps=10, d=2, E=2, seed=82)
        model_fa = MultiTraitMultiEnvLMM(vg_structure="fa(1)")
        nf_fa = model_fa.fit_null(data["Y_3d"], data["X0"], K=data["K"])

        model_uns = MultiTraitMultiEnvLMM(vg_structure="unstructured")
        nf_uns = model_uns.fit_null(data["Y_3d"], data["X0"], K=data["K"])

        assert nf_uns.log_likelihood >= nf_fa.log_likelihood - 5.0

    def test_fa_rank_too_large_raises(self):
        """FA rank >= dE should raise ValueError."""
        data = _simulate_mt_met_data(n=80, n_snps=10, d=2, E=2, seed=83)
        model = MultiTraitMultiEnvLMM(vg_structure="fa(4)")
        with pytest.raises(ValueError, match="FA rank"):
            model.fit_null(data["Y_3d"], data["X0"], K=data["K"])


class TestNullCalibration:
    """Verify that p-values are well-calibrated under the null."""

    def test_joint_pvalues_uniform_under_null(self):
        """Under no signal, joint p-values should be roughly uniform.

        Use KS test: p > 0.01 indicates uniform distribution.
        """
        import scipy.stats as sp_stats

        torch.manual_seed(90)
        n, m = 200, 100
        d, E = 2, 3
        dE = d * E
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        # Pure noise phenotype — no signal
        Y = torch.randn(n, dE, dtype=torch.float64)

        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(Y, X0, K=K, n_traits=d, n_envs=E)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        # KS test: null hypothesis is that p-values are uniform(0,1)
        ks_stat, ks_p = sp_stats.kstest(
            result.p.cpu().numpy(), "uniform"
        )
        assert ks_p > 0.01, (
            f"Joint p-values not uniform under null: KS stat={ks_stat:.4f}, "
            f"KS p={ks_p:.4f}"
        )

    def test_per_trait_pvalues_uniform_under_null(self):
        """Per-trait p-values should be approximately uniform under null."""
        import scipy.stats as sp_stats

        torch.manual_seed(91)
        n, m = 200, 100
        d, E = 2, 3
        dE = d * E
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dE, dtype=torch.float64)

        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(Y, X0, K=K, n_traits=d, n_envs=E)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        # Check each trait
        for t in range(d):
            ks_stat, ks_p = sp_stats.kstest(
                result.p_per_trait[:, t].cpu().numpy(), "uniform"
            )
            assert ks_p > 0.005, (
                f"Per-trait p-values (trait {t}) not uniform: "
                f"KS stat={ks_stat:.4f}, KS p={ks_p:.4f}"
            )


class TestProtocol:
    """BaseModel protocol conformance."""

    def test_has_fit_null(self):
        model = MultiTraitMultiEnvLMM()
        assert hasattr(model, "fit_null")
        assert callable(model.fit_null)

    def test_has_score_chunk(self):
        model = MultiTraitMultiEnvLMM()
        assert hasattr(model, "score_chunk")
        assert callable(model.score_chunk)
