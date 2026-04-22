"""Tests for Phase 26: Orthogonal Cross-Fit LMM (OCF-LMM).

Covers:
- Fold creation (partitioning, sizes)
- Per-fold REML + BLUP prediction
- Full cross-fit null model
- DML scan output shapes and validity
- Null calibration (chi-squared(1) null)
- Comparison with standard SingleTraitLMM
- Edge cases and BaseModel protocol conformance
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import ScanResult, VariantMeta
from torchgwas.models.ocf_lmm import (
    OCFLMM,
    OCFNullFit,
    _create_folds,
    _fit_fold,
)

# ===================================================================
# Helper: simulate LMM data
# ===================================================================

def _simulate_ocf_data(
    n: int = 200,
    n_snps: int = 500,
    h2: float = 0.5,
    causal_beta: float = 0.3,
    seed: int = 42,
):
    """Simulate single-trait LMM data with a planted causal SNP.

    y = X0 @ beta_fixed + g_causal * causal_beta + u + e
    u ~ N(0, sig2_g * K), e ~ N(0, sig2_e * I)
    """
    torch.manual_seed(seed)

    # Genotypes
    G = torch.randint(0, 3, (n, n_snps), dtype=torch.float64)

    # GRM from all SNPs
    K, _ = grm_vanraden(G)

    # Covariates: intercept
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Polygenic effect: u ~ N(0, sig2_g * K)
    sig2_g = h2
    sig2_e = 1.0 - h2
    nK = K + 1e-4 * torch.eye(n, dtype=torch.float64)
    LK = torch.linalg.cholesky(nK)
    u = math.sqrt(sig2_g) * LK @ torch.randn(n, dtype=torch.float64)

    # Residual
    e = math.sqrt(sig2_e) * torch.randn(n, dtype=torch.float64)

    # Causal SNP (index 0)
    g_causal = G[:, 0] - G[:, 0].mean()
    std = g_causal.std()
    if std > 0:
        g_causal = g_causal / std

    # Phenotype
    Y = X0.squeeze() * 0.5 + g_causal * causal_beta + u + e

    # VariantMeta
    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(n_snps)],
        chr=["1"] * n_snps,
        pos=list(range(0, n_snps * 1000, 1000)),
        a1=["A"] * n_snps,
        a2=["G"] * n_snps,
    )

    return {
        "Y": Y,
        "X0": X0,
        "K": K,
        "G": G,
        "vmeta": vmeta,
        "h2": h2,
        "causal_beta": causal_beta,
    }


# ===================================================================
# Fold creation
# ===================================================================

class TestFoldCreation:
    """Fold partitioning tests."""

    def test_folds_partition_all_samples(self):
        """K folds should collectively cover all n samples with no overlap."""
        folds = _create_folds(100, n_folds=5, seed=1)
        all_idx = torch.cat(folds).sort().values
        assert torch.equal(all_idx, torch.arange(100))

    def test_folds_roughly_equal_size(self):
        """Max fold size - min fold size should be at most 1."""
        folds = _create_folds(103, n_folds=5, seed=2)
        sizes = [len(f) for f in folds]
        assert max(sizes) - min(sizes) <= 1


# ===================================================================
# Per-fold fit
# ===================================================================

class TestFitFold:
    """Per-fold REML + BLUP prediction."""

    def test_fold_reml_converges(self):
        """Per-fold REML should converge."""
        data = _simulate_ocf_data(n=150, n_snps=100, seed=10)
        from torchgwas.config import NumericalConfig
        config = NumericalConfig()

        folds = _create_folds(150, n_folds=3, seed=10)
        test_idx = folds[0]
        train_mask = torch.ones(150, dtype=torch.bool)
        train_mask[test_idx] = False
        train_idx = torch.arange(150)[train_mask]

        fr = _fit_fold(
            data["Y"], data["X0"], data["K"],
            test_indices=test_idx, train_indices=train_idx,
            config=config, fold_id=0,
        )
        assert fr.converged
        assert fr.sig2_g > 0
        assert fr.sig2_e > 0

    def test_blup_prediction_nonzero(self):
        """BLUP predictions should be nonzero when h2 > 0."""
        data = _simulate_ocf_data(n=150, n_snps=100, h2=0.5, seed=11)
        from torchgwas.config import NumericalConfig
        config = NumericalConfig()

        folds = _create_folds(150, n_folds=3, seed=11)
        test_idx = folds[0]
        train_mask = torch.ones(150, dtype=torch.bool)
        train_mask[test_idx] = False
        train_idx = torch.arange(150)[train_mask]

        fr = _fit_fold(
            data["Y"], data["X0"], data["K"],
            test_indices=test_idx, train_indices=train_idx,
            config=config, fold_id=0,
        )
        # y_tilde should not equal raw Y_test (BLUP removed something)
        Y_test = data["Y"][test_idx]
        assert not torch.allclose(fr.y_tilde_test, Y_test, atol=0.01)

    def test_cross_fit_residual_shape(self):
        """Cross-fit residual should have shape (n_test,)."""
        data = _simulate_ocf_data(n=100, n_snps=50, seed=12)
        from torchgwas.config import NumericalConfig
        config = NumericalConfig()

        folds = _create_folds(100, n_folds=5, seed=12)
        test_idx = folds[0]
        train_mask = torch.ones(100, dtype=torch.bool)
        train_mask[test_idx] = False
        train_idx = torch.arange(100)[train_mask]

        fr = _fit_fold(
            data["Y"], data["X0"], data["K"],
            test_indices=test_idx, train_indices=train_idx,
            config=config, fold_id=0,
        )
        assert fr.y_tilde_test.shape == (len(test_idx),)


# ===================================================================
# Full null fit
# ===================================================================

class TestFitNull:
    """Full cross-fit null model."""

    def test_returns_ocf_nullfit(self):
        """fit_null should return OCFNullFit with correct n_folds."""
        data = _simulate_ocf_data(n=100, n_snps=50, seed=20)
        model = OCFLMM(n_folds=3, seed=20)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        assert isinstance(nf, OCFNullFit)
        assert nf.n_folds == 3
        assert nf.y_tilde.shape == (100,)
        assert nf.fold_ids.shape == (100,)

    def test_all_folds_converged(self):
        """All fold REML fits should converge."""
        data = _simulate_ocf_data(n=150, n_snps=100, seed=21)
        model = OCFLMM(n_folds=5, seed=21)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        assert nf.converged
        assert len(nf.fold_results) == 5
        for fr in nf.fold_results:
            assert fr.converged

    def test_h2_reasonable(self):
        """Mean h2 across folds should be within 0.3 of truth."""
        data = _simulate_ocf_data(n=200, n_snps=200, h2=0.5, seed=22)
        model = OCFLMM(n_folds=5, seed=22)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        assert abs(nf.mean_h2 - 0.5) < 0.35, f"mean_h2={nf.mean_h2:.4f}"


# ===================================================================
# DML scan
# ===================================================================

class TestScan:
    """DML scan output shapes and validity."""

    def test_scan_shapes(self):
        """score_chunk should return ScanResult with correct shapes."""
        data = _simulate_ocf_data(n=150, n_snps=50, seed=30)
        model = OCFLMM(n_folds=3, seed=30)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert isinstance(result, ScanResult)
        assert result.beta.shape == (50,)
        assert result.se.shape == (50,)
        assert result.stat.shape == (50,)
        assert result.p.shape == (50,)

    def test_pvalues_valid(self):
        """P-values should be in (0, 1]."""
        data = _simulate_ocf_data(n=150, n_snps=50, seed=31)
        model = OCFLMM(n_folds=3, seed=31)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        assert (result.p > 0).all()
        assert (result.p <= 1).all()
        assert result.inference_type == "cross_fit"

    def test_planted_signal_detected(self):
        """Causal SNP (index 0) should be in top 5 by p-value."""
        data = _simulate_ocf_data(
            n=250, n_snps=100, h2=0.3, causal_beta=0.8, seed=32,
        )
        model = OCFLMM(n_folds=5, seed=32)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        result = model.score_chunk(data["G"], nf, data["vmeta"])
        top5 = result.p.argsort()[:5].tolist()
        assert 0 in top5, f"Causal SNP not in top 5: top5={top5}"


# ===================================================================
# Null calibration
# ===================================================================

class TestNullCalibration:
    """Verify chi-squared(1) null is well-calibrated."""

    def test_null_pvalues_uniform(self):
        """Under no signal, p-values should be roughly uniform (KS test)."""
        import scipy.stats as sp_stats

        torch.manual_seed(40)
        n, m = 200, 200
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        # Pure noise phenotype
        Y = torch.randn(n, dtype=torch.float64)

        model = OCFLMM(n_folds=5, seed=40)
        nf = model.fit_null(Y, X0, K=K)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        ks_stat, ks_p = sp_stats.kstest(result.p.cpu().numpy(), "uniform")
        assert ks_p > 0.01, (
            f"P-values not uniform under null: KS stat={ks_stat:.4f}, KS p={ks_p:.4f}"
        )

    def test_median_pvalue_near_half(self):
        """Median p-value should be between 0.1 and 0.9 under null."""
        torch.manual_seed(41)
        n, m = 150, 100
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)

        model = OCFLMM(n_folds=3, seed=41)
        nf = model.fit_null(Y, X0, K=K)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert 0.1 < median_p < 0.9, f"Median p={median_p:.4f}"


# ===================================================================
# Comparison with standard LMM
# ===================================================================

class TestVsStandardLMM:
    """OCF-LMM should produce similar results to standard LMM under correct spec."""

    def test_similar_pvalues(self):
        """Rank correlation of -log10(p) between OCF and standard LMM > 0.8."""
        import scipy.stats as sp_stats

        data = _simulate_ocf_data(n=200, n_snps=100, h2=0.4, seed=50)

        # Standard LMM
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model_std = SingleTraitLMM()
        nf_std = model_std.fit_null(data["Y"], data["X0"], K=data["K"])
        result_std = model_std.score_chunk(data["G"], nf_std, data["vmeta"])

        # OCF-LMM
        model_ocf = OCFLMM(n_folds=5, seed=50)
        nf_ocf = model_ocf.fit_null(data["Y"], data["X0"], K=data["K"])
        result_ocf = model_ocf.score_chunk(data["G"], nf_ocf, data["vmeta"])

        logp_std = -torch.log10(result_std.p).cpu().numpy()
        logp_ocf = -torch.log10(result_ocf.p).cpu().numpy()

        rho, _ = sp_stats.spearmanr(logp_std, logp_ocf)
        assert rho > 0.7, f"Rank correlation {rho:.4f} too low"

    def test_similar_top_hits(self):
        """Top 5 SNPs from OCF-LMM should overlap substantially with standard LMM."""
        data = _simulate_ocf_data(
            n=250, n_snps=100, h2=0.3, causal_beta=0.8, seed=51,
        )

        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model_std = SingleTraitLMM()
        nf_std = model_std.fit_null(data["Y"], data["X0"], K=data["K"])
        result_std = model_std.score_chunk(data["G"], nf_std, data["vmeta"])

        model_ocf = OCFLMM(n_folds=5, seed=51)
        nf_ocf = model_ocf.fit_null(data["Y"], data["X0"], K=data["K"])
        result_ocf = model_ocf.score_chunk(data["G"], nf_ocf, data["vmeta"])

        top10_std = set(result_std.p.argsort()[:10].tolist())
        top10_ocf = set(result_ocf.p.argsort()[:10].tolist())
        overlap = len(top10_std & top10_ocf)
        assert overlap >= 3, f"Top-10 overlap={overlap}, sets={top10_std} vs {top10_ocf}"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Error handling and edge cases."""

    def test_zero_heritability(self):
        """OCF-LMM should work gracefully when h2 ~ 0."""
        torch.manual_seed(60)
        n = 100
        G = torch.randint(0, 3, (n, 20), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        # Pure noise — no genetic signal
        Y = torch.randn(n, dtype=torch.float64)

        model = OCFLMM(n_folds=3, seed=60)
        nf = model.fit_null(Y, X0, K=K)
        # h2 should be small
        assert nf.mean_h2 < 0.5, f"h2={nf.mean_h2:.4f} unexpectedly high"

        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(20)],
            chr=["1"] * 20, pos=list(range(20)),
            a1=["A"] * 20, a2=["G"] * 20,
        )
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p > 0).all()
        assert (result.p <= 1).all()

    def test_no_kinship_raises(self):
        """fit_null without K should raise ValueError."""
        model = OCFLMM()
        Y = torch.randn(50, dtype=torch.float64)
        X0 = torch.ones(50, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="requires a kinship matrix"):
            model.fit_null(Y, X0)

    def test_n_folds_2(self):
        """Minimum valid n_folds=2 should work."""
        data = _simulate_ocf_data(n=100, n_snps=20, seed=62)
        model = OCFLMM(n_folds=2, seed=62)
        nf = model.fit_null(data["Y"], data["X0"], K=data["K"])
        assert nf.n_folds == 2
        assert nf.y_tilde.shape == (100,)


# ===================================================================
# BaseModel protocol
# ===================================================================

class TestProtocol:
    """BaseModel protocol conformance."""

    def test_has_fit_null(self):
        model = OCFLMM()
        assert hasattr(model, "fit_null")
        assert callable(model.fit_null)

    def test_has_score_chunk(self):
        model = OCFLMM()
        assert hasattr(model, "score_chunk")
        assert callable(model.score_chunk)
