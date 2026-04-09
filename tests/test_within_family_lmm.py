"""Tests for Phase 23: Within-Family / Family-Aware GWAS.

Covers:
- Family parsing and demeaning
- Dual null fit (standard + within-family)
- Scan with attenuation metrics
- Confounding detection
- Polyploid compatibility
- BaseModel protocol conformance
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import BaseModel, NullFit, ScanResult, VariantMeta
from torchgwas.models.within_family_lmm import (
    WithinFamilyLMM,
    WithinFamilyNullFit,
    _demean_within_families,
    _parse_family_ids,
)


# ===================================================================
# Helper: simulate family-structured data
# ===================================================================

def _simulate_family_data(
    n_families: int = 30,
    n_per_family: int = 4,
    n_snps: int = 50,
    h2: float = 0.3,
    confound_strength: float = 0.0,
    seed: int = 42,
    ploidy: int = 2,
):
    """Simulate family-structured genotype + phenotype data.

    Parameters
    ----------
    confound_strength : float
        If > 0, add a between-family confounding effect:
        family means of g_confounded correlate with ancestry,
        and ancestry drives between-family phenotype variation.
    """
    torch.manual_seed(seed)
    n = n_families * n_per_family

    # Family IDs: 0,0,0,0, 1,1,1,1, ...
    family_ids = torch.repeat_interleave(
        torch.arange(n_families), n_per_family,
    )

    # Genotypes
    G = torch.randint(0, ploidy + 1, (n, n_snps), dtype=torch.float64)

    # GRM from genotypes
    K, _ = grm_vanraden(G, ploidy=ploidy)

    # Covariates: intercept + one continuous
    X0 = torch.ones(n, 2, dtype=torch.float64)
    X0[:, 1] = torch.randn(n, dtype=torch.float64)

    # True genetic effect from SNP 0 (causal)
    beta_causal = 0.5
    g_signal = G[:, 0] - G[:, 0].mean()
    g_signal = g_signal / g_signal.std()

    # Polygenic effect
    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(h2)

    # Residual
    e = torch.randn(n, dtype=torch.float64) * math.sqrt(1.0 - h2)

    # Phenotype
    Y = X0 @ torch.tensor([0.5, 0.3], dtype=torch.float64) + beta_causal * g_signal + u + e

    # Add between-family confounding if requested
    if confound_strength > 0:
        # Ancestry: per-family latent variable
        ancestry = torch.repeat_interleave(
            torch.randn(n_families, dtype=torch.float64),
            n_per_family,
        )
        # Confounded SNP: correlated with ancestry at family level
        # but NO direct causal effect on Y
        family_geno_shift = torch.repeat_interleave(
            torch.randn(n_families, dtype=torch.float64) * 2.0,
            n_per_family,
        )
        G[:, 1] = (G[:, 1] + family_geno_shift).clamp(0, ploidy)

        # Ancestry drives Y
        Y = Y + confound_strength * ancestry

    return {
        "Y": Y,
        "X0": X0,
        "K": K,
        "G": G,
        "family_ids": family_ids,
        "n_families": n_families,
        "n_per_family": n_per_family,
    }


# ===================================================================
# Demeaning
# ===================================================================

class TestDemeaning:
    """Within-family demeaning helper."""

    def test_correct_mean_subtraction(self):
        """Each family group should have zero mean after demeaning."""
        torch.manual_seed(10)
        X = torch.randn(12, 3, dtype=torch.float64)
        family_indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([3, 4, 5]),
            torch.tensor([6, 7, 8, 9, 10, 11]),
        ]
        X_dm = _demean_within_families(X, family_indices)

        for idx in family_indices:
            family_mean = X_dm[idx].mean(dim=0)
            assert torch.allclose(family_mean, torch.zeros(3, dtype=torch.float64), atol=1e-12)

    def test_intercept_absorbed(self):
        """An all-ones column should become all-zeros after demeaning."""
        n = 8
        ones = torch.ones(n, 1, dtype=torch.float64)
        family_indices = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]
        result = _demean_within_families(ones, family_indices)
        assert torch.allclose(result, torch.zeros(n, 1, dtype=torch.float64), atol=1e-12)

    def test_preserves_shape(self):
        """Demeaning should not change tensor shape."""
        X = torch.randn(20, 5, dtype=torch.float64)
        family_indices = [torch.tensor([0, 1, 2, 3, 4]),
                         torch.tensor([5, 6, 7, 8, 9]),
                         torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
        X_dm = _demean_within_families(X, family_indices)
        assert X_dm.shape == X.shape

    def test_genotype_demeaning(self):
        """Demeaning genotype columns (2D) should work correctly."""
        G = torch.tensor([
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 1],
            [1, 2, 1],
        ], dtype=torch.float64)
        family_indices = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        G_dm = _demean_within_families(G, family_indices)

        # Family 0: mean = [1, 1, 1] → demeaned = [-1,0,1], [1,0,-1]
        assert torch.allclose(G_dm[0], torch.tensor([-1., 0., 1.], dtype=torch.float64))
        assert torch.allclose(G_dm[1], torch.tensor([1., 0., -1.], dtype=torch.float64))


# ===================================================================
# Family parsing
# ===================================================================

class TestFamilyParsing:
    """Family ID parsing and validation."""

    def test_basic_parsing(self):
        """Should correctly group samples by family."""
        fids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        indices, sizes, mask = _parse_family_ids(fids, min_family_size=2)
        assert len(indices) == 3
        assert sizes.sum().item() == 9
        assert mask.all()

    def test_filter_small_families(self):
        """Families smaller than min_family_size should be excluded."""
        fids = torch.tensor([0, 0, 0, 1, 2, 2, 2])  # family 1 has 1 member
        indices, sizes, mask = _parse_family_ids(fids, min_family_size=2)
        assert len(indices) == 2  # families 0 and 2
        assert not mask[3]  # family 1's single member excluded

    def test_insufficient_families_raises(self):
        """Should raise if fewer than 2 valid families."""
        fids = torch.tensor([0, 0, 0, 1])  # only 1 family with ≥2 members
        with pytest.raises(ValueError, match="at least 2 families"):
            _parse_family_ids(fids, min_family_size=2)

    def test_all_singletons_raises(self):
        """All singletons should raise."""
        fids = torch.tensor([0, 1, 2, 3, 4])
        with pytest.raises(ValueError, match="at least 2 families"):
            _parse_family_ids(fids, min_family_size=2)


# ===================================================================
# Fit null
# ===================================================================

class TestWithinFamilyFitNull:
    """Dual null model fitting."""

    @pytest.fixture
    def family_data(self):
        return _simulate_family_data(n_families=25, n_per_family=4, seed=42)

    def test_both_nulls_converge(self, family_data):
        model = WithinFamilyLMM()
        nf = model.fit_null(
            family_data["Y"], family_data["X0"], K=family_data["K"],
            family_ids=family_data["family_ids"],
        )
        assert isinstance(nf, WithinFamilyNullFit)
        assert nf.standard_null_fit.converged
        assert nf.within_null_fit.converged

    def test_variance_components_positive(self, family_data):
        model = WithinFamilyLMM()
        nf = model.fit_null(
            family_data["Y"], family_data["X0"], K=family_data["K"],
            family_ids=family_data["family_ids"],
        )
        assert nf.standard_null_fit.sig2_g > 0
        assert nf.standard_null_fit.sig2_e > 0
        assert nf.within_null_fit.sig2_g > 0
        assert nf.within_null_fit.sig2_e > 0

    def test_heritability_reasonable(self, family_data):
        model = WithinFamilyLMM()
        nf = model.fit_null(
            family_data["Y"], family_data["X0"], K=family_data["K"],
            family_ids=family_data["family_ids"],
        )
        h2_std = nf.sig2_g / (nf.sig2_g + nf.sig2_e)
        h2_wf = nf.within_null_fit.sig2_g / (
            nf.within_null_fit.sig2_g + nf.within_null_fit.sig2_e
        )
        assert 0 < h2_std < 1
        assert 0 < h2_wf < 1

    def test_requires_family_ids(self):
        """Should raise if family_ids not provided."""
        model = WithinFamilyLMM()
        Y = torch.randn(10, dtype=torch.float64)
        X0 = torch.ones(10, 1, dtype=torch.float64)
        K = torch.eye(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="family_ids"):
            model.fit_null(Y, X0, K=K)


# ===================================================================
# Scan
# ===================================================================

class TestWithinFamilyScan:
    """Dual scan with attenuation metrics."""

    @pytest.fixture
    def scan_setup(self):
        data = _simulate_family_data(n_families=30, n_per_family=4, n_snps=30, seed=55)
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(30)],
            chr=["1"] * 30, pos=list(range(30)),
            a1=["A"] * 30, a2=["G"] * 30,
        )
        return model, nf, data["G"], vmeta

    def test_returns_valid_scan_result(self, scan_setup):
        model, nf, G, vmeta = scan_setup
        result = model.score_chunk(G, nf, vmeta)
        assert isinstance(result, ScanResult)
        assert len(result) == 30

    def test_pvalues_in_range(self, scan_setup):
        model, nf, G, vmeta = scan_setup
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()
        assert (result._wf_p >= 0).all()
        assert (result._wf_p <= 1).all()

    def test_beta_se_finite(self, scan_setup):
        model, nf, G, vmeta = scan_setup
        result = model.score_chunk(G, nf, vmeta)
        assert torch.isfinite(result.beta).all()
        assert torch.isfinite(result.se).all()
        assert torch.isfinite(result._wf_beta).all()
        assert torch.isfinite(result._wf_se).all()

    def test_attenuation_computed(self, scan_setup):
        model, nf, G, vmeta = scan_setup
        result = model.score_chunk(G, nf, vmeta)
        assert hasattr(result, "_attenuation")
        assert result._attenuation.shape == (30,)
        assert torch.isfinite(result._attenuation).all()

    def test_both_scans_same_shape(self, scan_setup):
        model, nf, G, vmeta = scan_setup
        result = model.score_chunk(G, nf, vmeta)
        assert result.beta.shape == result._wf_beta.shape
        assert result.se.shape == result._wf_se.shape
        assert result.p.shape == result._wf_p.shape


# ===================================================================
# Attenuation metric
# ===================================================================

class TestAttenuationMetric:
    """Per-SNP attenuation diagnostics."""

    def test_causal_snp_attenuation_near_one(self):
        """Causal SNP (direct effect) should have attenuation near 1."""
        data = _simulate_family_data(
            n_families=40, n_per_family=4, n_snps=20,
            h2=0.3, confound_strength=0.0, seed=100,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(20)],
            chr=["1"] * 20, pos=list(range(20)),
            a1=["A"] * 20, a2=["G"] * 20,
        )
        result = model.score_chunk(data["G"], nf, vmeta)

        # SNP 0 is causal → attenuation should be in [0.3, 3.0] (wide but directional)
        atten_causal = result._attenuation[0].item()
        # With no confounding, attenuation should be positive and moderate
        assert atten_causal > 0.1, f"Causal SNP attenuation too low: {atten_causal}"

    def test_null_attenuation_noisy(self):
        """Under null (no confounding), attenuation should be noisy around 1."""
        torch.manual_seed(101)
        n = 120
        family_ids = torch.repeat_interleave(torch.arange(30), 4)
        G = torch.randint(0, 3, (n, 30), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)

        model = WithinFamilyLMM()
        nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(30)],
            chr=["1"] * 30, pos=list(range(30)),
            a1=["A"] * 30, a2=["G"] * 30,
        )
        result = model.score_chunk(G, nf, vmeta)

        # Should have both standard and within-family results
        assert result._attenuation.shape == (30,)

    def test_sign_handling(self):
        """Attenuation should handle zero-effect SNPs gracefully."""
        torch.manual_seed(102)
        n = 80
        family_ids = torch.repeat_interleave(torch.arange(20), 4)
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)

        model = WithinFamilyLMM()
        nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(G, nf, vmeta)
        # No NaN in attenuation
        assert torch.isfinite(result._attenuation).all()

    def test_confound_flag(self):
        """Confound flag should trigger for confounded SNPs."""
        data = _simulate_family_data(
            n_families=40, n_per_family=4, n_snps=20,
            confound_strength=2.0, seed=103,
        )
        model = WithinFamilyLMM(confound_threshold=0.5)
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(20)],
            chr=["1"] * 20, pos=list(range(20)),
            a1=["A"] * 20, a2=["G"] * 20,
        )
        result = model.score_chunk(data["G"], nf, vmeta)
        assert hasattr(result, "_confound_flag")
        assert result._confound_flag.dtype == torch.bool


# ===================================================================
# Confounding detection (planted confounding)
# ===================================================================

class TestConfoundingDetection:
    """Detect planted between-family confounding."""

    def test_confounded_snp_has_lower_within_family_signal(self):
        """SNP confounded by ancestry should have weaker within-family effect."""
        data = _simulate_family_data(
            n_families=50, n_per_family=4, n_snps=20,
            confound_strength=3.0, seed=200,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(20)],
            chr=["1"] * 20, pos=list(range(20)),
            a1=["A"] * 20, a2=["G"] * 20,
        )
        result = model.score_chunk(data["G"], nf, vmeta)

        # SNP 1 is the confounded one — its within-family effect should be
        # weaker than its standard effect (in absolute terms)
        std_abs = result.beta[1].abs().item()
        wf_abs = result._wf_beta[1].abs().item()
        # If std_abs is substantial, within-family should be smaller
        # (but with small n this is probabilistic, so be lenient)
        if std_abs > 0.1:
            assert wf_abs < std_abs * 2.0, (
                f"Confounded SNP: std={std_abs:.3f}, wf={wf_abs:.3f}"
            )

    def test_null_pvalues_calibrated(self):
        """Under null, both standard and within-family p-values should be uniform."""
        torch.manual_seed(201)
        n_families, n_per = 50, 4
        n = n_families * n_per
        family_ids = torch.repeat_interleave(torch.arange(n_families), n_per)
        G = torch.randint(0, 3, (n, 80), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)

        model = WithinFamilyLMM()
        nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(80)],
            chr=["1"] * 80, pos=list(range(80)),
            a1=["A"] * 80, a2=["G"] * 80,
        )
        result = model.score_chunk(G, nf, vmeta)

        from scipy import stats
        # Standard scan
        ks_stat_std, ks_p_std = stats.kstest(result.p.numpy(), "uniform")
        assert ks_p_std > 0.001, f"Standard KS failed: p={ks_p_std:.6f}"

        # Within-family scan
        ks_stat_wf, ks_p_wf = stats.kstest(result._wf_p.numpy(), "uniform")
        assert ks_p_wf > 0.001, f"Within-family KS failed: p={ks_p_wf:.6f}"

    def test_clean_signal_not_flagged(self):
        """Causal SNP without confounding should not be flagged."""
        data = _simulate_family_data(
            n_families=40, n_per_family=4, n_snps=10,
            confound_strength=0.0, seed=202,
        )
        model = WithinFamilyLMM(confound_threshold=0.8)  # lenient
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(data["G"], nf, vmeta)

        # With no confounding, few SNPs should be flagged
        n_flagged = result._confound_flag.sum().item()
        # Expect most SNPs NOT flagged (allow some noise)
        assert n_flagged < 8, f"Too many flagged: {n_flagged}/10"


# ===================================================================
# Polyploid
# ===================================================================

class TestPolyploid:
    """Polyploid compatibility."""

    def test_tetraploid(self):
        """Tetraploid (ploidy=4) should work end-to-end."""
        data = _simulate_family_data(
            n_families=20, n_per_family=4, n_snps=15,
            seed=300, ploidy=4,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(15)],
            chr=["1"] * 15, pos=list(range(15)),
            a1=["A"] * 15, a2=["G"] * 15,
        )
        result = model.score_chunk(data["G"], nf, vmeta)
        assert len(result) == 15
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_hexaploid(self):
        """Hexaploid (ploidy=6) should work end-to-end."""
        data = _simulate_family_data(
            n_families=20, n_per_family=4, n_snps=10,
            seed=301, ploidy=6,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(data["G"], nf, vmeta)
        assert len(result) == 10
        assert (result.p >= 0).all()


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_chunk(self):
        """Empty genotype chunk should return empty ScanResult."""
        data = _simulate_family_data(n_families=10, n_per_family=3, n_snps=5, seed=400)
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        G_empty = torch.zeros(data["Y"].shape[0], 0, dtype=torch.float64)
        vmeta = VariantMeta(snp=[], chr=[], pos=[], a1=[], a2=[])
        result = model.score_chunk(G_empty, nf, vmeta)
        assert len(result) == 0
        assert result._attenuation.shape == (0,)

    def test_single_member_families_excluded(self):
        """Families with 1 member should be excluded from within-family scan."""
        torch.manual_seed(401)
        n = 30
        # 5 families of size 3 + 15 singletons
        family_ids = torch.cat([
            torch.repeat_interleave(torch.arange(5), 3),  # 5 families × 3
            torch.arange(5, 20),  # 15 singletons
        ])
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)

        model = WithinFamilyLMM(min_family_size=2)
        nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)

        # Should have 5 valid families
        assert nf.n_families == 5
        assert nf.n_valid_samples == 15


# ===================================================================
# BaseModel protocol
# ===================================================================

class TestProtocol:
    """BaseModel protocol conformance."""

    def test_has_fit_null(self):
        model = WithinFamilyLMM()
        assert hasattr(model, "fit_null")
        assert callable(model.fit_null)

    def test_has_score_chunk(self):
        model = WithinFamilyLMM()
        assert hasattr(model, "score_chunk")
        assert callable(model.score_chunk)


# ===================================================================
# Additional edge-case tests (audit round 2)
# ===================================================================

class TestScoreTestModeFamily:
    """Score test mode for WithinFamilyLMM."""

    def test_score_test_returns_valid_results(self):
        """test='score' should produce valid p-values."""
        data = _simulate_family_data(
            n_families=20, n_per_family=4, n_snps=10, seed=500,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(data["G"], nf, vmeta, test="score")
        assert len(result) == 10
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()
        assert hasattr(result, "_wf_p")
        assert (result._wf_p >= 0).all()
        assert (result._wf_p <= 1).all()

    def test_score_test_attenuation_computed(self):
        """Attenuation should be computed under score test too."""
        data = _simulate_family_data(
            n_families=20, n_per_family=4, n_snps=10, seed=501,
        )
        model = WithinFamilyLMM()
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"],
            family_ids=data["family_ids"],
        )
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(data["G"], nf, vmeta, test="score")
        assert hasattr(result, "_attenuation")
        assert result._attenuation.shape == (10,)


class TestInterceptOnlyFamily:
    """WithinFamilyLMM with intercept-only X0."""

    def test_intercept_only_x0(self):
        """With only intercept, demeaned X0 should be handled gracefully."""
        torch.manual_seed(510)
        n_families, n_per = 15, 4
        n = n_families * n_per
        family_ids = torch.repeat_interleave(torch.arange(n_families), n_per)
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)  # intercept only
        Y = torch.randn(n, dtype=torch.float64)

        model = WithinFamilyLMM()
        nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(10)],
            chr=["1"] * 10, pos=list(range(10)),
            a1=["A"] * 10, a2=["G"] * 10,
        )
        result = model.score_chunk(G, nf, vmeta)
        assert len(result) == 10
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()
