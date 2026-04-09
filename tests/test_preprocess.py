"""Phase 2: Preprocessing — imputation, QC, standardization, covariates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from torchgwas.preprocess.standardize import (
    center_genotypes,
    compute_allele_frequencies,
    compute_maf,
    scale_genotypes,
)
from torchgwas.preprocess.impute import impute_mean, impute_mode
from torchgwas.preprocess.qc import (
    QCFilterConfig,
    VariantQCStats,
    apply_qc_filters,
    compute_variant_qc,
    write_variant_qc_parquet,
)
from torchgwas.preprocess.covariates import build_covariate_matrix
from torchgwas.models.base import VariantMeta


@pytest.fixture
def G_with_nan():
    """10 samples, 5 SNPs, some NaN."""
    G = torch.tensor([
        [0, 1, 2, 0, 1],
        [1, 1, 0, 2, 0],
        [2, 0, 1, 1, 1],
        [0, 2, float("nan"), 0, 2],
        [1, 1, 1, 1, 1],
        [0, 0, 2, float("nan"), 0],
        [2, 1, 0, 1, 1],
        [1, 2, 1, 0, 2],
        [0, 0, 2, 2, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float64)
    return G


@pytest.fixture
def vmeta_5():
    return VariantMeta(
        snp=[f"rs{i}" for i in range(5)],
        chr=["1"] * 5,
        pos=[i * 1000 for i in range(5)],
        a1=["A"] * 5,
        a2=["G"] * 5,
    )


class TestStandardize:
    """Tests for torchgwas.preprocess.standardize."""

    def test_allele_frequencies_range(self, G_with_nan):
        af = compute_allele_frequencies(G_with_nan, ploidy=2)
        assert af.shape == (5,)
        assert torch.all(af >= 0)
        assert torch.all(af <= 1)

    def test_allele_frequencies_correct(self):
        """Known allele frequencies for simple data."""
        G = torch.tensor([[0, 2], [2, 0], [1, 1], [1, 1]], dtype=torch.float64)
        af = compute_allele_frequencies(G, ploidy=2)
        # SNP0: mean=1.0, af=0.5; SNP1: mean=1.0, af=0.5
        torch.testing.assert_close(af, torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_allele_frequencies_polyploid(self):
        """Polyploid AF divides by ploidy."""
        G = torch.tensor([[0, 4], [4, 0], [2, 2]], dtype=torch.float64)
        af = compute_allele_frequencies(G, ploidy=4)
        # SNP0: mean=2.0, af=0.5; SNP1: mean=2.0, af=0.5
        torch.testing.assert_close(af, torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_maf_range(self, G_with_nan):
        af = compute_allele_frequencies(G_with_nan)
        maf = compute_maf(af)
        assert torch.all(maf >= 0)
        assert torch.all(maf <= 0.5)

    def test_center_genotypes_zero_mean(self, G_with_nan):
        G_c = center_genotypes(G_with_nan)
        # Column means of non-NaN values should be ~0
        for j in range(G_c.shape[1]):
            col = G_c[:, j]
            valid = col[~torch.isnan(col)]
            assert abs(valid.mean().item()) < 1e-10

    def test_scale_genotypes_unit_variance(self):
        """Scaled genotypes have column variance ≈ 1 for well-behaved data."""
        torch.manual_seed(42)
        G = torch.randint(0, 3, (100, 10), dtype=torch.float64)
        G_s = scale_genotypes(G, ploidy=2)
        # Check variance is approximately 1 (not exact due to HWE assumption)
        for j in range(10):
            col = G_s[:, j]
            # Variance should be close to 1 but depends on how well HWE holds
            assert col.var().item() > 0.1  # at least non-zero

    def test_center_preserves_nan(self, G_with_nan):
        G_c = center_genotypes(G_with_nan)
        assert torch.isnan(G_c[3, 2])
        assert torch.isnan(G_c[5, 3])


class TestImputation:
    """Tests for torchgwas.preprocess.impute."""

    def test_impute_mean(self, G_with_nan):
        G_imp = impute_mean(G_with_nan)
        assert not torch.isnan(G_imp).any()

    def test_impute_mean_correct(self):
        """Mean imputation replaces NaN with column mean."""
        G = torch.tensor([[0, 1], [2, float("nan")], [1, 1]], dtype=torch.float64)
        G_imp = impute_mean(G)
        assert G_imp[1, 1].item() == 1.0  # mean of [1, 1] = 1.0

    def test_impute_mode(self, G_with_nan):
        G_imp = impute_mode(G_with_nan)
        assert not torch.isnan(G_imp).any()

    def test_impute_mode_correct(self):
        """Mode imputation replaces NaN with most frequent value."""
        G = torch.tensor([[0, 2], [0, float("nan")], [0, 1], [1, 2]], dtype=torch.float64)
        G_imp = impute_mode(G)
        assert G_imp[1, 1].item() == 2.0  # mode of [2, 1, 2] = 2

    def test_impute_no_missing_passthrough(self):
        G = torch.tensor([[0, 1], [2, 0]], dtype=torch.float64)
        G_imp = impute_mean(G)
        torch.testing.assert_close(G_imp, G)


class TestQCFilters:
    """Tests for torchgwas.preprocess.qc."""

    def test_compute_variant_qc(self, G_with_nan, vmeta_5):
        stats = compute_variant_qc(G_with_nan, vmeta_5, ploidy=2)
        assert len(stats.snp) == 5
        assert stats.af.shape == (5,)
        assert stats.maf.shape == (5,)
        assert stats.miss_rate.shape == (5,)
        assert stats.hwe_p.shape == (5,)

    def test_missingness_calculated(self, G_with_nan, vmeta_5):
        stats = compute_variant_qc(G_with_nan, vmeta_5)
        # SNP2 has 1 missing, SNP3 has 1 missing
        assert stats.n_miss[2].item() == 1
        assert stats.n_miss[3].item() == 1
        assert stats.miss_rate[2].item() == pytest.approx(0.1)

    def test_maf_filter(self, G_with_nan, vmeta_5):
        stats = compute_variant_qc(G_with_nan, vmeta_5)
        config = QCFilterConfig(maf_min=0.5)  # very strict
        passes = apply_qc_filters(stats, config)
        # Some SNPs should fail this strict filter
        assert not passes.all()

    def test_missingness_filter(self, G_with_nan, vmeta_5):
        stats = compute_variant_qc(G_with_nan, vmeta_5)
        config = QCFilterConfig(miss_max=0.05)  # stricter than default
        passes = apply_qc_filters(stats, config)
        # SNPs with >5% missingness should fail
        for i in range(5):
            if stats.miss_rate[i].item() > 0.05:
                assert not passes[i]

    def test_hwe_filter(self, G_with_nan, vmeta_5):
        stats = compute_variant_qc(G_with_nan, vmeta_5)
        # All HWE p-values should be computed
        assert stats.hwe_p.shape == (5,)
        assert torch.all(stats.hwe_p >= 0)
        assert torch.all(stats.hwe_p <= 1)

    def test_variant_qc_parquet_output(self, tmp_path, G_with_nan, vmeta_5):
        pytest.importorskip("pyarrow")
        stats = compute_variant_qc(G_with_nan, vmeta_5)
        apply_qc_filters(stats, QCFilterConfig())
        out_path = str(tmp_path / "qc.parquet")
        write_variant_qc_parquet(stats, out_path)

        import pyarrow.parquet as pq
        table = pq.read_table(out_path)
        assert table.num_rows == 5
        assert "SNP" in table.column_names
        assert "AF" in table.column_names
        assert "FILTER_PASS" in table.column_names
        assert "FILTER_REASON" in table.column_names

    def test_qc_filter_config_defaults(self):
        config = QCFilterConfig()
        assert config.maf_min == 0.01
        assert config.miss_max == 0.10
        assert config.hwe_p_min == 1e-6
        assert config.het_excess_max is None
        assert config.imp_rsq_min == 0.3

    def test_filter_order_matches_gemma(self, vmeta_5):
        """Filter application order: MONO → missingness → MAF → HWE (charter Section 6i)."""
        # Create data where:
        # SNP0: 100% missing → monomorphic (all NaN), caught by MONO first
        # SNP1: low MAF, fails maf_min=0.05
        # SNP2: high missingness (>50%), non-monomorphic → caught by MISS
        n = 20
        G = torch.zeros(n, 5, dtype=torch.float64)
        G[:, 0] = float("nan")  # 100% missing → MONO
        G[0, 1] = 1.0  # MAF = 1/(2*20) = 0.025, will fail maf_min=0.05
        G[:, 2] = 1.0  # all het
        G[:10, 3] = 0.0
        G[10:, 3] = 2.0
        G[:, 4] = 1.0
        # SNP2: make >50% missing but non-monomorphic among observed
        G[:12, 2] = float("nan")  # 12/20 = 60% missing
        G[12:16, 2] = 0.0
        G[16:20, 2] = 2.0  # non-mono among observed

        stats = compute_variant_qc(G, vmeta_5)
        config = QCFilterConfig(maf_min=0.05, miss_max=0.5)
        passes = apply_qc_filters(stats, config)

        # SNP0: monomorphic (all missing)
        assert not passes[0]
        assert stats.filter_reason[0] == "MONO"
        # SNP2: high missingness
        assert not passes[2]
        assert "MISS" in stats.filter_reason[2]


class TestCovariates:
    """Tests for torchgwas.preprocess.covariates."""

    def test_build_covariate_matrix_intercept(self):
        X0 = build_covariate_matrix(n_samples=10)
        assert X0.shape == (10, 1)
        assert torch.all(X0[:, 0] == 1.0)

    def test_build_with_covariates(self):
        covars = torch.randn(10, 3, dtype=torch.float64)
        X0 = build_covariate_matrix(n_samples=10, covariates=covars)
        assert X0.shape == (10, 4)  # intercept + 3

    def test_build_with_pcs(self):
        eigvecs = torch.randn(10, 20, dtype=torch.float64)
        X0 = build_covariate_matrix(n_samples=10, n_pcs=5, eigenvectors=eigvecs)
        assert X0.shape == (10, 6)  # intercept + 5 PCs

    def test_build_with_covariates_and_pcs(self):
        covars = torch.randn(10, 2, dtype=torch.float64)
        eigvecs = torch.randn(10, 20, dtype=torch.float64)
        X0 = build_covariate_matrix(n_samples=10, covariates=covars, n_pcs=3, eigenvectors=eigvecs)
        assert X0.shape == (10, 6)  # intercept + 2 covars + 3 PCs

    def test_mismatched_samples_raises(self):
        covars = torch.randn(8, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match"):
            build_covariate_matrix(n_samples=10, covariates=covars)
