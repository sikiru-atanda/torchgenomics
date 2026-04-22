"""Phase 2/8: Polyploid preprocessing and gene-action encoding tests."""

from __future__ import annotations

import pytest
import torch

from torchgwas.preprocess.polyploid import (
    detect_ploidy,
    list_gene_action_models,
    recode_gene_action,
)


@pytest.fixture
def G_diploid():
    return torch.tensor([
        [0, 1, 2, 0],
        [1, 1, 0, 2],
        [2, 0, 1, 1],
    ], dtype=torch.float64)


@pytest.fixture
def G_tetra():
    """Tetraploid dosage: values in {0, 1, 2, 3, 4}."""
    return torch.tensor([
        [0, 1, 4, 2],
        [2, 3, 0, 1],
        [4, 0, 2, 3],
        [1, 2, 3, 4],
    ], dtype=torch.float64)


class TestPolyploidEncoding:
    def test_detect_ploidy_diploid(self, G_diploid):
        assert detect_ploidy(G_diploid) == 2

    def test_detect_ploidy_tetraploid(self, G_tetra):
        assert detect_ploidy(G_tetra) == 4

    def test_recode_additive(self, G_diploid):
        result = recode_gene_action(G_diploid, "additive", ploidy=2)
        torch.testing.assert_close(result, G_diploid)

    def test_recode_1dom_diploid(self, G_diploid):
        """1-dom: 1 if dosage >= 1, else 0."""
        result = recode_gene_action(G_diploid, "1-dom", ploidy=2)
        expected = torch.tensor([
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
        ], dtype=torch.float64)
        torch.testing.assert_close(result, expected)

    def test_recode_1dom_tetraploid(self, G_tetra):
        """1-dom for tetraploid: 1 if dosage >= 1."""
        result = recode_gene_action(G_tetra, "1-dom", ploidy=4)
        expected = (G_tetra >= 1).to(torch.float64)
        torch.testing.assert_close(result, expected)

    def test_recode_2dom_tetraploid(self, G_tetra):
        """2-dom for tetraploid: 1 if dosage >= 2."""
        result = recode_gene_action(G_tetra, "2-dom", ploidy=4)
        expected = (G_tetra >= 2).to(torch.float64)
        torch.testing.assert_close(result, expected)

    def test_recode_diplo_additive(self, G_tetra):
        """Diplo-additive: min(dosage, k - dosage) for k=4."""
        result = recode_gene_action(G_tetra, "diplo-additive", ploidy=4)
        expected = torch.min(G_tetra, 4 - G_tetra)
        torch.testing.assert_close(result, expected)

    def test_recode_overdominant(self, G_tetra):
        """Overdominant: 1 if 0 < dosage < k."""
        result = recode_gene_action(G_tetra, "overdominant", ploidy=4)
        expected = ((G_tetra > 0) & (G_tetra < 4)).to(torch.float64)
        torch.testing.assert_close(result, expected)

    def test_recode_general(self, G_diploid):
        """General model returns (n, m, k-1) one-hot tensor."""
        result = recode_gene_action(G_diploid, "general", ploidy=2)
        assert result.shape == (3, 4, 1)  # k-1 = 1 class (dosage=1)

    def test_gene_action_all_lists(self):
        """list_gene_action_models returns correct models for ploidy 4."""
        models = list_gene_action_models(4)
        assert "additive" in models
        assert "1-dom" in models
        assert "2-dom" in models
        assert "3-dom" in models
        assert "diplo-additive" in models
        assert "overdominant" in models
        assert "general" in models

    def test_recode_with_nan(self):
        """NaN values are preserved through recoding."""
        G = torch.tensor([[0, float("nan")], [1, 2]], dtype=torch.float64)
        result = recode_gene_action(G, "1-dom", ploidy=2)
        assert torch.isnan(result[0, 1])
        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0

    def test_invalid_model_raises(self, G_diploid):
        with pytest.raises(ValueError, match="Unknown"):
            recode_gene_action(G_diploid, "nonexistent", ploidy=2)

    def test_invalid_jdom_raises(self, G_diploid):
        with pytest.raises(ValueError, match="j-dom requires"):
            recode_gene_action(G_diploid, "5-dom", ploidy=2)


class TestMaxGenoFreqFilter:
    """Tests for compute_max_genotype_freq (charter Section 9g)."""

    def test_binary_nearly_monomorphic(self):
        """After 1-dom encoding, a marker where nearly all samples have dose >= 1
        should have max genotype freq close to 1."""
        from torchgwas.preprocess.qc import compute_max_genotype_freq

        # 99 samples with dose >= 1 (encoded as 1), 1 sample with dose 0
        G_encoded = torch.ones(100, 3, dtype=torch.float64)
        G_encoded[0, 0] = 0.0  # 1st marker: 99% frequency of class 1
        G_encoded[:50, 1] = 0.0  # 2nd marker: 50/50
        # 3rd marker: all 1s (100%)

        freqs = compute_max_genotype_freq(G_encoded, ploidy=4)
        assert freqs.shape == (3,)
        assert freqs[0].item() == pytest.approx(0.99, abs=1e-6)
        assert freqs[1].item() == pytest.approx(0.50, abs=1e-6)
        assert freqs[2].item() == pytest.approx(1.00, abs=1e-6)

    def test_multicolumn_general_model(self):
        """For general model with (n, m, k-1), check each column independently
        and return the max across columns per marker."""
        from torchgwas.preprocess.qc import compute_max_genotype_freq

        # Shape (10, 2, 3) — 2 markers, 3 columns each
        # Mix values so no column is all-zero (which would give max_freq=1.0)
        G_encoded = torch.zeros(10, 2, 3, dtype=torch.float64)
        # Marker 0: col0 has 9 ones + 1 zero (max=0.9), col1 has 6 ones + 4 zeros (max=0.6),
        #           col2 has 5 ones + 5 zeros (max=0.5)
        G_encoded[:9, 0, 0] = 1.0
        G_encoded[:6, 0, 1] = 1.0
        G_encoded[:5, 0, 2] = 1.0
        # Marker 1: col0 has 3+7 (max=0.7), col1 has 5+5, col2 has 4+6
        G_encoded[:3, 1, 0] = 1.0
        G_encoded[:5, 1, 1] = 1.0
        G_encoded[:4, 1, 2] = 1.0

        freqs = compute_max_genotype_freq(G_encoded, ploidy=4)
        assert freqs.shape == (2,)
        # Marker 0: max across cols is col0 = 0.90
        assert freqs[0].item() == pytest.approx(0.90, abs=1e-6)
        # Marker 1: max across cols is col0 = 0.70
        assert freqs[1].item() == pytest.approx(0.70, abs=1e-6)

    def test_nan_handling(self):
        """NaN samples should be excluded from frequency computation."""
        from torchgwas.preprocess.qc import compute_max_genotype_freq

        G_encoded = torch.ones(10, 1, dtype=torch.float64)
        G_encoded[0, 0] = 0.0
        G_encoded[1, 0] = float("nan")
        # 8 ones, 1 zero out of 9 valid → max freq = 8/9

        freqs = compute_max_genotype_freq(G_encoded, ploidy=2)
        assert freqs[0].item() == pytest.approx(8.0 / 9.0, abs=1e-6)

    def test_additive_dosage(self):
        """Additive encoding with varied dosages — max freq is most common class."""
        from torchgwas.preprocess.qc import compute_max_genotype_freq

        G = torch.tensor([[0], [0], [1], [1], [1], [2], [2], [2], [2], [2]], dtype=torch.float64)
        freqs = compute_max_genotype_freq(G, ploidy=2)
        # dosage 2 appears 5/10 = 0.5
        assert freqs[0].item() == pytest.approx(0.50, abs=1e-6)


class TestPolyploidQC:
    """Tests for polyploid-specific QC: dosage class freq, per-class het,
    double-reduction HWE, and dosage certainty filter."""

    @pytest.fixture
    def vmeta_4(self):
        from torchgwas.models.base import VariantMeta
        return VariantMeta(
            snp=["s1", "s2", "s3", "s4"],
            chr=["1", "1", "2", "2"],
            pos=[100, 200, 100, 200],
            a1=["A", "A", "A", "A"],
            a2=["T", "T", "T", "T"],
        )

    @pytest.fixture
    def G_tetra_qc(self):
        """Tetraploid dosages (0-4) for 20 samples, 4 markers."""
        torch.manual_seed(99)
        n = 20
        # Marker 0: balanced (HWE-like)
        # Marker 1: excess homozygosity (suggestive of DR)
        # Marker 2: near-monomorphic (nearly all dose=4)
        # Marker 3: missing data
        G = torch.zeros(n, 4, dtype=torch.float64)

        # Marker 0: roughly binomial(4, 0.4) distribution
        G[:, 0] = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4], dtype=torch.float64)

        # Marker 1: excess dose=0 and dose=4 (double reduction signature)
        G[:, 1] = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4], dtype=torch.float64)

        # Marker 2: nearly monomorphic — 19 samples dose=4, 1 sample dose=3
        G[:, 2] = torch.full((n,), 4.0, dtype=torch.float64)
        G[0, 2] = 3.0

        # Marker 3: some missing
        G[:, 3] = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.float64)
        G[18, 3] = float("nan")
        G[19, 3] = float("nan")

        return G

    def test_dosage_class_freq_shape(self, G_tetra_qc, vmeta_4):
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        assert stats.dosage_class_freq is not None
        assert stats.dosage_class_freq.shape == (4, 5)  # (m, k+1)

    def test_dosage_class_freq_sums_to_one(self, G_tetra_qc, vmeta_4):
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        row_sums = stats.dosage_class_freq.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(4, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_dosage_class_freq_monomorphic(self, G_tetra_qc, vmeta_4):
        """Marker 2 is nearly monomorphic — dose=4 should dominate."""
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # dose=4 has 19/20 = 0.95
        assert stats.dosage_class_freq[2, 4].item() == pytest.approx(0.95, abs=1e-6)
        # dose=3 has 1/20 = 0.05
        assert stats.dosage_class_freq[2, 3].item() == pytest.approx(0.05, abs=1e-6)

    def test_het_per_class_shape(self, G_tetra_qc, vmeta_4):
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        assert stats.het_per_class is not None
        assert stats.het_per_class.shape == (4, 3)  # (m, k-1) = (4, 3)

    def test_het_per_class_values(self, G_tetra_qc, vmeta_4):
        """Per-class het should match dosage_class_freq for doses 1,2,3."""
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # het_per_class[:, h] = dosage_class_freq[:, h+1] for h in 0..k-2
        for h in range(3):
            torch.testing.assert_close(
                stats.het_per_class[:, h],
                stats.dosage_class_freq[:, h + 1],
                atol=1e-10, rtol=0,
            )

    def test_het_per_class_sums_to_total_het(self, G_tetra_qc, vmeta_4):
        """Sum of per-class het should equal total observed het."""
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        total_het_from_classes = stats.het_per_class.sum(dim=1)
        torch.testing.assert_close(total_het_from_classes, stats.het, atol=1e-10, rtol=0)

    def test_double_reduction_alpha_shape(self, G_tetra_qc, vmeta_4):
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        assert stats.double_reduction_alpha is not None
        assert stats.double_reduction_alpha.shape == (4,)
        # Alpha should be in [0, 1/6]
        assert torch.all(stats.double_reduction_alpha >= 0)
        assert torch.all(stats.double_reduction_alpha <= 1.0 / 6.0 + 1e-10)

    def test_hwe_p_dr_shape(self, G_tetra_qc, vmeta_4):
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        assert stats.hwe_p_dr is not None
        assert stats.hwe_p_dr.shape == (4,)
        assert torch.all(stats.hwe_p_dr >= 0)
        assert torch.all(stats.hwe_p_dr <= 1)

    def test_double_reduction_higher_for_excess_hom(self, G_tetra_qc, vmeta_4):
        """Marker 1 has excess homozygosity — should have higher DR alpha than marker 0."""
        from torchgwas.preprocess.qc import compute_variant_qc
        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # Marker 1 was designed with excess dose=0 and dose=4
        assert stats.double_reduction_alpha[1] >= stats.double_reduction_alpha[0]

    def test_dr_not_computed_for_diploid(self):
        """DR fields should be None for diploid data."""
        from torchgwas.models.base import VariantMeta
        from torchgwas.preprocess.qc import compute_variant_qc
        G = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.float64)
        vmeta = VariantMeta(snp=["s1", "s2"], chr=["1", "1"], pos=[1, 2], a1=["A", "A"], a2=["T", "T"])
        stats = compute_variant_qc(G, vmeta, ploidy=2)
        assert stats.dosage_class_freq is None
        assert stats.het_per_class is None
        assert stats.double_reduction_alpha is None
        assert stats.hwe_p_dr is None

    def test_dosage_certainty_filter(self):
        """Markers with high mean dosage variance should be filtered."""
        from torchgwas.models.base import VariantMeta
        from torchgwas.preprocess.qc import QCFilterConfig, apply_qc_filters, compute_variant_qc

        n, m, ploidy = 40, 3, 4
        # Well-behaved genotypes that will pass MAF/HWE: balanced dosages
        G = torch.zeros(n, m, dtype=torch.float64)
        for col in range(m):
            # Roughly binomial(4, 0.5): 8 classes spread
            G[:, col] = torch.tensor(
                [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
                dtype=torch.float64,
            )
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["T"] * m,
        )

        # Create dosage probs: marker 0 certain, marker 1 uncertain, marker 2 very uncertain
        probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
        # Marker 0: all probability on one class → low variance
        for i in range(n):
            d = int(G[i, 0].item())
            probs[i, 0, d] = 1.0
        # Marker 1: spread probability → high variance
        probs[:, 1, :] = 1.0 / (ploidy + 1)
        # Marker 2: also spread
        probs[:, 2, :] = 1.0 / (ploidy + 1)

        stats = compute_variant_qc(G, vmeta, ploidy=ploidy, dosage_probs=probs)
        assert stats.mean_dosage_var is not None
        assert stats.mean_dosage_var.shape == (m,)
        # Marker 0 should have ~0 variance, markers 1&2 should have higher
        assert stats.mean_dosage_var[0].item() < 0.01
        assert stats.mean_dosage_var[1].item() > 0.5

        # Apply dosage certainty filter (relax other filters)
        config = QCFilterConfig(dosage_var_max=0.1, maf_min=0.0, hwe_p_min=0.0)
        passes = apply_qc_filters(stats, config)
        assert passes[0]  # certain marker passes
        assert not passes[1]  # uncertain marker fails
        assert "DOSAGE_VAR" in stats.filter_reason[1]

    def test_dr_hwe_filter_integration(self, G_tetra_qc, vmeta_4):
        """use_double_reduction_hwe should use hwe_p_dr instead of hwe_p."""
        from torchgwas.preprocess.qc import QCFilterConfig, apply_qc_filters, compute_variant_qc

        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)

        # With standard HWE test
        config_std = QCFilterConfig(hwe_p_min=0.5, use_double_reduction_hwe=False)
        passes_std = apply_qc_filters(stats, config_std)

        # Reset filter state
        stats.filter_pass = [True] * len(stats.snp)
        stats.filter_reason = ["PASS"] * len(stats.snp)

        # With DR-aware HWE test (should be more lenient for markers with DR)
        config_dr = QCFilterConfig(hwe_p_min=0.5, use_double_reduction_hwe=True)
        passes_dr = apply_qc_filters(stats, config_dr)

        # Both should produce valid boolean masks
        assert passes_std.shape == (4,)
        assert passes_dr.shape == (4,)

    def test_polyploid_maf_correct(self, G_tetra_qc, vmeta_4):
        """MAF should be computed as mean(dosage)/(ploidy), not mean(dosage)/2."""
        from torchgwas.preprocess.qc import compute_variant_qc

        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # All MAF values should be in [0, 0.5]
        assert torch.all(stats.maf >= 0)
        assert torch.all(stats.maf <= 0.5)
        # AF = mean(dosage) / 4
        expected_af = G_tetra_qc[:, 0].mean() / 4.0
        assert stats.af[0].item() == pytest.approx(expected_af.item(), abs=1e-6)

    def test_polyploid_mac_uses_ploidy(self, G_tetra_qc, vmeta_4):
        """MAC = ploidy * n_obs * MAF for polyploids."""
        from torchgwas.preprocess.qc import compute_variant_qc

        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # Marker 0: no missing, so n_obs = 20
        expected_mac = 4 * 20 * stats.maf[0].item()
        assert stats.mac[0].item() == pytest.approx(expected_mac, abs=1e-4)

    def test_polyploid_het_counts_all_het_classes(self, G_tetra_qc, vmeta_4):
        """Observed het for polyploid should count all doses 1..k-1."""
        from torchgwas.preprocess.qc import compute_variant_qc

        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # For marker 0: doses 1,2,3 are heterozygous
        n = 20
        het_count = sum(1 for d in G_tetra_qc[:, 0] if 0 < d.item() < 4)
        expected_het = het_count / n
        assert stats.het[0].item() == pytest.approx(expected_het, abs=1e-6)

    def test_polyploid_hwe_uses_binomial_expansion(self, vmeta_4):
        """HWE test for polyploid uses C(k,d)*p^d*q^(k-d) expected frequencies."""

        from torchgwas.models.base import VariantMeta
        from torchgwas.preprocess.qc import compute_variant_qc

        # Create data that perfectly matches HWE expectation for p=0.5, k=4
        # Expected: C(4,d)*0.5^4 = [1, 4, 6, 4, 1]/16
        n = 160
        G = torch.zeros(n, 1, dtype=torch.float64)
        counts = [10, 40, 60, 40, 10]  # exactly HWE proportions
        idx = 0
        for d, c in enumerate(counts):
            G[idx:idx + c, 0] = float(d)
            idx += c

        vmeta = VariantMeta(snp=["s1"], chr=["1"], pos=[1], a1=["A"], a2=["T"])
        stats = compute_variant_qc(G, vmeta, ploidy=4)
        # Perfect HWE → high p-value
        assert stats.hwe_p[0].item() > 0.5

    def test_missing_data_excluded_from_polyploid_qc(self, G_tetra_qc, vmeta_4):
        """Missing samples should not affect dosage class freq computation."""
        from torchgwas.preprocess.qc import compute_variant_qc

        stats = compute_variant_qc(G_tetra_qc, vmeta_4, ploidy=4)
        # Marker 3 has 2 missing → n_obs = 18
        assert stats.n_obs[3].item() == 18
        # Dosage class freq should still sum to 1
        assert stats.dosage_class_freq[3].sum().item() == pytest.approx(1.0, abs=1e-10)
