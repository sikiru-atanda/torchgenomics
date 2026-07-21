"""Tests for GWAS↔TWAS p-value combination kernels and the
``combine_gwas_twas`` integration entry point.

Six classical kernels (Fisher, Brown, Empirical Brown, HMP,
truncated product, min-p) get closed-form gates with hand-computed
expectations. The two existing-kernel wrappers (Cauchy, Stouffer) get
parity-with-source tests. The six novel methods get sanity gates
showing they reduce to known limiting behaviours. The integration
entry point gets a smoke test against a synthetic 5-gene fixture.
"""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import chi2 as _chi2

from torchgenomics.postgwas import (
    CombinedGeneResult,
    CombinedResult,
    SumStats,
    TWASGeneResult,
    TWASResult,
    brown_combined,
    brown_ld_aware,
    cauchy_combined,
    cauchy_multi_tissue_plus_lead_snp,
    combine_gwas_twas,
    empirical_brown_combined,
    fisher_combined,
    fisher_polyploid_gene_action,
    gwas_twas_conditional,
    gwas_twas_hyprcoloc_gated,
    harmonic_mean_p,
    min_p_combined,
    stouffer_combined,
    stouffer_r2_weighted,
    truncated_product,
)

# ===========================================================================
# Fisher's combined test (closed-form gates)
# ===========================================================================

class TestFisherCombined:
    def test_two_equal_pvalues_closed_form(self):
        """T = -2*2*log(0.5); p = chi2.sf(T, df=4)."""
        stat, p = fisher_combined([0.5, 0.5])
        expected_stat = -2 * 2 * math.log(0.5)
        assert abs(stat - expected_stat) < 1e-10
        assert abs(p - float(_chi2.sf(expected_stat, df=4))) < 1e-10

    def test_extreme_p_gives_low_combined(self):
        """A p-value of 1e-8 should drive combined p tiny."""
        stat, p = fisher_combined([1e-8, 0.5])
        # T = -2(log 1e-8 + log 0.5) ≈ 38.2; chi2.sf(38.2, df=4) ≈ 1e-7
        assert p < 1e-6

    def test_all_uniform_p_means_one(self):
        """All p=1.0 -> stat=0 -> p_combined=1."""
        stat, p = fisher_combined([1.0, 1.0, 1.0])
        assert stat == 0.0
        assert abs(p - 1.0) < 1e-10

    def test_lancaster_weighted_two_equal_p(self):
        """Lancaster weighting: with weights [1, 1] we recover unweighted."""
        stat_u, p_u = fisher_combined([0.1, 0.1])
        stat_w, p_w = fisher_combined([0.1, 0.1], weights=[1.0, 1.0])
        assert abs(stat_u - stat_w) < 1e-10
        assert abs(p_u - p_w) < 1e-10

    def test_empty_input(self):
        stat, p = fisher_combined([])
        assert stat == 0.0
        assert p == 1.0


# ===========================================================================
# Brown's correlated p-value test
# ===========================================================================

class TestBrownCombined:
    def test_identity_cov_reduces_to_fisher(self):
        """With diagonal cov (Var=4) and no off-diagonal correlation,
        Brown's effective df → 2k, matching Fisher exactly."""
        p = [0.1, 0.1, 0.1]
        cov_indep = torch.eye(3, dtype=torch.float64) * 4.0  # Var(-2 log U) = 4
        _, p_brown = brown_combined(p, cov_indep)
        _, p_fisher = fisher_combined(p)
        assert abs(p_brown - p_fisher) < 1e-10

    def test_positive_correlation_inflates_p(self):
        """With positive off-diagonal cov, Brown's effective df shrinks
        → combined p should be LARGER (less significant) than Fisher's."""
        p = [0.01, 0.01, 0.01]
        cov_corr = torch.full((3, 3), 2.0, dtype=torch.float64)
        cov_corr.fill_diagonal_(4.0)
        _, p_brown = brown_combined(p, cov_corr)
        _, p_fisher = fisher_combined(p)
        assert p_brown > p_fisher

    def test_cov_shape_validation(self):
        with pytest.raises(ValueError, match="cov shape"):
            brown_combined([0.1, 0.2], torch.eye(3, dtype=torch.float64))


# ===========================================================================
# Empirical Brown (Kost-McDermott / Poole)
# ===========================================================================

class TestEmpiricalBrownCombined:
    def test_independent_columns_close_to_fisher(self):
        """When the data columns are nearly independent, EBM should
        approximate Fisher's. Spearman r on random samples leaves some
        residual covariance; allow 50% relative deviation."""
        gen = torch.Generator().manual_seed(0)
        data = torch.randn(500, 3, generator=gen, dtype=torch.float64)
        _, p_ebm = empirical_brown_combined([0.05, 0.05, 0.05], data)
        _, p_fisher = fisher_combined([0.05, 0.05, 0.05])
        assert abs(p_ebm - p_fisher) / p_fisher < 0.50

    def test_correlated_columns_inflate_p(self):
        """If columns are highly correlated, EBM's effective df shrinks
        → combined p larger than Fisher's."""
        gen = torch.Generator().manual_seed(1)
        a = torch.randn(50, generator=gen, dtype=torch.float64)
        # Three nearly-identical columns.
        data = torch.stack([a, a + 0.01 * torch.randn(50, generator=gen, dtype=torch.float64),
                            a + 0.01 * torch.randn(50, generator=gen, dtype=torch.float64)], dim=1)
        _, p_ebm = empirical_brown_combined([0.01, 0.01, 0.01], data)
        _, p_fisher = fisher_combined([0.01, 0.01, 0.01])
        assert p_ebm > p_fisher


# ===========================================================================
# Harmonic mean p (Wilson 2019)
# ===========================================================================

class TestHarmonicMeanP:
    def test_equal_pvalues(self):
        """HMP of k copies of p equals p exactly."""
        hmp, _ = harmonic_mean_p([0.3, 0.3, 0.3])
        assert abs(hmp - 0.3) < 1e-12

    def test_smaller_p_drives_combined_down(self):
        hmp_a, p_a = harmonic_mean_p([0.5, 0.5])
        hmp_b, p_b = harmonic_mean_p([0.01, 0.5])
        assert p_b < p_a

    def test_weights_validation(self):
        with pytest.raises(ValueError, match="weights"):
            harmonic_mean_p([0.5, 0.5], weights=[1.0, -0.5])


# ===========================================================================
# Truncated product (Zaykin)
# ===========================================================================

class TestTruncatedProduct:
    def test_no_p_below_threshold_returns_one(self):
        """If no p < τ, the product is empty and combined p = 1."""
        W, p = truncated_product([0.5, 0.6, 0.9], tau=0.05)
        assert W == 1.0
        assert p == 1.0

    def test_below_threshold_gives_small_p(self):
        """All three p well below τ; combined p should be small."""
        _, p = truncated_product([1e-4, 1e-4, 1e-4], tau=0.05)
        assert p < 0.1

    def test_tau_validation(self):
        with pytest.raises(ValueError, match="tau"):
            truncated_product([0.5], tau=1.5)


# ===========================================================================
# Min-p (Tippett)
# ===========================================================================

class TestMinPCombined:
    def test_two_equal_p_unweighted(self):
        """min-p(0.5, 0.5) = 1 - (1 - 0.5)^2 = 0.75."""
        pm, p = min_p_combined([0.5, 0.5])
        assert pm == 0.5
        assert abs(p - 0.75) < 1e-12

    def test_extreme_p_drops_combined(self):
        """p = 1 - (1 - 1e-10)^2 ≈ 2e-10."""
        _, p = min_p_combined([1e-10, 0.5])
        assert p < 3e-10

    def test_weights_unit_equivalent_to_unweighted(self):
        _, p_u = min_p_combined([0.1, 0.2])
        _, p_w = min_p_combined([0.1, 0.2], weights=[1.0, 1.0])
        assert abs(p_u - p_w) < 1e-12


# ===========================================================================
# Cauchy / Stouffer wrappers (parity with the existing kernels)
# ===========================================================================

class TestCauchyAndStoufferWrappers:
    def test_cauchy_matches_existing_kernel(self):
        """Wrapper output equals direct cauchy_combination call."""
        from torchgenomics.stats.cauchy import cauchy_combination
        p = [0.05, 0.20, 0.001]
        _, p_wrap = cauchy_combined(p)
        p_direct = float(cauchy_combination(
            torch.tensor(p, dtype=torch.float64).unsqueeze(0)
        )[0].item())
        assert abs(p_wrap - p_direct) < 1e-12

    def test_stouffer_two_equal_p_zero_z(self):
        z, p = stouffer_combined([0.5, 0.5])
        assert abs(z) < 1e-12
        assert abs(p - 1.0) < 1e-12

    def test_stouffer_with_directions_cancel(self):
        """Equal magnitude z's with opposite signs cancel."""
        z, p = stouffer_combined([0.05, 0.05], direction=[1.0, -1.0])
        assert abs(z) < 1e-12
        assert abs(p - 1.0) < 1e-10


# ===========================================================================
# Novel method 1: stouffer_r2_weighted
# ===========================================================================

class TestStoufferR2Weighted:
    def test_equal_r2_equivalent_to_unweighted(self):
        """Equal weights (after √R²) should give the same answer as
        plain Stouffer."""
        p = [0.01, 0.05]
        _, p_unw = stouffer_combined(p)
        _, p_r2 = stouffer_r2_weighted(p, [0.5, 0.5])  # √0.5 each
        assert abs(p_unw - p_r2) < 1e-12

    def test_low_r2_tissue_downweighted(self):
        """A high-significance p in a low-R² tissue should NOT pull
        the combined as hard as the same p in a high-R² tissue."""
        _, p_low_r2 = stouffer_r2_weighted([1e-6, 0.5], [0.01, 0.5])
        _, p_hi_r2 = stouffer_r2_weighted([1e-6, 0.5], [0.5, 0.5])
        assert p_low_r2 > p_hi_r2


# ===========================================================================
# Novel method 2: brown_ld_aware
# ===========================================================================

class TestBrownLdAware:
    def test_identity_ld_reduces_to_fisher(self):
        """Zero off-diagonal LD → brown_ld_aware ≈ Fisher's."""
        p = [0.01, 0.01, 0.01]
        ld_id = torch.eye(3, dtype=torch.float64)
        _, p_ld = brown_ld_aware(p, ld_id)
        _, p_fisher = fisher_combined(p)
        assert abs(p_ld - p_fisher) < 1e-10

    def test_high_ld_inflates_p(self):
        p = [0.01, 0.01, 0.01]
        ld_hi = torch.full((3, 3), 0.8, dtype=torch.float64)
        ld_hi.fill_diagonal_(1.0)
        _, p_ld = brown_ld_aware(p, ld_hi)
        _, p_fisher = fisher_combined(p)
        assert p_ld > p_fisher


# ===========================================================================
# Novel method 3: fisher_polyploid_gene_action
# ===========================================================================

class TestFisherPolyploidGeneAction:
    def test_runs_on_2d_input(self):
        gen = torch.Generator().manual_seed(2)
        p = torch.rand(10, 3, generator=gen, dtype=torch.float64)
        stat, p_comb = fisher_polyploid_gene_action(p)
        assert 0.0 <= p_comb <= 1.0
        assert stat >= 0.0

    def test_all_significant_actions_recover_signal(self):
        """If every gene-action model is significant, the combined p
        should be small."""
        p = torch.full((5, 3), 1e-5, dtype=torch.float64)
        _, p_comb = fisher_polyploid_gene_action(p)
        assert p_comb < 1e-3

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            fisher_polyploid_gene_action(torch.tensor([0.1, 0.2]))


# ===========================================================================
# Novel method 4: cauchy_multi_tissue_plus_lead_snp
# ===========================================================================

class TestCauchyMultiTissuePlusLeadSnp:
    def test_strong_lead_snp_drives_combined_down(self):
        """A very small lead-SNP p should drive the augmented Cauchy
        combination toward small p."""
        _, p_a = cauchy_multi_tissue_plus_lead_snp(
            [0.5, 0.6, 0.4], lead_snp_gwas_p=1e-8,
        )
        _, p_b = cauchy_multi_tissue_plus_lead_snp(
            [0.5, 0.6, 0.4], lead_snp_gwas_p=0.5,
        )
        assert p_a < p_b

    def test_p_in_bounds(self):
        _, p = cauchy_multi_tissue_plus_lead_snp([0.5, 0.5], 0.5)
        assert 0.0 <= p <= 1.0


# ===========================================================================
# Novel method 5: gwas_twas_conditional
# ===========================================================================

class TestGwasTwasConditional:
    def test_mediation_returns_p_twas(self):
        """When conditional GWAS p ≫ marginal GWAS p (≥ 10×), the
        function flags 'mediated' and returns p_twas."""
        p, _, status = gwas_twas_conditional(
            p_gwas=1e-8, p_twas=1e-6, p_gwas_conditional=0.5,
        )
        assert status == "mediated"
        assert p == 1e-6

    def test_independence_combines_via_method(self):
        """When the conditional p is similar to marginal, treat as
        independent and combine."""
        p, _, status = gwas_twas_conditional(
            p_gwas=1e-3, p_twas=1e-3, p_gwas_conditional=1e-3,
            method="fisher",
        )
        assert status == "independent"
        _, p_expected = fisher_combined([1e-3, 1e-3])
        assert abs(p - p_expected) < 1e-12

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            gwas_twas_conditional(0.1, 0.1, 0.1, method="bogus")


# ===========================================================================
# Novel method 6: gwas_twas_hyprcoloc_gated
# ===========================================================================

class TestGwasTwasHyprcolocGated:
    def test_low_ppfc_returns_p_twas_only(self):
        p, ppfc, status = gwas_twas_hyprcoloc_gated(
            p_gwas=1e-5, p_twas=1e-3, ppfc=0.3, ppfc_threshold=0.8,
        )
        assert status == "uncolocalised"
        assert p == 1e-3

    def test_high_ppfc_combines(self):
        p, ppfc, status = gwas_twas_hyprcoloc_gated(
            p_gwas=1e-3, p_twas=1e-3, ppfc=0.95, ppfc_threshold=0.8,
            method="fisher",
        )
        assert status == "colocalised"
        _, p_expected = fisher_combined([1e-3, 1e-3])
        assert abs(p - p_expected) < 1e-12

    def test_ppfc_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ppfc"):
            gwas_twas_hyprcoloc_gated(0.1, 0.1, ppfc=1.5)


# ===========================================================================
# Entry point: combine_gwas_twas
# ===========================================================================

def _make_combine_fixture():
    """Build a (SumStats, TWASResult) fixture with 3 genes on chr 1."""
    # 6 SNPs: 2 each per gene region.
    snps = ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"]
    chr_ = ["1"] * 6
    # Genes at (1000-2000), (5000-6000), (10000-11000).
    pos = [1100, 1900, 5300, 5800, 10500, 10900]
    beta = [0.3, 0.25, 0.05, 0.02, -0.2, -0.15]
    se = [0.05] * 6
    z = [b / s for b, s in zip(beta, se)]
    p = [float(torch.erfc(torch.tensor(abs(zi) / math.sqrt(2.0))).item()) for zi in z]
    ss = SumStats(
        chr=chr_, pos=pos, snp=snps, a1=["A"] * 6, a2=["G"] * 6,
        beta=torch.tensor(beta, dtype=torch.float64),
        se=torch.tensor(se, dtype=torch.float64),
        p=torch.tensor(p, dtype=torch.float64),
        n=torch.full((6,), 5000.0, dtype=torch.float64),
    )
    twas = TWASResult(
        genes=[
            TWASGeneResult(
                gene_id="G1", z_twas=4.2, p_twas=2.7e-05, n_cis_snps=0,
                r2_model=0.25, top_weight_snp=None, beta=0.21, se=0.05,
                chr="1", start=1000, end=2000, gene_name="GENE_1",
            ),
            TWASGeneResult(
                gene_id="G2", z_twas=0.8, p_twas=0.42, n_cis_snps=0,
                r2_model=0.05, top_weight_snp=None, beta=0.04, se=0.05,
                chr="1", start=5000, end=6000, gene_name="GENE_2",
            ),
            TWASGeneResult(
                gene_id="G3", z_twas=-3.5, p_twas=4.7e-04, n_cis_snps=0,
                r2_model=0.30, top_weight_snp=None, beta=-0.18, se=0.05,
                chr="1", start=10000, end=11000, gene_name="GENE_3",
            ),
        ],
        n_genes_tested=3, n_significant=2,
    )
    return ss, twas


class TestCombineGwasTwas:
    def test_smoke_fisher(self):
        ss, twas = _make_combine_fixture()
        res = combine_gwas_twas(ss, twas, method="fisher", cis_window_bp=200)
        assert res.n_genes_tested == 3
        # G1 has strong GWAS + strong TWAS → combined should be smaller
        # than either alone.
        g1 = next(g for g in res.genes if g.gene_id == "G1")
        assert g1.p_combined < min(g1.p_gwas, g1.p_twas)
        assert g1.n_snps_in_gene == 2
        # G2 has weak GWAS + weak TWAS → combined remains modest.
        g2 = next(g for g in res.genes if g.gene_id == "G2")
        assert g2.p_combined > 1e-3

    def test_direction_concordance(self):
        ss, twas = _make_combine_fixture()
        res = combine_gwas_twas(ss, twas, method="fisher", cis_window_bp=200)
        # G1: beta_gwas > 0, beta_twas > 0 → concordant.
        g1 = next(g for g in res.genes if g.gene_id == "G1")
        assert g1.direction_concordance == 1
        # G3: beta_gwas < 0, beta_twas < 0 → concordant.
        g3 = next(g for g in res.genes if g.gene_id == "G3")
        assert g3.direction_concordance == 1

    def test_bh_correction_applied(self):
        ss, twas = _make_combine_fixture()
        res = combine_gwas_twas(ss, twas, method="fisher",
                                 cis_window_bp=200, correction="bh")
        # p_adj populated on every gene.
        for g in res.genes:
            assert g.p_adj is not None
            assert 0.0 <= g.p_adj <= 1.0

    def test_unknown_method_raises(self):
        ss, twas = _make_combine_fixture()
        with pytest.raises(ValueError, match="Unknown method"):
            combine_gwas_twas(ss, twas, method="not_a_method")

    def test_r2_weighted_uses_r2(self):
        """weights='r2_model' should propagate the TWAS r2_model into
        the per-gene combination, changing p_combined relative to
        unweighted."""
        ss, twas = _make_combine_fixture()
        res_unw = combine_gwas_twas(ss, twas, method="stouffer",
                                     cis_window_bp=200, weights=None)
        res_r2 = combine_gwas_twas(ss, twas, method="stouffer",
                                    cis_window_bp=200, weights="r2_model")
        # G2 has r2_model=0.05 (small) — its TWAS contribution should
        # be down-weighted relative to GWAS, shifting p_combined.
        g2_u = next(g for g in res_unw.genes if g.gene_id == "G2")
        g2_r = next(g for g in res_r2.genes if g.gene_id == "G2")
        assert abs(g2_u.p_combined - g2_r.p_combined) > 0.0

    def test_returns_dataclasses(self):
        ss, twas = _make_combine_fixture()
        res = combine_gwas_twas(ss, twas, method="fisher", cis_window_bp=200)
        assert isinstance(res, CombinedResult)
        for g in res.genes:
            assert isinstance(g, CombinedGeneResult)
