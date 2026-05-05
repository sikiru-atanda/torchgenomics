"""Tier-2 behavioral coverage tests for ``torchgwas.postgwas`` MR, SMR/HEIDI,
and TWAS public symbols.

Bar (Pillar A spec section 4.3 Tier 2):
- Dataclasses: construct + round-trip + repr smoke.
- Functions: golden + edge + error tests.
  - ``mr_ivw``: closed-form IVW WLS estimate with hand-computed reference.
  - ``mr_egger``: zero-intercept invariant under no pleiotropy; intercept
    detects an additive constant.
  - ``mr_weighted_median``: robustness to a single outlier instrument.
  - ``mr_presso``: produces a valid global pleiotropy p-value in [0, 1].
  - ``mr_all``: returns one MRResult per method, with consensus estimates
    under valid instruments.
  - ``smr_test``: SMR chi-squared finite, p in [0, 1]; Wald-ratio causal
    estimate; degenerate beta_eqtl=0 falls back to inf SE.
  - ``heidi_test``: returns (p, stat, n_snps); empty / unmatched probe
    returns the documented null tuple.
  - ``smr_heidi``: returns SMRSummary aggregating both tests over a gene
    map.
  - ``twas_individual``: PrediXcan z-score significant when GReX recovers y.
  - ``twas_sumstat``: returns TWASResult with z and p; validates against
    a hand-computed S-PrediXcan formula on identity LD.

Covers 15 public symbols across 3 submodules:

- ``torchgwas.postgwas._mr.MRResult`` (dataclass)
- ``torchgwas.postgwas._mr.mr_ivw`` (function)
- ``torchgwas.postgwas._mr.mr_egger`` (function)
- ``torchgwas.postgwas._mr.mr_weighted_median`` (function)
- ``torchgwas.postgwas._mr.mr_presso`` (function)
- ``torchgwas.postgwas._mr.mr_all`` (function)
- ``torchgwas.postgwas._smr.SMRResult`` (dataclass)
- ``torchgwas.postgwas._smr.SMRSummary`` (dataclass)
- ``torchgwas.postgwas._smr.smr_test`` (function)
- ``torchgwas.postgwas._smr.heidi_test`` (function)
- ``torchgwas.postgwas._smr.smr_heidi`` (function)
- ``torchgwas.postgwas._twas.TWASGeneResult`` (dataclass)
- ``torchgwas.postgwas._twas.TWASResult`` (dataclass)
- ``torchgwas.postgwas._twas.twas_individual`` (function)
- ``torchgwas.postgwas._twas.twas_sumstat`` (function)

All re-exported via ``torchgwas.postgwas.__init__``.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from torchgwas.postgwas import (
    MRResult,
    SMRResult,
    SMRSummary,
    SumStats,
    TWASGeneResult,
    TWASResult,
    heidi_test,
    mr_all,
    mr_egger,
    mr_ivw,
    mr_presso,
    mr_weighted_median,
    smr_heidi,
    smr_test,
    twas_individual,
    twas_sumstat,
)

pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _build_mr_sumstats(
    bx: torch.Tensor,
    by: torch.Tensor,
    se_x: torch.Tensor | float = 0.05,
    se_y: torch.Tensor | float = 0.05,
    snp_prefix: str = "rs",
) -> tuple[SumStats, SumStats]:
    """Build aligned exposure / outcome SumStats for MR tests.

    Both SumStats share the same SNP order; betas come from ``bx`` / ``by``.
    """
    K = bx.shape[0]
    if not isinstance(se_x, torch.Tensor):
        se_x = torch.full((K,), float(se_x), dtype=torch.float64)
    if not isinstance(se_y, torch.Tensor):
        se_y = torch.full((K,), float(se_y), dtype=torch.float64)

    snps = [f"{snp_prefix}{i}" for i in range(K)]
    chr_list = ["1"] * K
    pos = list(range(1000, 1000 + K * 10, 10))
    a1 = ["A"] * K
    a2 = ["G"] * K
    n = torch.full((K,), 1000.0, dtype=torch.float64)
    af = torch.full((K,), 0.3, dtype=torch.float64)

    p_x = torch.erfc((bx / se_x).abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)
    p_y = torch.erfc((by / se_y).abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)

    expo = SumStats(
        chr=chr_list,
        pos=pos,
        snp=list(snps),
        a1=list(a1),
        a2=list(a2),
        beta=bx.to(torch.float64),
        se=se_x.to(torch.float64),
        p=p_x,
        n=n.clone(),
        af=af.clone(),
    )
    out = SumStats(
        chr=list(chr_list),
        pos=list(pos),
        snp=list(snps),
        a1=list(a1),
        a2=list(a2),
        beta=by.to(torch.float64),
        se=se_y.to(torch.float64),
        p=p_y,
        n=n.clone(),
        af=af.clone(),
    )
    return expo, out


def _build_smr_sumstats(
    snps: list[str],
    beta: torch.Tensor,
    se: torch.Tensor | float = 0.05,
) -> SumStats:
    """Build a SumStats for SMR/HEIDI tests."""
    K = len(snps)
    if not isinstance(se, torch.Tensor):
        se = torch.full((K,), float(se), dtype=torch.float64)
    z = beta / se
    p = torch.erfc(z.abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)
    return SumStats(
        chr=["1"] * K,
        pos=list(range(1000, 1000 + K * 100, 100)),
        snp=list(snps),
        a1=["A"] * K,
        a2=["G"] * K,
        beta=beta.to(torch.float64),
        se=se.to(torch.float64),
        p=p,
        n=torch.full((K,), 1000.0, dtype=torch.float64),
        af=torch.full((K,), 0.3, dtype=torch.float64),
    )


# ===========================================================================
# Dataclass smoke tests (5)
# ===========================================================================


class TestMRResult:
    """``MRResult`` carries a single MR estimator's output: method label,
    beta_hat / se / p / n_instruments, plus Egger- and PRESSO-specific
    fields with defaults."""

    def _make(self) -> MRResult:
        return MRResult(
            method="ivw",
            beta_hat=0.5,
            se=0.05,
            p_value=1e-10,
            n_instruments=10,
            intercept=0.01,
            intercept_se=0.02,
            intercept_p=0.6,
            egger_i2=0.95,
            global_rss=12.3,
            global_p=0.4,
            outlier_indices=[2, 5],
            n_outliers=2,
            beta_corrected=0.45,
            se_corrected=0.04,
            p_corrected=1e-9,
        )

    def test_construct(self):
        res = self._make()
        assert res.method == "ivw"
        assert res.beta_hat == pytest.approx(0.5)
        assert res.se == pytest.approx(0.05)
        assert res.n_instruments == 10
        assert res.intercept == pytest.approx(0.01)
        assert res.outlier_indices == [2, 5]
        assert res.n_outliers == 2

    def test_default_field_values(self):
        res = MRResult(method="ivw", beta_hat=0.1, se=0.05, p_value=0.05, n_instruments=3)
        assert res.intercept == 0.0
        assert res.intercept_se == 0.0
        assert res.intercept_p == pytest.approx(1.0)
        assert res.egger_i2 == 0.0
        assert res.global_rss == 0.0
        assert res.global_p == pytest.approx(1.0)
        assert res.outlier_indices == []
        assert res.n_outliers == 0
        assert res.beta_corrected == 0.0
        assert res.se_corrected == 0.0
        assert res.p_corrected == pytest.approx(1.0)

    def test_default_outlier_indices_independent(self):
        # Mutable default factory must not share state across instances.
        a = MRResult(method="ivw", beta_hat=0.1, se=0.05, p_value=0.05, n_instruments=3)
        b = MRResult(method="ivw", beta_hat=0.2, se=0.05, p_value=0.05, n_instruments=3)
        a.outlier_indices.append(99)
        assert b.outlier_indices == []

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "method",
            "beta_hat",
            "se",
            "p_value",
            "n_instruments",
            "intercept",
            "intercept_se",
            "intercept_p",
            "egger_i2",
            "global_rss",
            "global_p",
            "outlier_indices",
            "n_outliers",
            "beta_corrected",
            "se_corrected",
            "p_corrected",
        }
        res2 = MRResult(**{f.name: getattr(res, f.name) for f in fields(res)})
        assert res2 == res

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "MRResult" in s
        assert "method='ivw'" in s


class TestSMRResult:
    """``SMRResult`` carries per-gene SMR + HEIDI output: gene_id /
    probe_snp / beta_smr / se_smr / p_smr / chi2_smr / beta_gwas /
    beta_eqtl / p_heidi / n_heidi_snps / heidi_stat."""

    def _make(self) -> SMRResult:
        return SMRResult(
            gene_id="GENE1",
            probe_snp="rs100",
            beta_smr=0.4,
            se_smr=0.05,
            p_smr=1e-12,
            chi2_smr=49.0,
            beta_gwas=0.2,
            beta_eqtl=0.5,
            p_heidi=0.3,
            n_heidi_snps=10,
            heidi_stat=11.5,
        )

    def test_construct(self):
        res = self._make()
        assert res.gene_id == "GENE1"
        assert res.probe_snp == "rs100"
        assert res.beta_smr == pytest.approx(0.4)
        assert res.chi2_smr == pytest.approx(49.0)
        assert res.n_heidi_snps == 10

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "gene_id",
            "probe_snp",
            "beta_smr",
            "se_smr",
            "p_smr",
            "chi2_smr",
            "beta_gwas",
            "beta_eqtl",
            "p_heidi",
            "n_heidi_snps",
            "heidi_stat",
        }
        res2 = SMRResult(**{f.name: getattr(res, f.name) for f in fields(res)})
        assert res2 == res

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "SMRResult" in s
        assert "gene_id='GENE1'" in s


class TestSMRSummary:
    """``SMRSummary`` aggregates per-gene SMR results plus counts of total
    genes tested, SMR-significant, and HEIDI-pass."""

    def _make(self) -> SMRSummary:
        per_gene = SMRResult(
            gene_id="G1",
            probe_snp="rs1",
            beta_smr=0.1,
            se_smr=0.05,
            p_smr=0.04,
            chi2_smr=4.2,
            beta_gwas=0.05,
            beta_eqtl=0.5,
            p_heidi=0.6,
            n_heidi_snps=5,
            heidi_stat=2.5,
        )
        return SMRSummary(
            results=[per_gene],
            n_genes_tested=1,
            n_significant_smr=1,
            n_pass_heidi=1,
        )

    def test_construct(self):
        s = self._make()
        assert len(s.results) == 1
        assert s.n_genes_tested == 1
        assert s.n_significant_smr == 1
        assert s.n_pass_heidi == 1
        assert s.results[0].gene_id == "G1"

    def test_round_trip_via_dataclass_fields(self):
        s = self._make()
        names = {f.name for f in fields(s)}
        assert names == {
            "results",
            "n_genes_tested",
            "n_significant_smr",
            "n_pass_heidi",
        }
        s2 = SMRSummary(**{f.name: getattr(s, f.name) for f in fields(s)})
        assert s2 == s

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "SMRSummary" in s


class TestTWASGeneResult:
    """``TWASGeneResult`` carries per-gene TWAS output: gene_id, z_twas,
    p_twas, n_cis_snps, optional r2_model, top_weight_snp."""

    def _make(self) -> TWASGeneResult:
        return TWASGeneResult(
            gene_id="GeneX",
            z_twas=4.5,
            p_twas=6.8e-6,
            n_cis_snps=8,
            r2_model=0.12,
            top_weight_snp="rs7",
        )

    def test_construct(self):
        g = self._make()
        assert g.gene_id == "GeneX"
        assert g.z_twas == pytest.approx(4.5)
        assert g.p_twas == pytest.approx(6.8e-6)
        assert g.n_cis_snps == 8
        assert g.r2_model == pytest.approx(0.12)
        assert g.top_weight_snp == "rs7"

    def test_r2_model_optional_none(self):
        g = TWASGeneResult(
            gene_id="A",
            z_twas=0.0,
            p_twas=1.0,
            n_cis_snps=1,
            r2_model=None,
            top_weight_snp="rs0",
        )
        assert g.r2_model is None

    def test_round_trip_via_dataclass_fields(self):
        g = self._make()
        names = {f.name for f in fields(g)}
        assert names == {
            "gene_id",
            "z_twas",
            "p_twas",
            "n_cis_snps",
            "r2_model",
            "top_weight_snp",
        }
        g2 = TWASGeneResult(**{f.name: getattr(g, f.name) for f in fields(g)})
        assert g2 == g

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "TWASGeneResult" in s
        assert "gene_id='GeneX'" in s


class TestTWASResult:
    """``TWASResult`` aggregates per-gene TWAS results plus counts of
    total genes tested and significant after correction."""

    def _make(self) -> TWASResult:
        g1 = TWASGeneResult(
            gene_id="A", z_twas=4.5, p_twas=6.8e-6, n_cis_snps=8,
            r2_model=0.1, top_weight_snp="rs1",
        )
        g2 = TWASGeneResult(
            gene_id="B", z_twas=0.5, p_twas=0.6, n_cis_snps=5,
            r2_model=None, top_weight_snp="rs2",
        )
        return TWASResult(genes=[g1, g2], n_genes_tested=2, n_significant=1)

    def test_construct(self):
        r = self._make()
        assert len(r.genes) == 2
        assert r.n_genes_tested == 2
        assert r.n_significant == 1
        assert r.genes[0].gene_id == "A"
        assert r.genes[1].r2_model is None

    def test_round_trip_via_dataclass_fields(self):
        r = self._make()
        names = {f.name for f in fields(r)}
        assert names == {"genes", "n_genes_tested", "n_significant"}
        r2 = TWASResult(**{f.name: getattr(r, f.name) for f in fields(r)})
        assert r2 == r

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "TWASResult" in s


# ===========================================================================
# mr_ivw
# ===========================================================================


class TestMrIvw:
    """IVW WLS no-intercept: beta = sum(w*bx*by)/sum(w*bx^2),
    SE = 1/sqrt(sum(w*bx^2)) (Burgess 2013 second-order),
    with overdispersion inflation when Cochran's Q > K-1."""

    def test_closed_form_three_instruments(self):
        # Deterministic 3-instrument case: bx=[1,2,3], by=[0.5,1,1.5],
        # se_y all equal -> theta=0.5 exactly, Q=0, no overdispersion.
        bx = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        by = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        se_x = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        se_y = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        expo, out = _build_mr_sumstats(bx, by, se_x=se_x, se_y=se_y)

        res = mr_ivw(expo, out)

        # Manual reference: w = 1/se_y^2 (constant); beta = sum(bx*by)/sum(bx^2)
        ref = (bx * by).sum() / (bx**2).sum()
        assert isinstance(res, MRResult)
        assert res.method == "ivw"
        assert res.beta_hat == pytest.approx(ref.item(), abs=1e-12)
        assert res.beta_hat == pytest.approx(0.5, abs=1e-12)
        # No heterogeneity -> Q=0, SE = 1/sqrt(sum(w*bx^2))
        ref_se = (1.0 / (1.0 / se_y**2 * bx**2).sum().sqrt()).item()
        assert res.se == pytest.approx(ref_se, abs=1e-12)
        # n_instruments matches the input.
        assert res.n_instruments == 3
        # PRESSO-specific fields are at defaults.
        assert res.intercept == 0.0
        assert res.outlier_indices == []

    def test_recovers_true_effect_with_noise(self):
        # 5 instruments, true theta=0.5, mild outcome noise.
        torch.manual_seed(0)
        bx = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64)
        noise = torch.randn(5, dtype=torch.float64) * 0.01
        by = 0.5 * bx + noise
        expo, out = _build_mr_sumstats(bx, by, se_x=0.1, se_y=0.05)
        res = mr_ivw(expo, out)
        # Estimate is within 2 SE of the truth.
        assert abs(res.beta_hat - 0.5) <= 2.0 * res.se
        assert res.n_instruments == 5

    def test_invalid_mismatched_lengths(self):
        bx = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        by_full = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        # Build them separately to violate alignment.
        expo, _ = _build_mr_sumstats(bx, bx)
        _, out = _build_mr_sumstats(by_full, by_full)
        assert expo.m != out.m
        with pytest.raises(ValueError):
            mr_ivw(expo, out)

    def test_invalid_too_few_instruments(self):
        bx = torch.tensor([1.0, 2.0], dtype=torch.float64)
        by = torch.tensor([0.5, 1.0], dtype=torch.float64)
        expo, out = _build_mr_sumstats(bx, by)
        with pytest.raises(ValueError):
            mr_ivw(expo, out)


# ===========================================================================
# mr_egger
# ===========================================================================


class TestMrEgger:
    """MR-Egger fits ``by = alpha + beta * bx``. Under no pleiotropy the
    intercept ``alpha`` is ~0; an additive constant on by inflates alpha."""

    def test_returns_MRResult(self):
        torch.manual_seed(0)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.005
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        res = mr_egger(expo, out)
        assert isinstance(res, MRResult)
        assert res.method == "egger"
        assert math.isfinite(res.beta_hat)
        assert math.isfinite(res.intercept)
        assert math.isfinite(res.egger_i2)

    def test_no_pleiotropy_intercept_zero(self):
        # 6 valid instruments, no pleiotropy: intercept ~ 0 within 2 SE.
        torch.manual_seed(1)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.005
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        res = mr_egger(expo, out)
        # Intercept is small and within 2 SE of zero.
        assert abs(res.intercept) <= 2.0 * res.intercept_se + 1e-3

    def test_with_pleiotropy_detects_offset(self):
        # Add a deterministic offset of 0.3 to all by; Egger intercept
        # should detect this. (IVW would absorb it into a biased slope.)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        offset = 0.3
        # Use small-noise data so the intercept is well-estimated.
        torch.manual_seed(2)
        by_base = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.001
        by = by_base + offset
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        res = mr_egger(expo, out)
        # Egger intercept recovers the offset (within tolerance) and
        # is much larger than the no-pleiotropy case.
        assert res.intercept == pytest.approx(offset, abs=0.1)
        assert abs(res.intercept) > 5.0 * res.intercept_se

    def test_invalid_too_few_instruments(self):
        bx = torch.tensor([1.0, 2.0], dtype=torch.float64)
        by = torch.tensor([0.5, 1.0], dtype=torch.float64)
        expo, out = _build_mr_sumstats(bx, by)
        with pytest.raises(ValueError):
            mr_egger(expo, out)


# ===========================================================================
# mr_weighted_median
# ===========================================================================


class TestMrWeightedMedian:
    """Weighted median is consistent when >= 50% of weight is from valid
    instruments. With one outlier the estimate stays near the consensus."""

    def test_returns_MRResult(self):
        bx = torch.linspace(0.5, 3.0, 5, dtype=torch.float64)
        by = 0.5 * bx
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        res = mr_weighted_median(expo, out, n_boot=200, seed=42)
        assert isinstance(res, MRResult)
        assert res.method == "weighted_median"
        assert math.isfinite(res.beta_hat)
        assert math.isfinite(res.se)
        assert res.se >= 0.0
        assert 0.0 <= res.p_value <= 1.0
        assert res.n_instruments == 5

    def test_robust_to_outliers(self):
        # 5 instruments: 4 imply theta=0.5, 1 outlier implies theta=5.
        bx = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float64)
        by_consensus = 0.5 * bx
        by = by_consensus.clone()
        by[2] = 5.0 * bx[2]  # outlier instrument with very different ratio
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)

        wm = mr_weighted_median(expo, out, n_boot=300, seed=42)
        ivw = mr_ivw(expo, out)

        # Weighted median is closer to consensus (0.5) than IVW.
        assert abs(wm.beta_hat - 0.5) < abs(ivw.beta_hat - 0.5)
        # And substantially closer in absolute terms.
        assert abs(wm.beta_hat - 0.5) < 0.5

    def test_invalid_too_few_instruments(self):
        bx = torch.tensor([1.0, 2.0], dtype=torch.float64)
        by = torch.tensor([0.5, 1.0], dtype=torch.float64)
        expo, out = _build_mr_sumstats(bx, by)
        with pytest.raises(ValueError):
            mr_weighted_median(expo, out)


# ===========================================================================
# mr_presso
# ===========================================================================


class TestMrPresso:
    """MR-PRESSO returns a global pleiotropy p-value, per-SNP outlier
    indices, and a corrected IVW after outlier removal."""

    def test_returns_MRResult(self):
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        # Small n_perm to keep the test fast.
        res = mr_presso(expo, out, n_perm=50, outlier_threshold=0.05, seed=42)
        assert isinstance(res, MRResult)
        assert res.method == "mr_presso"
        assert math.isfinite(res.beta_hat)

    def test_global_test_p_value_in_unit_interval(self):
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        res = mr_presso(expo, out, n_perm=50, outlier_threshold=0.05, seed=42)
        assert 0.0 <= res.global_p <= 1.0
        assert res.global_rss >= 0.0
        assert res.n_outliers == len(res.outlier_indices)
        # Corrected estimate is finite.
        assert math.isfinite(res.beta_corrected)

    def test_outlier_inflates_global_rss(self):
        # An obvious outlier produces a much larger observed global RSS
        # than a clean fit at the same instruments. We compare global_rss
        # across two PRESSO runs (clean vs. polluted) rather than relying
        # on the permutation null at small n_perm.
        torch.manual_seed(3)
        bx = torch.linspace(0.5, 4.0, 8, dtype=torch.float64)
        by_clean = 0.5 * bx + torch.randn(8, dtype=torch.float64) * 0.001
        by_dirty = by_clean.clone()
        by_dirty[3] = 5.0 * bx[3]  # strong outlier

        expo, out_clean = _build_mr_sumstats(bx, by_clean, se_y=0.05)
        _, out_dirty = _build_mr_sumstats(bx, by_dirty, se_y=0.05)

        res_clean = mr_presso(expo, out_clean, n_perm=20,
                              outlier_threshold=0.05, seed=42)
        res_dirty = mr_presso(expo, out_dirty, n_perm=20,
                              outlier_threshold=0.05, seed=42)
        # RSS with the outlier is much larger than RSS without.
        assert res_dirty.global_rss > res_clean.global_rss * 100
        assert math.isfinite(res_dirty.global_rss)

    # ─── Phase 44 follow-up: parametric LOO bootstrap parity tests ──────

    def test_both_nulls_run(self):
        """Both ``null="parametric"`` and ``null="permutation"`` execute
        successfully and return a valid MRResult."""
        torch.manual_seed(7)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.005
        expo, out = _build_mr_sumstats(bx, by, se_x=0.02, se_y=0.05)

        res_param = mr_presso(
            expo, out, n_perm=100, outlier_threshold=0.05, seed=42,
            null="parametric",
        )
        res_perm = mr_presso(
            expo, out, n_perm=100, outlier_threshold=0.05, seed=42,
            null="permutation",
        )
        assert isinstance(res_param, MRResult)
        assert isinstance(res_perm, MRResult)
        assert 0.0 <= res_param.global_p <= 1.0
        assert 0.0 <= res_perm.global_p <= 1.0
        # Raw IVW slope is the same regardless of null choice (it is
        # computed independently of the bootstrap).
        assert abs(res_param.beta_hat - res_perm.beta_hat) < 1e-12

    def test_parametric_default(self):
        """Default null is parametric, matching MRPRESSO 1.0 behavior."""
        torch.manual_seed(8)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.005
        expo, out = _build_mr_sumstats(bx, by, se_x=0.02, se_y=0.05)

        res_default = mr_presso(
            expo, out, n_perm=100, outlier_threshold=0.05, seed=42,
        )
        res_explicit = mr_presso(
            expo, out, n_perm=100, outlier_threshold=0.05, seed=42,
            null="parametric",
        )
        assert res_default.global_p == res_explicit.global_p
        assert res_default.outlier_indices == res_explicit.outlier_indices
        assert abs(res_default.global_rss - res_explicit.global_rss) < 1e-12

    def test_invalid_null_raises(self):
        """Unknown ``null`` arg raises ValueError."""
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        with pytest.raises(ValueError):
            mr_presso(expo, out, n_perm=10, null="bogus_null")

    def test_parametric_matches_mrpresso_reference(self):
        """Parametric LOO bootstrap matches MRPRESSO 1.0 / TwoSampleMR
        on the K=30 + 3-planted-pleiotropic-SNP fixture used by
        ``validation/external/twosamplemr/compare.py``.

        Reference values come from the committed
        ``twosamplemr_results.json`` produced by the R harness with
        seed=42, NbDistribution=1000.
        """
        # Reproduce the simulate_mr.R fixture exactly:
        #     theta_true = 0.5, K = 30, pleio_indices = {15, 18, 21}
        #     se_x = 0.02, se_y = 0.03, pleio_sigma = 0.15, seed = 42
        # We use the bundled fixture rather than re-simulating to keep the
        # test independent of any RNG impl differences between R and torch.
        from pathlib import Path
        import csv as _csv
        repo_root = Path(__file__).resolve().parents[1]
        sumstats_path = (
            repo_root / "validation" / "external" / "twosamplemr"
            / "data" / "sumstats.tsv"
        )
        if not sumstats_path.exists():
            pytest.skip(f"fixture not present: {sumstats_path}")

        rows = list(_csv.DictReader(open(sumstats_path), delimiter="\t"))
        bx = torch.tensor(
            [float(r["beta_exposure"]) for r in rows], dtype=torch.float64,
        )
        se_x = torch.tensor(
            [float(r["se_exposure"]) for r in rows], dtype=torch.float64,
        )
        by = torch.tensor(
            [float(r["beta_outcome"]) for r in rows], dtype=torch.float64,
        )
        se_y = torch.tensor(
            [float(r["se_outcome"]) for r in rows], dtype=torch.float64,
        )
        expo, out = _build_mr_sumstats(bx, by, se_x=se_x, se_y=se_y)

        res = mr_presso(
            expo, out, n_perm=1000, outlier_threshold=0.05, seed=42,
        )

        # MRPRESSO 1.0 reference (from twosamplemr_results.json):
        #   global_rss   = 77.6197
        #   raw_b        = 0.4570
        #   corrected_b  = 0.4377
        #   outliers     = {15, 21} (zero-based)
        #   global_p     = 0.001
        # We assert exact equality on RSS / β_raw / β_corrected (closed
        # form, no bootstrap noise) and require the outlier set to match
        # exactly (the parametric bootstrap is stable enough at K=30,
        # NbDistribution=1000 that the planted pleiotropic SNPs
        # 15 and 21 always pass the Bonferroni threshold).
        assert abs(res.global_rss - 77.6197) < 1e-3
        assert abs(res.beta_hat - 0.4570) < 1e-3
        assert abs(res.beta_corrected - 0.4377) < 1e-3
        assert set(res.outlier_indices) == {15, 21}
        # MC noise tolerance for global_p ~ 1/sqrt(1000) ≈ 3e-2.
        assert abs(res.global_p - 0.001) < 5e-2


# ===========================================================================
# mr_all
# ===========================================================================


class TestMrAll:
    """``mr_all`` runs IVW, Egger, weighted median, and MR-PRESSO and
    returns one MRResult per method."""

    def test_returns_list_with_all_methods(self):
        torch.manual_seed(4)
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.005
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        results = mr_all(expo, out, n_boot=100, n_perm=50, seed=42)
        assert isinstance(results, list)
        assert len(results) == 4
        assert all(isinstance(r, MRResult) for r in results)
        methods = {r.method for r in results}
        assert methods == {"ivw", "egger", "weighted_median", "mr_presso"}

    def test_consistent_estimates_under_no_pleiotropy(self):
        # All instruments valid; estimators should agree to within 0.2.
        bx = torch.linspace(0.5, 3.0, 6, dtype=torch.float64)
        torch.manual_seed(5)
        by = 0.5 * bx + torch.randn(6, dtype=torch.float64) * 0.001
        expo, out = _build_mr_sumstats(bx, by, se_y=0.05)
        results = mr_all(expo, out, n_boot=100, n_perm=50, seed=42)
        betas = [r.beta_hat for r in results]
        spread = max(betas) - min(betas)
        assert spread < 0.2
        # All estimates close to the truth (0.5).
        assert all(abs(b - 0.5) < 0.1 for b in betas)


# ===========================================================================
# smr_test
# ===========================================================================


class TestSmrTest:
    """SMR chi^2 = (z_g^2 * z_e^2) / (z_g^2 + z_e^2); causal estimate is
    Wald ratio beta_g / beta_e with delta-method SE."""

    def test_smoke_finite_outputs(self):
        # GWAS and eQTL share probe SNP rs100 with strong signals.
        beta_g = torch.tensor([0.2, 0.05, 0.05], dtype=torch.float64)
        beta_e = torch.tensor([0.5, 0.1, 0.1], dtype=torch.float64)
        snps = ["rs100", "rs101", "rs102"]
        gwas = _build_smr_sumstats(snps, beta_g, se=0.04)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.04)

        res = smr_test(gwas, eqtl, gene_id="GENE1",
                       probe_snp="rs100",
                       eqtl_p_threshold=1.0)
        assert isinstance(res, SMRResult)
        assert res.gene_id == "GENE1"
        assert res.probe_snp == "rs100"
        assert math.isfinite(res.chi2_smr)
        assert 0.0 <= res.p_smr <= 1.0
        # Wald ratio reference.
        assert res.beta_smr == pytest.approx(0.2 / 0.5, abs=1e-12)
        assert res.beta_gwas == pytest.approx(0.2)
        assert res.beta_eqtl == pytest.approx(0.5)
        # SMR-only call leaves HEIDI fields at defaults.
        assert res.p_heidi == pytest.approx(1.0)
        assert res.n_heidi_snps == 0

    def test_chi2_formula_closed_form(self):
        # Probe SNP with z_g and z_e known by construction (se=0.05).
        beta_g = torch.tensor([0.5, 0.05], dtype=torch.float64)
        beta_e = torch.tensor([0.4, 0.1], dtype=torch.float64)
        snps = ["rsX", "rsY"]
        gwas = _build_smr_sumstats(snps, beta_g, se=0.05)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.05)
        res = smr_test(gwas, eqtl, gene_id="G", probe_snp="rsX",
                       eqtl_p_threshold=1.0)
        z_g = (0.5 / 0.05) ** 2
        z_e = (0.4 / 0.05) ** 2
        ref = (z_g * z_e) / (z_g + z_e)
        assert res.chi2_smr == pytest.approx(ref, abs=1e-9)

    def test_zero_beta_eqtl_returns_inf_se(self):
        # beta_eqtl ~ 0 -> Wald ratio ill-defined; impl returns 0 / inf.
        beta_g = torch.tensor([0.2, 0.1, 0.05], dtype=torch.float64)
        beta_e = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
        snps = ["rs100", "rs101", "rs102"]
        gwas = _build_smr_sumstats(snps, beta_g, se=0.05)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.05)
        res = smr_test(gwas, eqtl, gene_id="G", probe_snp="rs100",
                       eqtl_p_threshold=1.0)
        assert res.beta_smr == 0.0
        assert math.isinf(res.se_smr)

    def test_no_qualifying_eqtl_returns_null(self):
        # All eQTL p > threshold -> probe selection fails; return null result.
        beta_g = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)
        beta_e = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
        snps = ["rs1", "rs2", "rs3"]
        gwas = _build_smr_sumstats(snps, beta_g, se=0.5)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.5)
        res = smr_test(gwas, eqtl, gene_id="G",
                       eqtl_p_threshold=1e-300)
        assert isinstance(res, SMRResult)
        assert res.probe_snp == ""
        assert res.p_smr == pytest.approx(1.0)
        assert math.isinf(res.se_smr)


# ===========================================================================
# heidi_test
# ===========================================================================


class TestHeidiTest:
    """HEIDI test: returns (p_heidi, heidi_stat, n_snps). p_heidi is in
    [0, 1] when nearby SNPs have well-defined Wald ratios."""

    def test_returns_three_tuple(self):
        torch.manual_seed(6)
        K = 6
        snps = [f"rs{i}" for i in range(K)]
        # Probe is rs0; nearby SNPs have similar Wald ratios (single causal).
        beta_e = torch.tensor([0.5, 0.4, 0.45, 0.42, 0.38, 0.41],
                              dtype=torch.float64)
        beta_g = 0.4 * beta_e + torch.randn(K, dtype=torch.float64) * 0.001
        gwas = _build_smr_sumstats(snps, beta_g, se=0.04)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.04)
        out = heidi_test(gwas, eqtl, probe_snp="rs0",
                         nearby_snps=[s for s in snps if s != "rs0"],
                         max_snps=5)
        assert isinstance(out, tuple) and len(out) == 3
        p_h, stat, n = out
        assert 0.0 <= p_h <= 1.0
        assert math.isfinite(stat)
        assert stat >= 0.0
        assert n >= 1
        assert n <= 5

    def test_unknown_probe_returns_null(self):
        snps = ["rs0", "rs1", "rs2"]
        beta = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        eqtl = _build_smr_sumstats(snps, beta, se=0.05)
        # probe_snp not in either dataset -> documented null tuple.
        p_h, stat, n = heidi_test(gwas, eqtl, probe_snp="rs_missing",
                                  nearby_snps=snps)
        assert p_h == pytest.approx(1.0)
        assert stat == pytest.approx(0.0)
        assert n == 0

    def test_no_nearby_snps_returns_null(self):
        snps = ["rs0", "rs1", "rs2"]
        beta = torch.tensor([0.1, 0.05, 0.05], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        eqtl = _build_smr_sumstats(snps, beta, se=0.05)
        # No nearby SNPs -> n_snps == 0 and the documented null tuple.
        p_h, stat, n = heidi_test(gwas, eqtl, probe_snp="rs0",
                                  nearby_snps=[])
        assert n == 0
        assert p_h == pytest.approx(1.0)
        assert stat == pytest.approx(0.0)


# ===========================================================================
# smr_heidi
# ===========================================================================


class TestSmrHeidi:
    """``smr_heidi`` runs SMR + HEIDI across a gene_map and returns
    SMRSummary aggregating both tests."""

    def test_combined_returns_summary(self):
        torch.manual_seed(7)
        # Single gene with cis SNPs rs0 (probe) and rs1, rs2 (nearby).
        snps = ["rs0", "rs1", "rs2"]
        # Strong eQTL at rs0 (passes the p threshold), modest GWAS effects.
        beta_e = torch.tensor([0.6, 0.05, 0.05], dtype=torch.float64)
        # GWAS effects: probe nontrivial, nearby small.
        beta_g = torch.tensor([0.25, 0.02, 0.03], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta_g, se=0.04)
        eqtl = _build_smr_sumstats(snps, beta_e, se=0.04)
        gene_map = {"GENE_A": snps}

        s = smr_heidi(gwas, eqtl, gene_map,
                      eqtl_p_threshold=1e-3,
                      smr_p_threshold=0.5,
                      heidi_p_threshold=0.05,
                      heidi_max_snps=5)
        assert isinstance(s, SMRSummary)
        assert s.n_genes_tested == 1
        assert len(s.results) == 1
        res = s.results[0]
        assert isinstance(res, SMRResult)
        assert res.gene_id == "GENE_A"
        # SMR-significant counter is non-negative and bounded by tested.
        assert 0 <= s.n_significant_smr <= s.n_genes_tested
        assert 0 <= s.n_pass_heidi <= s.n_genes_tested

    def test_empty_gene_map(self):
        snps = ["rs0", "rs1"]
        beta = torch.tensor([0.1, 0.05], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        eqtl = _build_smr_sumstats(snps, beta, se=0.05)
        s = smr_heidi(gwas, eqtl, {})
        assert isinstance(s, SMRSummary)
        assert s.n_genes_tested == 0
        assert s.results == []
        assert s.n_significant_smr == 0
        assert s.n_pass_heidi == 0


# ===========================================================================
# twas_individual
# ===========================================================================


class TestTwasIndividual:
    """Individual-level (PrediXcan): GReX = X @ w, then OLS y ~ alpha +
    beta * GReX. When weights truly predict y from G, the z-score is
    significant."""

    def test_smoke_returns_TWASResult(self):
        torch.manual_seed(0)
        n, m = 50, 8
        G = torch.randn(n, m, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        snp_ids = [f"rs{i}" for i in range(m)]
        weights = {"GENE1": torch.randn(m, dtype=torch.float64)}
        snp_lists = {"GENE1": list(snp_ids)}

        res = twas_individual(G, y, weights, snp_lists, snp_ids)
        assert isinstance(res, TWASResult)
        assert res.n_genes_tested == 1
        assert len(res.genes) == 1
        g = res.genes[0]
        assert isinstance(g, TWASGeneResult)
        assert g.gene_id == "GENE1"
        assert math.isfinite(g.z_twas)
        assert 0.0 <= g.p_twas <= 1.0
        assert g.n_cis_snps == m
        assert g.r2_model is None
        assert g.top_weight_snp in snp_ids

    def test_perfect_prediction_significant(self):
        # y is constructed so it exactly matches GReX (up to tiny noise).
        torch.manual_seed(1)
        n, m = 80, 10
        G = torch.randn(n, m, dtype=torch.float64)
        true_w = torch.randn(m, dtype=torch.float64)
        grex = G @ true_w
        y = grex + torch.randn(n, dtype=torch.float64) * 0.01

        snp_ids = [f"rs{i}" for i in range(m)]
        weights = {"GENE1": true_w}
        snp_lists = {"GENE1": list(snp_ids)}

        res = twas_individual(G, y, weights, snp_lists, snp_ids)
        g = res.genes[0]
        # Effect is highly significant.
        assert g.p_twas < 1e-10
        # Top weight SNP is the one with the largest |true_w|.
        true_top_idx = int(true_w.abs().argmax().item())
        assert g.top_weight_snp == snp_ids[true_top_idx]

    def test_with_covariates(self):
        # Covariates are residualized out of y before the OLS test.
        torch.manual_seed(2)
        n, m = 40, 6
        G = torch.randn(n, m, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        cov = torch.randn(n, 2, dtype=torch.float64)
        # Add a column of ones to make the covariate matrix rank-2 with intercept.
        cov = torch.cat([torch.ones(n, 1, dtype=torch.float64), cov], dim=1)

        snp_ids = [f"rs{i}" for i in range(m)]
        weights = {"GENE1": torch.randn(m, dtype=torch.float64)}
        snp_lists = {"GENE1": list(snp_ids)}

        res = twas_individual(G, y, weights, snp_lists, snp_ids, covariates=cov)
        assert isinstance(res, TWASResult)
        assert len(res.genes) == 1
        assert math.isfinite(res.genes[0].z_twas)

    def test_invalid_empty_weights(self):
        n, m = 30, 5
        G = torch.randn(n, m, dtype=torch.float64)
        y = torch.randn(n, dtype=torch.float64)
        with pytest.raises(ValueError):
            twas_individual(G, y, {}, {}, [f"rs{i}" for i in range(m)])


# ===========================================================================
# twas_sumstat
# ===========================================================================


class TestTwasSumstat:
    """S-PrediXcan: z_twas = w^T z / sqrt(w^T Sigma w). On identity LD
    this collapses to a simple weighted z-statistic with a hand-checkable
    closed form."""

    def test_smoke_returns_TWASResult(self):
        m = 8
        snps = [f"rs{i}" for i in range(m)]
        z = torch.randn(m, dtype=torch.float64) * 1.5
        beta = z * 0.05
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        weights = {"GENE1": torch.randn(m, dtype=torch.float64)}
        snp_lists = {"GENE1": list(snps)}
        ld = {"GENE1": torch.eye(m, dtype=torch.float64)}
        r2_models = {"GENE1": 0.15}

        res = twas_sumstat(
            gwas, weights, snp_lists,
            ld_matrix=ld,
            r2_models=r2_models,
            p_threshold=0.05,
            correction="bonferroni",
        )
        assert isinstance(res, TWASResult)
        assert res.n_genes_tested == 1
        g = res.genes[0]
        assert g.gene_id == "GENE1"
        assert g.r2_model == pytest.approx(0.15)
        assert math.isfinite(g.z_twas)
        assert 0.0 <= g.p_twas <= 1.0

    def test_identity_ld_closed_form(self):
        # On identity LD, var_denom = sum(w^2) and z_twas = (w.z) / |w|.
        m = 5
        snps = [f"rs{i}" for i in range(m)]
        # Use deterministic GWAS effects so z = beta/se is known.
        beta = torch.tensor([0.10, 0.05, -0.05, 0.08, -0.03],
                            dtype=torch.float64)
        se_val = 0.04
        gwas = _build_smr_sumstats(snps, beta, se=se_val)
        z = beta / se_val
        w = torch.tensor([0.3, -0.2, 0.5, 0.1, -0.4], dtype=torch.float64)
        weights = {"GENE1": w}
        snp_lists = {"GENE1": list(snps)}
        ld = {"GENE1": torch.eye(m, dtype=torch.float64)}

        res = twas_sumstat(gwas, weights, snp_lists, ld_matrix=ld)
        ref_z = ((w * z).sum() / (w**2).sum().sqrt()).item()
        assert res.genes[0].z_twas == pytest.approx(ref_z, abs=1e-10)
        # Two-sided p from |z|.
        ref_p = float(torch.erfc(
            torch.tensor(abs(ref_z), dtype=torch.float64) / 2.0**0.5
        ).item())
        assert res.genes[0].p_twas == pytest.approx(ref_p, abs=1e-10)

    def test_no_ld_warns_and_uses_identity(self):
        # Without an LD matrix the impl warns and uses identity.
        m = 4
        snps = [f"rs{i}" for i in range(m)]
        beta = torch.tensor([0.1, -0.05, 0.07, 0.02], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        weights = {"GENE1": torch.tensor([0.3, 0.2, -0.1, 0.4],
                                         dtype=torch.float64)}
        snp_lists = {"GENE1": list(snps)}

        with pytest.warns(UserWarning):
            res = twas_sumstat(gwas, weights, snp_lists, ld_matrix=None)
        assert isinstance(res, TWASResult)
        assert len(res.genes) == 1
        assert math.isfinite(res.genes[0].z_twas)

    def test_invalid_empty_weights(self):
        snps = ["rs0", "rs1", "rs2"]
        beta = torch.tensor([0.1, 0.05, 0.0], dtype=torch.float64)
        gwas = _build_smr_sumstats(snps, beta, se=0.05)
        with pytest.raises(ValueError):
            twas_sumstat(gwas, {}, {})
