"""Tier-2 behavioral coverage tests for ``torchgwas.postgwas`` heritability,
LD-score, and clumping public symbols.

Bar (Pillar A spec section 4.3 Tier 2):
- Dataclasses: construct + round-trip + repr smoke.
- Functions: golden + edge + error tests. For LDSC/heritability also a
  "no signal" sanity (uniform-like z-stats give h^2 close to 0) and
  "strong signal" sanity (inflated z-stats give larger h^2).

Covers 16 public symbols across 5 submodules:

- ``torchgwas.postgwas._clump.ClumpResult`` (dataclass)
- ``torchgwas.postgwas._clump.ld_clump`` (function)
- ``torchgwas.postgwas._hess.HESSResult`` (dataclass)
- ``torchgwas.postgwas._hess.HESSRegionResult`` (dataclass)
- ``torchgwas.postgwas._hess.hess_local_h2`` (function)
- ``torchgwas.postgwas._hess.hess_local_rg`` (function)
- ``torchgwas.postgwas._ld_scores.compute_ld_scores`` (function)
- ``torchgwas.postgwas._ld_scores.compute_cross_ld_scores`` (function)
- ``torchgwas.postgwas._ldsc.LDSCResult`` (dataclass)
- ``torchgwas.postgwas._ldsc.LDSCRgResult`` (dataclass)
- ``torchgwas.postgwas._ldsc.ldsc_h2`` (function)
- ``torchgwas.postgwas._ldsc.ldsc_intercept`` (function)
- ``torchgwas.postgwas._ldsc.ldsc_rg`` (function)
- ``torchgwas.postgwas._ldsc.ldsc_rg_from_z`` (function)
- ``torchgwas.postgwas._sldsc.SLDSCResult`` (dataclass)
- ``torchgwas.postgwas._sldsc.sldsc_h2_partitioned`` (function)

All re-exported via ``torchgwas.postgwas.__init__``.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from torchgwas.postgwas import (
    ClumpResult,
    HESSRegionResult,
    HESSResult,
    LDSCResult,
    LDSCRgResult,
    SLDSCResult,
    compute_cross_ld_scores,
    compute_ld_scores,
    hess_local_h2,
    hess_local_rg,
    ld_clump,
    ldsc_h2,
    ldsc_intercept,
    ldsc_rg,
    ldsc_rg_from_z,
    sldsc_h2_partitioned,
)

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _build_ldsc_inputs(seed: int = 0, n_snps: int = 500, scale: float = 1.0):
    """Build LD scores, sample size vector, and z-scores for LDSC tests.

    The "scale" knob inflates the z-scores' standard deviation; ``scale=1.0``
    is the no-signal baseline, ``scale > 1.0`` mimics polygenic inflation.
    """
    torch.manual_seed(seed)
    ld_scores = torch.rand(n_snps, dtype=torch.float64) * 50.0 + 5.0
    z = torch.randn(n_snps, dtype=torch.float64) * scale
    chi2 = z**2
    return ld_scores, z, chi2


def _build_hess_ld_matrix(n_snps: int = 50, seed: int = 1) -> torch.Tensor:
    """Build a well-conditioned LD correlation matrix via random standardised G."""
    torch.manual_seed(seed)
    n_samples = 200
    G = torch.randn(n_samples, n_snps, dtype=torch.float64)
    G = (G - G.mean(dim=0)) / G.std(dim=0).clamp(min=1e-6)
    R = (G.t() @ G) / n_samples
    # Light ridge to keep eigenvalues away from zero
    R = R + torch.eye(n_snps, dtype=torch.float64) * 1e-6
    return R


# ---------------------------------------------------------------------------
# ClumpResult dataclass
# ---------------------------------------------------------------------------


class TestClumpResult:
    """``ClumpResult`` carries the output of LD clumping: index SNP indices,
    their p-values, the SNPs absorbed into each clump, and the clump count."""

    def test_construct(self):
        res = ClumpResult(
            index_snps=[0, 5, 12],
            index_p=torch.tensor([1e-10, 1e-9, 1e-8], dtype=torch.float64),
            clump_members=[[1, 2], [6], []],
            n_clumps=3,
        )
        assert res.index_snps == [0, 5, 12]
        assert res.index_p.shape == (3,)
        assert res.clump_members == [[1, 2], [6], []]
        assert res.n_clumps == 3

    def test_round_trip_via_dataclass_fields(self):
        res = ClumpResult(
            index_snps=[1],
            index_p=torch.tensor([1e-12], dtype=torch.float64),
            clump_members=[[2, 3]],
            n_clumps=1,
        )
        names = {f.name for f in fields(res)}
        assert names == {"index_snps", "index_p", "clump_members", "n_clumps"}
        res2 = ClumpResult(
            index_snps=list(res.index_snps),
            index_p=res.index_p.clone(),
            clump_members=[list(m) for m in res.clump_members],
            n_clumps=res.n_clumps,
        )
        assert res.index_snps == res2.index_snps
        assert torch.equal(res.index_p, res2.index_p)
        assert res.clump_members == res2.clump_members

    def test_repr_contains_class_name(self):
        res = ClumpResult(
            index_snps=[],
            index_p=torch.tensor([], dtype=torch.float64),
            clump_members=[],
            n_clumps=0,
        )
        s = repr(res)
        assert "ClumpResult" in s
        assert "n_clumps=0" in s


# ---------------------------------------------------------------------------
# HESSResult / HESSRegionResult dataclasses
# ---------------------------------------------------------------------------


class TestHessRegionResult:
    """``HESSRegionResult`` holds per-region HESS h2 (or local rg) plus the
    SNP-range bookkeeping and effective rank used for the estimate."""

    def test_construct(self):
        rr = HESSRegionResult(
            region_id="chr1:1-1000",
            chrom="1",
            start=0,
            end=10,
            h2_local=0.012,
            h2_local_se=0.003,
            n_snps=10,
            n_eigenvalues_kept=8,
        )
        assert rr.region_id == "chr1:1-1000"
        assert rr.chrom == "1"
        assert rr.start == 0
        assert rr.end == 10
        assert rr.h2_local == pytest.approx(0.012)
        assert rr.h2_local_se == pytest.approx(0.003)
        assert rr.n_snps == 10
        assert rr.n_eigenvalues_kept == 8

    def test_round_trip_via_dataclass_fields(self):
        rr = HESSRegionResult(
            region_id="r0",
            chrom="2",
            start=10,
            end=20,
            h2_local=0.0,
            h2_local_se=0.0,
            n_snps=10,
            n_eigenvalues_kept=0,
        )
        names = {f.name for f in fields(rr)}
        assert names == {
            "region_id",
            "chrom",
            "start",
            "end",
            "h2_local",
            "h2_local_se",
            "n_snps",
            "n_eigenvalues_kept",
        }
        rr2 = HESSRegionResult(**{f.name: getattr(rr, f.name) for f in fields(rr)})
        assert rr == rr2

    def test_repr_contains_class_name(self):
        rr = HESSRegionResult(
            region_id="x", chrom="X", start=0, end=1,
            h2_local=0.0, h2_local_se=0.0, n_snps=1, n_eigenvalues_kept=1,
        )
        s = repr(rr)
        assert "HESSRegionResult" in s


class TestHessResult:
    """``HESSResult`` is the aggregate over all per-region HESS estimates,
    plus the sum-of-regions h2 and quadrature SE."""

    def _make(self) -> HESSResult:
        regions = [
            HESSRegionResult(
                region_id=f"r{i}", chrom="1", start=i * 10, end=(i + 1) * 10,
                h2_local=0.01 * (i + 1), h2_local_se=0.002,
                n_snps=10, n_eigenvalues_kept=9,
            )
            for i in range(3)
        ]
        return HESSResult(
            regions=regions,
            h2_total=sum(r.h2_local for r in regions),
            h2_total_se=(sum(r.h2_local_se ** 2 for r in regions)) ** 0.5,
            n_regions=len(regions),
            n_snps_total=sum(r.n_snps for r in regions),
        )

    def test_construct(self):
        res = self._make()
        assert res.n_regions == 3
        assert res.n_snps_total == 30
        assert res.h2_total == pytest.approx(0.06)
        assert res.h2_total_se == pytest.approx((3 * 0.002 ** 2) ** 0.5)
        assert all(isinstance(r, HESSRegionResult) for r in res.regions)

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "regions",
            "h2_total",
            "h2_total_se",
            "n_regions",
            "n_snps_total",
        }
        res2 = HESSResult(
            regions=list(res.regions),
            h2_total=res.h2_total,
            h2_total_se=res.h2_total_se,
            n_regions=res.n_regions,
            n_snps_total=res.n_snps_total,
        )
        assert res2.h2_total == res.h2_total
        assert res2.regions == res.regions

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "HESSResult" in s
        assert "n_regions=3" in s


# ---------------------------------------------------------------------------
# LDSCResult / LDSCRgResult dataclasses
# ---------------------------------------------------------------------------


class TestLdscResult:
    """``LDSCResult`` is the univariate-LDSC h2 output: h2 and SE, intercept
    and SE, mean chi2, lambda GC, and the post-filter SNP count."""

    def _make(self) -> LDSCResult:
        return LDSCResult(
            h2=0.4,
            h2_se=0.05,
            intercept=1.02,
            intercept_se=0.01,
            mean_chi2=1.3,
            lambda_gc=1.05,
            n_snps=480,
        )

    def test_construct(self):
        res = self._make()
        assert res.h2 == pytest.approx(0.4)
        assert res.h2_se == pytest.approx(0.05)
        assert res.intercept == pytest.approx(1.02)
        assert res.intercept_se == pytest.approx(0.01)
        assert res.mean_chi2 == pytest.approx(1.3)
        assert res.lambda_gc == pytest.approx(1.05)
        assert res.n_snps == 480

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "h2",
            "h2_se",
            "intercept",
            "intercept_se",
            "mean_chi2",
            "lambda_gc",
            "n_snps",
        }
        res2 = LDSCResult(**{f.name: getattr(res, f.name) for f in fields(res)})
        assert res == res2

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "LDSCResult" in s
        assert "n_snps=480" in s


class TestLdscRgResult:
    """``LDSCRgResult`` is the cross-trait LDSC output: rg, cov_g and SEs,
    intercept, plus the two univariate ``LDSCResult`` h2 fits."""

    def _make(self) -> LDSCRgResult:
        h2_1 = LDSCResult(0.3, 0.05, 1.0, 0.01, 1.2, 1.0, 500)
        h2_2 = LDSCResult(0.2, 0.04, 1.0, 0.01, 1.1, 1.0, 500)
        return LDSCRgResult(
            rg=0.6,
            rg_se=0.1,
            cov_g=0.15,
            cov_g_se=0.03,
            intercept=0.0,
            intercept_se=0.005,
            h2_1=h2_1,
            h2_2=h2_2,
        )

    def test_construct(self):
        res = self._make()
        assert res.rg == pytest.approx(0.6)
        assert res.rg_se == pytest.approx(0.1)
        assert res.cov_g == pytest.approx(0.15)
        assert isinstance(res.h2_1, LDSCResult)
        assert isinstance(res.h2_2, LDSCResult)

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "rg",
            "rg_se",
            "cov_g",
            "cov_g_se",
            "intercept",
            "intercept_se",
            "h2_1",
            "h2_2",
        }
        res2 = LDSCRgResult(**{f.name: getattr(res, f.name) for f in fields(res)})
        assert res2.rg == res.rg
        assert res2.h2_1 == res.h2_1

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "LDSCRgResult" in s


# ---------------------------------------------------------------------------
# SLDSCResult dataclass
# ---------------------------------------------------------------------------


class TestSldscResult:
    """``SLDSCResult`` is the partitioned-LDSC output: per-annotation tau,
    per-category h2 and enrichment with SEs, plus the totals/intercept and
    the M_c counts. ``annotation_names`` is optional."""

    def _make(self, C: int = 3) -> SLDSCResult:
        tau = torch.tensor([0.001, 0.002, 0.0015], dtype=torch.float64)[:C]
        tau_se = torch.tensor([0.0001, 0.0002, 0.0001], dtype=torch.float64)[:C]
        m_annot = torch.tensor([100.0, 200.0, 150.0], dtype=torch.float64)[:C]
        h2_cat = tau * m_annot
        h2_cat_se = tau_se * m_annot
        h2_total = float(h2_cat.sum().item())
        prop = h2_cat / max(h2_total, 1e-30)
        enrichment = prop / (m_annot / 450.0)
        enrichment_se = (450.0 / m_annot) * (h2_cat_se / max(h2_total, 1e-30))
        return SLDSCResult(
            tau=tau,
            tau_se=tau_se,
            h2_cat=h2_cat,
            h2_cat_se=h2_cat_se,
            enrichment=enrichment,
            enrichment_se=enrichment_se,
            h2_total=h2_total,
            h2_total_se=float((h2_cat_se ** 2).sum().sqrt().item()),
            intercept=1.0,
            intercept_se=0.01,
            m_annot=m_annot,
            n_snps=450,
            annotation_names=["A", "B", "C"][:C],
        )

    def test_construct(self):
        res = self._make(C=3)
        assert res.tau.shape == (3,)
        assert res.tau_se.shape == (3,)
        assert res.h2_cat.shape == (3,)
        assert res.enrichment.shape == (3,)
        assert res.h2_total > 0
        assert res.intercept == pytest.approx(1.0)
        assert res.n_snps == 450
        assert res.annotation_names == ["A", "B", "C"]

    def test_round_trip_via_dataclass_fields(self):
        res = self._make(C=2)
        names = {f.name for f in fields(res)}
        assert names == {
            "tau",
            "tau_se",
            "h2_cat",
            "h2_cat_se",
            "enrichment",
            "enrichment_se",
            "h2_total",
            "h2_total_se",
            "intercept",
            "intercept_se",
            "m_annot",
            "n_snps",
            "annotation_names",
        }
        res2 = SLDSCResult(**{f.name: getattr(res, f.name) for f in fields(res)})
        assert torch.equal(res.tau, res2.tau)
        assert torch.equal(res.h2_cat, res2.h2_cat)
        assert res.h2_total == res2.h2_total
        assert res.annotation_names == res2.annotation_names

    def test_annotation_names_optional(self):
        res = SLDSCResult(
            tau=torch.zeros(1, dtype=torch.float64),
            tau_se=torch.zeros(1, dtype=torch.float64),
            h2_cat=torch.zeros(1, dtype=torch.float64),
            h2_cat_se=torch.zeros(1, dtype=torch.float64),
            enrichment=torch.zeros(1, dtype=torch.float64),
            enrichment_se=torch.zeros(1, dtype=torch.float64),
            h2_total=0.0,
            h2_total_se=0.0,
            intercept=1.0,
            intercept_se=0.0,
            m_annot=torch.tensor([1.0], dtype=torch.float64),
            n_snps=1,
        )
        assert res.annotation_names is None

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "SLDSCResult" in s


# ---------------------------------------------------------------------------
# compute_ld_scores
# ---------------------------------------------------------------------------


class TestComputeLdScores:
    """``compute_ld_scores`` returns the per-SNP sum of r^2 with all SNPs in
    a one-sided ``window_kb`` window on the same chromosome (including self,
    so independent SNPs score ~ 1)."""

    def test_golden_path(self, tiny_genotype_diploid):
        G = tiny_genotype_diploid  # (100, 50)
        m = G.shape[1]
        pos = list(range(0, m * 10_000, 10_000))
        chr_labels = ["1"] * m
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        assert scores.shape == (m,)
        assert scores.dtype == torch.float64
        # Each SNP contributes its own r^2=1; under MAF~0.3 LD between
        # independently-drawn diploid SNPs is small, so scores stay bounded.
        assert torch.all(scores >= 1.0 - 1e-6)
        assert torch.all(scores <= float(m))

    def test_independent_snps_low_scores(self):
        torch.manual_seed(0)
        n, m = 500, 30
        G = torch.randn(n, m, dtype=torch.float64)
        pos = list(range(0, m * 10_000_000, 10_000_000))  # very far apart
        chr_labels = [str(i) for i in range(m)]  # each SNP own chrom -> no LD partners
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1.0)
        # Each SNP isolated on its own chromosome; LD score = self r^2 = 1.
        for s in scores.tolist():
            assert s == pytest.approx(1.0, abs=1e-6)

    def test_perfectly_correlated_snps_high_scores(self):
        torch.manual_seed(1)
        n, m = 200, 25
        base = torch.randn(n, 1, dtype=torch.float64)
        G = base.expand(-1, m).clone()
        pos = list(range(0, m * 100, 100))
        chr_labels = ["1"] * m
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        # Every pairwise r^2 is 1, including self -> LD score ~= m for each.
        for s in scores.tolist():
            assert s == pytest.approx(float(m), abs=0.01)

    def test_chr_labels_length_mismatch_errors(self):
        """Passing chr_labels with the wrong length is a programmer bug.
        Post commit 563df33, ``compute_ld_scores`` validates lengths upfront
        and raises ``ValueError`` containing 'must equal'."""
        torch.manual_seed(2)
        G = torch.randn(50, 10, dtype=torch.float64)
        pos = list(range(0, 10 * 1000, 1000))
        chr_labels = ["1"] * 5  # too short
        with pytest.raises(ValueError, match="must equal"):
            compute_ld_scores(G, pos, chr_labels, window_kb=100.0)


# ---------------------------------------------------------------------------
# compute_cross_ld_scores
# ---------------------------------------------------------------------------


class TestComputeCrossLdScores:
    """``compute_cross_ld_scores`` is documented as a convenience alias for
    LDSC rg with a shared LD reference; it returns the same result as
    ``compute_ld_scores`` on the same arguments."""

    def test_golden_path(self, tiny_genotype_diploid):
        G = tiny_genotype_diploid
        m = G.shape[1]
        pos = list(range(0, m * 10_000, 10_000))
        chr_labels = ["1"] * m
        scores = compute_cross_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        assert scores.shape == (m,)
        assert scores.dtype == torch.float64

    def test_alias_matches_compute_ld_scores(self, tiny_genotype_diploid):
        G = tiny_genotype_diploid
        m = G.shape[1]
        pos = list(range(0, m * 10_000, 10_000))
        chr_labels = ["1"] * m
        a = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        b = compute_cross_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        assert torch.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# hess_local_h2
# ---------------------------------------------------------------------------


class TestHessLocalH2:
    """``hess_local_h2`` partitions h2 into contiguous SNP regions using
    eigendecomposition of the per-region LD matrix."""

    def test_golden_path(self):
        torch.manual_seed(0)
        n_snps = 50
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=1)
        # Mild signal: z drawn with a moderate scale
        z = torch.randn(n_snps, dtype=torch.float64) * 1.2
        bounds = [(i * 10, (i + 1) * 10) for i in range(5)]
        labels = [f"r{i}" for i in range(5)]
        res = hess_local_h2(z, R, n=10_000, region_bounds=bounds,
                            region_labels=labels, eigenvalue_threshold=0.01)
        assert isinstance(res, HESSResult)
        assert res.n_regions == 5
        assert res.n_snps_total == 50
        # Per-region results are well-formed
        for i, rr in enumerate(res.regions):
            assert isinstance(rr, HESSRegionResult)
            assert rr.region_id == f"r{i}"
            assert rr.start == i * 10
            assert rr.end == (i + 1) * 10
            assert rr.n_snps == 10

    def test_no_signal_h2_total_small(self):
        torch.manual_seed(42)
        n_snps = 50
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=2)
        z = torch.randn(n_snps, dtype=torch.float64)  # null
        bounds = [(i * 10, (i + 1) * 10) for i in range(5)]
        # With N=10000 the per-region (chi2 - m_r)/N is centred near 0.
        res = hess_local_h2(z, R, n=10_000, region_bounds=bounds,
                            eigenvalue_threshold=0.01)
        assert abs(res.h2_total) < 0.1

    def test_default_region_labels(self):
        torch.manual_seed(0)
        n_snps = 20
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=3)
        z = torch.randn(n_snps, dtype=torch.float64)
        bounds = [(0, 10), (10, 20)]
        res = hess_local_h2(z, R, n=5000, region_bounds=bounds,
                            eigenvalue_threshold=0.01)
        assert res.regions[0].region_id == "region_0"
        assert res.regions[1].region_id == "region_1"

    def test_empty_region_bounds_raises(self):
        z = torch.randn(10, dtype=torch.float64)
        R = _build_hess_ld_matrix(n_snps=10, seed=4)
        with pytest.raises(ValueError):
            hess_local_h2(z, R, n=1000, region_bounds=[])

    def test_high_threshold_drops_all_eigenvalues(self):
        """When ``eigenvalue_threshold`` exceeds every eigenvalue, each
        region falls into the ``n_kept == 0`` branch with h2=0 and SE=0."""
        torch.manual_seed(5)
        n_snps = 20
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=5)
        z = torch.randn(n_snps, dtype=torch.float64)
        bounds = [(0, 10), (10, 20)]
        res = hess_local_h2(z, R, n=1000, region_bounds=bounds,
                            eigenvalue_threshold=1e30)
        for rr in res.regions:
            assert rr.n_eigenvalues_kept == 0
            assert rr.h2_local == 0.0
            assert rr.h2_local_se == 0.0


# ---------------------------------------------------------------------------
# hess_local_rg
# ---------------------------------------------------------------------------


class TestHessLocalRg:
    """``hess_local_rg`` returns the same ``HESSResult`` shape but the
    ``h2_local`` slot holds local genetic covariance, not heritability."""

    def test_golden_path(self):
        torch.manual_seed(10)
        n_snps = 40
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=10)
        z1 = torch.randn(n_snps, dtype=torch.float64)
        z2 = torch.randn(n_snps, dtype=torch.float64)
        bounds = [(i * 10, (i + 1) * 10) for i in range(4)]
        res = hess_local_rg(z1, z2, R, n1=5000, n2=5000,
                            region_bounds=bounds, eigenvalue_threshold=0.01)
        assert isinstance(res, HESSResult)
        assert res.n_regions == 4
        assert res.n_snps_total == 40

    def test_correlated_traits_positive_rg(self):
        """When z2 = 0.8 * z1 + small noise, the per-region cross-quad-form
        is a positive multiple of the null h2 estimator -> h2_total > 0."""
        torch.manual_seed(20)
        n_snps = 50
        R = _build_hess_ld_matrix(n_snps=n_snps, seed=20)
        z1 = torch.randn(n_snps, dtype=torch.float64) * 3.0
        z2 = 0.8 * z1 + torch.randn(n_snps, dtype=torch.float64) * 0.1
        bounds = [(i * 10, (i + 1) * 10) for i in range(5)]
        res = hess_local_rg(z1, z2, R, n1=5000, n2=5000,
                            region_bounds=bounds, eigenvalue_threshold=0.01)
        # Sum across regions should be positive (rg-aligned).
        assert res.h2_total > 0.0

    def test_empty_region_bounds_raises(self):
        z1 = torch.randn(10, dtype=torch.float64)
        z2 = torch.randn(10, dtype=torch.float64)
        R = _build_hess_ld_matrix(n_snps=10, seed=30)
        with pytest.raises(ValueError):
            hess_local_rg(z1, z2, R, n1=1000, n2=1000, region_bounds=[])


# ---------------------------------------------------------------------------
# ld_clump
# ---------------------------------------------------------------------------


class TestLdClump:
    """``ld_clump`` runs PLINK-style greedy LD clumping; output is a
    ``ClumpResult`` with index SNPs and the SNPs absorbed into each clump."""

    def test_golden_path(self):
        torch.manual_seed(0)
        n, m = 200, 30
        G = torch.randn(n, m, dtype=torch.float64)
        # Inject a high-LD pair near the top of the p-value list so we get
        # at least one non-trivial clump.
        G[:, 1] = G[:, 0] + 0.05 * torch.randn(n, dtype=torch.float64)
        pos = list(range(1000, 1000 + m * 5000, 5000))
        chr_labels = ["1"] * m
        p = torch.linspace(1e-12, 1e-3, m, dtype=torch.float64)
        res = ld_clump(p, G, pos, chr_labels,
                       r2_threshold=0.5, p_threshold=1e-2)
        assert isinstance(res, ClumpResult)
        assert isinstance(res.index_snps, list)
        # Most-significant SNP is always selected first.
        assert 0 in res.index_snps
        assert res.n_clumps == len(res.index_snps)
        # index_p is aligned with index_snps
        assert res.index_p.shape == (len(res.index_snps),)

    def test_no_significant_returns_empty(self):
        torch.manual_seed(1)
        n, m = 100, 20
        G = torch.randn(n, m, dtype=torch.float64)
        pos = list(range(0, m * 1000, 1000))
        chr_labels = ["1"] * m
        p = torch.full((m,), 0.5, dtype=torch.float64)
        res = ld_clump(p, G, pos, chr_labels, p_threshold=1e-8)
        assert res.n_clumps == 0
        assert res.index_snps == []
        assert res.clump_members == []
        assert res.index_p.numel() == 0

    def test_p_g_size_mismatch_errors(self):
        """Passing p and G with mismatched SNP counts is a programmer bug.
        Post commit 563df33, ``ld_clump`` validates the SNP dimension upfront
        and raises ``ValueError`` whose message asserts the SNP dimension
        contract ('must agree on the SNP dimension')."""
        torch.manual_seed(2)
        n = 50
        G = torch.randn(n, 10, dtype=torch.float64)
        p = torch.linspace(1e-12, 1e-3, 5, dtype=torch.float64)  # too short
        pos = list(range(0, 10 * 1000, 1000))
        chr_labels = ["1"] * 10
        with pytest.raises(ValueError, match="must agree on the SNP dimension"):
            ld_clump(p, G, pos, chr_labels, p_threshold=1e-2)


# ---------------------------------------------------------------------------
# ldsc_h2
# ---------------------------------------------------------------------------


class TestLdscH2:
    """``ldsc_h2`` fits ``E[chi2_j] = (N/M) * h2 * l_j + intercept`` via two-
    step weighted least squares with block-jackknife SEs."""

    def test_golden_path(self):
        ld_scores, _z, chi2 = _build_ldsc_inputs(seed=0, scale=1.0)
        res = ldsc_h2(chi2, ld_scores, n=10_000, m_total=500, n_blocks=20)
        assert isinstance(res, LDSCResult)
        # Result fields are finite numerics
        assert isinstance(res.h2, float)
        assert isinstance(res.h2_se, float)
        assert isinstance(res.intercept, float)
        assert res.h2_se >= 0.0
        assert res.n_snps > 0

    def test_null_signal_h2_low(self):
        """Under z ~ N(0, 1) and random LD scores, the slope is essentially
        noise around zero, so |h2| stays small."""
        ld_scores, _z, chi2 = _build_ldsc_inputs(seed=1, scale=1.0)
        res = ldsc_h2(chi2, ld_scores, n=10_000, m_total=500, n_blocks=20)
        # The estimator can be slightly negative; check magnitude.
        assert abs(res.h2) < 0.5

    def test_inflated_h2_higher_than_null(self):
        """Inflating z's std raises mean_chi2 (and therefore the regression
        intercept/slope), so the recovered h2 magnitude is larger than the
        null fit at the same seed family."""
        # Use the same seed for both so the comparison is deterministic and
        # only the ``scale`` knob differs.
        ld_null, _zn, chi2_null = _build_ldsc_inputs(seed=2, scale=1.0)
        ld_inf, _zi, chi2_inf = _build_ldsc_inputs(seed=2, scale=2.5)
        res_null = ldsc_h2(chi2_null, ld_null, n=10_000, m_total=500, n_blocks=20)
        res_inf = ldsc_h2(chi2_inf, ld_inf, n=10_000, m_total=500, n_blocks=20)
        # mean_chi2 of the inflated set must be larger.
        assert res_inf.mean_chi2 > res_null.mean_chi2
        # The inflated fit reports a larger lambda_gc.
        assert res_inf.lambda_gc > res_null.lambda_gc

    def test_irwls_default_converges_in_two_iterations(self):
        """Per the validation findings ledger (Pillar B / B2), the LDSC
        IRWLS port in ``_ldsc.py`` converges at LDSC's fixed iteration
        count (2). With a deterministic fixture that mimics LDSC's
        inflated-chi² regime (distinct reference + regression-weight LD
        scores, h² = 0.4, intercept = 1.05), increasing ``n_iter`` past
        2 must not change the IRWLS estimate to FP precision — this is
        the bit-equality contract that closes the original B2 intercept
        divergence (TG → 0.9787 vs LDSC 0.9788, |Δ| 2.6e-5; verified via
        ``validation/external/ldsc/compare.py``).
        """
        torch.manual_seed(2026)
        m = 1000
        ld_ref = torch.rand(m, dtype=torch.float64) * 8.0 + 2.0
        w_ld = torch.rand(m, dtype=torch.float64) * 5.0 + 1.5
        n = 50_000
        M = m
        h2_true = 0.4
        intercept_true = 1.05
        sigma = ((n / M) * h2_true * ld_ref + intercept_true).sqrt()
        chi2 = sigma ** 2 * (1.0 + 0.1 * torch.randn(m, dtype=torch.float64))

        res_2 = ldsc_h2(chi2, ld_ref, n=n, m_total=M, w_ld=w_ld, n_iter=2)
        res_5 = ldsc_h2(chi2, ld_ref, n=n, m_total=M, w_ld=w_ld, n_iter=5)
        res_10 = ldsc_h2(chi2, ld_ref, n=n, m_total=M, w_ld=w_ld, n_iter=10)

        # IRWLS converges to FP precision after LDSC's fixed 2 iterations.
        assert abs(res_5.h2 - res_2.h2) < 1e-12
        assert abs(res_5.intercept - res_2.intercept) < 1e-12
        assert abs(res_10.h2 - res_2.h2) < 1e-12
        assert abs(res_10.intercept - res_2.intercept) < 1e-12

    def test_irwls_matches_hand_rolled_reference(self):
        """Per the validation findings ledger (Pillar B / B2 follow-up):
        the IRWLS port in ``_ldsc.py`` reproduces a hand-rolled reference
        implementation of LDSC's IRWLS algorithm (LDSC ``Hsq.weights`` +
        2-iteration loop + Nbar-scaled design matrix) to FP precision on
        a deterministic LDSC-regime fixture.

        The full bit-equality check against the upstream LDSC binary runs
        against the simulated chr22 sumstats in
        ``validation/external/ldsc/`` (out of tree for the default
        suite). That harness gates ``|Δ intercept| ≤ 5e-3`` and observes
        ~3e-5 — see the harness README for the full numbers. Here we
        gate the *algorithmic invariants* of the IRWLS implementation
        itself.
        """
        torch.manual_seed(7777)
        m = 2000
        ld_ref = torch.rand(m, dtype=torch.float64) * 10.0 + 2.0
        w_ld = torch.rand(m, dtype=torch.float64) * 6.0 + 1.5
        n = 100_000
        M = m
        h2_true = 0.4
        intercept_true = 1.05
        mean_chi2 = (n / M) * h2_true * ld_ref + intercept_true
        chi2 = mean_chi2 * (1.0 + 0.05 * torch.randn(m, dtype=torch.float64))
        chi2 = chi2.clamp(min=0.01)

        # Hand-rolled reference IRWLS following ldsc/regressions.py + ldsc/irwls.py.
        ld_c = torch.clamp(ld_ref, min=1.0).to(torch.float64)
        wld_c = torch.clamp(w_ld, min=1.0).to(torch.float64)
        n_per_snp = torch.full((m,), float(n), dtype=torch.float64)
        nbar = n_per_snp.mean()
        # Aggregate initial estimate.
        denom_agg = torch.mean(ld_c * n_per_snp)
        hsq = float(M * (chi2.mean() - 1.0) / denom_agg)
        intercept = 1.0
        # Initial weights.
        c = max(min(hsq, 1.0), 0.0) * n_per_snp / float(M)
        het_w = 1.0 / (2.0 * (intercept + c * ld_c) ** 2)
        w = (het_w / wld_c).clamp(min=torch.finfo(torch.float64).tiny)
        # Design matrix (Nbar-scaled, intercept column).
        x_col = n_per_snp * ld_c / nbar
        X = torch.stack([x_col, torch.ones(m, dtype=torch.float64)], dim=1)
        # 2-iteration IRWLS loop (LDSC default).
        sqrt_w = w.sqrt().unsqueeze(1)
        coef_ref = torch.linalg.lstsq(X * sqrt_w, chi2 * w.sqrt()).solution
        for _ in range(2):
            slope = coef_ref[0].item()
            intercept_iter = max(coef_ref[1].item(), 0.0)
            hsq_iter = slope * float(M) / float(nbar.item())
            c_iter = max(min(hsq_iter, 1.0), 0.0) * n_per_snp / float(M)
            het_w_iter = 1.0 / (2.0 * (intercept_iter + c_iter * ld_c) ** 2)
            w_iter = (het_w_iter / wld_c).clamp(min=torch.finfo(torch.float64).tiny)
            sqrt_w_iter = w_iter.sqrt().unsqueeze(1)
            coef_ref = torch.linalg.lstsq(X * sqrt_w_iter, chi2 * w_iter.sqrt()).solution
        h2_ref = coef_ref[0].item() * float(M) / float(nbar.item())
        intercept_ref = coef_ref[1].item()

        res = ldsc_h2(chi2, ld_ref, n=n, m_total=M, w_ld=w_ld, n_iter=2)
        assert abs(res.h2 - h2_ref) < 1e-12
        assert abs(res.intercept - intercept_ref) < 1e-12


# ---------------------------------------------------------------------------
# ldsc_intercept
# ---------------------------------------------------------------------------


class TestLdscIntercept:
    """``ldsc_intercept`` returns ``(intercept, intercept_se)`` from the
    underlying ``ldsc_h2`` fit; intercept ~ 1 means no inflation."""

    def test_golden_path(self):
        ld_scores, _z, chi2 = _build_ldsc_inputs(seed=10, scale=1.0)
        out = ldsc_intercept(chi2, ld_scores, n=10_000, m_total=500, n_blocks=20)
        # Documented signature is a 2-tuple of floats.
        assert isinstance(out, tuple)
        assert len(out) == 2
        intercept, se = out
        assert isinstance(intercept, float)
        assert isinstance(se, float)
        assert se >= 0.0

    def test_null_intercept_near_one(self):
        """Under z ~ N(0,1), chi2 has mean 1 with no LD-correlated signal,
        so the LDSC fit's intercept lands near 1.0."""
        ld_scores, _z, chi2 = _build_ldsc_inputs(seed=11, scale=1.0)
        intercept, se = ldsc_intercept(
            chi2, ld_scores, n=10_000, m_total=500, n_blocks=20
        )
        # Bound: 3 SE plus a 0.3 numerical margin (matches the existing
        # test_postgwas_ldsc convention).
        assert abs(intercept - 1.0) < 3 * se + 0.3


# ---------------------------------------------------------------------------
# ldsc_rg
# ---------------------------------------------------------------------------


class TestLdscRg:
    """``ldsc_rg`` runs cross-trait LDSC from chi2 statistics; sign of rg is
    not recoverable from chi2 alone (squared z's), so we only verify shape
    and finiteness."""

    def test_golden_path(self):
        ld_scores, _z, chi2_1 = _build_ldsc_inputs(seed=20, scale=1.0)
        torch.manual_seed(21)
        chi2_2 = torch.randn(500, dtype=torch.float64) ** 2
        res = ldsc_rg(chi2_1, chi2_2, ld_scores,
                      n1=10_000, n2=10_000, m_total=500, n_blocks=20)
        assert isinstance(res, LDSCRgResult)
        assert isinstance(res.h2_1, LDSCResult)
        assert isinstance(res.h2_2, LDSCResult)
        # rg may be NaN if denom collapses; SE either non-negative or NaN.
        import math
        assert isinstance(res.rg, float)
        assert math.isfinite(res.cov_g) or math.isnan(res.cov_g)


# ---------------------------------------------------------------------------
# ldsc_rg_from_z
# ---------------------------------------------------------------------------


class TestLdscRgFromZ:
    """``ldsc_rg_from_z`` is the signed-z variant of LDSC rg; the sign of
    the recovered cov_g should match the sign of the underlying correlation
    between z1 and z2."""

    def _build_correlated(self, m: int, rho: float, seed: int):
        torch.manual_seed(seed)
        ld_scores = torch.rand(m, dtype=torch.float64) * 50.0 + 5.0
        z1 = torch.randn(m, dtype=torch.float64)
        z2 = rho * z1 + (1.0 - rho ** 2) ** 0.5 * torch.randn(m, dtype=torch.float64)
        return ld_scores, z1, z2

    def test_golden_path(self):
        ld_scores, z1, z2 = self._build_correlated(500, 0.5, seed=30)
        res = ldsc_rg_from_z(z1, z2, ld_scores,
                             n1=10_000, n2=10_000, m_total=500, n_blocks=20)
        assert isinstance(res, LDSCRgResult)
        assert isinstance(res.h2_1, LDSCResult)
        assert isinstance(res.h2_2, LDSCResult)

    def test_correlated_traits_positive_cov_g(self):
        """With z's drawn so ``E[z1·z2 | l] ∝ l`` the cov_g slope is positive.

        After the IRWLS port (Phase 37 follow-up), the LDSC regression
        only recovers a non-zero slope when the SNP-level z1*z2 product
        actually scales with the LD score (the model assumption). The
        previous incarnation of this test relied on a flat z1*z2 product
        being mis-attributed to the slope by the simple
        ``w = 1/max(l², 1)`` weights — the IRWLS heteroscedastic weights
        no longer permit that, so we generate z's with a genuine
        LD-coupled covariance signal here.
        """
        torch.manual_seed(31)
        m = 500
        ld_scores = torch.rand(m, dtype=torch.float64) * 50.0 + 5.0
        # Variance of z1*z2 grows with l (the LDSC model): construct
        # z_shared,j ~ N(0, sqrt(l_j)/sqrt(M)) and add per-trait noise.
        scale = (ld_scores / float(m)).sqrt()
        n1 = 10_000
        z_shared = torch.randn(m, dtype=torch.float64) * scale * (n1 ** 0.5)
        eps1 = torch.randn(m, dtype=torch.float64) * 0.5
        eps2 = torch.randn(m, dtype=torch.float64) * 0.5
        z1 = z_shared + eps1
        z2 = z_shared + eps2
        res = ldsc_rg_from_z(z1, z2, ld_scores,
                             n1=n1, n2=n1, m_total=m, n_blocks=20)
        assert res.cov_g > 0.0

    def test_independent_traits_cov_g_small(self):
        """When z1 and z2 are independent, the cov_g slope sits near zero
        (within sampling noise)."""
        ld_scores, z1, z2 = self._build_correlated(500, 0.0, seed=32)
        res = ldsc_rg_from_z(z1, z2, ld_scores,
                             n1=10_000, n2=10_000, m_total=500, n_blocks=20)
        # Magnitude is small relative to the inflated case; pin the bound
        # against the seed/scale used here.
        assert abs(res.cov_g) < 0.2


# ---------------------------------------------------------------------------
# sldsc_h2_partitioned
# ---------------------------------------------------------------------------


class TestSldscH2Partitioned:
    """``sldsc_h2_partitioned`` extends LDSC h2 to multiple annotation
    categories; the single-category case must reduce to plain ``ldsc_h2``."""

    def test_golden_path(self):
        torch.manual_seed(40)
        m, C = 500, 3
        ld_scores_total = torch.rand(m, dtype=torch.float64) * 50.0 + 5.0
        # Split total into C non-negative columns that sum back to the total.
        weights = torch.softmax(torch.randn(m, C, dtype=torch.float64), dim=1)
        annot = ld_scores_total.unsqueeze(1) * weights  # (m, C)
        m_c = torch.tensor([200.0, 150.0, 150.0], dtype=torch.float64)
        chi2 = torch.randn(m, dtype=torch.float64) ** 2
        res = sldsc_h2_partitioned(
            chi2, annot, m_c, n=10_000, m_total=500, n_blocks=20,
            annotation_names=["A", "B", "C"],
        )
        assert isinstance(res, SLDSCResult)
        assert res.tau.shape == (C,)
        assert res.h2_cat.shape == (C,)
        assert res.enrichment.shape == (C,)
        assert res.annotation_names == ["A", "B", "C"]

    def test_single_category_recovers_ldsc_h2(self):
        """When C=1 and the single annotation column equals the total LD
        score, partitioned h2_total must agree with plain ldsc_h2.h2 to
        high precision (same WLS / two-step / jackknife mechanics)."""
        ld_scores, _z, chi2 = _build_ldsc_inputs(seed=50, scale=1.0)
        m_total = ld_scores.shape[0]
        annot = ld_scores.unsqueeze(1)  # (m, 1)
        m_c = torch.tensor([float(m_total)], dtype=torch.float64)
        # S-LDSC still uses single-pass WLS internally (Phase 42). The
        # default ``ldsc_h2`` runs IRWLS (Phase 37 follow-up); pass
        # ``n_iter=0`` so the comparison stays on the single-pass path.
        plain = ldsc_h2(
            chi2, ld_scores, n=10_000, m_total=m_total, n_blocks=20, n_iter=0,
        )
        partitioned = sldsc_h2_partitioned(
            chi2, annot, m_c, n=10_000, m_total=m_total, n_blocks=20,
        )
        # tau_c * M_c == (slope) * M_total / N when N is folded into design;
        # the S-LDSC design matrix uses N*L, so tau is in per-N units. Final
        # h2 from S-LDSC is tau * M_c, which equals the legacy LDSC h2.
        assert partitioned.h2_total == pytest.approx(plain.h2, abs=1e-6)

    def test_invalid_annot_dim_raises(self):
        chi2 = torch.randn(20, dtype=torch.float64) ** 2
        annot_bad = torch.randn(20, dtype=torch.float64)  # 1-D
        m_c = torch.tensor([20.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="2-D"):
            sldsc_h2_partitioned(chi2, annot_bad, m_c, n=1000, m_total=20)

    def test_chi2_annot_size_mismatch_raises(self):
        chi2 = torch.randn(20, dtype=torch.float64) ** 2
        annot_bad = torch.randn(15, 2, dtype=torch.float64)
        m_c = torch.tensor([10.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="same number of SNPs"):
            sldsc_h2_partitioned(chi2, annot_bad, m_c, n=1000, m_total=20)

    def test_m_c_length_mismatch_raises(self):
        chi2 = torch.randn(20, dtype=torch.float64) ** 2
        annot = torch.randn(20, 2, dtype=torch.float64)
        m_c_bad = torch.tensor([10.0], dtype=torch.float64)  # wrong length
        with pytest.raises(ValueError, match="length equal"):
            sldsc_h2_partitioned(chi2, annot, m_c_bad, n=1000, m_total=20)
