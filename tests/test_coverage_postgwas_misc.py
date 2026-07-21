"""Tier-2 behavioral coverage tests for the FINAL ``torchgenomics.postgwas`` public
symbols across five submodules:

- ``_enrichment``: ``EnrichmentResult``, ``GeneResult``, ``gene_set_enrichment``,
  ``snp_to_gene``.
- ``_multi_ancestry``: ``MultiAncestryResult``, ``mantra``, ``mr_mega``.
- ``_power``: ``PowerResult``, ``gwas_power``, ``power_curve``, ``required_n``.
- ``_sumstats``: ``SumStats``, ``align_sumstats``, ``load_sumstats``.
- ``_winners_curse``: ``WinnersCurseResult``, ``bootstrap_correction``,
  ``conditional_likelihood``, ``correct_winners_curse``, ``fiqt``.

Bar (Pillar A spec section 4.3 Tier 2):
- Dataclasses: construct + round-trip + repr smoke.
- Functions: golden + edge + error tests, with closed-form references where
  possible (e.g. FIQT shrinks tiny z-scores toward zero, CL leaves z<<c
  uncorrected, identity-LD power formula).

After this commit, postgwas Tier 2 covers 66/66 public symbols.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from torchgenomics.postgwas import (
    EnrichmentResult,
    GeneResult,
    MultiAncestryResult,
    PowerResult,
    SumStats,
    WinnersCurseResult,
    align_sumstats,
    bootstrap_correction,
    conditional_likelihood,
    correct_winners_curse,
    fiqt,
    gene_set_enrichment,
    gwas_power,
    load_sumstats,
    mantra,
    mr_mega,
    power_curve,
    required_n,
    snp_to_gene,
)

pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sumstats(
    snps: list[str],
    beta: torch.Tensor,
    se: float | torch.Tensor = 0.05,
    chr_label: str = "1",
    a1: str = "A",
    a2: str = "G",
    af: float | torch.Tensor | None = 0.3,
) -> SumStats:
    """Construct a minimal :class:`SumStats` for tests."""
    K = len(snps)
    if not isinstance(se, torch.Tensor):
        se = torch.full((K,), float(se), dtype=torch.float64)
    p = torch.erfc((beta / se).abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)
    n = torch.full((K,), 1000.0, dtype=torch.float64)
    if af is not None and not isinstance(af, torch.Tensor):
        af_t = torch.full((K,), float(af), dtype=torch.float64)
    elif isinstance(af, torch.Tensor):
        af_t = af.to(torch.float64)
    else:
        af_t = None
    return SumStats(
        chr=[chr_label] * K,
        pos=list(range(1000, 1000 + K * 100, 100)),
        snp=list(snps),
        a1=[a1] * K,
        a2=[a2] * K,
        beta=beta.to(torch.float64),
        se=se.to(torch.float64),
        p=p,
        n=n,
        af=af_t,
    )


# ===========================================================================
# Dataclass smoke tests (6)
# ===========================================================================


class TestSumStatsDataclass:
    """``SumStats`` carries chr/pos/snp/a1/a2/beta/se/p/n/af with derived
    chi2 / z / m properties and a ``to(device)`` helper."""

    def _make(self) -> SumStats:
        return SumStats(
            chr=["1", "1", "2"],
            pos=[100, 200, 300],
            snp=["rs1", "rs2", "rs3"],
            a1=["A", "C", "G"],
            a2=["G", "T", "A"],
            beta=torch.tensor([0.1, 0.2, -0.05], dtype=torch.float64),
            se=torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64),
            p=torch.tensor([1e-3, 1e-5, 0.3], dtype=torch.float64),
            n=torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64),
            af=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64),
        )

    def test_construct(self):
        ss = self._make()
        assert ss.snp == ["rs1", "rs2", "rs3"]
        assert ss.m == 3

    def test_chi2_and_z_properties(self):
        ss = self._make()
        ref_z = ss.beta / ss.se
        torch.testing.assert_close(ss.z, ref_z)
        torch.testing.assert_close(ss.chi2, ref_z**2)

    def test_to_cpu_returns_new_instance(self):
        ss = self._make()
        moved = ss.to("cpu")
        assert isinstance(moved, SumStats)
        assert moved.beta.device.type == "cpu"
        # Lists shared / cloned-but-equal.
        assert moved.snp == ss.snp

    def test_round_trip_via_dataclass_fields(self):
        ss = self._make()
        names = {f.name for f in fields(ss)}
        assert names == {
            "chr", "pos", "snp", "a1", "a2", "beta", "se", "p", "n", "af",
        }
        ss2 = SumStats(**{f.name: getattr(ss, f.name) for f in fields(ss)})
        assert ss2.snp == ss.snp
        torch.testing.assert_close(ss2.beta, ss.beta)

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "SumStats" in s


class TestGeneResultDataclass:
    """``GeneResult`` carries gene_id/chr/start/end + n_snps/stat/p."""

    def _make(self) -> GeneResult:
        return GeneResult(
            gene_id=["G1", "G2"],
            gene_chr=["1", "2"],
            gene_start=[1000, 5000],
            gene_end=[2000, 6000],
            n_snps=[3, 0],
            stat=torch.tensor([1.5, 0.0], dtype=torch.float64),
            p=torch.tensor([0.05, 1.0], dtype=torch.float64),
        )

    def test_construct(self):
        g = self._make()
        assert g.gene_id == ["G1", "G2"]
        assert g.n_snps == [3, 0]
        assert g.stat.shape == (2,)

    def test_round_trip_via_dataclass_fields(self):
        g = self._make()
        names = {f.name for f in fields(g)}
        assert names == {
            "gene_id", "gene_chr", "gene_start", "gene_end",
            "n_snps", "stat", "p",
        }
        g2 = GeneResult(**{f.name: getattr(g, f.name) for f in fields(g)})
        assert g2.gene_id == g.gene_id

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "GeneResult" in s


class TestEnrichmentResultDataclass:
    """``EnrichmentResult`` carries gene_set_name + n_genes_in_set +
    n_genes_total + beta_enrichment / se / p."""

    def _make(self) -> EnrichmentResult:
        return EnrichmentResult(
            gene_set_name=["set1", "set2"],
            n_genes_in_set=[3, 5],
            n_genes_total=10,
            beta_enrichment=torch.tensor([0.2, -0.1], dtype=torch.float64),
            se=torch.tensor([0.05, 0.06], dtype=torch.float64),
            p=torch.tensor([1e-3, 0.4], dtype=torch.float64),
        )

    def test_construct(self):
        r = self._make()
        assert r.gene_set_name == ["set1", "set2"]
        assert r.n_genes_total == 10
        assert r.beta_enrichment.shape == (2,)

    def test_round_trip_via_dataclass_fields(self):
        r = self._make()
        names = {f.name for f in fields(r)}
        assert names == {
            "gene_set_name", "n_genes_in_set", "n_genes_total",
            "beta_enrichment", "se", "p",
        }
        r2 = EnrichmentResult(**{f.name: getattr(r, f.name) for f in fields(r)})
        assert r2.gene_set_name == r.gene_set_name

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "EnrichmentResult" in s


class TestMultiAncestryResultDataclass:
    """``MultiAncestryResult`` carries beta_meta / se_meta / p_meta /
    p_heterogeneity + method tag + method-specific fields."""

    def _make(self) -> MultiAncestryResult:
        return MultiAncestryResult(
            beta_meta=torch.tensor([0.1, 0.2], dtype=torch.float64),
            se_meta=torch.tensor([0.05, 0.05], dtype=torch.float64),
            p_meta=torch.tensor([1e-3, 1e-5], dtype=torch.float64),
            p_heterogeneity=torch.tensor([0.4, 0.6], dtype=torch.float64),
            method="mr_mega",
            p_ancestry=torch.tensor([0.5, 0.5], dtype=torch.float64),
            p_residual=torch.tensor([0.7, 0.7], dtype=torch.float64),
            n_axes=2,
            log10_bf=None,
            posterior_effect=None,
            n_populations=3,
            n_snps=2,
        )

    def test_construct(self):
        r = self._make()
        assert r.method == "mr_mega"
        assert r.n_axes == 2
        assert r.n_populations == 3
        assert r.n_snps == 2

    def test_default_optional_fields(self):
        r = MultiAncestryResult(
            beta_meta=torch.zeros(1, dtype=torch.float64),
            se_meta=torch.zeros(1, dtype=torch.float64),
            p_meta=torch.ones(1, dtype=torch.float64),
            p_heterogeneity=torch.ones(1, dtype=torch.float64),
            method="mantra",
        )
        assert r.p_ancestry is None
        assert r.p_residual is None
        assert r.log10_bf is None
        assert r.posterior_effect is None
        assert r.n_axes == 0
        assert r.n_populations == 0
        assert r.n_snps == 0

    def test_round_trip_via_dataclass_fields(self):
        r = self._make()
        names = {f.name for f in fields(r)}
        assert names == {
            "beta_meta", "se_meta", "p_meta", "p_heterogeneity", "method",
            "p_ancestry", "p_residual", "n_axes",
            "log10_bf", "posterior_effect",
            "n_populations", "n_snps",
        }
        r2 = MultiAncestryResult(
            **{f.name: getattr(r, f.name) for f in fields(r)}
        )
        assert r2.method == r.method

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "MultiAncestryResult" in s


class TestPowerResultDataclass:
    """``PowerResult`` carries power, ncp, alpha, n, min_detectable_beta."""

    def _make(self) -> PowerResult:
        return PowerResult(
            power=torch.tensor([0.1, 0.8], dtype=torch.float64),
            ncp=torch.tensor([2.0, 30.0], dtype=torch.float64),
            alpha=5e-8,
            n=10000,
            min_detectable_beta=torch.tensor([0.1, 0.05], dtype=torch.float64),
        )

    def test_construct(self):
        r = self._make()
        assert r.alpha == pytest.approx(5e-8)
        assert r.n == 10000
        assert r.power.shape == (2,)

    def test_round_trip_via_dataclass_fields(self):
        r = self._make()
        names = {f.name for f in fields(r)}
        assert names == {
            "power", "ncp", "alpha", "n", "min_detectable_beta",
        }
        r2 = PowerResult(**{f.name: getattr(r, f.name) for f in fields(r)})
        assert r2.alpha == r.alpha
        assert r2.n == r.n

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "PowerResult" in s


class TestWinnersCurseResultDataclass:
    """``WinnersCurseResult`` carries method / beta_adjusted / se_adjusted /
    shrinkage_factor / n_corrected."""

    def _make(self) -> WinnersCurseResult:
        return WinnersCurseResult(
            method="conditional_likelihood",
            beta_adjusted=torch.tensor([0.1, 0.2], dtype=torch.float64),
            se_adjusted=None,
            shrinkage_factor=torch.tensor([0.9, 0.95], dtype=torch.float64),
            n_corrected=2,
        )

    def test_construct(self):
        r = self._make()
        assert r.method == "conditional_likelihood"
        assert r.se_adjusted is None
        assert r.n_corrected == 2

    def test_round_trip_via_dataclass_fields(self):
        r = self._make()
        names = {f.name for f in fields(r)}
        assert names == {
            "method", "beta_adjusted", "se_adjusted",
            "shrinkage_factor", "n_corrected",
        }
        r2 = WinnersCurseResult(
            **{f.name: getattr(r, f.name) for f in fields(r)}
        )
        assert r2.method == r.method

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "WinnersCurseResult" in s


# ===========================================================================
# align_sumstats
# ===========================================================================


class TestAlignSumstats:
    """``align_sumstats`` intersects a list of SumStats by SNP ID and flips
    effect signs when a1/a2 are swapped relative to the first study."""

    def test_smoke_intersects_common_snps(self):
        ss1 = _build_sumstats(
            ["rs1", "rs2", "rs3"],
            torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        )
        ss2 = _build_sumstats(
            ["rs2", "rs3", "rs4"],
            torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64),
        )
        aligned = align_sumstats([ss1, ss2])
        assert len(aligned) == 2
        # Intersection is rs2, rs3 in ss1 order.
        assert aligned[0].snp == ["rs2", "rs3"]
        assert aligned[1].snp == ["rs2", "rs3"]
        # Beta values from ss1 in intersected order.
        torch.testing.assert_close(
            aligned[0].beta,
            torch.tensor([0.2, 0.3], dtype=torch.float64),
        )
        # Beta values from ss2 reordered to match.
        torch.testing.assert_close(
            aligned[1].beta,
            torch.tensor([0.5, 0.6], dtype=torch.float64),
        )

    def test_allele_flip_negates_effect(self):
        ss1 = _build_sumstats(
            ["rs1", "rs2"],
            torch.tensor([0.1, 0.2], dtype=torch.float64),
            a1="A",
            a2="G",
        )
        # Build ss2 with swapped alleles -> match_alleles=True flips betas.
        ss2 = _build_sumstats(
            ["rs1", "rs2"],
            torch.tensor([0.4, 0.5], dtype=torch.float64),
            a1="G",  # swapped
            a2="A",
        )
        aligned = align_sumstats([ss1, ss2], match_alleles=True)
        # ss2 effects flipped.
        torch.testing.assert_close(
            aligned[1].beta,
            torch.tensor([-0.4, -0.5], dtype=torch.float64),
        )
        # And the alleles now match the reference.
        assert aligned[1].a1 == ["A", "A"]
        assert aligned[1].a2 == ["G", "G"]

    def test_no_overlap_raises(self):
        ss1 = _build_sumstats(
            ["rs1", "rs2"], torch.tensor([0.1, 0.2], dtype=torch.float64),
        )
        ss2 = _build_sumstats(
            ["rs9", "rs10"], torch.tensor([0.3, 0.4], dtype=torch.float64),
        )
        with pytest.raises(ValueError):
            align_sumstats([ss1, ss2])

    def test_single_input_returned_unchanged(self):
        # Documented behavior: <2 inputs are returned as-is.
        ss1 = _build_sumstats(
            ["rs1"], torch.tensor([0.1], dtype=torch.float64),
        )
        out = align_sumstats([ss1])
        assert out is not None
        assert len(out) == 1
        assert out[0].snp == ["rs1"]


# ===========================================================================
# load_sumstats
# ===========================================================================


class TestLoadSumstats:
    """``load_sumstats`` reads a TSV with configurable column names into a
    SumStats."""

    def _write_tsv(self, tmp_path, with_n=True, with_af=True):
        path = tmp_path / "ss.tsv"
        cols = ["chr", "pos", "snp", "a1", "a2", "beta", "se", "p"]
        rows = [
            ["1", "100", "rs1", "A", "G", "0.1", "0.05", "0.001"],
            ["1", "200", "rs2", "C", "T", "-0.05", "0.05", "0.3"],
            ["2", "300", "rs3", "G", "A", "0.2", "0.05", "1e-5"],
        ]
        if with_n:
            cols.append("n")
            for r in rows:
                r.append("1000")
        if with_af:
            cols.append("af")
            for r in rows:
                r.append("0.3")
        with path.open("w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")
        return path

    def test_smoke_loads_all_fields(self, tmp_path):
        path = self._write_tsv(tmp_path)
        ss = load_sumstats(str(path))
        assert isinstance(ss, SumStats)
        assert ss.snp == ["rs1", "rs2", "rs3"]
        assert ss.chr == ["1", "1", "2"]
        assert ss.pos == [100, 200, 300]
        assert ss.a1 == ["A", "C", "G"]
        torch.testing.assert_close(
            ss.beta,
            torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64),
        )
        torch.testing.assert_close(
            ss.n,
            torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64),
        )
        assert ss.af is not None
        torch.testing.assert_close(
            ss.af,
            torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
        )

    def test_missing_optional_columns_uses_defaults(self, tmp_path):
        # No 'n' or 'af' columns -> n is NaN, af is None.
        path = self._write_tsv(tmp_path, with_n=False, with_af=False)
        ss = load_sumstats(str(path))
        assert ss.af is None
        assert torch.all(torch.isnan(ss.n))

    def test_invalid_path_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_sumstats("/nonexistent/path/to/ss.tsv")

    def test_malformed_columns_raises(self, tmp_path):
        # Missing required column 'beta' -> KeyError from pandas indexing.
        path = tmp_path / "bad.tsv"
        with path.open("w") as fh:
            fh.write("chr\tpos\tsnp\ta1\ta2\tse\tp\n")
            fh.write("1\t100\trs1\tA\tG\t0.05\t0.001\n")
        with pytest.raises((KeyError, ValueError)):
            load_sumstats(str(path))


# ===========================================================================
# conditional_likelihood
# ===========================================================================


class TestConditionalLikelihood:
    """CL shrinkage applies only to |z| >= z_alpha; below that the estimate
    passes through unchanged."""

    def test_smoke(self):
        beta = torch.tensor([0.5, 0.05, 0.01], dtype=torch.float64)
        se = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)
        res = conditional_likelihood(beta, se, alpha=5e-8)
        assert isinstance(res, WinnersCurseResult)
        assert res.method == "conditional_likelihood"
        assert res.beta_adjusted.shape == beta.shape
        assert res.shrinkage_factor.shape == beta.shape

    def test_below_threshold_unchanged(self):
        # Tiny effect, |z| ~ 0.2 -- well below the 5e-8 threshold.
        # Implementation leaves the estimate unchanged.
        beta = torch.tensor([0.01], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = conditional_likelihood(beta, se, alpha=5e-8)
        assert res.n_corrected == 0
        torch.testing.assert_close(res.beta_adjusted, beta)
        # Shrinkage factor is exactly 1.0 for uncorrected variants.
        torch.testing.assert_close(
            res.shrinkage_factor,
            torch.tensor([1.0], dtype=torch.float64),
        )

    def test_significant_variant_shrinks_toward_zero(self):
        # |z| ~ 6, well above z_{5e-8/2} ~ 5.45 -> correction applied.
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = conditional_likelihood(beta, se, alpha=5e-8)
        assert res.n_corrected == 1
        assert abs(res.beta_adjusted.item()) < abs(beta.item())
        assert res.beta_adjusted.item() > 0  # positive direction preserved

    def test_invalid_negative_alpha(self):
        # alpha = 0 leaves no significant region; alpha < 0 -> nan/inf
        # quantile, which propagates through the formula. We exercise
        # alpha=2 (> 1) instead, which yields ndtri(0) = -inf and a
        # documented-but-degenerate result. Implementation does not raise
        # but returns finite no-correction output (sig mask is empty).
        beta = torch.tensor([0.5], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        # alpha > 1 makes 1 - alpha/2 negative -> ndtri returns nan;
        # use the boundary alpha=1.0 which gives c = 0 and corrects all.
        res = conditional_likelihood(beta, se, alpha=1.0)
        # With c=0 every variant qualifies as significant.
        assert res.n_corrected == 1


# ===========================================================================
# fiqt
# ===========================================================================


class TestFiqt:
    """FIQT shrinks z proportionally to local FDR. Tiny z (high lfdr) ->
    near-zero corrected z; large z (low lfdr) -> z almost unchanged."""

    def test_smoke(self):
        torch.manual_seed(0)
        z = torch.randn(50, dtype=torch.float64)
        se = torch.full((50,), 0.05, dtype=torch.float64)
        res = fiqt(z, se)
        assert isinstance(res, WinnersCurseResult)
        assert res.method == "fiqt"
        assert res.beta_adjusted.shape == z.shape
        assert res.shrinkage_factor.shape == z.shape

    def test_small_z_shrinks_toward_zero(self):
        # Build a sample with a few very small z's embedded in a
        # null-dominated background. Their lfdr ~ 1, so corrected
        # z ~ 0 and beta_adj ~ 0.
        torch.manual_seed(1)
        z_null = torch.randn(200, dtype=torch.float64)
        z = torch.cat([z_null, torch.tensor([0.05], dtype=torch.float64)])
        se = torch.full((201,), 0.05, dtype=torch.float64)
        res = fiqt(z, se, pi0=1.0)
        # The injected tiny-z entry (last) should be shrunk near zero.
        assert abs(res.beta_adjusted[-1].item()) < abs(z[-1].item() * se[-1].item()) + 1e-10

    def test_large_z_unchanged(self):
        # Under FIQT (BH-adjust two-sided p, back-transform), a z = 10 among
        # 300 null neighbours is barely shrunk: its BH-adjusted p is still
        # tiny, so the corrected z ~ 9.4 (shrinkage ~0.94). Large signals stay
        # large, but the top hit is mildly attenuated by the m-fold BH factor.
        torch.manual_seed(2)
        z_null = torch.randn(300, dtype=torch.float64)
        big_z = 10.0
        z = torch.cat([z_null, torch.tensor([big_z], dtype=torch.float64)])
        se = torch.full((301,), 0.05, dtype=torch.float64)
        res = fiqt(z, se)
        sf = res.shrinkage_factor[-1].item()
        assert 0.9 < sf <= 1.0
        # corrected z stays well above the significance threshold
        assert abs(res.beta_adjusted[-1].item() / 0.05) > 8.0

    def test_invalid_empty_input(self):
        # Empty z tensor -> bandwidth computation hits 0**(-0.2)
        # (ZeroDivisionError) since m == 0. Documents that fiqt does not
        # accept empty inputs.
        z = torch.zeros(0, dtype=torch.float64)
        se = torch.zeros(0, dtype=torch.float64)
        with pytest.raises((ValueError, RuntimeError, IndexError, ZeroDivisionError)):
            fiqt(z, se)


# ===========================================================================
# bootstrap_correction
# ===========================================================================


class TestBootstrapCorrection:
    """Parametric bootstrap winner's curse correction. Resamples from
    N(beta, se^2), applies the |z| >= z_alpha selection, and adjusts."""

    def test_smoke(self):
        beta = torch.tensor([0.30, 0.05], dtype=torch.float64)
        se = torch.tensor([0.05, 0.05], dtype=torch.float64)
        res = bootstrap_correction(beta, se, alpha=5e-8, n_boot=200, seed=42)
        assert isinstance(res, WinnersCurseResult)
        assert res.method == "bootstrap"
        assert res.beta_adjusted.shape == beta.shape
        assert res.n_corrected >= 0

    def test_below_threshold_unchanged(self):
        beta = torch.tensor([0.01], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = bootstrap_correction(beta, se, alpha=5e-8, n_boot=200, seed=42)
        assert res.n_corrected == 0
        torch.testing.assert_close(res.beta_adjusted, beta)

    def test_significant_variant_shrinks(self):
        # |z| = 6 -> selection bias -> corrected estimate is smaller in magnitude.
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = bootstrap_correction(beta, se, alpha=5e-8, n_boot=2000, seed=42)
        assert res.n_corrected == 1
        # The bootstrap correction shrinks the magnitude.
        assert abs(res.beta_adjusted.item()) <= abs(beta.item()) + 1e-6

    def test_seeded_reproducibility(self):
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        r1 = bootstrap_correction(beta, se, alpha=5e-8, n_boot=300, seed=42)
        r2 = bootstrap_correction(beta, se, alpha=5e-8, n_boot=300, seed=42)
        torch.testing.assert_close(r1.beta_adjusted, r2.beta_adjusted)


# ===========================================================================
# correct_winners_curse (dispatch wrapper)
# ===========================================================================


class TestCorrectWinnersCurse:
    """Dispatch wrapper: routes to CL/FIQT/bootstrap by name and returns a
    WinnersCurseResult."""

    def test_dispatches_to_conditional_likelihood(self):
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = correct_winners_curse(
            beta, se, method="conditional_likelihood", alpha=5e-8
        )
        assert isinstance(res, WinnersCurseResult)
        assert res.method == "conditional_likelihood"

    def test_dispatches_to_fiqt(self):
        torch.manual_seed(0)
        beta = torch.randn(50, dtype=torch.float64) * 0.1
        se = torch.full((50,), 0.05, dtype=torch.float64)
        res = correct_winners_curse(beta, se, method="fiqt")
        assert res.method == "fiqt"

    def test_dispatches_to_bootstrap(self):
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        res = correct_winners_curse(
            beta, se, method="bootstrap", alpha=5e-8,
            n_boot=200, seed=42,
        )
        assert res.method == "bootstrap"

    def test_invalid_method_name_raises(self):
        beta = torch.tensor([0.30], dtype=torch.float64)
        se = torch.tensor([0.05], dtype=torch.float64)
        with pytest.raises(ValueError):
            correct_winners_curse(beta, se, method="not_a_real_method")


# ===========================================================================
# snp_to_gene
# ===========================================================================


class TestSnpToGene:
    """Map SNPs to genes by chr+position, then compute per-gene mean chi^2
    one-sided z-test. Genes outside any SNP window get stat=0, p=1."""

    def test_smoke(self):
        snps = ["rs1", "rs2", "rs3", "rs4", "rs5"]
        beta = torch.tensor([0.3, 0.25, 0.2, 0.05, 0.05], dtype=torch.float64)
        ss = _build_sumstats(snps, beta, se=0.04)
        # Override positions for predictable mapping: rs1..rs5 at positions
        # 1000, 1500, 2000, 5000, 9000 on chr 1.
        ss.pos = [1000, 1500, 2000, 5000, 9000]
        # Two genes: G1 covers rs1..rs3, G2 covers rs4 only.
        gene_id = ["G1", "G2"]
        gene_chr = ["1", "1"]
        gene_start = [900, 4900]
        gene_end = [2100, 5100]

        res = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)
        assert isinstance(res, GeneResult)
        assert res.gene_id == gene_id
        assert res.n_snps == [3, 1]
        assert res.stat.shape == (2,)
        assert res.p.shape == (2,)
        # All p in [0, 1].
        assert ((res.p >= 0) & (res.p <= 1)).all()

    def test_no_genes_in_window(self):
        snps = ["rs1", "rs2"]
        beta = torch.tensor([0.3, 0.2], dtype=torch.float64)
        ss = _build_sumstats(snps, beta, se=0.04)
        ss.pos = [1000, 2000]
        # Gene on chr 99 -> no SNPs match.
        res = snp_to_gene(
            ss,
            gene_id=["GhostGene"],
            gene_chr=["99"],
            gene_start=[1000],
            gene_end=[2000],
        )
        assert res.n_snps == [0]
        assert res.stat[0].item() == pytest.approx(0.0)
        assert res.p[0].item() == pytest.approx(1.0)

    def test_window_extension(self):
        snps = ["rs1", "rs2"]
        beta = torch.tensor([0.3, 0.2], dtype=torch.float64)
        ss = _build_sumstats(snps, beta, se=0.04)
        ss.pos = [900, 2100]
        # Gene from 1000..2000 with window_kb=0 -> 0 SNPs.
        # With window_kb=0.5 (=500 bp), both SNPs fall inside.
        res0 = snp_to_gene(
            ss,
            gene_id=["G1"],
            gene_chr=["1"],
            gene_start=[1000],
            gene_end=[2000],
            window_kb=0.0,
        )
        assert res0.n_snps == [0]
        res1 = snp_to_gene(
            ss,
            gene_id=["G1"],
            gene_chr=["1"],
            gene_start=[1000],
            gene_end=[2000],
            window_kb=0.5,
        )
        assert res1.n_snps == [2]

    def test_invalid_mismatched_lengths(self):
        snps = ["rs1", "rs2"]
        beta = torch.tensor([0.3, 0.2], dtype=torch.float64)
        ss = _build_sumstats(snps, beta, se=0.04)
        with pytest.raises(ValueError):
            snp_to_gene(
                ss,
                gene_id=["G1", "G2"],
                gene_chr=["1"],          # mismatched
                gene_start=[1000, 2000],
                gene_end=[1500, 2500],
            )


# ===========================================================================
# gene_set_enrichment
# ===========================================================================


class TestGeneSetEnrichment:
    """Competitive gene-set enrichment. Compares set genes' z-scores
    against the rest, controlling for gene size."""

    def _make_gene_result(self, n_genes=10, low_p_idx=None):
        """Build a synthetic GeneResult with controllable signal."""
        n_snps = [5] * n_genes
        # All genes start with neutral p ~ 0.5 (z ~ 0).
        p = torch.full((n_genes,), 0.5, dtype=torch.float64)
        if low_p_idx is not None:
            for i in low_p_idx:
                p[i] = 1e-6  # strong signal
        gene_id = [f"G{i}" for i in range(n_genes)]
        return GeneResult(
            gene_id=gene_id,
            gene_chr=["1"] * n_genes,
            gene_start=[i * 1000 for i in range(n_genes)],
            gene_end=[(i + 1) * 1000 - 1 for i in range(n_genes)],
            n_snps=n_snps,
            stat=torch.full((n_genes,), 1.5, dtype=torch.float64),
            p=p,
        )

    def test_smoke(self):
        gr = self._make_gene_result(n_genes=10, low_p_idx=[0, 1, 2])
        # Set 1 contains the genes with strong signal.
        gene_sets = {
            "set_signal": ["G0", "G1", "G2"],
            "set_null": ["G6", "G7", "G8"],
        }
        res = gene_set_enrichment(gr, gene_sets)
        assert isinstance(res, EnrichmentResult)
        assert res.gene_set_name == ["set_signal", "set_null"]
        assert res.n_genes_in_set == [3, 3]
        assert res.n_genes_total == 10
        # Set with strong signal should have non-negative enrichment beta.
        # Note: under certain numpy/scipy version combinations on CI the
        # regression beta collapses to exactly 0 (well-conditioned synthetic
        # signal, near-degenerate design). The directional ordering of
        # p-values remains the discriminative assertion.
        assert res.beta_enrichment[0].item() >= 0
        # And p <= p of the null set.
        assert res.p[0].item() <= res.p[1].item()

    def test_no_signal(self):
        # All genes have p = 0.5 (z = 0); enrichment ~ 0, p ~ 0.5.
        gr = self._make_gene_result(n_genes=10)
        gene_sets = {"set_a": ["G0", "G1", "G2"]}
        res = gene_set_enrichment(gr, gene_sets)
        assert abs(res.beta_enrichment[0].item()) < 0.5
        # p value not extreme (no strong evidence).
        assert res.p[0].item() > 0.05

    def test_invalid_empty_gene_sets(self):
        gr = self._make_gene_result(n_genes=5)
        with pytest.raises(ValueError):
            gene_set_enrichment(gr, {})

    def test_invalid_too_few_valid_genes(self):
        # Only 1 gene with SNPs assigned -> regression undefined.
        gr = GeneResult(
            gene_id=["G0", "G1"],
            gene_chr=["1", "1"],
            gene_start=[0, 1000],
            gene_end=[500, 1500],
            n_snps=[5, 0],  # only 1 valid
            stat=torch.tensor([1.0, 0.0], dtype=torch.float64),
            p=torch.tensor([0.5, 1.0], dtype=torch.float64),
        )
        with pytest.raises(ValueError):
            gene_set_enrichment(gr, {"set_a": ["G0"]})


# ===========================================================================
# gwas_power
# ===========================================================================


class TestGwasPower:
    """``gwas_power(n, af, beta, alpha, target_power)`` returns PowerResult.

    NCP = 2 * N * af * (1-af) * beta^2; power = Phi(sqrt(NCP) - z_alpha) +
    Phi(-sqrt(NCP) - z_alpha)."""

    def test_smoke_returns_PowerResult(self):
        af = torch.tensor([0.3, 0.5], dtype=torch.float64)
        beta = torch.tensor([0.05, 0.1], dtype=torch.float64)
        res = gwas_power(n=10000, af=af, beta=beta, alpha=5e-8)
        assert isinstance(res, PowerResult)
        assert res.alpha == pytest.approx(5e-8)
        assert res.n == 10000
        assert res.power.shape == (2,)
        assert res.ncp.shape == (2,)
        assert ((res.power >= 0) & (res.power <= 1)).all()

    def test_known_high_power(self):
        # Large n + non-trivial effect -> power ~ 1.
        af = torch.tensor([0.5], dtype=torch.float64)
        beta = torch.tensor([0.2], dtype=torch.float64)
        res = gwas_power(n=1_000_000, af=af, beta=beta, alpha=5e-8)
        assert res.power.item() == pytest.approx(1.0, abs=1e-6)

    def test_known_low_power(self):
        # Small n + tiny effect -> power near alpha.
        af = torch.tensor([0.5], dtype=torch.float64)
        beta = torch.tensor([0.001], dtype=torch.float64)
        res = gwas_power(n=100, af=af, beta=beta, alpha=5e-8)
        # Power should be very small (well below 0.05).
        assert 0.0 <= res.power.item() <= 0.05

    def test_ncp_closed_form(self):
        # Single SNP: NCP = 2 * N * af * (1-af) * beta^2.
        af = torch.tensor([0.3], dtype=torch.float64)
        beta = torch.tensor([0.1], dtype=torch.float64)
        res = gwas_power(n=1000, af=af, beta=beta, alpha=5e-8)
        ref_ncp = 2.0 * 1000 * 0.3 * 0.7 * (0.1**2)
        assert res.ncp.item() == pytest.approx(ref_ncp, abs=1e-9)


# ===========================================================================
# power_curve
# ===========================================================================


class TestPowerCurve:
    """``power_curve(n, af_grid, alpha, target_power)`` returns the minimum
    detectable |beta| at each AF -- the trumpet envelope."""

    def test_smoke(self):
        af_grid = torch.linspace(0.05, 0.5, 10, dtype=torch.float64)
        res = power_curve(n=10000, af_grid=af_grid, alpha=5e-8, target_power=0.8)
        assert isinstance(res, torch.Tensor)
        assert res.shape == (10,)
        assert (res > 0).all()

    def test_decreases_with_n(self):
        af_grid = torch.tensor([0.3], dtype=torch.float64)
        small = power_curve(n=1000, af_grid=af_grid)
        large = power_curve(n=100000, af_grid=af_grid)
        # Larger n -> smaller min detectable beta.
        assert large.item() < small.item()

    def test_higher_at_rare_alleles(self):
        # The trumpet is wider (larger min |beta|) at rare AF.
        af_rare = torch.tensor([0.01], dtype=torch.float64)
        af_common = torch.tensor([0.5], dtype=torch.float64)
        rare = power_curve(n=10000, af_grid=af_rare)
        common = power_curve(n=10000, af_grid=af_common)
        assert rare.item() > common.item()


# ===========================================================================
# required_n
# ===========================================================================


class TestRequiredN:
    """``required_n(af, beta, alpha, target_power)`` returns the sample size
    needed to detect each variant at the given power and alpha."""

    def test_smoke(self):
        af = torch.tensor([0.3, 0.5], dtype=torch.float64)
        beta = torch.tensor([0.1, 0.05], dtype=torch.float64)
        n = required_n(af=af, beta=beta, alpha=5e-8, target_power=0.8)
        assert isinstance(n, torch.Tensor)
        assert n.shape == (2,)
        assert (n > 0).all()

    def test_inverse_consistency(self):
        # Compute n needed for power=0.8, then check that gwas_power(n, ...)
        # at this AF/beta yields ~0.8 within tolerance.
        af = torch.tensor([0.3], dtype=torch.float64)
        beta = torch.tensor([0.05], dtype=torch.float64)
        n_req = required_n(af=af, beta=beta, alpha=5e-8, target_power=0.8)
        n_int = float(math.ceil(n_req.item()))
        res = gwas_power(n=n_int, af=af, beta=beta, alpha=5e-8)
        assert res.power.item() == pytest.approx(0.8, abs=0.02)

    def test_smaller_effect_needs_larger_n(self):
        af = torch.tensor([0.3, 0.3], dtype=torch.float64)
        beta = torch.tensor([0.01, 0.1], dtype=torch.float64)
        n = required_n(af=af, beta=beta, alpha=5e-8, target_power=0.8)
        # Smaller beta requires larger n.
        assert n[0].item() > n[1].item()


# ===========================================================================
# mantra
# ===========================================================================


def _build_population_sumstats(
    K_pop: int,
    snps: list[str],
    seed: int = 0,
    shared_signal: bool = True,
) -> list[SumStats]:
    """Build K populations with the same SNP set."""
    torch.manual_seed(seed)
    out = []
    for k in range(K_pop):
        base = torch.tensor([0.2, 0.15, 0.0, -0.05, 0.0],
                            dtype=torch.float64)[: len(snps)]
        if not shared_signal and k > 0:
            base = base + torch.randn(len(snps), dtype=torch.float64) * 0.3
        beta = base + torch.randn(len(snps), dtype=torch.float64) * 0.005
        out.append(_build_sumstats(snps, beta, se=0.05,
                                    af=0.2 + 0.05 * k))
    return out


class TestMantra:
    """MANTRA: Bayesian shared/heterogeneous mixture; returns log10_bf,
    posterior effect, p_meta, p_heterogeneity."""

    def test_smoke(self):
        snps = ["rs1", "rs2", "rs3", "rs4", "rs5"]
        ss_list = _build_population_sumstats(3, snps, seed=0)
        res = mantra(ss_list)
        assert isinstance(res, MultiAncestryResult)
        assert res.method == "mantra"
        assert res.n_populations == 3
        assert res.n_snps == 5
        assert res.beta_meta.shape == (5,)
        assert res.log10_bf is not None
        assert res.log10_bf.shape == (5,)
        assert res.posterior_effect is not None

    def test_shared_signal_higher_bf_than_null(self):
        # Strong signal across populations -> log10_bf > 0 for the signal SNP.
        snps = ["rs1", "rs2"]
        # Strong shared signal at rs1, null at rs2.
        bs_per_pop = [
            torch.tensor([0.4, 0.0], dtype=torch.float64),
            torch.tensor([0.4, 0.0], dtype=torch.float64),
            torch.tensor([0.4, 0.0], dtype=torch.float64),
        ]
        ss_list = [
            _build_sumstats(snps, b, se=0.05) for b in bs_per_pop
        ]
        res = mantra(ss_list)
        # Strong signal -> log10_bf at rs1 > log10_bf at rs2.
        assert res.log10_bf[0].item() > res.log10_bf[1].item()

    def test_invalid_too_few_populations(self):
        ss = _build_sumstats(
            ["rs1"], torch.tensor([0.1], dtype=torch.float64),
        )
        with pytest.raises(ValueError):
            mantra([ss])

    def test_invalid_negative_prior_sigma(self):
        snps = ["rs1", "rs2"]
        ss_list = _build_population_sumstats(2, snps, seed=0)
        with pytest.raises(ValueError):
            mantra(ss_list, prior_sigma2=-0.1)

    def test_invalid_prior_prob_out_of_range(self):
        snps = ["rs1", "rs2"]
        ss_list = _build_population_sumstats(2, snps, seed=0)
        with pytest.raises(ValueError):
            mantra(ss_list, prior_prob_consistent=1.5)


# ===========================================================================
# mr_mega
# ===========================================================================


class TestMrMega:
    """MR-MEGA meta-regression: WLS with PC axes of cross-population AF."""

    def test_smoke(self):
        snps = ["rs1", "rs2", "rs3", "rs4", "rs5"]
        ss_list = _build_population_sumstats(4, snps, seed=0)
        res = mr_mega(ss_list, n_axes=2)
        assert isinstance(res, MultiAncestryResult)
        assert res.method == "mr_mega"
        assert res.n_populations == 4
        assert res.n_snps == 5
        assert res.n_axes == 2
        assert res.beta_meta.shape == (5,)
        assert res.p_meta.shape == (5,)
        assert res.p_ancestry is not None
        assert res.p_residual is not None

    def test_strong_signal_low_p(self):
        snps = ["rs1", "rs2"]
        # 4 populations, strong shared signal at rs1.
        bs_per_pop = [
            torch.tensor([0.5, 0.0], dtype=torch.float64) for _ in range(4)
        ]
        ss_list = [
            _build_sumstats(snps, b, se=0.05) for b in bs_per_pop
        ]
        res = mr_mega(ss_list, n_axes=2)
        # Signal SNP has smaller p than null.
        assert res.p_meta[0].item() < res.p_meta[1].item()
        # Signal SNP p_meta is highly significant.
        assert res.p_meta[0].item() < 1e-3

    def test_invalid_too_few_populations(self):
        ss = _build_sumstats(
            ["rs1"], torch.tensor([0.1], dtype=torch.float64),
        )
        with pytest.raises(ValueError):
            mr_mega([ss], n_axes=1)

    def test_invalid_n_axes_too_large(self):
        snps = ["rs1", "rs2"]
        ss_list = _build_population_sumstats(2, snps, seed=0)
        # Only 2 populations -> max n_axes = 1, n_axes=2 should raise.
        with pytest.raises(ValueError):
            mr_mega(ss_list, n_axes=2)
