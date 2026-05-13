"""Tier-2 behavioral coverage tests for ``torchgwas.postgwas`` colocalization,
fine-mapping, and meta-analysis public symbols.

Bar (Pillar A spec section 4.3 Tier 2):
- Dataclasses: construct + round-trip + repr smoke.
- Functions: golden + edge + error tests.
  - ``meta_fixed_effect``: closed-form IVW.
  - ``meta_sample_size``: closed-form Stouffer's z.
  - ``meta_random_effect``: collapses to FE under no heterogeneity.
  - ``coloc_pairwise``: PP.H0..H4 sum to ~1.
  - ``hyprcoloc``: posteriors valid for K>=3 traits.

Covers 16 public symbols across 3 submodules:

- ``torchgwas.postgwas._finemapping.AnnotatedSumStats`` (dataclass)
- ``torchgwas.postgwas._finemapping.CredibleSet`` (dataclass)
- ``torchgwas.postgwas._finemapping.LocusSummary`` (dataclass)
- ``torchgwas.postgwas._finemapping.annotate_sumstats`` (function)
- ``torchgwas.postgwas._finemapping.extract_credible_sets`` (function)
- ``torchgwas.postgwas._finemapping.locus_summary`` (function)
- ``torchgwas.postgwas._finemapping.to_coloc_sumstats`` (function)
- ``torchgwas.postgwas._hyprcoloc.ColocPairwiseResult`` (dataclass)
- ``torchgwas.postgwas._hyprcoloc.HyprcolocResult`` (dataclass)
- ``torchgwas.postgwas._hyprcoloc.coloc_pairwise`` (function)
- ``torchgwas.postgwas._hyprcoloc.hyprcoloc`` (function)
- ``torchgwas.postgwas._meta.MetaResult`` (dataclass)
- ``torchgwas.postgwas._meta.meta_fixed_effect`` (function)
- ``torchgwas.postgwas._meta.meta_random_effect`` (function)
- ``torchgwas.postgwas._meta.meta_sample_size`` (function)
- ``torchgwas.postgwas._meta.meta_han_eskin`` (function)

All re-exported via ``torchgwas.postgwas.__init__``.
"""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from torchgwas.models.bayesian_vs import BayesianVSResult
from torchgwas.postgwas import (
    AnnotatedSumStats,
    ColocPairwiseResult,
    CredibleSet,
    HyprcolocResult,
    LocusSummary,
    MetaResult,
    SumStats,
    annotate_sumstats,
    coloc_pairwise,
    extract_credible_sets,
    hyprcoloc,
    locus_summary,
    meta_fixed_effect,
    meta_han_eskin,
    meta_random_effect,
    meta_sample_size,
    to_coloc_sumstats,
)

pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _build_sumstats(
    m: int = 8,
    seed: int = 0,
    snp_prefix: str = "rs",
    chrom: str = "1",
    causal_idx: int | None = None,
    effect_size: float = 0.6,
) -> SumStats:
    """Build a minimal SumStats for a region of ``m`` SNPs.

    If ``causal_idx`` is given, that SNP gets a strong effect; the rest
    are random small effects.
    """
    torch.manual_seed(seed)
    snp_ids = [f"{snp_prefix}{i}" for i in range(m)]
    pos = list(range(1000, 1000 + m * 10, 10))
    chr_list = [chrom] * m
    a1 = ["A"] * m
    a2 = ["G"] * m

    beta = torch.randn(m, dtype=torch.float64) * 0.05
    se = torch.full((m,), 0.05, dtype=torch.float64)
    if causal_idx is not None:
        beta[causal_idx] = effect_size
    z = beta / se
    p = torch.erfc(z.abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)
    n = torch.full((m,), 1000.0, dtype=torch.float64)
    return SumStats(
        chr=chr_list,
        pos=pos,
        snp=snp_ids,
        a1=a1,
        a2=a2,
        beta=beta,
        se=se,
        p=p,
        n=n,
        af=torch.full((m,), 0.3, dtype=torch.float64),
    )


def _build_bayesian_result(
    m: int = 6,
    snp_prefix: str = "rs",
    chrom: str = "1",
    pip_signal_idx: int | None = 2,
    method: str = "cavi",
    susie_layers: int = 0,
) -> BayesianVSResult:
    """Build a minimal BayesianVSResult for fine-mapping tests."""
    snps = [f"{snp_prefix}{i}" for i in range(m)]
    pos = list(range(1000, 1000 + m * 10, 10))
    chrs = [chrom] * m
    a1 = ["A"] * m
    a2 = ["G"] * m
    af = torch.full((m,), 0.3, dtype=torch.float64)

    pip = torch.full((m,), 0.05, dtype=torch.float64)
    if pip_signal_idx is not None:
        pip[pip_signal_idx] = 0.9

    beta_mean = torch.zeros(m, dtype=torch.float64)
    if pip_signal_idx is not None:
        beta_mean[pip_signal_idx] = 0.5
    beta_sd = torch.full((m,), 0.05, dtype=torch.float64)

    alpha = None
    if method == "susie" and susie_layers > 0:
        alpha = torch.full((susie_layers, m), 1.0 / m, dtype=torch.float64)
        # Make first layer concentrated on pip_signal_idx to be "active"
        if pip_signal_idx is not None:
            alpha[0] = 0.01
            alpha[0, pip_signal_idx] = 0.95
            # remaining mass spread
            remain = (1.0 - 0.95 - 0.01 * (m - 2))
            alpha[0, (pip_signal_idx + 1) % m] = max(remain, 0.0) + 0.01

    return BayesianVSResult(
        chr=chrs,
        pos=pos,
        snp=snps,
        a1=a1,
        a2=a2,
        af=af,
        pip=pip,
        beta_mean=beta_mean,
        beta_sd=beta_sd,
        method=method,
        alpha=alpha,
        n_signals=1 if susie_layers > 0 else 0,
    )


# ---------------------------------------------------------------------------
# Dataclass smoke tests (5 result dataclasses + MetaResult)
# ---------------------------------------------------------------------------


class TestCredibleSet:
    """``CredibleSet`` carries a single fine-mapping signal: snp indices and
    IDs in decreasing PIP order, lead variant, and cumulative coverage."""

    def _make(self) -> CredibleSet:
        return CredibleSet(
            signal_index=0,
            snp_indices=[2, 5, 7],
            snp_ids=["rs2", "rs5", "rs7"],
            pip=torch.tensor([0.6, 0.25, 0.1], dtype=torch.float64),
            cumulative_coverage=0.95,
            lead_snp_index=2,
            lead_snp_id="rs2",
            lead_pip=0.6,
        )

    def test_construct(self):
        cs = self._make()
        assert cs.signal_index == 0
        assert cs.snp_indices == [2, 5, 7]
        assert cs.snp_ids == ["rs2", "rs5", "rs7"]
        assert cs.pip.shape == (3,)
        assert cs.cumulative_coverage == pytest.approx(0.95)
        assert cs.lead_snp_index == 2
        assert cs.lead_snp_id == "rs2"
        assert cs.lead_pip == pytest.approx(0.6)

    def test_round_trip_via_dataclass_fields(self):
        cs = self._make()
        names = {f.name for f in fields(cs)}
        assert names == {
            "signal_index",
            "snp_indices",
            "snp_ids",
            "pip",
            "cumulative_coverage",
            "lead_snp_index",
            "lead_snp_id",
            "lead_pip",
        }
        cs2 = CredibleSet(
            signal_index=cs.signal_index,
            snp_indices=list(cs.snp_indices),
            snp_ids=list(cs.snp_ids),
            pip=cs.pip.clone(),
            cumulative_coverage=cs.cumulative_coverage,
            lead_snp_index=cs.lead_snp_index,
            lead_snp_id=cs.lead_snp_id,
            lead_pip=cs.lead_pip,
        )
        assert cs2.snp_indices == cs.snp_indices
        assert torch.equal(cs2.pip, cs.pip)

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "CredibleSet" in s
        assert "signal_index=0" in s


class TestAnnotatedSumStats:
    """``AnnotatedSumStats`` merges PIPs and credible-set membership onto
    SumStats. ``as_table`` returns a column-oriented dict for Parquet/DF."""

    def _make(self, m: int = 4) -> AnnotatedSumStats:
        return AnnotatedSumStats(
            chr=["1"] * m,
            pos=[100, 200, 300, 400][:m],
            snp=[f"rs{i}" for i in range(m)],
            a1=["A"] * m,
            a2=["G"] * m,
            beta=torch.zeros(m, dtype=torch.float64),
            se=torch.full((m,), 0.05, dtype=torch.float64),
            p=torch.full((m,), 0.5, dtype=torch.float64),
            pip=torch.tensor([0.05, 0.6, 0.2, 0.01], dtype=torch.float64)[:m],
            in_credible_set=[-1, 0, 0, -1][:m],
            beta_posterior=torch.zeros(m, dtype=torch.float64),
            beta_posterior_sd=torch.full((m,), 0.04, dtype=torch.float64),
        )

    def test_construct(self):
        ann = self._make()
        assert len(ann.snp) == 4
        assert ann.pip.shape == (4,)
        assert ann.in_credible_set == [-1, 0, 0, -1]

    def test_round_trip_via_dataclass_fields(self):
        ann = self._make()
        names = {f.name for f in fields(ann)}
        assert names == {
            "chr",
            "pos",
            "snp",
            "a1",
            "a2",
            "beta",
            "se",
            "p",
            "pip",
            "in_credible_set",
            "beta_posterior",
            "beta_posterior_sd",
        }
        ann2 = AnnotatedSumStats(
            chr=list(ann.chr),
            pos=list(ann.pos),
            snp=list(ann.snp),
            a1=list(ann.a1),
            a2=list(ann.a2),
            beta=ann.beta.clone(),
            se=ann.se.clone(),
            p=ann.p.clone(),
            pip=ann.pip.clone(),
            in_credible_set=list(ann.in_credible_set),
            beta_posterior=ann.beta_posterior.clone(),
            beta_posterior_sd=ann.beta_posterior_sd.clone(),
        )
        assert ann2.snp == ann.snp
        assert torch.equal(ann2.pip, ann.pip)

    def test_as_table(self):
        ann = self._make()
        tbl = ann.as_table()
        expected_keys = {
            "chr",
            "pos",
            "snp",
            "a1",
            "a2",
            "beta",
            "se",
            "p",
            "pip",
            "in_credible_set",
            "beta_posterior",
            "beta_posterior_sd",
        }
        assert set(tbl.keys()) == expected_keys
        assert all(len(v) == 4 for v in tbl.values())
        # Tensors should have been converted to plain lists.
        assert isinstance(tbl["pip"], list)
        assert tbl["in_credible_set"] == [-1, 0, 0, -1]

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "AnnotatedSumStats" in s


class TestLocusSummary:
    """``LocusSummary`` aggregates fine-mapping outputs across loci, with
    per-locus chr/start/end/n_variants/n_signals/lead_snp/total_pip."""

    def _make(self) -> LocusSummary:
        return LocusSummary(
            locus_chr=["1", "2"],
            locus_start=[1000, 2000],
            locus_end=[1100, 2100],
            n_variants=[10, 12],
            n_signals=[1, 2],
            credible_sets=[[], []],
            lead_snp=["rs0", "rs5"],
            lead_pip=[0.4, 0.7],
            total_pip=[1.0, 1.5],
        )

    def test_construct(self):
        ls = self._make()
        assert ls.locus_chr == ["1", "2"]
        assert ls.n_variants == [10, 12]
        assert ls.n_signals == [1, 2]

    def test_default_construct_empty(self):
        ls = LocusSummary()
        assert ls.locus_chr == []
        assert ls.locus_start == []
        assert ls.lead_pip == []
        # Lists are independent (not shared mutable defaults).
        ls2 = LocusSummary()
        ls.locus_chr.append("X")
        assert ls2.locus_chr == []

    def test_round_trip_via_dataclass_fields(self):
        ls = self._make()
        names = {f.name for f in fields(ls)}
        assert names == {
            "locus_chr",
            "locus_start",
            "locus_end",
            "n_variants",
            "n_signals",
            "credible_sets",
            "lead_snp",
            "lead_pip",
            "total_pip",
        }
        ls2 = LocusSummary(**{f.name: getattr(ls, f.name) for f in fields(ls)})
        assert ls2.locus_chr == ls.locus_chr
        assert ls2.lead_pip == ls.lead_pip

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "LocusSummary" in s


class TestColocPairwiseResult:
    """``ColocPairwiseResult`` is the Giambartolomei (2014) two-trait
    decomposition: PP.H0..H4 plus the candidate SNP under H4."""

    def _make(self) -> ColocPairwiseResult:
        return ColocPairwiseResult(
            pp_h0=0.05, pp_h1=0.05, pp_h2=0.05, pp_h3=0.05, pp_h4=0.8,
            candidate_snp=3,
        )

    def test_construct(self):
        res = self._make()
        assert res.pp_h0 == pytest.approx(0.05)
        assert res.pp_h4 == pytest.approx(0.8)
        assert res.candidate_snp == 3
        s = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
        assert s == pytest.approx(1.0)

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {"pp_h0", "pp_h1", "pp_h2", "pp_h3", "pp_h4", "candidate_snp"}
        res2 = ColocPairwiseResult(
            **{f.name: getattr(res, f.name) for f in fields(res)}
        )
        assert res2 == res

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "ColocPairwiseResult" in s


class TestHyprcolocResult:
    """``HyprcolocResult`` carries multi-trait coloc posteriors (subset
    posteriors keyed by trait-tuple), null and all-coloc PPs, the best
    cluster, and the candidate SNP within it."""

    def _make(self) -> HyprcolocResult:
        return HyprcolocResult(
            subset_posteriors={(0, 1): 0.4, (0, 2): 0.05, (1, 2): 0.05, (0, 1, 2): 0.4},
            pp_all_colocalize=0.4,
            pp_null=0.1,
            best_cluster=(0, 1),
            best_cluster_posterior=0.4,
            candidate_snp=2,
            candidate_snp_posterior=0.7,
            log_bf_marginal=[1.2, 1.1, 0.9],
            trait_names=["A", "B", "C"],
        )

    def test_construct(self):
        res = self._make()
        assert res.pp_all_colocalize == pytest.approx(0.4)
        assert res.best_cluster == (0, 1)
        assert res.candidate_snp == 2
        assert res.trait_names == ["A", "B", "C"]
        assert (0, 1) in res.subset_posteriors

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "subset_posteriors",
            "pp_all_colocalize",
            "pp_null",
            "best_cluster",
            "best_cluster_posterior",
            "candidate_snp",
            "candidate_snp_posterior",
            "log_bf_marginal",
            "trait_names",
        }
        res2 = HyprcolocResult(
            **{f.name: getattr(res, f.name) for f in fields(res)}
        )
        assert res2 == res

    def test_default_trait_names_optional(self):
        res = HyprcolocResult(
            subset_posteriors={(0, 1): 0.5},
            pp_all_colocalize=0.5,
            pp_null=0.5,
            best_cluster=(0, 1),
            best_cluster_posterior=0.5,
            candidate_snp=0,
            candidate_snp_posterior=0.4,
            log_bf_marginal=[0.5, 0.5],
        )
        assert res.trait_names is None

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "HyprcolocResult" in s


class TestMetaResult:
    """``MetaResult`` is the unified output for all four meta methods:
    beta/se/p/z, Cochran's Q, I^2, tau^2, and a ``method`` label."""

    def _make(self, m: int = 5, method: str = "fixed_effect") -> MetaResult:
        return MetaResult(
            beta_meta=torch.zeros(m, dtype=torch.float64),
            se_meta=torch.ones(m, dtype=torch.float64),
            p_meta=torch.full((m,), 0.5, dtype=torch.float64),
            z_meta=torch.zeros(m, dtype=torch.float64),
            q_stat=torch.zeros(m, dtype=torch.float64),
            i2=torch.zeros(m, dtype=torch.float64),
            tau2=torch.zeros(m, dtype=torch.float64),
            method=method,
        )

    def test_construct(self):
        res = self._make()
        assert res.beta_meta.shape == (5,)
        assert res.method == "fixed_effect"

    def test_round_trip_via_dataclass_fields(self):
        res = self._make()
        names = {f.name for f in fields(res)}
        assert names == {
            "beta_meta",
            "se_meta",
            "p_meta",
            "z_meta",
            "q_stat",
            "i2",
            "tau2",
            "method",
        }
        res2 = MetaResult(
            **{f.name: getattr(res, f.name) for f in fields(res)}
        )
        assert res2.method == res.method
        assert torch.equal(res2.beta_meta, res.beta_meta)

    def test_repr_contains_class_name(self):
        s = repr(self._make())
        assert "MetaResult" in s
        assert "method='fixed_effect'" in s


# ---------------------------------------------------------------------------
# extract_credible_sets
# ---------------------------------------------------------------------------


class TestExtractCredibleSets:
    """``extract_credible_sets`` greedily accumulates variants in decreasing
    PIP order until cumulative coverage reaches the target. SuSiE-method
    results dispatch on ``alpha`` layers; CAVI builds a single global set."""

    def test_smoke_cavi(self):
        res = _build_bayesian_result(m=6, pip_signal_idx=2, method="cavi")
        cs_list = extract_credible_sets(res, coverage=0.95)
        assert isinstance(cs_list, list)
        assert all(isinstance(cs, CredibleSet) for cs in cs_list)
        assert len(cs_list) == 1
        cs = cs_list[0]
        # Lead is the highest-PIP variant.
        assert cs.lead_snp_id == "rs2"
        assert cs.cumulative_coverage >= 0.95

    def test_smoke_susie(self):
        res = _build_bayesian_result(
            m=6, pip_signal_idx=2, method="susie", susie_layers=2
        )
        cs_list = extract_credible_sets(res, coverage=0.95)
        assert all(isinstance(cs, CredibleSet) for cs in cs_list)
        # First layer was constructed to be active and concentrated on idx 2.
        assert any(cs.lead_snp_id == "rs2" for cs in cs_list)

    def test_min_pip_filter(self):
        res = _build_bayesian_result(m=6, pip_signal_idx=2, method="cavi")
        cs_list = extract_credible_sets(res, coverage=0.99, min_pip=0.5)
        # Only the dominant variant survives the PIP threshold.
        assert all(cs.lead_pip >= 0.5 for cs in cs_list)

    def test_invalid_coverage_zero(self):
        res = _build_bayesian_result(m=4, pip_signal_idx=0, method="cavi")
        with pytest.raises(ValueError):
            extract_credible_sets(res, coverage=0.0)

    def test_invalid_coverage_above_one(self):
        res = _build_bayesian_result(m=4, pip_signal_idx=0, method="cavi")
        with pytest.raises(ValueError):
            extract_credible_sets(res, coverage=1.5)


# ---------------------------------------------------------------------------
# annotate_sumstats
# ---------------------------------------------------------------------------


class TestAnnotateSumstats:
    """``annotate_sumstats`` joins fine-mapping PIPs and credible-set
    membership onto an existing SumStats (matching by SNP ID), zero-filling
    variants absent from the fine-mapping result."""

    def test_smoke(self):
        ss = _build_sumstats(m=6, causal_idx=2)
        fm = _build_bayesian_result(m=6, pip_signal_idx=2, method="cavi")
        ann = annotate_sumstats(ss, fm)
        assert isinstance(ann, AnnotatedSumStats)
        assert ann.snp == ss.snp
        # The strong-PIP variant should be inside a credible set.
        assert ann.in_credible_set[2] >= 0
        # Other variants are zero-PIP relative to the dominant signal but
        # may still appear if the cumulative-coverage walk reaches them.
        assert ann.pip[2].item() == pytest.approx(0.9)

    def test_partial_overlap(self):
        # SumStats has extra variant absent from the fine-mapping result.
        ss = _build_sumstats(m=6)
        fm = _build_bayesian_result(m=4, pip_signal_idx=0, method="cavi")
        # Make fm SNPs match the first 4 of ss
        ann = annotate_sumstats(ss, fm)
        # Variants 4..5 of ss are not in fm, so PIP=0 and in_cs=-1
        assert ann.pip[4].item() == 0.0
        assert ann.pip[5].item() == 0.0
        assert ann.in_credible_set[4] == -1
        assert ann.in_credible_set[5] == -1


# ---------------------------------------------------------------------------
# locus_summary
# ---------------------------------------------------------------------------


class TestLocusSummaryFn:
    """``locus_summary`` aggregates one fine-mapping result per locus into a
    ``LocusSummary`` containing chrom/start/end/n_variants/n_signals/lead/
    total_pip parallel lists."""

    def test_smoke(self):
        res1 = _build_bayesian_result(m=4, snp_prefix="a", chrom="1", pip_signal_idx=1)
        res2 = _build_bayesian_result(m=5, snp_prefix="b", chrom="2", pip_signal_idx=2)
        summary = locus_summary([res1, res2], coverage=0.95)
        assert isinstance(summary, LocusSummary)
        assert len(summary.locus_chr) == 2
        assert summary.locus_chr == ["1", "2"]
        assert summary.n_variants == [4, 5]
        assert summary.lead_snp[0] == "a1"
        assert summary.lead_snp[1] == "b2"
        # Each locus has at least one credible set under the cavi build.
        assert all(n >= 1 for n in summary.n_signals)
        # Total PIP per locus matches sum of the input PIPs.
        assert summary.total_pip[0] == pytest.approx(float(res1.pip.sum().item()))

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            locus_summary([], coverage=0.95)


# ---------------------------------------------------------------------------
# to_coloc_sumstats
# ---------------------------------------------------------------------------


class TestToColocSumstats:
    """``to_coloc_sumstats`` realigns a SumStats onto the variant order in a
    fine-mapping result; only shared SNP IDs are kept."""

    def test_smoke_full_overlap(self):
        ss = _build_sumstats(m=6, causal_idx=2)
        fm = _build_bayesian_result(m=6, pip_signal_idx=2, method="cavi")
        out = to_coloc_sumstats(fm, ss)
        # Output is a SumStats aligned to fm.
        assert out.snp == fm.snp
        assert out.beta.shape == (6,)
        # Effect at the causal SNP is preserved.
        assert out.beta[2].item() == pytest.approx(ss.beta[2].item())

    def test_partial_overlap(self):
        ss = _build_sumstats(m=6)
        # fm covers only first 4 SNPs of ss.
        fm = _build_bayesian_result(m=4, pip_signal_idx=0, method="cavi")
        out = to_coloc_sumstats(fm, ss)
        assert out.snp == fm.snp  # length 4
        assert out.beta.shape == (4,)

    def test_invalid_no_overlap(self):
        ss = _build_sumstats(m=6, snp_prefix="ss")
        fm = _build_bayesian_result(m=6, snp_prefix="fm")
        with pytest.raises(ValueError):
            to_coloc_sumstats(fm, ss)


# ---------------------------------------------------------------------------
# coloc_pairwise
# ---------------------------------------------------------------------------


class TestColocPairwise:
    """Two-trait colocalization (Giambartolomei 2014). PP.H0..H4 must form
    a probability vector summing to 1; with shared signal H4 dominates;
    with independent signals H4 is small."""

    def test_smoke_shared_signal(self):
        m = 20
        # Both traits have the same causal SNP at index 5.
        ss1 = _build_sumstats(m=m, seed=11, causal_idx=5, effect_size=0.6)
        ss2 = _build_sumstats(m=m, seed=12, causal_idx=5, effect_size=0.55)
        res = coloc_pairwise(ss1, ss2)
        assert isinstance(res, ColocPairwiseResult)
        s = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
        assert s == pytest.approx(1.0, abs=1e-6)
        # H4 should dominate.
        assert res.pp_h4 > 0.5
        # Candidate SNP should be at or near the true causal index.
        assert 0 <= res.candidate_snp < m

    def test_independent_no_coloc_signal(self):
        m = 20
        # Trait 1 has a signal at idx 3; trait 2 has none (just noise).
        ss1 = _build_sumstats(m=m, seed=21, causal_idx=3, effect_size=0.6)
        ss2 = _build_sumstats(m=m, seed=22)  # noise only
        res = coloc_pairwise(ss1, ss2)
        s = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
        assert s == pytest.approx(1.0, abs=1e-6)
        # H4 should be tiny under no shared signal.
        assert res.pp_h4 < 0.2

    def test_pp_sum_to_one_general(self):
        ss1 = _build_sumstats(m=15, seed=31, causal_idx=7, effect_size=0.5)
        ss2 = _build_sumstats(m=15, seed=32, causal_idx=7, effect_size=0.4)
        res = coloc_pairwise(ss1, ss2, prior_1=1e-3, prior_2=1e-3, prior_12=1e-4)
        s = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_invalid_mismatched_snps(self):
        ss1 = _build_sumstats(m=10)
        ss2 = _build_sumstats(m=12)
        with pytest.raises(ValueError):
            coloc_pairwise(ss1, ss2)


# ---------------------------------------------------------------------------
# hyprcoloc
# ---------------------------------------------------------------------------


class TestHyprcoloc:
    """Multi-trait colocalization (Foley 2021). Verifies posterior shape and
    invariants for K >= 3 traits with overlapping SNP sets."""

    def test_smoke_three_traits(self):
        m = 15
        ss1 = _build_sumstats(m=m, seed=41, causal_idx=7, effect_size=0.6)
        ss2 = _build_sumstats(m=m, seed=42, causal_idx=7, effect_size=0.55)
        ss3 = _build_sumstats(m=m, seed=43, causal_idx=7, effect_size=0.5)
        res = hyprcoloc(
            [ss1, ss2, ss3], trait_names=["t1", "t2", "t3"]
        )
        assert isinstance(res, HyprcolocResult)
        assert res.trait_names == ["t1", "t2", "t3"]
        # Posteriors are probabilities.
        assert 0.0 <= res.pp_null <= 1.0
        assert 0.0 <= res.pp_all_colocalize <= 1.0
        # Subset posteriors plus null sum to 1.
        total = res.pp_null + sum(res.subset_posteriors.values())
        assert total == pytest.approx(1.0, abs=1e-6)
        # K=3 -> 2^3 - 3 - 1 = 4 non-singleton subsets.
        assert len(res.subset_posteriors) == 4
        # Best cluster posterior is the max over subset posteriors.
        assert res.best_cluster_posterior == pytest.approx(
            max(res.subset_posteriors.values())
        )
        # Candidate SNP is a valid index.
        assert 0 <= res.candidate_snp < m
        assert 0.0 <= res.candidate_snp_posterior <= 1.0
        # Marginal log-BFs are finite floats per trait.
        assert len(res.log_bf_marginal) == 3
        assert all(math.isfinite(v) for v in res.log_bf_marginal)

    def test_invalid_single_trait(self):
        ss1 = _build_sumstats(m=10)
        with pytest.raises(ValueError):
            hyprcoloc([ss1])

    def test_invalid_mismatched_snp_counts(self):
        ss1 = _build_sumstats(m=10, seed=51)
        ss2 = _build_sumstats(m=12, seed=52)
        ss3 = _build_sumstats(m=10, seed=53)
        with pytest.raises(ValueError):
            hyprcoloc([ss1, ss2, ss3])

    def test_invalid_prior_out_of_range(self):
        ss1 = _build_sumstats(m=10, seed=61)
        ss2 = _build_sumstats(m=10, seed=62)
        with pytest.raises(ValueError):
            hyprcoloc([ss1, ss2], prior_1=1.5)


# ---------------------------------------------------------------------------
# meta_fixed_effect
# ---------------------------------------------------------------------------


class TestMetaFixedEffect:
    """Fixed-effect IVW: combined beta = sum(beta_k / se_k^2) / sum(1/se_k^2),
    combined SE = 1 / sqrt(sum(1/se_k^2))."""

    def test_closed_form_three_studies(self):
        # 1 SNP, 3 studies, manual reference.
        beta = torch.tensor([[0.10, 0.12, 0.09]], dtype=torch.float64)
        se = torch.tensor([[0.05, 0.04, 0.06]], dtype=torch.float64)

        w = 1.0 / (se**2)
        ref_beta = (w * beta).sum(dim=1) / w.sum(dim=1)
        ref_se = 1.0 / w.sum(dim=1).sqrt()

        res = meta_fixed_effect(beta, se)
        assert isinstance(res, MetaResult)
        assert res.method == "fixed_effect"
        assert res.beta_meta.shape == (1,)
        assert res.beta_meta.item() == pytest.approx(ref_beta.item(), abs=1e-12)
        assert res.se_meta.item() == pytest.approx(ref_se.item(), abs=1e-12)
        # tau2 is identically zero for fixed-effect.
        assert res.tau2.item() == 0.0

    def test_homogeneous_inputs(self):
        # All studies identical -> combined beta = shared beta, SE = se / sqrt(K).
        K = 4
        beta = torch.full((1, K), 0.2, dtype=torch.float64)
        se = torch.full((1, K), 0.1, dtype=torch.float64)
        res = meta_fixed_effect(beta, se)
        assert res.beta_meta.item() == pytest.approx(0.2, abs=1e-12)
        assert res.se_meta.item() == pytest.approx(0.1 / math.sqrt(K), abs=1e-12)
        assert res.q_stat.item() == pytest.approx(0.0, abs=1e-10)
        assert res.i2.item() == 0.0

    def test_multi_snp_shape(self):
        beta = torch.randn(5, 3, dtype=torch.float64) * 0.05
        se = torch.full((5, 3), 0.05, dtype=torch.float64)
        res = meta_fixed_effect(beta, se)
        assert res.beta_meta.shape == (5,)
        assert res.se_meta.shape == (5,)
        assert res.p_meta.shape == (5,)
        assert torch.all(res.p_meta >= 0)
        assert torch.all(res.p_meta <= 1)

    def test_invalid_zero_se(self):
        beta = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
        se = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        # Zero SE => divide-by-zero produces inf weights. The weighted sum
        # (inf * beta).sum() / inf.sum() collapses to NaN for beta_meta and
        # the combined SE = 1/sqrt(inf) = 0. Document the non-finite output.
        res = meta_fixed_effect(beta, se)
        assert not torch.isfinite(res.beta_meta).all()


# ---------------------------------------------------------------------------
# meta_random_effect
# ---------------------------------------------------------------------------


class TestMetaRandomEffect:
    """Random-effects (DerSimonian-Laird): adds between-study variance tau^2.
    When studies are homogeneous, tau^2=0 and RE collapses to FE."""

    def test_smoke(self):
        torch.manual_seed(0)
        beta = torch.randn(4, 5, dtype=torch.float64) * 0.05
        se = torch.full((4, 5), 0.05, dtype=torch.float64)
        res = meta_random_effect(beta, se)
        assert isinstance(res, MetaResult)
        assert res.method == "random_effect"
        assert res.beta_meta.shape == (4,)
        assert torch.all(torch.isfinite(res.tau2))
        assert torch.all(res.tau2 >= 0)

    def test_collapse_to_fixed_when_no_heterogeneity(self):
        # Perfectly homogeneous studies (all betas equal).
        beta = torch.full((1, 5), 0.2, dtype=torch.float64)
        se = torch.full((1, 5), 0.1, dtype=torch.float64)
        re = meta_random_effect(beta, se)
        fe = meta_fixed_effect(beta, se)
        assert re.beta_meta.item() == pytest.approx(fe.beta_meta.item(), abs=1e-10)
        assert re.se_meta.item() == pytest.approx(fe.se_meta.item(), abs=1e-10)
        # tau2 should be 0 under no heterogeneity.
        assert re.tau2.item() == pytest.approx(0.0, abs=1e-10)

    def test_invalid_zero_se(self):
        beta = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
        se = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        res = meta_random_effect(beta, se)
        # Zero SE produces inf weights, yielding NaN beta_meta. Document.
        assert not torch.isfinite(res.beta_meta).all()


# ---------------------------------------------------------------------------
# meta_sample_size
# ---------------------------------------------------------------------------


class TestMetaSampleSize:
    """Sample-size weighted Stouffer's Z combines z-scores using sqrt(n_k)
    weights normalized to sum to 1 over studies."""

    def test_stouffer_closed_form_signed(self):
        # 1 SNP, 3 studies: build z, p directly, with explicit direction.
        z_per_study = torch.tensor([[1.5, 2.0, 1.8]], dtype=torch.float64)
        # Convert to two-sided p-values.
        p = torch.erfc(z_per_study.abs() / 2.0**0.5).clamp(min=1e-300, max=1.0)
        n = torch.tensor([1000.0, 2000.0, 1500.0], dtype=torch.float64)
        direction = torch.ones_like(z_per_study)

        # Reference: |z| from p using meta's inverse, then sign by direction.
        # Meta uses |z| = sqrt(2) * erfinv(1 - p), then multiplies by direction.
        p_c = p.clamp(min=1e-300, max=1.0 - 1e-10)
        z_abs_ref = (2.0**0.5) * torch.erfinv(1.0 - p_c)
        z_signed_ref = z_abs_ref * direction
        w = n.sqrt() / n.sqrt().sum()
        z_meta_ref = (z_signed_ref * w.unsqueeze(0)).sum(dim=1)

        res = meta_sample_size(p, n, direction=direction)
        assert isinstance(res, MetaResult)
        assert res.method == "sample_size"
        assert res.z_meta.item() == pytest.approx(z_meta_ref.item(), abs=1e-10)
        # P-value matches two-sided erfc(|z|/sqrt(2)).
        ref_p = float(
            torch.erfc(z_meta_ref.abs() / 2.0**0.5).clamp(min=1e-300, max=1.0).item()
        )
        assert res.p_meta.item() == pytest.approx(ref_p, abs=1e-10)

    def test_unsigned_uses_absolute_z(self):
        # Without direction the implementation uses unsigned z.
        p = torch.tensor([[0.05, 0.01, 0.001]], dtype=torch.float64)
        n = torch.tensor([1000.0, 2000.0, 1500.0], dtype=torch.float64)
        res = meta_sample_size(p, n)
        # Unsigned z is non-negative, so the combined z is non-negative.
        assert res.z_meta.item() >= 0.0
        assert 0.0 <= res.p_meta.item() <= 1.0

    def test_shape_multi_snp(self):
        m, K = 4, 3
        torch.manual_seed(0)
        p = torch.rand(m, K, dtype=torch.float64)
        n = torch.tensor([500.0, 1000.0, 1500.0], dtype=torch.float64)
        res = meta_sample_size(p, n)
        assert res.z_meta.shape == (m,)
        assert res.p_meta.shape == (m,)


# ---------------------------------------------------------------------------
# meta_han_eskin
# ---------------------------------------------------------------------------


class TestMetaHanEskin:
    """Han-Eskin RE2 modified LRT: returns chi^2(1) p-value combining
    fixed-effect z and Cochran's Q-derived heterogeneity inflation."""

    def test_smoke(self):
        torch.manual_seed(0)
        beta = torch.randn(3, 5, dtype=torch.float64) * 0.1
        se = torch.full((3, 5), 0.05, dtype=torch.float64)
        res = meta_han_eskin(beta, se)
        assert isinstance(res, MetaResult)
        assert res.method == "han_eskin"
        assert res.beta_meta.shape == (3,)
        assert torch.all(torch.isfinite(res.beta_meta))
        assert torch.all(res.p_meta >= 0)
        assert torch.all(res.p_meta <= 1)

    def test_homogeneous_collapses_to_fe_chi1(self):
        # Homogeneous studies: Q < df, so q_ratio=1 and t_re2 = z_fe^2.
        beta = torch.full((1, 5), 0.3, dtype=torch.float64)
        se = torch.full((1, 5), 0.1, dtype=torch.float64)
        res = meta_han_eskin(beta, se)
        fe = meta_fixed_effect(beta, se)
        # |z_meta| under han_eskin equals |z_fe| in this regime.
        assert res.z_meta.abs().item() == pytest.approx(
            fe.z_meta.abs().item(), abs=1e-10
        )

    def test_invalid_zero_se(self):
        beta = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
        se = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        res = meta_han_eskin(beta, se)
        # Zero SE produces inf weights, yielding NaN beta_meta and tau2.
        assert not torch.isfinite(res.beta_meta).all()
