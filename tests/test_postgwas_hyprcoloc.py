"""Tests for multi-trait colocalization (hyprcoloc) and classical coloc.

Phase 42b: Foley et al. (2021) hyprcoloc and the Giambartolomei (2014)
pairwise coloc decomposition.

Strategy: construct small synthetic regions where the colocalization
structure is known by design (all traits share one causal SNP, or distinct
causal SNPs, or no signal) and check that the posteriors agree with the
ground truth. We avoid testing the exact numerical values against the R
``coloc`` / ``hyprcoloc`` implementations — the priors are the same, but
small differences in ABF / log-sum-exp conventions mean we only expect the
*ordering* of posteriors to match.
"""

from __future__ import annotations

import math

import torch

from torchgwas.postgwas._hyprcoloc import (
    ColocPairwiseResult,
    HyprcolocResult,
    coloc_pairwise,
    hyprcoloc,
)
from torchgwas.postgwas._sumstats import SumStats


def _make_sumstats(beta: torch.Tensor, se: torch.Tensor) -> SumStats:
    m = beta.shape[0]
    return SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=beta,
        se=se,
        p=torch.ones(m, dtype=torch.float64),
        n=torch.full((m,), 1000.0, dtype=torch.float64),
    )


def _region_with_shared_signal(m: int, causal_idx: int, k_traits: int,
                               effect: float, seed: int):
    """Build k_traits aligned sumstats where all traits have the same causal
    SNP (``causal_idx``) with effect size ``effect`` and small noise
    elsewhere."""
    torch.manual_seed(seed)
    sslist = []
    for _ in range(k_traits):
        beta = torch.randn(m, dtype=torch.float64) * 0.01
        beta[causal_idx] += effect
        se = torch.full((m,), 0.02, dtype=torch.float64)
        sslist.append(_make_sumstats(beta, se))
    return sslist


def _region_with_distinct_signals(m: int, causal_idxs: list[int],
                                  effect: float, seed: int):
    """Each trait gets its own distinct causal SNP."""
    torch.manual_seed(seed)
    sslist = []
    for j in causal_idxs:
        beta = torch.randn(m, dtype=torch.float64) * 0.01
        beta[j] += effect
        se = torch.full((m,), 0.02, dtype=torch.float64)
        sslist.append(_make_sumstats(beta, se))
    return sslist


# ---------------------------------------------------------------------------
# hyprcoloc
# ---------------------------------------------------------------------------


def test_hyprcoloc_two_traits_shared_signal_pp_all_high():
    """Two traits sharing a strong causal SNP should give PP(all) near 1."""
    ss = _region_with_shared_signal(m=50, causal_idx=17, k_traits=2,
                                    effect=0.5, seed=1)
    res = hyprcoloc(ss, prior_1=1e-4, prior_2=0.98)
    assert isinstance(res, HyprcolocResult)
    assert res.pp_all_colocalize > 0.9
    assert res.best_cluster == (0, 1)
    assert res.candidate_snp == 17


def test_hyprcoloc_three_traits_shared_signal():
    """All three traits sharing a causal SNP should give best_cluster = (0,1,2)."""
    ss = _region_with_shared_signal(m=80, causal_idx=40, k_traits=3,
                                    effect=0.6, seed=2)
    res = hyprcoloc(ss, prior_1=1e-4, prior_2=0.98)
    assert res.best_cluster == (0, 1, 2)
    assert res.pp_all_colocalize > 0.8
    assert res.candidate_snp == 40


def test_hyprcoloc_null_region_low_all_posterior():
    """No signal in any trait: PP(all colocalize) should be negligible and
    the null hypothesis should dominate."""
    torch.manual_seed(5)
    m = 60
    ss = [
        _make_sumstats(
            torch.randn(m, dtype=torch.float64) * 0.01,
            torch.full((m,), 0.02, dtype=torch.float64),
        )
        for _ in range(3)
    ]
    res = hyprcoloc(ss)
    assert res.pp_all_colocalize < 0.1
    assert res.pp_null > 0.5


def test_hyprcoloc_distinct_causal_variants():
    """If each trait has its own distinct causal SNP in a large region, the
    best cluster should *not* be "all three colocalize" (because they don't
    share). We invoke with a neutral conditional sharing prior so the
    data-driven evidence dominates the prior.

    Updated 2026-05-15 after the F3 #1 patch: the default ``prior_2 = 0.98``
    encodes Foley 2021's strong prior belief in sharing (`c = 0.98` means
    "given a cluster exists, 98% chance each additional trait joins").
    Under that default, even weakly-shared data can push PP(all-traits) >
    0.5 because the prior favors larger clusters. This test exercises
    the *data-driven* discrimination by setting ``prior_2 = 0.5`` (the
    neutral midpoint where the prior is uninformative about cluster
    size), and asserts that the distinct-signal architecture is
    correctly identified.
    """
    ss = _region_with_distinct_signals(
        m=100, causal_idxs=[5, 50, 90], effect=0.5, seed=9
    )
    res = hyprcoloc(ss, prior_2=0.5)
    # PP(all three colocalize) should not dominate under a neutral
    # conditional prior.
    assert res.pp_all_colocalize < 0.5


def test_hyprcoloc_foley_2021_conditional_prior_post_f3_patch():
    """Regression: F3 #1 (2026-05-15) — under a 3-trait shared-causal
    architecture with strong signal, R hyprcoloc reports cluster
    membership [T1, T2, T3] (size = K = 3). Pre-patch TG returned
    [T1, T3] (size = 2) because the product prior
    `prior_1^|S| * prior_2^(|S|-1) * (1 - prior_1)^(K - |S|)` imposed
    a ~1e-4 penalty per additional trait. The F3 #1 patch implements
    Foley 2021 Eq. 2 conditional prior
    `prior_1 * prior_2^(|S|-1) * (1 - prior_2)^(K - |S|)` so the
    prior ratio between |S| = K and |S| = K - 1 is ~prior_2 / (1 - prior_2)
    when prior_2 is high. Validated against R hyprcoloc at commit
    0348bbd via validation/external/hyprcoloc/ (cluster + candidate
    SNP agreement now exact; regional_pp Pearson r ≈ 1.0).

    This test gates the patch: on a strong 3-trait shared fixture
    with default prior_2 = 0.98, the best cluster must include ALL
    three traits.
    """
    # Strong shared signal at index 50 on all 3 traits.
    ss = _region_with_shared_signal(m=100, causal_idx=50, k_traits=3,
                                    effect=0.6, seed=42)
    res = hyprcoloc(ss)
    assert res.best_cluster == (0, 1, 2), (
        f"best_cluster = {res.best_cluster!r}; F3 #1 patch (Foley 2021 "
        "conditional prior) likely regressed. Pre-patch best_cluster "
        "was (0, 2); see docs/validation_findings.md."
    )
    assert res.best_cluster_posterior >= 0.95, (
        f"best_cluster_posterior = {res.best_cluster_posterior:.4f}; "
        "expected >= 0.95 under the corrected conditional prior."
    )


def test_hyprcoloc_identical_traits_trivially_colocalize():
    """Two byte-identical SumStats must collapse to PP(all) ≈ 1."""
    torch.manual_seed(13)
    m = 40
    beta = torch.randn(m, dtype=torch.float64) * 0.05
    beta[20] = 0.7
    se = torch.full((m,), 0.03, dtype=torch.float64)
    ss = _make_sumstats(beta, se)
    res = hyprcoloc([ss, ss])
    assert res.pp_all_colocalize > 0.95
    assert res.candidate_snp == 20


def test_hyprcoloc_rejects_mismatched_snp_counts():
    ss1 = _make_sumstats(torch.zeros(30, dtype=torch.float64),
                         torch.ones(30, dtype=torch.float64))
    ss2 = _make_sumstats(torch.zeros(25, dtype=torch.float64),
                         torch.ones(25, dtype=torch.float64))
    try:
        hyprcoloc([ss1, ss2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched SNP counts")


def test_hyprcoloc_requires_at_least_two_traits():
    ss = _make_sumstats(torch.zeros(10, dtype=torch.float64),
                        torch.ones(10, dtype=torch.float64))
    try:
        hyprcoloc([ss])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for K < 2")


def test_hyprcoloc_subset_posteriors_sum_to_under_one():
    ss = _region_with_shared_signal(m=30, causal_idx=10, k_traits=3,
                                    effect=0.4, seed=21)
    res = hyprcoloc(ss)
    total = res.pp_null + sum(res.subset_posteriors.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Two-trait coloc (Giambartolomei 2014)
# ---------------------------------------------------------------------------


def test_coloc_pairwise_shared_signal_pph4_dominates():
    ss = _region_with_shared_signal(m=50, causal_idx=25, k_traits=2,
                                    effect=0.5, seed=31)
    res = coloc_pairwise(ss[0], ss[1])
    assert isinstance(res, ColocPairwiseResult)
    total = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
    assert math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert res.pp_h4 > 0.8
    assert res.candidate_snp == 25


def test_coloc_pairwise_distinct_signals_pph3_dominates():
    """Distinct causal variants should push posterior mass to H3."""
    ss = _region_with_distinct_signals(m=60, causal_idxs=[10, 45],
                                       effect=0.5, seed=33)
    res = coloc_pairwise(ss[0], ss[1])
    # H3 (distinct causal) should beat H4 (shared).
    assert res.pp_h3 > res.pp_h4


def test_coloc_pairwise_distinct_signals_pph3_near_unity_post_f3_patch():
    """Regression: F3 #2 (2026-05-15) — under a strong distinct-causal
    architecture, R coloc::coloc.abf reports PP.H3 ≈ 0.997. Pre-patch
    TG returned PP.H3 ≈ 0.854 with the missing 14% leaked into PP.H1
    because (a) H3 used the full outer product instead of
    `outer - diag` and (b) the spurious `-log m` factor on the
    per-hypothesis weights inverted the Bayes-factor balance between
    H3 and H1. After the F3 #2 patch the corrected closed form
    matches R to FP precision on the validation harness fixture
    (validation/external/coloc/, scenario='distinct': max |Δ PP| = 2e-5).

    This test gates the closed form: PP.H3 must be ≥ 0.95 on a
    well-separated distinct-signal architecture, and the H1/H2
    mass-leak must remain below 0.05 combined. Pre-patch this test
    fails (H3 ≈ 0.85, H1 ≈ 0.14); post-patch it passes.
    """
    ss = _region_with_distinct_signals(m=60, causal_idxs=[10, 45],
                                       effect=0.5, seed=33)
    res = coloc_pairwise(ss[0], ss[1])
    assert res.pp_h3 >= 0.95, (
        f"PP.H3 = {res.pp_h3:.4f}; F3 #2 patch (H3 outer-minus-diagonal + drop "
        "log_m factor) likely regressed. Pre-patch value was ~0.85; "
        "see docs/validation_findings.md."
    )
    assert (res.pp_h1 + res.pp_h2) < 0.05, (
        f"PP.H1 + PP.H2 = {res.pp_h1 + res.pp_h2:.4f}; the F3 #2 patch "
        "should keep H1 + H2 mass below 0.05 under a well-separated "
        "distinct-signal architecture."
    )


def test_coloc_pairwise_null_signal_pph0_dominates():
    torch.manual_seed(37)
    m = 60
    beta1 = torch.randn(m, dtype=torch.float64) * 0.005
    beta2 = torch.randn(m, dtype=torch.float64) * 0.005
    se = torch.full((m,), 0.02, dtype=torch.float64)
    res = coloc_pairwise(_make_sumstats(beta1, se), _make_sumstats(beta2, se))
    assert res.pp_h0 > 0.5
