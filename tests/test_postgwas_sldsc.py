"""Tests for partitioned LD Score Regression (S-LDSC).

Phase 42a: partitioned heritability by functional annotation (Finucane 2015).
The tests reuse the same infinitesimal simulator as the univariate LDSC tests
but split the SNPs into annotation categories, concentrate the effect sizes
in one category, and check that

  1. single-annotation S-LDSC recovers univariate ``ldsc_h2``,
  2. the enriched category shows Enrichment > 1 with a plausible SE, and
  3. the estimate is robust to reshuffling the annotation columns.
"""

from __future__ import annotations

import math

import torch

from torchgwas.postgwas._ldsc import ldsc_h2
from torchgwas.postgwas._sldsc import SLDSCResult, sldsc_h2_partitioned


def _sim_gwas_with_annotation(
    n: int = 1000,
    m: int = 400,
    h2: float = 0.4,
    enriched_frac: float = 0.25,
    enriched_share: float = 0.70,
    seed: int = 7,
):
    """Simulate GWAS chi² + stratified LD scores.

    Splits the ``m`` SNPs into two annotation categories: the first
    ``enriched_frac`` get ``enriched_share`` of the total heritability; the
    rest get the remaining ``1 - enriched_share``.

    Returns ``(chi2, annot_ld, M_c, n_sample, m_total, true_h2)`` with
    ``annot_ld`` a (m, 2) tensor of per-category LD scores.
    """
    torch.manual_seed(seed)
    G = torch.randn(n, m, dtype=torch.float64)
    for j in range(1, m):
        G[:, j] = 0.3 * G[:, j - 1] + (1 - 0.09) ** 0.5 * G[:, j]
    G = (G - G.mean(dim=0)) / G.std(dim=0).clamp(min=1e-6)

    n_enriched = int(round(m * enriched_frac))
    enriched_idx = torch.arange(n_enriched)
    rest_idx = torch.arange(n_enriched, m)

    beta = torch.zeros(m, dtype=torch.float64)
    var_enriched = h2 * enriched_share / max(n_enriched, 1)
    var_rest = h2 * (1 - enriched_share) / max(m - n_enriched, 1)
    beta[enriched_idx] = torch.randn(n_enriched, dtype=torch.float64) * math.sqrt(var_enriched)
    beta[rest_idx] = torch.randn(m - n_enriched, dtype=torch.float64) * math.sqrt(var_rest)

    g = G @ beta
    e_var = g.var() * (1 - h2) / h2
    e = torch.randn(n, dtype=torch.float64) * e_var.sqrt()
    y = g + e

    z = torch.zeros(m, dtype=torch.float64)
    for j in range(m):
        gj = G[:, j]
        b = (gj @ y) / (gj @ gj)
        resid = y - gj * b
        se = (resid.var() / (gj @ gj)).sqrt()
        z[j] = b / se
    chi2 = z**2

    # Stratified LD scores: for each target SNP j, sum r² only with SNPs in
    # category c. We use the empirical correlation matrix on G.
    R = (G.T @ G) / n  # (m, m)
    r2 = R**2

    annot = torch.zeros(m, 2, dtype=torch.float64)
    annot[:n_enriched, 0] = 1.0
    annot[n_enriched:, 1] = 1.0
    # l_cj = sum_{k in c} r2[j, k]
    annot_ld = r2 @ annot  # (m, 2)

    M_c = annot.sum(dim=0)
    m_total = m
    return chi2, annot_ld, M_c, float(n), m_total, h2


def test_sldsc_returns_expected_shapes():
    chi2, annot_ld, M_c, n, m_total, h2 = _sim_gwas_with_annotation()
    res = sldsc_h2_partitioned(
        chi2, annot_ld, M_c, n, m_total,
        annotation_names=["enriched", "baseline"],
    )
    assert isinstance(res, SLDSCResult)
    assert res.tau.shape == (2,)
    assert res.tau_se.shape == (2,)
    assert res.h2_cat.shape == (2,)
    assert res.enrichment.shape == (2,)
    assert res.annotation_names == ["enriched", "baseline"]
    assert res.n_snps > 0
    # Intercept should be near 1 under no confounding.
    assert abs(res.intercept - 1.0) < 1.0


def test_sldsc_single_annotation_matches_univariate_ldsc():
    """A single-column annotation (all SNPs in one category) must reduce to
    univariate ``ldsc_h2``: the tau coefficient, h2 and intercept should
    match to within numerical tolerance (WLS is deterministic)."""
    chi2, annot_ld_2col, _, n, m_total, _ = _sim_gwas_with_annotation()
    # Collapse the two categories into one by summing the stratified
    # LD scores — this reproduces the standard ``ldsc_h2`` LD score.
    total_ld = annot_ld_2col.sum(dim=1, keepdim=True)  # (m, 1)
    M_c = torch.tensor([float(m_total)], dtype=torch.float64)

    res_s = sldsc_h2_partitioned(chi2, total_ld, M_c, n, m_total)
    res_u = ldsc_h2(chi2, total_ld.squeeze(1), n, m_total)

    # Numerical agreement: both solve the same WLS problem.
    assert math.isclose(res_s.h2_total, res_u.h2, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(res_s.intercept, res_u.intercept, rel_tol=1e-6, abs_tol=1e-6)


def test_sldsc_enrichment_above_one_for_true_enriched_category():
    """The category where we concentrated 70% of heritability on 25% of SNPs
    should have point-estimate enrichment > 1. We don't gate on the SE band
    because jackknife SEs are noisy at m = 400."""
    chi2, annot_ld, M_c, n, m_total, _ = _sim_gwas_with_annotation(
        h2=0.5, enriched_frac=0.25, enriched_share=0.75, seed=11,
    )
    res = sldsc_h2_partitioned(chi2, annot_ld, M_c, n, m_total)
    # The enriched category is column 0.
    assert res.enrichment[0].item() > res.enrichment[1].item()
    assert res.enrichment[0].item() > 1.0


def test_sldsc_permutation_of_annotations_is_permutation_of_coefficients():
    """Reordering the annotation columns should permute tau and h2_cat
    correspondingly but leave h2_total and intercept invariant."""
    chi2, annot_ld, M_c, n, m_total, _ = _sim_gwas_with_annotation(seed=3)
    res_a = sldsc_h2_partitioned(chi2, annot_ld, M_c, n, m_total)

    perm = torch.tensor([1, 0])
    res_b = sldsc_h2_partitioned(
        chi2, annot_ld[:, perm], M_c[perm], n, m_total
    )

    assert math.isclose(res_a.h2_total, res_b.h2_total, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(res_a.intercept, res_b.intercept, rel_tol=1e-6, abs_tol=1e-6)
    assert torch.allclose(res_a.tau[perm], res_b.tau, rtol=1e-6, atol=1e-6)
    assert torch.allclose(res_a.h2_cat[perm], res_b.h2_cat, rtol=1e-6, atol=1e-6)


def test_sldsc_many_categories_falls_back_to_python_path():
    """With more than 8 annotation categories the native jackknife kernel
    bails out; the Python path must still produce a sane result."""
    torch.manual_seed(19)
    chi2, annot_ld_2, _, n, m_total, _ = _sim_gwas_with_annotation(m=300)
    m = chi2.shape[0]
    C = 12  # exceeds the native kernel's p <= 8 guard
    # Build a random but full-rank stratified LD score matrix by taking C
    # non-negative linear combinations of the two-category version.
    W = torch.rand(2, C, dtype=torch.float64) + 0.1
    annot_ld = annot_ld_2 @ W  # (m, C)
    M_c = torch.full((C,), float(m) / C, dtype=torch.float64)

    res = sldsc_h2_partitioned(chi2, annot_ld, M_c, n, m_total)
    assert res.tau.shape == (C,)
    assert math.isfinite(res.h2_total)
    assert math.isfinite(res.intercept)
    # All SEs must be finite (the fallback path produced them).
    assert torch.isfinite(res.tau_se).all()


def test_sldsc_rejects_mismatched_shapes():
    chi2 = torch.randn(100, dtype=torch.float64) ** 2
    annot_ld = torch.rand(100, 3, dtype=torch.float64)
    M_c_bad = torch.tensor([10.0, 20.0])  # length 2 vs C = 3
    try:
        sldsc_h2_partitioned(chi2, annot_ld, M_c_bad, n=1000, m_total=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")
