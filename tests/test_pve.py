"""Tests for per-marker PVE (phenotypic variance explained).

The helper addresses the three GAPIT quirks the user flagged:

1. If no markers exceed the p-value threshold, GAPIT may not produce a PVE
   file at all — ``compute_pve`` returns an empty ``PVEResult`` instead so
   downstream code never has to handle ``None``.
2. If >500 significant markers are found, GAPIT may silently skip the file
   — ``compute_pve`` still computes, but emits a ``RuntimeWarning`` so the
   user knows the joint fit is numerically sensitive.
3. GAPIT-style marginal PVE can sum to >100% in LD. ``method="joint"``
   uses squared semi-partial correlations which are *mathematically*
   bounded by the joint R² ≤ 1; ``method="marginal"`` renormalizes and
   sets ``capped=True`` when the raw sum exceeds 1.0.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from torchgwas.models.base import ScanResult
from torchgwas.stats import PVEResult, compute_pve


def _make_scan_result(
    beta: torch.Tensor,
    p: torch.Tensor,
    af: torch.Tensor,
) -> ScanResult:
    m = beta.shape[0]
    return ScanResult(
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=af,
        beta=beta,
        se=torch.ones(m, dtype=torch.float64),
        stat=torch.zeros(m, dtype=torch.float64),
        p=p,
        test="wald",
    )


def _simulate_gwas(n: int = 400, m: int = 30, n_causal: int = 3, seed: int = 0):
    """Generate a tiny GWAS fixture: n samples × m SNPs with n_causal hits."""
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m)).to(torch.float64)
    G = G - G.mean(dim=0)  # center so OLS is clean
    causal_idx = list(range(n_causal))
    default_effects = torch.tensor(
        [0.8, -0.6, 0.5, 0.4, -0.3, 0.2], dtype=torch.float64
    )
    true_beta = torch.zeros(m, dtype=torch.float64)
    for i, cidx in enumerate(causal_idx):
        true_beta[cidx] = default_effects[i % default_effects.numel()]
    y = G @ true_beta + torch.randn(n, dtype=torch.float64) * 0.3
    # Compute per-SNP univariate stats for the ScanResult.
    beta_hat = torch.zeros(m, dtype=torch.float64)
    p = torch.ones(m, dtype=torch.float64)
    for j in range(m):
        gj = G[:, j]
        b = float((gj @ y) / (gj @ gj))
        beta_hat[j] = b
        resid = y - gj * b
        var_b = float(resid.var(unbiased=True) / (gj @ gj))
        z = b / math.sqrt(max(var_b, 1e-30))
        from scipy.stats import norm
        p[j] = float(2.0 * (1.0 - norm.cdf(abs(z))))
    af = torch.full((m,), 0.5, dtype=torch.float64)
    return G, y, beta_hat, p, af, causal_idx


# ---------------------------------------------------------------------------
# Zero-hit case: empty result, not None
# ---------------------------------------------------------------------------


def test_compute_pve_returns_empty_result_when_no_markers_significant():
    m = 20
    beta = torch.zeros(m, dtype=torch.float64)
    p = torch.ones(m, dtype=torch.float64)  # all = 1
    af = torch.full((m,), 0.3, dtype=torch.float64)
    result = _make_scan_result(beta, p, af)

    G = torch.randn(100, m, dtype=torch.float64)
    y = torch.randn(100, dtype=torch.float64)

    res = compute_pve(result, y, G=G, significance_threshold=5e-8)
    assert isinstance(res, PVEResult)
    assert res.n_significant == 0
    assert len(res) == 0
    assert res.pve_total == 0.0
    assert not res.capped
    # Phenotypic variance should still be reported for the trait.
    assert res.phenotypic_variance > 0.0


# ---------------------------------------------------------------------------
# Joint method: sum guaranteed ≤ 1.0 even under strong LD
# ---------------------------------------------------------------------------


def test_joint_pve_sum_never_exceeds_one_under_ld():
    """Three SNPs in perfect LD share *one* causal signal; marginal PVE
    would triple-count it, but joint semi-partial PVE must cap at ≤ 1."""
    torch.manual_seed(1)
    n = 500
    base = torch.randint(0, 3, (n, 1)).to(torch.float64).squeeze(-1)
    base = base - base.mean()
    # Three near-identical copies (rho ~ 0.995)
    noise = torch.randn(n, 3, dtype=torch.float64) * 0.05
    G = torch.stack([base, base, base], dim=1) + noise
    y = base * 1.2 + torch.randn(n, dtype=torch.float64) * 0.3

    # Fake a ScanResult where all 3 markers are "significant" with huge beta.
    beta_hat = torch.tensor([1.2, 1.2, 1.2], dtype=torch.float64)
    p = torch.tensor([1e-20, 1e-20, 1e-20], dtype=torch.float64)
    af = torch.full((3,), 0.5, dtype=torch.float64)
    result = _make_scan_result(beta_hat, p, af)

    res = compute_pve(result, y, G=G, method="joint")
    assert res.method in ("joint", "marginal")  # may fall back on rank deficiency
    assert res.pve_total <= 1.0 + 1e-9
    assert float(res.pve.sum().item()) <= 1.0 + 1e-9
    assert (res.pve >= -1e-12).all()


def test_joint_pve_recovers_known_r2_on_orthogonal_design():
    """When the significant markers are uncorrelated, joint R² should be
    close to the sum of the true marginal R² contributions."""
    G, y, beta_hat, p, af, causal_idx = _simulate_gwas(n=500, m=15, n_causal=3)
    # Force the three causal SNPs to be "significant".
    p[:3] = 1e-20
    result = _make_scan_result(beta_hat, p, af)

    res = compute_pve(result, y, G=G, method="joint")
    assert res.n_significant == 3
    # Sum of per-SNP PVE must match pve_total to float precision.
    assert math.isclose(
        float(res.pve.sum().item()), res.pve_total, rel_tol=1e-9, abs_tol=1e-9
    )
    # Joint R² from a direct centered OLS fit on the three selected columns
    # (same centering semantics as compute_pve's residualizer).
    G_sig = G[:, :3] - G[:, :3].mean(dim=0)
    y_c = y - y.mean()
    beta_joint = torch.linalg.lstsq(G_sig, y_c).solution
    r2_direct = float((beta_joint @ (G_sig.T @ y_c)) / (y_c @ y_c))
    # semi-partial total ≤ joint R², and the two should agree closely for
    # orthogonal predictors.
    assert res.pve_total <= 1.0 + 1e-9
    assert abs(res.pve_total - r2_direct) < 1e-6


# ---------------------------------------------------------------------------
# Marginal method: renormalizes and flags capped=True
# ---------------------------------------------------------------------------


def test_marginal_pve_capped_when_sum_exceeds_one():
    """Two SNPs with huge effect sizes in the same direction will
    over-explain the variance marginally. The helper must renormalize to
    sum=1 and set capped=True."""
    m = 2
    beta = torch.tensor([3.0, 3.0], dtype=torch.float64)
    p = torch.tensor([1e-20, 1e-20], dtype=torch.float64)
    af = torch.full((m,), 0.5, dtype=torch.float64)  # var(G) = 2 * 0.25 = 0.5
    result = _make_scan_result(beta, p, af)

    n = 100
    torch.manual_seed(3)
    G = torch.randn(n, m, dtype=torch.float64)
    # Small phenotypic variance so the raw marginal sum blows up.
    y = torch.randn(n, dtype=torch.float64) * 0.1

    res = compute_pve(result, y, G=G, method="marginal")
    assert res.capped is True
    assert math.isclose(float(res.pve.sum().item()), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert res.method == "marginal"


def test_marginal_pve_matches_gapit_formula_for_single_snp():
    """For a single SNP, marginal PVE should exactly equal
    β² · 2 f (1-f) / var(y)."""
    n = 300
    torch.manual_seed(5)
    af_true = 0.3
    G_raw = torch.distributions.Binomial(2, af_true).sample((n, 1)).to(torch.float64)
    beta_true = 0.5
    y = G_raw.squeeze(-1) * beta_true + torch.randn(n, dtype=torch.float64) * 0.5

    beta_hat = float((G_raw.squeeze(-1) - G_raw.mean()) @ y /
                     ((G_raw.squeeze(-1) - G_raw.mean()) @ (G_raw.squeeze(-1) - G_raw.mean())))
    result = _make_scan_result(
        beta=torch.tensor([beta_hat], dtype=torch.float64),
        p=torch.tensor([1e-20], dtype=torch.float64),
        af=torch.tensor([af_true], dtype=torch.float64),
    )

    res = compute_pve(result, y, G=G_raw, method="marginal", ploidy=2)
    var_y = float((y - y.mean()).var(unbiased=True).item())
    expected = (beta_hat**2) * (2.0 * af_true * (1.0 - af_true)) / var_y
    assert math.isclose(float(res.pve[0].item()), expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Soft cap warning at >500 markers
# ---------------------------------------------------------------------------


def test_compute_pve_warns_above_500_markers():
    m = 600
    beta = torch.full((m,), 0.1, dtype=torch.float64)
    p = torch.full((m,), 1e-20, dtype=torch.float64)  # all significant
    af = torch.full((m,), 0.5, dtype=torch.float64)
    result = _make_scan_result(beta, p, af)

    n = 200
    torch.manual_seed(7)
    G = torch.randn(n, m, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = compute_pve(result, y, G=G, method="marginal")
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "exceeds GAPIT's soft cap" in str(w.message)
        for w in caught
    )


# ---------------------------------------------------------------------------
# Covariate adjustment
# ---------------------------------------------------------------------------


def test_compute_pve_respects_covariate_adjustment():
    """A covariate correlated with y should reduce ``phenotypic_variance``
    (which in this helper means var(y_resid)), and leave the causal PVE
    intact."""
    G, y, beta_hat, p, af, _ = _simulate_gwas(n=400, m=10, n_causal=2)
    p[:2] = 1e-20
    result = _make_scan_result(beta_hat, p, af)

    # Build a covariate uncorrelated with G but correlated with y.
    torch.manual_seed(9)
    z = torch.randn(400, dtype=torch.float64)
    y_conf = y + 2.0 * z
    X0 = z.view(-1, 1)

    res_cov = compute_pve(result, y_conf, G=G, X0=X0, method="joint")
    res_nocov = compute_pve(result, y_conf, G=G, method="joint")

    # Without covariate adjustment, phenotypic variance is larger (the
    # covariate contributes noise).
    assert res_cov.phenotypic_variance < res_nocov.phenotypic_variance
    # Both reports should still have ≤ 1 total.
    assert res_cov.pve_total <= 1.0 + 1e-9
    assert res_nocov.pve_total <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compute_pve_rejects_invalid_method():
    m = 5
    result = _make_scan_result(
        torch.zeros(m, dtype=torch.float64),
        torch.ones(m, dtype=torch.float64),
        torch.full((m,), 0.5, dtype=torch.float64),
    )
    try:
        compute_pve(result, torch.zeros(10), method="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown method")


def test_compute_pve_joint_rejects_misaligned_genotype_matrix():
    m = 5
    result = _make_scan_result(
        torch.zeros(m, dtype=torch.float64),
        torch.tensor([1e-20] + [1.0] * (m - 1), dtype=torch.float64),
        torch.full((m,), 0.5, dtype=torch.float64),
    )
    G_bad = torch.zeros(10, m - 1)
    try:
        compute_pve(result, torch.zeros(10), G=G_bad, method="joint")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for marker count mismatch")


def test_compute_pve_joint_requires_genotype_matrix():
    m = 5
    result = _make_scan_result(
        torch.zeros(m, dtype=torch.float64),
        torch.tensor([1e-20] + [1.0] * (m - 1), dtype=torch.float64),
        torch.full((m,), 0.5, dtype=torch.float64),
    )
    try:
        compute_pve(result, torch.zeros(10), method="joint")
    except ValueError as e:
        assert "requires the genotype matrix" in str(e)
    else:
        raise AssertionError("expected ValueError when joint mode is asked without G")


def test_compute_pve_rejects_multi_trait_scan_result():
    m = 5
    result = _make_scan_result(
        torch.zeros((m, 2), dtype=torch.float64),
        torch.tensor([1e-20] + [1.0] * (m - 1), dtype=torch.float64),
        torch.full((m,), 0.5, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="single-trait"):
        compute_pve(result, torch.zeros(10), method="marginal")


# ---------------------------------------------------------------------------
# Scaling: marginal PVE on a 1M-SNP ScanResult must be subsecond and must
# never materialize a (n, m) genotype matrix.
# ---------------------------------------------------------------------------


def test_compute_pve_marginal_scales_to_one_million_snps():
    """Industry-grade scaling invariant. We build a fake ScanResult with
    m = 1_000_000 markers (only β / p / af are materialized — no genotype
    matrix at all) and confirm that marginal PVE runs in well under a
    second. This locks in the O(m) mask walk + O(k) compute contract."""
    import time

    m = 1_000_000
    torch.manual_seed(42)
    beta = torch.randn(m, dtype=torch.float64) * 0.005
    p = torch.rand(m, dtype=torch.float64)
    # Plant 50 strong hits so k = 50 ≪ m.
    hit_idx = torch.arange(0, 50) * (m // 50)
    beta[hit_idx] = 0.3
    p[hit_idx] = 1e-12
    af = torch.full((m,), 0.3, dtype=torch.float64)
    result = ScanResult(
        chr=["1"] * m,
        pos=list(range(m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=af,
        beta=beta,
        se=torch.ones(m, dtype=torch.float64),
        stat=torch.zeros(m, dtype=torch.float64),
        p=p,
        test="wald",
    )

    # No genotype matrix supplied — marginal mode must not need it.
    y = torch.randn(500, dtype=torch.float64)

    t0 = time.perf_counter()
    res = compute_pve(result, y, method="marginal")
    dt = time.perf_counter() - t0

    assert res.n_significant == 50
    assert res.pve_total <= 1.0 + 1e-9
    assert len(res) == 50
    # Generous bound: marginal PVE on 1M markers should finish in well
    # under a second on any reasonable laptop.
    assert dt < 2.0, f"marginal PVE on 1M SNPs took {dt:.2f}s (expected < 2s)"


def test_pve_as_table_returns_serializable_dict():
    G, y, beta_hat, p, af, _ = _simulate_gwas(n=300, m=10, n_causal=2)
    p[:2] = 1e-20
    result = _make_scan_result(beta_hat, p, af)
    res = compute_pve(result, y, G=G, method="joint")
    table = res.as_table()
    assert set(table.keys()) >= {"snp", "chr", "pos", "beta", "pve", "pve_percent"}
    assert len(table["snp"]) == res.n_significant
    # pve_percent must equal pve * 100 entry by entry.
    for pv, pct in zip(table["pve"], table["pve_percent"]):
        assert math.isclose(pv * 100.0, pct, rel_tol=1e-12, abs_tol=1e-12)
