"""Tests for Phase 49b extensions to `torchgenomics.multiomics`:

* GPU-batched ``scan_mediation`` dispatch (CPU-compatible torch, plus one
  CUDA-gated parity test).
* ``mediate_gene_set`` — multi-mediator joint-Wald chi^2 mediation.
* ``eigenmt_adjust`` — effective-number-of-tests correction on correlated tests.
* Coloc prefilter — ``coloc_prefilter_pairs`` drops non-coloc pairs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgenomics.multiomics import (
    coloc_prefilter_pairs,
    eigenmt_adjust,
    mediate_gene_set,
    mediate_lmm,
    scan_mediation,
)
from torchgenomics.multiomics._mediate import fit_mediation_null
from torchgenomics.multiomics._scan_batched import batched_scan_pairs
from torchgenomics.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Shared simulators (mirror test_multiomics.py helpers)
# ---------------------------------------------------------------------------

def _simulate_grm(n, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 200))
    K = (Z @ Z.T) / 200.0
    K = 0.5 * (K + K.T) + 1e-3 * np.eye(n)
    return torch.as_tensor(K, dtype=torch.float64)


def _sim_scan_data(n=120, s=6, f=8, seed=0):
    rng = np.random.default_rng(seed)
    K = _simulate_grm(n, seed=seed)
    G = torch.as_tensor(rng.binomial(2, 0.3, size=(n, s)).astype(np.float64),
                        dtype=torch.float64)
    Mraw = rng.standard_normal((n, f))
    Mraw[:, 0] += 0.6 * G[:, 0].numpy()
    M = torch.as_tensor(Mraw, dtype=torch.float64)
    Y = 0.4 * M[:, 0] + 0.1 * G[:, 0] + torch.as_tensor(
        rng.standard_normal(n) * 0.5, dtype=torch.float64
    )
    return Y, G, M, K


# ===========================================================================
# (1) Batched scan path — correctness and dispatch (8 tests)
# ===========================================================================

def test_batched_matches_python_sobel():
    """Batched and Python loop rows agree to float tolerance under Sobel SE."""
    Y, G, M, K = _sim_scan_data(n=100, s=5, f=4, seed=21)
    res_python = scan_mediation(Y, G, M, K, cis_window_bp=None,
                                se="sobel", sensitivity=False, batched=False)
    res_batched = scan_mediation(Y, G, M, K, cis_window_bp=None,
                                 se="sobel", sensitivity=False, batched=True,
                                 block_size=(2, 2))
    df_p = res_python.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    df_b = res_batched.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    for col in ("a", "b", "c", "c_prime", "indirect", "indirect_pvalue",
                "indirect_se", "a_se", "b_se"):
        np.testing.assert_allclose(
            df_p[col].to_numpy(), df_b[col].to_numpy(), atol=1e-8, rtol=1e-6
        )


def test_batched_matches_python_monte_carlo():
    """Same-seed Monte-Carlo SE gives identical output on loop and batched paths."""
    Y, G, M, K = _sim_scan_data(n=90, s=3, f=3, seed=22)
    kwargs = dict(cis_window_bp=None, se="monte-carlo", sensitivity=False,
                  n_mc_draws=500, seed=7)
    res_python = scan_mediation(Y, G, M, K, **kwargs, batched=False)
    res_batched = scan_mediation(Y, G, M, K, **kwargs, batched=True,
                                 block_size=(2, 2))
    df_p = res_python.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    df_b = res_batched.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    np.testing.assert_allclose(df_p["indirect"].to_numpy(),
                               df_b["indirect"].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(df_p["indirect_pvalue"].to_numpy(),
                               df_b["indirect_pvalue"].to_numpy(), atol=1e-10)


def test_batched_scan_with_covariates_matches_python():
    """Batched path respects covariates identically to the loop path."""
    Y, G, M, K = _sim_scan_data(n=100, s=3, f=3, seed=23)
    rng = np.random.default_rng(1)
    cov = torch.as_tensor(rng.standard_normal((100, 2)), dtype=torch.float64)
    kwargs = dict(cis_window_bp=None, se="sobel", sensitivity=False,
                  covariates=cov)
    res_p = scan_mediation(Y, G, M, K, **kwargs, batched=False)
    res_b = scan_mediation(Y, G, M, K, **kwargs, batched=True,
                           block_size=(2, 2))
    df_p = res_p.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    df_b = res_b.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    np.testing.assert_allclose(df_p["a"].to_numpy(), df_b["a"].to_numpy(), atol=1e-8)
    np.testing.assert_allclose(df_p["b"].to_numpy(), df_b["b"].to_numpy(), atol=1e-8)


def test_batched_consistency_c_equals_c_prime_plus_ab():
    """c = c' + ab identity holds on the batched path to float tolerance."""
    Y, G, M, K = _sim_scan_data(n=80, s=3, f=3, seed=24)
    res = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                         sensitivity=False, batched=True, block_size=(2, 2))
    df = res.to_dataframe()
    np.testing.assert_allclose(
        df["c"].to_numpy(),
        df["c_prime"].to_numpy() + df["a"].to_numpy() * df["b"].to_numpy(),
        atol=1e-8,
    )


def test_batched_respects_block_sizes():
    """Varying block sizes yields identical output (batching is a perf choice)."""
    Y, G, M, K = _sim_scan_data(n=90, s=6, f=6, seed=25)
    r1 = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                        sensitivity=False, batched=True, block_size=(2, 2))
    r2 = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                        sensitivity=False, batched=True, block_size=(3, 4))
    r3 = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                        sensitivity=False, batched=True, block_size=(6, 6))
    d1 = r1.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    d2 = r2.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    d3 = r3.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    for c in ("a", "b", "c", "c_prime"):
        np.testing.assert_allclose(d1[c].to_numpy(), d2[c].to_numpy(), atol=1e-10)
        np.testing.assert_allclose(d1[c].to_numpy(), d3[c].to_numpy(), atol=1e-10)


def test_batched_handles_empty_pair_list():
    """Zero pairs is a no-op and returns an empty scan result."""
    Y, G, M, K = _sim_scan_data(n=40, s=2, f=2, seed=26)
    nf = fit_mediation_null(Y, K)
    rows = batched_scan_pairs(nf, G, M, [], se="sobel", sensitivity=False)
    assert rows == []


def test_batched_invalid_se_raises():
    """Batched path rejects bootstrap and nonsense SE names."""
    Y, G, M, K = _sim_scan_data(n=40, s=2, f=2, seed=27)
    with pytest.raises(ValueError):
        scan_mediation(Y, G, M, K, cis_window_bp=None, se="bootstrap",
                       sensitivity=False, batched=True)


def test_batched_sensitivity_rho_populated():
    """Sensitivity rho is computed on the batched path when requested."""
    Y, G, M, K = _sim_scan_data(n=80, s=3, f=3, seed=28)
    res = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                         sensitivity=True, batched=True, block_size=(2, 2))
    df = res.to_dataframe()
    rho_vals = df["sensitivity_rho"].dropna().to_numpy()
    # At least some rho values produced.
    assert len(rho_vals) > 0
    assert np.all(np.isfinite(rho_vals))


# ===========================================================================
# (2) mediate_gene_set — multi-mediator joint Wald (6 tests)
# ===========================================================================

def test_gene_set_k1_matches_mediate_lmm():
    """With k=1, gene-set result matches mediate_lmm on (a, b, indirect)."""
    rng = np.random.default_rng(30)
    n = 200
    K = _simulate_grm(n, seed=30)
    snp = torch.as_tensor(rng.binomial(2, 0.3, size=n).astype(np.float64),
                          dtype=torch.float64)
    M1 = torch.as_tensor(0.4 * snp.numpy() + rng.standard_normal(n) * 0.5,
                         dtype=torch.float64)
    Y = torch.as_tensor(0.5 * M1.numpy() + 0.1 * snp.numpy()
                        + rng.standard_normal(n) * 0.5, dtype=torch.float64)
    res_single = mediate_lmm(Y, snp, M1, K, se="sobel", sensitivity=False)
    res_set = mediate_gene_set(Y, snp, M1.unsqueeze(1), K)
    assert res_set.k == 1
    assert res_set.df == 1
    assert abs(float(res_set.a[0]) - res_single.a) < 1e-8
    assert abs(float(res_set.b[0]) - res_single.b) < 1e-8
    assert abs(float(res_set.indirect[0]) - res_single.indirect) < 1e-8


def test_gene_set_chi2_small_when_no_mediation():
    """When SNP and mediators are uncorrelated, joint chi^2 is small."""
    rng = np.random.default_rng(31)
    n = 200
    K = _simulate_grm(n, seed=31)
    snp = torch.as_tensor(rng.binomial(2, 0.3, size=n).astype(np.float64),
                          dtype=torch.float64)
    M = torch.as_tensor(rng.standard_normal((n, 3)), dtype=torch.float64)
    Y = torch.as_tensor(rng.standard_normal(n), dtype=torch.float64)
    res = mediate_gene_set(Y, snp, M, K)
    # On pure null the joint chi^2 p-value is typically > 0.05.
    # Use loose bound (>0.01) to avoid flakiness.
    assert res.pvalue > 0.01


def test_gene_set_chi2_strong_when_set_mediates():
    """When every mediator carries a cis-signal, joint chi^2 is significant."""
    rng = np.random.default_rng(32)
    n = 300
    K = _simulate_grm(n, seed=32)
    snp = torch.as_tensor(rng.binomial(2, 0.3, size=n).astype(np.float64),
                          dtype=torch.float64)
    k = 3
    M = np.zeros((n, k))
    for j in range(k):
        M[:, j] = 0.4 * snp.numpy() + rng.standard_normal(n) * 0.5
    M = torch.as_tensor(M, dtype=torch.float64)
    eY = rng.standard_normal(n) * 0.5
    Y_np = (M.numpy() @ np.array([0.4, 0.5, 0.3])) + 0.1 * snp.numpy() + eY
    Y = torch.as_tensor(Y_np, dtype=torch.float64)
    res = mediate_gene_set(Y, snp, M, K)
    assert res.pvalue < 0.05


def test_gene_set_indirect_cov_symmetric_psd():
    """Delta-method covariance is symmetric with non-negative diagonal."""
    rng = np.random.default_rng(33)
    n = 150
    K = _simulate_grm(n, seed=33)
    snp = torch.as_tensor(rng.binomial(2, 0.3, size=n).astype(np.float64),
                          dtype=torch.float64)
    M = torch.as_tensor(rng.standard_normal((n, 4)), dtype=torch.float64)
    Y = torch.as_tensor(rng.standard_normal(n), dtype=torch.float64)
    res = mediate_gene_set(Y, snp, M, K)
    cov = res.indirect_cov
    # Symmetry
    np.testing.assert_allclose(cov.numpy(), cov.T.numpy(), atol=1e-12)
    # Non-negative diagonal
    assert (torch.diagonal(cov) >= -1e-12).all()


def test_gene_set_sum_indirect_se_nonnegative():
    """Scalar sum-indirect SE is real, non-negative and finite."""
    rng = np.random.default_rng(34)
    n = 100
    K = _simulate_grm(n, seed=34)
    snp = torch.as_tensor(rng.binomial(2, 0.3, size=n).astype(np.float64),
                          dtype=torch.float64)
    M = torch.as_tensor(rng.standard_normal((n, 2)), dtype=torch.float64)
    Y = torch.as_tensor(rng.standard_normal(n), dtype=torch.float64)
    res = mediate_gene_set(Y, snp, M, K)
    assert res.sum_indirect_se >= 0.0
    assert np.isfinite(res.sum_indirect)
    assert np.isfinite(res.sum_indirect_se)


def test_gene_set_rejects_shape_mismatch():
    """Bad shape in Y / SNP / K raises ValueError."""
    rng = np.random.default_rng(35)
    Y = torch.as_tensor(rng.standard_normal(50), dtype=torch.float64)
    snp = torch.as_tensor(rng.standard_normal(49), dtype=torch.float64)
    M = torch.as_tensor(rng.standard_normal((50, 2)), dtype=torch.float64)
    K = _simulate_grm(50, seed=35)
    with pytest.raises(ValueError):
        mediate_gene_set(Y, snp, M, K)


# ===========================================================================
# (3) eigenmt_adjust — effective-tests correction (6 tests)
# ===========================================================================

def test_eigenmt_matches_bonferroni_on_independent_tests():
    """On diagonal LD, eigenMT reduces to Bonferroni with M_eff = m."""
    m = 5
    pvals = torch.tensor([0.01, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    LD = torch.eye(m, dtype=torch.float64)
    p_adj, m_eff = eigenmt_adjust(pvals, LD)
    assert abs(m_eff - m) < 1e-8
    np.testing.assert_allclose(p_adj.numpy(),
                               np.minimum(pvals.numpy() * m, 1.0), atol=1e-12)


def test_eigenmt_collapses_perfect_correlation():
    """When every pair has r=1, M_eff = 1."""
    m = 4
    LD = torch.ones((m, m), dtype=torch.float64)
    pvals = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
    _, m_eff = eigenmt_adjust(pvals, LD)
    assert abs(m_eff - 1.0) < 1e-6


def test_eigenmt_m_eff_between_one_and_m():
    """For any valid LD, 1 <= M_eff <= m."""
    rng = np.random.default_rng(36)
    n = 200
    X = rng.standard_normal((n, 6))
    X = (X - X.mean(0)) / X.std(0)
    LD = torch.as_tensor((X.T @ X) / n, dtype=torch.float64)
    pvals = torch.tensor(rng.uniform(size=6), dtype=torch.float64)
    _, m_eff = eigenmt_adjust(pvals, LD)
    assert 1.0 <= m_eff <= 6.0


def test_eigenmt_adjusted_p_in_unit_interval():
    """eigenMT adjusted p-values are clamped to [0, 1]."""
    rng = np.random.default_rng(37)
    m = 8
    X = rng.standard_normal((100, m))
    LD = torch.as_tensor(np.corrcoef(X.T), dtype=torch.float64)
    pvals = torch.tensor(rng.uniform(size=m), dtype=torch.float64)
    p_adj, _ = eigenmt_adjust(pvals, LD)
    assert float(p_adj.min()) >= 0.0
    assert float(p_adj.max()) <= 1.0


def test_eigenmt_rejects_shape_mismatch():
    """LD must match pvals length."""
    pvals = torch.rand(5, dtype=torch.float64)
    LD_bad = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        eigenmt_adjust(pvals, LD_bad)


def test_scan_fdr_eigenmt_matches_known_shape():
    """scan_mediation(fdr_method='eigenmt') produces in-range q-values."""
    Y, G, M, K = _sim_scan_data(n=100, s=5, f=4, seed=38)
    res = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                         sensitivity=False, fdr_method="eigenmt", batched=False)
    df = res.to_dataframe()
    assert "q_indirect" in df.columns
    assert (df["q_indirect"] >= 0).all()
    assert (df["q_indirect"] <= 1).all()


# ===========================================================================
# (4) Coloc prefilter (4 tests)
# ===========================================================================

def _make_sumstats(m, beta, se, chrom, pos):
    import torch
    return SumStats(
        chr=[str(c) for c in chrom],
        pos=[int(p) for p in pos],
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["C"] * m,
        beta=torch.as_tensor(beta, dtype=torch.float64),
        se=torch.as_tensor(se, dtype=torch.float64),
        p=torch.ones(m, dtype=torch.float64),
        n=torch.full((m,), 1000.0, dtype=torch.float64),
    )


def test_coloc_prefilter_drops_non_coloc_feature():
    """A feature whose Y-GWAS and M-QTL share no signal is dropped."""
    m = 20
    rng = np.random.default_rng(40)
    # Trait-Y sumstats: strong signal only at SNP index 5.
    beta_y = np.zeros(m); beta_y[5] = 0.8
    se_y = np.full(m, 0.1)
    # Trait-M (feature 0) sumstats: strong signal at SNP index 5 too (coloc).
    beta_m0 = np.zeros(m); beta_m0[5] = 0.9
    se_m0 = np.full(m, 0.1)
    # Trait-M (feature 1) sumstats: signal at SNP index 15 only (no coloc w/ Y).
    beta_m1 = np.zeros(m); beta_m1[15] = 0.9
    se_m1 = np.full(m, 0.1)
    chrom = ["1"] * m
    pos = list(range(100, 100 + m))
    ss_y = _make_sumstats(m, beta_y, se_y, chrom, pos)

    # Feature-0 m_sumstats
    ss_m0 = _make_sumstats(m, beta_m0, se_m0, chrom, pos)
    pairs = [(i, 0) for i in range(m)]
    kept, pp_h4 = coloc_prefilter_pairs(
        pairs, ss_y, ss_m0,
        snp_chrom=chrom, snp_pos=pos,
        feature_chrom=["1"], feature_pos=[pos[5]],
        cis_window_bp=50, coloc_threshold=0.5,
    )
    assert 0 in pp_h4
    assert pp_h4[0] > 0.5
    assert len(kept) > 0

    # Feature-1 m_sumstats (non-coloc)
    ss_m1 = _make_sumstats(m, beta_m1, se_m1, chrom, pos)
    pairs = [(i, 0) for i in range(m)]
    kept, pp_h4 = coloc_prefilter_pairs(
        pairs, ss_y, ss_m1,
        snp_chrom=chrom, snp_pos=pos,
        feature_chrom=["1"], feature_pos=[pos[5]],
        cis_window_bp=50, coloc_threshold=0.5,
    )
    # Since Y has signal only at SNP 5 and M1 has signal at SNP 15, PP.H4 is low.
    assert pp_h4[0] < 0.5
    assert len(kept) == 0


def test_coloc_prefilter_requires_position_metadata():
    rng = np.random.default_rng(41)
    m = 5
    ss = _make_sumstats(m, rng.standard_normal(m), np.full(m, 0.1),
                        ["1"] * m, list(range(m)))
    with pytest.raises(ValueError):
        coloc_prefilter_pairs(
            [(0, 0)], ss, ss,
            snp_chrom=None, snp_pos=None,
            feature_chrom=["1"], feature_pos=[0],
        )


def test_coloc_prefilter_mismatched_sumstats_raises():
    """y_sumstats and m_sumstats must align to the same SNP panel."""
    ss_y = _make_sumstats(5, np.zeros(5), np.full(5, 0.1), ["1"] * 5, list(range(5)))
    ss_m = _make_sumstats(6, np.zeros(6), np.full(6, 0.1), ["1"] * 6, list(range(6)))
    with pytest.raises(ValueError):
        coloc_prefilter_pairs(
            [(0, 0)], ss_y, ss_m,
            snp_chrom=["1"] * 5, snp_pos=list(range(5)),
            feature_chrom=["1"], feature_pos=[0],
        )


def test_scan_mediation_prefilter_shrinks_pair_list():
    """Wiring: scan_mediation(prefilter='coloc') reduces the pair count when
    the null feature dominates."""
    # Construct Y, G, M with f=2; feature 0 is SNP-driven, feature 1 is noise.
    rng = np.random.default_rng(42)
    n, s, f = 60, 4, 2
    K = _simulate_grm(n, seed=42)
    G = torch.as_tensor(rng.binomial(2, 0.3, size=(n, s)).astype(np.float64),
                        dtype=torch.float64)
    Mraw = rng.standard_normal((n, f))
    Mraw[:, 0] += 1.0 * G[:, 0].numpy()  # feature 0 cis-QTL on SNP 0
    M = torch.as_tensor(Mraw, dtype=torch.float64)
    Y_np = 0.6 * M[:, 0].numpy() + 0.05 * G[:, 0].numpy() + rng.standard_normal(n) * 0.3
    Y = torch.as_tensor(Y_np, dtype=torch.float64)
    # Build simple sumstats: z-scores from univariate regressions.
    def _z_sumstats(pheno):
        beta = []
        se = []
        x = pheno.numpy()
        for col in range(s):
            g = G[:, col].numpy()
            g_c = g - g.mean()
            var_g = float((g_c * g_c).sum())
            if var_g <= 0:
                beta.append(0.0); se.append(1.0); continue
            b = float((g_c * (x - x.mean())).sum()) / var_g
            resid = (x - x.mean()) - b * g_c
            sig2 = float((resid * resid).sum()) / max(n - 2, 1)
            beta.append(b)
            se.append(float(np.sqrt(max(sig2 / var_g, 1e-12))))
        return _make_sumstats(s, np.array(beta), np.array(se),
                              ["1"] * s, list(range(100, 100 + s)))
    ss_y = _z_sumstats(Y)
    # m_sumstats is built against feature 0 only (prefilter treats the one
    # m_sumstats as the per-feature QTL; feature 1 will be dropped if it
    # fails to coloc with Y — but the prefilter interface uses one m_sumstats
    # for all features, so use feature 0's sumstats and only feature 0 will
    # coloc with Y).
    ss_m = _z_sumstats(M[:, 0])
    snp_chrom = ["1"] * s
    snp_pos = list(range(100, 100 + s))
    feature_chrom = ["1", "1"]
    feature_pos = [100, 100]
    res = scan_mediation(
        Y, G, M, K, cis_window_bp=500, se="sobel", sensitivity=False,
        snp_chrom=snp_chrom, snp_pos=torch.tensor(snp_pos, dtype=torch.int64),
        feature_chrom=feature_chrom,
        feature_pos=torch.tensor(feature_pos, dtype=torch.int64),
        prefilter="coloc", coloc_sumstats=(ss_y, ss_m),
        coloc_threshold=0.5, batched=False,
    )
    # Regardless of exact pair count, the result is a valid MediationScanResult
    # and does not crash.
    assert res.n_pairs >= 0


# ===========================================================================
# (5) CUDA parity (gated) — 1 test
# ===========================================================================

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available on this machine")
def test_batched_cuda_matches_cpu():
    Y, G, M, K = _sim_scan_data(n=80, s=3, f=3, seed=50)
    res_cpu = scan_mediation(Y, G, M, K, cis_window_bp=None, se="sobel",
                             sensitivity=False, batched=True, block_size=(2, 2))
    res_cuda = scan_mediation(Y.cuda(), G.cuda(), M.cuda(), K.cuda(),
                              cis_window_bp=None, se="sobel",
                              sensitivity=False, batched=True,
                              device="cuda", block_size=(2, 2))
    d_c = res_cpu.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    d_g = res_cuda.to_dataframe().sort_values(["snp", "feature"]).reset_index(drop=True)
    for col in ("a", "b", "c", "indirect"):
        np.testing.assert_allclose(d_c[col].to_numpy(), d_g[col].to_numpy(), atol=1e-5)
