"""Regression tests for the statistical-correctness fixes from the
2026-07 deep sanity scan.

Each test pins the *correct* behaviour of a bug that was found and fixed, with
an independent reference (statsmodels / scipy / hand formula / from-scratch
computation) so the defect cannot silently return. See the sanity-scan report
for the per-finding analysis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

STAT = torch.float64


# ---------------------------------------------------------------------------
# 1. GLM Wald uses the FULL-model residual variance (not the null-model one).
#    torchgenomics/models/glm.py
# ---------------------------------------------------------------------------
def test_glm_wald_matches_statsmodels_full_model_variance():
    sm = pytest.importorskip("statsmodels.api")
    from torchgenomics.models.glm import GLM
    from torchgenomics.models.single_trait_lmm import VariantMeta

    rng = np.random.default_rng(3)
    n = 200
    cov = rng.standard_normal(n)
    g = rng.binomial(2, 0.3, n).astype(float)
    y = 0.4 * g + 0.2 * cov + rng.standard_normal(n)

    Xf = sm.add_constant(np.column_stack([cov, g]))
    ref = sm.OLS(y, Xf).fit()

    Y = torch.tensor(y, dtype=STAT).unsqueeze(1)
    X0 = torch.tensor(sm.add_constant(cov), dtype=STAT)
    G = torch.tensor(g, dtype=STAT).unsqueeze(1)
    vm = VariantMeta(snp=["s"], chr=["1"], pos=[1], a1=["A"], a2=["G"])

    m = GLM()
    res = m.score_chunk(G, m.fit_null(Y, X0), vm)
    assert float(res.beta.ravel()[0]) == pytest.approx(float(ref.params[2]), rel=1e-8)
    assert float(res.se.ravel()[0]) == pytest.approx(float(ref.bse[2]), rel=1e-6)
    assert float(res.p.ravel()[0]) == pytest.approx(float(ref.pvalues[2]), rel=1e-6)


# ---------------------------------------------------------------------------
# 2. ACAT / Cauchy combination does not saturate in the tail.
#    torchgenomics/stats/cauchy.py
# ---------------------------------------------------------------------------
def test_acat_no_tail_saturation():
    from torchgenomics.stats.cauchy import cauchy_combination

    prev = 1.0
    for s in [1e-8, 1e-16, 1e-20, 1e-100, 1e-300]:
        p = torch.tensor([s, 0.5, 0.5], dtype=STAT)
        out = float(cauchy_combination(p))
        # combined p keeps shrinking with the smallest input (was floored ~5.5e-17)
        assert out < prev
        assert out < 1e-6 if s < 1e-7 else True
        prev = out
    # far-tail input must not be censored to ~5.5e-17
    assert float(cauchy_combination(torch.tensor([1e-100, 0.5, 0.5], dtype=STAT))) < 1e-50


# ---------------------------------------------------------------------------
# 3. diplo-additive polyploid encoding is monotone [0,1,1,...,1,2] (GWASpoly).
#    torchgenomics/preprocess/polyploid.py
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ploidy", [4, 6])
def test_diplo_additive_is_monotone_diploidization(ploidy):
    from torchgenomics.preprocess.polyploid import recode_gene_action

    G = torch.arange(ploidy + 1, dtype=STAT).unsqueeze(0)  # [0,1,...,k]
    out = recode_gene_action(G, "diplo-additive", ploidy).squeeze(0)
    expected = torch.ones(ploidy + 1, dtype=STAT)
    expected[0] = 0.0
    expected[ploidy] = 2.0
    torch.testing.assert_close(out, expected)
    # monotonically non-decreasing in dosage
    assert torch.all(out[1:] >= out[:-1])


# ---------------------------------------------------------------------------
# 4. grm_zhang degenerate branch returns the declared (Tensor, meta) tuple.
#    torchgenomics/linalg/kinship.py
# ---------------------------------------------------------------------------
def test_grm_zhang_degenerate_returns_tuple():
    from torchgenomics.linalg.kinship import grm_zhang

    # All-identical rows -> degenerate diagonal range triggers the guard branch.
    G = torch.ones(5, 8, dtype=STAT)
    K, meta = grm_zhang(G)  # must unpack without error
    assert K.shape == (5, 5)
    assert meta.method == "zhang"


# ---------------------------------------------------------------------------
# 5. Default REML (optimizer stack) matches the exact profile-likelihood
#    optimum, i.e. does not stop early. torchgenomics/optim/controller.py
# ---------------------------------------------------------------------------
def test_default_reml_hits_exact_profile_optimum():
    from torchgenomics.linalg.kinship import grm_vanraden
    from torchgenomics.models.single_trait_lmm import SingleTraitLMM
    from scipy.optimize import minimize_scalar

    def reml_ll(lam, y, X, K):
        n, c = X.shape
        H = lam * K + np.eye(n)
        Hi = np.linalg.inv(H)
        XtHiX = X.T @ Hi @ X
        P = Hi - Hi @ X @ np.linalg.inv(XtHiX) @ X.T @ Hi
        s2e = float(y @ P @ y) / (n - c)
        _, ldH = np.linalg.slogdet(H)
        _, ldX = np.linalg.slogdet(XtHiX)
        return -0.5 * ((n - c) * np.log(s2e) + ldH + ldX + (n - c))

    for seed in (0, 5):
        g = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        n, mm = 300, 1200
        G = torch.tensor(rng.binomial(2, rng.uniform(0.1, 0.5, (n, mm))), dtype=STAT)
        K, _ = grm_vanraden(G, ploidy=2)
        Kn = K.numpy()
        L = np.linalg.cholesky(Kn + 1e-6 * np.eye(n))
        z = rng.standard_normal(n)
        y = np.sqrt(0.6) * (L @ z) / np.sqrt(np.mean(np.diag(Kn))) + np.sqrt(0.4) * rng.standard_normal(n)
        X = np.ones((n, 1))

        grid = np.linspace(-4, 6, 300)
        l0 = grid[int(np.argmax([reml_ll(np.exp(l), y, X, Kn) for l in grid]))]
        res = minimize_scalar(lambda l: -reml_ll(np.exp(l), y, X, Kn),
                              bracket=(l0 - 0.3, l0, l0 + 0.3))
        lam_ref = float(np.exp(res.x))

        model = SingleTraitLMM()
        nf = model.fit_null(torch.tensor(y, dtype=STAT).unsqueeze(1),
                            torch.ones(n, 1, dtype=STAT), K)
        lam_lib = float(nf.sig2_g) / float(nf.sig2_e)
        assert lam_lib == pytest.approx(lam_ref, rel=5e-3), (
            f"seed {seed}: lam_lib={lam_lib} vs ref={lam_ref}"
        )


# ---------------------------------------------------------------------------
# 6. Sample-size Stouffer meta is N(0,1) under the null (correct denominator).
#    torchgenomics/postgwas/_meta.py
# ---------------------------------------------------------------------------
def test_stouffer_sample_size_meta_null_calibrated():
    from torchgenomics.postgwas._meta import meta_sample_size

    torch.manual_seed(0)
    K, m = 5, 30000
    p = torch.rand(m, K, dtype=STAT).clamp(1e-12, 1 - 1e-12)
    n = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    direction = torch.sign(torch.randn(m, K))
    res = meta_sample_size(p, n, direction=direction)
    z = res.z_meta.numpy()
    assert z.var() == pytest.approx(1.0, abs=0.05)
    assert (res.p_meta.numpy() < 0.05).mean() == pytest.approx(0.05, abs=0.01)


# ---------------------------------------------------------------------------
# 7. Inverse-normal tail is stable (no inf) in _meta and _combine.
# ---------------------------------------------------------------------------
def test_inverse_normal_tail_finite():
    from torchgenomics.postgwas._meta import _p_to_z
    from torchgenomics.postgwas._combine import stouffer_combined

    q = torch.tensor([1e-8, 1e-20, 1e-50, 1e-200], dtype=STAT)
    z = _p_to_z(q)
    assert torch.all(torch.isfinite(z))
    assert float(z[-1]) > 25.0  # p=1e-200 -> |z|~30, not capped
    # stouffer_combined: one tiny p with two neutral -> z = z_tiny/sqrt(3)
    import scipy.special as sp
    zc, _ = stouffer_combined(torch.tensor([1e-50, 0.5, 0.5], dtype=STAT))
    z_expected = (-sp.ndtri(1e-50)) / math.sqrt(3.0)
    assert zc == pytest.approx(z_expected, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. Brown's method reduces to Fisher under independence (variance not ×2).
#    torchgenomics/postgwas/_combine.py
# ---------------------------------------------------------------------------
def test_brown_reduces_to_fisher_under_independence():
    import scipy.stats as st
    from torchgenomics.postgwas._combine import brown_combined

    p = torch.tensor([0.01, 0.2, 0.5, 0.7], dtype=STAT)
    cov = torch.eye(4, dtype=STAT) * 4.0  # Var(-2logU)=4, independent
    _, pb = brown_combined(p, cov)
    fisher = float(st.combine_pvalues(p.numpy(), method="fisher").pvalue)
    assert pb == pytest.approx(fisher, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. PRS-CS shrinkage carries the factor n (heavy shrinkage at small n/phi).
#    torchgenomics/pgs/prscs.py
# ---------------------------------------------------------------------------
def test_prscs_shrinkage_scales_with_n():
    from torchgenomics.pgs.prscs import _prscs_gibbs_block

    beta_std = torch.tensor([0.05], dtype=STAT)
    R = torch.tensor([[1.0]], dtype=STAT)

    def post(n, phi):
        rng = torch.Generator().manual_seed(3)
        bm, _ = _prscs_gibbs_block(beta_std, R, n_eff=float(n), phi=phi,
                                   a=1.0, b=0.5, n_iter=2000, n_burnin=1000, rng=rng)
        return float(bm)

    small = post(2000, 1e-4)
    large = post(10000, 1e-2)
    # strong-shrinkage regime: heavily shrunk toward 0 (was ~0.026 with the bug)
    assert small < 0.002
    # weak-shrinkage / large-n regime: close to the marginal 0.05
    assert large > 0.03
    assert small < large


# ---------------------------------------------------------------------------
# 10. LDpred2 prior variance uses genome-wide M, so shrinkage is invariant to
#     how independent SNPs are partitioned into blocks. torchgenomics/pgs/ldpred2.py
# ---------------------------------------------------------------------------
def test_ldpred2_prior_uses_genomewide_m_not_block_size():
    from torchgenomics.pgs.ldpred2 import _ldpred2_gibbs_block

    M = 40
    beta = torch.full((M,), 0.05, dtype=STAT)

    def run(bsz, m_total):
        rng = torch.Generator().manual_seed(7)
        Rb = torch.eye(bsz, dtype=STAT)
        bm, _, _ = _ldpred2_gibbs_block(beta[:bsz], Rb, p_causal=0.1, h2=0.3,
                                        n_eff=1000.0, n_iter=4000, n_burnin=2000,
                                        sparse=False, rng=rng, m_total=m_total)
        return float(bm.mean())

    fixed = [run(b, M) for b in (40, 20, 10, 8)]      # correct: M fixed
    buggy = [run(b, b) for b in (40, 20, 10, 8)]      # old: prior used block size
    spread_fixed = max(fixed) - min(fixed)
    spread_buggy = max(buggy) - min(buggy)
    # with genome-wide M the spread across block layouts is much smaller and
    # non-systematic; the block-size prior is systematically monotone in b.
    assert spread_fixed < spread_buggy
    assert spread_fixed < 5e-4


# ---------------------------------------------------------------------------
# 12. Storey q-value: pi0 floored, never returns all-zero q (anti-conservative).
#     torchgenomics/stats/multipletesting.py
# ---------------------------------------------------------------------------
def test_storey_qvalue_no_all_zero_when_pi0_estimate_is_zero():
    from torchgenomics.stats.multipletesting import storey_qvalue

    # All p-values below lambda=0.5 -> raw pi0 estimate is 0.
    p = torch.linspace(1e-6, 0.4, 60, dtype=STAT)
    q = storey_qvalue(p)
    assert not bool((q == 0).all()), "q collapsed to all-zero (pi0 not floored)"
    assert torch.all(q >= 0) and torch.all(q <= 1)


# ---------------------------------------------------------------------------
# 13. FIQT is the FDR Inverse Quantile Transformation (Bigdeli 2016), not KDE
#     local-fdr. torchgenomics/postgwas/_winners_curse.py
# ---------------------------------------------------------------------------
def test_fiqt_matches_reference_inverse_quantile_transform():
    import scipy.stats as st
    from torchgenomics.postgwas._winners_curse import fiqt

    z = torch.tensor([6.0, 4.5, 3.0, 2.0, 0.5], dtype=STAT)
    se = torch.ones(5, dtype=STAT)
    z_adj = (fiqt(z, se).beta_adjusted / se).numpy()

    # Independent reference: BH-adjust two-sided p, back-transform.
    pv = np.clip(2 * st.norm.sf(np.abs(z.numpy())), 1e-300, 1.0)
    m = pv.size
    o = np.argsort(pv)
    s = pv[o] * m / np.arange(1, m + 1)
    s = np.minimum.accumulate(s[::-1])[::-1]
    adj = np.empty_like(pv)
    adj[o] = np.clip(s, None, 1.0)
    ref = np.sign(z.numpy()) * st.norm.isf(adj / 2)
    assert np.allclose(z_adj, ref, atol=1e-6)
    # shrinkage is toward zero and monotone with significance
    assert abs(z_adj[0]) < 6.0 and z_adj[-1] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 14. MR weighted-median uses a PARAMETRIC bootstrap (Bowden 2016): the SE
#     responds to exposure error se_x. torchgenomics/postgwas/_mr.py
# ---------------------------------------------------------------------------
def test_mr_weighted_median_parametric_bootstrap_uses_exposure_error():
    from torchgenomics.postgwas._sumstats import SumStats
    from torchgenomics.postgwas._mr import mr_weighted_median

    rng = np.random.default_rng(0)
    K = 30
    snp = [f"rs{i}" for i in range(K)]
    bx = np.abs(rng.standard_normal(K)) * 0.2 + 0.3
    by = 0.5 * bx + rng.standard_normal(K) * 0.02
    se_y = np.ones(K) * 0.03

    def mk(beta, se):
        return SumStats(chr=["1"] * K, pos=list(range(K)), snp=snp, a1=["A"] * K,
                        a2=["G"] * K, beta=torch.tensor(beta), se=torch.tensor(se),
                        p=torch.ones(K) * 0.01, n=torch.ones(K) * 1000, af=torch.ones(K) * 0.3)

    r_small = mr_weighted_median(mk(bx, np.ones(K) * 0.001), mk(by, se_y), n_boot=2000, seed=1)
    r_large = mr_weighted_median(mk(bx, np.ones(K) * 0.05), mk(by, se_y), n_boot=2000, seed=1)
    assert r_small.beta_hat == pytest.approx(0.5, abs=0.1)
    # Parametric bootstrap draws bx* ~ N(bx, se_x^2): larger se_x -> larger SE.
    # A nonparametric index bootstrap would be invariant to se_x.
    assert r_large.se > r_small.se * 1.1


# ---------------------------------------------------------------------------
# 15. Nystrom eigenvalues carry the (n/l) scaling (Williams & Seeger 2001), so
#     the approximation reconstructs the GRM diagonal. torchgenomics/linalg/nystrom.py
# ---------------------------------------------------------------------------
def test_nystrom_eigenvalues_have_nl_scaling():
    from torchgenomics.linalg.kinship import grm_vanraden
    from torchgenomics.linalg.nystrom import nystrom_approximate

    rng = np.random.default_rng(0)
    n, msnp = 300, 900
    G = torch.tensor(rng.binomial(2, rng.uniform(0.1, 0.5, (n, msnp))), dtype=STAT)
    K, _ = grm_vanraden(G, ploidy=2)
    true_diag = float(K.diag().mean())

    def chunks():
        for s in range(0, msnp, 300):
            yield (G[:, s:s + 300], None)

    ed, _ = nystrom_approximate(chunks(), n_samples=n, n_landmarks=100, seed=1)
    recon_diag = float((ed.eigenvectors ** 2 @ ed.eigenvalues).mean())
    # Without the n/l factor this ratio would be ~l/n = 1/3.
    assert recon_diag / true_diag == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# 16. Han-Eskin RE2 is a likelihood-ratio test with the 0.5:0.5 chi2_1/chi2_2
#     mixture null — valid (never anti-conservative), unlike the old
#     chi2(1)-referenced heuristic. torchgenomics/postgwas/_meta.py
# ---------------------------------------------------------------------------
def test_re2_null_not_anticonservative_and_powered():
    import scipy.stats as st
    from torchgenomics.postgwas._meta import meta_han_eskin

    rng = np.random.default_rng(0)
    m, K = 40000, 6
    se = torch.tensor(rng.uniform(0.02, 0.08, (m, K)), dtype=STAT)

    # NULL: beta_hat ~ N(0, se^2). p must NOT be anti-conservative (old code
    # gave frac<0.05 ~ 0.084). The asymptotic is conservative for small K.
    beta0 = torch.tensor(rng.standard_normal((m, K)), dtype=STAT) * se
    p0 = meta_han_eskin(beta0, se).p_meta.numpy()
    assert (p0 < 0.05).mean() <= 0.055
    assert (p0 < 0.01).mean() <= 0.012

    # POWER under a heterogeneous positive effect.
    rng2 = np.random.default_rng(1)
    beta_t = 0.15 + rng2.standard_normal((m, K)) * 0.10
    beta1 = torch.tensor(beta_t, dtype=STAT) + torch.tensor(rng2.standard_normal((m, K)), dtype=STAT) * se
    p1 = meta_han_eskin(beta1, se).p_meta.numpy()
    assert (p1 < 0.05).mean() > 0.9

    # The mixture survival function matches scipy exactly.
    S = torch.tensor([1.0, 4.0, 9.0], dtype=STAT)
    torch_sf = (0.5 * torch.special.gammaincc(torch.full_like(S, 0.5), S / 2)
                + 0.5 * torch.special.gammaincc(torch.ones_like(S), S / 2)).numpy()
    scipy_sf = 0.5 * st.chi2.sf(S.numpy(), 1) + 0.5 * st.chi2.sf(S.numpy(), 2)
    assert np.allclose(torch_sf, scipy_sf, atol=1e-12)


# ---------------------------------------------------------------------------
# 17. Harmonic-mean-p uses the Wilson (2019) Landau asymptotic, so the null is
#     calibrated (old linear form gave P(p<0.05)~0.14 at K=20) and the far tail
#     does not underflow. torchgenomics/postgwas/_combine.py
# ---------------------------------------------------------------------------
def test_harmonic_mean_p_landau_null_calibrated_and_tail():
    from torchgenomics.postgwas._combine import harmonic_mean_p

    rng = np.random.default_rng(3)
    pv = np.array([harmonic_mean_p(torch.tensor(rng.uniform(0, 1, 20), dtype=STAT))[1]
                   for _ in range(1500)])
    # Calibrated null (not the old ~0.14 over-rejection).
    assert (pv < 0.05).mean() == pytest.approx(0.05, abs=0.015)
    assert 0.45 < (pv < 0.5).mean() < 0.55

    # Far tail must not underflow to 0 for a highly significant combination.
    _, p_sig = harmonic_mean_p(torch.tensor([1e-100] + [0.5] * 9, dtype=STAT))
    assert 0 < p_sig < 1e-90
    # HMP is sensitive to the minimum: combined ~ O(k) * min-p, not floored.
    assert p_sig > 1e-100


# ---------------------------------------------------------------------------
# 18. General polyploid model is full-rank: k dummies (one reference class), so
#     the two homozygotes are distinguishable. torchgenomics/preprocess/polyploid.py
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ploidy", [2, 4, 6])
def test_general_model_full_rank_distinguishes_homozygotes(ploidy):
    from torchgenomics.preprocess.polyploid import recode_gene_action
    from torchgenomics.stats.best_model import _n_params

    doses = torch.arange(ploidy + 1, dtype=STAT).unsqueeze(0)  # [0..k], shape (1, k+1)
    enc = recode_gene_action(doses, "general", ploidy)  # (1, k+1, k)
    assert enc.shape == (1, ploidy + 1, ploidy)
    # nulliplex (dosage 0) is the all-zeros reference.
    assert torch.all(enc[0, 0] == 0)
    # the two homozygotes (dosage 0 and dosage k) must NOT be identical.
    assert not torch.equal(enc[0, 0], enc[0, ploidy])
    # each non-reference class activates exactly one distinct dummy.
    for d in range(1, ploidy + 1):
        assert enc[0, d].sum() == 1.0
    assert _n_params("general", ploidy) == ploidy


# ---------------------------------------------------------------------------
# 19. LD-module native paths gate on device: a CUDA tensor runs the torch body
#     instead of being force-copied to host for the C++ path (and pelt's
#     x.numpy() no longer crashes on CUDA). torchgenomics/ld/*.py
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ld_native_paths_device_gated():
    from torchgenomics.ld._changepoint import dp_changepoint
    from torchgenomics.ld._graph_utils import connected_components

    g = torch.Generator().manual_seed(0)
    sig = torch.rand(40, generator=g, dtype=STAT)
    # pelt: CUDA must not crash (previously x.numpy() on CUDA raised) and must
    # match the CPU result.
    assert dp_changepoint(sig, penalty=1.0) == dp_changepoint(sig.cuda(), penalty=1.0)

    A = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=STAT)
    assert connected_components(A) == connected_components(A.cuda())
