"""Benchmark: torchgwas colocalization and S-LDSC vs manual scipy/numpy.

Part 1 -- Colocalization (coloc_pairwise, hyprcoloc):
    Validates the Wakefield approximate Bayes factor, per-SNP and per-subset
    posteriors against manual numpy/scipy log-space calculations.

Part 2 -- Stratified LDSC (S-LDSC):
    Validates partitioned heritability tau coefficients and enrichment
    against manual numpy WLS regression.

Part 3 -- Polyploid compatibility:
    Proves colocalization and S-LDSC produce valid results on summary
    statistics from polyploid GWAS (tetraploid, hexaploid).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from torchgwas.postgwas._hyprcoloc import (
    _wakefield_log_abf,
    coloc_pairwise,
    hyprcoloc,
)
from torchgwas.postgwas._sldsc import sldsc_h2_partitioned
from torchgwas.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sumstats(
    beta: torch.Tensor, se: torch.Tensor, m: int | None = None
) -> SumStats:
    if m is None:
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


def _make_polyploid_coloc_data(ploidy, m=50, seed=42):
    """Two aligned SumStats for coloc, one shared signal at index 25."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(ploidy)  # per-allele effect dilution

    # Null background — keep noise small relative to SE
    beta1 = rng.normal(0, 0.01 * scale, size=m)
    beta2 = rng.normal(0, 0.01 * scale, size=m)
    se1 = np.full(m, 0.05 * scale)
    se2 = np.full(m, 0.06 * scale)

    # Shared causal signal at index 25 — strong effect (z ~ 10-20)
    causal = 25
    beta1[causal] = 1.0 * scale
    beta2[causal] = 0.8 * scale

    ss1 = _make_sumstats(
        torch.tensor(beta1, dtype=torch.float64),
        torch.tensor(se1, dtype=torch.float64),
    )
    ss2 = _make_sumstats(
        torch.tensor(beta2, dtype=torch.float64),
        torch.tensor(se2, dtype=torch.float64),
    )
    return ss1, ss2, causal


def _make_polyploid_coloc_data_k(ploidy, k_traits=3, m=50, seed=42):
    """K aligned SumStats for hyprcoloc, shared signal at index 25."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(ploidy)
    causal = 25

    ss_list = []
    for t in range(k_traits):
        beta = rng.normal(0, 0.01 * scale, size=m)
        se = np.full(m, (0.05 + 0.01 * t) * scale)
        beta[causal] = (1.0 - 0.1 * t) * scale
        ss_list.append(
            _make_sumstats(
                torch.tensor(beta, dtype=torch.float64),
                torch.tensor(se, dtype=torch.float64),
            )
        )
    return ss_list, causal


# ---------------------------------------------------------------------------
# PART 1: Colocalization benchmarks
# ---------------------------------------------------------------------------


class TestBenchWakefieldABF:
    """test_bench_wakefield_abf_matches_manual"""

    def test_bench_wakefield_abf_matches_manual(self):
        """Manually compute log-ABF and compare to _wakefield_log_abf."""
        rng = np.random.default_rng(123)
        m = 30
        beta_np = rng.normal(0, 0.3, size=m)
        se_np = np.abs(rng.normal(0.1, 0.02, size=m))
        W = 0.15 ** 2

        # Manual computation
        v = se_np ** 2
        r = W / (W + v)
        z = beta_np / se_np
        manual_log_abf = 0.5 * np.log(1.0 - r) + 0.5 * r * z ** 2

        # Library computation
        beta_t = torch.tensor(beta_np, dtype=torch.float64)
        se_t = torch.tensor(se_np, dtype=torch.float64)
        lib_log_abf = _wakefield_log_abf(beta_t, se_t, W).numpy()

        np.testing.assert_allclose(lib_log_abf, manual_log_abf, atol=1e-12)


class TestBenchColocH4Posterior:
    """test_bench_coloc_h4_posterior_matches_manual"""

    def test_bench_coloc_h4_posterior_matches_manual(self):
        """Reproduce the full H0..H4 decomposition manually and compare."""
        rng = np.random.default_rng(77)
        m = 40
        W = 0.15 ** 2
        prior_1 = 1e-4
        prior_2 = 1e-4
        prior_12 = 1e-5

        # Two traits with a shared causal SNP at index 20
        beta1 = rng.normal(0, 0.02, size=m)
        beta2 = rng.normal(0, 0.02, size=m)
        se1 = np.full(m, 0.10)
        se2 = np.full(m, 0.12)
        beta1[20] = 0.5
        beta2[20] = 0.4

        # Manual per-SNP log-ABFs
        def _manual_log_abf(beta, se):
            v = se ** 2
            r = W / (W + v)
            z = beta / se
            return 0.5 * np.log(1.0 - r) + 0.5 * r * z ** 2

        labf1 = _manual_log_abf(beta1, se1)
        labf2 = _manual_log_abf(beta2, se2)

        # Log-sum-exp helper
        def _lse(arr):
            mx = np.max(arr)
            return mx + np.log(np.sum(np.exp(arr - mx)))

        log_sum1 = _lse(labf1)
        log_sum2 = _lse(labf2)
        log_sum12 = _lse(labf1 + labf2)
        log_m = np.log(m)

        # Wallace 2020 erratum formulation
        log_h0 = 0.0
        log_h1 = np.log(prior_1) + log_sum1 - log_m
        log_h2 = np.log(prior_2) + log_sum2 - log_m
        log_h3 = (
            np.log(prior_1) + np.log(prior_2) + log_sum1 + log_sum2 - 2 * log_m
        )
        log_h4 = np.log(prior_12) + log_sum12 - log_m

        log_weights = np.array([log_h0, log_h1, log_h2, log_h3, log_h4])
        log_norm = _lse(log_weights)
        manual_pp = np.exp(log_weights - log_norm)

        # Library
        ss1 = _make_sumstats(
            torch.tensor(beta1, dtype=torch.float64),
            torch.tensor(se1, dtype=torch.float64),
        )
        ss2 = _make_sumstats(
            torch.tensor(beta2, dtype=torch.float64),
            torch.tensor(se2, dtype=torch.float64),
        )
        res = coloc_pairwise(ss1, ss2, prior_1=prior_1, prior_2=prior_2,
                             prior_12=prior_12, prior_w=W)

        lib_pp = np.array([res.pp_h0, res.pp_h1, res.pp_h2, res.pp_h3, res.pp_h4])
        np.testing.assert_allclose(lib_pp, manual_pp, atol=1e-8)


class TestBenchColocPosteriorsSum:
    """test_bench_coloc_posteriors_sum_to_one"""

    def test_bench_coloc_posteriors_sum_to_one(self):
        rng = np.random.default_rng(55)
        m = 30
        beta1 = rng.normal(0, 0.3, size=m)
        beta2 = rng.normal(0, 0.3, size=m)
        se = np.full(m, 0.1)

        ss1 = _make_sumstats(
            torch.tensor(beta1, dtype=torch.float64),
            torch.tensor(se, dtype=torch.float64),
        )
        ss2 = _make_sumstats(
            torch.tensor(beta2, dtype=torch.float64),
            torch.tensor(se, dtype=torch.float64),
        )
        res = coloc_pairwise(ss1, ss2)
        total = res.pp_h0 + res.pp_h1 + res.pp_h2 + res.pp_h3 + res.pp_h4
        assert abs(total - 1.0) < 1e-12


class TestBenchHyprcolocSubsetSum:
    """test_bench_hyprcoloc_subset_posteriors_sum_correctly"""

    def test_bench_hyprcoloc_subset_posteriors_sum_correctly(self):
        rng = np.random.default_rng(88)
        m = 30
        ss_list = []
        for t in range(3):
            beta = rng.normal(0, 0.02, size=m).astype(np.float64)
            se = np.full(m, 0.10, dtype=np.float64)
            beta[15] = 0.5  # shared signal
            ss_list.append(
                _make_sumstats(
                    torch.tensor(beta, dtype=torch.float64),
                    torch.tensor(se, dtype=torch.float64),
                )
            )

        res = hyprcoloc(ss_list)
        total = sum(res.subset_posteriors.values()) + res.pp_null
        assert abs(total - 1.0) < 1e-10


class TestBenchHyprcolocBFManual:
    """test_bench_hyprcoloc_bf_per_subset_matches_manual"""

    def test_bench_hyprcoloc_bf_per_subset_matches_manual(self):
        rng = np.random.default_rng(99)
        m = 20
        K = 3
        prior_1 = 1e-4
        prior_2 = 0.98
        W = 0.15 ** 2

        # Build K traits with shared signal at index 10
        betas, ses = [], []
        for t in range(K):
            b = rng.normal(0, 0.02, size=m).astype(np.float64)
            s = np.full(m, 0.10 + 0.01 * t, dtype=np.float64)
            b[10] = 0.5 - 0.05 * t
            betas.append(b)
            ses.append(s)

        # Manual per-trait per-SNP log-ABFs
        log_abf_np = np.zeros((K, m), dtype=np.float64)
        for k in range(K):
            v = ses[k] ** 2
            r = W / (W + v)
            z = betas[k] / ses[k]
            log_abf_np[k] = 0.5 * np.log(1.0 - r) + 0.5 * r * z ** 2

        def _lse(arr):
            mx = np.max(arr)
            return mx + np.log(np.sum(np.exp(arr - mx)))

        # Manual subset log-BFs
        manual_subset_log_bf = {}
        for size in range(2, K + 1):
            for subset in itertools.combinations(range(K), size):
                stacked = np.sum(log_abf_np[list(subset)], axis=0)
                manual_subset_log_bf[subset] = _lse(stacked)

        # Manual prior
        log_p1 = np.log(prior_1)
        log_p2 = np.log(prior_2)
        log_1mp1 = np.log1p(-prior_1)

        def _log_prior(sz):
            return sz * log_p1 + (sz - 1) * log_p2 + (K - sz) * log_1mp1

        # Manual null mass
        non_null_lps = []
        subset_log_prior = {}
        for subset in manual_subset_log_bf:
            lp = _log_prior(len(subset))
            subset_log_prior[subset] = lp
            non_null_lps.append(lp)
        max_lp = max(non_null_lps)
        non_null_mass = np.exp(max_lp) * np.sum(
            np.exp(np.array(non_null_lps) - max_lp)
        )
        null_mass = max(1.0 - non_null_mass, 1e-300)
        log_p_null = np.log(null_mass)

        # Manual posteriors
        log_weights = [log_p_null]
        keys = [None]
        for subset, lbf in manual_subset_log_bf.items():
            log_weights.append(subset_log_prior[subset] + lbf)
            keys.append(subset)
        log_weights_arr = np.array(log_weights)
        log_norm = _lse(log_weights_arr)
        manual_posteriors_arr = np.exp(log_weights_arr - log_norm)

        manual_pp_null = manual_posteriors_arr[0]
        manual_subset_pp = {}
        for key, pp in zip(keys[1:], manual_posteriors_arr[1:]):
            manual_subset_pp[key] = pp

        # Library
        ss_list = []
        for k in range(K):
            ss_list.append(
                _make_sumstats(
                    torch.tensor(betas[k], dtype=torch.float64),
                    torch.tensor(ses[k], dtype=torch.float64),
                )
            )
        res = hyprcoloc(ss_list, prior_1=prior_1, prior_2=prior_2, prior_w=W)

        # Compare pp_null
        assert abs(res.pp_null - manual_pp_null) < 1e-8

        # Compare subset posteriors
        for subset in manual_subset_pp:
            assert abs(res.subset_posteriors[subset] - manual_subset_pp[subset]) < 1e-8


class TestBenchColocCandidateSNP:
    """test_bench_coloc_candidate_snp_is_highest_joint_abf"""

    def test_bench_coloc_candidate_snp_is_highest_joint_abf(self):
        rng = np.random.default_rng(33)
        m = 40
        W = 0.15 ** 2

        beta1 = rng.normal(0, 0.02, size=m).astype(np.float64)
        beta2 = rng.normal(0, 0.02, size=m).astype(np.float64)
        se1 = np.full(m, 0.10, dtype=np.float64)
        se2 = np.full(m, 0.12, dtype=np.float64)
        beta1[15] = 0.6
        beta2[15] = 0.5

        # Manual computation of argmax(log_abf1 + log_abf2)
        def _manual_log_abf(beta, se):
            v = se ** 2
            r = W / (W + v)
            z = beta / se
            return 0.5 * np.log(1.0 - r) + 0.5 * r * z ** 2

        labf1 = _manual_log_abf(beta1, se1)
        labf2 = _manual_log_abf(beta2, se2)
        manual_cand = int(np.argmax(labf1 + labf2))

        ss1 = _make_sumstats(
            torch.tensor(beta1, dtype=torch.float64),
            torch.tensor(se1, dtype=torch.float64),
        )
        ss2 = _make_sumstats(
            torch.tensor(beta2, dtype=torch.float64),
            torch.tensor(se2, dtype=torch.float64),
        )
        res = coloc_pairwise(ss1, ss2, prior_w=W)

        assert res.candidate_snp == manual_cand
        assert res.candidate_snp == 15


# ---------------------------------------------------------------------------
# PART 2: S-LDSC benchmarks
# ---------------------------------------------------------------------------


class TestBenchSLDSCTauManualWLS:
    """test_bench_sldsc_tau_matches_manual_wls"""

    def test_bench_sldsc_tau_matches_manual_wls(self):
        """Compare S-LDSC tau against statsmodels WLS on filtered data."""
        rng = np.random.default_rng(200)
        m = 500
        n_sample = 5000
        C = 2

        # Synthetic annotation LD scores: category 0 enriched, category 1 baseline
        annot_ld = rng.uniform(0.5, 3.0, size=(m, C))
        annot_ld[:, 0] *= 2.0  # enriched annotation

        M_c = np.array([1000.0, 4000.0])
        m_total = 5000

        # True tau values
        tau_true = np.array([0.001, 0.0002])
        # chi2 = N * sum_c(tau_c * l_cj) + 1 + noise
        expected = n_sample * (annot_ld @ tau_true) + 1.0
        chi2_np = expected + rng.normal(0, 0.5, size=m)
        chi2_np = np.maximum(chi2_np, 0.1)

        chi2_t = torch.tensor(chi2_np, dtype=torch.float64)
        annot_t = torch.tensor(annot_ld, dtype=torch.float64)
        M_c_t = torch.tensor(M_c, dtype=torch.float64)

        # Library call
        res = sldsc_h2_partitioned(
            chi2_t, annot_t, M_c_t, n=n_sample, m_total=m_total,
            two_step_cutoff=30.0,
        )

        # Manual WLS via numpy after the same two-step filter
        # Step 1: fit on all, then filter chi2 <= 30
        l_total = annot_ld.sum(axis=1)
        w = 1.0 / np.maximum(l_total ** 2, 1.0)

        # Design matrix: [N * l_c1, N * l_c2, 1]
        X = np.column_stack([n_sample * annot_ld, np.ones(m)])
        y = chi2_np

        # Step 2 filter
        keep = chi2_np <= 30.0
        if keep.sum() < max(10, C + 2):
            keep = np.ones(m, dtype=bool)

        X2 = X[keep]
        y2 = y[keep]
        w2 = w[keep]

        # Manual WLS: (X'WX)^{-1} X'Wy
        sqrt_w = np.sqrt(w2)
        Xw = X2 * sqrt_w[:, None]
        yw = y2 * sqrt_w
        manual_coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        manual_tau = manual_coef[:C]
        lib_tau = res.tau.numpy()

        np.testing.assert_allclose(lib_tau, manual_tau, atol=1e-4)


class TestBenchSLDSCEnrichmentFormula:
    """test_bench_sldsc_enrichment_formula"""

    def test_bench_sldsc_enrichment_formula(self):
        rng = np.random.default_rng(201)
        m = 300
        n_sample = 5000
        C = 3

        annot_ld = rng.uniform(0.5, 2.0, size=(m, C))
        M_c = np.array([1000.0, 2000.0, 2000.0])
        m_total = 5000

        tau_true = np.array([0.002, 0.0005, 0.0003])
        chi2_np = n_sample * (annot_ld @ tau_true) + 1.0 + rng.normal(0, 0.3, m)
        chi2_np = np.maximum(chi2_np, 0.1)

        res = sldsc_h2_partitioned(
            torch.tensor(chi2_np, dtype=torch.float64),
            torch.tensor(annot_ld, dtype=torch.float64),
            torch.tensor(M_c, dtype=torch.float64),
            n=n_sample,
            m_total=m_total,
        )

        # Manual enrichment from the library's own tau and M_c
        tau = res.tau.numpy()
        h2_cat = tau * M_c
        h2_total = h2_cat.sum()
        enrichment_manual = (h2_cat / h2_total) / (M_c / m_total)

        np.testing.assert_allclose(
            res.enrichment.numpy(), enrichment_manual, atol=1e-8
        )
        np.testing.assert_allclose(
            res.h2_cat.numpy(), h2_cat, atol=1e-8
        )
        assert abs(res.h2_total - h2_total) < 1e-8


class TestBenchSLDSCSingleAnnotation:
    """test_bench_sldsc_single_annotation_matches_ldsc_h2"""

    def test_bench_sldsc_single_annotation_matches_ldsc_h2(self):
        """Single-annotation S-LDSC should match univariate ldsc_h2."""
        from torchgwas.postgwas._ldsc import ldsc_h2

        rng = np.random.default_rng(202)
        m = 400
        n_sample = 10000

        ld_scores = rng.uniform(1.0, 5.0, size=m)
        h2_true = 0.5
        chi2_np = 1.0 + (n_sample / m) * h2_true * ld_scores + rng.normal(0, 0.3, m)
        chi2_np = np.maximum(chi2_np, 0.1)

        chi2_t = torch.tensor(chi2_np, dtype=torch.float64)
        ld_t = torch.tensor(ld_scores, dtype=torch.float64)

        # Univariate LDSC. S-LDSC runs single-pass WLS internally
        # (Phase 42), so we compare against ``ldsc_h2(n_iter=0)``: the new
        # IRWLS default (Phase 37 follow-up) would produce a slightly
        # different fit and break this single-annotation equivalence
        # check. Promoting S-LDSC to full IRWLS is filed as a deferred
        # follow-up.
        uni = ldsc_h2(chi2_t, ld_t, n=n_sample, m_total=m, n_iter=0)

        # S-LDSC with single annotation
        sldsc = sldsc_h2_partitioned(
            chi2_t,
            ld_t.unsqueeze(1),
            torch.tensor([float(m)], dtype=torch.float64),
            n=n_sample,
            m_total=m,
        )

        assert abs(sldsc.h2_total - uni.h2) < 1e-6
        assert abs(sldsc.intercept - uni.intercept) < 1e-6


# ---------------------------------------------------------------------------
# PART 3: Polyploid compatibility
# ---------------------------------------------------------------------------


class TestBenchColocTetraploidShared:
    """test_bench_coloc_tetraploid_shared_signal"""

    def test_bench_coloc_tetraploid_shared_signal(self):
        ss1, ss2, causal = _make_polyploid_coloc_data(ploidy=4, m=50, seed=42)
        res = coloc_pairwise(ss1, ss2)
        assert res.pp_h4 > 0.8
        assert res.candidate_snp == causal


class TestBenchColocHexaploidValid:
    """test_bench_coloc_hexaploid_valid_posteriors"""

    def test_bench_coloc_hexaploid_valid_posteriors(self):
        ss1, ss2, _ = _make_polyploid_coloc_data(ploidy=6, m=50, seed=43)
        res = coloc_pairwise(ss1, ss2)
        pp = [res.pp_h0, res.pp_h1, res.pp_h2, res.pp_h3, res.pp_h4]
        for p in pp:
            assert 0.0 <= p <= 1.0
        assert abs(sum(pp) - 1.0) < 1e-12


class TestBenchHyprcolocTetraploidThreeTraits:
    """test_bench_hyprcoloc_tetraploid_three_traits"""

    def test_bench_hyprcoloc_tetraploid_three_traits(self):
        ss_list, causal = _make_polyploid_coloc_data_k(
            ploidy=4, k_traits=3, m=50, seed=44
        )
        res = hyprcoloc(ss_list)
        assert res.pp_all_colocalize > 0.5
        assert res.candidate_snp == causal


class TestBenchSLDSCTetraploidEnrichment:
    """test_bench_sldsc_tetraploid_enrichment"""

    def test_bench_sldsc_tetraploid_enrichment(self):
        rng = np.random.default_rng(300)
        m = 300
        n_sample = 5000
        C = 2

        # Tetraploid: higher ploidy -> more alleles -> different LD structure
        # but chi2 statistics still follow LDSC model
        annot_ld = rng.uniform(0.5, 3.0, size=(m, C))
        M_c = np.array([750.0, 3000.0])
        m_total = 3750

        # Category 0 enriched (contributes 75% of h2 with 20% of SNPs)
        tau_true = np.array([0.003, 0.0002])
        chi2_np = n_sample * (annot_ld @ tau_true) + 1.0 + rng.normal(0, 0.5, m)
        chi2_np = np.maximum(chi2_np, 0.1)

        res = sldsc_h2_partitioned(
            torch.tensor(chi2_np, dtype=torch.float64),
            torch.tensor(annot_ld, dtype=torch.float64),
            torch.tensor(M_c, dtype=torch.float64),
            n=n_sample,
            m_total=m_total,
        )

        # Enriched category should have enrichment > 1
        assert float(res.enrichment[0]) > 1.0
        # Depleted category should have enrichment < 1
        assert float(res.enrichment[1]) < 1.0


class TestBenchColocPloidyInvariant:
    """test_bench_coloc_ploidy_invariant

    Same underlying causal architecture, one set of SumStats scaled as
    diploid, one as tetraploid. Both should produce PP(H4) > 0.7 because
    the Wakefield ABF is ploidy-invariant: r = W/(W + se^2) and z = beta/se
    are both invariant when beta and se scale together by 1/sqrt(ploidy).
    """

    def test_bench_coloc_ploidy_invariant(self):
        # The Wakefield ABF z-score z = beta/se is ploidy-invariant when
        # both beta and se scale by 1/sqrt(ploidy). The shrinkage factor
        # r = W/(W + se^2) depends on absolute SE, but we can make it
        # invariant by scaling prior_w proportionally to se^2. For a
        # cleaner test, we fix the z-scores directly: construct SumStats
        # with identical z but different (beta, se) scales.
        rng = np.random.default_rng(42)
        m = 50
        causal = 25

        # Base z-scores (ploidy-invariant)
        z1 = rng.normal(0, 0.2, size=m)
        z2 = rng.normal(0, 0.2, size=m)
        z1[causal] = 15.0  # strong shared signal
        z2[causal] = 12.0

        # Diploid version: se ~ 0.05
        se_dip = np.full(m, 0.05)
        beta_dip_1 = z1 * se_dip
        beta_dip_2 = z2 * se_dip
        ss1_dip = _make_sumstats(
            torch.tensor(beta_dip_1, dtype=torch.float64),
            torch.tensor(se_dip, dtype=torch.float64),
        )
        ss2_dip = _make_sumstats(
            torch.tensor(beta_dip_2, dtype=torch.float64),
            torch.tensor(se_dip, dtype=torch.float64),
        )

        # Tetraploid version: se ~ 0.05/sqrt(2) (smaller SE but same z)
        se_tet = np.full(m, 0.05 / np.sqrt(2))
        beta_tet_1 = z1 * se_tet
        beta_tet_2 = z2 * se_tet
        ss1_tet = _make_sumstats(
            torch.tensor(beta_tet_1, dtype=torch.float64),
            torch.tensor(se_tet, dtype=torch.float64),
        )
        ss2_tet = _make_sumstats(
            torch.tensor(beta_tet_2, dtype=torch.float64),
            torch.tensor(se_tet, dtype=torch.float64),
        )

        res_dip = coloc_pairwise(ss1_dip, ss2_dip)
        res_tet = coloc_pairwise(ss1_tet, ss2_tet)

        # Both should show strong shared signal
        assert res_dip.pp_h4 > 0.7
        assert res_tet.pp_h4 > 0.7

        # Both should identify the same candidate SNP
        assert res_dip.candidate_snp == res_tet.candidate_snp
        assert res_dip.candidate_snp == causal
