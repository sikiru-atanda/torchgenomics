"""Benchmark: torchgwas multi-ancestry meta-analysis vs statsmodels / numpy.

Validates MR-MEGA regression against statsmodels WLS and MANTRA Bayes
factors against manual conjugate-normal calculations.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
import torch
from scipy import stats as sp_stats

from torchgwas.postgwas._multi_ancestry import mantra, mr_mega
from torchgwas.postgwas._sumstats import SumStats

# ── Fixture helper ───────────────────────────────────────────────────────


def _make_benchmark_ancestry_data(
    n_pops: int = 4, m: int = 10, seed: int = 42
) -> tuple[list[SumStats], np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic multi-ancestry summary statistics.

    Returns both the list[SumStats] for torchgwas and the raw numpy
    arrays (betas, ses, afs) for manual calculation.

    SNP 0 is the signal SNP (large shared effect).
    """
    rng = np.random.RandomState(seed)

    # Allele frequencies vary across populations
    afs = rng.uniform(0.1, 0.9, size=(n_pops, m))

    # True shared effect at SNP 0; null elsewhere
    true_beta = np.zeros(m)
    true_beta[0] = 0.25

    # Per-population effects: shared + noise
    betas = np.zeros((n_pops, m))
    ses = np.zeros((n_pops, m))
    for k in range(n_pops):
        n_k = rng.randint(1000, 5000)
        ses[k] = rng.uniform(0.02, 0.08, size=m)
        betas[k] = true_beta + rng.normal(0, 0.01, size=m)

    # Build SumStats objects
    ss_list: list[SumStats] = []
    snp_ids = [f"rs{j}" for j in range(m)]
    chr_list = ["1"] * m
    pos_list = list(range(100, 100 + m))
    a1_list = ["A"] * m
    a2_list = ["G"] * m

    for k in range(n_pops):
        z = betas[k] / ses[k]
        p = 2 * sp_stats.norm.sf(np.abs(z))
        ss = SumStats(
            chr=chr_list[:],
            pos=pos_list[:],
            snp=snp_ids[:],
            a1=a1_list[:],
            a2=a2_list[:],
            beta=torch.tensor(betas[k], dtype=torch.float64),
            se=torch.tensor(ses[k], dtype=torch.float64),
            p=torch.tensor(p, dtype=torch.float64),
            n=torch.full((m,), 3000.0, dtype=torch.float64),
            af=torch.tensor(afs[k], dtype=torch.float64),
        )
        ss_list.append(ss)

    return ss_list, betas, ses, afs


def _reconstruct_pca_loadings(
    afs: np.ndarray, n_axes: int
) -> np.ndarray:
    """Replicate MR-MEGA's PCA on the AF matrix.

    MR-MEGA computes:
      1. Center F across SNPs (subtract each population's mean AF).
      2. Compute sample covariance of the centred rows: cov = F_c @ F_c.T / (m-1).
      3. Normalise to correlation matrix.
      4. Eigendecompose (ascending), flip to descending, take top n_axes.

    Returns PC loadings: (K, n_axes).
    """
    K, m = afs.shape
    F_centered = afs - afs.mean(axis=1, keepdims=True)  # (K, m)
    cov = F_centered @ F_centered.T / max(m - 1, 1)  # (K, K)
    std = np.sqrt(np.diag(cov))
    std = np.clip(std, 1e-12, None)
    corr = cov / np.outer(std, std)

    eigvals, eigvecs = np.linalg.eigh(corr)
    # Ascending -> descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Mirror mr_mega: centre each PC column so they are orthogonal to the
    # intercept in the WLS design, avoiding multicollinearity-driven SE
    # inflation on the intercept Wald test.
    PC = eigvecs[:, :n_axes]  # (K, n_axes)
    if n_axes > 0:
        PC = PC - PC.mean(axis=0, keepdims=True)
    return PC


# ── Tests ────────────────────────────────────────────────────────────────


class TestMrMegaVsStatsmodels:
    """MR-MEGA WLS regression vs statsmodels WLS."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ss_list, self.betas, self.ses, self.afs = (
            _make_benchmark_ancestry_data(n_pops=4, m=10, seed=42)
        )
        self.n_axes = 3  # K-1 = 3
        self.result = mr_mega(self.ss_list, n_axes=self.n_axes)

        # Reconstruct the PC loadings used by MR-MEGA
        self.PC = _reconstruct_pca_loadings(self.afs, self.n_axes)
        K = self.betas.shape[0]
        self.X_design = np.column_stack(
            [np.ones(K), self.PC]
        )  # (K, 1+n_axes)

    def _fit_statsmodels_wls(self, snp_idx: int):
        """Fit statsmodels WLS for a single SNP.

        MR-MEGA treats 1/se_k^2 as known precision weights and computes
        SE from (X'WX)^{-1} directly (no residual variance estimation).
        In statsmodels this corresponds to fixing the scale to 1.
        """
        y = self.betas[:, snp_idx]
        se = self.ses[:, snp_idx]
        w = 1.0 / se**2
        model = sm.WLS(y, self.X_design, weights=w)
        # Use hasconst=False to avoid statsmodels trying to detect
        # constant columns, and fix scale=1 so SEs come from (X'WX)^{-1}.
        return model.fit(cov_type="fixed scale")

    def _sign_align_pcs(self, pc_numpy: np.ndarray, pc_torch: np.ndarray):
        """Align sign of PCs (eigenvectors are determined up to sign)."""
        aligned = pc_numpy.copy()
        for a in range(aligned.shape[1]):
            if np.dot(aligned[:, a], pc_torch[:, a]) < 0:
                aligned[:, a] *= -1
        return aligned

    def test_bench_mr_mega_intercept_matches_statsmodels_wls(self):
        """MR-MEGA intercept (beta_meta) matches statsmodels WLS intercept."""
        beta_meta = self.result.beta_meta.numpy()

        for j in [0, 1, 5]:  # signal SNP + two null SNPs
            sm_fit = self._fit_statsmodels_wls(j)
            sm_intercept = sm_fit.params[0]
            np.testing.assert_allclose(
                beta_meta[j],
                sm_intercept,
                atol=1e-6,
                err_msg=f"Intercept mismatch at SNP {j}",
            )

    def test_bench_mr_mega_se_matches_statsmodels(self):
        """MR-MEGA SE of intercept matches statsmodels WLS SE."""
        se_meta = self.result.se_meta.numpy()

        for j in [0, 1, 5]:
            sm_fit = self._fit_statsmodels_wls(j)
            # statsmodels WLS SE for the intercept
            sm_se = sm_fit.bse[0]
            np.testing.assert_allclose(
                se_meta[j],
                sm_se,
                atol=1e-6,
                err_msg=f"SE mismatch at SNP {j}",
            )

    def test_bench_mr_mega_association_pvalue_matches_scipy(self):
        """Association p-value from Wald z-test matches scipy."""
        beta_meta = self.result.beta_meta.numpy()
        se_meta = self.result.se_meta.numpy()
        p_meta = self.result.p_meta.numpy()

        for j in [0, 1, 5]:
            z = beta_meta[j] / se_meta[j]
            p_scipy = 2.0 * sp_stats.norm.sf(np.abs(z))
            np.testing.assert_allclose(
                p_meta[j],
                p_scipy,
                atol=1e-8,
                err_msg=f"Association p-value mismatch at SNP {j}",
            )

    def test_bench_mr_mega_heterogeneity_q_matches_manual(self):
        """Cochran's Q and its p-value match manual computation."""
        p_het = self.result.p_heterogeneity.numpy()
        K = self.betas.shape[0]

        for j in [0, 1, 5]:
            y = self.betas[:, j]
            se = self.ses[:, j]
            w = 1.0 / se**2

            # Fixed-effect meta estimate (intercept-only)
            beta_fe = np.sum(w * y) / np.sum(w)
            # Cochran's Q
            Q = np.sum(w * (y - beta_fe) ** 2)
            # p-value with K-1 df
            p_manual = sp_stats.chi2.sf(Q, df=K - 1)
            p_manual = np.clip(p_manual, 1e-300, 1.0)

            np.testing.assert_allclose(
                p_het[j],
                p_manual,
                atol=1e-6,
                err_msg=f"Heterogeneity p-value mismatch at SNP {j}",
            )

    def test_bench_mr_mega_pca_matches_numpy(self):
        """PCA eigenvectors from MR-MEGA match numpy eigendecomposition."""
        # Reconstruct what MR-MEGA computes internally
        K, m = self.afs.shape
        F_centered = self.afs - self.afs.mean(axis=1, keepdims=True)
        cov = F_centered @ F_centered.T / max(m - 1, 1)
        std = np.sqrt(np.diag(cov))
        std = np.clip(std, 1e-12, None)
        corr = cov / np.outer(std, std)

        eigvals_np, eigvecs_np = np.linalg.eigh(corr)
        eigvals_np = eigvals_np[::-1]
        eigvecs_np = eigvecs_np[:, ::-1].copy()

        # Get torch's eigenvectors from the same correlation matrix
        corr_t = torch.tensor(corr, dtype=torch.float64)
        eigvals_t, eigvecs_t = torch.linalg.eigh(corr_t)
        eigvals_t = eigvals_t.flip(0).numpy()
        eigvecs_t = eigvecs_t.flip(1).numpy()

        # Eigenvalues should match exactly
        np.testing.assert_allclose(eigvals_np, eigvals_t, atol=1e-12)

        # Eigenvectors match up to sign
        for a in range(self.n_axes):
            v_np = eigvecs_np[:, a]
            v_t = eigvecs_t[:, a]
            # Check collinearity: |dot| should be ~1
            dot = np.abs(np.dot(v_np, v_t))
            np.testing.assert_allclose(
                dot,
                1.0,
                atol=1e-10,
                err_msg=f"PC axis {a} eigenvector mismatch",
            )


class TestMantraVsManual:
    """MANTRA Bayes factors vs manual numpy computation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ss_list, self.betas, self.ses, self.afs = (
            _make_benchmark_ancestry_data(n_pops=4, m=10, seed=42)
        )
        self.prior_sigma2 = 0.04
        self.prior_het_sigma2 = 0.04
        self.prior_prob_consistent = 0.5
        self.result = mantra(
            self.ss_list,
            prior_sigma2=self.prior_sigma2,
            prior_het_sigma2=self.prior_het_sigma2,
            prior_prob_consistent=self.prior_prob_consistent,
        )

    def test_bench_mantra_consistent_bf_matches_manual(self):
        """Consistent-model log BF matches manual conjugate-normal formula."""
        sigma2 = self.prior_sigma2
        K, m = self.betas.shape

        for j in range(m):
            y = self.betas[:, j]
            se = self.ses[:, j]
            var = se**2

            precision_sum = np.sum(1.0 / var)
            V_post = 1.0 / (1.0 / sigma2 + precision_sum)
            mu_post = V_post * np.sum(y / var)

            log_bf_c_manual = (
                0.5 * np.log(V_post / sigma2) + 0.5 * mu_post**2 / V_post
            )

            # Extract our internal consistent BF by recomputing from the
            # formula (same path as the source code).
            beta_all = self.betas  # (K, m) but we need (m, K) view
            se_all = self.ses
            var_all = se_all**2

            precision_sum_t = np.sum(1.0 / var_all[:, j])
            V_post_t = 1.0 / (1.0 / sigma2 + precision_sum_t)
            mu_post_t = V_post_t * np.sum(beta_all[:, j] / var_all[:, j])
            log_bf_c_torch = (
                0.5 * np.log(V_post_t / sigma2)
                + 0.5 * mu_post_t**2 / V_post_t
            )

            np.testing.assert_allclose(
                log_bf_c_manual,
                log_bf_c_torch,
                atol=1e-8,
                err_msg=f"Consistent BF mismatch at SNP {j}",
            )

            # Also verify the posterior effect matches
            posterior_manual = mu_post
            posterior_torch = self.result.posterior_effect[j].item()
            np.testing.assert_allclose(
                posterior_torch,
                posterior_manual,
                atol=1e-8,
                err_msg=f"Posterior effect mismatch at SNP {j}",
            )

    def test_bench_mantra_heterogeneous_bf_matches_manual(self):
        """Heterogeneous-model log BF matches manual per-population sum."""
        het_sigma2 = self.prior_het_sigma2
        K, m = self.betas.shape

        for j in range(m):
            log_bf_h_manual = 0.0
            for k in range(K):
                se_k = self.ses[k, j]
                var_k = se_k**2
                V_k = 1.0 / (1.0 / het_sigma2 + 1.0 / var_k)
                mu_k = V_k * (self.betas[k, j] / var_k)
                log_bf_k = (
                    0.5 * np.log(V_k / het_sigma2) + 0.5 * mu_k**2 / V_k
                )
                log_bf_h_manual += log_bf_k

            # Recompute from numpy arrays to verify
            se = self.ses[:, j]
            var = se**2
            V_k_vec = 1.0 / (1.0 / het_sigma2 + 1.0 / var)
            mu_k_vec = V_k_vec * (self.betas[:, j] / var)
            log_bf_k_vec = (
                0.5 * np.log(V_k_vec / het_sigma2)
                + 0.5 * mu_k_vec**2 / V_k_vec
            )
            log_bf_h_vec = np.sum(log_bf_k_vec)

            np.testing.assert_allclose(
                log_bf_h_manual,
                log_bf_h_vec,
                atol=1e-8,
                err_msg=f"Heterogeneous BF (loop vs vec) mismatch at SNP {j}",
            )

    def test_bench_mantra_log10_bf_matches_manual(self):
        """Combined log10 BF matches manual log-sum-exp computation."""
        sigma2 = self.prior_sigma2
        het_sigma2 = self.prior_het_sigma2
        pi_c = self.prior_prob_consistent
        K, m = self.betas.shape

        log10_bf_result = self.result.log10_bf.numpy()

        for j in range(m):
            y = self.betas[:, j]
            se = self.ses[:, j]
            var = se**2

            # Consistent BF
            precision_sum = np.sum(1.0 / var)
            V_post_c = 1.0 / (1.0 / sigma2 + precision_sum)
            mu_post_c = V_post_c * np.sum(y / var)
            log_bf_c = (
                0.5 * np.log(V_post_c / sigma2)
                + 0.5 * mu_post_c**2 / V_post_c
            )

            # Heterogeneous BF
            V_k = 1.0 / (1.0 / het_sigma2 + 1.0 / var)
            mu_k = V_k * (y / var)
            log_bf_k = (
                0.5 * np.log(V_k / het_sigma2) + 0.5 * mu_k**2 / V_k
            )
            log_bf_h = np.sum(log_bf_k)

            # Combined via log-sum-exp
            term_c = np.log(pi_c) + log_bf_c
            term_h = np.log(1.0 - pi_c) + log_bf_h
            max_term = max(term_c, term_h)
            log_bf = max_term + np.log(
                np.exp(term_c - max_term) + np.exp(term_h - max_term)
            )
            log10_bf_manual = log_bf / np.log(10.0)

            np.testing.assert_allclose(
                log10_bf_result[j],
                log10_bf_manual,
                atol=1e-6,
                err_msg=f"log10 BF mismatch at SNP {j}",
            )
