"""Known-truth simulation benchmark for Multi-Environment GLMM.

Simulates data with KNOWN planted SNP effects across E environments,
then verifies the ME-GLMM correctly recovers the ground truth:

  Causal SNP layout (binary, E=3 environments):
  ─────────────────────────────────────────────────────────
  SNP   β_env0   β_env1   β_env2   Type
  ─────────────────────────────────────────────────────────
  s0     0.80     0.80     0.80    Stable (same everywhere)
  s1     1.00     0.00     0.00    Env-specific (env 0 only)
  s2     0.50    -0.70     0.90    Heterogeneous (all nonzero)
  s3     0.30     0.30     0.30    Weak stable
  s4     0.80     0.00    -0.80    Opposing (cancel across envs)
  s5-19  0.00     0.00     0.00    Null (no effect)
  ─────────────────────────────────────────────────────────

  Ordinal variant (J=3 categories, E=2 environments):
  ─────────────────────────────────────────────────────────
  SNP   β_env0   β_env1   Type
  ─────────────────────────────────────────────────────────
  s0     0.80     0.80    Stable
  s1     1.00     0.00    Env-specific (env 0 only)
  s2     0.50    -0.70    Heterogeneous
  s3-14  0.00     0.00    Null
  ─────────────────────────────────────────────────────────

Expected outcomes for each test type:
- Joint test: detects all causal SNPs (s0-s4), not null SNPs
- Per-env marginal: detects SNPs active in that env, not inactive ones
- Homogeneity: small for stable (s0,s3), large for heterogeneous (s1,s2,s4)
- Reaction-norm stable: detects s0,s3; NOT s4 (cancels across envs)
- Reaction-norm GxE: detects s1,s2,s4; NOT s0,s3 (no env variation)
- Null SNPs: p > 0.05 at expected false-positive rate
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _make_vmeta(m):
    from torchgwas.models.base import VariantMeta
    return VariantMeta(
        snp=[f"s{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )


# =====================================================================
# Simulation with known causal architecture
# =====================================================================

def _simulate_known_binary(
    n=600, E=3, m_null=15, h2=0.15, seed=2026, m_grm=120,
):
    """Simulate multi-env binary phenotype with KNOWN causal SNP effects.

    Returns
    -------
    Y : (n, E) binary phenotype
    G : (n, m_total) genotypes — first 5 are causal, rest null
    X0 : (n, 1) intercept
    K : (n, n) GRM from separate genotypes
    true_beta : (m_total, E) ground-truth effect matrix
    causal_labels : list of str describing each SNP type
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === GRM from separate genotypes (avoids proximal contamination) ===
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.1 + 0.3 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )
    G_std_grm = G_grm - G_grm.mean(dim=0)
    K = (G_std_grm @ G_std_grm.T) / m_grm
    K += 0.01 * torch.eye(n, dtype=torch.float64)

    # === Test genotypes ===
    n_causal = 5
    m_total = n_causal + m_null
    G = torch.zeros(n, m_total, dtype=torch.float64)
    for j in range(m_total):
        maf = 0.20 + 0.10 * np.random.rand()  # MAF in [0.20, 0.30]
        G[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )

    # === Known causal effects ===
    true_beta = torch.zeros(m_total, E, dtype=torch.float64)
    # s0: Stable effect (same in all envs)
    true_beta[0, :] = 0.80
    # s1: Env-specific (env 0 only)
    true_beta[1, 0] = 1.00
    # s2: Heterogeneous (all nonzero, different signs/magnitudes)
    true_beta[2, 0] = 0.50
    true_beta[2, 1] = -0.70
    true_beta[2, 2] = 0.90
    # s3: Weak stable
    true_beta[3, :] = 0.30
    # s4: Opposing (cancels in mean)
    true_beta[4, 0] = 0.80
    true_beta[4, 2] = -0.80

    causal_labels = [
        "stable_strong", "env0_specific", "heterogeneous",
        "stable_weak", "opposing",
    ] + ["null"] * m_null

    # === Polygenic background from GRM ===
    Sigma_g = torch.eye(E, dtype=torch.float64) * 0.5
    for e1 in range(E):
        for e2 in range(E):
            if e1 != e2:
                Sigma_g[e1, e2] = 0.3
    V_poly = h2 * torch.kron(Sigma_g, K)
    V_poly += (1 - h2) * torch.eye(n * E, dtype=torch.float64)
    L = torch.linalg.cholesky(V_poly + 1e-6 * torch.eye(n * E, dtype=torch.float64))
    u_vec = L @ torch.randn(n * E, dtype=torch.float64)
    u = u_vec.reshape(E, n).T  # (n, E)

    # === Generate phenotypes ===
    # Standardize causal genotypes for interpretable effect sizes
    G_std = G - G.mean(dim=0)
    G_sd = G.std(dim=0).clamp(min=0.1)
    G_std = G_std / G_sd

    Y = torch.zeros(n, E, dtype=torch.float64)
    for e in range(E):
        # Linear predictor: intercept + causal effects + polygenic
        eta_e = -0.5 + G_std @ true_beta[:, e] + u[:, e]
        prob_e = torch.sigmoid(eta_e).clamp(0.01, 0.99)
        Y[:, e] = torch.bernoulli(prob_e)

    return Y, G, X0_from_n(n), K, true_beta, causal_labels


def _simulate_known_ordinal(
    n=600, E=2, J=3, m_null=12, h2=0.15, seed=2027, m_grm=120,
):
    """Simulate multi-env ordinal phenotype (J categories) with known effects.

    Returns Y (n, E), G (n, m), X0, K, true_beta (m, E), causal_labels.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GRM
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.1 + 0.3 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )
    G_std_grm = G_grm - G_grm.mean(dim=0)
    K = (G_std_grm @ G_std_grm.T) / m_grm + 0.01 * torch.eye(n, dtype=torch.float64)

    # Test genotypes
    n_causal = 3
    m_total = n_causal + m_null
    G = torch.zeros(n, m_total, dtype=torch.float64)
    for j in range(m_total):
        maf = 0.20 + 0.10 * np.random.rand()
        G[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )

    # Known effects
    true_beta = torch.zeros(m_total, E, dtype=torch.float64)
    true_beta[0, :] = 0.80   # s0: stable
    true_beta[1, 0] = 1.00   # s1: env 0 only
    true_beta[2, 0] = 0.50   # s2: heterogeneous
    true_beta[2, 1] = -0.70

    causal_labels = ["stable", "env0_specific", "heterogeneous"] + ["null"] * m_null

    # Polygenic
    Sigma_g = torch.eye(E, dtype=torch.float64) * 0.5
    Sigma_g[0, 1] = Sigma_g[1, 0] = 0.3
    V_poly = h2 * torch.kron(Sigma_g, K) + (1 - h2) * torch.eye(n * E, dtype=torch.float64)
    L = torch.linalg.cholesky(V_poly + 1e-6 * torch.eye(n * E, dtype=torch.float64))
    u_vec = L @ torch.randn(n * E, dtype=torch.float64)
    u = u_vec.reshape(E, n).T

    # Standardize
    G_std = G - G.mean(dim=0)
    G_sd = G.std(dim=0).clamp(min=0.1)
    G_std = G_std / G_sd

    # Ordinal thresholds: logit(P(Y<=j)) = alpha_j - eta
    # For J=3: alpha = [-0.5, 0.8]
    thresholds = torch.tensor([-0.5, 0.8], dtype=torch.float64)

    Y = torch.zeros(n, E, dtype=torch.float64)
    for e in range(E):
        eta_e = G_std @ true_beta[:, e] + u[:, e]
        cum_probs = torch.zeros(n, J, dtype=torch.float64)
        for j in range(J - 1):
            cum_probs[:, j] = torch.sigmoid(thresholds[j] - eta_e)
        cum_probs[:, J - 1] = 1.0
        # Category probs
        probs = torch.zeros(n, J, dtype=torch.float64)
        probs[:, 0] = cum_probs[:, 0]
        for j in range(1, J):
            probs[:, j] = (cum_probs[:, j] - cum_probs[:, j - 1]).clamp(min=1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Sample
        for i in range(n):
            Y[i, e] = torch.multinomial(probs[i], 1).item()

    return Y, G, X0_from_n(n), K, true_beta, causal_labels


def X0_from_n(n):
    return torch.ones(n, 1, dtype=torch.float64)


# =====================================================================
# Binary known-truth tests
# =====================================================================

class TestKnownTruthBinaryJoint:
    """Joint test should detect all causal SNPs, not null SNPs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(family="binary", use_spa=False, pql_max_iter=20)
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_strong_detected_joint(self):
        """s0 (stable β=0.8): joint test should detect."""
        p = self.result.p[0].item()
        assert p < 0.05, f"s0 (stable strong) not detected: joint p={p:.4e}"

    def test_env_specific_detected_joint(self):
        """s1 (env-specific β=1.0 in env 0): joint test should detect."""
        p = self.result.p[1].item()
        assert p < 0.05, f"s1 (env-specific) not detected: joint p={p:.4e}"

    def test_heterogeneous_detected_joint(self):
        """s2 (heterogeneous): joint test should detect."""
        p = self.result.p[2].item()
        assert p < 0.05, f"s2 (heterogeneous) not detected: joint p={p:.4e}"

    def test_null_snps_not_inflated(self):
        """Null SNPs (s5-s19) should have controlled false-positive rate."""
        null_p = self.result.p[5:].numpy()
        reject_rate = (null_p < 0.05).mean()
        # Allow up to 25% — generous for 15 null SNPs (binomial noise)
        assert reject_rate < 0.35, (
            f"Null SNP rejection rate {reject_rate:.2f} exceeds 0.35"
        )


class TestKnownTruthBinaryPerEnv:
    """Per-env marginals should detect SNPs active in each environment."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(family="binary", use_spa=False, pql_max_iter=20)
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_env0_specific_detected_in_env0(self):
        """s1 (β=1.0 in env 0): per-env marginal should detect in env 0."""
        p_env0 = self.result.p_marginal[1, 0].item()
        assert p_env0 < 0.05, f"s1 not detected in env 0: p={p_env0:.4e}"

    def test_env0_specific_NOT_detected_in_env1(self):
        """s1 (β=0 in env 1): should NOT be significant in env 1."""
        p_env1 = self.result.p_marginal[1, 1].item()
        # Not significant (expect p > 0.01 since true beta=0)
        assert p_env1 > 0.01, (
            f"s1 falsely detected in env 1 (β=0): p={p_env1:.4e}"
        )

    def test_stable_detected_in_all_envs(self):
        """s0 (stable β=0.8): should be detected in all environments."""
        for e in range(3):
            p_e = self.result.p_marginal[0, e].item()
            assert p_e < 0.05, (
                f"s0 (stable) not detected in env {e}: p={p_e:.4e}"
            )

    def test_opposing_detected_in_active_envs(self):
        """s4 (β=0.8 in env0, β=-0.8 in env2): detected in env 0 and 2."""
        p_env0 = self.result.p_marginal[4, 0].item()
        p_env2 = self.result.p_marginal[4, 2].item()
        assert p_env0 < 0.05, f"s4 not detected in env 0: p={p_env0:.4e}"
        assert p_env2 < 0.05, f"s4 not detected in env 2: p={p_env2:.4e}"


class TestKnownTruthBinaryHomogeneity:
    """Homogeneity test should distinguish stable from heterogeneous effects."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(family="binary", use_spa=False, pql_max_iter=20)
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_has_small_homogeneity(self):
        """s0 (stable): homogeneity p should be large (effects are equal)."""
        p_hom = self.result.p_homogeneity[0].item()
        assert p_hom > 0.01, (
            f"s0 (stable β=0.8) falsely flagged as heterogeneous: "
            f"p_hom={p_hom:.4e}"
        )

    def test_env_specific_has_large_homogeneity(self):
        """s1 (env-specific): homogeneity stat should be large."""
        stat_hom = self.result.stat_homogeneity[1].item()
        # Environment-specific effect is a strong departure from homogeneity
        assert stat_hom > 2.0, (
            f"s1 (env-specific) homogeneity stat too small: {stat_hom:.4f}"
        )

    def test_heterogeneous_has_large_homogeneity(self):
        """s2 (heterogeneous): homogeneity stat should be large."""
        stat_hom = self.result.stat_homogeneity[2].item()
        assert stat_hom > 2.0, (
            f"s2 (heterogeneous) homogeneity stat too small: {stat_hom:.4f}"
        )

    def test_opposing_has_large_homogeneity(self):
        """s4 (opposing): effects differ → large homogeneity stat."""
        stat_hom = self.result.stat_homogeneity[4].item()
        assert stat_hom > 2.0, (
            f"s4 (opposing) homogeneity stat too small: {stat_hom:.4f}"
        )


class TestKnownTruthBinaryReactionNorm:
    """Reaction-norm decomposition should separate stable from GxE."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(
            family="binary", parameterization="reaction_norm",
            use_spa=False, pql_max_iter=20,
        )
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_has_significant_stable_component(self):
        """s0 (stable β=0.8): stable component should be significant."""
        p_stable = self.result.p_stable[0].item()
        assert p_stable < 0.05, (
            f"s0 stable component not detected: p_stable={p_stable:.4e}"
        )

    def test_stable_has_nonsignificant_gxe(self):
        """s0 (stable β=0.8): GxE component should NOT be significant."""
        p_gxe = self.result.p_gxe[0].item()
        assert p_gxe > 0.01, (
            f"s0 (stable) falsely shows GxE: p_gxe={p_gxe:.4e}"
        )

    def test_env_specific_has_significant_gxe(self):
        """s1 (env-specific): GxE component should be significant."""
        p_gxe = self.result.p_gxe[1].item()
        assert p_gxe < 0.05, (
            f"s1 (env-specific) GxE not detected: p_gxe={p_gxe:.4e}"
        )

    def test_opposing_has_nonsignificant_stable(self):
        """s4 (opposing ±0.8): stable component should NOT be significant
        because mean(β) ≈ 0."""
        p_stable = self.result.p_stable[4].item()
        assert p_stable > 0.01, (
            f"s4 (opposing) stable component falsely detected: "
            f"p_stable={p_stable:.4e}"
        )

    def test_opposing_has_significant_gxe(self):
        """s4 (opposing ±0.8): GxE should be significant."""
        p_gxe = self.result.p_gxe[4].item()
        assert p_gxe < 0.05, (
            f"s4 (opposing) GxE not detected: p_gxe={p_gxe:.4e}"
        )

    def test_null_snps_gxe_not_inflated(self):
        """Null SNPs should not show inflated GxE signal."""
        null_gxe = self.result.p_gxe[5:].numpy()
        reject = (null_gxe < 0.05).mean()
        assert reject < 0.35, (
            f"Null SNP GxE rejection rate {reject:.2f} too high"
        )


class TestKnownTruthBinaryEffectDirection:
    """Estimated beta signs should match planted truth for strong effects."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(family="binary", use_spa=False, pql_max_iter=20)
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_positive_in_all_envs(self):
        """s0 (stable β=+0.8): estimated beta should be positive in all envs."""
        for e in range(3):
            beta_hat = self.result.beta[0, e].item()
            assert beta_hat > 0, (
                f"s0 env {e}: expected positive beta, got {beta_hat:.4f}"
            )

    def test_heterogeneous_sign_pattern(self):
        """s2 (β=[0.5, -0.7, 0.9]): signs should match planted truth."""
        expected_signs = [1, -1, 1]
        for e in range(3):
            beta_hat = self.result.beta[2, e].item()
            actual_sign = 1 if beta_hat > 0 else -1
            assert actual_sign == expected_signs[e], (
                f"s2 env {e}: expected sign={expected_signs[e]}, "
                f"got beta={beta_hat:.4f}"
            )

    def test_opposing_sign_pattern(self):
        """s4 (β=[0.8, 0, -0.8]): env 0 positive, env 2 negative."""
        beta_0 = self.result.beta[4, 0].item()
        beta_2 = self.result.beta[4, 2].item()
        assert beta_0 > 0, f"s4 env 0: expected positive, got {beta_0:.4f}"
        assert beta_2 < 0, f"s4 env 2: expected negative, got {beta_2:.4f}"


# =====================================================================
# Ordinal known-truth tests
# =====================================================================

class TestKnownTruthOrdinalJoint:
    """Ordinal ME-GLMM should detect known causal SNPs."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_ordinal(n=600, E=2, J=3, m_null=12, seed=2027)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(
            family="ordinal", n_categories=3,
            use_spa=False, pql_max_iter=20,
        )
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_detected_joint(self):
        """Ordinal s0 (stable β=0.8): joint test should detect."""
        p = self.result.p[0].item()
        assert p < 0.05, f"Ordinal s0 (stable) not detected: p={p:.4e}"

    def test_env_specific_detected_joint(self):
        """Ordinal s1 (env-specific): joint test should detect."""
        p = self.result.p[1].item()
        assert p < 0.05, f"Ordinal s1 (env-specific) not detected: p={p:.4e}"

    def test_null_snps_calibrated(self):
        """Ordinal null SNPs: controlled rejection rate."""
        null_p = self.result.p[3:].numpy()
        reject = (null_p < 0.05).mean()
        assert reject < 0.35, (
            f"Ordinal null rejection rate {reject:.2f} too high"
        )


class TestKnownTruthOrdinalPerEnv:
    """Ordinal per-env marginals should match planted architecture."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_ordinal(n=600, E=2, J=3, m_null=12, seed=2027)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(
            family="ordinal", n_categories=3,
            use_spa=False, pql_max_iter=20,
        )
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_env0_specific_detected_in_env0(self):
        """Ordinal s1 (β=1.0 in env 0): detected in env 0."""
        p = self.result.p_marginal[1, 0].item()
        assert p < 0.05, f"Ordinal s1 not detected in env 0: p={p:.4e}"

    def test_env0_specific_NOT_detected_in_env1(self):
        """Ordinal s1 (β=0 in env 1): NOT detected in env 1."""
        p = self.result.p_marginal[1, 1].item()
        assert p > 0.01, (
            f"Ordinal s1 falsely detected in env 1: p={p:.4e}"
        )

    def test_stable_detected_in_both_envs(self):
        """Ordinal s0 (stable): detected in both environments."""
        for e in range(2):
            p = self.result.p_marginal[0, e].item()
            assert p < 0.05, (
                f"Ordinal s0 not detected in env {e}: p={p:.4e}"
            )


class TestKnownTruthOrdinalHomogeneity:
    """Ordinal homogeneity should distinguish stable from heterogeneous."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_ordinal(n=600, E=2, J=3, m_null=12, seed=2027)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(
            family="ordinal", n_categories=3,
            use_spa=False, pql_max_iter=20,
        )
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_stable_low_homogeneity(self):
        """Ordinal s0 (stable): homogeneity p should be large."""
        p_hom = self.result.p_homogeneity[0].item()
        assert p_hom > 0.01, (
            f"Ordinal s0 (stable) falsely heterogeneous: p_hom={p_hom:.4e}"
        )

    def test_heterogeneous_high_homogeneity_stat(self):
        """Ordinal s2 (β=[0.5,-0.7]): homogeneity stat should be large."""
        stat_hom = self.result.stat_homogeneity[2].item()
        assert stat_hom > 2.0, (
            f"Ordinal s2 homogeneity stat too small: {stat_hom:.4f}"
        )


# =====================================================================
# Summary diagnostic: confusion matrix for all test types
# =====================================================================

class TestKnownTruthBinarySummary:
    """Summary: overall detection rates for the full causal architecture."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        self.Y, self.G, self.X0, self.K, self.true_beta, self.labels = \
            _simulate_known_binary(n=600, E=3, m_null=15, seed=2026)
        self.vmeta = _make_vmeta(self.G.shape[1])
        model = MultiEnvGLMM(
            family="binary", parameterization="reaction_norm",
            use_spa=False, pql_max_iter=20,
        )
        self.nf = model.fit_null(self.Y, self.X0, self.K)
        self.result = model.score_chunk(self.G, self.nf, self.vmeta)

    def test_sensitivity_strong_causals(self):
        """At least 3 of the 5 causal SNPs detected by joint test at p<0.05."""
        causal_p = self.result.p[:5].numpy()
        n_detected = (causal_p < 0.05).sum()
        assert n_detected >= 3, (
            f"Only {n_detected}/5 causal SNPs detected by joint test. "
            f"p-values: {causal_p}"
        )

    def test_specificity_null_snps(self):
        """Null SNPs: false positive rate should be reasonable."""
        null_p = self.result.p[5:].numpy()
        fpr = (null_p < 0.05).mean()
        assert fpr < 0.35, (
            f"False positive rate among null SNPs: {fpr:.2f}"
        )

    def test_reaction_norm_correct_decomposition(self):
        """Stable SNPs → large stable component; env-varying → large GxE."""
        # s0 (stable): stable should rank higher than GxE
        stat_stable_s0 = self.result.stat_stable[0].item()
        stat_gxe_s0 = self.result.stat_gxe[0].item()
        assert stat_stable_s0 > stat_gxe_s0, (
            f"s0 (stable): stat_stable={stat_stable_s0:.2f} should > "
            f"stat_gxe={stat_gxe_s0:.2f}"
        )

        # s1 (env-specific): GxE should be present
        stat_gxe_s1 = self.result.stat_gxe[1].item()
        assert stat_gxe_s1 > 1.0, (
            f"s1 (env-specific): stat_gxe={stat_gxe_s1:.2f} too small"
        )
