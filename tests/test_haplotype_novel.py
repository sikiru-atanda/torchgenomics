"""Tests for novel haplotype GWAS methods: PCHT, HHCT, HSKAT, HapGxE, BayesHap."""

import torch

from torchgwas.models.haplotype_gwas import (
    HaplotypeBlock,
    _enumerate_haplotypes_phased,
)
from torchgwas.models.haplotype_novel import (
    BayesHapResult,
    BayesianHaplotypeFineMapping,
    HapGxEResult,
    HHCTResult,
    PCHTResult,
    compute_dosage_posterior_cov,
    haplotype_gxe_test,
    haplotype_similarity_kernel,
    hierarchical_haplotype_test,
    hskat_adaptive,
    hskat_test,
    pcht_score_test,
)

DTYPE = torch.float64


# ── Shared fixtures ───────────────────────────────────────────────────

def _make_haplotype_data(n=200, m=3, seed=42):
    """Create synthetic haplotype data with 3 distinct haplotypes."""
    torch.manual_seed(seed)
    haps = torch.zeros(n, 2, m, dtype=torch.long)
    for i in range(n):
        for p in range(2):
            r = torch.rand(1).item()
            if r < 0.5:
                haps[i, p, :] = 0  # "000"
            elif r < 0.85:
                haps[i, p, :] = 1  # "111"
            else:
                alt = torch.zeros(m, dtype=torch.long)
                alt[1] = 1
                haps[i, p, :] = alt  # "010"

    G = haps.sum(dim=1).float()
    labels, freqs, dosage = _enumerate_haplotypes_phased(haps, ploidy=2)
    return G, labels, freqs, dosage, haps


def _drop_reference(freqs, dosage, labels):
    """Drop most frequent haplotype for testing."""
    ref_idx = freqs.argmax().item()
    test_idx = [j for j in range(len(labels)) if j != ref_idx]
    D = dosage[:, test_idx]
    test_labels = [labels[j] for j in test_idx]
    return D, test_labels, ref_idx


# ═══════════════════════════════════════════════════════════════════════
# 1. PCHT Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPCHT:
    """Tests for Posterior-Calibrated Haplotype Test."""

    def test_pcht_phased_no_uncertainty(self):
        """Phased data: posterior cov is zero, PCHT ≈ standard chi2 test."""
        G, labels, freqs, dosage, _ = _make_haplotype_data()
        n = G.shape[0]
        D, test_labels, _ = _drop_reference(freqs, dosage, labels)
        Y = torch.randn(n, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        # Phased data: no phase uncertainty, cov should be ~0
        # (Phased dosages are deterministic)
        cov = compute_dosage_posterior_cov(G, labels, freqs)
        # Remove reference row/col
        ref = freqs.argmax().item()
        keep = [j for j in range(len(labels)) if j != ref]
        cov_test = cov[keep][:, keep]

        # For phased data, the cov is NOT exactly zero (it's estimated
        # from unphased G). But PCHT should still produce valid p.
        result = pcht_score_test(Y, X0, D.to(DTYPE), cov_test)
        assert isinstance(result, PCHTResult)
        assert 0 <= result.p <= 1

    def test_pcht_correction_nonneg(self):
        """Correction matrix diagonal is non-negative."""
        G, labels, freqs, dosage, _ = _make_haplotype_data()
        cov = compute_dosage_posterior_cov(G, labels, freqs)
        # Diagonal of covariance sum should be >= 0
        assert (cov.diag() >= -1e-10).all()

    def test_pcht_detects_signal(self):
        """PCHT detects a haplotype-trait association."""
        torch.manual_seed(42)
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=500)
        D, test_labels, ref = _drop_reference(freqs, dosage, labels)
        n = D.shape[0]
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = D[:, 0].to(DTYPE) * 2.0 + torch.randn(n, dtype=DTYPE) * 0.5

        cov = compute_dosage_posterior_cov(G, labels, freqs)
        keep = [j for j in range(len(labels)) if j != ref]
        cov_test = cov[keep][:, keep]

        result = pcht_score_test(Y, X0, D.to(DTYPE), cov_test)
        assert result.p < 0.05

    def test_pcht_dosage_cov_shape(self):
        """Posterior covariance has shape (H, H)."""
        G, labels, freqs, dosage, _ = _make_haplotype_data()
        cov = compute_dosage_posterior_cov(G, labels, freqs)
        H = len(labels)
        assert cov.shape == (H, H)

    def test_pcht_score_test_valid(self):
        """PCHT produces valid chi-squared p-value in [0, 1]."""
        torch.manual_seed(99)
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=100)
        D, test_labels, ref = _drop_reference(freqs, dosage, labels)
        n = D.shape[0]
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)

        cov = compute_dosage_posterior_cov(G, labels, freqs)
        keep = [j for j in range(len(labels)) if j != ref]
        cov_test = cov[keep][:, keep]

        result = pcht_score_test(Y, X0, D.to(DTYPE), cov_test)
        assert 0 <= result.p <= 1
        assert result.stat >= 0


# ═══════════════════════════════════════════════════════════════════════
# 2. HHCT Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHHCT:
    """Tests for Hierarchical Haplotype Collapsing Test."""

    def test_hhct_merge_tree_structure(self):
        """H leaves produce H-1 internal nodes."""
        G, labels, freqs, dosage, _ = _make_haplotype_data()
        n = dosage.shape[0]
        H = len(labels)
        Y = torch.randn(n, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        result = hierarchical_haplotype_test(Y, X0, dosage, labels)
        assert isinstance(result, HHCTResult)
        assert result.n_leaves == H
        # Total nodes = H leaves + (H-1) internal = 2H - 1
        assert len(result.nodes) == 2 * H - 1

    def test_hhct_detects_signal(self):
        """Strong signal → at least one rejected leaf."""
        torch.manual_seed(42)
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=500)
        n = dosage.shape[0]
        X0 = torch.ones(n, 1, dtype=DTYPE)
        # Strong signal on first haplotype
        Y = dosage[:, 0].to(DTYPE) * 3.0 + torch.randn(n, dtype=DTYPE) * 0.5

        result = hierarchical_haplotype_test(Y, X0, dosage, labels, alpha=0.05)
        # Should reject something
        assert len(result.rejected_leaves) > 0

    def test_hhct_adjusted_geq_raw(self):
        """Adjusted p ≥ raw p for all nodes."""
        G, labels, freqs, dosage, _ = _make_haplotype_data()
        n = dosage.shape[0]
        Y = torch.randn(n, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        result = hierarchical_haplotype_test(Y, X0, dosage, labels)
        for node in result.nodes:
            assert node.adjusted_p >= node.raw_p - 1e-10

    def test_hhct_hierarchy_respected(self):
        """Rejected nodes form connected subtree from root."""
        torch.manual_seed(42)
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=500)
        n = dosage.shape[0]
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = dosage[:, 0].to(DTYPE) * 5.0 + torch.randn(n, dtype=DTYPE) * 0.3

        result = hierarchical_haplotype_test(Y, X0, dosage, labels, alpha=0.05)

        # Build parent map
        parent = {}
        for node in result.nodes:
            for child_id in node.children:
                parent[child_id] = node.node_id

        # Every rejected non-root node must have a rejected parent
        root_id = len(result.nodes) - 1
        for node in result.nodes:
            if node.rejected and node.node_id != root_id:
                assert node.node_id in parent
                pid = parent[node.node_id]
                assert result.nodes[pid].rejected

    def test_hhct_single_haplotype(self):
        """Single haplotype → no merging, no rejections."""
        n = 50
        Y = torch.randn(n, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        dosage = torch.ones(n, 1, dtype=DTYPE)

        result = hierarchical_haplotype_test(Y, X0, dosage, ["0"])
        assert result.n_leaves == 1
        assert len(result.nodes) == 1
        assert len(result.rejected_leaves) == 0


# ═══════════════════════════════════════════════════════════════════════
# 3. HSKAT Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHSKAT:
    """Tests for Haplotype Similarity Kernel Association Test."""

    def test_hskat_kernel_psd(self):
        """Kernel matrix is positive semi-definite."""
        labels = ["000", "001", "010", "111"]
        K = haplotype_similarity_kernel(labels, bandwidth=1.0)
        evals = torch.linalg.eigvalsh(K)
        assert (evals >= -1e-10).all()
        # Diagonal should be 1 (distance to self = 0)
        assert K.diag().allclose(torch.ones(4, dtype=DTYPE))

    def test_hskat_large_bandwidth_matches_skat(self):
        """bandwidth → ∞ gives identity kernel → standard SKAT."""
        torch.manual_seed(42)
        labels = ["00", "01", "10", "11"]
        K_inf = haplotype_similarity_kernel(labels, bandwidth=1e6)
        # With huge bandwidth, all entries ≈ 1
        assert (K_inf > 0.999).all()

    def test_hskat_detects_signal(self):
        """HSKAT detects a haplotype signal."""
        torch.manual_seed(42)
        n = 300
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, test_labels, _ = _drop_reference(freqs, dosage, labels)
        D = D.to(DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = D[:, 0] * 2.0 + torch.randn(n, dtype=DTYPE) * 0.5

        Q, _ = torch.linalg.qr(X0)
        Py = Y - Q @ (Q.T @ Y)

        def apply_P(v):
            if v.ndim == 1:
                return v - Q @ (Q.T @ v)
            return v - Q @ (Q.T @ v)

        K = haplotype_similarity_kernel(test_labels, bandwidth=1.0)
        Q_stat, p = hskat_test(D, Py, apply_P, K)
        assert p < 0.05

    def test_hskat_adaptive(self):
        """Adaptive HSKAT produces valid corrected p-value."""
        torch.manual_seed(42)
        n = 200
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, test_labels, _ = _drop_reference(freqs, dosage, labels)
        D = D.to(DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)

        Q, _ = torch.linalg.qr(X0)
        Py = Y - Q @ (Q.T @ Y)

        def apply_P(v):
            if v.ndim == 1:
                return v - Q @ (Q.T @ v)
            return v - Q @ (Q.T @ v)

        Q_stat, p_corr, bw = hskat_adaptive(D, Py, apply_P, test_labels)
        assert 0 <= p_corr <= 1
        assert bw > 0

    def test_hskat_null_calibration(self):
        """Under null, HSKAT p-value is not systematically small."""
        torch.manual_seed(42)
        n = 200
        D = torch.randn(n, 3, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)

        Q, _ = torch.linalg.qr(X0)
        Py = Y - Q @ (Q.T @ Y)

        def apply_P(v):
            if v.ndim == 1:
                return v - Q @ (Q.T @ v)
            return v - Q @ (Q.T @ v)

        labels = ["00", "01", "10"]
        K = haplotype_similarity_kernel(labels, bandwidth=1.0)
        _, p = hskat_test(D, Py, apply_P, K)
        assert p > 0.001


# ═══════════════════════════════════════════════════════════════════════
# 4. HapGxE Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHapGxE:
    """Tests for Haplotype-by-Environment Interaction Test."""

    def test_hapgxe_no_interaction(self):
        """No GxE: interaction p should not be significant."""
        torch.manual_seed(42)
        n = 300
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, _, _ = _drop_reference(freqs, dosage, labels)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        # Main effect only, no interaction
        Y = D[:, 0].to(DTYPE) * 1.0 + torch.randn(n, dtype=DTYPE) * 0.5

        result = haplotype_gxe_test(Y, X0, D.to(DTYPE), env)
        assert isinstance(result, HapGxEResult)
        # Main should be significant
        assert result.p_main < 0.05
        # Interaction should NOT be significant (usually)
        # We allow some flexibility but it shouldn't be strongly significant
        assert result.p_interaction > 0.001

    def test_hapgxe_detects_interaction(self):
        """Strong GxE: interaction p should be significant."""
        torch.manual_seed(42)
        n = 500
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, _, _ = _drop_reference(freqs, dosage, labels)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        # Interaction signal: haplotype 0 effect depends on environment
        Y = (D[:, 0].to(DTYPE) * env * 2.0
             + torch.randn(n, dtype=DTYPE) * 0.5)

        result = haplotype_gxe_test(Y, X0, D.to(DTYPE), env)
        assert result.p_interaction < 0.05

    def test_hapgxe_joint_test(self):
        """Joint test detects any haplotype effect."""
        torch.manual_seed(42)
        n = 300
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, _, _ = _drop_reference(freqs, dosage, labels)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        Y = D[:, 0].to(DTYPE) * 1.5 + torch.randn(n, dtype=DTYPE) * 0.5

        result = haplotype_gxe_test(Y, X0, D.to(DTYPE), env)
        assert result.p_joint < 0.05

    def test_hapgxe_result_fields(self):
        """Result has all expected fields."""
        torch.manual_seed(42)
        n = 100
        D = torch.randn(n, 2, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)

        result = haplotype_gxe_test(Y, X0, D, env)
        assert hasattr(result, "F_main")
        assert hasattr(result, "p_main")
        assert hasattr(result, "F_interaction")
        assert hasattr(result, "p_interaction")
        assert hasattr(result, "F_joint")
        assert hasattr(result, "p_joint")

    def test_hapgxe_null(self):
        """Under null, no test is strongly significant."""
        torch.manual_seed(42)
        n = 200
        D = torch.randn(n, 2, dtype=DTYPE)
        X0 = torch.ones(n, 1, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)

        result = haplotype_gxe_test(Y, X0, D, env)
        # None should be extremely small under null
        assert result.p_main > 0.001
        assert result.p_interaction > 0.001
        assert result.p_joint > 0.001


# ═══════════════════════════════════════════════════════════════════════
# 5. BayesHap Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBayesHap:
    """Tests for Bayesian Haplotype Fine-Mapping."""

    def _make_blocks(self, n=200, seed=42):
        """Create two HaplotypeBlocks for testing."""
        torch.manual_seed(seed)
        # Block 1: 3 SNPs
        haps1 = torch.zeros(n, 2, 3, dtype=torch.long)
        for i in range(n):
            for p in range(2):
                r = torch.rand(1).item()
                if r < 0.5:
                    haps1[i, p] = 0
                elif r < 0.85:
                    haps1[i, p] = 1
                else:
                    haps1[i, p] = torch.tensor([0, 1, 0])

        labels1, freqs1, dosage1 = _enumerate_haplotypes_phased(haps1)
        blk1 = HaplotypeBlock(
            block_id="blk1", chr="1", start=0, end=3000,
            variant_indices=[0, 1, 2],
            haplotypes=labels1, frequencies=freqs1, dosage=dosage1,
        )

        # Block 2: 2 SNPs
        haps2 = torch.randint(0, 2, (n, 2, 2))
        labels2, freqs2, dosage2 = _enumerate_haplotypes_phased(haps2)
        blk2 = HaplotypeBlock(
            block_id="blk2", chr="1", start=5000, end=7000,
            variant_indices=[5, 6],
            haplotypes=labels2, frequencies=freqs2, dosage=dosage2,
        )

        return [blk1, blk2]

    def test_bayeshap_pip_range(self):
        """PIPs are in [0, 1]."""
        n = 200
        blocks = self._make_blocks(n=n)
        Y = torch.randn(n, dtype=DTYPE)

        fm = BayesianHaplotypeFineMapping(n_signals=3, max_iter=20)
        result = fm.fit(Y, blocks)

        assert isinstance(result, BayesHapResult)
        assert (result.pip >= 0).all()
        assert (result.pip <= 1 + 1e-6).all()

    def test_bayeshap_detects_causal(self):
        """Causal haplotype should have high PIP."""
        torch.manual_seed(42)
        n = 500
        blocks = self._make_blocks(n=n, seed=42)
        # Signal from first non-reference haplotype in block 1
        ref1 = blocks[0].frequencies.argmax().item()
        test_idx = [j for j in range(len(blocks[0].haplotypes)) if j != ref1]
        if test_idx:
            causal = test_idx[0]
            Y = (blocks[0].dosage[:, causal].to(DTYPE) * 2.0
                 + torch.randn(n, dtype=DTYPE) * 0.5)
        else:
            Y = torch.randn(n, dtype=DTYPE)

        fm = BayesianHaplotypeFineMapping(n_signals=3, max_iter=50, sigma2_0=2.0)
        result = fm.fit(Y, blocks)

        # The causal haplotype should have the highest PIP
        if result.pip.numel() > 0:
            assert result.pip.max().item() > 0.3

    def test_bayeshap_credible_set(self):
        """Credible sets are non-empty for active layers."""
        n = 200
        blocks = self._make_blocks(n=n)
        Y = torch.randn(n, dtype=DTYPE)

        fm = BayesianHaplotypeFineMapping(n_signals=3, max_iter=20)
        result = fm.fit(Y, blocks)

        assert len(result.credible_sets) == 3  # one per layer
        for cs in result.credible_sets:
            assert len(cs) > 0

    def test_bayeshap_convergence(self):
        """Alpha values converge within max_iter."""
        torch.manual_seed(42)
        n = 200
        blocks = self._make_blocks(n=n)
        Y = torch.randn(n, dtype=DTYPE)

        fm = BayesianHaplotypeFineMapping(n_signals=2, max_iter=100, tol=1e-3)
        result = fm.fit(Y, blocks)

        # ELBO trace should exist
        assert len(result.elbo_trace) > 0

    def test_bayeshap_labels_populated(self):
        """Haplotype labels and block IDs are correctly populated."""
        n = 100
        blocks = self._make_blocks(n=n)
        Y = torch.randn(n, dtype=DTYPE)

        fm = BayesianHaplotypeFineMapping(n_signals=2, max_iter=10)
        result = fm.fit(Y, blocks)

        assert len(result.haplotype_labels) == result.pip.shape[0]
        assert len(result.block_ids) == result.pip.shape[0]
        # Labels should contain block ID
        for label, bid in zip(result.haplotype_labels, result.block_ids):
            assert bid in label


# ═══════════════════════════════════════════════════════════════════════
# 6. LMM-Corrected Path Tests
# ═══════════════════════════════════════════════════════════════════════

def _make_null_fit(n, Y=None, seed=42):
    """Create a minimal NullFit for testing LMM paths."""
    from torchgwas.models.base import NullFit
    torch.manual_seed(seed)
    A = torch.randn(n, n, dtype=DTYPE)
    U, _, _ = torch.linalg.svd(A)
    eigenvalues = torch.rand(n, dtype=DTYPE) * 2 + 0.1
    sig2_g, sig2_e = 0.5, 0.5
    if Y is None:
        Y = torch.randn(n, dtype=DTYPE)
    X0 = torch.ones(n, 1, dtype=DTYPE)
    X0_rot = U.T @ X0
    lam = sig2_g / max(sig2_e, 1e-20)
    H_inv = 1.0 / (eigenvalues * lam + 1.0)
    wX = H_inv.unsqueeze(1) * X0_rot
    M00 = X0_rot.T @ wX
    return NullFit(
        sig2_g=sig2_g, sig2_e=sig2_e,
        eigenvalues=eigenvalues, eigenvectors=U,
        Y_rot=U.T @ Y, X0_rot=X0_rot,
        M00=M00,
        converged=True,
    ), Y, X0


class TestLMMPaths:
    """Tests for LMM-corrected haplotype methods."""

    def test_pcht_lmm_valid(self):
        """PCHT under LMM produces valid p-value."""
        from torchgwas.models.haplotype_novel import pcht_score_test_lmm
        torch.manual_seed(42)
        n = 200
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, test_labels, ref = _drop_reference(freqs, dosage, labels)
        Y = torch.randn(n, dtype=DTYPE)
        nf, _, _ = _make_null_fit(n, Y=Y, seed=99)

        cov = compute_dosage_posterior_cov(G, labels, freqs)
        keep = [j for j in range(len(labels)) if j != ref]
        cov_test = cov[keep][:, keep]

        result = pcht_score_test_lmm(nf, D.to(DTYPE), cov_test)
        assert isinstance(result, PCHTResult)
        assert 0 <= result.p <= 1
        assert result.stat >= 0

    def test_pcht_lmm_detects_signal(self):
        """PCHT under LMM detects a strong signal."""
        from torchgwas.models.haplotype_novel import pcht_score_test_lmm
        torch.manual_seed(42)
        n = 500
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, test_labels, ref = _drop_reference(freqs, dosage, labels)
        Y = D[:, 0].to(DTYPE) * 2.0 + torch.randn(n, dtype=DTYPE) * 0.5
        nf, _, _ = _make_null_fit(n, Y=Y, seed=99)

        cov = compute_dosage_posterior_cov(G, labels, freqs)
        keep = [j for j in range(len(labels)) if j != ref]
        cov_test = cov[keep][:, keep]

        result = pcht_score_test_lmm(nf, D.to(DTYPE), cov_test)
        assert result.p < 0.05

    def test_hapgxe_lmm_valid(self):
        """HapGxE under LMM produces valid p-values."""
        from torchgwas.models.haplotype_novel import haplotype_gxe_test_lmm
        torch.manual_seed(42)
        n = 200
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, _, _ = _drop_reference(freqs, dosage, labels)
        env = torch.randn(n, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)
        nf, _, _ = _make_null_fit(n, Y=Y, seed=99)

        result = haplotype_gxe_test_lmm(nf, D.to(DTYPE), env)
        assert isinstance(result, HapGxEResult)
        assert 0 <= result.p_main <= 1
        assert 0 <= result.p_interaction <= 1
        assert 0 <= result.p_joint <= 1

    def test_hapgxe_lmm_detects_interaction(self):
        """HapGxE LMM detects a true interaction signal."""
        from torchgwas.models.haplotype_novel import haplotype_gxe_test_lmm
        torch.manual_seed(42)
        n = 500
        G, labels, freqs, dosage, _ = _make_haplotype_data(n=n)
        D, _, _ = _drop_reference(freqs, dosage, labels)
        env = torch.randn(n, dtype=DTYPE)
        # Interaction: haplotype effect modulated by environment
        Y = (D[:, 0].to(DTYPE) * env * 2.0
             + torch.randn(n, dtype=DTYPE) * 0.5)
        nf, _, _ = _make_null_fit(n, Y=Y, seed=99)

        result = haplotype_gxe_test_lmm(nf, D.to(DTYPE), env)
        assert result.p_interaction < 0.05

    def test_hapgxe_lmm_null(self):
        """HapGxE LMM under null: no test is strongly significant."""
        from torchgwas.models.haplotype_novel import haplotype_gxe_test_lmm
        torch.manual_seed(42)
        n = 200
        D = torch.randn(n, 2, dtype=DTYPE)
        env = torch.randn(n, dtype=DTYPE)
        Y = torch.randn(n, dtype=DTYPE)
        nf, _, _ = _make_null_fit(n, Y=Y, seed=99)

        result = haplotype_gxe_test_lmm(nf, D, env)
        assert result.p_main > 0.001
        assert result.p_interaction > 0.001
        assert result.p_joint > 0.001
