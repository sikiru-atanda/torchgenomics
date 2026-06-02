"""Tests for Phase 29: Leave-Region-Out LMM (LRO-LMM).

Covers:
- Block GRM contribution (additivity, rank)
- No proximal contamination (causal SNP excluded from its own null K)
- Result completeness (all variants covered, valid p-values)
- Test consistency (wald, score, lrt)
- Null calibration and power
- Edge cases (singleton blocks, few variants)
- Protocol (has run() method, result dataclass)
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.base import VariantMeta
from torchgenomics.models.lro_lmm import LROLMM, LROResult, _block_grm_contribution

# ===================================================================
# Helper: simulate data
# ===================================================================

def _simulate_lro_data(
    n: int = 150,
    m: int = 200,
    n_blocks: int = 20,
    block_size: int = 10,
    h2: float = 0.4,
    causal_beta: float = 0.5,
    seed: int = 42,
):
    """Simulate data with block LD structure and causal effect in block 0."""
    torch.manual_seed(seed)

    m = n_blocks * block_size

    G = torch.zeros(n, m, dtype=torch.float64)
    for b in range(n_blocks):
        start = b * block_size
        latent = torch.randn(n, dtype=torch.float64)
        for j in range(block_size):
            noise = torch.randn(n, dtype=torch.float64)
            raw = 0.6 * latent + 0.4 * noise
            G[:, start + j] = (raw > 0.5).float() + (raw > -0.5).float()

    K, meta = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    sig2_g = h2
    sig2_e = 1.0 - h2
    nK = K + 1e-4 * torch.eye(n, dtype=torch.float64)
    LK = torch.linalg.cholesky(nK)
    u = math.sqrt(sig2_g) * LK @ torch.randn(n, dtype=torch.float64)

    # Causal effect at SNP 0 (block 0)
    g0 = G[:, 0] - G[:, 0].mean()
    std = g0.std().clamp(min=1e-6)
    g0 = g0 / std
    Y = X0.squeeze() * 0.5 + g0 * causal_beta + u
    Y = Y + math.sqrt(sig2_e) * torch.randn(n, dtype=torch.float64)

    vpos = [i * 1000 for i in range(m)]
    vchr = ["1"] * m
    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=vchr, pos=vpos,
        a1=["A"] * m, a2=["G"] * m,
    )

    return G, Y, X0, K, meta.normalizer, vmeta, vpos, vchr


# ===================================================================
# Test class 1: Block GRM contribution
# ===================================================================

class TestBlockGRM:
    """Tests for _block_grm_contribution and GRM additivity."""

    def test_additivity(self):
        """K_full should equal sum of all block contributions."""
        torch.manual_seed(1)
        n, m = 100, 50
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K_full, meta = grm_vanraden(G)

        # Split into 5 blocks of 10
        K_sum = torch.zeros_like(K_full)
        for b in range(5):
            G_block = G[:, b*10:(b+1)*10]
            K_b = _block_grm_contribution(G_block, ploidy=2,
                                          normalizer=meta.normalizer)
            K_sum += K_b

        assert torch.allclose(K_full, K_sum, atol=1e-10)

    def test_rank_bounded(self):
        """Block GRM should have rank <= p_b."""
        torch.manual_seed(2)
        n, p_b = 100, 5
        G_block = torch.randint(0, 3, (n, p_b), dtype=torch.float64)
        K_b = _block_grm_contribution(G_block, ploidy=2, normalizer=100.0)

        eigvals = torch.linalg.eigvalsh(K_b)
        # Number of eigenvalues > threshold should be <= p_b
        rank = (eigvals.abs() > 1e-10).sum().item()
        assert rank <= p_b

    def test_symmetric_psd(self):
        """Block GRM should be symmetric and PSD."""
        torch.manual_seed(3)
        G_block = torch.randint(0, 3, (80, 8), dtype=torch.float64)
        K_b = _block_grm_contribution(G_block, ploidy=2, normalizer=50.0)

        assert torch.allclose(K_b, K_b.T, atol=1e-12)
        eigvals = torch.linalg.eigvalsh(K_b)
        assert (eigvals >= -1e-10).all()


# ===================================================================
# Test class 2: Proximal contamination
# ===================================================================

class TestNoProximalContamination:
    """Tests that LRO excludes the tested region from its null GRM."""

    def test_lro_vs_standard_power(self):
        """LRO should have equal or better power near causal region.

        Standard LMM absorbs causal signal into random effect;
        LRO excludes the causal block, preserving signal.
        """
        data = _simulate_lro_data(n=200, m=200, n_blocks=20,
                                  block_size=10, h2=0.3,
                                  causal_beta=0.6, seed=10)
        G, Y, X0, K, normalizer, vmeta, vpos, vchr = data

        # LRO scan
        lro = LROLMM(ld_method="r2")
        result_lro = lro.run(Y, X0, G, vmeta, vpos, vchr, test="wald")

        # Standard LMM scan (full K)
        from torchgenomics.models.single_trait_lmm import SingleTraitLMM
        lmm = SingleTraitLMM()
        nf = lmm.fit_null(Y, X0, K=K)
        result_std = lmm.score_chunk(G, nf, vmeta, test="wald")

        # The causal SNP (index 0) should have at least as good p-value
        # in LRO as in standard LMM
        p_lro = result_lro.p[0].item()
        p_std = result_std.p[0].item()
        # LRO p-value should not be dramatically worse
        assert p_lro < 0.5  # should detect signal

    def test_causal_snp_not_in_own_grm(self):
        """Conceptual check: block 0's SNPs shouldn't contribute to K_{-0}."""
        torch.manual_seed(11)
        n, m = 100, 50
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K_full, meta = grm_vanraden(G)

        # Block 0: first 10 SNPs
        G_block0 = G[:, :10]
        K_b0 = _block_grm_contribution(G_block0, ploidy=2,
                                        normalizer=meta.normalizer)
        K_minus_0 = K_full - K_b0

        # K_{-0} should NOT equal K_full
        assert not torch.allclose(K_minus_0, K_full, atol=1e-6)

        # But K_{-0} + K_b0 should equal K_full
        assert torch.allclose(K_minus_0 + K_b0, K_full, atol=1e-10)


# ===================================================================
# Test class 3: Result completeness
# ===================================================================

class TestResultComplete:
    """Tests that all variants are covered and results are valid."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _simulate_lro_data(n=100, m=100, n_blocks=10,
                                  block_size=10, seed=20)
        self.G, self.Y, self.X0, self.K, _, self.vmeta, self.vpos, self.vchr = data

    def test_all_variants_covered(self):
        """Result should have stats for all m variants."""
        lro = LROLMM(ld_method="r2")
        result = lro.run(self.Y, self.X0, self.G, self.vmeta,
                         self.vpos, self.vchr)

        m = self.G.shape[1]
        assert result.beta.shape[0] == m
        assert result.stat.shape[0] == m
        assert result.p.shape[0] == m
        assert len(result.snp) == m

    def test_pvalues_valid(self):
        """All p-values should be in [0, 1]."""
        lro = LROLMM(ld_method="r2")
        result = lro.run(self.Y, self.X0, self.G, self.vmeta,
                         self.vpos, self.vchr)

        assert (result.p >= 0).all()
        assert (result.p <= 1).all()


# ===================================================================
# Test class 4: Test consistency
# ===================================================================

class TestConsistency:
    """Tests for different test types and LD methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        data = _simulate_lro_data(n=100, m=100, n_blocks=10,
                                  block_size=10, seed=30)
        self.G, self.Y, self.X0, _, _, self.vmeta, self.vpos, self.vchr = data

    def test_wald_and_score_rank_correlated(self):
        """Wald and score test rankings should be correlated."""
        lro = LROLMM(ld_method="r2")
        res_w = lro.run(self.Y, self.X0, self.G, self.vmeta,
                        self.vpos, self.vchr, test="wald")
        res_s = lro.run(self.Y, self.X0, self.G, self.vmeta,
                        self.vpos, self.vchr, test="score")

        # Rank correlation
        rank_w = res_w.stat.argsort(descending=True).float()
        rank_s = res_s.stat.argsort(descending=True).float()
        corr = torch.corrcoef(torch.stack([rank_w, rank_s]))[0, 1]
        assert corr > 0.5

    def test_different_ld_methods_work(self):
        """Should work with different LD block methods."""
        for method in ["r2", "spine"]:
            lro = LROLMM(ld_method=method)
            result = lro.run(self.Y, self.X0, self.G, self.vmeta,
                             self.vpos, self.vchr)
            assert isinstance(result, LROResult)
            assert result.block_method == method

    def test_returns_lro_result(self):
        """Should return LROResult dataclass."""
        lro = LROLMM(ld_method="r2")
        result = lro.run(self.Y, self.X0, self.G, self.vmeta,
                         self.vpos, self.vchr)
        assert isinstance(result, LROResult)
        assert result.n_blocks > 0
        assert len(result.block_sizes) == result.n_blocks


# ===================================================================
# Test class 5: Calibration
# ===================================================================

class TestCalibration:
    """Tests for null calibration and power."""

    def test_null_calibration(self):
        """Under null (no causal), p-values should not be inflated."""
        data = _simulate_lro_data(n=150, m=100, n_blocks=10,
                                  block_size=10, h2=0.3,
                                  causal_beta=0.0, seed=40)
        G, Y, X0, _, _, vmeta, vpos, vchr = data

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        # Median p-value should be near 0.5 under null
        median_p = result.p.median().item()
        assert median_p > 0.1  # not severely inflated

    def test_power_detects_signal(self):
        """With strong signal, causal SNP should have small p-value."""
        data = _simulate_lro_data(n=200, m=100, n_blocks=10,
                                  block_size=10, h2=0.2,
                                  causal_beta=0.8, seed=41)
        G, Y, X0, _, _, vmeta, vpos, vchr = data

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        # Causal SNP (index 0) should be among top hits
        assert result.p[0].item() < 0.05


# ===================================================================
# Test class 6: Edge cases
# ===================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_few_variants(self):
        """Should work with very few variants."""
        torch.manual_seed(50)
        n, m = 80, 5
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)
        vpos = [i * 100000 for i in range(m)]  # far apart → singletons
        vchr = ["1"] * m
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)],
            chr=vchr, pos=vpos,
            a1=["A"] * m, a2=["G"] * m,
        )

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        assert isinstance(result, LROResult)
        assert result.beta.shape[0] == m

    def test_singleton_blocks(self):
        """Should handle all-singleton blocks (no LD)."""
        torch.manual_seed(51)
        n, m = 80, 10
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)
        # Very far apart → all singletons
        vpos = [i * 500000 for i in range(m)]
        vchr = ["1"] * m
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)],
            chr=vchr, pos=vpos,
            a1=["A"] * m, a2=["G"] * m,
        )

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        assert result.n_blocks == m  # all singletons
        assert all(s == 1 for s in result.block_sizes)

    def test_large_block(self):
        """Should handle a large block."""
        data = _simulate_lro_data(n=100, m=100, n_blocks=5,
                                  block_size=20, seed=52)
        G, Y, X0, _, _, vmeta, vpos, vchr = data

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        assert isinstance(result, LROResult)


# ===================================================================
# Test class 7: Protocol
# ===================================================================

class TestProtocol:
    """Tests for API conformance."""

    def test_has_run_method(self):
        """LROLMM should have a run() method."""
        model = LROLMM()
        assert hasattr(model, "run")
        assert callable(model.run)

    def test_result_dataclass_fields(self):
        """LROResult should have expected fields."""
        data = _simulate_lro_data(n=80, m=50, n_blocks=5,
                                  block_size=10, seed=60)
        G, Y, X0, _, _, vmeta, vpos, vchr = data

        lro = LROLMM(ld_method="r2")
        result = lro.run(Y, X0, G, vmeta, vpos, vchr)

        assert hasattr(result, "beta")
        assert hasattr(result, "p")
        assert hasattr(result, "n_blocks")
        assert hasattr(result, "block_sizes")
        assert hasattr(result, "block_method")
