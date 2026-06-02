"""Tests for LD clumping."""

import pytest
import torch

from torchgenomics.postgwas._clump import ld_clump


def _make_genotypes(n=200, m=50, seed=0):
    """Generate genotypes with some LD structure."""
    torch.manual_seed(seed)
    G = torch.randn(n, m, dtype=torch.float64)
    # Add LD blocks: SNPs 0-9 correlated, 10-19 correlated, etc.
    for block_start in range(0, m, 10):
        block_end = min(block_start + 10, m)
        base = torch.randn(n, 1, dtype=torch.float64)
        for j in range(block_start, block_end):
            G[:, j] = 0.8 * base.squeeze() + 0.6 * G[:, j]
    pos = list(range(1000, 1000 + m * 5000, 5000))
    chr_labels = ["1"] * m
    return G, pos, chr_labels


class TestLDClump:
    """Tests for PLINK-style LD clumping."""

    def test_most_significant_selected(self):
        """The most significant SNP should always be an index SNP."""
        G, pos, chr_labels = _make_genotypes()
        m = G.shape[1]
        p = torch.ones(m, dtype=torch.float64) * 0.5
        p[25] = 1e-10  # most significant
        result = ld_clump(p, G, pos, chr_labels, p_threshold=1e-5)
        assert 25 in result.index_snps

    def test_no_significant_returns_empty(self):
        """No SNPs below threshold -> no clumps."""
        G, pos, chr_labels = _make_genotypes()
        m = G.shape[1]
        p = torch.ones(m, dtype=torch.float64) * 0.1
        result = ld_clump(p, G, pos, chr_labels, p_threshold=1e-8)
        assert result.n_clumps == 0
        assert len(result.index_snps) == 0

    def test_r2_threshold_respected(self):
        """High r2 SNPs should be clumped; low r2 should remain independent."""
        torch.manual_seed(10)
        n = 300
        g1 = torch.randn(n, 1, dtype=torch.float64)
        g2 = g1.clone()  # perfect LD
        g3 = torch.randn(n, 1, dtype=torch.float64)  # independent
        G = torch.cat([g1, g2, g3], dim=1)
        pos = [1000, 2000, 3000]
        chr_labels = ["1", "1", "1"]
        p = torch.tensor([1e-10, 1e-9, 1e-10], dtype=torch.float64)

        # With high r2 threshold, g1 and g2 should be clumped
        result = ld_clump(p, G, pos, chr_labels, r2_threshold=0.5, p_threshold=1e-5)
        assert 0 in result.index_snps  # most significant
        assert 2 in result.index_snps  # independent
        # g2 should be clumped with g1
        assert 1 not in result.index_snps

    def test_window_respected(self):
        """Distant SNPs should not be clumped even with high LD."""
        torch.manual_seed(20)
        n = 200
        g = torch.randn(n, 1, dtype=torch.float64)
        G = torch.cat([g, g], dim=1)
        # 500 kb apart -> outside 250 kb window
        pos = [0, 500_001]
        chr_labels = ["1", "1"]
        p = torch.tensor([1e-10, 1e-10], dtype=torch.float64)
        result = ld_clump(p, G, pos, chr_labels, r2_threshold=0.1,
                          p_threshold=1e-5, window_kb=250.0)
        # Both should be index SNPs (outside window)
        assert len(result.index_snps) == 2

    def test_cross_chromosome_independence(self):
        """SNPs on different chromosomes should never be clumped."""
        torch.manual_seed(30)
        n = 200
        g = torch.randn(n, 1, dtype=torch.float64)
        G = torch.cat([g, g], dim=1)
        pos = [1000, 1000]
        chr_labels = ["1", "2"]
        p = torch.tensor([1e-10, 1e-10], dtype=torch.float64)
        result = ld_clump(p, G, pos, chr_labels, r2_threshold=0.01, p_threshold=1e-5)
        assert len(result.index_snps) == 2

    def test_clump_members_tracked(self):
        """Clump members should be properly recorded."""
        torch.manual_seed(40)
        n = 300
        g1 = torch.randn(n, 1, dtype=torch.float64)
        G = torch.cat([g1, g1 + 0.01 * torch.randn(n, 1, dtype=torch.float64),
                        g1 + 0.01 * torch.randn(n, 1, dtype=torch.float64),
                        torch.randn(n, 1, dtype=torch.float64)], dim=1)
        pos = [1000, 2000, 3000, 4000]
        chr_labels = ["1"] * 4
        p = torch.tensor([1e-12, 1e-8, 1e-9, 1e-10], dtype=torch.float64)

        result = ld_clump(p, G, pos, chr_labels, r2_threshold=0.5, p_threshold=1e-5)
        assert 0 in result.index_snps  # most significant
        # SNPs 1 and 2 should be clumped with SNP 0
        idx0_pos = result.index_snps.index(0)
        members = result.clump_members[idx0_pos]
        assert 1 in members or 2 in members

    def test_p_threshold_filters(self):
        """Only SNPs below p_threshold should be considered."""
        G, pos, chr_labels = _make_genotypes()
        m = G.shape[1]
        p = torch.ones(m, dtype=torch.float64) * 1e-6
        p[0] = 1e-10
        p[25] = 1e-4  # above 5e-8 threshold
        result = ld_clump(p, G, pos, chr_labels, p_threshold=5e-8)
        # SNP 25 should NOT be an index SNP (p > threshold)
        assert 25 not in result.index_snps

    def test_index_p_values_correct(self):
        """Index p-values should match original p-values."""
        G, pos, chr_labels = _make_genotypes()
        m = G.shape[1]
        p = torch.ones(m, dtype=torch.float64)
        p[5] = 1e-15
        p[30] = 1e-12
        result = ld_clump(p, G, pos, chr_labels, p_threshold=1e-8)
        for i, idx in enumerate(result.index_snps):
            assert result.index_p[i].item() == pytest.approx(p[idx].item(), rel=1e-6)
