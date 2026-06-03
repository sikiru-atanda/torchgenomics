"""Tests for haplotype block detection and pairwise LD statistics."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.ld import (
    LDBlock,
    compute_dprime_matrix,
    compute_pairwise_ld,
    compute_r2_matrix,
    detect_blocks,
    save_blocks_bed,
)
from torchgenomics.ld._dprime_ci import dprime_confidence_interval
from torchgenomics.ld._em_haplotype import build_genotype_counts, em_haplotype_freq
from torchgenomics.ld._pairwise import (
    compute_dprime_phased,
    compute_dprime_unphased,
    compute_r2_pairs,
)

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def simple_genotypes():
    """5 SNPs, 100 samples with known LD structure.

    SNPs 0-2 are in perfect LD (copies of each other + noise).
    SNPs 3-4 are independent.
    """
    torch.manual_seed(42)
    n = 100
    # Base SNP
    g0 = torch.randint(0, 3, (n,), dtype=torch.float64)
    # Correlated SNPs (same as g0 with minor noise)
    g1 = g0.clone()
    g2 = g0.clone()
    # Flip a few values for imperfect LD
    flip_mask = torch.rand(n) < 0.05
    g1[flip_mask] = (2 - g1[flip_mask]).clamp(0, 2)
    # Independent SNPs
    g3 = torch.randint(0, 3, (n,), dtype=torch.float64)
    g4 = torch.randint(0, 3, (n,), dtype=torch.float64)

    G = torch.stack([g0, g1, g2, g3, g4], dim=1)
    pos = [1000, 2000, 3000, 50000, 60000]
    chrs = ["1"] * 5
    ids = [f"rs{i}" for i in range(5)]
    return G, pos, chrs, ids


@pytest.fixture
def phased_haplotypes():
    """Phased haplotypes with known block structure.

    Block 1: SNPs 0-3 (no recombination within)
    Block 2: SNPs 4-6 (no recombination within)
    Recombination between SNPs 3 and 4.
    """
    torch.manual_seed(99)
    n, ploidy, m = 200, 2, 7

    H = torch.zeros(n, ploidy, m, dtype=torch.float64)
    for s in range(n):
        for p in range(ploidy):
            # Block 1: all 4 SNPs have same haplotype
            allele1 = torch.randint(0, 2, (1,)).item()
            H[s, p, 0:4] = allele1
            # Block 2: independent haplotype
            allele2 = torch.randint(0, 2, (1,)).item()
            H[s, p, 4:7] = allele2

    # Add some noise (minor recombination within blocks)
    noise = torch.rand(n, ploidy, m) < 0.02
    H[noise] = 1.0 - H[noise]

    pos = [1000, 2000, 3000, 4000, 20000, 21000, 22000]
    chrs = ["1"] * 7
    ids = [f"rs{i}" for i in range(7)]
    return H, pos, chrs, ids


# ── r-squared tests ─────────────────────────────────────────────────

class TestR2:
    def test_r2_matrix_shape(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        assert r2.shape == (5, 5)

    def test_r2_diagonal_is_one(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        assert torch.allclose(r2.diag(), torch.ones(5, dtype=torch.float64), atol=1e-10)

    def test_r2_symmetric(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        assert torch.allclose(r2, r2.T, atol=1e-10)

    def test_r2_correlated_snps_high(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        # SNPs 0 and 1 are highly correlated
        assert r2[0, 1] > 0.8

    def test_r2_independent_snps_low(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        # SNPs 0 and 3 are independent
        assert r2[0, 3] < 0.3

    def test_r2_pairs_matches_matrix(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2_mat = compute_r2_matrix(G)
        idx_i = torch.tensor([0, 0, 1])
        idx_j = torch.tensor([1, 3, 2])
        r2_pairs = compute_r2_pairs(G, idx_i, idx_j)
        for k in range(3):
            assert abs(r2_pairs[k].item() - r2_mat[idx_i[k], idx_j[k]].item()) < 1e-8

    def test_r2_in_unit_interval(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        r2 = compute_r2_matrix(G)
        assert (r2 >= -1e-10).all()
        assert (r2 <= 1.0 + 1e-10).all()


# ── EM haplotype frequency tests ────────────────────────────────────

class TestEM:
    def test_em_frequencies_sum_to_one(self):
        """EM-estimated haplotype frequencies must sum to 1."""
        torch.manual_seed(10)
        n = 200
        G = torch.randint(0, 3, (n, 4), dtype=torch.float64)
        idx_i = torch.tensor([0, 1])
        idx_j = torch.tensor([2, 3])
        counts = build_genotype_counts(G, idx_i, idx_j)
        hapfreq = em_haplotype_freq(counts)
        sums = hapfreq.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2, dtype=torch.float64), atol=1e-6)

    def test_em_matches_phased(self, phased_haplotypes):
        """EM D' from unphased genotypes should approximate phased D'."""
        H, pos, chrs, ids = phased_haplotypes
        # Derive unphased genotypes
        G = H.sum(dim=1)  # (n, m)

        idx_i = torch.tensor([0, 4])
        idx_j = torch.tensor([1, 5])

        dp_phased = compute_dprime_phased(H, idx_i, idx_j)
        dp_unphased = compute_dprime_unphased(G, idx_i, idx_j)

        # Should be similar (not exact due to EM approximation)
        assert torch.allclose(dp_phased.abs(), dp_unphased.abs(), atol=0.15), (
            f"Phased D': {dp_phased}, Unphased D': {dp_unphased}"
        )

    def test_em_all_frequencies_positive(self):
        """All haplotype frequencies should be non-negative."""
        torch.manual_seed(7)
        G = torch.randint(0, 3, (100, 6), dtype=torch.float64)
        idx_i = torch.tensor([0, 1, 2])
        idx_j = torch.tensor([3, 4, 5])
        counts = build_genotype_counts(G, idx_i, idx_j)
        hapfreq = em_haplotype_freq(counts)
        assert (hapfreq >= 0).all()


# ── D' tests ────────────────────────────────────────────────────────

class TestDprime:
    def test_dprime_phased_perfect_ld(self):
        """D' should be 1.0 for perfectly linked SNPs."""
        n, ploidy = 100, 2
        H = torch.zeros(n, ploidy, 2, dtype=torch.float64)
        for s in range(n):
            for p in range(ploidy):
                a = torch.randint(0, 2, (1,)).item()
                H[s, p, :] = a  # same allele at both loci

        idx_i = torch.tensor([0])
        idx_j = torch.tensor([1])
        dp = compute_dprime_phased(H, idx_i, idx_j)
        assert dp.abs().item() > 0.95

    def test_dprime_in_range(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        idx_i = torch.tensor([0, 0, 1])
        idx_j = torch.tensor([1, 3, 4])
        dp = compute_dprime_unphased(G, idx_i, idx_j)
        assert (dp.abs() <= 1.0 + 1e-8).all()

    def test_dprime_matrix_shape(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        dp = compute_dprime_matrix(G)
        assert dp.shape == (5, 5)

    def test_dprime_matrix_symmetric(self, simple_genotypes):
        G, _, _, _ = simple_genotypes
        dp = compute_dprime_matrix(G)
        assert torch.allclose(dp, dp.T, atol=1e-8)


# ── D' CI tests ─────────────────────────────────────────────────────

class TestDprimeCI:
    def test_ci_contains_mle(self):
        """CI should contain the MLE."""
        torch.manual_seed(5)
        G = torch.randint(0, 3, (200, 4), dtype=torch.float64)
        idx_i = torch.tensor([0, 1])
        idx_j = torch.tensor([2, 3])
        counts = build_genotype_counts(G, idx_i, idx_j)
        dp_hat, ci_low, ci_high = dprime_confidence_interval(counts)
        assert (ci_low <= dp_hat + 1e-6).all()
        assert (ci_high >= dp_hat - 1e-6).all()

    def test_ci_within_range(self):
        """CI bounds should be in [-1, 1]."""
        torch.manual_seed(8)
        G = torch.randint(0, 3, (150, 4), dtype=torch.float64)
        idx_i = torch.tensor([0, 1])
        idx_j = torch.tensor([2, 3])
        counts = build_genotype_counts(G, idx_i, idx_j)
        _, ci_low, ci_high = dprime_confidence_interval(counts)
        assert (ci_low >= -1.0 - 1e-6).all()
        assert (ci_high <= 1.0 + 1e-6).all()

    def test_strong_ld_narrow_ci(self, phased_haplotypes):
        """Strongly linked SNPs should have narrow CI near 1."""
        H, _, _, _ = phased_haplotypes
        G = H.sum(dim=1)
        # SNPs 0 and 1 are in same block
        idx_i = torch.tensor([0])
        idx_j = torch.tensor([1])
        counts = build_genotype_counts(G, idx_i, idx_j)
        dp_hat, ci_low, ci_high = dprime_confidence_interval(counts)
        # CI should be high for linked SNPs
        assert ci_low.item() > 0.3, f"CI_low={ci_low.item():.3f} too low for linked SNPs"


# ── Block detection tests ───────────────────────────────────────────

class TestFourGamete:
    def test_detects_recombination_breakpoint(self, phased_haplotypes):
        """Should detect breakpoint between block 1 (0-3) and block 2 (4-6)."""
        H, pos, chrs, ids = phased_haplotypes
        blocks = detect_blocks(
            H.sum(dim=1), pos, chrs, ids,
            method="four_gamete",
            haplotypes=H,
        )
        assert len(blocks) >= 1

    def test_requires_haplotypes(self, simple_genotypes):
        """four_gamete should raise without haplotypes."""
        G, pos, chrs, ids = simple_genotypes
        with pytest.raises(ValueError, match="phased haplotypes"):
            detect_blocks(G, pos, chrs, ids, method="four_gamete")

    def test_blocks_are_contiguous(self, phased_haplotypes):
        H, pos, chrs, ids = phased_haplotypes
        blocks = detect_blocks(
            H.sum(dim=1), pos, chrs, ids,
            method="four_gamete",
            haplotypes=H,
        )
        for block in blocks:
            indices = block.variant_indices
            assert indices == list(range(indices[0], indices[-1] + 1))


class TestR2Blocks:
    def test_r2_blocks_detected(self, simple_genotypes):
        """r2 method should detect at least one block."""
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="r2", r2_threshold=0.3)
        assert len(blocks) >= 1

    def test_r2_blocks_correlated_snps_together(self, simple_genotypes):
        """Correlated SNPs (0, 1, 2) should be in the same block."""
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="r2",
            r2_threshold=0.3, max_kb=100.0,
        )
        # Find the block containing SNP 0
        block_with_0 = [b for b in blocks if 0 in b.variant_indices]
        if block_with_0:
            assert 1 in block_with_0[0].variant_indices


class TestSpineBlocks:
    def test_spine_produces_blocks(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="spine",
            d_prime_threshold=0.5, max_kb=100.0,
        )
        # Should produce at least one block from correlated SNPs
        assert isinstance(blocks, list)


class TestGabrielBlocks:
    def test_gabriel_produces_blocks(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="gabriel",
            max_kb=100.0, ci_low=0.5, ci_high=0.7,
        )
        assert isinstance(blocks, list)

    def test_gabriel_blocks_have_correct_method(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="gabriel",
            max_kb=100.0, ci_low=0.5, ci_high=0.7,
        )
        for b in blocks:
            assert b.method == "gabriel"


# ── Integration tests ───────────────────────────────────────────────

class TestIntegration:
    def test_pairwise_ld_output(self, simple_genotypes):
        G, pos, chrs, _ = simple_genotypes
        pld = compute_pairwise_ld(G, pos, chrs, max_kb=100.0)
        assert pld.r2.shape == pld.dprime.shape
        assert pld.idx_i.shape == pld.idx_j.shape
        assert len(pld.r2) > 0

    def test_save_blocks_bed(self, simple_genotypes, tmp_path):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="r2", r2_threshold=0.3)
        bed_path = tmp_path / "blocks.bed"
        save_blocks_bed(blocks, str(bed_path))
        assert bed_path.exists()
        lines = bed_path.read_text().strip().split("\n")
        # First line is header, remaining lines are blocks
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == len(blocks)

    def test_ldblock_has_region(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="r2", r2_threshold=0.3)
        for b in blocks:
            assert b.region is not None
            assert b.region.chr == "1"
            assert b.region.start >= 0
            assert b.region.end > b.region.start

    def test_unknown_method_raises(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        with pytest.raises(ValueError, match="Unknown method"):
            detect_blocks(G, pos, chrs, ids, method="unknown")

    def test_all_methods_produce_valid_output(self, simple_genotypes, phased_haplotypes):
        G, pos, chrs, ids = simple_genotypes
        H, hpos, hchrs, hids = phased_haplotypes

        for method in ["gabriel", "spine", "r2"]:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=100.0,
                ci_low=0.5, ci_high=0.7,
                d_prime_threshold=0.5,
                r2_threshold=0.3,
            )
            assert isinstance(blocks, list)
            for b in blocks:
                assert isinstance(b, LDBlock)
                assert b.n_variants >= 2
                assert 0.0 <= b.mean_r2 <= 1.0

        # Four gamete needs haplotypes
        blocks = detect_blocks(
            H.sum(dim=1), hpos, hchrs, hids,
            method="four_gamete", haplotypes=H,
        )
        assert isinstance(blocks, list)
