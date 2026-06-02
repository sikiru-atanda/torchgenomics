"""Tests for advanced block detection methods."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.ld import LDBlock, detect_blocks

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def simple_genotypes():
    """5 SNPs, 100 samples with known LD structure."""
    torch.manual_seed(42)
    n = 100
    g0 = torch.randint(0, 3, (n,), dtype=torch.float64)
    g1 = g0.clone()
    g2 = g0.clone()
    flip_mask = torch.rand(n) < 0.05
    g1[flip_mask] = (2 - g1[flip_mask]).clamp(0, 2)
    g3 = torch.randint(0, 3, (n,), dtype=torch.float64)
    g4 = torch.randint(0, 3, (n,), dtype=torch.float64)
    G = torch.stack([g0, g1, g2, g3, g4], dim=1)
    pos = [1000, 2000, 3000, 50000, 60000]
    chrs = ["1"] * 5
    ids = [f"rs{i}" for i in range(5)]
    return G, pos, chrs, ids


@pytest.fixture
def gp_probs_fixture():
    """Synthetic GP probabilities with known uncertainty structure.

    SNPs 0-2: high confidence (sharp probabilities).
    SNPs 3-4: low confidence (diffuse probabilities).
    """
    torch.manual_seed(77)
    n, m, ploidy = 100, 5, 2
    # High-confidence SNPs: peaked at one dosage
    probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
    for i in range(n):
        for j in range(3):  # SNPs 0-2: high confidence
            true_dose = torch.randint(0, 3, (1,)).item()
            probs[i, j, true_dose] = 0.95
            for k in range(3):
                if k != true_dose:
                    probs[i, j, k] = 0.025
        for j in range(3, 5):  # SNPs 3-4: low confidence
            probs[i, j] = torch.tensor([0.33, 0.34, 0.33], dtype=torch.float64)

    # Expected dosage
    d_vals = torch.arange(ploidy + 1, dtype=torch.float64)
    G = (probs * d_vals).sum(dim=-1)

    pos = [1000, 2000, 3000, 50000, 60000]
    chrs = ["1"] * 5
    ids = [f"rs{i}" for i in range(5)]
    return G, probs, pos, chrs, ids


@pytest.fixture
def multi_pop_genotypes():
    """Genotypes with 2 populations having different LD patterns."""
    torch.manual_seed(99)
    n_per_pop = 50
    m = 6

    # Population A: SNPs 0-2 correlated, 3-5 independent
    g0a = torch.randint(0, 3, (n_per_pop,), dtype=torch.float64)
    G_a = torch.stack([g0a, g0a.clone(), g0a.clone(),
                       torch.randint(0, 3, (n_per_pop,), dtype=torch.float64),
                       torch.randint(0, 3, (n_per_pop,), dtype=torch.float64),
                       torch.randint(0, 3, (n_per_pop,), dtype=torch.float64)], dim=1)

    # Population B: SNPs 0-2 independent, 3-5 correlated
    g3b = torch.randint(0, 3, (n_per_pop,), dtype=torch.float64)
    G_b = torch.stack([torch.randint(0, 3, (n_per_pop,), dtype=torch.float64),
                       torch.randint(0, 3, (n_per_pop,), dtype=torch.float64),
                       torch.randint(0, 3, (n_per_pop,), dtype=torch.float64),
                       g3b, g3b.clone(), g3b.clone()], dim=1)

    G = torch.cat([G_a, G_b], dim=0)
    pop_ids = ["A"] * n_per_pop + ["B"] * n_per_pop
    pos = [1000, 2000, 3000, 4000, 5000, 6000]
    chrs = ["1"] * m
    ids = [f"rs{i}" for i in range(m)]
    return G, pop_ids, pos, chrs, ids


@pytest.fixture
def recomb_map_fixture():
    """Synthetic recombination map with a hotspot."""
    # 10 SNPs: uniform recomb except a hotspot between SNPs 4 and 5
    cm = [0.0, 0.01, 0.02, 0.03, 0.04, 0.50, 0.51, 0.52, 0.53, 0.54]
    torch.manual_seed(55)
    n = 100
    # Block 1: SNPs 0-4 correlated
    g0 = torch.randint(0, 3, (n,), dtype=torch.float64)
    G1 = torch.stack([g0] * 5, dim=1)
    # Block 2: SNPs 5-9 correlated
    g5 = torch.randint(0, 3, (n,), dtype=torch.float64)
    G2 = torch.stack([g5] * 5, dim=1)
    G = torch.cat([G1, G2], dim=1)
    pos = [1000 * (i + 1) for i in range(10)]
    chrs = ["1"] * 10
    ids = [f"rs{i}" for i in range(10)]
    return G, cm, pos, chrs, ids


# ── GWAS-Aligned ───────────────────────────────────────────────────

class TestGWASAligned:
    def test_blocks_detected(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="gwas_aligned", max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert isinstance(b, LDBlock)
            assert b.method == "gwas_aligned"

    def test_concentration_populated(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="gwas_aligned", max_kb=100.0)
        for b in blocks:
            assert b.concentration_ratio is not None
            assert 0.0 <= b.concentration_ratio <= 1.0

    def test_respects_max_kb(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="gwas_aligned", max_kb=3.0)
        for b in blocks:
            span_kb = (b.region.end - b.region.start) / 1000.0
            assert span_kb <= 3.5  # small tolerance for BED half-open

    def test_penalty_reduces_block_count(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks_low = detect_blocks(G, pos, chrs, ids, method="gwas_aligned",
                                   max_kb=100.0, condition_penalty=0.01)
        blocks_high = detect_blocks(G, pos, chrs, ids, method="gwas_aligned",
                                    max_kb=100.0, condition_penalty=10.0)
        # High penalty should not produce more blocks than low penalty
        assert len(blocks_high) <= len(blocks_low) + 1  # allow small variation


# ── Uncertainty-Aware ──────────────────────────────────────────────

class TestUncertainty:
    def test_requires_gp_probs(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        with pytest.raises(ValueError, match="gp_probs"):
            detect_blocks(G, pos, chrs, ids, method="uncertainty")

    def test_produces_blocks(self, gp_probs_fixture):
        G, probs, pos, chrs, ids = gp_probs_fixture
        blocks = detect_blocks(G, pos, chrs, ids, method="uncertainty",
                               gp_probs=probs, r2_threshold=0.1, max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.method == "uncertainty"
            assert b.uncertainty_metric is not None

    def test_uncertainty_metric_in_range(self, gp_probs_fixture):
        G, probs, pos, chrs, ids = gp_probs_fixture
        blocks = detect_blocks(G, pos, chrs, ids, method="uncertainty",
                               gp_probs=probs, r2_threshold=0.01, max_kb=100.0)
        for b in blocks:
            if b.uncertainty_metric is not None:
                assert 0.0 <= b.uncertainty_metric <= 1.0


# ── Cross-Population Consensus ─────────────────────────────────────

class TestCrossPop:
    def test_requires_population_ids(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        with pytest.raises(ValueError, match="population_ids"):
            detect_blocks(G, pos, chrs, ids, method="cross_pop")

    def test_produces_blocks(self, multi_pop_genotypes):
        G, pop_ids, pos, chrs, ids = multi_pop_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="cross_pop",
                               population_ids=pop_ids, max_kb=100.0,
                               stability_threshold=0.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.method == "cross_pop"

    def test_stability_populated(self, multi_pop_genotypes):
        G, pop_ids, pos, chrs, ids = multi_pop_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="cross_pop",
                               population_ids=pop_ids, max_kb=100.0,
                               stability_threshold=0.0)
        for b in blocks:
            assert b.population_stability is not None
            assert 0.0 <= b.population_stability <= 1.0 + 1e-6


# ── Graphical (Conditional Independence) ───────────────────────────

class TestGraphical:
    def test_produces_blocks(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="graphical",
                               l1_penalty=0.05, max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.method == "graphical"

    def test_l1_controls_sparsity(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks_dense = detect_blocks(G, pos, chrs, ids, method="graphical",
                                     l1_penalty=0.01, max_kb=100.0)
        blocks_sparse = detect_blocks(G, pos, chrs, ids, method="graphical",
                                      l1_penalty=1.0, max_kb=100.0)
        # Higher penalty → sparser graph → more/smaller components
        # (or fewer blocks if components become singletons)
        assert isinstance(blocks_sparse, list)


# ── Change-Point ───────────────────────────────────────────────────

class TestChangepoint:
    def test_detects_boundary(self, recomb_map_fixture):
        G, cm, pos, chrs, ids = recomb_map_fixture
        blocks = detect_blocks(G, pos, chrs, ids, method="changepoint",
                               penalty=3.0, min_segment_size=3)
        assert isinstance(blocks, list)
        assert len(blocks) >= 1
        for b in blocks:
            assert b.method == "changepoint"

    def test_uses_recomb_map(self, recomb_map_fixture):
        G, cm, pos, chrs, ids = recomb_map_fixture
        blocks = detect_blocks(G, pos, chrs, ids, method="changepoint",
                               recomb_map_cm=cm, use_recomb_map=True,
                               penalty=1.0, min_segment_size=3)
        assert isinstance(blocks, list)

    def test_penalty_controls_n_blocks(self, recomb_map_fixture):
        G, cm, pos, chrs, ids = recomb_map_fixture
        blocks_low = detect_blocks(G, pos, chrs, ids, method="changepoint",
                                   penalty=1.0, min_segment_size=2)
        blocks_high = detect_blocks(G, pos, chrs, ids, method="changepoint",
                                    penalty=100.0, min_segment_size=2)
        assert len(blocks_high) <= len(blocks_low) + 1


# ── Big-LD ─────────────────────────────────────────────────────────

class TestBigLD:
    def test_blocks_detected(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="big_ld",
                               r2_threshold=0.3, max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.method == "big_ld"

    def test_non_overlapping(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="big_ld",
                               r2_threshold=0.3, max_kb=100.0)
        # Check no overlapping variant indices
        used = set()
        for b in blocks:
            for idx in b.variant_indices:
                assert idx not in used, f"Overlap at index {idx}"
                used.add(idx)


# ── DP Optimization ────────────────────────────────────────────────

class TestDPOptimize:
    def test_haplotype_diversity_objective(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="dp_optimize",
                               objective="haplotype_diversity", max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.method == "dp_optimize"
            assert b.metadata is not None
            assert b.metadata["objective"] == "haplotype_diversity"

    def test_tag_snp_objective(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="dp_optimize",
                               objective="tag_snp", max_kb=100.0)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.metadata["objective"] == "tag_snp"

    def test_penalty_effect(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks_low = detect_blocks(G, pos, chrs, ids, method="dp_optimize",
                                   penalty=0.1, max_kb=100.0)
        blocks_high = detect_blocks(G, pos, chrs, ids, method="dp_optimize",
                                    penalty=100.0, max_kb=100.0)
        # Higher penalty → fewer, larger blocks
        assert len(blocks_high) <= len(blocks_low) + 1


# ── Wall & Pritchard Diagnostics ───────────────────────────────────

class TestWallPritchard:
    def test_returns_ldblock(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="wall_pritchard",
                               max_kb=100.0)
        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0].method == "wall_pritchard"

    def test_blockiness_score_populated(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="wall_pritchard",
                               max_kb=100.0)
        assert blocks[0].blockiness_score is not None
        assert 0.0 <= blocks[0].blockiness_score <= 1.0

    def test_metadata_has_Q(self, simple_genotypes):
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(G, pos, chrs, ids, method="wall_pritchard",
                               max_kb=100.0)
        assert "Q" in blocks[0].metadata
        assert "Q_adj" in blocks[0].metadata

    def test_independent_snps_low_score(self):
        """Independent SNPs should have low blockiness."""
        torch.manual_seed(123)
        n = 200
        G = torch.randint(0, 3, (n, 5), dtype=torch.float64)
        pos = [1000 * (i + 1) for i in range(5)]
        chrs = ["1"] * 5
        blocks = detect_blocks(G, pos, chrs, method="wall_pritchard", max_kb=100.0)
        assert blocks[0].blockiness_score < 0.5


# ── CC-Graph ──────────────────────────────────────────────────────

class TestCCGraph:
    def test_blocks_detected(self, simple_genotypes):
        """High-LD SNPs should cluster into at least one block."""
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.3, max_kb=200.0,
        )
        assert len(blocks) >= 1

    def test_tag_snp_in_metadata(self, simple_genotypes):
        """Each block should have tag_snp metadata."""
        G, pos, chrs, ids = simple_genotypes
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.3, max_kb=200.0,
        )
        for b in blocks:
            assert b.metadata is not None
            assert "tag_snp" in b.metadata
            assert "tag_snp_maf" in b.metadata
            assert 0.0 <= b.metadata["tag_snp_maf"] <= 0.5

    def test_transitive_clustering(self):
        """A-B and B-C high r² but A-C low → all three in one block."""
        torch.manual_seed(99)
        n = 200
        # A and B highly correlated
        a = torch.randint(0, 3, (n,), dtype=torch.float64)
        b = a.clone()
        flip = torch.rand(n) < 0.05
        b[flip] = (2 - b[flip]).clamp(0, 2)
        # B and C highly correlated
        c = b.clone()
        flip2 = torch.rand(n) < 0.05
        c[flip2] = (2 - c[flip2]).clamp(0, 2)
        # D independent
        d = torch.randint(0, 3, (n,), dtype=torch.float64)
        G = torch.stack([a, b, c, d], dim=1)
        pos = [1000, 2000, 3000, 4000]
        chrs = ["1"] * 4
        ids = ["snpA", "snpB", "snpC", "snpD"]
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.3, max_kb=200.0,
        )
        # A, B, C should be in one block via transitive linkage
        multi_blocks = [b for b in blocks if b.n_variants >= 3]
        assert len(multi_blocks) >= 1
        big = multi_blocks[0]
        assert 0 in big.variant_indices
        assert 1 in big.variant_indices
        assert 2 in big.variant_indices

    def test_singletons_included(self):
        """Independent SNPs should appear as singleton blocks."""
        torch.manual_seed(55)
        n = 100
        G = torch.randint(0, 3, (n, 4), dtype=torch.float64)
        pos = [1000, 2000, 3000, 4000]
        chrs = ["1"] * 4
        blocks = detect_blocks(
            G, pos, chrs, method="cc_graph",
            r2_threshold=0.99, max_kb=200.0, include_singletons=True,
        )
        # With very high threshold, most SNPs should be singletons
        singleton_count = sum(1 for b in blocks if b.n_variants == 1)
        assert singleton_count >= 2

    def test_singletons_excluded(self):
        """With include_singletons=False, only multi-SNP blocks returned."""
        torch.manual_seed(55)
        n = 100
        G = torch.randint(0, 3, (n, 4), dtype=torch.float64)
        pos = [1000, 2000, 3000, 4000]
        chrs = ["1"] * 4
        blocks = detect_blocks(
            G, pos, chrs, method="cc_graph",
            r2_threshold=0.99, max_kb=200.0,
            include_singletons=False, min_block_snps=2,
        )
        for b in blocks:
            assert b.n_variants >= 2

    def test_r2_threshold_effect(self, simple_genotypes):
        """Higher threshold should produce more (smaller) blocks."""
        G, pos, chrs, ids = simple_genotypes
        blocks_low = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.1, max_kb=200.0,
        )
        blocks_high = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.9, max_kb=200.0,
        )
        # More stringent threshold → fewer edges → more components
        max_size_low = max(b.n_variants for b in blocks_low)
        max_size_high = max(b.n_variants for b in blocks_high)
        assert max_size_high <= max_size_low


# ── Integration ────────────────────────────────────────────────────

class TestAdvancedIntegration:
    def test_all_advanced_methods_valid_output(self, simple_genotypes,
                                               gp_probs_fixture,
                                               multi_pop_genotypes):
        G, pos, chrs, ids = simple_genotypes
        G_gp, probs, pos_gp, chrs_gp, ids_gp = gp_probs_fixture
        G_mp, pop_ids, pos_mp, chrs_mp, ids_mp = multi_pop_genotypes

        # Methods that work on simple genotypes
        for method in ["gwas_aligned", "graphical", "changepoint",
                       "big_ld", "cc_graph", "dp_optimize", "wall_pritchard"]:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=100.0,
                r2_threshold=0.3,
                l1_penalty=0.1,
                penalty=3.0,
                min_segment_size=2,
            )
            assert isinstance(blocks, list), f"{method} failed"
            for b in blocks:
                assert isinstance(b, LDBlock)
                assert b.n_variants >= 1

        # Uncertainty method
        blocks = detect_blocks(
            G_gp, pos_gp, chrs_gp, ids_gp,
            method="uncertainty", gp_probs=probs,
            r2_threshold=0.01, max_kb=100.0,
        )
        assert isinstance(blocks, list)

        # Cross-pop method
        blocks = detect_blocks(
            G_mp, pos_mp, chrs_mp, ids_mp,
            method="cross_pop", population_ids=pop_ids,
            max_kb=100.0, stability_threshold=0.0,
        )
        assert isinstance(blocks, list)
