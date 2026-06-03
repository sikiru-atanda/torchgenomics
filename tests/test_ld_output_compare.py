"""Tests for standardized block output, PLINK compatibility, and comparison metrics."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.io.regions import Region
from torchgenomics.ld import (
    BlockComparisonResult,
    LDBlock,
    compare_all_methods,
    compare_blocks,
    detect_blocks,
    format_comparison_table,
    ldblocks_to_plink_det,
    load_plink_blocks_det,
    plink_blocks_to_ldblocks,
    save_blocks_bed,
    save_blocks_det,
    save_blocks_summary,
)

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_blocks() -> list[LDBlock]:
    """Three blocks on chr1 with varied optional fields."""
    return [
        LDBlock(
            region=Region("blk1", "1", 1000, 3001),
            n_variants=3, variant_indices=[0, 1, 2],
            method="gabriel", mean_r2=0.85, mean_dprime=0.92,
        ),
        LDBlock(
            region=Region("blk2", "1", 5000, 8001),
            n_variants=4, variant_indices=[3, 4, 5, 6],
            method="gabriel", mean_r2=0.78, mean_dprime=0.88,
            blockiness_score=0.72,
        ),
        LDBlock(
            region=Region("blk3", "2", 1000, 2001),
            n_variants=2, variant_indices=[7, 8],
            method="gabriel", mean_r2=0.91, mean_dprime=0.95,
            concentration_ratio=0.65,
        ),
    ]


@pytest.fixture
def genotypes_for_all_methods():
    """Genotype data suitable for running all 13 methods."""
    torch.manual_seed(42)
    n = 100
    # Create LD structure: g0-g2 correlated, g3-g4 independent
    g0 = torch.randint(0, 3, (n,), dtype=torch.float64)
    g1 = g0.clone()
    g2 = g0.clone()
    flip1 = torch.rand(n) < 0.05
    flip2 = torch.rand(n) < 0.05
    g1[flip1] = (2 - g1[flip1]).clamp(0, 2)
    g2[flip2] = (2 - g2[flip2]).clamp(0, 2)
    g3 = torch.randint(0, 3, (n,), dtype=torch.float64)
    g4 = torch.randint(0, 3, (n,), dtype=torch.float64)
    G = torch.stack([g0, g1, g2, g3, g4], dim=1)
    pos = [1000, 2000, 3000, 50000, 60000]
    chrs = ["1"] * 5
    ids = ["rs1", "rs2", "rs3", "rs4", "rs5"]
    return G, pos, chrs, ids


# ── Standardized BED output ────────────────────────────────────────

class TestSaveBlocksBed:
    def test_header_present(self, sample_blocks, tmp_path):
        out = tmp_path / "blocks.bed"
        save_blocks_bed(sample_blocks, out)
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#chr")
        assert "blockiness_score" in lines[0]
        assert "concentration_ratio" in lines[0]
        assert "changepoint_confidence" in lines[0]

    def test_fixed_column_count(self, sample_blocks, tmp_path):
        """All rows should have exactly 13 columns."""
        out = tmp_path / "blocks.bed"
        save_blocks_bed(sample_blocks, out)
        lines = out.read_text().strip().split("\n")
        # Skip header
        for line in lines[1:]:
            cols = line.split("\t")
            assert len(cols) == 13, f"Expected 13 columns, got {len(cols)}: {line}"

    def test_na_for_missing_fields(self, sample_blocks, tmp_path):
        out = tmp_path / "blocks.bed"
        save_blocks_bed(sample_blocks, out)
        lines = out.read_text().strip().split("\n")
        # First block has no optional fields — should have NA for all 5
        cols = lines[1].split("\t")
        assert cols[8] == "NA"   # blockiness_score
        assert cols[9] == "NA"   # concentration_ratio
        assert cols[10] == "NA"  # population_stability
        assert cols[11] == "NA"  # uncertainty_metric
        assert cols[12] == "NA"  # changepoint_confidence

    def test_populated_optional_fields(self, sample_blocks, tmp_path):
        out = tmp_path / "blocks.bed"
        save_blocks_bed(sample_blocks, out)
        lines = out.read_text().strip().split("\n")
        # Second block has blockiness_score=0.72
        cols = lines[2].split("\t")
        assert cols[8] == "0.7200"
        assert cols[9] == "NA"
        # Third block has concentration_ratio=0.65
        cols = lines[3].split("\t")
        assert cols[8] == "NA"
        assert cols[9] == "0.6500"

    def test_all_methods_same_format(self, genotypes_for_all_methods, tmp_path):
        """Every method should produce 13-column BED output."""
        G, pos, chrs, ids = genotypes_for_all_methods
        methods_simple = [
            "gabriel", "spine", "r2",
            "gwas_aligned", "graphical", "changepoint",
            "big_ld", "cc_graph", "dp_optimize", "wall_pritchard",
        ]
        for method in methods_simple:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            out = tmp_path / f"{method}.bed"
            save_blocks_bed(blocks, out)
            lines = out.read_text().strip().split("\n")
            assert lines[0].startswith("#chr"), f"{method}: missing header"
            for i, line in enumerate(lines[1:], 1):
                cols = line.split("\t")
                assert len(cols) == 13, (
                    f"{method} line {i}: expected 13 columns, got {len(cols)}"
                )


# ── PLINK .blocks.det output ──────────────────────────────────────

class TestSaveBlocksDet:
    def test_header_present(self, sample_blocks, tmp_path):
        out = tmp_path / "blocks.det"
        save_blocks_det(sample_blocks, out, variant_ids=[f"rs{i}" for i in range(9)])
        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("CHR")
        assert "NSNPS" in lines[0]
        assert "SNPS" in lines[0]

    def test_plink_coordinate_convention(self, sample_blocks, tmp_path):
        """BP1/BP2 should be 1-based inclusive."""
        out = tmp_path / "blocks.det"
        save_blocks_det(sample_blocks, out)
        lines = out.read_text().strip().split("\n")
        cols = lines[1].split("\t")
        # First block: region start=1000 (0-based) → BP1=1001 (1-based)
        assert cols[1] == "1001"

    def test_snps_pipe_delimited(self, sample_blocks, tmp_path):
        out = tmp_path / "blocks.det"
        ids = [f"rs{i}" for i in range(9)]
        save_blocks_det(sample_blocks, out, variant_ids=ids)
        lines = out.read_text().strip().split("\n")
        cols = lines[1].split("\t")
        # First block: indices [0,1,2] → "rs0|rs1|rs2"
        assert cols[-1] == "rs0|rs1|rs2"


# ── Summary output ────────────────────────────────────────────────

class TestSaveBlocksSummary:
    def test_writes_summary(self, sample_blocks, tmp_path):
        out = tmp_path / "summary.txt"
        save_blocks_summary(sample_blocks, out)
        text = out.read_text()
        assert "Block Detection Summary" in text
        assert "gabriel" in text
        assert "Total blocks:" in text
        assert "3" in text

    def test_per_chr_breakdown(self, sample_blocks, tmp_path):
        out = tmp_path / "summary.txt"
        save_blocks_summary(sample_blocks, out)
        text = out.read_text()
        assert "Per-Chromosome Breakdown" in text

    def test_empty_blocks(self, tmp_path):
        out = tmp_path / "summary.txt"
        save_blocks_summary([], out)
        text = out.read_text()
        assert "No blocks" in text


# ── PLINK block parser ────────────────────────────────────────────

class TestPLINKParser:
    def test_parse_blocks_det(self, tmp_path):
        """Parse a synthetic PLINK .blocks.det file."""
        det_file = tmp_path / "plink.blocks.det"
        det_file.write_text(
            "CHR\tBP1\tBP2\tKB\tNSNPS\tSNPS\n"
            "1\t1001\t3001\t2.001\t3\trs1|rs2|rs3\n"
            "1\t5001\t8001\t3.001\t4\trs4|rs5|rs6|rs7\n"
            "2\t1001\t2001\t1.001\t2\trs8|rs9\n"
        )
        blocks = load_plink_blocks_det(det_file)
        assert len(blocks) == 3
        assert blocks[0].chr == "1"
        assert blocks[0].bp1 == 1001
        assert blocks[0].bp2 == 3001
        assert blocks[0].n_snps == 3
        assert blocks[0].snps == ["rs1", "rs2", "rs3"]

    def test_skip_malformed_lines(self, tmp_path):
        det_file = tmp_path / "bad.blocks.det"
        det_file.write_text(
            "CHR\tBP1\tBP2\tKB\tNSNPS\tSNPS\n"
            "1\t1001\t3001\t2.0\t3\trs1|rs2|rs3\n"
            "bad line\n"
            "2\t1001\t2001\t1.0\t2\trs4|rs5\n"
        )
        blocks = load_plink_blocks_det(det_file)
        assert len(blocks) == 2

    def test_plink_to_ldblocks(self, tmp_path):
        det_file = tmp_path / "plink.blocks.det"
        det_file.write_text(
            "CHR\tBP1\tBP2\tKB\tNSNPS\tSNPS\n"
            "1\t1001\t3001\t2.001\t3\trs1|rs2|rs3\n"
        )
        pblocks = load_plink_blocks_det(det_file)
        variant_ids = ["rs0", "rs1", "rs2", "rs3", "rs4"]
        ld_blocks = plink_blocks_to_ldblocks(pblocks, variant_ids=variant_ids)
        assert len(ld_blocks) == 1
        assert ld_blocks[0].method == "plink"
        assert ld_blocks[0].n_variants == 3
        # Variant indices should be resolved from IDs
        assert ld_blocks[0].variant_indices == [1, 2, 3]

    def test_ldblocks_to_plink_det(self, sample_blocks, tmp_path):
        """Round-trip: LDBlocks → PLINK det string."""
        ids = [f"rs{i}" for i in range(9)]
        content = ldblocks_to_plink_det(sample_blocks, variant_ids=ids)
        assert "CHR\tBP1\tBP2" in content
        assert "rs0|rs1|rs2" in content

    def test_roundtrip(self, sample_blocks, tmp_path):
        """Write PLINK det, read back, compare."""
        ids = [f"rs{i}" for i in range(9)]
        det_path = tmp_path / "rt.blocks.det"
        ldblocks_to_plink_det(sample_blocks, variant_ids=ids, output_path=det_path)

        pblocks = load_plink_blocks_det(det_path)
        assert len(pblocks) == 3
        assert pblocks[0].n_snps == 3


# ── Block comparison metrics ──────────────────────────────────────

class TestCompareBlocks:
    def test_identical_blocks(self, sample_blocks):
        """Comparing blocks to themselves should give perfect scores."""
        result = compare_blocks(sample_blocks, sample_blocks)
        assert result.jaccard_index == 1.0
        assert result.dice_coefficient == 1.0
        assert result.overlap_coefficient == 1.0
        assert result.match_rate_a == 1.0
        assert result.match_rate_b == 1.0
        assert result.mean_boundary_dist == 0.0

    def test_disjoint_blocks(self):
        """Completely non-overlapping blocks should give zero Jaccard."""
        blocks_a = [LDBlock(
            region=Region("a", "1", 0, 100), n_variants=2,
            variant_indices=[0, 1], method="a", mean_r2=0.5, mean_dprime=0.5,
        )]
        blocks_b = [LDBlock(
            region=Region("b", "1", 200, 300), n_variants=2,
            variant_indices=[5, 6], method="b", mean_r2=0.5, mean_dprime=0.5,
        )]
        result = compare_blocks(blocks_a, blocks_b)
        assert result.jaccard_index == 0.0
        assert result.dice_coefficient == 0.0
        assert result.n_matched_a == 0
        assert result.n_matched_b == 0

    def test_partial_overlap(self):
        """Blocks sharing some variants."""
        blocks_a = [LDBlock(
            region=Region("a", "1", 0, 100), n_variants=3,
            variant_indices=[0, 1, 2], method="a", mean_r2=0.5, mean_dprime=0.5,
        )]
        blocks_b = [LDBlock(
            region=Region("b", "1", 0, 100), n_variants=3,
            variant_indices=[1, 2, 3], method="b", mean_r2=0.5, mean_dprime=0.5,
        )]
        result = compare_blocks(blocks_a, blocks_b)
        # intersection={1,2}, union={0,1,2,3}
        assert abs(result.jaccard_index - 0.5) < 1e-10
        # Dice: 2*2 / (3+3) = 4/6 = 0.6667
        assert abs(result.dice_coefficient - 2 / 3) < 1e-10

    def test_empty_blocks(self):
        result = compare_blocks([], [])
        assert result.jaccard_index == 1.0
        assert result.n_blocks_a == 0
        assert result.n_blocks_b == 0

    def test_per_chr_breakdown(self, sample_blocks):
        result = compare_blocks(sample_blocks, sample_blocks)
        assert "1" in result.per_chr
        assert "2" in result.per_chr
        assert result.per_chr["1"]["jaccard"] == 1.0

    def test_result_dataclass(self, sample_blocks):
        result = compare_blocks(sample_blocks, sample_blocks)
        assert isinstance(result, BlockComparisonResult)
        assert result.method_a == "gabriel"
        assert result.method_b == "gabriel"


class TestCompareAllMethods:
    def test_pairwise(self, genotypes_for_all_methods):
        G, pos, chrs, ids = genotypes_for_all_methods
        methods = {"r2": [], "big_ld": [], "cc_graph": []}
        for m in methods:
            methods[m] = detect_blocks(
                G, pos, chrs, ids, method=m,
                max_kb=200.0, r2_threshold=0.3,
            )
        results = compare_all_methods(methods)
        # 3 methods → 3 pairwise comparisons
        assert len(results) == 3
        for key, r in results.items():
            assert isinstance(r, BlockComparisonResult)
            assert 0.0 <= r.jaccard_index <= 1.0

    def test_reference_mode(self, genotypes_for_all_methods):
        G, pos, chrs, ids = genotypes_for_all_methods
        methods = {}
        for m in ["r2", "big_ld", "cc_graph"]:
            methods[m] = detect_blocks(
                G, pos, chrs, ids, method=m,
                max_kb=200.0, r2_threshold=0.3,
            )
        results = compare_all_methods(methods, reference="r2")
        # 2 comparisons: big_ld vs r2, cc_graph vs r2
        assert len(results) == 2
        for key, r in results.items():
            assert r.method_b == "r2"


class TestFormatComparisonTable:
    def test_table_format(self, sample_blocks):
        result = compare_blocks(sample_blocks, sample_blocks, method_a="A", method_b="B")
        table = format_comparison_table({("A", "B"): result})
        lines = table.strip().split("\n")
        assert lines[0].startswith("Method_A")
        assert len(lines) == 2  # header + 1 row
        cols = lines[1].split("\t")
        assert cols[0] == "A"
        assert cols[1] == "B"


# ── Cross-method output consistency ───────────────────────────────

class TestAllMethodsConsistentOutput:
    """Verify all 13 methods produce consistent LDBlock fields."""

    SIMPLE_METHODS = [
        "gabriel", "spine", "r2",
        "gwas_aligned", "graphical", "changepoint",
        "big_ld", "cc_graph", "dp_optimize", "wall_pritchard",
    ]

    def test_all_blocks_have_required_fields(self, genotypes_for_all_methods):
        G, pos, chrs, ids = genotypes_for_all_methods
        for method in self.SIMPLE_METHODS:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            for b in blocks:
                assert isinstance(b.region, Region), f"{method}: bad region"
                assert isinstance(b.n_variants, int), f"{method}: bad n_variants"
                assert b.n_variants >= 1, f"{method}: n_variants < 1"
                assert isinstance(b.variant_indices, list), f"{method}: bad indices"
                assert isinstance(b.method, str), f"{method}: bad method"
                assert isinstance(b.mean_r2, float), f"{method}: bad mean_r2"
                assert 0.0 <= b.mean_r2 <= 1.0 + 1e-6, f"{method}: mean_r2 out of range"
                assert isinstance(b.mean_dprime, float), f"{method}: bad mean_dprime"
                assert b.mean_dprime >= 0.0, f"{method}: negative mean_dprime"

    def test_all_blocks_have_valid_region(self, genotypes_for_all_methods):
        G, pos, chrs, ids = genotypes_for_all_methods
        for method in self.SIMPLE_METHODS:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            for b in blocks:
                assert b.region.chr in {"1", "2"}, f"{method}: unexpected chr"
                assert b.region.start < b.region.end, f"{method}: start >= end"
                assert isinstance(b.region.region_id, str), f"{method}: bad region_id"

    def test_variant_indices_within_range(self, genotypes_for_all_methods):
        G, pos, chrs, ids = genotypes_for_all_methods
        m = G.shape[1]
        for method in self.SIMPLE_METHODS:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            for b in blocks:
                for idx in b.variant_indices:
                    assert 0 <= idx < m, f"{method}: index {idx} out of range"

    def test_all_methods_produce_valid_bed(self, genotypes_for_all_methods, tmp_path):
        """All 10 simple methods produce parseable BED with 13 columns."""
        G, pos, chrs, ids = genotypes_for_all_methods
        for method in self.SIMPLE_METHODS:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            out = tmp_path / f"{method}.bed"
            save_blocks_bed(blocks, out)

            lines = out.read_text().strip().split("\n")
            assert lines[0].startswith("#"), f"{method}: no header"
            for line in lines[1:]:
                cols = line.split("\t")
                assert len(cols) == 13, f"{method}: {len(cols)} cols != 13"

    def test_all_methods_produce_valid_det(self, genotypes_for_all_methods, tmp_path):
        """All 10 simple methods produce parseable .blocks.det."""
        G, pos, chrs, ids = genotypes_for_all_methods
        for method in self.SIMPLE_METHODS:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            out = tmp_path / f"{method}.det"
            save_blocks_det(blocks, out, variant_ids=ids)

            lines = out.read_text().strip().split("\n")
            assert lines[0].startswith("CHR"), f"{method}: no header"
            for line in lines[1:]:
                cols = line.split("\t")
                assert len(cols) == 9, f"{method}: {len(cols)} cols != 9"

    def test_pairwise_comparison_all_methods(self, genotypes_for_all_methods):
        """All simple methods can be compared pairwise."""
        G, pos, chrs, ids = genotypes_for_all_methods
        blocks_by_method = {}
        for method in ["r2", "big_ld", "cc_graph", "changepoint"]:
            blocks_by_method[method] = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                penalty=3.0, min_segment_size=2,
            )
        results = compare_all_methods(blocks_by_method, reference="r2")
        assert len(results) == 3
        table = format_comparison_table(results)
        assert "Jaccard" in table
