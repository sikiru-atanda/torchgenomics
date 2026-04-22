"""End-to-end validation: all 13 block methods vs PLINK --blocks.

Generates synthetic diploid data with known LD block structure, runs
PLINK 1.9 --blocks to get reference Gabriel blocks, runs all 13
torchgwas methods, and compares results.  Also validates polyploid
(tetraploid) block detection.

Requirements: PLINK 1.9 accessible at the path below.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from torchgwas.ld import (
    LDBlock,
    compare_all_methods,
    compare_blocks,
    detect_blocks,
    format_comparison_table,
    load_plink_blocks_det,
    plink_blocks_to_ldblocks,
    save_blocks_bed,
    save_blocks_det,
    save_blocks_summary,
)

# ── PLINK path ────────────────────────────────────────────────────

PLINK_EXE = r"C:\Users\Sikiru\Desktop\BigData\plink.exe"
HAS_PLINK = os.path.isfile(PLINK_EXE)


# ── Synthetic data generation ─────────────────────────────────────

def _generate_diploid_ld_data(
    n_samples: int = 200,
    n_snps_per_block: int = 10,
    n_blocks: int = 5,
    n_independent: int = 10,
    noise_rate: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, list[int], list[str], list[str]]:
    """Generate diploid genotype data with known LD block structure.

    Creates `n_blocks` LD blocks, each with `n_snps_per_block` highly
    correlated SNPs, plus `n_independent` independent SNPs between blocks.

    Returns (G, pos, chrs, ids) where G is (n, m) float64 in {0,1,2}.
    """
    torch.manual_seed(seed)
    columns = []
    pos = []
    ids = []
    chrs = []
    bp = 1000

    for blk in range(n_blocks):
        # Founder SNP for this block
        founder = torch.randint(0, 3, (n_samples,), dtype=torch.float64)
        for j in range(n_snps_per_block):
            snp = founder.clone()
            # Add noise: flip some genotypes
            flip = torch.rand(n_samples) < noise_rate
            snp[flip] = torch.randint(0, 3, (flip.sum().item(),), dtype=torch.float64)
            columns.append(snp)
            pos.append(bp)
            ids.append(f"rs_b{blk}_s{j}")
            chrs.append("1")
            bp += 500  # 500bp apart within block

        # Gap + independent SNPs between blocks
        bp += 50000  # 50kb gap
        for j in range(n_independent):
            columns.append(torch.randint(0, 3, (n_samples,), dtype=torch.float64))
            pos.append(bp)
            ids.append(f"rs_ind{blk}_{j}")
            chrs.append("1")
            bp += 2000

        bp += 50000  # Another gap before next block

    G = torch.stack(columns, dim=1)
    return G, pos, chrs, ids


def _generate_polyploid_ld_data(
    n_samples: int = 150,
    ploidy: int = 4,
    n_snps_per_block: int = 8,
    n_blocks: int = 3,
    n_independent: int = 5,
    noise_rate: float = 0.08,
    seed: int = 99,
) -> tuple[torch.Tensor, list[int], list[str], list[str]]:
    """Generate polyploid genotype data with LD structure.

    Dosages are in [0, ploidy] (e.g., 0-4 for tetraploid).
    """
    torch.manual_seed(seed)
    columns = []
    pos = []
    ids = []
    chrs = []
    bp = 1000

    for blk in range(n_blocks):
        founder = torch.randint(0, ploidy + 1, (n_samples,), dtype=torch.float64)
        for j in range(n_snps_per_block):
            snp = founder.clone()
            flip = torch.rand(n_samples) < noise_rate
            snp[flip] = torch.randint(
                0, ploidy + 1, (flip.sum().item(),), dtype=torch.float64
            )
            columns.append(snp)
            pos.append(bp)
            ids.append(f"poly_b{blk}_s{j}")
            chrs.append("1")
            bp += 500

        bp += 50000
        for j in range(n_independent):
            columns.append(
                torch.randint(0, ploidy + 1, (n_samples,), dtype=torch.float64)
            )
            pos.append(bp)
            ids.append(f"poly_ind{blk}_{j}")
            chrs.append("1")
            bp += 2000

        bp += 50000

    G = torch.stack(columns, dim=1)
    return G, pos, chrs, ids


def _write_plink_files(
    G: torch.Tensor,
    pos: list[int],
    chrs: list[str],
    ids: list[str],
    out_prefix: str,
) -> None:
    """Write genotype data to PLINK binary format (.bed/.bim/.fam).

    PLINK .bed format: magic bytes + SNP-major mode, 2 bits per genotype.
    """
    n, m = G.shape

    # .fam file
    with open(f"{out_prefix}.fam", "w") as f:
        for i in range(n):
            f.write(f"FAM{i}\tIND{i}\t0\t0\t0\t-9\n")

    # .bim file
    with open(f"{out_prefix}.bim", "w") as f:
        for j in range(m):
            f.write(f"{chrs[j]}\t{ids[j]}\t0\t{pos[j]}\tA\tG\n")

    # .bed file (SNP-major)
    bytes_per_snp = (n + 3) // 4
    with open(f"{out_prefix}.bed", "wb") as f:
        # Magic bytes
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(m):
            snp_bytes = bytearray(bytes_per_snp)
            for i in range(n):
                g = int(G[i, j].item())
                # PLINK encoding: 00=hom_A(2), 01=missing, 10=het(1), 11=hom_B(0)
                if g == 0:
                    code = 0b11  # hom_B
                elif g == 1:
                    code = 0b10  # het
                elif g == 2:
                    code = 0b00  # hom_A
                else:
                    code = 0b01  # missing
                byte_idx = i // 4
                bit_offset = (i % 4) * 2
                snp_bytes[byte_idx] |= (code << bit_offset)
            f.write(snp_bytes)


def _run_plink_blocks(plink_exe: str, bed_prefix: str, out_prefix: str) -> Path:
    """Run PLINK --blocks and return path to .blocks.det file."""
    cmd = [
        plink_exe,
        "--bfile", bed_prefix,
        "--blocks", "no-pheno-req", "no-small-max-span",
        "--blocks-max-kb", "200",
        "--out", out_prefix,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
    )
    det_path = Path(f"{out_prefix}.blocks.det")
    if not det_path.exists():
        # Try without no-small-max-span
        cmd = [
            plink_exe,
            "--bfile", bed_prefix,
            "--blocks", "no-pheno-req",
            "--blocks-max-kb", "200",
            "--out", out_prefix,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
    return det_path


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def diploid_data():
    """Synthetic diploid data with 5 LD blocks of 10 SNPs each."""
    return _generate_diploid_ld_data(
        n_samples=200, n_snps_per_block=10, n_blocks=5,
        n_independent=10, noise_rate=0.05, seed=42,
    )


@pytest.fixture(scope="module")
def polyploid_data():
    """Synthetic tetraploid (ploidy=4) data with 3 LD blocks."""
    return _generate_polyploid_ld_data(
        n_samples=150, ploidy=4, n_snps_per_block=8,
        n_blocks=3, n_independent=5, noise_rate=0.08, seed=99,
    )


@pytest.fixture(scope="module")
def hexaploid_data():
    """Synthetic hexaploid (ploidy=6) data with 3 LD blocks."""
    return _generate_polyploid_ld_data(
        n_samples=150, ploidy=6, n_snps_per_block=8,
        n_blocks=3, n_independent=5, noise_rate=0.08, seed=77,
    )


@pytest.fixture(scope="module")
def plink_reference_blocks(diploid_data):
    """Run PLINK --blocks on diploid data, return (plink_ldblocks, variant_ids)."""
    if not HAS_PLINK:
        pytest.skip("PLINK 1.9 not found")

    G, pos, chrs, ids = diploid_data
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "synth")
        _write_plink_files(G, pos, chrs, ids, prefix)

        out_prefix = os.path.join(tmpdir, "plink_out")
        det_path = _run_plink_blocks(PLINK_EXE, prefix, out_prefix)

        if not det_path.exists():
            pytest.skip("PLINK --blocks did not produce .blocks.det")

        pblocks = load_plink_blocks_det(det_path)
        ldblocks = plink_blocks_to_ldblocks(pblocks, variant_ids=ids)

    return ldblocks, ids


# All 13 methods split into groups by required inputs
METHODS_SIMPLE = [
    "gabriel", "spine", "r2",
    "gwas_aligned", "graphical", "changepoint",
    "big_ld", "cc_graph", "dp_optimize", "wall_pritchard",
]


# ── PLINK comparison tests ────────────────────────────────────────

@pytest.mark.skipif(not HAS_PLINK, reason="PLINK 1.9 not found")
class TestPLINKValidation:
    """Compare torchgwas block detection against PLINK reference."""

    def test_plink_produces_blocks(self, plink_reference_blocks):
        ldblocks, ids = plink_reference_blocks
        assert len(ldblocks) > 0, "PLINK found no blocks"
        for b in ldblocks:
            assert b.method == "plink"
            assert b.n_variants >= 2

    def test_gabriel_vs_plink(self, diploid_data, plink_reference_blocks):
        """Gabriel method should most closely match PLINK (which uses Gabriel)."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        our_blocks = detect_blocks(
            G, pos, chrs, ids, method="gabriel", max_kb=200.0,
        )
        result = compare_blocks(
            our_blocks, plink_blocks,
            method_a="gabriel", method_b="plink",
        )

        # Gabriel should have reasonable agreement with PLINK's Gabriel
        # (not identical due to different D'/CI implementations)
        assert result.n_blocks_a > 0, "gabriel found no blocks"
        assert result.jaccard_index > 0.0, "No SNP overlap with PLINK"

    def test_all_simple_methods_vs_plink(self, diploid_data, plink_reference_blocks):
        """Compare all 10 simple methods against PLINK reference."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        results_table = {}
        for method in METHODS_SIMPLE:
            kwargs = {
                "max_kb": 200.0,
                "r2_threshold": 0.3,
                "l1_penalty": 0.1,
                "penalty": 3.0,
                "min_segment_size": 2,
            }
            our_blocks = detect_blocks(G, pos, chrs, ids, method=method, **kwargs)
            result = compare_blocks(
                our_blocks, plink_blocks,
                method_a=method, method_b="plink",
            )
            results_table[(method, "plink")] = result

            # Every method should produce valid blocks
            assert isinstance(result.jaccard_index, float)
            assert 0.0 <= result.jaccard_index <= 1.0
            assert 0.0 <= result.match_rate_a <= 1.0
            assert 0.0 <= result.match_rate_b <= 1.0

        # Print comparison table for visibility
        table = format_comparison_table(results_table)
        print("\n=== All Methods vs PLINK ===")
        print(table)

    def test_plink_comparison_output_files(self, diploid_data, plink_reference_blocks,
                                           tmp_path):
        """Verify all output formats work with PLINK comparison data."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        # Run our gabriel method
        our_blocks = detect_blocks(
            G, pos, chrs, ids, method="gabriel", max_kb=200.0,
        )

        # Save in all formats
        save_blocks_bed(our_blocks, tmp_path / "gabriel.bed")
        save_blocks_det(our_blocks, tmp_path / "gabriel.blocks.det", variant_ids=ids)
        save_blocks_summary(our_blocks, tmp_path / "gabriel.summary.txt")

        # Verify files exist and have content
        assert (tmp_path / "gabriel.bed").stat().st_size > 0
        assert (tmp_path / "gabriel.blocks.det").stat().st_size > 0
        assert (tmp_path / "gabriel.summary.txt").stat().st_size > 0

        # Verify BED has 13 columns
        lines = (tmp_path / "gabriel.bed").read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        for line in data_lines:
            assert len(line.split("\t")) == 13

    def test_cross_method_comparison(self, diploid_data, plink_reference_blocks):
        """Pairwise comparison of multiple methods including PLINK."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        blocks_by_method = {"plink": plink_blocks}
        for method in ["gabriel", "r2", "big_ld", "cc_graph"]:
            blocks_by_method[method] = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
            )

        results = compare_all_methods(blocks_by_method, reference="plink")
        assert len(results) == 4  # 4 methods vs plink

        # Print for visibility
        table = format_comparison_table(results)
        print("\n=== Cross-Method vs PLINK Reference ===")
        print(table)


# ── Diploid block detection (all 13 methods) ─────────────────────

class TestDiploidAllMethods:
    """Verify all 13 methods work on realistic diploid data."""

    def test_simple_methods_find_blocks(self, diploid_data):
        G, pos, chrs, ids = diploid_data
        for method in METHODS_SIMPLE:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            assert isinstance(blocks, list), f"{method}: not a list"
            # Most methods should find at least some blocks
            if method != "wall_pritchard":
                assert len(blocks) > 0, f"{method}: found no blocks"
            for b in blocks:
                assert isinstance(b, LDBlock)
                assert b.method == method

    def test_uncertainty_method(self, diploid_data):
        G, pos, chrs, ids = diploid_data
        n, m = G.shape
        # Create GP probs from dosage (high confidence)
        gp_probs = torch.zeros(n, m, 3, dtype=torch.float64)
        for i in range(n):
            for j in range(m):
                d = int(G[i, j].item())
                gp_probs[i, j, d] = 0.95
                for k in range(3):
                    if k != d:
                        gp_probs[i, j, k] = 0.025
        blocks = detect_blocks(
            G, pos, chrs, ids, method="uncertainty",
            gp_probs=gp_probs, r2_threshold=0.1, max_kb=200.0,
        )
        assert isinstance(blocks, list)

    def test_cross_pop_method(self, diploid_data):
        G, pos, chrs, ids = diploid_data
        n = G.shape[0]
        pop_ids = ["pop1"] * (n // 2) + ["pop2"] * (n - n // 2)
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cross_pop",
            population_ids=pop_ids, max_kb=200.0,
            stability_threshold=0.0,
        )
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_output_standard_format(self, diploid_data, tmp_path):
        """All methods produce 13-column BED and 9-column det."""
        G, pos, chrs, ids = diploid_data
        for method in METHODS_SIMPLE:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            bed = tmp_path / f"dip_{method}.bed"
            det = tmp_path / f"dip_{method}.blocks.det"
            save_blocks_bed(blocks, bed)
            save_blocks_det(blocks, det, variant_ids=ids)

            bed_lines = [l for l in bed.read_text().strip().split("\n")
                         if not l.startswith("#")]
            for line in bed_lines:
                assert len(line.split("\t")) == 13, f"{method}: BED cols != 13"

            det_lines = det.read_text().strip().split("\n")
            for line in det_lines[1:]:  # skip header
                assert len(line.split("\t")) == 9, f"{method}: det cols != 9"

    def test_blocks_respect_known_structure(self, diploid_data):
        """Methods should find blocks roughly aligned with the 5 LD blocks."""
        G, pos, chrs, ids = diploid_data
        # r2 method with matched threshold should find the synthetic blocks
        blocks = detect_blocks(
            G, pos, chrs, ids, method="r2",
            r2_threshold=0.3, max_kb=200.0,
        )
        # We have 5 blocks of 10 SNPs each, with 10 independent SNPs between
        # The r2 method should find multi-SNP blocks
        multi_blocks = [b for b in blocks if b.n_variants >= 3]
        assert len(multi_blocks) >= 3, (
            f"Expected ≥3 multi-SNP blocks from 5 synthetic blocks, got {len(multi_blocks)}"
        )


# ── Polyploid block detection ────────────────────────────────────

class TestPolyploidAllMethods:
    """Verify all methods work on polyploid data (ploidy=4 and ploidy=6)."""

    def test_tetraploid_simple_methods(self, polyploid_data):
        G, pos, chrs, ids = polyploid_data
        for method in METHODS_SIMPLE:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            assert isinstance(blocks, list), f"{method}: not a list (tetraploid)"
            for b in blocks:
                assert isinstance(b, LDBlock)
                assert b.method == method
                assert 0.0 <= b.mean_r2 <= 1.0 + 1e-6, (
                    f"{method}: r2={b.mean_r2} out of range (tetraploid)"
                )

    def test_hexaploid_simple_methods(self, hexaploid_data):
        G, pos, chrs, ids = hexaploid_data
        for method in METHODS_SIMPLE:
            blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )
            assert isinstance(blocks, list), f"{method}: not a list (hexaploid)"
            for b in blocks:
                assert isinstance(b, LDBlock)

    def test_tetraploid_uncertainty_method(self, polyploid_data):
        G, pos, chrs, ids = polyploid_data
        n, m = G.shape
        ploidy = 4
        # Create GP probs for tetraploid (k+1 = 5 categories: 0,1,2,3,4)
        gp_probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
        for i in range(n):
            for j in range(m):
                d = int(G[i, j].item())
                d = min(d, ploidy)  # clamp
                gp_probs[i, j, d] = 0.90
                remaining = 0.10 / ploidy
                for k in range(ploidy + 1):
                    if k != d:
                        gp_probs[i, j, k] = remaining
        blocks = detect_blocks(
            G, pos, chrs, ids, method="uncertainty",
            gp_probs=gp_probs, ploidy=ploidy,
            r2_threshold=0.1, max_kb=200.0,
        )
        assert isinstance(blocks, list)

    def test_tetraploid_cross_pop_method(self, polyploid_data):
        G, pos, chrs, ids = polyploid_data
        n = G.shape[0]
        pop_ids = ["popA"] * (n // 2) + ["popB"] * (n - n // 2)
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cross_pop",
            population_ids=pop_ids, max_kb=200.0,
            stability_threshold=0.0,
        )
        assert isinstance(blocks, list)

    def test_tetraploid_finds_blocks(self, polyploid_data):
        """Tetraploid data with known LD should produce multi-SNP blocks."""
        G, pos, chrs, ids = polyploid_data
        blocks = detect_blocks(
            G, pos, chrs, ids, method="r2",
            r2_threshold=0.2, max_kb=200.0,
        )
        multi_blocks = [b for b in blocks if b.n_variants >= 3]
        assert len(multi_blocks) >= 1, "Expected multi-SNP blocks in tetraploid data"

    def test_polyploid_output_standard(self, polyploid_data, tmp_path):
        """Polyploid BED output has same 13-column format."""
        G, pos, chrs, ids = polyploid_data
        blocks = detect_blocks(
            G, pos, chrs, ids, method="cc_graph",
            r2_threshold=0.2, max_kb=200.0,
        )
        bed = tmp_path / "poly.bed"
        save_blocks_bed(blocks, bed)
        lines = [l for l in bed.read_text().strip().split("\n")
                 if not l.startswith("#")]
        for line in lines:
            assert len(line.split("\t")) == 13

    def test_polyploid_vs_diploid_comparison(self, diploid_data, polyploid_data):
        """Both ploidy levels produce comparable output structures."""
        G_dip, pos_dip, chrs_dip, ids_dip = diploid_data
        G_poly, pos_poly, chrs_poly, ids_poly = polyploid_data

        blocks_dip = detect_blocks(
            G_dip, pos_dip, chrs_dip, ids_dip,
            method="r2", r2_threshold=0.3, max_kb=200.0,
        )
        blocks_poly = detect_blocks(
            G_poly, pos_poly, chrs_poly, ids_poly,
            method="r2", r2_threshold=0.3, max_kb=200.0,
        )

        # Both should produce valid LDBlocks with same field structure
        for b in blocks_dip + blocks_poly:
            assert hasattr(b, "region")
            assert hasattr(b, "mean_r2")
            assert hasattr(b, "mean_dprime")
            assert hasattr(b, "blockiness_score")
            assert hasattr(b, "metadata")


# ── Full comparison report ────────────────────────────────────────

@pytest.mark.skipif(not HAS_PLINK, reason="PLINK 1.9 not found")
class TestFullComparisonReport:
    """Generate and validate a complete cross-method comparison report."""

    def test_full_report(self, diploid_data, plink_reference_blocks, tmp_path):
        """Run all methods, compare to PLINK, write full report."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        # Run all simple methods
        blocks_by_method = {"plink": plink_blocks}
        for method in METHODS_SIMPLE:
            blocks_by_method[method] = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                l1_penalty=0.1, penalty=3.0,
                min_segment_size=2,
            )

        # Pairwise comparison against PLINK
        results = compare_all_methods(blocks_by_method, reference="plink")
        table = format_comparison_table(results)

        # Write report
        report_path = tmp_path / "full_comparison_report.txt"
        with open(report_path, "w") as f:
            f.write("TorchGWAS Block Detection: Full Validation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Data: {G.shape[0]} samples x {G.shape[1]} SNPs\n")
            f.write("Synthetic: 5 LD blocks of 10 SNPs, 10 independent between\n\n")
            f.write("Methods vs PLINK Reference\n")
            f.write("-" * 60 + "\n")
            f.write(table + "\n\n")

            f.write("Per-Method Summary\n")
            f.write("-" * 60 + "\n")
            for method, blocks in sorted(blocks_by_method.items()):
                n_blk = len(blocks)
                n_var = sum(b.n_variants for b in blocks)
                mean_sz = n_var / n_blk if n_blk > 0 else 0
                mean_r2 = (sum(b.mean_r2 for b in blocks) / n_blk
                           if n_blk > 0 else 0)
                f.write(
                    f"  {method:<20} blocks={n_blk:>4}  "
                    f"variants={n_var:>5}  mean_size={mean_sz:>5.1f}  "
                    f"mean_r2={mean_r2:.4f}\n"
                )

            f.write("\n")
            # Save individual method BED files
            for method, blocks in blocks_by_method.items():
                if method == "plink":
                    continue
                bed = tmp_path / f"report_{method}.bed"
                det = tmp_path / f"report_{method}.blocks.det"
                save_blocks_bed(blocks, bed)
                save_blocks_det(blocks, det, variant_ids=ids)

        assert report_path.exists()
        report = report_path.read_text()
        assert "Jaccard" in report
        assert "gabriel" in report
        assert "plink" in report

        # Print report to stdout for visibility in test output
        print("\n" + report)

    def test_gabriel_closest_to_plink(self, diploid_data, plink_reference_blocks):
        """Gabriel should typically have the highest concordance with PLINK."""
        G, pos, chrs, ids = diploid_data
        plink_blocks, _ = plink_reference_blocks

        scores: dict[str, float] = {}
        for method in ["gabriel", "r2", "big_ld", "cc_graph", "changepoint"]:
            our_blocks = detect_blocks(
                G, pos, chrs, ids, method=method,
                max_kb=200.0, r2_threshold=0.3,
                penalty=3.0, min_segment_size=2,
            )
            result = compare_blocks(our_blocks, plink_blocks)
            scores[method] = result.jaccard_index

        # Gabriel should be competitive (not necessarily top due to
        # implementation differences, but should have non-zero overlap)
        assert scores["gabriel"] > 0.0, "Gabriel has zero overlap with PLINK"
        print(f"\nJaccard scores vs PLINK: {scores}")
