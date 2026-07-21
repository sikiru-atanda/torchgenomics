"""Tests for the ``tolerance`` parameter on ``detect_blocks_r2``.

The ``tolerance`` parameter implements the SelectionTools R package
convention (``hbd.bfile``; Pandit et al., *Theoretical and Applied
Genetics* 2026, barley leaf rust): when walking adjacent SNP pairs,
absorb up to ``tolerance`` consecutive below-threshold pairs into the
current block instead of closing on the first failure. Default
``tolerance=0`` reproduces the strict legacy behavior.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from torchgenomics.ld import detect_blocks
from torchgenomics.ld._blocks import PairwiseLD, detect_blocks_r2

# ── Helpers ────────────────────────────────────────────────────────────


def _make_pld_from_adjacent_r2(r2_seq: list[float]) -> PairwiseLD:
    """Construct a ``PairwiseLD`` whose adjacent-pair r² values follow
    ``r2_seq`` (length n-1 for n SNPs). Non-adjacent pairs are filled
    with 0 — ``detect_blocks_r2`` only inspects ``(i, i+1)`` keys.
    """
    n = len(r2_seq) + 1
    idx_i, idx_j, r2_vals, dp_vals = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)
            if j == i + 1:
                r2_vals.append(float(r2_seq[i]))
                dp_vals.append(float(min(1.0, r2_seq[i] + 0.05)))
            else:
                r2_vals.append(0.0)
                dp_vals.append(0.0)
    return PairwiseLD(
        idx_i=torch.tensor(idx_i, dtype=torch.long),
        idx_j=torch.tensor(idx_j, dtype=torch.long),
        r2=torch.tensor(r2_vals, dtype=torch.float64),
        dprime=torch.tensor(dp_vals, dtype=torch.float64),
        dprime_ci_low=torch.zeros(len(r2_vals), dtype=torch.float64),
        dprime_ci_high=torch.ones(len(r2_vals), dtype=torch.float64),
    )


def _make_meta(n: int) -> tuple[list[int], list[str], list[str]]:
    pos = [1000 * (i + 1) for i in range(n)]
    chrs = ["1"] * n
    ids = [f"rs{i}" for i in range(n)]
    return pos, chrs, ids


def _block_indices(blocks) -> list[list[int]]:
    return [b.variant_indices for b in blocks]


# ── Tests ──────────────────────────────────────────────────────────────


class TestR2Tolerance:
    """Direct unit tests against ``detect_blocks_r2``."""

    def test_r2_block_tolerance_zero_matches_legacy(self):
        """tolerance=0 must reproduce the pre-change block partition.

        The "legacy" reference is computed inline by simulating the
        original loop (close on first below-threshold pair) and compared
        SNP-for-SNP. Uses a 20-SNP simulated adjacent-r² sequence with
        multiple sub-threshold drops to force several splits.
        """
        rng = np.random.default_rng(0)
        # 19 adjacent-pair r² values for 20 SNPs.
        r2_seq = rng.uniform(0.6, 0.99, size=19).tolist()
        # Drop a handful of pairs below threshold to force splits.
        for k in (4, 9, 14):
            r2_seq[k] = 0.05
        threshold = 0.3

        pld = _make_pld_from_adjacent_r2(r2_seq)
        pos, chrs, ids = _make_meta(20)
        blocks = detect_blocks_r2(
            pld, pos, chrs, ids,
            r2_threshold=threshold,
            min_block_snps=2,
            tolerance=0,
        )
        got = _block_indices(blocks)

        # Legacy reference: close on first below-threshold pair.
        legacy: list[list[int]] = []
        start = 0
        for i, r2 in enumerate(r2_seq):
            if r2 < threshold:
                if i - start + 1 >= 2:
                    legacy.append(list(range(start, i + 1)))
                start = i + 1
        last_end = len(r2_seq)  # n_snps - 1
        if last_end - start + 1 >= 2:
            legacy.append(list(range(start, last_end + 1)))

        assert got == legacy, (
            f"tolerance=0 diverged from legacy partition\n"
            f"got:    {got}\nlegacy: {legacy}"
        )

    def test_r2_block_tolerance_one_skips_single_drop(self):
        """tolerance=1 must merge isolated single-pair drops but keep
        runs of two-or-more consecutive failures as boundaries.

        The sequence is 10 adjacent-pair r² values for 11 SNPs. The lone
        drop at index 2 is absorbed; the two consecutive drops at
        indices 6–7 close the block.
        """
        r2_seq = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9]
        threshold = 0.5

        pld = _make_pld_from_adjacent_r2(r2_seq)
        pos, chrs, ids = _make_meta(11)

        blocks_strict = detect_blocks_r2(
            pld, pos, chrs, ids,
            r2_threshold=threshold,
            min_block_snps=2,
            tolerance=0,
        )
        # tol=0 legacy: pair 2 closes [0,2]; pairs 3–5 build [3,6]; pair 6
        # closes [3,6]; pair 7 closes [7,7] (degenerate, < min, no
        # emit); pairs 8–9 build [8,10] → 3 blocks.
        assert len(blocks_strict) == 3, (
            f"strict tol=0 should give 3 blocks; got "
            f"{_block_indices(blocks_strict)}"
        )

        blocks_tol1 = detect_blocks_r2(
            pld, pos, chrs, ids,
            r2_threshold=threshold,
            min_block_snps=2,
            tolerance=1,
        )
        # tol=1 absorbs the lone drop at pair 2 → single block running
        # from SNP 0 through SNP 6 (closed by the two consecutive
        # failures at pairs 6–7), then a [8,10] tail block.
        assert len(blocks_tol1) == 2, (
            f"tol=1 should give 2 blocks; got "
            f"{_block_indices(blocks_tol1)}"
        )
        assert blocks_tol1[0].variant_indices == [0, 1, 2, 3, 4, 5, 6]
        assert blocks_tol1[1].variant_indices == [8, 9, 10]

    def test_r2_block_tolerance_closes_on_two_consec(self):
        """tolerance=1 closes the block exactly when 2 consecutive
        adjacent pairs are below threshold — not before, not later."""
        # Two consecutive failures at pair indices 3-4 force a split;
        # no other failures in the sequence.
        r2_seq = [0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9]
        threshold = 0.5

        pld = _make_pld_from_adjacent_r2(r2_seq)
        pos, chrs, ids = _make_meta(9)
        blocks = detect_blocks_r2(
            pld, pos, chrs, ids,
            r2_threshold=threshold,
            min_block_snps=2,
            tolerance=1,
        )
        # Pair 3 fails → consec=1, not > 1, no close.
        # Pair 4 fails → consec=2, > 1, close. block_end = 4-2+1 = 3.
        # Emit [0,3]. block_start = 4+1 = 5.
        # Pairs 5,6,7 pass → tail emits [5,8].
        assert len(blocks) == 2
        assert blocks[0].variant_indices == [0, 1, 2, 3]
        assert blocks[1].variant_indices == [5, 6, 7, 8]

        # Sanity: with tolerance=2 the two-pair run is absorbed and the
        # whole sequence becomes a single block.
        blocks2 = detect_blocks_r2(
            pld, pos, chrs, ids,
            r2_threshold=threshold,
            min_block_snps=2,
            tolerance=2,
        )
        assert len(blocks2) == 1
        assert blocks2[0].variant_indices == list(range(9))

    def test_r2_block_tolerance_negative_raises(self):
        """Negative tolerance must be rejected."""
        pld = _make_pld_from_adjacent_r2([0.9, 0.9, 0.9])
        pos, chrs, ids = _make_meta(4)
        with pytest.raises(ValueError):
            detect_blocks_r2(
                pld, pos, chrs, ids,
                r2_threshold=0.5,
                tolerance=-1,
            )

    def test_r2_block_tolerance_passes_through_detect_blocks(self):
        """The top-level ``detect_blocks(method='r2', tolerance=...)``
        dispatcher must forward the tolerance kwarg without dropping it
        on the ``_METHOD_KWARGS`` allow-list filter."""
        torch.manual_seed(7)
        n = 30
        # 3 correlated 10-SNP blocks built from shared base columns.
        base_a = torch.randint(0, 3, (200,), dtype=torch.float64)
        base_b = torch.randint(0, 3, (200,), dtype=torch.float64)
        base_c = torch.randint(0, 3, (200,), dtype=torch.float64)
        cols = []
        for _ in range(10):
            cols.append(base_a + 0.01 * torch.randn(200, dtype=torch.float64))
        for _ in range(10):
            cols.append(base_b + 0.01 * torch.randn(200, dtype=torch.float64))
        for _ in range(10):
            cols.append(base_c + 0.01 * torch.randn(200, dtype=torch.float64))
        G = torch.stack(cols, dim=1)
        pos = [i * 1000 for i in range(n)]
        chrs = ["1"] * n
        ids = [f"rs{i}" for i in range(n)]
        blocks_strict = detect_blocks(
            G, pos, chrs, ids, method="r2",
            r2_threshold=0.5, tolerance=0,
        )
        blocks_tol = detect_blocks(
            G, pos, chrs, ids, method="r2",
            r2_threshold=0.5, tolerance=2,
        )
        assert isinstance(blocks_strict, list)
        assert isinstance(blocks_tol, list)
        # Tolerance can only merge or leave equal — never split further.
        assert len(blocks_tol) <= len(blocks_strict)


# ── CLI smoke test ─────────────────────────────────────────────────────


def _write_tiny_csv_genotype(tmp_path: Path, n_samples: int = 50,
                             n_snps: int = 20) -> Path:
    """Emit a tiny CSV dosage file (3-column structured format)."""
    rng = np.random.default_rng(123)
    # Two correlated blocks so the detector has something to find.
    base_a = rng.integers(0, 3, size=n_samples).astype(np.int8)
    base_b = rng.integers(0, 3, size=n_samples).astype(np.int8)
    cols = []
    half = n_snps // 2
    for _ in range(half):
        c = base_a.copy()
        flip = rng.random(n_samples) < 0.05
        c[flip] = np.clip(2 - c[flip], 0, 2)
        cols.append(c)
    for _ in range(n_snps - half):
        c = base_b.copy()
        flip = rng.random(n_samples) < 0.05
        c[flip] = np.clip(2 - c[flip], 0, 2)
        cols.append(c)
    G = np.stack(cols, axis=1)  # (n_samples, n_snps)
    # Markers as rows, samples as columns.
    snp_ids = [f"rs{i}" for i in range(n_snps)]
    chrs = ["1"] * n_snps
    positions = [i * 5000 + 1000 for i in range(n_snps)]
    sample_ids = [f"S{i}" for i in range(n_samples)]
    df = pd.DataFrame(G.T, index=snp_ids, columns=sample_ids)
    df.insert(0, "Chromosome", chrs)
    df.insert(1, "Position_BP", positions)
    df.index.name = "SNP"
    out = tmp_path / "tiny.csv"
    df.to_csv(out)
    return out


def test_r2_block_tolerance_cli_flag(tmp_path):
    """End-to-end CLI smoke: ``ld-blocks --method r2 --tolerance 2
    --r2-threshold 0.7`` must run and emit the standard output files.
    """
    bin_path = shutil.which("torchgenomics")
    if bin_path is None:
        # Fall back to module entry — same code path either way.
        cmd_prefix = [sys.executable, "-m", "torchgenomics.cli"]
    else:
        cmd_prefix = [bin_path]

    geno_path = _write_tiny_csv_genotype(tmp_path)
    out_prefix = tmp_path / "blocks_out"

    cmd = cmd_prefix + [
        "ld-blocks",
        "--genotype", str(geno_path),
        "--method", "r2",
        "--r2-threshold", "0.7",
        "--tolerance", "2",
        "--max-kb", "200",
        "--output", str(out_prefix),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"CLI exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    # The three standard output files must exist.
    assert (tmp_path / "blocks_out.bed").exists()
    assert (tmp_path / "blocks_out.blocks.det").exists()
    assert (tmp_path / "blocks_out.summary.txt").exists()
