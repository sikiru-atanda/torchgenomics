"""Tests for LD-window peak pruning (charter Section 9g)."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.models.base import ScanResult
from torchgenomics.stats.peak_pruning import prune_peaks


def _make_scan_result(chrs, positions, p_values):
    m = len(p_values)
    return ScanResult(
        chr=chrs,
        pos=positions,
        snp=[f"snp_{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.5),
        beta=torch.zeros(m),
        se=torch.ones(m),
        stat=torch.ones(m),
        p=torch.tensor(p_values, dtype=torch.float64),
        test="wald",
    )


class TestPrunePeaks:
    def test_single_peak(self):
        """Single significant marker returns one peak."""
        sr = _make_scan_result(["1"], [1000], [1e-6])
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 1
        assert peaks[0].snp == "snp_0"

    def test_two_peaks_same_chr_far_apart(self):
        """Two markers on same chr but > bp_window apart → two peaks."""
        sr = _make_scan_result(
            ["1", "1"],
            [1_000_000, 5_000_000],
            [1e-6, 1e-5],
        )
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 2

    def test_cluster_collapses_to_one(self):
        """Three markers within bp_window on same chr → one peak (the best)."""
        sr = _make_scan_result(
            ["1", "1", "1"],
            [1_000_000, 1_000_500, 1_001_000],
            [1e-5, 1e-8, 1e-6],
        )
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 1
        assert peaks[0].snp == "snp_1"  # best p = 1e-8
        assert peaks[0].n_markers_in_window == 3

    def test_different_chromosomes_independent(self):
        """Markers on different chromosomes are always independent."""
        sr = _make_scan_result(
            ["1", "2", "3"],
            [1_000_000, 1_000_000, 1_000_000],
            [1e-6, 1e-5, 1e-7],
        )
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 3

    def test_no_significant_returns_empty(self):
        """No markers below threshold → empty list."""
        sr = _make_scan_result(["1", "1"], [1000, 2000], [0.5, 0.8])
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 0

    def test_best_model_labels(self):
        """Best model labels are attached to peaks."""
        sr = _make_scan_result(["1", "1"], [1000, 5_000_000], [1e-6, 1e-5])
        peaks = prune_peaks(
            sr, bp_window=1_000_000, p_threshold=1e-4,
            best_models=["additive", "1-dom"],
        )
        assert peaks[0].best_model == "additive"
        assert peaks[1].best_model == "1-dom"

    def test_sorted_by_neglog10p(self):
        """Peaks returned sorted by significance (most significant first)."""
        sr = _make_scan_result(
            ["1", "2", "3"],
            [1000, 1000, 1000],
            [1e-5, 1e-8, 1e-3],
        )
        peaks = prune_peaks(sr, bp_window=1_000_000, p_threshold=1e-2)
        assert peaks[0].neglog10p > peaks[1].neglog10p
        assert peaks[1].neglog10p > peaks[2].neglog10p

    def test_best_models_length_mismatch_raises(self):
        sr = _make_scan_result(["1"], [1000], [1e-6])
        with pytest.raises(ValueError, match="best_models length"):
            prune_peaks(sr, best_models=["additive", "1-dom"])
