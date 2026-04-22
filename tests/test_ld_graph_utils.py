"""Tests for LD graph utilities: Laplacian, spectral partition, CC, MWIS, changepoint."""

from __future__ import annotations

import torch

from torchgwas.ld._changepoint import dp_changepoint, ld_decay_signal
from torchgwas.ld._graph_utils import (
    adjacency_to_laplacian,
    connected_components,
    greedy_mwis,
    normalized_laplacian,
    spectral_partition,
)

# ── Laplacian tests ────────────────────────────────────────────────

class TestLaplacian:
    def test_row_sums_zero(self):
        A = torch.tensor([
            [0.0, 0.5, 0.3],
            [0.5, 0.0, 0.8],
            [0.3, 0.8, 0.0],
        ], dtype=torch.float64)
        L = adjacency_to_laplacian(A)
        assert torch.allclose(L.sum(dim=1), torch.zeros(3, dtype=torch.float64), atol=1e-10)

    def test_normalized_eigenvalues_in_range(self):
        A = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=torch.float64)
        L = normalized_laplacian(A)
        eigvals = torch.linalg.eigvalsh(L)
        assert (eigvals >= -1e-10).all()
        assert (eigvals <= 2.0 + 1e-10).all()

    def test_laplacian_symmetric(self):
        A = torch.rand(5, 5, dtype=torch.float64)
        A = (A + A.T) / 2
        A.fill_diagonal_(0.0)
        L = adjacency_to_laplacian(A)
        assert torch.allclose(L, L.T, atol=1e-10)


# ── Spectral partition tests ───────────────────────────────────────

class TestSpectralPartition:
    def test_two_clusters_block_diagonal(self):
        """Block-diagonal adjacency should produce 2 clusters."""
        A = torch.zeros(6, 6, dtype=torch.float64)
        # Block 1: nodes 0,1,2
        A[0, 1] = A[1, 0] = 1.0
        A[0, 2] = A[2, 0] = 1.0
        A[1, 2] = A[2, 1] = 1.0
        # Block 2: nodes 3,4,5
        A[3, 4] = A[4, 3] = 1.0
        A[3, 5] = A[5, 3] = 1.0
        A[4, 5] = A[5, 4] = 1.0

        clusters = spectral_partition(A, n_clusters=2)
        assert len(clusters) == 2
        flat = sorted([sorted(c) for c in clusters])
        assert flat == [[0, 1, 2], [3, 4, 5]]

    def test_single_cluster(self):
        """Fully connected graph should produce 1 cluster with k=1."""
        A = torch.ones(4, 4, dtype=torch.float64)
        A.fill_diagonal_(0.0)
        clusters = spectral_partition(A, n_clusters=1)
        assert len(clusters) >= 1
        all_nodes = sorted(n for c in clusters for n in c)
        assert all_nodes == [0, 1, 2, 3]


# ── Connected components tests ─────────────────────────────────────

class TestConnectedComponents:
    def test_disconnected_graph(self):
        A = torch.zeros(4, 4, dtype=torch.float64)
        A[0, 1] = A[1, 0] = 1.0
        A[2, 3] = A[3, 2] = 1.0
        components = connected_components(A)
        assert len(components) == 2
        assert sorted(components[0]) == [0, 1]
        assert sorted(components[1]) == [2, 3]

    def test_fully_connected(self):
        A = torch.ones(3, 3, dtype=torch.float64)
        A.fill_diagonal_(0.0)
        components = connected_components(A)
        assert len(components) == 1
        assert components[0] == [0, 1, 2]

    def test_threshold_disconnects(self):
        A = torch.tensor([
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ], dtype=torch.float64)
        components = connected_components(A, threshold=0.5)
        assert len(components) == 2


# ── MWIS tests ─────────────────────────────────────────────────────

class TestMWIS:
    def test_non_overlapping_selection(self):
        intervals = [(0, 2, 5.0), (3, 5, 3.0), (1, 4, 10.0)]
        selected = greedy_mwis(intervals)
        # Should select (1,4,10) as highest weight, then nothing else overlaps
        # Actually (0,2,5) and (3,5,3) don't overlap with each other
        # but (1,4,10) overlaps with both. Since 10 > 5+3=8, we pick (1,4,10)
        assert len(selected) >= 1
        # Verify no overlaps
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                s1, e1, _ = selected[i]
                s2, e2, _ = selected[j]
                assert e1 < s2 or e2 < s1

    def test_weight_ordering(self):
        intervals = [(0, 1, 10.0), (2, 3, 5.0), (4, 5, 1.0)]
        selected = greedy_mwis(intervals)
        assert len(selected) == 3  # All non-overlapping

    def test_empty_input(self):
        assert greedy_mwis([]) == []


# ── Change-point tests ─────────────────────────────────────────────

class TestChangepoint:
    def test_single_changepoint(self):
        """Signal with clear level shift should detect 1 change-point."""
        signal = torch.cat([
            torch.ones(20, dtype=torch.float64) * 0.8,
            torch.ones(20, dtype=torch.float64) * 0.2,
        ])
        cps = dp_changepoint(signal, penalty=1.0, min_seg=5)
        assert len(cps) >= 1
        # Change-point should be near index 20
        assert any(15 <= cp <= 25 for cp in cps)

    def test_no_changepoints(self):
        """Constant signal with high penalty should have no change-points."""
        signal = torch.ones(30, dtype=torch.float64) * 0.5
        cps = dp_changepoint(signal, penalty=100.0, min_seg=5)
        assert len(cps) == 0

    def test_multiple_changepoints(self):
        """Signal with 2 level shifts."""
        signal = torch.cat([
            torch.ones(15, dtype=torch.float64) * 0.9,
            torch.ones(15, dtype=torch.float64) * 0.1,
            torch.ones(15, dtype=torch.float64) * 0.9,
        ])
        cps = dp_changepoint(signal, penalty=3.0, min_seg=5)
        assert len(cps) >= 2

    def test_penalty_controls_sensitivity(self):
        """Higher penalty should produce fewer change-points."""
        signal = torch.cat([
            torch.ones(10, dtype=torch.float64) * (0.8 if i % 2 == 0 else 0.2)
            for i in range(6)
        ])
        cps_low = dp_changepoint(signal, penalty=1.0, min_seg=3)
        cps_high = dp_changepoint(signal, penalty=50.0, min_seg=3)
        assert len(cps_high) <= len(cps_low)

    def test_ld_decay_signal(self):
        """LD decay signal should have correct shape."""
        r2 = torch.rand(9, dtype=torch.float64)
        idx_i = torch.arange(9, dtype=torch.long)
        idx_j = torch.arange(1, 10, dtype=torch.long)
        signal = ld_decay_signal(r2, idx_i, idx_j, n_variants=10)
        assert signal.shape == (10,)
        assert (signal >= 0).all()
