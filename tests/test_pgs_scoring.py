"""Tests for torchgenomics.pgs.scoring — score_individuals."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.pgs.base import PGSResult
from torchgenomics.pgs.scoring import score_individuals


def _make_result(snp, a1, a2, w):
    m = len(snp)
    return PGSResult(
        method="test",
        snp=list(snp),
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        a1=list(a1),
        a2=list(a2),
        weight=torch.tensor(w, dtype=torch.float64),
    )


def test_score_matches_naive_dot_product():
    n = 5
    G = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    res = _make_result(["rs1", "rs2", "rs3"], ["A", "A", "A"], ["G", "G", "G"], [0.5, -0.2, 0.1])
    out = score_individuals(G, ["rs1", "rs2", "rs3"], ["A", "A", "A"], ["G", "G", "G"], res)
    expected = G @ torch.tensor([0.5, -0.2, 0.1], dtype=torch.float64)
    assert torch.allclose(out.pgs, expected)
    assert out.n_snp_used == 3
    assert out.n_snp_missing == 0
    assert out.n_snp_flipped == 0


def test_score_flips_swapped_alleles():
    G = torch.tensor([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    # target rs1 is swapped relative to result
    res = _make_result(["rs1", "rs2"], ["A", "A"], ["G", "G"], [0.5, 0.3])
    out = score_individuals(
        G,
        ["rs1", "rs2"],
        ["G", "A"],  # rs1 swapped
        ["A", "G"],
        res,
    )
    # Effective weight for rs1 becomes -0.5
    expected = G @ torch.tensor([-0.5, 0.3], dtype=torch.float64)
    assert torch.allclose(out.pgs, expected)
    assert out.n_snp_flipped == 1


def test_score_intersects_and_counts_missing_snps():
    G = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0]], dtype=torch.float64)
    # weight file has an extra SNP not present in target
    res = _make_result(
        ["rs1", "rs2", "rs99"],
        ["A", "A", "A"],
        ["G", "G", "G"],
        [0.5, -0.2, 0.9],
    )
    out = score_individuals(
        G,
        ["rs1", "rs2", "rs3"],
        ["A", "A", "A"],
        ["G", "G", "G"],
        res,
    )
    assert out.n_snp_used == 2
    assert out.n_snp_missing == 1  # rs99 missing in target
    expected = G[:, :2] @ torch.tensor([0.5, -0.2], dtype=torch.float64)
    assert torch.allclose(out.pgs, expected)


def test_score_mean_imputes_nan_genotypes():
    G = torch.tensor(
        [
            [0.0, 1.0],
            [float("nan"), 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    res = _make_result(["rs1", "rs2"], ["A", "A"], ["G", "G"], [1.0, 0.5])
    out = score_individuals(G, ["rs1", "rs2"], ["A", "A"], ["G", "G"], res, handle_missing="mean")
    # rs1 mean over non-nan = (0 + 2)/2 = 1.0
    G_imp = G.clone()
    G_imp[1, 0] = 1.0
    expected = G_imp @ torch.tensor([1.0, 0.5], dtype=torch.float64)
    assert torch.allclose(out.pgs, expected)
    assert out.n_snp_used == 2


def test_score_drops_snps_with_nan_under_drop_mode():
    G = torch.tensor(
        [
            [0.0, 1.0],
            [float("nan"), 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    res = _make_result(["rs1", "rs2"], ["A", "A"], ["G", "G"], [1.0, 0.5])
    out = score_individuals(G, ["rs1", "rs2"], ["A", "A"], ["G", "G"], res, handle_missing="drop")
    assert out.n_snp_used == 1
    assert out.snp_used == ["rs2"]
    expected = G[:, 1] * 0.5
    assert torch.allclose(out.pgs, expected)


def test_score_chunked_matches_full():
    torch.manual_seed(0)
    n, m = 20, 17
    G = torch.randn(n, m, dtype=torch.float64).abs().clamp(max=2.0)
    snp = [f"rs{i}" for i in range(m)]
    res = _make_result(snp, ["A"] * m, ["G"] * m, torch.randn(m).tolist())
    r_full = score_individuals(G, snp, ["A"] * m, ["G"] * m, res, chunk_size=1000)
    r_chunk = score_individuals(G, snp, ["A"] * m, ["G"] * m, res, chunk_size=3)
    assert torch.allclose(r_full.pgs, r_chunk.pgs)


def test_score_standardize_zero_centers():
    G = torch.tensor(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], dtype=torch.float64
    )
    res = _make_result(["rs1", "rs2"], ["A", "A"], ["G", "G"], [1.0, 0.0])
    out = score_individuals(G, ["rs1", "rs2"], ["A", "A"], ["G", "G"], res, standardize=True)
    # rs2 weight is zero so PGS should sum to zero across individuals
    assert float(out.pgs.sum()) == pytest.approx(0.0, abs=1e-10)


def test_score_empty_intersection_returns_zero_pgs():
    G = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
    res = _make_result(["rs99", "rs100"], ["A", "A"], ["G", "G"], [0.5, 0.5])
    out = score_individuals(G, ["rs1", "rs2"], ["A", "A"], ["G", "G"], res)
    assert out.n_snp_used == 0
    assert out.n_snp_missing == 2
    assert torch.allclose(out.pgs, torch.zeros(2, dtype=torch.float64))
