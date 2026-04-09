"""Tests for torchgwas.pgs.ld_ref."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgwas.pgs.ld_ref import (
    _ledoit_wolf_shrink,
    build_ld_reference,
    load_ld_reference,
    save_ld_reference,
)


def _sim_genotypes(n=200, m=10, seed=0, ploidy=2):
    g = torch.Generator().manual_seed(seed)
    af = 0.1 + 0.7 * torch.rand(m, generator=g, dtype=torch.float64)
    G = torch.binomial(
        torch.full((n, m), float(ploidy), dtype=torch.float64),
        af.expand(n, m),
        generator=g,
    )
    return G, af


def _meta(m: int) -> dict:
    return {
        "snp": [f"rs{i}" for i in range(m)],
        "chr": ["1"] * m,
        "pos": list(range(100, 100 + m)),
        "a1": ["A"] * m,
        "a2": ["G"] * m,
    }


def test_build_full_ld_diagonal_is_one():
    G, _ = _sim_genotypes(n=300, m=8, seed=1)
    ld = build_ld_reference(G, _meta(8), mode="full")
    assert ld.mode == "full"
    assert ld.R_full.shape == (8, 8)
    diag = torch.diag(ld.R_full)
    assert torch.allclose(diag, torch.ones(8, dtype=torch.float64), atol=1e-10)
    # symmetric
    assert torch.allclose(ld.R_full, ld.R_full.T)
    # AF in (0, 1)
    assert (ld.af > 0).all() and (ld.af < 1).all()


def test_build_full_matches_numpy_corrcoef():
    G, _ = _sim_genotypes(n=200, m=6, seed=2)
    ld = build_ld_reference(G, _meta(6), mode="full")
    R_np = np.corrcoef(G.numpy(), rowvar=False)
    assert torch.allclose(
        ld.R_full, torch.tensor(R_np, dtype=torch.float64), atol=1e-10
    )


def test_block_mode_equals_full_when_one_block():
    G, _ = _sim_genotypes(n=200, m=6, seed=3)
    full = build_ld_reference(G, _meta(6), mode="full")
    block = build_ld_reference(
        G, _meta(6), mode="block", block_assignments=[0] * 6
    )
    assert block.mode == "block"
    assert len(block.R_blocks) == 1
    assert torch.allclose(block.R_blocks[0], full.R_full, atol=1e-12)


def test_block_mode_distinct_blocks():
    G, _ = _sim_genotypes(n=200, m=6, seed=4)
    block = build_ld_reference(
        G, _meta(6), mode="block", block_assignments=[0, 0, 0, 1, 1, 1]
    )
    assert len(block.R_blocks) == 2
    assert block.R_blocks[0].shape == (3, 3)
    assert block.R_blocks[1].shape == (3, 3)
    # within-block: matches np.corrcoef of the corresponding columns
    R0 = np.corrcoef(G[:, :3].numpy(), rowvar=False)
    R1 = np.corrcoef(G[:, 3:].numpy(), rowvar=False)
    assert torch.allclose(block.R_blocks[0], torch.tensor(R0, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(block.R_blocks[1], torch.tensor(R1, dtype=torch.float64), atol=1e-10)


def test_ledoit_wolf_shrinkage_increases_diag_dominance():
    G, _ = _sim_genotypes(n=80, m=10, seed=5)
    R = build_ld_reference(G, _meta(10), mode="full").R_full
    R_shrunk = _ledoit_wolf_shrink(R, intensity=0.5)
    # mixture identity
    expected = 0.5 * R + 0.5 * torch.eye(10, dtype=torch.float64)
    assert torch.allclose(R_shrunk, expected, atol=1e-12)
    # off-diagonal magnitudes shrink toward zero
    off = R - torch.diag(torch.diag(R))
    off_s = R_shrunk - torch.diag(torch.diag(R_shrunk))
    assert off_s.abs().sum() < off.abs().sum()


def test_save_load_roundtrip_full(tmp_path):
    G, _ = _sim_genotypes(n=150, m=5, seed=6)
    ld = build_ld_reference(G, _meta(5), mode="full", shrinkage=0.1)
    p = tmp_path / "ld_full.pt"
    save_ld_reference(ld, str(p))
    ld2 = load_ld_reference(str(p))
    assert ld2.mode == "full"
    assert ld2.snp == ld.snp
    assert torch.allclose(ld2.R_full, ld.R_full)
    assert torch.allclose(ld2.af, ld.af)
    assert ld2.n_ref == ld.n_ref


def test_save_load_roundtrip_block(tmp_path):
    G, _ = _sim_genotypes(n=150, m=8, seed=7)
    ld = build_ld_reference(
        G, _meta(8), mode="block", block_assignments=[0, 0, 0, 1, 1, 2, 2, 2]
    )
    p = tmp_path / "ld_block.pt"
    save_ld_reference(ld, str(p))
    ld2 = load_ld_reference(str(p))
    assert ld2.mode == "block"
    assert len(ld2.R_blocks) == 3
    for a, b in zip(ld.R_blocks, ld2.R_blocks):
        assert torch.allclose(a, b)
    assert torch.equal(ld2.block_index, ld.block_index)


def test_build_ld_reference_validates_meta_lengths():
    G, _ = _sim_genotypes(n=50, m=4, seed=8)
    bad_meta = _meta(4)
    bad_meta["snp"] = ["rs0", "rs1"]  # wrong length
    with pytest.raises(ValueError, match="length"):
        build_ld_reference(G, bad_meta, mode="full")
