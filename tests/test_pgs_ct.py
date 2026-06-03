"""Tests for torchgenomics.pgs.ct — Clumping + Thresholding."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.pgs.ct import ClumpingThresholding
from torchgenomics.pgs.ld_ref import build_ld_reference
from torchgenomics.postgwas._sumstats import SumStats


def _sim_two_loci(n=300, seed=0):
    """Simulate 6 SNPs in 2 LD blocks of 3, with one strong signal per block."""
    g = torch.Generator().manual_seed(seed)
    # Block 1: 3 highly correlated SNPs (shared latent)
    z1 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    block1 = z1 + 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    # Block 2: 3 highly correlated SNPs
    z2 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    block2 = z2 + 0.1 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    # Convert to dosage in [0, 2]
    G = torch.cat([block1, block2], dim=1)
    G = G - G.min(dim=0, keepdim=True).values
    G = 2.0 * G / (G.max(dim=0, keepdim=True).values + 1e-12)
    return G


def _meta(m, chr_labels=None, positions=None):
    if chr_labels is None:
        chr_labels = ["1"] * m
    if positions is None:
        positions = list(range(100, 100 + m))
    return {
        "snp": [f"rs{i}" for i in range(m)],
        "chr": chr_labels,
        "pos": positions,
        "a1": ["A"] * m,
        "a2": ["G"] * m,
    }


def _make_sumstats(snp_list, chr_labels, positions, betas, ps, af=None):
    m = len(snp_list)
    return SumStats(
        chr=list(chr_labels),
        pos=list(positions),
        snp=list(snp_list),
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.tensor(betas, dtype=torch.float64),
        se=torch.full((m,), 0.05, dtype=torch.float64),
        p=torch.tensor(ps, dtype=torch.float64),
        n=torch.full((m,), 5000.0),
        af=af,  # leave as None to skip MAF mismatch check
    )


def test_ct_picks_one_snp_per_ld_block():
    G = _sim_two_loci(n=400, seed=1)
    m = 6
    pos = [100, 200, 300, 1000, 1100, 1200]
    ld = build_ld_reference(
        G, _meta(m, positions=pos), mode="block",
        block_assignments=[0, 0, 0, 1, 1, 1],
    )
    # Strong signal at SNP 1 (block 0) and SNP 4 (block 1)
    ps = [1e-7, 1e-12, 1e-6, 1e-5, 1e-15, 1e-4]
    betas = [0.1, 0.4, 0.05, 0.05, 0.5, 0.02]
    ss = _make_sumstats(
        [f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps,
    )
    res = ClumpingThresholding().fit(ss, ld, p_threshold=5e-8, r2_threshold=0.1)
    nz = (res.weight != 0).nonzero(as_tuple=True)[0].tolist()
    # Should retain exactly one SNP from each block (the most significant)
    assert nz == [1, 4]
    assert float(res.weight[1]) == pytest.approx(0.4)
    assert float(res.weight[4]) == pytest.approx(0.5)


def test_ct_thresholding_drops_nonsignificant():
    G = _sim_two_loci(n=300, seed=2)
    m = 6
    pos = list(range(100, 100 + m))
    ld = build_ld_reference(
        G, _meta(m, positions=pos), mode="block",
        block_assignments=[0, 0, 0, 1, 1, 1],
    )
    ps = [0.5] * m  # all non-significant
    betas = [0.1] * m
    ss = _make_sumstats([f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps)
    res = ClumpingThresholding().fit(ss, ld, p_threshold=5e-8)
    assert int((res.weight != 0).sum().item()) == 0


def test_ct_window_filtering_does_not_clump_distant_snps():
    G = _sim_two_loci(n=300, seed=3)
    m = 6
    # Place all 6 SNPs far apart so window kicks in
    pos = [100, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]
    ld = build_ld_reference(
        G, _meta(m, positions=pos), mode="full",
    )
    ps = [1e-10] * m
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ss = _make_sumstats([f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps)
    res = ClumpingThresholding().fit(
        ss, ld, p_threshold=5e-8, r2_threshold=0.1, window_kb=250.0
    )
    # Window 250kb means each SNP is its own clump
    nz = (res.weight != 0).nonzero(as_tuple=True)[0].tolist()
    assert len(nz) == 6


def test_ct_cross_chromosome_independence():
    G = _sim_two_loci(n=300, seed=4)
    m = 6
    chr_labels = ["1", "1", "1", "2", "2", "2"]
    pos = [100, 200, 300, 100, 200, 300]
    ld = build_ld_reference(
        G, _meta(m, chr_labels=chr_labels, positions=pos), mode="block",
        block_assignments=[0, 0, 0, 1, 1, 1],
    )
    ps = [1e-12, 1e-10, 1e-9, 1e-12, 1e-10, 1e-9]
    betas = [0.5, 0.4, 0.3, 0.5, 0.4, 0.3]
    ss = _make_sumstats(
        [f"rs{i}" for i in range(m)], chr_labels, pos, betas, ps,
    )
    res = ClumpingThresholding().fit(ss, ld, p_threshold=5e-8)
    nz = (res.weight != 0).nonzero(as_tuple=True)[0].tolist()
    # One per chromosome (which equals one per block here)
    assert nz == [0, 3]


def test_ct_grid_metadata():
    G = _sim_two_loci(n=200, seed=5)
    m = 6
    pos = list(range(100, 100 + m))
    ld = build_ld_reference(
        G, _meta(m, positions=pos), mode="block",
        block_assignments=[0, 0, 0, 1, 1, 1],
    )
    ps = [1e-9] * m
    betas = [0.1] * m
    ss = _make_sumstats([f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps)
    res = ClumpingThresholding().fit(
        ss, ld, p_threshold=1e-8, r2_threshold=0.2, window_kb=100.0,
    )
    assert res.method == "ct"
    assert res.grid["p_threshold"] == 1e-8
    assert res.grid["r2_threshold"] == 0.2
    assert res.grid["window_kb"] == 100.0
    assert res.grid["n_index_snps"] == int((res.weight != 0).sum().item())
    assert res.converged is True


def test_ct_empty_result_when_no_significant_snps():
    G = _sim_two_loci(n=200, seed=6)
    m = 6
    pos = list(range(100, 100 + m))
    ld = build_ld_reference(
        G, _meta(m, positions=pos), mode="full",
    )
    ps = [0.9] * m
    betas = [0.0] * m
    ss = _make_sumstats([f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps)
    res = ClumpingThresholding().fit(ss, ld, p_threshold=5e-8)
    assert int((res.weight != 0).sum().item()) == 0
    assert res.grid["n_index_snps"] == 0


def test_ct_full_mode_matches_block_mode_when_one_block():
    G = _sim_two_loci(n=300, seed=7)
    m = 6
    pos = list(range(100, 100 + m))
    ld_full = build_ld_reference(G, _meta(m, positions=pos), mode="full")
    ld_block = build_ld_reference(
        G, _meta(m, positions=pos), mode="block", block_assignments=[0] * m,
    )
    ps = [1e-12, 1e-10, 1e-9, 1e-11, 1e-10, 1e-9]
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ss = _make_sumstats([f"rs{i}" for i in range(m)], ["1"] * m, pos, betas, ps)
    r_full = ClumpingThresholding().fit(ss, ld_full, p_threshold=5e-8)
    r_block = ClumpingThresholding().fit(ss, ld_block, p_threshold=5e-8)
    nz_f = (r_full.weight != 0).nonzero(as_tuple=True)[0].tolist()
    nz_b = (r_block.weight != 0).nonzero(as_tuple=True)[0].tolist()
    assert nz_f == nz_b
