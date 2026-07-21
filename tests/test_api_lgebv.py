"""Tests for :func:`torchgenomics.api.lgebv`.

Coverage:
 1. Recovery of a planted block effect (diploid).
 2. Tetraploid sanity (runs, shapes, h2_used in (0, 1)).
 3. ``favorable_direction`` flip.
 4. ``h2`` provided vs auto-estimated agree at the truth.
 5. Integration with :func:`torchgenomics.ld.detect_blocks`.
 6. Integration with :class:`torchgenomics.models.HaplotypeGWAS`.
 7. :meth:`LGEBVResult.to_dataframe` columns.
 8. :meth:`LGEBVResult.top_blocks` sort.
"""
from __future__ import annotations

import numpy as np
import torch

import torchgenomics as tg
from torchgenomics.api import LGEBVResult, lgebv
from torchgenomics.io.regions import Region
from torchgenomics.ld import LDBlock, detect_blocks

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_diploid_fixture(seed: int = 42, n: int = 200, m: int = 60):
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(n, m)).astype(np.float64)
    G_t = torch.tensor(G, dtype=torch.float64)
    return G_t


def _make_simple_blocks(m: int, block_size: int = 6, chrom: str = "1") -> list:
    """Build a simple uniform LDBlock list for testing."""
    blocks = []
    for b_start in range(0, m, block_size):
        b_end = min(b_start + block_size, m)
        vi = list(range(b_start, b_end))
        if not vi:
            continue
        region = Region(
            region_id=f"blk_{b_start}",
            chr=chrom,
            start=b_start * 1000,         # arbitrary bp positions
            end=(b_end - 1) * 1000 + 1,
        )
        blocks.append(
            LDBlock(
                region=region,
                n_variants=len(vi),
                variant_indices=vi,
                method="manual",
                mean_r2=0.0,
                mean_dprime=0.0,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Test 1: Planted block effect recovery (diploid)
# ---------------------------------------------------------------------------

def test_diploid_lgebv_recovers_known_block_effect():
    """Simulate a diploid dataset where one block carries the true effect.

    Block 0 (variants 0..7) has a non-zero true effect β = -2 on y; all
    other blocks are independent of y. Assert that block 0 has the
    most-negative LGEBV (true direction) and the largest block_variance.
    """
    rng = np.random.default_rng(2026)
    n, m = 250, 60
    G_np = rng.binomial(2, 0.3, size=(n, m)).astype(np.float64)
    G = torch.tensor(G_np, dtype=torch.float64)

    true_idx = list(range(0, 8))
    # Standardize the planted block before injecting the effect so the signal
    # is on the same scale the model fits.
    block_dose = G[:, true_idx].sum(dim=1)
    block_dose -= block_dose.mean()
    block_dose /= block_dose.std()
    y = -2.0 * block_dose + 1.0 * torch.randn(n, dtype=torch.float64)

    blocks = _make_simple_blocks(m, block_size=8)
    assert len(blocks) > 1

    result = lgebv(y, G, blocks)
    assert isinstance(result, LGEBVResult)
    assert len(result.lgebv) == len(blocks)

    # The first block carries the truth → it should have the most-negative
    # lgebv and the largest block_variance.
    lgebv_vec = result.lgebv.cpu().numpy()
    var_vec = result.block_variance.cpu().numpy()

    true_block_idx = 0
    # Most negative
    assert lgebv_vec.argmin() == true_block_idx, (
        f"Expected block 0 to have the most-negative lgebv; got argmin={lgebv_vec.argmin()} "
        f"with values {lgebv_vec}"
    )
    # Largest variance
    assert var_vec.argmax() == true_block_idx, (
        f"Expected block 0 to have the largest variance; got argmax={var_vec.argmax()} "
        f"with values {var_vec}"
    )
    # Other blocks have |lgebv| < 0.5 * |true|
    true_abs = abs(float(lgebv_vec[true_block_idx]))
    for i, v in enumerate(lgebv_vec):
        if i == true_block_idx:
            continue
        assert abs(v) < 0.5 * true_abs, (
            f"Block {i} has |lgebv|={abs(v):.3f} not < 0.5 * {true_abs:.3f}"
        )

    # favorable for resistance (negative direction) → true block is favorable.
    assert result.favorable[true_block_idx] is True


# ---------------------------------------------------------------------------
# Test 2: Tetraploid sanity
# ---------------------------------------------------------------------------

def test_tetraploid_lgebv_runs():
    """Tetraploid [0,4] dosages, 3 blocks of 6 SNPs each."""
    rng = np.random.default_rng(7)
    n, m = 120, 18
    G_np = rng.binomial(4, 0.4, size=(n, m)).astype(np.float64)
    G = torch.tensor(G_np, dtype=torch.float64)
    y = torch.tensor(rng.standard_normal(n), dtype=torch.float64)

    blocks = _make_simple_blocks(m, block_size=6)
    assert len(blocks) == 3

    result = lgebv(y, G, blocks)
    assert len(result.lgebv) == 3
    assert result.lgebv.shape == (3,)
    assert result.block_variance.shape == (3,)
    assert 0.0 < result.h2_used < 1.0, f"h2_used out of range: {result.h2_used}"
    # variances are non-negative
    assert (result.block_variance >= 0).all()


# ---------------------------------------------------------------------------
# Test 3: favorable_direction flip
# ---------------------------------------------------------------------------

def test_favorable_direction_flip():
    """Same data with negative vs positive favorable direction → inverted."""
    rng = np.random.default_rng(3)
    n, m = 150, 30
    G = torch.tensor(rng.binomial(2, 0.3, size=(n, m)).astype(np.float64), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal(n), dtype=torch.float64)
    blocks = _make_simple_blocks(m, block_size=5)

    r_neg = lgebv(y, G, blocks, favorable_direction="negative")
    r_pos = lgebv(y, G, blocks, favorable_direction="positive")

    # Skip exactly-zero LGEBV blocks (degenerate sign) when comparing flip.
    flipped = 0
    compared = 0
    for i, (vneg, vpos, lg) in enumerate(zip(r_neg.favorable, r_pos.favorable, r_neg.lgebv.tolist())):
        if abs(lg) < 1e-12:
            continue
        compared += 1
        assert vneg != vpos, (
            f"Block {i} (lgebv={lg:.6f}) did not flip: neg={vneg}, pos={vpos}"
        )
        flipped += 1
    assert flipped == compared
    assert compared > 0


# ---------------------------------------------------------------------------
# Test 4: h2 provided vs estimated agree at truth
# ---------------------------------------------------------------------------

def test_h2_provided_vs_estimated_agree_at_truth():
    """Fixture with h²=0.6 → h2=0.6 vs h2=None Pearson-correlate > 0.95."""
    rng = np.random.default_rng(11)
    n, m = 300, 100
    G_np = rng.binomial(2, 0.3, size=(n, m)).astype(np.float64)
    G = torch.tensor(G_np, dtype=torch.float64)

    # Simulate a truly polygenic trait with h² = 0.6
    p = G_np.mean(axis=0) / 2.0
    scale = np.sqrt(2.0 * p * (1.0 - p))
    scale[scale < 1e-10] = 1.0
    Z = (G_np - 2.0 * p) / scale
    # True effects: small i.i.d. polygenic
    true_u = rng.standard_normal(m)
    g = Z @ true_u
    g_var = float(np.var(g))
    target_h2 = 0.6
    sig2_e = g_var * (1.0 - target_h2) / target_h2
    e = rng.standard_normal(n) * np.sqrt(sig2_e)
    y_np = g + e
    y = torch.tensor(y_np, dtype=torch.float64)

    blocks = _make_simple_blocks(m, block_size=10)

    r_fixed = lgebv(y, G, blocks, h2=0.6)
    r_auto = lgebv(y, G, blocks, h2=None)

    a = r_fixed.lgebv.cpu().numpy()
    b = r_auto.lgebv.cpu().numpy()
    # Pearson correlation
    a_z = (a - a.mean()) / (a.std() + 1e-30)
    b_z = (b - b.mean()) / (b.std() + 1e-30)
    rho = float(np.mean(a_z * b_z))
    assert rho > 0.95, f"Pearson correlation too low: rho={rho:.4f}"


# ---------------------------------------------------------------------------
# Test 5: detect_blocks integration
# ---------------------------------------------------------------------------

def test_blocks_from_ld_detect_blocks_integration():
    """Run detect_blocks then feed the LDBlock list directly to lgebv()."""
    rng = np.random.default_rng(42)
    n, m = 120, 40
    G = torch.tensor(rng.binomial(2, 0.3, size=(n, m)).astype(np.float64), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal(n), dtype=torch.float64)

    positions = [i * 1000 for i in range(m)]
    chrs = ["1"] * m

    blocks = detect_blocks(
        G,
        positions,
        chrs,
        method="r2",
        r2_threshold=0.2,
        tolerance=1,
    )
    # If detect_blocks happens to return zero blocks for this draw, fall back
    # to single-SNP blocks so the integration assertion still tests the duck-typing.
    if len(blocks) == 0:
        blocks = _make_simple_blocks(m, block_size=4)

    result = lgebv(y, G, blocks)
    assert isinstance(result, LGEBVResult)
    assert len(result.lgebv) == len(blocks)


# ---------------------------------------------------------------------------
# Test 6: HaplotypeGWAS integration
# ---------------------------------------------------------------------------

def test_blocks_from_haplotype_gwas_integration():
    """Feed HaplotypeBlock list to lgebv() — duck-type on .variant_indices."""
    from torchgenomics.models.haplotype_gwas import HaplotypeGWAS

    rng = np.random.default_rng(5)
    n, m = 120, 24
    G = torch.tensor(rng.binomial(2, 0.3, size=(n, m)).astype(np.float64), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal(n), dtype=torch.float64)

    positions = [i * 1000 for i in range(m)]
    chrs = ["1"] * m

    # Use moving-window construction to avoid block-detection sensitivity.
    hg = HaplotypeGWAS(method="window", window_size=6, step=6, ploidy=2)
    haplo_blocks = hg.construct_haplotypes(G, positions, chrs)
    assert len(haplo_blocks) > 0
    # Sanity-check duck-typing
    assert all(hasattr(b, "variant_indices") for b in haplo_blocks)

    result = lgebv(y, G, haplo_blocks)
    assert isinstance(result, LGEBVResult)
    assert len(result.lgebv) == len(haplo_blocks)


# ---------------------------------------------------------------------------
# Test 7: to_dataframe columns
# ---------------------------------------------------------------------------

def test_to_dataframe_columns():
    rng = np.random.default_rng(9)
    n, m = 80, 20
    G = torch.tensor(rng.binomial(2, 0.3, size=(n, m)).astype(np.float64), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal(n), dtype=torch.float64)
    blocks = _make_simple_blocks(m, block_size=5)

    result = lgebv(y, G, blocks)
    df = result.to_dataframe()
    expected = {
        "block_id", "chrom", "start", "end", "n_variants",
        "lgebv", "block_variance", "favorable",
    }
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )
    assert len(df) == len(blocks)


# ---------------------------------------------------------------------------
# Test 8: top_blocks ordering
# ---------------------------------------------------------------------------

def test_top_blocks_filters_correctly():
    rng = np.random.default_rng(13)
    n, m = 100, 50
    G_np = rng.binomial(2, 0.3, size=(n, m)).astype(np.float64)
    G = torch.tensor(G_np, dtype=torch.float64)
    # Plant a strong signal in block 0 so variance ordering is non-trivial.
    block_dose = G[:, 0:10].sum(dim=1)
    block_dose -= block_dose.mean()
    block_dose /= (block_dose.std() + 1e-12)
    y = -1.5 * block_dose + 1.0 * torch.randn(n, dtype=torch.float64)

    blocks = _make_simple_blocks(m, block_size=10)
    result = lgebv(y, G, blocks)

    top = result.top_blocks(k=3, by="variance")
    assert len(top) == 3
    vars_ = top["block_variance"].to_list()
    # Descending sort
    for a, b in zip(vars_, vars_[1:]):
        assert a >= b, f"top_blocks not sorted descending: {vars_}"

    # by="abs_lgebv" — descending |lgebv|
    top2 = result.top_blocks(k=3, by="abs_lgebv")
    abs_l = [abs(v) for v in top2["lgebv"].to_list()]
    for a, b in zip(abs_l, abs_l[1:]):
        assert a >= b, f"top_blocks abs_lgebv not sorted descending: {abs_l}"


# ---------------------------------------------------------------------------
# Sanity: re-exports
# ---------------------------------------------------------------------------

def test_lgebv_top_level_reexport():
    assert tg.lgebv is lgebv
    assert tg.LGEBVResult is LGEBVResult


def test_lgebv_registered_as_mcp_tool():
    from torchgenomics.api import registered_tools
    reg = registered_tools()
    assert "tg_lgebv" in reg, "tg_lgebv not in MCP tool registry"
    entry = reg["tg_lgebv"]
    assert entry.func is lgebv
    assert entry.category == "pgs"
