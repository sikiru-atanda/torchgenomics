"""Tests for the native cross-population block stability accelerator.

The native module replaces the per-cluster, per-population, per-pair
``r2[..].item()`` round-trip loop in
``torchgwas.ld._blocks_novel.detect_blocks_cross_pop``. The pure-Python loop
remains in-tree as the algorithmic spec; the dispatcher routes to C++ when
the build is present and ``TORCHGWAS_DISABLE_NATIVE`` is unset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_CROSS_POP, _cross_pop_native
from torchgwas.ld._blocks_novel import detect_blocks_cross_pop

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_CROSS_POP, reason="native cross-pop extension not built"
)


# ── direct-kernel tests ──────────────────────────────────────────────


def test_build_sanity():
    assert HAS_NATIVE_CROSS_POP
    assert hasattr(_cross_pop_native, "cross_pop_stability")


def test_single_population_returns_one():
    # With a single population min == max so stability == 1 for any cluster
    # with at least one pair.
    rng = np.random.default_rng(0)
    R = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float64)
    R = (R + R.T) / 2.0
    pop_r2 = R[None, :, :]  # (1, 8, 8)
    starts = np.array([0, 4], dtype=np.int64)
    indices = np.array([0, 2, 5, 7], dtype=np.int64)
    out = _cross_pop_native.cross_pop_stability(pop_r2, starts, indices)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(1.0, abs=1e-12)


def test_cluster_too_small_returns_zero():
    pop_r2 = np.ones((2, 5, 5), dtype=np.float64)
    starts = np.array([0, 1], dtype=np.int64)  # b=1, no pairs
    indices = np.array([2], dtype=np.int64)
    out = _cross_pop_native.cross_pop_stability(pop_r2, starts, indices)
    assert out[0] == 0.0


def test_two_populations_min_over_max():
    # Pop A has all r²=0.9 in the cluster, Pop B has all r²=0.3 → 0.3/0.9
    R_a = np.full((6, 6), 0.9, dtype=np.float64)
    R_b = np.full((6, 6), 0.3, dtype=np.float64)
    pop_r2 = np.stack([R_a, R_b], axis=0)
    starts = np.array([0, 4], dtype=np.int64)
    indices = np.array([0, 1, 2, 3], dtype=np.int64)
    out = _cross_pop_native.cross_pop_stability(pop_r2, starts, indices)
    assert out[0] == pytest.approx(0.3 / 0.9, rel=1e-12)


def test_invalid_pop_r2_shape_rejected():
    bad = np.ones((3, 4, 5), dtype=np.float64)  # not square
    starts = np.array([0, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    with pytest.raises(Exception):
        _cross_pop_native.cross_pop_stability(bad, starts, indices)


def test_index_out_of_range_rejected():
    pop_r2 = np.ones((1, 4, 4), dtype=np.float64)
    starts = np.array([0, 2], dtype=np.int64)
    indices = np.array([0, 99], dtype=np.int64)  # 99 is out of range
    with pytest.raises(Exception):
        _cross_pop_native.cross_pop_stability(pop_r2, starts, indices)


# ── native↔Python equivalence inside the kernel ──────────────────────


def _python_stability(pop_r2_np: np.ndarray, cluster: list[int]) -> float:
    if len(cluster) < 2:
        return 0.0
    pop_means = []
    for p in range(pop_r2_np.shape[0]):
        R = pop_r2_np[p]
        vals = []
        for ii in range(len(cluster)):
            for jj in range(ii + 1, len(cluster)):
                vals.append(R[cluster[ii], cluster[jj]])
        if vals:
            pop_means.append(sum(vals) / len(vals))
    if not pop_means:
        return 0.0
    mx = max(pop_means)
    if mx <= 0.0:
        return 0.0
    return min(pop_means) / mx


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_matches_python_random_clusters(seed):
    rng = np.random.default_rng(seed)
    P, m = 3, 25
    pop_r2 = np.empty((P, m, m), dtype=np.float64)
    for p in range(P):
        M = rng.uniform(0.0, 1.0, size=(m, m))
        M = (M + M.T) / 2.0
        np.fill_diagonal(M, 1.0)
        pop_r2[p] = M

    n_clusters = 5
    starts = [0]
    indices: list[int] = []
    py_expected: list[float] = []
    for _ in range(n_clusters):
        b = int(rng.integers(2, 8))
        cl = sorted(rng.choice(m, size=b, replace=False).tolist())
        indices.extend(cl)
        starts.append(len(indices))
        py_expected.append(_python_stability(pop_r2, cl))

    out = _cross_pop_native.cross_pop_stability(
        pop_r2,
        np.asarray(starts, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
    )
    assert out.shape == (n_clusters,)
    np.testing.assert_allclose(out, np.asarray(py_expected), rtol=1e-12, atol=1e-12)


# ── end-to-end dispatcher equivalence (native vs Python fallback) ────


def _make_two_pop_fixture(seed=0):
    """Two populations, three latent blocks; pop A has stronger LD than pop B."""
    rng = np.random.default_rng(seed)
    n_per_pop = 80
    block_sizes = [6, 5, 7]
    m = sum(block_sizes)
    pop_ids: list[str] = ["A"] * n_per_pop + ["B"] * n_per_pop
    n = 2 * n_per_pop

    G = np.zeros((n, m), dtype=np.float64)
    col = 0
    for bsz in block_sizes:
        # Latent factor + small noise → strong LD inside the block
        z_a = rng.normal(0.0, 1.0, size=n_per_pop)
        z_b = rng.normal(0.0, 1.0, size=n_per_pop)
        for k in range(bsz):
            noise_a = rng.normal(0.0, 0.4, size=n_per_pop)
            # Weaker LD for B: more noise
            noise_b = rng.normal(0.0, 0.9, size=n_per_pop)
            xa = z_a + noise_a
            xb = z_b + noise_b
            # Quantize to 0/1/2 dosages
            G[:n_per_pop, col] = np.round(np.clip(xa - xa.min(), 0, 2))
            G[n_per_pop:, col] = np.round(np.clip(xb - xb.min(), 0, 2))
            col += 1

    variant_pos = list(range(m))
    variant_chr = ["1"] * m
    return torch.tensor(G), pop_ids, variant_pos, variant_chr


def test_dispatcher_native_matches_python(monkeypatch):
    G, pop_ids, variant_pos, variant_chr = _make_two_pop_fixture(seed=42)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_cross_pop(
        G, pop_ids, None, variant_pos, variant_chr,
        n_blocks_hint=3, stability_threshold=0.0, min_block_snps=2,
    )

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_cross_pop(
        G, pop_ids, None, variant_pos, variant_chr,
        n_blocks_hint=3, stability_threshold=0.0, min_block_snps=2,
    )

    assert len(blocks_native) == len(blocks_python)
    for bn, bp in zip(blocks_native, blocks_python):
        assert bn.region.start == bp.region.start
        assert bn.region.end == bp.region.end
        assert bn.n_variants == bp.n_variants
        assert bn.population_stability == pytest.approx(
            bp.population_stability, rel=1e-10, abs=1e-12
        )


def test_dispatcher_three_pop_matches_python(monkeypatch):
    rng = np.random.default_rng(7)
    n_per_pop = 60
    m = 24
    G = torch.tensor(rng.normal(1.0, 0.7, size=(3 * n_per_pop, m)))
    G = G.clamp(0.0, 2.0)
    pop_ids = ["A"] * n_per_pop + ["B"] * n_per_pop + ["C"] * n_per_pop
    variant_pos = list(range(m))
    variant_chr = ["1"] * m

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    bn = detect_blocks_cross_pop(
        G, pop_ids, None, variant_pos, variant_chr,
        n_blocks_hint=4, stability_threshold=0.0, min_block_snps=2,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    bp = detect_blocks_cross_pop(
        G, pop_ids, None, variant_pos, variant_chr,
        n_blocks_hint=4, stability_threshold=0.0, min_block_snps=2,
    )
    assert len(bn) == len(bp)
    for a, b in zip(bn, bp):
        assert a.population_stability == pytest.approx(
            b.population_stability, rel=1e-10, abs=1e-12
        )
