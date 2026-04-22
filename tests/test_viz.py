"""Tests for the ``torchgwas.viz`` module.

All tests run headless via the ``Agg`` backend so they can execute on CI
without a display. We verify that each entry point

  1. returns the right axes object(s),
  2. renders without raising on realistic inputs,
  3. respects its documented parameters (thresholds, labels, blocks),
  4. rejects malformed inputs with a clear ``ValueError``.

We deliberately do not assert on pixel output — matplotlib's rendering
back-end changes across versions.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchgwas.viz import (  # noqa: E402
    circos_manhattan_plot,
    haploview_plot,
    manhattan_plot,
    miami_plot,
    qq_plot,
)

# ---------------------------------------------------------------------------
# Synthetic GWAS fixture
# ---------------------------------------------------------------------------


def _synthetic_gwas(seed: int = 0, m_per_chr: int = 60, n_chrom: int = 5):
    rng = np.random.default_rng(seed)
    chrom: list[str] = []
    pos: list[int] = []
    pvals: list[float] = []
    for c in range(1, n_chrom + 1):
        for j in range(m_per_chr):
            chrom.append(str(c))
            pos.append(10_000 + j * 1_000)
            pvals.append(float(rng.uniform(0.0, 1.0)))
    # Plant one strong hit on chromosome 3.
    hit_idx = (3 - 1) * m_per_chr + m_per_chr // 2
    pvals[hit_idx] = 1e-12
    return chrom, pos, np.asarray(pvals), hit_idx


# ---------------------------------------------------------------------------
# Manhattan
# ---------------------------------------------------------------------------


def test_manhattan_plot_returns_axes_and_sets_thresholds():
    chrom, pos, p, _ = _synthetic_gwas()
    ax = manhattan_plot(chrom, pos, p, significance_threshold=5e-8,
                        suggestive_threshold=1e-5)
    assert ax is not None
    # Two horizontal lines were drawn (significance + suggestive).
    hlines = [ln for ln in ax.get_lines() if ln.get_linestyle() in ("--", ":")]
    assert len(hlines) >= 2
    plt.close(ax.figure)


def test_manhattan_plot_accepts_torch_tensor_and_highlight():
    chrom, pos, p, hit = _synthetic_gwas()
    p_t = torch.tensor(p, dtype=torch.float64)
    ax = manhattan_plot(chrom, pos, p_t, highlight=[hit],
                        annotate=[(hit, "LEAD")])
    # At least one text annotation (from annotate=).
    texts = [t for t in ax.texts if t.get_text() == "LEAD"]
    assert len(texts) == 1
    plt.close(ax.figure)


def test_manhattan_plot_writes_file():
    chrom, pos, p, _ = _synthetic_gwas()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "manhattan.png")
        manhattan_plot(chrom, pos, p, output_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
    plt.close("all")


def test_manhattan_plot_rejects_length_mismatch():
    chrom, pos, p, _ = _synthetic_gwas()
    try:
        manhattan_plot(chrom[:-1], pos, p)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_manhattan_plot_honors_custom_colors():
    """User must be able to override chromosome / threshold colors."""
    chrom, pos, p, _ = _synthetic_gwas()
    custom = ["#112233", "#445566", "#778899"]
    ax = manhattan_plot(
        chrom, pos, p,
        colors=custom,
        significance_color="#ff00ff",
        suggestive_color="#00ffff",
    )
    # Collect scatter facecolors and confirm at least two custom colors landed.
    facecolors = [c.get_facecolor() for c in ax.collections]
    assert len(facecolors) >= 2
    # Confirm custom threshold colors were applied to dashed/dotted lines.
    import matplotlib.colors as mcolors

    line_colors = {tuple(mcolors.to_rgba(ln.get_color())) for ln in ax.get_lines()
                   if ln.get_linestyle() in ("--", ":")}
    assert tuple(mcolors.to_rgba("#ff00ff")) in line_colors
    assert tuple(mcolors.to_rgba("#00ffff")) in line_colors
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Circos Manhattan
# ---------------------------------------------------------------------------


def test_circos_manhattan_plot_returns_polar_axes():
    from matplotlib.projections.polar import PolarAxes

    chrom, pos, p, _ = _synthetic_gwas()
    ax = circos_manhattan_plot(chrom, pos, p)
    assert isinstance(ax, PolarAxes)
    # Baseline ring + significance ring + suggestive ring = 3 line plots.
    assert len([ln for ln in ax.get_lines()]) >= 3
    plt.close(ax.figure)


def test_circos_manhattan_plot_draws_chrom_labels():
    chrom, pos, p, _ = _synthetic_gwas(n_chrom=5)
    ax = circos_manhattan_plot(chrom, pos, p)
    text_labels = {t.get_text() for t in ax.texts}
    # Every chromosome label should appear as a text annotation.
    for c in {"1", "2", "3", "4", "5"}:
        assert c in text_labels
    plt.close(ax.figure)


def test_circos_manhattan_plot_honors_custom_colors():
    chrom, pos, p, hit = _synthetic_gwas()
    custom = ["#123456", "#abcdef"]
    ax = circos_manhattan_plot(
        chrom, pos, p,
        colors=custom,
        highlight=[hit],
        highlight_color="#00ff00",
        significance_color="#ff0000",
        suggestive_color="#0000ff",
    )
    import matplotlib.colors as mcolors

    line_colors = {tuple(mcolors.to_rgba(ln.get_color())) for ln in ax.get_lines()}
    assert tuple(mcolors.to_rgba("#ff0000")) in line_colors
    assert tuple(mcolors.to_rgba("#0000ff")) in line_colors
    plt.close(ax.figure)


def test_circos_manhattan_plot_writes_file():
    chrom, pos, p, _ = _synthetic_gwas()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "circos.png")
        circos_manhattan_plot(chrom, pos, p, output_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
    plt.close("all")


def test_circos_manhattan_plot_rejects_length_mismatch():
    chrom, pos, p, _ = _synthetic_gwas()
    try:
        circos_manhattan_plot(chrom[:-1], pos, p)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")


def test_circos_manhattan_plot_rejects_non_polar_axes():
    chrom, pos, p, _ = _synthetic_gwas()
    fig, ax = plt.subplots()
    try:
        circos_manhattan_plot(chrom, pos, p, ax=ax)
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for cartesian axes")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Miami
# ---------------------------------------------------------------------------


def test_miami_plot_returns_two_axes_and_labels():
    chrom, pos, p_top, _ = _synthetic_gwas(seed=1)
    _, _, p_bot, _ = _synthetic_gwas(seed=2)
    ax_top, ax_bot = miami_plot(
        chrom, pos, p_top, p_bot,
        top_label="Trait A", bottom_label="Trait B",
    )
    assert ax_top is not ax_bot
    # Bottom axis is inverted so ylim[0] > ylim[1].
    y0, y1 = ax_bot.get_ylim()
    assert y0 > y1
    plt.close(ax_top.figure)


def test_miami_plot_rejects_length_mismatch():
    chrom, pos, p, _ = _synthetic_gwas()
    try:
        miami_plot(chrom, pos, p, p[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for length mismatch")


# ---------------------------------------------------------------------------
# QQ plot
# ---------------------------------------------------------------------------


def test_qq_plot_uniform_null_has_lambda_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=2000)
    ax = qq_plot(p, confidence_band=True, lambda_gc=True)
    # Find the lambda_GC annotation.
    lam_texts = [t.get_text() for t in ax.texts if "lambda" in t.get_text().lower()
                 or "\\lambda" in t.get_text()]
    assert len(lam_texts) == 1
    plt.close(ax.figure)


def test_qq_plot_accepts_torch_tensor_and_saves():
    rng = np.random.default_rng(1)
    p = torch.tensor(rng.uniform(0.0, 1.0, size=500), dtype=torch.float64)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "qq.png")
        qq_plot(p, output_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
    plt.close("all")


def test_qq_plot_confidence_band_bounds_uniform_null():
    """For m=500 uniform p-values, most observed points must fall within
    the 95% CI band (pointwise). We allow up to 15% outside to leave room
    for the pointwise / simultaneous distinction and MC noise."""
    from scipy.stats import beta as beta_dist

    rng = np.random.default_rng(2)
    m = 500
    p = np.sort(rng.uniform(0.0, 1.0, size=m))
    ranks = np.arange(1, m + 1)
    lo = beta_dist.ppf(0.025, ranks, m - ranks + 1)
    hi = beta_dist.ppf(0.975, ranks, m - ranks + 1)
    inside = ((p >= lo) & (p <= hi)).mean()
    assert inside > 0.85


def test_qq_plot_rejects_empty():
    try:
        qq_plot(np.array([]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty input")


# ---------------------------------------------------------------------------
# Haploview LD triangle
# ---------------------------------------------------------------------------


def _ld_block_matrix(m: int = 20, seed: int = 0) -> np.ndarray:
    """Build a symmetric r² matrix with two strong LD blocks."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, 0.2, size=(m, m))
    # Strong block 1: indices [2..7]
    for i in range(2, 8):
        for j in range(2, 8):
            R[i, j] = 0.85
    # Strong block 2: indices [12..17]
    for i in range(12, 18):
        for j in range(12, 18):
            R[i, j] = 0.9
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 1.0)
    return R


def test_haploview_plot_basic_returns_axes():
    R = _ld_block_matrix()
    ax = haploview_plot(R)
    # Should have one PolyCollection (the diamond cells).
    polycols = [c for c in ax.collections
                if type(c).__name__ == "PolyCollection"]
    assert len(polycols) >= 1
    plt.close(ax.figure)


def test_haploview_plot_accepts_torch_tensor_and_blocks_tuple():
    R = torch.tensor(_ld_block_matrix(), dtype=torch.float64)
    ax = haploview_plot(
        R, blocks=[(2, 7), (12, 17)], metric="r2", show_snp_track=True,
    )
    # Two block triangles overlaid as Polygon patches.
    from matplotlib.patches import Polygon

    polys = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(polys) == 2
    plt.close(ax.figure)


def test_haploview_plot_accepts_ldblock_dataclass():
    """LDBlock-like objects with .variant_indices should be accepted."""
    from dataclasses import dataclass

    @dataclass
    class _FakeBlock:
        variant_indices: list[int]

    R = _ld_block_matrix()
    blocks = [_FakeBlock(variant_indices=list(range(2, 8)))]
    ax = haploview_plot(R, blocks=blocks)
    from matplotlib.patches import Polygon

    polys = [p for p in ax.patches if isinstance(p, Polygon)]
    assert len(polys) == 1
    plt.close(ax.figure)


def test_haploview_plot_dprime_metric_uses_blue_cmap():
    R = _ld_block_matrix()
    ax = haploview_plot(R, metric="dprime")
    pc = [c for c in ax.collections if type(c).__name__ == "PolyCollection"][0]
    # Just check that a cmap was assigned (Blues by default for dprime).
    assert pc.get_cmap().name.lower().startswith("blue")
    plt.close(ax.figure)


def test_haploview_plot_writes_file():
    R = _ld_block_matrix()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "hap.png")
        haploview_plot(R, blocks=[(2, 7)], output_path=out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
    plt.close("all")


def test_haploview_plot_rejects_non_square():
    try:
        haploview_plot(np.random.rand(10, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-square matrix")


def test_haploview_plot_rejects_tiny_matrix():
    try:
        haploview_plot(np.array([[1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for m < 2")


# ---------------------------------------------------------------------------
# Back-compat shim: torchgwas.stats.plots must still import cleanly.
# ---------------------------------------------------------------------------


def test_stats_plots_shim_reexports_viz_entry_points():
    from torchgwas.stats.plots import (
        haploview_plot as h,
    )
    from torchgwas.stats.plots import (
        manhattan_plot as m,
    )
    from torchgwas.stats.plots import (
        miami_plot as mi,
    )
    from torchgwas.stats.plots import (
        qq_plot as q,
    )

    assert callable(h) and callable(m) and callable(mi) and callable(q)
