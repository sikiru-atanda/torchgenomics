"""Manhattan plots.

This module ships two Manhattan *types* (selectable by the user), plus a
bonus two-trait variant:

* :func:`manhattan_plot`        — classic linear single-trait Manhattan
  with per-SNP scatter, alternating chromosome colors, genome-wide and
  suggestive significance lines, optional highlight / annotate overlays.
* :func:`circos_manhattan_plot` — circular (Circos-style) Manhattan. The
  genome is unrolled around a polar axis, with chromosomes arranged as
  contiguous arcs and ``-log10(p)`` plotted radially outward from the ring.
* :func:`miami_plot`            — two-trait back-to-back Manhattan for
  visual comparison of two GWAS scans over the same SNP set.

Colors for chromosome alternation are user-overridable in every entry
point via the ``colors=`` parameter (any sequence of matplotlib colors,
cycled across chromosomes). Significance / suggestive line colors and
highlight colors are likewise parameterized.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ._common import chrom_to_int, neglog10_p, to_numpy


def _layout_positions(
    chrom_int: np.ndarray, pos: np.ndarray, gap: float = 0.02
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out variants on a shared x-axis, one chromosome after another.

    Returns ``(x, tick_positions, chrom_ids)`` where ``x`` has one entry per
    variant and ``tick_positions`` marks the center of each chromosome.
    """
    uniq = np.unique(chrom_int)
    x = np.empty_like(pos, dtype=np.float64)
    ticks = np.empty(uniq.size, dtype=np.float64)
    offset = 0.0
    for i, c in enumerate(uniq):
        mask = chrom_int == c
        p = pos[mask].astype(np.float64)
        if p.size == 0:
            ticks[i] = offset
            continue
        # Rescale each chromosome to its own [0, span] where span is the
        # fraction of total BP so big chromosomes get more visual space.
        pmin, pmax = p.min(), p.max()
        span = max(pmax - pmin, 1.0)
        x[mask] = offset + (p - pmin)
        ticks[i] = offset + span / 2.0
        offset += span * (1.0 + gap)
    return x, ticks, uniq


def manhattan_plot(
    chrom: Sequence[Any],
    pos: Sequence[int],
    p: Any,
    *,
    significance_threshold: float = 5e-8,
    suggestive_threshold: Optional[float] = 1e-5,
    highlight: Optional[Sequence[int]] = None,
    annotate: Optional[Sequence[tuple[int, str]]] = None,
    colors: Sequence[str] = ("#1f77b4", "#d62728"),
    highlight_color: str = "#2ca02c",
    significance_color: str = "red",
    suggestive_color: str = "grey",
    ax=None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    point_size: float = 6.0,
):
    """Classic linear Manhattan plot.

    Parameters
    ----------
    chrom, pos : sequences of length m
        Chromosome label and base-pair position per variant.
    p : (m,) array-like
        Association p-values. Tensors / numpy / lists all accepted.
    significance_threshold : float
        Horizontal red line; default 5e-8 (genome-wide).
    suggestive_threshold : float or None
        Horizontal grey line; set to None to disable.
    highlight : sequence of int indices or boolean mask
        Variants to paint in ``highlight_color``.
    annotate : sequence of ``(index, label)``
        SNPs to label with arrow + text.
    colors : pair of matplotlib colors
        Alternating chromosome colors.
    ax : matplotlib Axes or None
        If None, a new figure is created.
    output_path : str or None
        Save the figure to this path.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    p = to_numpy(p).astype(np.float64)
    pos_arr = to_numpy(pos).astype(np.int64)
    if p.size != pos_arr.size or p.size != len(chrom):
        raise ValueError("chrom, pos, p must have the same length")
    if p.size == 0:
        raise ValueError("empty input to manhattan_plot")

    chrom_int, chrom_labels = chrom_to_int(chrom)
    x, ticks, uniq = _layout_positions(chrom_int, pos_arr)
    y = neglog10_p(p)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    for i, c in enumerate(uniq):
        mask = chrom_int == c
        ax.scatter(
            x[mask],
            y[mask],
            s=point_size,
            c=colors[i % len(colors)],
            linewidths=0,
            rasterized=True,
        )

    if highlight is not None:
        hl = np.asarray(highlight)
        if hl.dtype == bool:
            hl_idx = np.nonzero(hl)[0]
        else:
            hl_idx = hl.astype(np.int64)
        if hl_idx.size:
            ax.scatter(
                x[hl_idx], y[hl_idx], s=point_size * 3.0,
                c=highlight_color, linewidths=0, zorder=3,
            )

    if significance_threshold is not None:
        ax.axhline(
            -np.log10(significance_threshold), color=significance_color,
            linestyle="--", linewidth=0.9, zorder=1,
        )
    if suggestive_threshold is not None:
        ax.axhline(
            -np.log10(suggestive_threshold), color=suggestive_color,
            linestyle=":", linewidth=0.8, zorder=1,
        )

    if annotate is not None:
        for idx, label in annotate:
            ax.annotate(
                label,
                xy=(float(x[idx]), float(y[idx])),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
            )

    ax.set_xticks(ticks)
    ax.set_xticklabels(chrom_labels, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, max(float(y.max()) * 1.1, 8.0))
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)

    return ax


def miami_plot(
    chrom: Sequence[Any],
    pos: Sequence[int],
    p_top: Any,
    p_bottom: Any,
    *,
    significance_threshold: float = 5e-8,
    suggestive_threshold: Optional[float] = 1e-5,
    top_label: Optional[str] = None,
    bottom_label: Optional[str] = None,
    colors: Sequence[str] = ("#1f77b4", "#d62728"),
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (11, 6),
):
    """Two-trait back-to-back Manhattan plot.

    The top panel shows ``-log10(p_top)`` pointing up; the bottom panel
    shows ``-log10(p_bottom)`` mirrored below the x-axis so the two scans
    share a single chromosome axis. Classic use: compare a GWAS result
    against a replication / alternate trait / meta-analysis.

    Returns
    -------
    tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        ``(ax_top, ax_bottom)``.
    """
    import matplotlib.pyplot as plt

    p_top = to_numpy(p_top).astype(np.float64)
    p_bot = to_numpy(p_bottom).astype(np.float64)
    if p_top.size != p_bot.size or p_top.size != len(chrom):
        raise ValueError("chrom, p_top, p_bottom must have equal length")

    pos_arr = to_numpy(pos).astype(np.int64)
    chrom_int, chrom_labels = chrom_to_int(chrom)
    x, ticks, uniq = _layout_positions(chrom_int, pos_arr)
    y_top = neglog10_p(p_top)
    y_bot = neglog10_p(p_bot)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    for i, c in enumerate(uniq):
        mask = chrom_int == c
        col = colors[i % len(colors)]
        ax_top.scatter(x[mask], y_top[mask], s=6, c=col, linewidths=0, rasterized=True)
        ax_bot.scatter(x[mask], y_bot[mask], s=6, c=col, linewidths=0, rasterized=True)

    if significance_threshold is not None:
        thr = -np.log10(significance_threshold)
        ax_top.axhline(thr, color="red", linestyle="--", linewidth=0.9)
        ax_bot.axhline(thr, color="red", linestyle="--", linewidth=0.9)
    if suggestive_threshold is not None:
        sug = -np.log10(suggestive_threshold)
        ax_top.axhline(sug, color="grey", linestyle=":", linewidth=0.8)
        ax_bot.axhline(sug, color="grey", linestyle=":", linewidth=0.8)

    top_max = max(float(y_top.max()) * 1.1, 8.0)
    bot_max = max(float(y_bot.max()) * 1.1, 8.0)
    ax_top.set_ylim(0, top_max)
    ax_bot.set_ylim(bot_max, 0)  # inverted axis for mirrored look
    ax_bot.set_xticks(ticks)
    ax_bot.set_xticklabels(chrom_labels, fontsize=8)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels([])

    ax_top.set_ylabel(r"$-\log_{10}(p)$")
    ax_bot.set_ylabel(r"$-\log_{10}(p)$")
    ax_bot.set_xlabel("Chromosome")

    if top_label:
        ax_top.text(
            0.01, 0.92, top_label, transform=ax_top.transAxes,
            fontsize=10, fontweight="bold",
        )
    if bottom_label:
        ax_bot.text(
            0.01, 0.08, bottom_label, transform=ax_bot.transAxes,
            fontsize=10, fontweight="bold",
        )

    for a in (ax_top, ax_bot):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    if title:
        ax_top.set_title(title)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return ax_top, ax_bot


# ---------------------------------------------------------------------------
# Circos Manhattan
# ---------------------------------------------------------------------------


def circos_manhattan_plot(
    chrom: Sequence[Any],
    pos: Sequence[int],
    p: Any,
    *,
    significance_threshold: float = 5e-8,
    suggestive_threshold: Optional[float] = 1e-5,
    highlight: Optional[Sequence[int]] = None,
    annotate: Optional[Sequence[tuple[int, str]]] = None,
    colors: Sequence[str] = ("#1f77b4", "#d62728"),
    highlight_color: str = "#2ca02c",
    significance_color: str = "red",
    suggestive_color: str = "grey",
    inner_radius: float = 1.0,
    ring_width: float = 1.0,
    gap_frac: float = 0.01,
    ax=None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    point_size: float = 6.0,
    chrom_label_fontsize: float = 8.0,
    figsize: tuple[float, float] = (7.5, 7.5),
):
    """Circos-style Manhattan plot.

    Chromosomes are laid out around a polar axis as contiguous arcs, each
    taking an angular span proportional to its base-pair extent (plus a
    fixed inter-chromosome gap). ``-log10(p)`` is plotted radially outward
    from ``inner_radius`` toward ``inner_radius + ring_width``.

    Parameters
    ----------
    chrom, pos, p
        Same as :func:`manhattan_plot`.
    significance_threshold, suggestive_threshold
        Radial dashed / dotted threshold rings. Set to ``None`` to disable.
    highlight : sequence of int indices or bool mask
        Variants painted in ``highlight_color``.
    annotate : sequence of ``(index, label)``
        SNPs to label outside the ring.
    colors : sequence of matplotlib colors
        Cycled across chromosomes.
    highlight_color, significance_color, suggestive_color : str
        User-customizable colors for highlight points and threshold rings.
    inner_radius : float
        Radius at which the ``-log10(p) = 0`` baseline sits.
    ring_width : float
        Radial distance reserved for the GWAS signal. Actual plotted points
        are scaled so the highest observed ``-log10(p)`` reaches
        ``inner_radius + ring_width``.
    gap_frac : float
        Fraction of the full circle reserved for gaps between chromosomes.
    ax : matplotlib Axes or None
        Must be a polar axes if supplied. Otherwise one is created.
    output_path : str or None
        Save the figure.

    Returns
    -------
    matplotlib.axes.Axes
        The polar axes that was drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes

    p = to_numpy(p).astype(np.float64)
    pos_arr = to_numpy(pos).astype(np.int64)
    if p.size != pos_arr.size or p.size != len(chrom):
        raise ValueError("chrom, pos, p must have the same length")
    if p.size == 0:
        raise ValueError("empty input to circos_manhattan_plot")

    chrom_int, chrom_labels = chrom_to_int(chrom)
    uniq = np.unique(chrom_int)
    n_chrom = uniq.size

    # Per-chromosome BP span, used to size each arc.
    spans: list[float] = []
    for c in uniq:
        mask = chrom_int == c
        pc = pos_arr[mask]
        spans.append(float(max(pc.max() - pc.min(), 1)))
    total_span = float(sum(spans))

    # Angular budget: 2π total; gap_frac * 2π reserved for inter-chrom gaps.
    gap_rad = 2.0 * np.pi * gap_frac / max(n_chrom, 1)
    data_budget = 2.0 * np.pi - gap_rad * n_chrom

    # Map each variant to a (theta, r) polar coordinate.
    y = neglog10_p(p)
    y_max = max(float(y.max()) * 1.05, 8.0)
    theta = np.empty_like(y, dtype=np.float64)
    cursor = 0.0
    arc_centers: list[float] = []
    arc_ranges: list[tuple[float, float]] = []
    for span, c in zip(spans, uniq):
        arc = data_budget * (span / total_span)
        mask = chrom_int == c
        pc = pos_arr[mask].astype(np.float64)
        pc_min = pc.min()
        theta[mask] = cursor + (pc - pc_min) / span * arc
        arc_centers.append(cursor + arc / 2.0)
        arc_ranges.append((cursor, cursor + arc))
        cursor += arc + gap_rad

    r = inner_radius + (y / y_max) * ring_width

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
    else:
        if not isinstance(ax, PolarAxes):
            raise TypeError("circos_manhattan_plot requires a polar Axes")
        fig = ax.figure

    # Draw SNPs, colored by chromosome.
    for i, c in enumerate(uniq):
        mask = chrom_int == c
        ax.scatter(
            theta[mask], r[mask],
            s=point_size,
            c=colors[i % len(colors)],
            linewidths=0,
            rasterized=True,
        )

    # Highlight overlay.
    if highlight is not None:
        hl = np.asarray(highlight)
        if hl.dtype == bool:
            hl_idx = np.nonzero(hl)[0]
        else:
            hl_idx = hl.astype(np.int64)
        if hl_idx.size:
            ax.scatter(
                theta[hl_idx], r[hl_idx],
                s=point_size * 3.0, c=highlight_color,
                linewidths=0, zorder=4,
            )

    # Threshold rings (dashed red for genome-wide, dotted grey for suggestive).
    theta_full = np.linspace(0.0, 2.0 * np.pi, 361)
    if significance_threshold is not None:
        r_sig = inner_radius + (-np.log10(significance_threshold) / y_max) * ring_width
        ax.plot(theta_full, np.full_like(theta_full, r_sig),
                color=significance_color, linestyle="--", linewidth=0.9, zorder=2)
    if suggestive_threshold is not None:
        r_sug = inner_radius + (-np.log10(suggestive_threshold) / y_max) * ring_width
        ax.plot(theta_full, np.full_like(theta_full, r_sug),
                color=suggestive_color, linestyle=":", linewidth=0.8, zorder=2)

    # Baseline ring at -log10(p) = 0.
    ax.plot(theta_full, np.full_like(theta_full, inner_radius),
            color="black", linewidth=0.6, zorder=1)

    # Chromosome labels just outside the outer ring.
    outer = inner_radius + ring_width
    label_r = outer + 0.15 * ring_width
    for label, center in zip(chrom_labels, arc_centers):
        ax.text(
            center, label_r, str(label),
            ha="center", va="center",
            fontsize=chrom_label_fontsize, fontweight="bold",
        )

    if annotate is not None:
        for idx, lab in annotate:
            ax.annotate(
                str(lab),
                xy=(float(theta[idx]), float(r[idx])),
                xytext=(float(theta[idx]), outer + 0.35 * ring_width),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black"),
            )

    # Strip polar spines and angular ticks for a clean Circos look.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, outer + 0.5 * ring_width)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    if title:
        ax.set_title(title, pad=20)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return ax
