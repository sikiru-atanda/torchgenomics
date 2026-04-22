"""Trumpet plot: allele frequency vs effect size with power curve overlays.

Implements the visualization from Garcia-Gonzalez et al. (2023, Gigabyte),
"TrumpetPlots: Visualizing the relationship between allele frequency and
effect size in genetic association studies".

The plot displays individual variants as a scatter of allele frequency
(x-axis, log10 scale by default) against effect size (y-axis), with
overlaid power curves showing the detection boundary at different sample
sizes. The "trumpet" shape emerges because power decreases at extreme
allele frequencies, so the minimum detectable effect size forms a
U-shaped envelope.

Power curves are computed via the NCP-based normal approximation from
:func:`torchgwas.postgwas._power.power_curve`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._common import to_numpy


def trumpet_plot(
    af: Any,
    beta: Any,
    *,
    n: int | float | None = None,
    n_curves: Sequence[int | float] | None = None,
    alpha: float = 5e-8,
    target_power: float = 0.8,
    p: Any | None = None,
    significance_threshold: float = 5e-8,
    log_af: bool = True,
    mirror: bool = True,
    af_range: tuple[float, float] = (1e-4, 0.5),
    curve_colors: Sequence[str] | None = None,
    point_color: str = "#1f77b4",
    sig_color: str = "#d62728",
    nonsig_color: str = "#aaaaaa",
    curve_label_fmt: str = "N={:,.0f}",
    point_size: float = 8.0,
    ax: Any = None,
    output_path: str | None = None,
    title: str | None = None,
) -> Any:
    """Trumpet plot: allele frequency vs effect size with power curves.

    Parameters
    ----------
    af : (m,) array-like
        Minor / effect allele frequencies.
    beta : (m,) array-like
        Effect sizes (log-OR for binary, beta for quantitative).
    n : int or float, optional
        Sample size for a single power curve overlay.
    n_curves : sequence of int/float, optional
        Multiple sample sizes for multiple power curves. Overrides ``n``.
    alpha : float
        Significance threshold for the power calculation (default 5e-8).
    target_power : float
        Power level for the detection boundary curves (default 0.8).
    p : (m,) array-like, optional
        P-values; if given, points are colored by significance.
    significance_threshold : float
        P-value threshold for coloring significant vs non-significant.
    log_af : bool
        Use log10 scale for the AF axis (default True).
    mirror : bool
        Plot ``|beta|`` and show symmetric curves (default True).
    af_range : (float, float)
        AF range for the power curve grid.
    curve_colors : sequence of str, optional
        Colors for each power curve. Cycled if shorter than n_curves.
    point_color : str
        Default point color when no p-values are given.
    sig_color : str
        Color for significant variants (when p is provided).
    nonsig_color : str
        Color for non-significant variants (when p is provided).
    curve_label_fmt : str
        Format string for power curve legend labels.
    point_size : float
        Marker size for the scatter points.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, creates a new figure.
    output_path : str, optional
        If given, saves the figure to this path.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``af`` and ``beta`` have different lengths or are empty.
    """
    import matplotlib.pyplot as plt

    af_arr = to_numpy(af)
    beta_arr = to_numpy(beta)

    if af_arr.size == 0 or beta_arr.size == 0:
        raise ValueError("af and beta must be non-empty.")
    if af_arr.shape[0] != beta_arr.shape[0]:
        raise ValueError(
            f"af and beta must have the same length, got {af_arr.shape[0]} "
            f"and {beta_arr.shape[0]}."
        )

    if mirror:
        beta_arr = np.abs(beta_arr)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Scatter points
    if p is not None:
        p_arr = to_numpy(p)
        sig_mask = p_arr < significance_threshold
        ax.scatter(
            af_arr[~sig_mask], beta_arr[~sig_mask],
            s=point_size, c=nonsig_color, alpha=0.5, edgecolors="none",
            label="Non-significant", zorder=2,
        )
        ax.scatter(
            af_arr[sig_mask], beta_arr[sig_mask],
            s=point_size, c=sig_color, alpha=0.8, edgecolors="none",
            label="Significant", zorder=3,
        )
    else:
        ax.scatter(
            af_arr, beta_arr,
            s=point_size, c=point_color, alpha=0.6, edgecolors="none",
            zorder=2,
        )

    # Power curves
    sample_sizes: list[float] = []
    if n_curves is not None:
        sample_sizes = list(n_curves)
    elif n is not None:
        sample_sizes = [float(n)]

    if sample_sizes:
        import torch

        from torchgwas.postgwas._power import power_curve as _power_curve

        af_grid = torch.logspace(
            float(np.log10(af_range[0])),
            float(np.log10(af_range[1])),
            200, dtype=torch.float64,
        )
        af_grid_np = af_grid.numpy()

        default_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
                          "#e377c2", "#17becf"]
        if curve_colors is None:
            curve_colors = default_colors

        for i, ns in enumerate(sample_sizes):
            min_beta = _power_curve(
                ns, af_grid, alpha=alpha, target_power=target_power
            ).numpy()
            color = curve_colors[i % len(curve_colors)]
            label = curve_label_fmt.format(ns)
            ax.plot(af_grid_np, min_beta, color=color, linewidth=1.5,
                    label=label, zorder=4)
            if not mirror:
                ax.plot(af_grid_np, -min_beta, color=color, linewidth=1.5,
                        zorder=4)

    # Axis formatting
    if log_af:
        ax.set_xscale("log")

    ax.set_xlabel("Allele Frequency")
    y_label = "|Effect Size|" if mirror else "Effect Size"
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if sample_sizes or p is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ax.grid(True, alpha=0.3)

    # Save if requested
    if output_path is not None:
        ax.get_figure().savefig(output_path, dpi=150, bbox_inches="tight")

    return ax
