"""Gene-level TWAS visualizations.

Three convenience plots for ``TWASResult`` outputs:

- :func:`manhattan_twas` — gene-level Manhattan with one point per
  gene (x-axis: chromosomal position when ``gene.start`` is set,
  otherwise gene order; y-axis: ``-log10(p_twas)``).
- :func:`qq_twas` — QQ plot of TWAS p-values against the uniform null.
- :func:`genomic_inflation_factor_twas` — λ_TWAS = median(z²) / 0.456,
  the gene-level analogue of the genomic-control inflation factor.

All three accept either a :class:`~torchgwas.postgwas.TWASResult` or a
flat list of :class:`~torchgwas.postgwas.TWASGeneResult` (or any
iterable of objects with ``gene_id``, ``p_twas``, ``z_twas``,
``chr``, ``start``, ``gene_name`` attributes).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ._common import chrom_to_int, neglog10_p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genes_from(result_or_list: Any) -> list[Any]:
    """Accept either a TWASResult or an iterable of gene-result dataclasses."""
    if hasattr(result_or_list, "genes"):
        return list(result_or_list.genes)
    return list(result_or_list)


# ---------------------------------------------------------------------------
# Gene-level Manhattan
# ---------------------------------------------------------------------------

def manhattan_twas(
    result: Any,
    *,
    ax: Any = None,
    significance: float = 2.5e-6,
    suggestive: float | None = 1e-4,
    label_top: int = 10,
    colors: tuple[str, ...] = ("#1f77b4", "#aec7e8"),
    significance_color: str = "red",
    suggestive_color: str = "grey",
    label_color: str = "black",
    point_size: float = 18.0,
    figsize: tuple[float, float] = (12.0, 5.0),
    show_grid: bool = True,
):
    """Gene-level Manhattan plot for a TWAS result.

    Each gene is plotted as a single point. When ``gene.chr`` and
    ``gene.start`` are populated (via ``gene_annotation``), points are
    arranged along the genome and colors alternate per chromosome. When
    chromosome / position information is missing the points are
    arranged in their input order along the x-axis.

    Parameters
    ----------
    result : TWASResult or iterable of TWASGeneResult
        TWAS output to plot.
    ax : matplotlib Axes, optional
        Plot on this axis; otherwise create a new figure.
    significance : float
        Genome-wide significance threshold. Default ``2.5e-6`` (≈
        ``0.05 / 20000`` for the GTEx whole-gene-set test).
    suggestive : float or None
        Suggestive threshold (lighter line). Set to ``None`` to hide.
    label_top : int
        Annotate the top-``N`` genes (by ``p_twas``) with their
        ``gene_name`` (fallback ``gene_id``).
    colors, significance_color, suggestive_color, label_color
        Colour overrides.
    point_size : float
        Scatter point size.
    figsize : tuple
        Figure size when ``ax`` is None.
    show_grid : bool
        Show light grid.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib figure and axis (figure may be None if ``ax`` was
        supplied).
    """
    import matplotlib.pyplot as plt

    genes = _genes_from(result)
    if not genes:
        raise ValueError("manhattan_twas: empty gene list.")

    p_arr = np.array([g.p_twas for g in genes], dtype=np.float64)
    neglog = neglog10_p(p_arr)

    has_coords = all(
        getattr(g, "chr", None) is not None and getattr(g, "start", None) is not None
        for g in genes
    )

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if has_coords:
        chrom_int, _uniq = chrom_to_int([g.chr for g in genes])
        starts = np.array([g.start for g in genes], dtype=np.float64)
        # Order by (chrom, start) so the genome reads left-to-right.
        order = np.lexsort((starts, chrom_int))
        chrom_int = chrom_int[order]
        starts = starts[order]
        neglog = neglog[order]
        genes_ordered = [genes[i] for i in order]

        unique_chroms = np.unique(chrom_int)
        # Build cumulative x positions.
        x_positions = np.zeros_like(starts)
        chrom_ticks = []
        offset = 0.0
        for ci in unique_chroms:
            mask = chrom_int == ci
            chrom_starts = starts[mask]
            x_positions[mask] = offset + chrom_starts
            chrom_ticks.append((offset + chrom_starts.mean(), str(ci)))
            offset += float(chrom_starts.max() - chrom_starts.min() + 1)
        # Plot with alternating chrom colors.
        for j, ci in enumerate(unique_chroms):
            mask = chrom_int == ci
            ax.scatter(
                x_positions[mask], neglog[mask],
                s=point_size, color=colors[j % len(colors)],
                edgecolor="none",
            )
        ax.set_xticks([t[0] for t in chrom_ticks])
        ax.set_xticklabels([t[1] for t in chrom_ticks])
        ax.set_xlabel("Chromosome")
    else:
        # No coordinates: just plot in input order.
        x_positions = np.arange(len(genes), dtype=np.float64)
        ax.scatter(
            x_positions, neglog,
            s=point_size, color=colors[0], edgecolor="none",
        )
        ax.set_xlabel("Gene index")
        genes_ordered = genes

    ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{TWAS}})$")
    ax.axhline(-np.log10(significance), color=significance_color,
               linestyle="--", linewidth=0.9,
               label=f"Genome-wide ({significance:.1e})")
    if suggestive is not None:
        ax.axhline(-np.log10(suggestive), color=suggestive_color,
                   linestyle=":", linewidth=0.8,
                   label=f"Suggestive ({suggestive:.1e})")
    if show_grid:
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    # Annotate top-N hits.
    if label_top > 0:
        top_idx = np.argsort(neglog)[::-1][:label_top]
        for i in top_idx:
            g = genes_ordered[i]
            label = getattr(g, "gene_name", None) or g.gene_id
            ax.annotate(
                label, (x_positions[i], neglog[i]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color=label_color,
            )

    ax.set_title("TWAS gene-level Manhattan")
    return fig, ax


# ---------------------------------------------------------------------------
# Gene-level QQ
# ---------------------------------------------------------------------------

def qq_twas(
    result: Any,
    *,
    ax: Any = None,
    point_color: str = "#1f77b4",
    diag_color: str = "grey",
    confidence: bool = True,
    confidence_color: str = "#cccccc",
    figsize: tuple[float, float] = (5.5, 5.5),
):
    """QQ plot of TWAS p-values against the uniform null.

    Plots observed vs expected ``-log10(p)`` quantiles. Includes the
    y = x reference line and (optionally) a 95% Brown-Forsythe-style
    pointwise confidence band derived from the Beta(k, m+1-k) marginal
    of the order statistic.

    Parameters
    ----------
    result : TWASResult or iterable of TWASGeneResult
    ax : matplotlib Axes, optional
    point_color, diag_color, confidence_color : str
    confidence : bool
        Draw the 95 % confidence band.
    figsize : tuple

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    from scipy.stats import beta as _beta  # type: ignore[import-untyped]

    genes = _genes_from(result)
    p_arr = np.array([g.p_twas for g in genes], dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        raise ValueError("qq_twas: empty p-value list.")

    sorted_p = np.sort(p_arr)
    expected = (np.arange(1, m + 1) - 0.5) / m
    observed_log = -np.log10(np.clip(sorted_p, 1e-300, 1.0))
    expected_log = -np.log10(expected)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if confidence:
        k = np.arange(1, m + 1)
        lo = _beta.ppf(0.025, k, m - k + 1)
        hi = _beta.ppf(0.975, k, m - k + 1)
        ax.fill_between(
            expected_log,
            -np.log10(np.clip(hi, 1e-300, 1.0)),
            -np.log10(np.clip(lo, 1e-300, 1.0)),
            color=confidence_color, alpha=0.4, label="95% CI",
        )

    ax.scatter(
        expected_log, observed_log, s=10, color=point_color,
        edgecolor="none", label="Observed",
    )
    lim_max = max(expected_log.max(), observed_log.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color=diag_color, linewidth=0.8,
            linestyle="--", label="y = x")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title("TWAS QQ plot")
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Genomic inflation factor (λ_TWAS)
# ---------------------------------------------------------------------------

def genomic_inflation_factor_twas(result: Any) -> float:
    """λ_TWAS = median(z²) / median(χ²₁) = median(z²) / 0.4549...

    The chi²(1) median is approximately 0.4549364231 (numerically:
    ``scipy.stats.chi2.ppf(0.5, df=1)``). λ_TWAS ~ 1 under the null;
    > 1 indicates inflation (residual confounding); < 1 indicates
    deflation (overcorrection, e.g., too-aggressive covariate
    residualization).

    Parameters
    ----------
    result : TWASResult or iterable of TWASGeneResult

    Returns
    -------
    float
        Inflation factor.
    """
    from scipy.stats import chi2 as _chi2  # type: ignore[import-untyped]
    genes = _genes_from(result)
    if not genes:
        raise ValueError("genomic_inflation_factor_twas: empty gene list.")
    z = np.array([g.z_twas for g in genes], dtype=np.float64)
    chi2_median = float(_chi2.ppf(0.5, df=1))  # 0.4549364231...
    return float(np.median(z ** 2) / chi2_median)
