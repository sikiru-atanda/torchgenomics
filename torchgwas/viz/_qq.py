"""QQ plot for GWAS p-values.

Expected quantiles under the uniform null are ``(i - 0.5) / m``; the 95%
confidence band at each rank ``i`` comes from the Beta(i, m - i + 1)
order-statistic distribution, with the upper and lower bounds transformed
through ``-log10`` to match the axis.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._common import to_numpy


def _lambda_gc(p: np.ndarray) -> float:
    """Genomic inflation factor ``median(chi2) / chi2_0.5(df=1)``."""
    from scipy.stats import chi2

    chi2_obs = chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / chi2.isf(0.5, df=1))


def qq_plot(
    p: Any,
    *,
    confidence_band: bool = True,
    lambda_gc: bool = True,
    ax=None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    point_color: str = "#1f77b4",
    band_color: str = "#cccccc",
    point_size: float = 8.0,
):
    """Draw a QQ plot of observed vs expected ``-log10(p)``.

    Parameters
    ----------
    p : (m,) array-like
        Association p-values.
    confidence_band : bool
        Shade the 95% pointwise confidence band under the uniform null.
    lambda_gc : bool
        Annotate the genomic inflation factor λ_GC in the top-left corner.
    ax : matplotlib Axes or None
        Draw onto this axis if supplied; otherwise create a new figure.
    output_path : str or None
        Save the figure to disk if set.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist

    p = to_numpy(p).astype(np.float64)
    if p.size == 0:
        raise ValueError("empty input to qq_plot")
    p = np.clip(p, 1e-300, 1.0)
    p_sorted = np.sort(p)
    m = p_sorted.size

    expected_p = (np.arange(1, m + 1) - 0.5) / m
    x = -np.log10(expected_p)
    y = -np.log10(p_sorted)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if confidence_band:
        # Beta(i, m - i + 1) 95% CI on the raw p-value axis.
        ranks = np.arange(1, m + 1)
        lo = beta_dist.ppf(0.025, ranks, m - ranks + 1)
        hi = beta_dist.ppf(0.975, ranks, m - ranks + 1)
        y_lo = -np.log10(np.clip(hi, 1e-300, 1.0))  # high p -> low -log10
        y_hi = -np.log10(np.clip(lo, 1e-300, 1.0))
        # x-axis increases with rank-based -log10 expected_p.
        ax.fill_between(x[::-1], y_lo[::-1], y_hi[::-1], color=band_color,
                        alpha=0.6, linewidth=0, zorder=1)

    lim = max(float(x.max()), float(y.max())) * 1.02
    ax.plot([0, lim], [0, lim], color="red", linewidth=0.9, zorder=2)
    ax.scatter(x, y, s=point_size, c=point_color, linewidths=0,
               zorder=3, rasterized=True)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if lambda_gc:
        try:
            lam = _lambda_gc(p)
            ax.text(
                0.04, 0.93, rf"$\lambda_{{GC}} = {lam:.3f}$",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="none", alpha=0.7),
            )
        except Exception:
            pass

    if title:
        ax.set_title(title)

    if output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)

    return ax
