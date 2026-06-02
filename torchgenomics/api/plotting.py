"""Plotting tier-1 API: :func:`manhattan`, :func:`qq`.

Both functions return :class:`PlotResult` objects that carry either a
matplotlib Figure (for notebook use) or a base64 PNG (for MCP transport
through JSON), or both. Output PNG path is set when an ``output`` arg
is provided.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import resolve_output_dir, timed
from ._results import PlotResult


def _read_sumstats(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Coerce a path or DataFrame to a sumstats DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def _figure_to_outputs(
    fig,
    output: str | Path | None,
    fmt: str,
    dpi: int,
    return_base64: bool,
) -> tuple[Path | None, str | None]:
    """Save the figure to disk and/or to base64 PNG. Returns (png_path, png_b64)."""
    png_path: Path | None = None
    png_b64: str | None = None
    if output is not None:
        png_path = Path(output)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(png_path), format=fmt, dpi=dpi, bbox_inches="tight")
    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return png_path, png_b64


@tool(
    name="tg_manhattan",
    title="Manhattan plot of GWAS results",
    description=(
        "Generate a Manhattan plot (variant -log10(p) vs chromosome position) "
        "from a GWAS sumstats file or DataFrame. Returns PlotResult carrying "
        "the matplotlib Figure (for inline notebook display) and, when "
        "`return_base64=True`, a base64-encoded PNG that travels well through "
        "JSON / MCP tool responses."
    ),
    long_running=False,
    category="plot",
    tags=["plot", "manhattan", "visualization", "gwas"],
)
def manhattan(
    sumstats: str | Path | pd.DataFrame,
    *,
    output: str | Path | None = None,
    significance_threshold: float = 5e-8,
    suggestive_threshold: float | None = 1e-5,
    title: str | None = None,
    chr_col: str = "CHR",
    pos_col: str = "POS",
    p_col: str = "P",
    width_in: float = 12.0,
    height_in: float = 4.5,
    dpi: int = 100,
    return_base64: bool = False,
    fmt: Literal["png", "pdf", "svg"] = "png",
) -> PlotResult:
    """Render a Manhattan plot."""
    with timed() as elapsed:
        import matplotlib.pyplot as plt

        from ..viz import manhattan_plot

        df = _read_sumstats(sumstats)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        manhattan_plot(
            chrom=df[chr_col].astype(str).tolist(),
            pos=df[pos_col].astype(int).tolist(),
            p=df[p_col].astype(float).tolist(),
            significance_threshold=significance_threshold,
            suggestive_threshold=suggestive_threshold,
            title=title,
            ax=ax,
        )
        png_path, png_b64 = _figure_to_outputs(fig, output, fmt, dpi, return_base64)

        output_files = {"image": png_path} if png_path else {}

        return PlotResult(
            runtime_s=elapsed(),
            output_files=output_files,
            figure=fig,
            png_path=png_path,
            png_base64=png_b64,
            width_px=int(width_in * dpi),
            height_px=int(height_in * dpi),
        )


@tool(
    name="tg_qq",
    title="Q-Q plot of GWAS p-values",
    description=(
        "Generate a Q-Q (quantile-quantile) plot of observed vs expected "
        "p-values to check calibration of a GWAS scan. Inflation (λ_GC > 1) "
        "and deflation are visually obvious on this plot. Returns PlotResult "
        "with the Figure and optional base64 PNG."
    ),
    long_running=False,
    category="plot",
    tags=["plot", "qq", "calibration", "lambda-gc"],
)
def qq(
    sumstats: str | Path | pd.DataFrame,
    *,
    output: str | Path | None = None,
    p_col: str = "P",
    title: str | None = None,
    width_in: float = 6.0,
    height_in: float = 6.0,
    dpi: int = 100,
    return_base64: bool = False,
    fmt: Literal["png", "pdf", "svg"] = "png",
) -> PlotResult:
    """Render a Q-Q plot."""
    with timed() as elapsed:
        import matplotlib.pyplot as plt

        from ..viz import qq_plot

        df = _read_sumstats(sumstats)
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        qq_plot(
            p=df[p_col].astype(float).tolist(),
            title=title,
            ax=ax,
        )
        png_path, png_b64 = _figure_to_outputs(fig, output, fmt, dpi, return_base64)

        output_files = {"image": png_path} if png_path else {}

        return PlotResult(
            runtime_s=elapsed(),
            output_files=output_files,
            figure=fig,
            png_path=png_path,
            png_base64=png_b64,
            width_px=int(width_in * dpi),
            height_px=int(height_in * dpi),
        )
