"""Back-compat shim — the real implementations now live in :mod:`torchgenomics.viz`."""

from __future__ import annotations

from torchgenomics.viz import (
    circos_manhattan_plot,
    haploview_plot,
    manhattan_plot,
    miami_plot,
    qq_plot,
    trumpet_plot,
)

__all__ = [
    "manhattan_plot",
    "circos_manhattan_plot",
    "miami_plot",
    "qq_plot",
    "haploview_plot",
    "trumpet_plot",
]
