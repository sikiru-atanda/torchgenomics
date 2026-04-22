"""GWAS visualization: Manhattan plots (classic + Miami), QQ plots, and
Haploview-style LD triangle views.

All entry points accept numpy arrays, torch tensors, or Python lists and
return the matplotlib ``Axes`` (or tuple of Axes) they drew on, so they
compose with external figures. Passing ``output_path`` also writes the
figure to disk.
"""

from __future__ import annotations

from ._haploview import haploview_plot
from ._manhattan import circos_manhattan_plot, manhattan_plot, miami_plot
from ._qq import qq_plot
from ._trumpet import trumpet_plot

__all__ = [
    "manhattan_plot",
    "circos_manhattan_plot",
    "miami_plot",
    "qq_plot",
    "haploview_plot",
    "trumpet_plot",
]
