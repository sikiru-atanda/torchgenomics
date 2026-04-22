"""Haploview-style LD triangle plot.

Draws a 45°-rotated triangular heatmap of pairwise LD (r² or D') that
mirrors the visual output of Haploview (Barrett et al. 2005). Cells are
diamond-shaped patches whose color encodes the LD magnitude, and optional
block overlays are rendered as thick black triangles over the cells that
belong to each detected haplotype block.

Input conventions:

* ``ld_matrix``: either a full ``(m, m)`` symmetric matrix, or just the
  upper triangle — both are accepted and the code reads only the
  ``i < j`` entries.
* ``blocks``: either a sequence of ``(start, end)`` index pairs
  (end-inclusive) or a sequence of ``LDBlock`` dataclasses (in which case
  ``block.variant_indices`` is used and the block is rendered as the
  convex range ``[min(idx), max(idx)]``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ._common import to_numpy


def _block_ranges(blocks: Iterable[Any]) -> list[tuple[int, int]]:
    """Normalize ``blocks`` input to a list of ``(start, end)`` int pairs."""
    out: list[tuple[int, int]] = []
    for b in blocks:
        if hasattr(b, "variant_indices"):
            idxs = list(b.variant_indices)
            if not idxs:
                continue
            out.append((int(min(idxs)), int(max(idxs))))
        elif isinstance(b, (tuple, list)) and len(b) == 2:
            out.append((int(b[0]), int(b[1])))
        else:
            raise TypeError(
                "blocks entries must be (start, end) tuples or LDBlock "
                f"objects; got {type(b).__name__}"
            )
    return out


def haploview_plot(
    ld_matrix: Any,
    *,
    blocks: Sequence[Any] | None = None,
    metric: str = "r2",
    cmap: str | None = None,
    show_snp_track: bool = True,
    snp_labels: Sequence[str] | None = None,
    ax=None,
    output_path: str | None = None,
    title: str | None = None,
    colorbar: bool = True,
):
    """Haploview-style LD triangle.

    Parameters
    ----------
    ld_matrix : (m, m) array-like
        Pairwise LD matrix (``r²`` or ``|D'|``). Only the upper triangle is
        read.
    blocks : sequence of LDBlock or (start, end) tuples, optional
        Haplotype blocks to overlay as black triangular outlines.
    metric : {"r2", "dprime"}
        Drives the default colormap and colorbar label.
    cmap : str or None
        Matplotlib colormap name; defaults to ``Reds`` (r²) or ``Blues``
        (D').
    show_snp_track : bool
        Draw tick marks along the top for each SNP position.
    snp_labels : sequence of str or None
        Per-SNP labels drawn above the track; supply sparsely (e.g. only
        the lead SNPs) to avoid overlap.
    ax : matplotlib Axes or None
        Draw on this Axes; otherwise create a new figure.
    output_path : str or None
        Save the figure.
    title : str or None
        Axes title.
    colorbar : bool
        Draw a vertical colorbar on the right.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon

    L = to_numpy(ld_matrix).astype(np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("ld_matrix must be a square 2-D array")
    m = L.shape[0]
    if m < 2:
        raise ValueError("haploview_plot requires at least 2 variants")

    if cmap is None:
        cmap = "Reds" if metric == "r2" else "Blues"

    # Build diamond cells for every i < j pair.
    ii, jj = np.triu_indices(m, k=1)
    values = L[ii, jj]
    # Clip to [0, 1] — both r² and |D'| are in that range.
    values = np.clip(values, 0.0, 1.0)

    # Each diamond center: (cx, cy) with cy <= 0 (triangle hangs below row 0).
    cx = (ii + jj) / 2.0
    cy = -(jj - ii) / 2.0
    # Vertices: up / right / down / left around the center with
    # half-diagonal 0.5 (so that neighbouring cells tile perfectly).
    offsets = np.array([[0.0, 0.5], [0.5, 0.0], [0.0, -0.5], [-0.5, 0.0]])
    verts = np.stack(
        [np.column_stack([cx, cy]) + offsets[k] for k in range(4)],
        axis=1,
    )  # (n_pairs, 4, 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(min(1 + 0.3 * m, 14), 5))
    else:
        fig = ax.figure

    pc = PolyCollection(
        verts, array=values, cmap=cmap,
        edgecolors="white", linewidths=0.3,
    )
    pc.set_clim(0.0, 1.0)
    ax.add_collection(pc)

    if colorbar:
        cbar = fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(r"$r^2$" if metric == "r2" else r"$|D'|$")

    # Block overlays: a triangle whose base spans SNPs [s, e] along y=0 and
    # whose apex sits at the (s, e) pair cell.
    if blocks is not None:
        for s, e in _block_ranges(blocks):
            if e <= s or s < 0 or e >= m:
                continue
            base_left = (s - 0.5, 0.0)
            base_right = (e + 0.5, 0.0)
            apex = ((s + e) / 2.0, -(e - s + 1) / 2.0)
            tri = Polygon(
                [base_left, base_right, apex],
                closed=True, fill=False,
                edgecolor="black", linewidth=1.8, zorder=5,
            )
            ax.add_patch(tri)

    # SNP marker track along y = 0.
    if show_snp_track:
        ax.scatter(
            np.arange(m), np.zeros(m),
            s=14, marker="|", color="black", zorder=6,
        )
        if snp_labels is not None:
            if len(snp_labels) != m:
                raise ValueError("snp_labels must have length m")
            for i, lab in enumerate(snp_labels):
                if lab:
                    ax.text(
                        i, 0.4, str(lab), rotation=90,
                        ha="center", va="bottom", fontsize=7,
                    )

    ax.set_xlim(-0.8, m - 0.2)
    ax.set_ylim(-(m / 2.0) - 0.5, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("SNP index")
    ax.set_yticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    if title:
        ax.set_title(title)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return ax
