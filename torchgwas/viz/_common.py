"""Shared helpers for the viz module."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert tensor / list / numpy array to a 1-D float64 numpy array."""
    if x is None:
        return np.asarray([], dtype=np.float64)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype.kind not in ("f", "i", "u"):
        arr = arr.astype(np.float64)
    return arr


def neglog10_p(p: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """``-log10(p)`` with a floor to avoid ``inf`` from zero p-values."""
    return -np.log10(np.clip(p, eps, 1.0))


def chrom_to_int(chrom: Iterable[Any]) -> np.ndarray:
    """Map chromosome labels to a stable integer ordering.

    Numeric chromosomes sort numerically (1, 2, ..., 22); sex/other
    chromosomes (X, Y, MT/M, XY) follow in their canonical order; anything
    else is appended alphabetically. Returns a 0-indexed (m,) int array.
    """
    labels = [str(c) for c in chrom]
    uniq = sorted(set(labels), key=_chrom_sort_key)
    order = {c: i for i, c in enumerate(uniq)}
    return np.asarray([order[c] for c in labels], dtype=np.int64), uniq


def _chrom_sort_key(label: str) -> tuple[int, float, str]:
    lab = label.upper().lstrip("CHR")
    specials = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}
    if lab.isdigit():
        return (0, float(lab), lab)
    if lab in specials:
        return (1, float(specials[lab]), lab)
    return (2, 0.0, lab)
