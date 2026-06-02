"""Shared helpers for :mod:`torchgenomics.api` tier-1 functions."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

logger = logging.getLogger("torchgenomics.api")

ProgressCallback = Callable[[float, str], None]
"""Signature: ``progress_callback(fraction_complete: float, message: str) -> None``.

Used by long-running api functions to emit progress updates. The MCP layer
wraps this to translate calls into MCP ``progress`` notifications.
"""


def resolve_output_dir(output: str | Path | None, default_prefix: str = "torchgenomics_results") -> Path:
    """Create + return an output directory; auto-name a unique one when missing.

    Behaviour:
      - ``output`` is None → ``./<default_prefix>/<8-char-id>/`` (unique per call).
      - ``output`` is an existing directory → reused.
      - ``output`` is a non-existent path → created as a directory.

    Always returns an absolute :class:`Path` with the directory ensured to exist.
    """
    if output is None:
        run_id = uuid.uuid4().hex[:8]
        path = Path.cwd() / default_prefix / run_id
    else:
        path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def select_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` → ``"cuda"`` if available, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def top_hits(df: pd.DataFrame, k: int = 50, p_column: str = "P") -> pd.DataFrame:
    """Return the K smallest-p rows from ``df`` (no copy if already small)."""
    if df.empty or p_column not in df.columns:
        return df.head(k)
    return df.nsmallest(min(k, len(df)), p_column).reset_index(drop=True)


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    """Context manager yielding a ``.elapsed()`` callable.

    Usage::

        with timed() as elapsed:
            do_work()
        runtime_s = elapsed()
    """
    start = time.monotonic()
    captured = [None]  # type: ignore[var-annotated]

    def _elapsed() -> float:
        if captured[0] is None:
            return time.monotonic() - start
        return captured[0]

    try:
        yield _elapsed
    finally:
        captured[0] = time.monotonic() - start


def emit_progress(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    """Best-effort progress emit; swallows exceptions so a buggy callback never crashes work."""
    if callback is None:
        return
    try:
        callback(max(0.0, min(1.0, float(fraction))), str(message))
    except Exception:  # noqa: BLE001
        logger.debug("progress_callback raised; ignoring.", exc_info=True)
