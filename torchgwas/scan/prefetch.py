"""Async prefetch: overlap CPU I/O with GPU compute via pinned memory.

Uses a background thread + bounded queue to load the next chunk while
the GPU processes the current one.  Only active when device is CUDA;
falls back to simple passthrough on CPU (no threading overhead).
"""

from __future__ import annotations

import logging
from queue import Queue
from threading import Thread
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)

_SENTINEL = object()  # end-of-stream marker


class PrefetchIterator:
    """Wraps a genotype chunk iterator with async prefetch.

    While the GPU processes the current chunk, the next chunk is loaded
    from disk and transferred to pinned memory on a background thread.
    When the GPU is ready, the pinned chunk is moved to GPU with
    non-blocking transfer — overlapping compute and I/O.

    Only active when device is CUDA. Falls back to simple passthrough
    on CPU (no threading overhead).
    """

    def __init__(
        self,
        chunk_iter: Iterator[Tuple[Tensor, VariantMeta]],
        device: torch.device,
        n_prefetch: int = 2,
    ) -> None:
        self._iter = chunk_iter
        self._device = device
        self._n_prefetch = n_prefetch
        self._use_prefetch = device.type == "cuda"

    def __iter__(self) -> Iterator[Tuple[Tensor, VariantMeta]]:
        if not self._use_prefetch:
            yield from self._iter
            return

        # Threaded producer-consumer with bounded queue for backpressure
        queue: Queue = Queue(maxsize=self._n_prefetch)

        def _producer():
            try:
                for G_chunk, vmeta in self._iter:
                    # Pin memory for fast GPU transfer
                    if not G_chunk.is_pinned():
                        G_chunk = G_chunk.pin_memory()
                    queue.put((G_chunk, vmeta))
            except Exception as exc:
                # Propagate exception to consumer
                queue.put(exc)
            finally:
                queue.put(_SENTINEL)

        thread = Thread(target=_producer, daemon=True)
        thread.start()

        try:
            while True:
                item = queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                G_chunk, vmeta = item
                # Non-blocking transfer to GPU
                G_gpu = G_chunk.to(self._device, non_blocking=True)
                yield G_gpu, vmeta
        finally:
            thread.join(timeout=5.0)


def move_nullfit_to_device(null_fit, device: torch.device):
    """Move all tensors in a NullFit (or custom null fit) to the target device.

    Handles both standard NullFit and custom types (OCFNullFit,
    WithinFamilyNullFit, etc.) by moving known fields first,
    then scanning for any additional tensor attributes.
    """
    if getattr(null_fit, "device", None) == device:
        return null_fit

    # Known NullFit fields (fast path)
    for attr in ("eigenvalues", "eigenvectors", "Y_rot", "X0_rot",
                 "M00", "b0", "weights", "Vg", "Ve"):
        val = getattr(null_fit, attr, None)
        if val is not None and isinstance(val, Tensor):
            setattr(null_fit, attr, val.to(device))

    # Generic: move any additional tensor attributes (for custom null fits)
    for attr in vars(null_fit):
        if attr.startswith("_"):
            continue
        val = getattr(null_fit, attr, None)
        if isinstance(val, Tensor) and val.device != device:
            setattr(null_fit, attr, val.to(device))

    null_fit.device = device
    return null_fit
