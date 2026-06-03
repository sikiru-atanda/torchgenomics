"""Centralized device / size / capability dispatcher for hot loops.

Hot loops in TorchGenomics now have up to three execution paths:

- ``"gpu"``    — torch ops on a CUDA device. Best for big embarrassingly
                 parallel work whose data is *already* on the GPU. Never
                 reached by proactively moving CPU data to the GPU — that
                 PCIe round-trip almost always loses to staying local.
- ``"native"`` — a hand-rolled C++ extension under ``torchgenomics._native``.
                 Best for sequential / branchy code on CPU-resident data.
                 Always reads/writes via ``numpy`` views of host memory.
- ``"python"`` — the pure-Python / pure-torch reference implementation.
                 Always available; serves as the algorithmic spec and the
                 fallback when no compiler / no GPU / problem too small.

The decision is a function of *(device, problem size, build capability)*.
Every dispatcher in the codebase should call :func:`select_path` rather
than rolling its own ``HAS_NATIVE_*`` check, so the heuristics live in
exactly one place.

Environment knobs (both opt-out, default off):

- ``TORCHGENOMICS_DISABLE_NATIVE=1`` — never use the C++ path. Used by the
  test suite to exercise the Python reference and by users debugging
  numerical drift.
- ``TORCHGENOMICS_DISABLE_GPU=1`` — never run the dedicated GPU path. The
  data may still live on CUDA; the dispatcher just falls through to the
  generic torch (``"python"``) branch, which is fine because torch ops
  run wherever the tensor lives.
"""

from __future__ import annotations

import os
import warnings
from typing import Literal, Optional

import torch

Path = Literal["gpu", "native", "python"]


_NEW_ENV_PREFIX = "TORCHGENOMICS_"
_OLD_ENV_PREFIX = "TORCHGWAS_"
_warned_env_aliases: set[str] = set()


def env_var(name: str) -> Optional[str]:
    """Read a ``TORCHGENOMICS_*`` env var with backwards-compat for the legacy
    ``TORCHGWAS_*`` name.

    If only the legacy name is set, emits a one-shot :class:`DeprecationWarning`
    the first time it is read in the process and returns its value. The legacy
    alias is removed in v1.0.0.
    """
    new_val = os.environ.get(name)
    if new_val is not None:
        return new_val
    if not name.startswith(_NEW_ENV_PREFIX):
        return None
    old_name = _OLD_ENV_PREFIX + name[len(_NEW_ENV_PREFIX):]
    old_val = os.environ.get(old_name)
    if old_val is None:
        return None
    if old_name not in _warned_env_aliases:
        _warned_env_aliases.add(old_name)
        warnings.warn(
            f"Environment variable `{old_name}` is deprecated; use `{name}` instead. "
            "The legacy alias will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return old_val


# Default size thresholds. These are deliberately conservative — below the
# native threshold, even the C++ extension's overhead (numpy view + GIL
# release) does not pay back; below the GPU threshold, the kernel-launch
# fixed cost (~5–50 µs) dominates any GPU win.
DEFAULT_NATIVE_THRESHOLD = 1_000
DEFAULT_GPU_THRESHOLD = 10_000


def native_disabled() -> bool:
    """Whether ``TORCHGENOMICS_DISABLE_NATIVE`` is set in the environment.

    Also accepts the legacy ``TORCHGWAS_DISABLE_NATIVE`` (with one-shot
    DeprecationWarning) through v0.x; legacy alias removed in v1.0.0.
    """
    return bool(env_var("TORCHGENOMICS_DISABLE_NATIVE"))


def gpu_disabled() -> bool:
    """Whether ``TORCHGENOMICS_DISABLE_GPU`` is set in the environment.

    Also accepts the legacy ``TORCHGWAS_DISABLE_GPU`` (with one-shot
    DeprecationWarning) through v0.x; legacy alias removed in v1.0.0.
    """
    return bool(env_var("TORCHGENOMICS_DISABLE_GPU"))


def select_path(
    tensor: torch.Tensor,
    *,
    has_native: bool,
    has_gpu_kernel: bool = False,
    gpu_threshold: int = DEFAULT_GPU_THRESHOLD,
    native_threshold: int = DEFAULT_NATIVE_THRESHOLD,
) -> Path:
    """Pick the execution path for a hot loop given an input tensor.

    Parameters
    ----------
    tensor : Tensor
        The principal input. Its ``.device`` and ``.numel()`` drive the
        decision; its dtype is *not* inspected here — callers can add
        their own dtype guard if the C++ / GPU paths are dtype-restricted.
    has_native : bool
        Whether the relevant ``torchgenomics._native._<name>_native`` extension
        is loaded. Pass the module-level ``HAS_NATIVE_<NAME>`` flag.
    has_gpu_kernel : bool
        Whether the caller has a *dedicated* torch-GPU kernel for this
        operation (a hand-tuned implementation that beats the generic
        Python path on CUDA). Defaults to ``False``: most current
        dispatchers rely on the generic torch path on GPU, which is what
        the ``"python"`` branch does anyway.
    gpu_threshold, native_threshold : int
        Per-call overrides for the size thresholds. Most callers should
        leave these at the defaults.

    Returns
    -------
    {"gpu", "native", "python"}
        The label the caller should branch on. The dispatcher itself
        never moves data — the caller is expected to honour the chosen
        path on the tensor's existing device.

    Notes
    -----
    The function never pulls a CUDA tensor to host. If a CUDA tensor is
    too small for the GPU path *and* there is no dedicated GPU kernel,
    the result is ``"python"`` — torch ops run on whatever device the
    tensor already lives on, so no PCIe round-trip is required.
    """
    on_cuda = tensor.device.type == "cuda"
    n = tensor.numel()

    if on_cuda:
        if has_gpu_kernel and not gpu_disabled() and n >= gpu_threshold:
            return "gpu"
        # Stay on the GPU via the generic torch path. Never copy to host
        # just to use the C++ extension — the transfer would dominate.
        return "python"

    # CPU device.
    if has_native and not native_disabled() and n >= native_threshold:
        return "native"
    return "python"


__all__ = [
    "Path",
    "DEFAULT_NATIVE_THRESHOLD",
    "DEFAULT_GPU_THRESHOLD",
    "native_disabled",
    "gpu_disabled",
    "select_path",
]
