"""Tests for ``torchgenomics._dispatch.select_path``.

These exercise the device / size / capability decision matrix that every
hot-loop dispatcher consults. CUDA branches are skipped on machines
without a GPU so the suite still runs in CPU-only CI.
"""

from __future__ import annotations

import pytest
import torch

from torchgenomics._dispatch import (
    DEFAULT_GPU_THRESHOLD,
    DEFAULT_NATIVE_THRESHOLD,
    gpu_disabled,
    native_disabled,
    select_path,
)

# ---------------------------------------------------------------------------
# CPU branches
# ---------------------------------------------------------------------------


def _clear_env(monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_GPU", raising=False)


def test_cpu_native_when_built_and_large(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_NATIVE_THRESHOLD)
    assert select_path(t, has_native=True) == "native"


def test_cpu_python_when_no_native_build(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_NATIVE_THRESHOLD * 10)
    assert select_path(t, has_native=False) == "python"


def test_cpu_python_when_native_disabled_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    t = torch.zeros(DEFAULT_NATIVE_THRESHOLD * 10)
    assert select_path(t, has_native=True) == "python"
    assert native_disabled() is True


def test_cpu_python_below_native_threshold(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_NATIVE_THRESHOLD - 1)
    assert select_path(t, has_native=True) == "python"


def test_cpu_threshold_override(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(50)
    # Smaller threshold flips the decision.
    assert select_path(t, has_native=True, native_threshold=10) == "native"
    assert select_path(t, has_native=True, native_threshold=100) == "python"


# ---------------------------------------------------------------------------
# CUDA branches (skipped without a GPU)
# ---------------------------------------------------------------------------


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@cuda_required
def test_cuda_uses_gpu_path_when_kernel_available(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_GPU_THRESHOLD, device="cuda")
    assert select_path(t, has_native=True, has_gpu_kernel=True) == "gpu"


@cuda_required
def test_cuda_falls_through_to_python_without_gpu_kernel(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_GPU_THRESHOLD, device="cuda")
    # No dedicated GPU kernel — must NOT pull to host for the C++ path.
    assert select_path(t, has_native=True, has_gpu_kernel=False) == "python"


@cuda_required
def test_cuda_below_gpu_threshold_falls_through(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_GPU_THRESHOLD - 1, device="cuda")
    assert select_path(t, has_native=True, has_gpu_kernel=True) == "python"


@cuda_required
def test_cuda_disabled_via_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_GPU", "1")
    t = torch.zeros(DEFAULT_GPU_THRESHOLD * 10, device="cuda")
    assert select_path(t, has_native=True, has_gpu_kernel=True) == "python"
    assert gpu_disabled() is True


# ---------------------------------------------------------------------------
# Invariant: dispatcher never moves data
# ---------------------------------------------------------------------------


def test_select_path_does_not_touch_tensor_device(monkeypatch):
    _clear_env(monkeypatch)
    t = torch.zeros(DEFAULT_NATIVE_THRESHOLD * 2)
    dev_before = t.device
    _ = select_path(t, has_native=True)
    assert t.device == dev_before
