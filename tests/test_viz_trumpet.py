"""Tests for torchgenomics.viz.trumpet_plot (trumpet plot visualization)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from torchgenomics.viz import trumpet_plot

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_data(m=200, seed=42):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.01, 0.5, size=m)
    beta = rng.normal(0.0, 0.1, size=m)
    return af, beta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trumpet_returns_axes():
    """Basic call returns a matplotlib Axes object."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_trumpet_accepts_torch():
    """Torch tensor inputs are accepted without error."""
    af, beta = _make_data()
    ax = trumpet_plot(torch.tensor(af), torch.tensor(beta))
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_trumpet_single_power_curve():
    """Passing n=10000 draws at least one power-curve line."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta, n=10000)
    lines = ax.get_lines()
    assert len(lines) >= 1
    plt.close("all")


def test_trumpet_multiple_curves():
    """Passing n_curves=[10000, 50000] draws at least two lines."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta, n_curves=[10000, 50000])
    lines = ax.get_lines()
    assert len(lines) >= 2
    plt.close("all")


def test_trumpet_p_coloring():
    """Providing p-values creates scatter collections for sig/nonsig."""
    af, beta = _make_data()
    rng = np.random.default_rng(99)
    p = rng.uniform(0, 1, size=len(af))
    # Force a few significant hits
    p[:5] = 1e-10

    ax = trumpet_plot(af, beta, p=p)
    # Should have at least two PathCollections (sig + nonsig scatter)
    collections = ax.collections
    assert len(collections) >= 2
    plt.close("all")


def test_trumpet_log_scale():
    """log_af=True sets the x-axis to log scale."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta, log_af=True)
    assert ax.get_xscale() == "log"
    plt.close("all")


def test_trumpet_linear_scale():
    """log_af=False keeps the x-axis linear."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta, log_af=False)
    assert ax.get_xscale() == "linear"
    plt.close("all")


def test_trumpet_mirror_mode():
    """mirror=True means all plotted y values (scatter) are >= 0."""
    af, beta = _make_data()
    ax = trumpet_plot(af, beta, mirror=True)
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            y_vals = offsets[:, 1]
            assert np.all(y_vals >= 0), "mirror=True should plot |beta|"
    plt.close("all")


def test_trumpet_output_path(tmp_path):
    """output_path writes a file to disk."""
    af, beta = _make_data()
    out = str(tmp_path / "trumpet.png")
    trumpet_plot(af, beta, output_path=out)
    import os
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
    plt.close("all")


def test_trumpet_empty_raises():
    """Empty arrays raise ValueError."""
    with pytest.raises(ValueError):
        trumpet_plot(np.array([]), np.array([]))


def test_trumpet_length_mismatch():
    """Mismatched af/beta lengths raise ValueError."""
    with pytest.raises(ValueError):
        trumpet_plot(np.array([0.1, 0.2, 0.3]), np.array([0.01, 0.02]))
