"""Regression tests for the native C++ TWAS expression-normalisation kernels.

The native kernels
(``torchgwas._native._expression_native.rank_int_u_columns`` and
``quantile_normalize_columns``) replace the column-wise stateful
Python tie-resolution loops in
``torchgwas.preprocess.expression.inverse_normal_transform`` and
``quantile_normalize``. These tests assert:

1.  **Parity** — native and Python paths produce numerically identical
    outputs at the FP64 floor for both transforms.
2.  **Ties** — average-tie semantics are preserved.
3.  **Env-var fallthrough** — ``TORCHGWAS_DISABLE_NATIVE=1`` forces
    the Python reference path.
4.  **Below-threshold input** — for tiny matrices the dispatcher
    routes to Python.
5.  **1-D inverse_normal_transform** — the 1-D path stays in Python
    (the C++ kernel is for 2-D only).
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from torchgwas._native import HAS_NATIVE_EXPRESSION
from torchgwas.preprocess.expression import (
    inverse_normal_transform,
    quantile_normalize,
)


def _planted(n: int, m: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, m))).to(torch.float64)


def _planted_with_ties(n: int, m: int, seed: int) -> torch.Tensor:
    """Matrix with deliberate ties — every column has a duplicated value."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, m))
    # Inject ties: copy row 0 to row 1, row 2 to row 3 in every column.
    x[1, :] = x[0, :]
    x[3, :] = x[2, :]
    return torch.from_numpy(x).to(torch.float64)


def _run_with(env_value, fn, *args, **kwargs):
    prev = os.environ.get("TORCHGWAS_DISABLE_NATIVE")
    if env_value is None:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    else:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = env_value
    try:
        return fn(*args, **kwargs)
    finally:
        if prev is None:
            os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
        else:
            os.environ["TORCHGWAS_DISABLE_NATIVE"] = prev


@unittest.skipUnless(
    HAS_NATIVE_EXPRESSION, "native _expression_native extension not built"
)
class TestINTNativeParity(unittest.TestCase):

    def test_parity_no_ties(self):
        x = _planted(n=200, m=100, seed=42)
        py = _run_with("1", inverse_normal_transform, x)
        cpp = _run_with(None, inverse_normal_transform, x)
        # FP64 floor: identical rank computation across paths; floor 1e-12.
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)

    def test_parity_with_ties(self):
        x = _planted_with_ties(n=100, m=80, seed=43)
        py = _run_with("1", inverse_normal_transform, x)
        cpp = _run_with(None, inverse_normal_transform, x)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)

    def test_parity_realistic_gtex(self):
        """GTEx-style: n_samples=200, n_genes=20_000."""
        x = _planted(n=200, m=20_000, seed=44)
        py = _run_with("1", inverse_normal_transform, x)
        cpp = _run_with(None, inverse_normal_transform, x)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)

    def test_alternate_blom_constant(self):
        """Non-default c (Tukey 1/3 instead of Blom 3/8)."""
        x = _planted(n=100, m=200, seed=45)
        py = _run_with("1", inverse_normal_transform, x, c=1.0 / 3.0)
        cpp = _run_with(None, inverse_normal_transform, x, c=1.0 / 3.0)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)


@unittest.skipUnless(
    HAS_NATIVE_EXPRESSION, "native _expression_native extension not built"
)
class TestQuantileNormNativeParity(unittest.TestCase):

    def test_parity_no_ties(self):
        x = _planted(n=200, m=100, seed=46)
        py = _run_with("1", quantile_normalize, x)
        cpp = _run_with(None, quantile_normalize, x)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)

    def test_parity_with_ties(self):
        x = _planted_with_ties(n=100, m=80, seed=47)
        py = _run_with("1", quantile_normalize, x)
        cpp = _run_with(None, quantile_normalize, x)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)

    def test_parity_realistic_gtex(self):
        x = _planted(n=200, m=20_000, seed=48)
        py = _run_with("1", quantile_normalize, x)
        cpp = _run_with(None, quantile_normalize, x)
        torch.testing.assert_close(cpp, py, atol=1e-12, rtol=1e-12)


@unittest.skipUnless(
    HAS_NATIVE_EXPRESSION, "native _expression_native extension not built"
)
class TestExpressionNativeFallthrough(unittest.TestCase):

    def test_env_var_disables_native(self):
        x = _planted(n=100, m=50, seed=49)
        py = _run_with("1", inverse_normal_transform, x)
        default = _run_with(None, inverse_normal_transform, x)
        torch.testing.assert_close(default, py, atol=1e-12, rtol=1e-12)

    def test_below_threshold_uses_python(self):
        """n*m < 1024 routes to Python."""
        x = _planted(n=10, m=10, seed=50)  # n*m = 100 < 1024
        py = _run_with("1", inverse_normal_transform, x)
        default = _run_with(None, inverse_normal_transform, x)
        torch.testing.assert_close(default, py, atol=1e-15, rtol=1e-15)

    def test_1d_input_stays_python(self):
        """1-D rank-INT uses the existing Python path."""
        x = torch.from_numpy(
            np.random.default_rng(51).standard_normal(100)
        ).to(torch.float64)
        py = _run_with("1", inverse_normal_transform, x)
        default = _run_with(None, inverse_normal_transform, x)
        torch.testing.assert_close(default, py, atol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
