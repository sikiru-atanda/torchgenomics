"""Regression tests for the native C++ OrdinalGLMM threshold-NR kernel.

The native kernel
(``torchgwas._native._ordinal_threshold_native.ordinal_threshold_score_hessian``)
replaces the nested ``for jj / for kk`` Python score + Hessian
assembly inside the OrdinalGLMM PQL null fit. These tests assert:

1.  **Parity** — native and Python paths produce identical score
    vector and Hessian for the same (gamma_nr, Y) input at the FP64
    floor.
2.  **End-to-end fit_null parity** — the full PQL null fit converges
    to the same fixed point under both code paths.
3.  **Env-var fallthrough** — TORCHGWAS_DISABLE_NATIVE=1 forces
    the Python reference path.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from torchgwas._native import (
    HAS_NATIVE_ORDINAL_THRESHOLD,
    _ordinal_threshold_native,
)


def _make_gamma_Y(n: int, J: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate plausible cumulative-link gammas + Y category labels.

    gamma_nr[i, jj] = P(Y_i <= jj) under the cumulative-logit model.
    They must satisfy 0 < gamma[:, 0] < gamma[:, 1] < ... < gamma[:, J-2] < 1.
    """
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(n) * 1.0
    thresholds = np.linspace(-1.5, 1.5, J - 1)
    gamma = np.empty((n, J - 1))
    for jj in range(J - 1):
        gamma[:, jj] = 1.0 / (1.0 + np.exp(-(thresholds[jj] - eta)))
    Y = rng.integers(0, J, size=n)
    return (
        torch.tensor(gamma, dtype=torch.float64),
        torch.tensor(Y, dtype=torch.int64),
    )


def _python_score_hessian(
    gamma_nr: torch.Tensor, Y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python reference (mirrors the algorithmic spec in ordinal_glmm.py)."""
    n_thresh = gamma_nr.shape[1]
    score = torch.zeros(n_thresh, dtype=torch.float64)
    H = torch.zeros(n_thresh, n_thresh, dtype=torch.float64)
    for jj in range(n_thresh):
        D_jj = (jj >= Y).double()
        g_jj = gamma_nr[:, jj]
        score[jj] = (D_jj - g_jj).sum()
        H[jj, jj] = -(g_jj * (1.0 - g_jj)).sum()
        for kk in range(jj + 1, n_thresh):
            g_kk = gamma_nr[:, kk]
            cross = -(g_jj * (1.0 - g_kk)).sum()
            H[jj, kk] = cross
            H[kk, jj] = cross
    return score, H


@unittest.skipUnless(
    HAS_NATIVE_ORDINAL_THRESHOLD,
    "native _ordinal_threshold_native extension not built",
)
class TestOrdinalThresholdNativeParity(unittest.TestCase):

    def test_parity_J3(self):
        gamma, Y = _make_gamma_Y(n=500, J=3, seed=42)
        score_py, H_py = _python_score_hessian(gamma, Y)
        score_cpp = torch.zeros_like(score_py)
        H_cpp = torch.zeros_like(H_py)
        _ordinal_threshold_native.ordinal_threshold_score_hessian(
            gamma.contiguous().numpy(),
            Y.contiguous().numpy(),
            score_cpp.numpy(),
            H_cpp.numpy(),
        )
        torch.testing.assert_close(score_cpp, score_py, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(H_cpp, H_py, atol=1e-12, rtol=1e-12)

    def test_parity_J5(self):
        gamma, Y = _make_gamma_Y(n=2000, J=5, seed=43)
        score_py, H_py = _python_score_hessian(gamma, Y)
        score_cpp = torch.zeros_like(score_py)
        H_cpp = torch.zeros_like(H_py)
        _ordinal_threshold_native.ordinal_threshold_score_hessian(
            gamma.contiguous().numpy(),
            Y.contiguous().numpy(),
            score_cpp.numpy(),
            H_cpp.numpy(),
        )
        torch.testing.assert_close(score_cpp, score_py, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(H_cpp, H_py, atol=1e-12, rtol=1e-12)

    def test_parity_J10(self):
        gamma, Y = _make_gamma_Y(n=5000, J=10, seed=44)
        score_py, H_py = _python_score_hessian(gamma, Y)
        score_cpp = torch.zeros_like(score_py)
        H_cpp = torch.zeros_like(H_py)
        _ordinal_threshold_native.ordinal_threshold_score_hessian(
            gamma.contiguous().numpy(),
            Y.contiguous().numpy(),
            score_cpp.numpy(),
            H_cpp.numpy(),
        )
        torch.testing.assert_close(score_cpp, score_py, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(H_cpp, H_py, atol=1e-12, rtol=1e-12)

    def test_hessian_symmetric(self):
        """Output Hessian must be symmetric (by construction)."""
        gamma, Y = _make_gamma_Y(n=500, J=5, seed=45)
        score = torch.zeros(4, dtype=torch.float64)
        H = torch.zeros(4, 4, dtype=torch.float64)
        _ordinal_threshold_native.ordinal_threshold_score_hessian(
            gamma.contiguous().numpy(),
            Y.contiguous().numpy(),
            score.numpy(),
            H.numpy(),
        )
        torch.testing.assert_close(H, H.T, atol=1e-15, rtol=1e-15)


@unittest.skipUnless(
    HAS_NATIVE_ORDINAL_THRESHOLD,
    "native _ordinal_threshold_native extension not built",
)
class TestOrdinalGLMMEndToEndParity(unittest.TestCase):
    """End-to-end fit_null must match across code paths."""

    def test_fit_null_converges_under_both_paths(self):
        # Use the model directly to exercise the full PQL loop with
        # the native threshold kernel wired in.
        from torchgwas.models.ordinal_glmm import OrdinalGLMM
        rng = np.random.default_rng(46)
        n = 300
        J = 3
        K = torch.eye(n, dtype=torch.float64) + 0.1 * torch.ones((n, n), dtype=torch.float64)
        X0 = torch.from_numpy(rng.standard_normal((n, 2))).to(torch.float64)
        eta_true = X0 @ torch.tensor([0.3, -0.2], dtype=torch.float64)
        # Discretize to ordinal labels.
        cutpoints = np.quantile(eta_true.numpy(), [1.0 / J * j for j in range(1, J)])
        Y_int = np.digitize(eta_true.numpy(), cutpoints)
        Y = torch.from_numpy(Y_int).to(torch.int64)

        def fit_once(env_value):
            prev = os.environ.get("TORCHGWAS_DISABLE_NATIVE")
            if env_value is None:
                os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
            else:
                os.environ["TORCHGWAS_DISABLE_NATIVE"] = env_value
            try:
                m = OrdinalGLMM(n_categories=J)
                return m.fit_null(Y=Y, X0=X0, K=K)
            finally:
                if prev is None:
                    os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
                else:
                    os.environ["TORCHGWAS_DISABLE_NATIVE"] = prev

        nf_py = fit_once("1")
        nf_cpp = fit_once(None)
        self.assertAlmostEqual(nf_py.sig2_g, nf_cpp.sig2_g, places=10)
        self.assertAlmostEqual(nf_py.sig2_e, nf_cpp.sig2_e, places=10)


if __name__ == "__main__":
    unittest.main()
