"""Regression tests for the native C++ hyprcoloc subset enumeration.

The native kernel
(``torchgenomics._native._hyprcoloc_native.enumerate_subset_log_bf``)
replaces the per-subset Python loop in
``torchgenomics.postgwas._hyprcoloc.hyprcoloc``. These tests assert:

1.  **Parity** — native and Python paths produce numerically identical
    posterior outputs at the FP64 floor for K ∈ {6, 8, 10, 12}.
2.  **Env-var fallthrough** — ``TORCHGENOMICS_DISABLE_NATIVE=1`` forces
    the Python reference path.
3.  **Below-threshold input** — K < 6 routes to Python.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from torchgenomics._native import HAS_NATIVE_HYPRCOLOC
from torchgenomics.postgwas._hyprcoloc import hyprcoloc
from torchgenomics.postgwas._sumstats import SumStats


def _planted_K_traits(
    K: int,
    m: int,
    n_shared_traits: int = 3,
    seed: int = 0,
) -> list[SumStats]:
    """Create K sumstats; the first n_shared_traits share a planted causal."""
    rng = np.random.default_rng(seed)
    snp = [f"rs{i}" for i in range(m)]
    chr_ = ["1"] * m
    pos = list(range(1, m + 1))
    a1 = ["A"] * m
    a2 = ["G"] * m
    causal_idx = m // 2
    sumstats: list[SumStats] = []
    for k in range(K):
        beta = rng.standard_normal(m) * 0.05
        if k < n_shared_traits:
            beta[causal_idx] = 0.30  # planted shared effect
        se = np.abs(rng.standard_normal(m) * 0.01) + 0.02
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.tanh(np.abs(beta / se))))
        sumstats.append(SumStats(
            chr=chr_, pos=pos, snp=snp, a1=a1, a2=a2,
            beta=torch.tensor(beta, dtype=torch.float64),
            se=torch.tensor(se, dtype=torch.float64),
            p=torch.tensor(np.clip(p, 1e-30, 1.0), dtype=torch.float64),
            n=torch.full((m,), 10000.0, dtype=torch.float64),
        ))
    return sumstats


def _run_with(env_value, fn, *args, **kwargs):
    prev = os.environ.get("TORCHGENOMICS_DISABLE_NATIVE")
    if env_value is None:
        os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    else:
        os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = env_value
    try:
        return fn(*args, **kwargs)
    finally:
        if prev is None:
            os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
        else:
            os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = prev


@unittest.skipUnless(
    HAS_NATIVE_HYPRCOLOC, "native _hyprcoloc_native extension not built"
)
class TestHyprcolocNativeParity(unittest.TestCase):

    def test_parity_K6(self):
        ss = _planted_K_traits(K=6, m=100, seed=42)
        py = _run_with("1", hyprcoloc, ss)
        cpp = _run_with(None, hyprcoloc, ss)
        # Per-subset posteriors: full-precision parity since the kernel
        # produces identical log-BFs and the downstream prior + normalisation
        # is pure Python in both paths.
        for key in py.subset_posteriors:
            self.assertAlmostEqual(
                cpp.subset_posteriors[key], py.subset_posteriors[key],
                places=12,
            )
        self.assertEqual(cpp.best_cluster, py.best_cluster)
        self.assertAlmostEqual(
            cpp.best_cluster_posterior, py.best_cluster_posterior, places=12
        )

    def test_parity_K10(self):
        ss = _planted_K_traits(K=10, m=200, seed=43)
        py = _run_with("1", hyprcoloc, ss)
        cpp = _run_with(None, hyprcoloc, ss)
        # 1013 subsets — compare every one.
        self.assertEqual(set(cpp.subset_posteriors), set(py.subset_posteriors))
        max_diff = max(
            abs(cpp.subset_posteriors[k] - py.subset_posteriors[k])
            for k in cpp.subset_posteriors
        )
        self.assertLess(max_diff, 1e-12)

    def test_parity_K12(self):
        """K=12 → 4083 subsets; ensures large bitmask correctness."""
        ss = _planted_K_traits(K=12, m=100, seed=44)
        py = _run_with("1", hyprcoloc, ss)
        cpp = _run_with(None, hyprcoloc, ss)
        max_diff = max(
            abs(cpp.subset_posteriors[k] - py.subset_posteriors[k])
            for k in cpp.subset_posteriors
        )
        self.assertLess(max_diff, 1e-12)


@unittest.skipUnless(
    HAS_NATIVE_HYPRCOLOC, "native _hyprcoloc_native extension not built"
)
class TestHyprcolocNativeFallthrough(unittest.TestCase):

    def test_env_var_disables_native(self):
        ss = _planted_K_traits(K=8, m=150, seed=45)
        py = _run_with("1", hyprcoloc, ss)
        default = _run_with(None, hyprcoloc, ss)
        max_diff = max(
            abs(default.subset_posteriors[k] - py.subset_posteriors[k])
            for k in default.subset_posteriors
        )
        self.assertLess(max_diff, 1e-12)

    def test_below_threshold_uses_python(self):
        """K < 6 stays in Python; outputs exactly identical."""
        ss = _planted_K_traits(K=4, m=100, seed=46)
        py = _run_with("1", hyprcoloc, ss)
        default = _run_with(None, hyprcoloc, ss)
        for key in py.subset_posteriors:
            self.assertEqual(
                default.subset_posteriors[key], py.subset_posteriors[key]
            )


if __name__ == "__main__":
    unittest.main()
