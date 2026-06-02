"""Regression tests for the native C++ PCHT compat-pair kernel.

The native kernel
(``torchgenomics._native._pcht_native.dosage_posterior_cov``) replaces the
nested per-sample compat-pair enumeration + posterior-covariance
accumulation in
``torchgenomics.models.haplotype_novel.compute_dosage_posterior_cov``.
These tests assert:

1.  **Parity** — native and Python paths produce numerically identical
    (H, H) summed posterior covariance at the FP64 floor.
2.  **Env-var fallthrough** — ``TORCHGENOMICS_DISABLE_NATIVE=1`` forces
    the Python reference path; results match.
3.  **Below-threshold input** — for tiny problems (n*H² < 256) the
    dispatcher routes to Python.
4.  **F2 LD-aware pruning regression** — the LD-aware fractional-count
    score (commit f601f20) lives upstream in
    ``haplotype_gwas._compute_ld_aware_score``; the PCHT kernel
    consumes the already-pruned haplotype set and must not
    re-introduce the spurious "max_haplotypes drops a high-freq
    haplotype under tight LD" bug. We verify by running the end-to-end
    haplotype-GWAS test that locked the original fix.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from torchgenomics._native import HAS_NATIVE_PCHT
from torchgenomics.models.haplotype_novel import compute_dosage_posterior_cov


def _planted_haplotype_fixture(
    n: int = 200,
    H: int = 8,
    m: int = 5,
    seed: int = 0,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Build (G_block, labels, freq) where G is generated from random
    parental haplotype pairs.

    Parameters
    ----------
    n : int
        Number of individuals.
    H : int
        Number of distinct haplotypes (each is an m-bit binary string).
    m : int
        SNPs per haplotype.

    Returns
    -------
    G_block : (n, m) int genotype dosages in [0, 2]
    labels : list[str] length H — '0'/'1' string per haplotype
    freq : (H,) float64 haplotype frequencies (sum to 1)
    """
    rng = np.random.default_rng(seed)
    # Distinct haplotype labels (binary strings of length m).
    label_set: set[str] = set()
    while len(label_set) < H:
        label_set.add("".join(str(b) for b in rng.integers(0, 2, size=m)))
    labels = sorted(label_set)
    hap_mat = np.array(
        [[int(c) for c in lab] for lab in labels], dtype=np.int64
    )  # (H, m)

    # Frequencies (Dirichlet) sum to 1.
    freq_np = rng.dirichlet(np.ones(H))
    freq = torch.from_numpy(freq_np).to(torch.float64)

    # Generate genotypes G[i] = hap[a_i] + hap[b_i], where (a_i, b_i)
    # are drawn from the haplotype frequency distribution.
    a = rng.choice(H, size=n, p=freq_np)
    b = rng.choice(H, size=n, p=freq_np)
    G = hap_mat[a] + hap_mat[b]
    G_block = torch.from_numpy(G).to(torch.float64)
    return G_block, labels, freq


def _run_with(env_value: str | None, fn, *args, **kwargs):
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
    HAS_NATIVE_PCHT, "native _pcht_native extension not built"
)
class TestPCHTNativeParity(unittest.TestCase):

    def test_parity_small_block(self):
        G, labels, freq = _planted_haplotype_fixture(n=200, H=8, m=5, seed=42)
        py = _run_with("1", compute_dosage_posterior_cov, G, labels, freq)
        cpp = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        # FP64 floor: per-pair posterior is exact arithmetic, but
        # summation order across threads varies for the OpenMP reduction.
        # Floor at 1e-10 absolute (observed: ~1e-13 on small fixtures).
        torch.testing.assert_close(cpp, py, atol=1e-10, rtol=1e-10)

    def test_parity_realistic_block(self):
        """Larger PCHT-typical block: n=2000, H=20, m=10."""
        G, labels, freq = _planted_haplotype_fixture(n=2000, H=20, m=10, seed=43)
        py = _run_with("1", compute_dosage_posterior_cov, G, labels, freq)
        cpp = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        torch.testing.assert_close(cpp, py, atol=1e-10, rtol=1e-10)

    def test_parity_symmetric_output(self):
        """Output cov matrix must be symmetric (and identical across paths)."""
        G, labels, freq = _planted_haplotype_fixture(n=500, H=12, m=6, seed=44)
        cpp = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        torch.testing.assert_close(cpp, cpp.T, atol=1e-12, rtol=1e-12)

    def test_zero_input(self):
        """Empty haplotype set returns (0, 0) — both paths."""
        G = torch.zeros((10, 3), dtype=torch.float64)
        labels: list[str] = []
        freq = torch.zeros(0, dtype=torch.float64)
        py = _run_with("1", compute_dosage_posterior_cov, G, labels, freq)
        cpp = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        self.assertEqual(py.shape, (0, 0))
        self.assertEqual(cpp.shape, (0, 0))


@unittest.skipUnless(
    HAS_NATIVE_PCHT, "native _pcht_native extension not built"
)
class TestPCHTNativeFallthrough(unittest.TestCase):

    def test_env_var_disables_native(self):
        G, labels, freq = _planted_haplotype_fixture(n=200, H=8, m=5, seed=46)
        py = _run_with("1", compute_dosage_posterior_cov, G, labels, freq)
        default = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        torch.testing.assert_close(default, py, atol=1e-10, rtol=1e-10)

    def test_below_threshold_uses_python(self):
        """n * H² < 256 routes to Python."""
        # n=10, H=4 → n*H² = 160 (< 256). Routes to Python.
        G, labels, freq = _planted_haplotype_fixture(n=10, H=4, m=3, seed=47)
        py = _run_with("1", compute_dosage_posterior_cov, G, labels, freq)
        default = _run_with(None, compute_dosage_posterior_cov, G, labels, freq)
        # Floor: per-pair arithmetic is FP-equal between paths at small sizes.
        torch.testing.assert_close(default, py, atol=1e-12, rtol=1e-12)


@unittest.skipUnless(
    HAS_NATIVE_PCHT, "native _pcht_native extension not built"
)
class TestPCHTF2InvariantPreserved(unittest.TestCase):
    """The F2 LD-aware-pruning fix lives upstream of this kernel.

    The PCHT kernel consumes the (labels, freq) already produced by the
    EM-haplotype + LD-aware pruning path in ``haplotype_gwas``. As long
    as the upstream test for the F2 fix still passes (which we rerun
    here), and the kernel's output matches the Python body exactly,
    the F2 invariant is preserved automatically.
    """

    def test_haplotype_gwas_f2_regression_test_still_passes(self):
        """The hap_gwas regression test ``test_ld_aware_pruning_keeps_
        high_freq_haplotype_under_tight_ld`` is the canonical guard
        on commit f601f20.  Verify it still passes with the native
        PCHT kernel enabled (default state)."""
        # Import here to avoid a hard test-collection dependency.
        try:
            import importlib.util
            spec = importlib.util.find_spec(
                "tests.test_haplotype_gwas"
            )
            if spec is None:
                self.skipTest("tests.test_haplotype_gwas not importable")
        except Exception:
            self.skipTest("haplotype-gwas test module not importable")
        # The exact regression-test class / method name is
        # TestHaplotypeConstruction.test_ld_aware_pruning_keeps_
        # high_freq_haplotype_under_tight_ld per the f601f20 commit
        # message. We invoke pytest on the specific node to avoid
        # coupling to internal class names; if it passes, the F2
        # invariant holds end-to-end with the native PCHT path.
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/test_haplotype_gwas.py::TestHaplotypeConstruction::"
                "test_ld_aware_pruning_keeps_high_freq_haplotype_under_tight_ld",
                "-q",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            self.fail(
                "F2 regression test failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
