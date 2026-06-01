"""Regression tests for the native C++ MAGMA snp_to_gene window scan.

The native kernel
(``torchgwas._native._snp_to_gene_native.snp_to_gene_scan``) replaces
the per-gene Python loop in
``torchgwas.postgwas._enrichment.snp_to_gene``. These tests assert:

1.  **Parity** — native and Python paths produce identical gene-level
    statistics, p-values, and n_snps counts at the FP64 floor.
2.  **Env-var fallthrough** — ``TORCHGWAS_DISABLE_NATIVE=1`` forces
    the Python reference path.
3.  **Below-threshold input** — for fewer than 64 genes the dispatcher
    routes to Python.
4.  **Edge cases** — empty SNP set, genes without matching chromosome,
    window kb extension, multiple chromosomes.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch

from torchgwas._native import HAS_NATIVE_SNP_TO_GENE
from torchgwas.postgwas._enrichment import snp_to_gene
from torchgwas.postgwas._sumstats import SumStats


def _make_sumstats(
    m: int = 1000,
    n_chrs: int = 5,
    seed: int = 0,
) -> SumStats:
    rng = np.random.default_rng(seed)
    chrs = rng.integers(1, n_chrs + 1, size=m).astype(str).tolist()
    pos = rng.integers(1, 100_000_000, size=m).tolist()
    snp_ids = [f"rs{i}" for i in range(m)]
    # Random p-values; chi² inferred via chi² = -2 ln p (placeholder formula
    # is fine — SumStats computes chi² as (β/se)² internally if those are
    # provided, but the snp_to_gene function only reads ss.chi2).
    p = rng.uniform(1e-5, 1.0, size=m).tolist()
    beta = rng.standard_normal(m).tolist()
    se = np.abs(rng.standard_normal(m) + 0.1).tolist()
    # SumStats expects torch tensors / lists.
    return SumStats(
        chr=chrs,
        pos=pos,
        snp=snp_ids,
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.tensor(beta, dtype=torch.float64),
        se=torch.tensor(se, dtype=torch.float64),
        p=torch.tensor(p, dtype=torch.float64),
        n=torch.full((m,), 10_000.0, dtype=torch.float64),
    )


def _make_genes(
    n_genes: int = 200,
    n_chrs: int = 5,
    seed: int = 1,
) -> tuple[list[str], list[str], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    gene_id = [f"g{i}" for i in range(n_genes)]
    gene_chr = rng.integers(1, n_chrs + 1, size=n_genes).astype(str).tolist()
    centers = rng.integers(1_000_000, 99_000_000, size=n_genes)
    widths = rng.integers(5_000, 100_000, size=n_genes)
    gene_start = (centers - widths // 2).tolist()
    gene_end = (centers + widths // 2).tolist()
    return gene_id, gene_chr, gene_start, gene_end


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
    HAS_NATIVE_SNP_TO_GENE, "native _snp_to_gene_native extension not built"
)
class TestSnpToGeneNativeParity(unittest.TestCase):

    def test_parity_basic(self):
        ss = _make_sumstats(m=2000, n_chrs=5, seed=42)
        gid, gchr, gstart, gend = _make_genes(n_genes=200, n_chrs=5, seed=43)
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend)
        cpp = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend)
        self.assertEqual(cpp.n_snps, py.n_snps)
        # Floor at 1e-12 abs: arithmetic is bit-equal except for OpenMP
        # reduction order on sum_chi2.
        torch.testing.assert_close(cpp.stat, py.stat, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(cpp.p, py.p, atol=1e-12, rtol=1e-12)

    def test_parity_with_window(self):
        """Non-zero window_kb extension should preserve parity."""
        ss = _make_sumstats(m=2000, n_chrs=5, seed=44)
        gid, gchr, gstart, gend = _make_genes(n_genes=200, n_chrs=5, seed=45)
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend,
                       window_kb=50.0)
        cpp = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend,
                        window_kb=50.0)
        self.assertEqual(cpp.n_snps, py.n_snps)
        torch.testing.assert_close(cpp.stat, py.stat, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(cpp.p, py.p, atol=1e-12, rtol=1e-12)

    def test_parity_large(self):
        """Realistic-size scan: m=10K SNPs × 20K genes."""
        ss = _make_sumstats(m=10_000, n_chrs=22, seed=46)
        gid, gchr, gstart, gend = _make_genes(
            n_genes=20_000, n_chrs=22, seed=47
        )
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend,
                       window_kb=10.0)
        cpp = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend,
                        window_kb=10.0)
        self.assertEqual(cpp.n_snps, py.n_snps)
        torch.testing.assert_close(cpp.stat, py.stat, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(cpp.p, py.p, atol=1e-12, rtol=1e-12)

    def test_genes_with_no_matching_chromosome(self):
        """Genes on a chromosome with no SNPs return stat=0, p=1, n=0."""
        ss = _make_sumstats(m=200, n_chrs=2, seed=48)  # chrs "1", "2"
        gid = [f"g{i}" for i in range(70)]
        gchr = ["99"] * 70  # no SNPs on chromosome 99
        gstart = [1_000_000] * 70
        gend = [2_000_000] * 70
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend)
        cpp = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend)
        self.assertEqual(cpp.n_snps, py.n_snps)
        self.assertTrue(all(n == 0 for n in cpp.n_snps))
        self.assertTrue(torch.all(cpp.p == 1.0))
        self.assertTrue(torch.all(cpp.stat == 0.0))


@unittest.skipUnless(
    HAS_NATIVE_SNP_TO_GENE, "native _snp_to_gene_native extension not built"
)
class TestSnpToGeneNativeFallthrough(unittest.TestCase):

    def test_env_var_disables_native(self):
        ss = _make_sumstats(m=500, n_chrs=3, seed=49)
        gid, gchr, gstart, gend = _make_genes(n_genes=200, n_chrs=3, seed=50)
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend)
        default = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend)
        torch.testing.assert_close(default.stat, py.stat, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(default.p, py.p, atol=1e-12, rtol=1e-12)

    def test_below_threshold_uses_python(self):
        """n_genes < 64 routes to Python; identical results."""
        ss = _make_sumstats(m=500, n_chrs=3, seed=51)
        gid, gchr, gstart, gend = _make_genes(n_genes=32, n_chrs=3, seed=52)
        py = _run_with("1", snp_to_gene, ss, gid, gchr, gstart, gend)
        default = _run_with(None, snp_to_gene, ss, gid, gchr, gstart, gend)
        # Exact FP equality because both paths take the Python branch.
        torch.testing.assert_close(default.stat, py.stat, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close(default.p, py.p, atol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    unittest.main()
