"""Phase 7: FarmCPU iterative multi-locus model tests."""

from __future__ import annotations

import pytest
import torch

from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.models.farmcpu import FarmCPU
from torchgwas.models.iterative import IterativeGWASLoop

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def farmcpu_data():
    """Simulate data with 2 causal SNPs for multi-locus detection.

    n=200, m=100. SNPs 10 and 60 are causal with strong effects.
    Positions spread across two chromosomes.
    """
    torch.manual_seed(77)
    n, m = 200, 100

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[10] = 2.0
    beta_true[60] = 1.5

    Y = 5.0 + G @ beta_true + torch.randn(n, dtype=torch.float64) * 0.5

    chrs = ["1"] * 50 + ["2"] * 50
    pos = list(range(1000, 1000 + 50 * 10000, 10000)) + \
          list(range(1000, 1000 + 50 * 10000, 10000))

    return {
        "Y": Y, "X0": X0, "G": G, "n": n, "m": m,
        "chrs": chrs, "pos": pos,
    }


@pytest.fixture
def farmcpu_null_data():
    """Null data with no signal."""
    torch.manual_seed(88)
    n, m = 200, 80

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    Y = torch.randn(n, dtype=torch.float64) * 2.0

    chrs = ["1"] * m
    pos = list(range(1000, 1000 + m * 5000, 5000))

    return {"Y": Y, "X0": X0, "G": G, "n": n, "m": m, "chrs": chrs, "pos": pos}


def _make_vmeta(m: int, chrs=None, pos=None) -> VariantMeta:
    return VariantMeta(
        snp=[f"snp_{i}" for i in range(m)],
        chr=chrs or (["1"] * m),
        pos=pos or list(range(1, m + 1)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


# ---------------------------------------------------------------
# IterativeGWASLoop base class tests
# ---------------------------------------------------------------

class TestIterativeBase:
    def test_select_pseudo_qtns_empty(self):
        loop = IterativeGWASLoop()
        p = torch.ones(50, dtype=torch.float64)  # all non-significant
        qtns = loop.select_pseudo_qtns(p, threshold=0.01)
        assert len(qtns) == 0

    def test_select_pseudo_qtns_top_k(self):
        loop = IterativeGWASLoop()
        p = torch.ones(50, dtype=torch.float64)
        p[5] = 0.001
        p[10] = 0.005
        p[30] = 0.008
        qtns = loop.select_pseudo_qtns(p, threshold=0.01, max_qtns=2)
        assert len(qtns) == 2
        assert 5 in qtns.tolist()
        assert 10 in qtns.tolist()

    def test_convergence_same_set(self):
        loop = IterativeGWASLoop()
        a = torch.tensor([3, 7, 12])
        b = torch.tensor([12, 3, 7])  # same set, different order
        assert loop.check_convergence(a, b) is True

    def test_convergence_different_set(self):
        loop = IterativeGWASLoop()
        a = torch.tensor([3, 7])
        b = torch.tensor([3, 8])
        assert loop.check_convergence(a, b) is False

    def test_convergence_empty(self):
        loop = IterativeGWASLoop()
        a = torch.tensor([], dtype=torch.long)
        b = torch.tensor([], dtype=torch.long)
        assert loop.check_convergence(a, b) is True


# ---------------------------------------------------------------
# FarmCPU model tests
# ---------------------------------------------------------------

class TestFarmCPU:
    def test_fit_null(self, farmcpu_data):
        model = FarmCPU()
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        assert isinstance(nf, NullFit)
        assert nf.sig2_e > 0
        assert nf.converged is True

    def test_scan_returns_scanresult(self, farmcpu_data):
        model = FarmCPU(max_iter=3)
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        vmeta = _make_vmeta(
            farmcpu_data["m"], farmcpu_data["chrs"], farmcpu_data["pos"],
        )
        result = model.score_chunk(farmcpu_data["G"], nf, vmeta)

        assert isinstance(result, ScanResult)
        assert result.beta.shape == (farmcpu_data["m"],)
        assert result.p.shape == (farmcpu_data["m"],)
        assert result.test == "wald"

    def test_pvalues_valid(self, farmcpu_data):
        model = FarmCPU(max_iter=3)
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        vmeta = _make_vmeta(
            farmcpu_data["m"], farmcpu_data["chrs"], farmcpu_data["pos"],
        )
        result = model.score_chunk(farmcpu_data["G"], nf, vmeta)

        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)
        assert torch.all(result.stat >= 0)

    def test_causal_snps_detected(self, farmcpu_data):
        """Causal SNPs should have small p-values after iteration."""
        model = FarmCPU(max_iter=5, p_threshold=0.05)
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        vmeta = _make_vmeta(
            farmcpu_data["m"], farmcpu_data["chrs"], farmcpu_data["pos"],
        )
        result = model.score_chunk(farmcpu_data["G"], nf, vmeta)

        # At least one causal SNP should be significant
        causal_pvals = result.p[[10, 60]]
        assert causal_pvals.min().item() < 0.05

    def test_null_no_false_positives(self, farmcpu_null_data):
        """Under null, most p-values should be non-significant."""
        model = FarmCPU(max_iter=3)
        nf = model.fit_null(farmcpu_null_data["Y"], farmcpu_null_data["X0"])
        vmeta = _make_vmeta(
            farmcpu_null_data["m"], farmcpu_null_data["chrs"],
            farmcpu_null_data["pos"],
        )
        result = model.score_chunk(farmcpu_null_data["G"], nf, vmeta)

        # Under null, fewer than 20% should be < 0.05 (allowing some inflation)
        frac_sig = (result.p < 0.05).float().mean().item()
        assert frac_sig < 0.20

    def test_invalid_test_raises(self, farmcpu_data):
        model = FarmCPU()
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        vmeta = _make_vmeta(farmcpu_data["m"])
        with pytest.raises(ValueError, match="wald"):
            model.score_chunk(farmcpu_data["G"], nf, vmeta, test="lrt")

    def test_max_iter_respected(self, farmcpu_data):
        """FarmCPU should not exceed max_iter iterations."""
        model = FarmCPU(max_iter=2)
        nf = model.fit_null(farmcpu_data["Y"], farmcpu_data["X0"])
        vmeta = _make_vmeta(
            farmcpu_data["m"], farmcpu_data["chrs"], farmcpu_data["pos"],
        )
        # Should complete without error (2 iterations max)
        result = model.score_chunk(farmcpu_data["G"], nf, vmeta)
        assert isinstance(result, ScanResult)
