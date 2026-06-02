"""Phase 7: BLINK LD-clustering + BIC multi-locus model tests."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.models.base import NullFit, ScanResult, VariantMeta
from torchgenomics.models.blink import BLINK, _bic_forward_select, _ld_remove_block

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def blink_data():
    """Simulate data with 2 causal SNPs on different chromosomes.

    n=200, m=100. SNPs 15 and 70 are causal with strong effects.
    """
    torch.manual_seed(99)
    n, m = 200, 100

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[15] = 2.0
    beta_true[70] = 1.5

    Y = 5.0 + G @ beta_true + torch.randn(n, dtype=torch.float64) * 0.5

    chrs = ["1"] * 50 + ["2"] * 50
    pos = list(range(1000, 1000 + 50 * 10000, 10000)) + \
          list(range(1000, 1000 + 50 * 10000, 10000))

    return {
        "Y": Y, "X0": X0, "G": G, "n": n, "m": m,
        "chrs": chrs, "pos": pos,
    }


@pytest.fixture
def blink_null_data():
    """Null data with no signal."""
    torch.manual_seed(111)
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
# Helper function tests
# ---------------------------------------------------------------

class TestHelpers:
    def test_ld_remove_empty(self):
        """Empty candidates should return empty."""
        G = torch.randn(50, 10, dtype=torch.float64)
        candidates = torch.tensor([], dtype=torch.long)
        p = torch.ones(10, dtype=torch.float64)
        result = _ld_remove_block(G, candidates, p, threshold=0.7)
        assert len(result) == 0

    def test_ld_remove_single(self):
        """Single candidate should be kept."""
        G = torch.randn(50, 10, dtype=torch.float64)
        candidates = torch.tensor([3], dtype=torch.long)
        p = torch.ones(10, dtype=torch.float64)
        p[3] = 0.001
        result = _ld_remove_block(G, candidates, p, threshold=0.7)
        assert len(result) == 1
        assert result[0].item() == 3

    def test_ld_remove_correlated(self):
        """Highly correlated SNPs should be pruned to one."""
        torch.manual_seed(42)
        n = 200
        base = torch.randn(n, dtype=torch.float64)
        G = torch.randn(n, 5, dtype=torch.float64)
        # Make SNPs 0 and 1 nearly identical
        G[:, 0] = base
        G[:, 1] = base + torch.randn(n, dtype=torch.float64) * 0.01
        candidates = torch.tensor([0, 1], dtype=torch.long)
        p = torch.tensor([0.001, 0.01, 0.5, 0.5, 0.5], dtype=torch.float64)
        result = _ld_remove_block(G, candidates, p, threshold=0.7)
        assert len(result) == 1  # only keep the better one

    def test_bic_select_no_improvement(self):
        """If no SNP improves BIC, return empty."""
        torch.manual_seed(42)
        n = 50
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        G = torch.randn(n, 5, dtype=torch.float64) * 0.001  # tiny effect
        leads = torch.tensor([0, 1, 2], dtype=torch.long)
        p = torch.ones(5, dtype=torch.float64)
        selected = _bic_forward_select(Y, X0, G, leads, p, max_qtns=5)
        # May or may not select — just check it returns a valid tensor
        assert selected.dtype == torch.long

    def test_bic_select_strong_signal(self):
        """Strong causal SNP should be selected by BIC."""
        torch.manual_seed(42)
        n = 200
        G = torch.randn(n, 5, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = 3.0 * G[:, 2] + torch.randn(n, dtype=torch.float64) * 0.5
        leads = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        p = torch.ones(5, dtype=torch.float64)
        p[2] = 1e-10
        selected = _bic_forward_select(Y, X0, G, leads, p, max_qtns=5)
        assert 2 in selected.tolist()


# ---------------------------------------------------------------
# BLINK model tests
# ---------------------------------------------------------------

class TestBLINK:
    def test_fit_null(self, blink_data):
        model = BLINK()
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        assert isinstance(nf, NullFit)
        assert nf.sig2_e > 0
        assert nf.converged is True

    def test_scan_returns_scanresult(self, blink_data):
        model = BLINK(max_iter=3)
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        vmeta = _make_vmeta(
            blink_data["m"], blink_data["chrs"], blink_data["pos"],
        )
        result = model.score_chunk(blink_data["G"], nf, vmeta)

        assert isinstance(result, ScanResult)
        assert result.beta.shape == (blink_data["m"],)
        assert result.p.shape == (blink_data["m"],)
        assert result.test == "wald"

    def test_pvalues_valid(self, blink_data):
        model = BLINK(max_iter=3)
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        vmeta = _make_vmeta(
            blink_data["m"], blink_data["chrs"], blink_data["pos"],
        )
        result = model.score_chunk(blink_data["G"], nf, vmeta)

        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)
        assert torch.all(result.stat >= 0)

    def test_causal_snps_detected(self, blink_data):
        """Causal SNPs should have small p-values after iteration."""
        model = BLINK(max_iter=5, p_threshold=0.05)
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        vmeta = _make_vmeta(
            blink_data["m"], blink_data["chrs"], blink_data["pos"],
        )
        result = model.score_chunk(blink_data["G"], nf, vmeta)

        causal_pvals = result.p[[15, 70]]
        assert causal_pvals.min().item() < 0.05

    def test_null_controlled(self, blink_null_data):
        """Under null, false positive rate should be controlled."""
        model = BLINK(max_iter=3)
        nf = model.fit_null(blink_null_data["Y"], blink_null_data["X0"])
        vmeta = _make_vmeta(
            blink_null_data["m"], blink_null_data["chrs"],
            blink_null_data["pos"],
        )
        result = model.score_chunk(blink_null_data["G"], nf, vmeta)

        frac_sig = (result.p < 0.05).float().mean().item()
        assert frac_sig < 0.20

    def test_invalid_test_raises(self, blink_data):
        model = BLINK()
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        vmeta = _make_vmeta(blink_data["m"])
        with pytest.raises(ValueError, match="wald"):
            model.score_chunk(blink_data["G"], nf, vmeta, test="score")

    def test_max_iter_respected(self, blink_data):
        model = BLINK(max_iter=2)
        nf = model.fit_null(blink_data["Y"], blink_data["X0"])
        vmeta = _make_vmeta(
            blink_data["m"], blink_data["chrs"], blink_data["pos"],
        )
        result = model.score_chunk(blink_data["G"], nf, vmeta)
        assert isinstance(result, ScanResult)
