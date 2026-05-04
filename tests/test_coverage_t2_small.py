"""Tier-2 behavioral coverage tests for small-batch public symbols across
``torchgwas.ld``, ``torchgwas.multiomics``, and ``torchgwas.pgs``.

Bar (Pillar A spec section 4.3 Tier 2):
- Dataclasses: construct + round-trip + repr smoke.
- ``HAS_NATIVE_*`` constants: type=bool + cross-check against the
  corresponding ``torchgwas._native`` extension import.

Covers 10 public symbols:

- ``torchgwas.ld._plink_compat.PLINKBlock`` (dataclass)
- ``torchgwas.ld._blocks.PairwiseLD`` (dataclass)
- ``torchgwas.multiomics._types.GeneSetMediationResult`` (dataclass)
- ``torchgwas.pgs.validation.PGSValidation`` (dataclass)
- ``torchgwas.pgs.scoring.ScoringResult`` (dataclass)
- ``torchgwas.pgs.ct.HAS_NATIVE_CT`` (bool constant)
- ``torchgwas.pgs.diagnostics.HAS_NATIVE_ESS`` (bool constant)
- ``torchgwas.pgs.ldpred2.HAS_NATIVE_LDPRED2`` (bool constant)
- ``torchgwas.pgs.prscs.HAS_NATIVE_PRSCS`` (bool constant)
- ``torchgwas.pgs.validation.pi`` (stray ``math.pi`` re-export from
  ``from math import exp, log, pi, sqrt`` — smoke-tested; cleanup queued
  for T12)
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.ld import PairwiseLD, PLINKBlock
from torchgwas.multiomics import GeneSetMediationResult
from torchgwas.pgs import validation as pgs_validation
from torchgwas.pgs.ct import HAS_NATIVE_CT
from torchgwas.pgs.diagnostics import HAS_NATIVE_ESS
from torchgwas.pgs.ldpred2 import HAS_NATIVE_LDPRED2
from torchgwas.pgs.prscs import HAS_NATIVE_PRSCS
from torchgwas.pgs.scoring import ScoringResult
from torchgwas.pgs.validation import PGSValidation


pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# torchgwas.ld._plink_compat.PLINKBlock
# ---------------------------------------------------------------------------


class TestPlinkBlock:
    """``PLINKBlock`` is a dataclass mirroring one row of a PLINK 1.9
    ``.blocks.det`` file (chr / bp1 / bp2 / kb / n_snps / snps)."""

    def test_construct(self):
        """Build with documented fields and verify each attribute round-trips."""
        block = PLINKBlock(
            chr="1",
            bp1=1000,
            bp2=5000,
            kb=4.0,
            n_snps=3,
            snps=["rs1", "rs2", "rs3"],
        )
        assert block.chr == "1"
        assert block.bp1 == 1000
        assert block.bp2 == 5000
        assert block.kb == pytest.approx(4.0)
        assert block.n_snps == 3
        assert block.snps == ["rs1", "rs2", "rs3"]

    def test_round_trip_via_dataclass_fields(self):
        """Dataclass fields are addressable by name and survive copy."""
        from dataclasses import asdict, fields

        block = PLINKBlock(
            chr="X",
            bp1=10,
            bp2=20,
            kb=0.01,
            n_snps=2,
            snps=["rsA", "rsB"],
        )
        # asdict round-trip: re-instantiate from the dict
        d = asdict(block)
        block2 = PLINKBlock(**d)
        assert block == block2
        # All declared fields are present
        names = {f.name for f in fields(block)}
        assert names == {"chr", "bp1", "bp2", "kb", "n_snps", "snps"}

    def test_repr_contains_class_name(self):
        """``repr`` includes the class name and field values (default
        dataclass repr)."""
        block = PLINKBlock(chr="2", bp1=1, bp2=2, kb=0.001, n_snps=1, snps=["x"])
        s = repr(block)
        assert "PLINKBlock" in s
        assert "chr='2'" in s
        assert "n_snps=1" in s


# ---------------------------------------------------------------------------
# torchgwas.ld._blocks.PairwiseLD
# ---------------------------------------------------------------------------


class TestPairwiseLD:
    """``PairwiseLD`` bundles per-pair LD statistics: idx_i, idx_j, r2,
    dprime, dprime_ci_low, dprime_ci_high. All tensors of shape (P,)."""

    def _make_pld(self, P: int = 4) -> PairwiseLD:
        idx_i = torch.arange(P, dtype=torch.long)
        idx_j = torch.arange(P, dtype=torch.long) + 1
        r2 = torch.full((P,), 0.5, dtype=torch.float64)
        dprime = torch.full((P,), 0.7, dtype=torch.float64)
        ci_low = torch.full((P,), 0.6, dtype=torch.float64)
        ci_high = torch.full((P,), 0.9, dtype=torch.float64)
        return PairwiseLD(
            idx_i=idx_i,
            idx_j=idx_j,
            r2=r2,
            dprime=dprime,
            dprime_ci_low=ci_low,
            dprime_ci_high=ci_high,
        )

    def test_construct(self):
        """Construct with documented tensors and verify attribute access."""
        pld = self._make_pld(P=5)
        assert pld.idx_i.shape == (5,)
        assert pld.idx_j.shape == (5,)
        assert pld.r2.shape == (5,)
        assert pld.dprime.shape == (5,)
        assert pld.dprime_ci_low.shape == (5,)
        assert pld.dprime_ci_high.shape == (5,)
        assert torch.all(pld.r2 == 0.5)
        assert torch.all(pld.dprime == 0.7)
        assert torch.all(pld.dprime_ci_low == 0.6)
        assert torch.all(pld.dprime_ci_high == 0.9)

    def test_round_trip_via_dataclass_fields(self):
        """Dataclass fields are addressable; reconstruction yields equal tensors."""
        from dataclasses import fields

        pld = self._make_pld(P=3)
        names = {f.name for f in fields(pld)}
        assert names == {
            "idx_i",
            "idx_j",
            "r2",
            "dprime",
            "dprime_ci_low",
            "dprime_ci_high",
        }
        # Reconstruct from the same tensor handles
        pld2 = PairwiseLD(
            idx_i=pld.idx_i,
            idx_j=pld.idx_j,
            r2=pld.r2,
            dprime=pld.dprime,
            dprime_ci_low=pld.dprime_ci_low,
            dprime_ci_high=pld.dprime_ci_high,
        )
        assert torch.equal(pld.idx_i, pld2.idx_i)
        assert torch.equal(pld.r2, pld2.r2)
        assert torch.equal(pld.dprime_ci_high, pld2.dprime_ci_high)

    def test_repr_contains_class_name(self):
        """``repr`` includes the class name (default dataclass repr)."""
        pld = self._make_pld(P=2)
        s = repr(pld)
        assert "PairwiseLD" in s


# ---------------------------------------------------------------------------
# torchgwas.multiomics._types.GeneSetMediationResult
# ---------------------------------------------------------------------------


class TestGeneSetMediationResult:
    """``GeneSetMediationResult`` is a dataclass holding the joint-Wald
    output of multi-mediator mediation. Includes a ``to_dict`` method
    that converts tensors to nested lists."""

    def _make_result(self, k: int = 3) -> GeneSetMediationResult:
        a = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)[:k]
        a_cov = torch.eye(k, dtype=torch.float64) * 0.01
        b = torch.tensor([0.5, -0.4, 0.2], dtype=torch.float64)[:k]
        b_cov = torch.eye(k, dtype=torch.float64) * 0.02
        indirect = a * b
        indirect_cov = torch.eye(k, dtype=torch.float64) * 0.005
        return GeneSetMediationResult(
            a=a,
            a_cov=a_cov,
            b=b,
            b_cov=b_cov,
            indirect=indirect,
            indirect_cov=indirect_cov,
            chi2=12.34,
            df=k,
            pvalue=0.006,
            sum_indirect=float(indirect.sum().item()),
            sum_indirect_se=0.05,
            sum_indirect_pvalue=0.04,
            n=200,
            k=k,
            rank_deficient=False,
        )

    def test_construct(self):
        """Build with documented fields and round-trip every attribute."""
        res = self._make_result(k=3)
        assert res.a.shape == (3,)
        assert res.a_cov.shape == (3, 3)
        assert res.b.shape == (3,)
        assert res.b_cov.shape == (3, 3)
        assert res.indirect.shape == (3,)
        assert res.indirect_cov.shape == (3, 3)
        assert res.chi2 == pytest.approx(12.34)
        assert res.df == 3
        assert res.pvalue == pytest.approx(0.006)
        assert res.n == 200
        assert res.k == 3
        assert res.rank_deficient is False

    def test_round_trip_via_to_dict(self):
        """``to_dict`` exposes every dataclass field; tensors become lists."""
        res = self._make_result(k=2)
        d = res.to_dict()
        # All declared fields show up in the dict
        for fname in res.__dataclass_fields__:
            assert fname in d
        # Tensor fields are lists; scalars round-trip as floats / ints / bools
        assert isinstance(d["a"], list)
        assert isinstance(d["a_cov"], list)
        assert isinstance(d["chi2"], float)
        assert isinstance(d["df"], int)
        assert isinstance(d["rank_deficient"], bool)
        # Element-wise consistency for the (k,) vector fields
        assert len(d["a"]) == 2
        assert len(d["b"]) == 2

    def test_repr_contains_class_name(self):
        """``repr`` includes the class name (default dataclass repr)."""
        res = self._make_result(k=2)
        s = repr(res)
        assert "GeneSetMediationResult" in s


# ---------------------------------------------------------------------------
# torchgwas.pgs.validation.PGSValidation
# ---------------------------------------------------------------------------


class TestPgsValidation:
    """``PGSValidation`` is a dataclass with continuous-trait metrics
    (R², incremental R², Pearson, Spearman, MAE) and optional
    binary-trait metrics (AUC, Nagelkerke R², liability R²). Has a
    custom ``__repr__`` that formats values in a multi-line block."""

    def test_construct_continuous(self):
        """Build with continuous-trait fields; binary fields default to None."""
        v = PGSValidation(
            r2=0.25,
            r2_incremental=0.05,
            pearson_r=0.5,
            spearman_r=0.48,
            mae=0.3,
            n=1000,
            trait_type="continuous",
        )
        assert v.r2 == pytest.approx(0.25)
        assert v.r2_incremental == pytest.approx(0.05)
        assert v.pearson_r == pytest.approx(0.5)
        assert v.spearman_r == pytest.approx(0.48)
        assert v.mae == pytest.approx(0.3)
        assert v.n == 1000
        assert v.trait_type == "continuous"
        # Documented defaults for the binary block
        assert v.auc is None
        assert v.nagelkerke_r2 is None
        assert v.liability_r2 is None
        assert v.prevalence is None

    def test_construct_binary(self):
        """Build with binary-trait fields populated."""
        v = PGSValidation(
            r2=0.1,
            r2_incremental=0.02,
            pearson_r=0.3,
            spearman_r=0.28,
            mae=0.4,
            auc=0.75,
            nagelkerke_r2=0.12,
            liability_r2=0.08,
            n=500,
            trait_type="binary",
            prevalence=0.1,
        )
        assert v.auc == pytest.approx(0.75)
        assert v.nagelkerke_r2 == pytest.approx(0.12)
        assert v.liability_r2 == pytest.approx(0.08)
        assert v.prevalence == pytest.approx(0.1)
        assert v.trait_type == "binary"

    def test_round_trip_via_dataclass_fields(self):
        """All declared fields are accessible by name; reconstruction succeeds."""
        from dataclasses import fields

        v = PGSValidation(
            r2=0.2, r2_incremental=0.04, pearson_r=0.45, spearman_r=0.43,
            mae=0.35, n=100,
        )
        names = {f.name for f in fields(v)}
        assert "r2" in names
        assert "r2_incremental" in names
        assert "pearson_r" in names
        assert "spearman_r" in names
        assert "mae" in names
        assert "auc" in names
        assert "nagelkerke_r2" in names
        assert "liability_r2" in names
        assert "n" in names
        assert "trait_type" in names
        assert "prevalence" in names

    def test_repr_contains_class_name_and_metrics(self):
        """The custom ``__repr__`` opens with ``PGSValidation(...)`` and
        includes the formatted continuous-trait metrics."""
        v = PGSValidation(
            r2=0.25,
            r2_incremental=0.05,
            pearson_r=0.5,
            spearman_r=0.48,
            mae=0.3,
            n=1000,
        )
        s = repr(v)
        assert "PGSValidation" in s
        assert "n=1000" in s
        assert "trait_type='continuous'" in s
        # Custom repr formats R² to 6 decimals
        assert "0.250000" in s


# ---------------------------------------------------------------------------
# torchgwas.pgs.scoring.ScoringResult
# ---------------------------------------------------------------------------


class TestScoringResult:
    """``ScoringResult`` is a dataclass holding the per-individual PGS
    plus alignment audit (n_snp_used, n_snp_missing, n_snp_flipped,
    snp_used)."""

    def test_construct(self):
        """Build with documented fields and round-trip every attribute."""
        pgs = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        res = ScoringResult(
            pgs=pgs,
            n_snp_used=10,
            n_snp_missing=2,
            n_snp_flipped=1,
            snp_used=[f"rs{i}" for i in range(10)],
        )
        assert torch.equal(res.pgs, pgs)
        assert res.n_snp_used == 10
        assert res.n_snp_missing == 2
        assert res.n_snp_flipped == 1
        assert len(res.snp_used) == 10
        assert res.snp_used[0] == "rs0"

    def test_round_trip_via_dataclass_fields(self):
        """All fields are addressable; reconstruction yields equal values."""
        from dataclasses import fields

        pgs = torch.zeros(5, dtype=torch.float64)
        res = ScoringResult(
            pgs=pgs, n_snp_used=0, n_snp_missing=0, n_snp_flipped=0, snp_used=[]
        )
        names = {f.name for f in fields(res)}
        assert names == {
            "pgs",
            "n_snp_used",
            "n_snp_missing",
            "n_snp_flipped",
            "snp_used",
        }
        res2 = ScoringResult(
            pgs=res.pgs,
            n_snp_used=res.n_snp_used,
            n_snp_missing=res.n_snp_missing,
            n_snp_flipped=res.n_snp_flipped,
            snp_used=list(res.snp_used),
        )
        assert torch.equal(res.pgs, res2.pgs)
        assert res.n_snp_used == res2.n_snp_used
        assert res.snp_used == res2.snp_used

    def test_repr_contains_class_name(self):
        """``repr`` includes the class name (default dataclass repr)."""
        res = ScoringResult(
            pgs=torch.zeros(2),
            n_snp_used=2,
            n_snp_missing=0,
            n_snp_flipped=0,
            snp_used=["rs1", "rs2"],
        )
        s = repr(res)
        assert "ScoringResult" in s


# ---------------------------------------------------------------------------
# torchgwas.pgs.ct.HAS_NATIVE_CT
# ---------------------------------------------------------------------------


class TestHasNativeCt:
    """``HAS_NATIVE_CT`` reflects whether the C++ ``_ct_native`` extension
    was built and is importable from ``torchgwas._native``."""

    def test_is_bool(self):
        """Constant is a Python bool (not bool-y)."""
        assert isinstance(HAS_NATIVE_CT, bool)

    def test_consistent_with_native_module(self):
        """Cross-check against the underlying native import: True iff the
        ``_ct_native`` module imports cleanly and is non-None."""
        try:
            from torchgwas._native import _ct_native
            native_present = _ct_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_CT == native_present


# ---------------------------------------------------------------------------
# torchgwas.pgs.diagnostics.HAS_NATIVE_ESS
# ---------------------------------------------------------------------------


class TestHasNativeEss:
    """``HAS_NATIVE_ESS`` reflects whether the C++ ``_ess_native``
    extension was built."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_ESS, bool)

    def test_consistent_with_native_module(self):
        try:
            from torchgwas._native import _ess_native
            native_present = _ess_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_ESS == native_present


# ---------------------------------------------------------------------------
# torchgwas.pgs.ldpred2.HAS_NATIVE_LDPRED2
# ---------------------------------------------------------------------------


class TestHasNativeLdpred2:
    """``HAS_NATIVE_LDPRED2`` reflects whether the C++ ``_ldpred2_native``
    extension was built."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_LDPRED2, bool)

    def test_consistent_with_native_module(self):
        try:
            from torchgwas._native import _ldpred2_native
            native_present = _ldpred2_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_LDPRED2 == native_present


# ---------------------------------------------------------------------------
# torchgwas.pgs.prscs.HAS_NATIVE_PRSCS
# ---------------------------------------------------------------------------


class TestHasNativePrscs:
    """``HAS_NATIVE_PRSCS`` reflects whether the C++ ``_prscs_native``
    extension was built."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_PRSCS, bool)

    def test_consistent_with_native_module(self):
        try:
            from torchgwas._native import _prscs_native
            native_present = _prscs_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_PRSCS == native_present


# ---------------------------------------------------------------------------
# torchgwas.pgs.validation.pi
# ---------------------------------------------------------------------------


class TestPgsValidationPi:
    """Investigation: ``torchgwas.pgs.validation.pi`` is reachable because
    ``validation.py`` imports ``from math import exp, log, pi, sqrt`` at
    module top-level. The auditor flagged it as a public symbol because
    nothing imports ``pi`` *from* ``torchgwas.pgs.validation``; it is a
    stray re-export with no callers.

    Decision: smoke-test that it equals ``math.pi`` and queue cleanup
    (changing the import to ``from math import exp, log, sqrt`` and
    using ``math.pi`` inline at the one call site that uses it) for T12.
    Not severe enough for an F3 fix-now: it's correct, just cluttered.
    """

    def test_is_math_pi(self):
        """Re-exported ``pi`` equals the canonical ``math.pi``."""
        # NOTE: ``pgs.validation.pi`` is a stray top-level
        # ``from math import ... pi ...`` re-export. Used in the module
        # only inside an internal logistic-fit routine; nothing imports
        # it from this module path. Cleanup queued for T12 — not F3
        # because the value is correct and the symbol is unused
        # externally.
        assert pgs_validation.pi == math.pi

    def test_is_float(self):
        """Type contract: ``pi`` is a Python float."""
        assert isinstance(pgs_validation.pi, float)
