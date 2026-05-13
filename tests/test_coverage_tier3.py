"""Tier-3 (smoke) coverage tests for torchgwas.annotate and torchgwas.cli.

Bar (spec section 4.3 Tier 3):
- One test per public function: import + invoke with simplest valid input.
- Assert: doesn't raise, returns documented type, non-trivial output.

Symbols covered:
- torchgwas.annotate.NCBIError (exception class)
- torchgwas.annotate.AnnotatedGene (dataclass)
- torchgwas.annotate.AnnotatedHit (dataclass)

Removed (post-campaign cleanup): ``torchgwas.cli.TYPE_CHECKING`` was a
``typing.TYPE_CHECKING`` re-export at module top-level. The cli.py
import was renamed to ``from typing import TYPE_CHECKING as
_TYPE_CHECKING`` so the symbol no longer surfaces in audit
public-symbol enumeration; the smoke test (``TestCliTypeChecking``)
was deleted with the symbol.
"""

from __future__ import annotations

import pytest

from torchgwas.annotate import AnnotatedGene, AnnotatedHit, NCBIError

pytestmark = pytest.mark.timeout(30)


class TestNCBIError:
    """NCBIError is the exception type raised on non-retryable NCBI failures."""

    def test_is_exception_subclass(self):
        assert issubclass(NCBIError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(NCBIError, match="test message"):
            raise NCBIError("test message")

    def test_message_round_trip(self):
        e = NCBIError("hello")
        assert str(e) == "hello"


class TestAnnotatedGene:
    """Dataclass mirroring NCBI gene metadata for one gene overlapping a SNP."""

    def _make(self, **overrides):
        defaults = dict(
            gene_id="123",
            symbol="GENE1",
            description="Test gene",
            gene_type="protein_coding",
            chrom="1",
            start=100,
            end=2000,
            orientation="+",
            genomic_distance_to_snp=500,
        )
        defaults.update(overrides)
        return AnnotatedGene(**defaults)

    def test_construct_with_required_fields(self):
        gene = self._make()
        assert gene.gene_id == "123"
        assert gene.symbol == "GENE1"
        assert gene.start == 100
        assert gene.end == 2000

    def test_default_factory_lists_are_per_instance(self):
        g1 = self._make()
        g2 = self._make()
        g1.go_terms.append({"id": "GO:0001"})
        assert g2.go_terms == [], "default-factory lists must not alias"

    def test_repr_does_not_crash(self):
        s = repr(self._make())
        assert "AnnotatedGene" in s


class TestAnnotatedHit:
    """Dataclass bundling one SNP hit and its overlapping genes."""

    def test_construct_with_required_fields(self):
        hit = AnnotatedHit(snp="rs1", chrom="1", pos=1500, p=1e-8)
        assert hit.snp == "rs1"
        assert hit.status == "ok"
        assert hit.genes == []

    def test_genes_default_factory_isolated(self):
        h1 = AnnotatedHit(snp="rs1", chrom="1", pos=1, p=1e-5)
        h2 = AnnotatedHit(snp="rs2", chrom="1", pos=2, p=1e-5)
        h1.genes.append(AnnotatedGene(
            gene_id="x", symbol="X", description="", gene_type="",
            chrom="1", start=1, end=2, orientation="+", genomic_distance_to_snp=0,
        ))
        assert h2.genes == [], "default-factory lists must not alias"

    def test_repr_does_not_crash(self):
        s = repr(AnnotatedHit(snp="rs9", chrom="3", pos=42, p=0.001))
        assert "AnnotatedHit" in s


# TestCliTypeChecking removed post-campaign: ``torchgwas.cli.TYPE_CHECKING``
# was a stdlib re-export at module top-level. The cli.py import is now
# ``from typing import TYPE_CHECKING as _TYPE_CHECKING`` so the symbol
# no longer appears in public-symbol enumeration. Conditional type-only
# imports inside cli.py still use the renamed binding correctly.
