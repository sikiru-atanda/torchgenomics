"""Golden tests: TorchGWAS vs GAPIT3 reference outputs.

GAPIT3 references cover FarmCPU and BLINK models.
Tolerances follow Section 16 of the charter.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.golden


class TestGAPITFarmCPU:
    """GAPIT3 FarmCPU reference comparison."""

    def test_pvalues(self):
        """FarmCPU p-values match GAPIT3 to 4th decimal."""
        pytest.skip("Golden data not yet available")

    def test_pseudo_qtns(self):
        """Pseudo-QTN selection converges to similar set as GAPIT3."""
        pytest.skip("Golden data not yet available")


class TestGAPITBLINK:
    """GAPIT3 BLINK reference comparison."""

    def test_pvalues(self):
        """BLINK p-values match GAPIT3 to 4th decimal."""
        pytest.skip("Golden data not yet available")
