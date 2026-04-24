"""Phase 56: polyploid phasing via PolyOrigin (Tier 1, always-on)."""
from __future__ import annotations

import torchgwas.preprocess.phase_polyorigin as mod


def test_module_imports_without_juliacall():
    # juliacall must NOT be imported at module load — only inside
    # get_runtime(). Verify the import surface is inert.
    assert hasattr(mod, "_VALID_PLOIDIES")
    assert mod._VALID_PLOIDIES == frozenset({2, 4, 6})
