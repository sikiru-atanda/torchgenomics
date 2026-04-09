"""Phase 4-5: Single-trait LMM tests."""

from __future__ import annotations

import pytest


class TestSingleTraitLMM:
    """Tests for torchgwas.models.lmm_single.SingleTraitLMM."""

    def test_lmm_conforms_to_protocol(self):
        """SingleTraitLMM implements BaseModel protocol."""
        pytest.skip("Phase 4: not yet implemented")

    def test_reml_convergence(self):
        """REML optimizer converges (converged=True in NullFit)."""
        pytest.skip("Phase 4: not yet implemented")

    def test_variance_components_positive(self):
        """sig2_g and sig2_e are positive."""
        pytest.skip("Phase 4: not yet implemented")

    def test_heritability_range(self):
        """h2 = sig2_g / (sig2_g + sig2_e) is in [0, 1]."""
        pytest.skip("Phase 4: not yet implemented")

    def test_rotated_residuals_orthogonal(self):
        """Y_rot and X0_rot have expected shapes after rotation."""
        pytest.skip("Phase 4: not yet implemented")

    def test_loco_scan(self):
        """LOCO scan excludes target chromosome from GRM."""
        pytest.skip("Phase 5: not yet implemented")

    def test_optimizer_fallback(self):
        """Optimizer falls back from PX-EM to AI-REML on failure."""
        pytest.skip("Phase 4: not yet implemented")
