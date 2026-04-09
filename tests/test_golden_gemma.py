"""Golden tests: TorchGWAS vs GEMMA reference outputs.

These tests compare TorchGWAS results against saved GEMMA outputs on canonical
datasets. Tolerances follow Section 16 of the charter:
  - Log-likelihood:      < 1e-4 relative
  - Variance components: < 1e-4 relative
  - Beta (effect size):  < 1e-4 relative
  - SE:                  < 1e-4 relative
  - P-value:             match to 4th decimal (< 5e-5 absolute for p > 1e-4)
  - GRM elements:        < 1e-6 relative
"""

from __future__ import annotations

import pytest

# Mark all tests in this module as golden
pytestmark = pytest.mark.golden


class TestGEMMALMMSingleTrait:
    """GEMMA univariate LMM reference comparison."""

    def test_variance_components(self):
        """sig2_g and sig2_e match GEMMA output within 1e-4 relative."""
        pytest.skip("Golden data not yet available")

    def test_log_likelihood(self):
        """Log-likelihood matches GEMMA within 1e-4 relative."""
        pytest.skip("Golden data not yet available")

    def test_beta_se(self):
        """Effect sizes and standard errors match GEMMA within 1e-4 relative."""
        pytest.skip("Golden data not yet available")

    def test_pvalues(self):
        """P-values match GEMMA to 4th decimal."""
        pytest.skip("Golden data not yet available")

    def test_grm_elements(self):
        """GRM diagonal and off-diagonal match GEMMA within 1e-6 relative."""
        pytest.skip("Golden data not yet available")


class TestGEMMAMvLMM:
    """GEMMA multivariate LMM reference comparison."""

    def test_vg_ve_matrices(self):
        """Vg and Ve matrices match GEMMA within 1e-4 relative."""
        pytest.skip("Golden data not yet available")

    def test_multi_trait_pvalues(self):
        """Multi-trait p-values match GEMMA to 4th decimal."""
        pytest.skip("Golden data not yet available")
