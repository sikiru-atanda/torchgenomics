"""Tests for update_null() — ASReml-R-style warm-start optimizer resumption."""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.config import NumericalConfig
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import update_null
from torchgwas.models.single_trait_lmm import SingleTraitLMM

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def lmm_data():
    """Simulated single-trait LMM data (200 samples, h2~0.5)."""
    torch.manual_seed(42)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(1.0)
    Y = u + torch.randn(n, dtype=torch.float64)

    return Y, X0, K


@pytest.fixture
def partial_fit(lmm_data):
    """NullFit from a deliberately short run (max_iter=3, likely non-converged)."""
    Y, X0, K = lmm_data
    config = NumericalConfig()
    config.reml_max_iter = 3
    model = SingleTraitLMM(config=config)
    nf = model.fit_null(Y, X0, K)
    return nf, model


@pytest.fixture
def full_fit(lmm_data):
    """NullFit from a full run (max_iter=100, converged)."""
    Y, X0, K = lmm_data
    model = SingleTraitLMM()
    nf = model.fit_null(Y, X0, K)
    return nf, model


# ── Single-trait tests ────────────────────────────────────────────────

class TestUpdateNullSingleTrait:
    def test_warm_start_converges(self, partial_fit):
        """A short initial fit + update should converge."""
        nf, model = partial_fit
        updated = model.update_null(nf, max_iter=100)
        assert updated.converged
        assert updated.sig2_g > 0
        assert updated.sig2_e > 0

    def test_matches_fresh_fit(self, partial_fit, full_fit):
        """Updated fit should reach similar variance components as a fresh full fit."""
        nf_partial, model = partial_fit
        nf_full, _ = full_fit

        updated = model.update_null(nf_partial, max_iter=100)

        # Variance components should be close (within 10%)
        assert abs(updated.sig2_g - nf_full.sig2_g) < 0.1 * nf_full.sig2_g + 1e-6
        assert abs(updated.sig2_e - nf_full.sig2_e) < 0.1 * nf_full.sig2_e + 1e-6

    def test_trace_accumulates(self, partial_fit):
        """Updated trace should be longer than original trace."""
        nf, model = partial_fit
        orig_len = len(nf.optimizer_trace)
        updated = model.update_null(nf, max_iter=50)
        assert len(updated.optimizer_trace) > orig_len

    def test_trace_has_update_marker(self, partial_fit):
        """New trace entries should have update=True marker."""
        nf, model = partial_fit
        orig_len = len(nf.optimizer_trace)
        updated = model.update_null(nf, max_iter=50)

        # Old entries should NOT have update marker
        for entry in updated.optimizer_trace[:orig_len]:
            assert "update" not in entry or entry.get("update") is not True

        # New entries should have update=True
        new_entries = updated.optimizer_trace[orig_len:]
        assert len(new_entries) > 0
        for entry in new_entries:
            assert entry.get("update") is True

    def test_skips_eigendecomposition(self, partial_fit):
        """Update should reuse eigenvectors from the original fit."""
        nf, model = partial_fit
        orig_eigvecs = nf.eigenvectors
        updated = model.update_null(nf, max_iter=50)
        # Should be the exact same object (not recomputed)
        assert updated.eigenvectors is orig_eigvecs

    def test_already_converged_stable(self, full_fit):
        """Updating an already-converged fit should return similar VCs."""
        nf, model = full_fit
        assert nf.converged

        updated = model.update_null(nf, max_iter=50)
        assert updated.converged
        # Warm-started re-optimization may shift slightly due to different
        # PX-EM vs direct start path; 1% relative tolerance is sufficient
        assert abs(updated.sig2_g - nf.sig2_g) < 0.01 * nf.sig2_g + 1e-4
        assert abs(updated.sig2_e - nf.sig2_e) < 0.01 * nf.sig2_e + 1e-4

    def test_emma_to_aireml_transition(self, lmm_data):
        """Update of an EMMA fit should use AI-REML stack (not EMMA again)."""
        Y, X0, K = lmm_data
        config = NumericalConfig()
        config.reml_method = "emma"
        model = SingleTraitLMM(config=config)
        nf_emma = model.fit_null(Y, X0, K)

        # EMMA always converges
        assert nf_emma.converged

        # Now update — should use AI-REML stack (not EMMA)
        config_update = NumericalConfig()  # default is AI-REML stack
        updated = update_null(nf_emma, max_iter=50, config=config_update)

        modes = [e.get("mode", "") for e in updated.optimizer_trace]
        # The update entries (after EMMA entries) should include AI-REML
        has_ai_reml = any("AI-REML" in m for m in modes)
        assert has_ai_reml, f"Expected AI-REML in trace, got modes: {set(modes)}"


# ── Multi-trait tests ─────────────────────────────────────────────────

class TestUpdateNullMultiTrait:
    def test_multitrait_warm_start(self):
        """Multi-trait update should converge from partial fit."""
        torch.manual_seed(7)
        n, m = 100, 300
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = L @ torch.randn(n, dtype=torch.float64) + torch.randn(n, dtype=torch.float64)
        Y[:, 1] = L @ torch.randn(n, dtype=torch.float64) + torch.randn(n, dtype=torch.float64)

        from torchgwas.models.multi_trait_lmm import MultiTraitLMM

        config = NumericalConfig()
        config.reml_max_iter = 3
        model = MultiTraitLMM(config=config)
        nf = model.fit_null(Y, X0, K)

        updated = model.update_null(nf, max_iter=100)
        assert updated.Vg is not None
        assert updated.Ve is not None
        assert updated.log_likelihood >= nf.log_likelihood - 1e-4

    def test_vg_ve_positive_definite(self):
        """Updated Vg and Ve should remain positive definite."""
        torch.manual_seed(99)
        n, m = 80, 200
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = L @ torch.randn(n, dtype=torch.float64) + torch.randn(n, dtype=torch.float64)
        Y[:, 1] = L @ torch.randn(n, dtype=torch.float64) + torch.randn(n, dtype=torch.float64)

        from torchgwas.models.multi_trait_lmm import MultiTraitLMM

        config = NumericalConfig()
        config.reml_max_iter = 5
        model = MultiTraitLMM(config=config)
        nf = model.fit_null(Y, X0, K)
        updated = model.update_null(nf, max_iter=100)

        # Check positive definite
        eig_g = torch.linalg.eigvalsh(updated.Vg)
        eig_e = torch.linalg.eigvalsh(updated.Ve)
        assert (eig_g > -1e-8).all(), f"Vg not PD: min eigval={eig_g.min()}"
        assert (eig_e > -1e-8).all(), f"Ve not PD: min eigval={eig_e.min()}"


# ── Multi-kernel tests ────────────────────────────────────────────────

class TestUpdateNullMultiKernel:
    def test_multikernel_warm_start(self):
        """Multi-kernel update should improve log-likelihood."""
        torch.manual_seed(123)
        n, m = 80, 200
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        K_a, _ = grm_vanraden(G)
        K_d = K_a * K_a  # Hadamard for epistatic-like
        X0 = torch.ones(n, 1, dtype=torch.float64)

        L = torch.linalg.cholesky(K_a + 1e-6 * torch.eye(n, dtype=torch.float64))
        Y = L @ torch.randn(n, dtype=torch.float64) + torch.randn(n, dtype=torch.float64)

        from torchgwas.models.multi_kernel_lmm import MultiKernelLMM

        config = NumericalConfig()
        config.reml_max_iter = 3
        model = MultiKernelLMM(config=config)
        nf = model.fit_null(Y, X0, kernels=[K_a, K_d], kernel_names=["additive", "epistatic"])

        updated = model.update_null(nf, max_iter=100)
        assert updated.sig2_g > 0
        assert updated.sig2_e > 0
        assert updated.log_likelihood >= nf.log_likelihood - 1e-4
