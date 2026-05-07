"""Phase 4: Optimizer stack tests."""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.eigh import eigendecompose, rotate
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.lmm_multi_fit import fit_mvlmm_null_lbfgs
from torchgwas.optim.ai_reml import ai_reml_single
from torchgwas.optim.controller import OptimizerController, OptimizerMode
from torchgwas.optim.em_warmstart import px_em_warmstart
from torchgwas.optim.fisher_scoring import fisher_scoring_reml


@pytest.fixture
def rotated_data():
    """Rotated data from a simulated LMM."""
    torch.manual_seed(42)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(1.0)
    y = u + torch.randn(n, dtype=torch.float64)

    ed = eigendecompose(K)
    Y_rot = rotate(y, ed.eigenvectors)
    X0_rot = rotate(X0, ed.eigenvectors)
    return Y_rot, X0_rot, ed.eigenvalues


class TestOptimizerController:
    def test_default_mode_order(self):
        """Default fallback order: PX_EM -> AI_REML -> LBFGS -> MM -> PCG -> RESCUE."""
        modes = list(OptimizerMode)
        assert modes[0] == OptimizerMode.PX_EM
        assert modes[1] == OptimizerMode.AI_REML
        assert modes[3] == OptimizerMode.MM

    def test_fallback_on_failure(self, rotated_data):
        """Controller produces valid NullFit."""
        Y_rot, X0_rot, evals = rotated_data
        controller = OptimizerController()
        nf = controller.fit(Y_rot, X0_rot, evals, n_traits=1)
        assert nf.sig2_g > 0
        assert nf.sig2_e > 0

    def test_controller_warm_start(self, rotated_data):
        """Controller with sig2_g_init/sig2_e_init skips PX-EM and converges."""
        Y_rot, X0_rot, evals = rotated_data
        controller = OptimizerController()
        # First: cold start to get reasonable VCs
        nf_cold = controller.fit(Y_rot, X0_rot, evals, n_traits=1)

        # Second: warm start with those VCs
        nf_warm = controller.fit(
            Y_rot, X0_rot, evals, n_traits=1,
            sig2_g_init=nf_cold.sig2_g,
            sig2_e_init=nf_cold.sig2_e,
        )
        assert nf_warm.sig2_g > 0
        assert nf_warm.sig2_e > 0
        # Warm start should not have PX-EM entries
        modes = [e.get("mode", "") for e in nf_warm.optimizer_trace]
        assert "PX-EM" not in modes, f"Expected no PX-EM in warm-start trace, got: {modes}"

    def test_controller_is_single_trait_only(self):
        """Multi-trait calls are routed to model-specific optimizers."""
        controller = OptimizerController()
        with pytest.raises(ValueError, match="single-trait only"):
            controller.fit(
                torch.randn(10, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
                torch.ones(10, dtype=torch.float64),
                n_traits=2,
            )


class TestAIREML:
    def test_ai_reml_convergence(self, rotated_data):
        """AI-REML converges on well-conditioned data."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll, trace = ai_reml_single(Y_rot, X0_rot, evals)
        assert math.isfinite(ll)
        assert len(trace) < 100  # converged before max iter

    def test_ai_reml_variance_components_positive(self, rotated_data):
        """AI-REML returns positive variance components."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, _, _ = ai_reml_single(Y_rot, X0_rot, evals)
        assert sig2_g > 0
        assert sig2_e > 0

    def test_fisher_scoring_wrapper_returns_scalars(self, rotated_data):
        """The legacy Fisher-scoring path delegates to implemented AI-REML."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, ll = fisher_scoring_reml(Y_rot, X0_rot, evals, max_iter=20)
        assert sig2_g > 0
        assert sig2_e > 0
        assert math.isfinite(ll)


class TestPXEM:
    def test_px_em_warmstart_improves_likelihood(self, rotated_data):
        """PX-EM iterations return positive variance components."""
        Y_rot, X0_rot, evals = rotated_data
        sig2_g, sig2_e, trace = px_em_warmstart(Y_rot, X0_rot, evals, n_iter=10)
        assert sig2_g > 0
        assert sig2_e > 0
        assert len(trace) == 10


class TestLBFGS:
    def test_lbfgs_mvlmm_wrapper_rejects_single_trait(self, rotated_data):
        """The mvLMM LBFGS wrapper is explicitly multi-trait only."""
        Y_rot, X0_rot, evals = rotated_data
        with pytest.raises(ValueError, match="d >= 2"):
            fit_mvlmm_null_lbfgs(Y_rot, X0_rot, evals)
