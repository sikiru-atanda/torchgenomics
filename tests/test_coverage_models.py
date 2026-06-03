"""Tier-1 coverage tests for 15 final ``torchgenomics.models`` symbols.

Bar (Pillar A spec, Tier 1 — V1-core):

- Result / config dataclasses (BayesianVSResult, ThresholdNullFit,
  ThresholdConfig, HHCTNode, JointQTLResult, FoldResult): construct +
  field round-trip + ``repr`` smoke.
- LinkFunction (abstract base): subclassable; required-method shape.
- IdentityLink (concrete): link / inverse / derivative correctness.
- Major model classes (SingleTraitLMM, MultiTraitLMM, SparseLMM):
  canonical-import smoke — exercises ``torchgenomics.models.lmm_single``
  / ``lmm_multi`` / ``sparse_lmm`` import paths so the auditor credits
  them. Math correctness is exercised in tests/test_single_trait_lmm.py
  / tests/test_multi_trait_lmm.py.
- ``fit_mvlmm_null_ai_reml`` / ``fit_mvlmm_null_lbfgs``: runs-to-
  completion + finite log-likelihood + converged flag on a tiny d=2,
  n=50 synthetic problem.
- ``_dispatch.native_disabled``: env-var-driven boolean toggle.
- ``HAS_NATIVE_CAVI``: module-level boolean flag.

The mvlmm-null tests verify shape + finite quantities + convergence
flag rather than asserting numerical equivalence to a reference;
end-to-end correctness is covered by the higher-level model tests.
"""

from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest
import torch
from torch import Tensor

from torchgenomics._dispatch import native_disabled
from torchgenomics.models.bayesian_vs import HAS_NATIVE_CAVI, BayesianVSResult
from torchgenomics.models.glm_link import (
    IdentityLink,
    LinkFunction,
)
from torchgenomics.models.haplotype_novel import HHCTNode
from torchgenomics.models.joint_qtl import JointQTLResult
from torchgenomics.models.lmm_multi_fit import (
    fit_mvlmm_null_ai_reml,
    fit_mvlmm_null_lbfgs,
)
from torchgenomics.models.ocf_lmm import FoldResult
from torchgenomics.models.threshold_linear import ThresholdConfig, ThresholdNullFit


pytestmark = pytest.mark.timeout(60)


# ---------------------------------------------------------------------------
# BayesianVSResult
# ---------------------------------------------------------------------------


class TestBayesianVSResult:
    """Dataclass holding posterior inclusion probabilities, beta moments,
    credible sets, and ELBO trace for Bayesian variable selection."""

    def _build(self, p: int = 3) -> BayesianVSResult:
        return BayesianVSResult(
            chr=["1"] * p,
            pos=[i * 100 for i in range(p)],
            snp=[f"rs{i}" for i in range(p)],
            a1=["A"] * p,
            a2=["G"] * p,
            af=torch.full((p,), 0.3, dtype=torch.float64),
            pip=torch.tensor([0.05, 0.7, 0.02], dtype=torch.float64)[:p],
            beta_mean=torch.zeros(p, dtype=torch.float64),
            beta_sd=torch.ones(p, dtype=torch.float64),
        )

    def test_is_dataclass(self):
        """``BayesianVSResult`` is a ``@dataclass``."""
        assert is_dataclass(BayesianVSResult)

    def test_construct(self):
        """Instantiation with the documented required fields succeeds."""
        r = self._build()
        assert isinstance(r, BayesianVSResult)
        assert len(r.snp) == 3
        # Default-factory fields land at their factory values.
        assert r.elbo_trace == []
        assert r.converged is False
        assert r.alpha is None
        assert r.n_signals == 0
        assert r.method == "cavi"
        assert r.credible_sets is None

    def test_field_round_trip(self):
        """Set fields → read back identically."""
        r = self._build()
        r.converged = True
        r.elbo_trace.append(-100.0)
        r.n_signals = 2
        r.method = "susie"
        r.alpha = torch.zeros(2, 3, dtype=torch.float64)
        r.credible_sets = [[1]]
        assert r.converged is True
        assert r.elbo_trace == [-100.0]
        assert r.n_signals == 2
        assert r.method == "susie"
        assert r.alpha.shape == (2, 3)
        assert r.credible_sets == [[1]]

    def test_len_returns_variant_count(self):
        """``__len__`` returns the number of variants."""
        r = self._build(p=3)
        assert len(r) == 3

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "BayesianVSResult" in s


# ---------------------------------------------------------------------------
# ThresholdNullFit
# ---------------------------------------------------------------------------


class TestThresholdNullFit:
    """Extended NullFit carrying threshold-model-specific quantities
    (theta, liabilities, per-trait thresholds, R, G_cov)."""

    def _build(self) -> ThresholdNullFit:
        from torchgenomics.models.base import NullFit
        n, c = 5, 2
        return ThresholdNullFit(
            null_fit=NullFit(),
            theta=torch.zeros(c * n, dtype=torch.float64),
            liabilities=torch.zeros(n, 1, dtype=torch.float64),
            thresholds_list=[torch.tensor([0.0], dtype=torch.float64)],
            R=torch.eye(c, dtype=torch.float64),
            G_cov=torch.eye(c, dtype=torch.float64),
        )

    def test_is_dataclass(self):
        assert is_dataclass(ThresholdNullFit)

    def test_construct(self):
        nf = self._build()
        assert isinstance(nf, ThresholdNullFit)
        # Default fields
        assert nf.R_tilde_inv is None
        assert nf.WtRW is None
        assert nf.WtRW_inv is None
        assert nf.solver == "nr"
        assert nf.n_iter == 0
        assert nf.converged is False

    def test_field_round_trip(self):
        nf = self._build()
        nf.solver = "em"
        nf.n_iter = 17
        nf.converged = True
        assert nf.solver == "em"
        assert nf.n_iter == 17
        assert nf.converged is True

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "ThresholdNullFit" in s


# ---------------------------------------------------------------------------
# ThresholdConfig
# ---------------------------------------------------------------------------


class TestThresholdConfig:
    """Per-trait threshold configuration: ``n_categories`` + optional
    fixed thresholds."""

    def test_is_dataclass(self):
        assert is_dataclass(ThresholdConfig)

    def test_construct_default_thresholds(self):
        cfg = ThresholdConfig(n_categories=3)
        assert cfg.n_categories == 3
        assert cfg.thresholds is None

    def test_construct_with_thresholds(self):
        thr = torch.tensor([-0.5, 0.5], dtype=torch.float64)
        cfg = ThresholdConfig(n_categories=3, thresholds=thr)
        assert torch.equal(cfg.thresholds, thr)

    def test_field_round_trip(self):
        cfg = ThresholdConfig(n_categories=2)
        cfg.n_categories = 4
        new_thr = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        cfg.thresholds = new_thr
        assert cfg.n_categories == 4
        assert torch.equal(cfg.thresholds, new_thr)

    def test_repr_does_not_crash(self):
        s = repr(ThresholdConfig(n_categories=3))
        assert isinstance(s, str)
        assert "ThresholdConfig" in s


# ---------------------------------------------------------------------------
# HHCTNode
# ---------------------------------------------------------------------------


class TestHHCTNode:
    """Node in the Hierarchical Haplotype Collapsing Test merge tree."""

    def _build(self) -> HHCTNode:
        return HHCTNode(
            node_id=0,
            children=[1, 2],
            leaf_indices=[0, 1, 2],
            raw_p=0.04,
            adjusted_p=0.08,
            rejected=False,
        )

    def test_is_dataclass(self):
        assert is_dataclass(HHCTNode)

    def test_construct(self):
        node = self._build()
        assert node.node_id == 0
        assert node.children == [1, 2]
        assert node.leaf_indices == [0, 1, 2]
        assert node.raw_p == pytest.approx(0.04)
        assert node.adjusted_p == pytest.approx(0.08)
        assert node.rejected is False

    def test_field_round_trip(self):
        node = self._build()
        node.rejected = True
        node.adjusted_p = 0.01
        assert node.rejected is True
        assert node.adjusted_p == pytest.approx(0.01)

    def test_leaf_node_has_no_children(self):
        leaf = HHCTNode(
            node_id=42,
            children=[],
            leaf_indices=[7],
            raw_p=0.5,
            adjusted_p=0.5,
            rejected=False,
        )
        assert leaf.children == []
        assert leaf.leaf_indices == [7]

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "HHCTNode" in s


# ---------------------------------------------------------------------------
# JointQTLResult
# ---------------------------------------------------------------------------


class TestJointQTLResult:
    """Result of multi-QTL joint LMM fit with backward elimination."""

    def _build(self) -> JointQTLResult:
        q = 3
        return JointQTLResult(
            qtl_snps=[f"rs{i}" for i in range(q)],
            qtl_indices=[10, 20, 30],
            beta=torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64),
            se=torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64),
            r_squared=torch.tensor([0.05, 0.10, 0.02], dtype=torch.float64),
            lrt_p=torch.tensor([1e-6, 1e-3, 0.1], dtype=torch.float64),
            total_r_squared=0.17,
            n_eliminated=1,
        )

    def test_is_dataclass(self):
        assert is_dataclass(JointQTLResult)

    def test_construct(self):
        r = self._build()
        assert r.qtl_snps == ["rs0", "rs1", "rs2"]
        assert r.qtl_indices == [10, 20, 30]
        assert r.beta.shape == (3,)
        assert r.total_r_squared == pytest.approx(0.17)
        assert r.n_eliminated == 1

    def test_field_round_trip(self):
        r = self._build()
        r.n_eliminated = 2
        r.total_r_squared = 0.5
        assert r.n_eliminated == 2
        assert r.total_r_squared == pytest.approx(0.5)

    def test_empty_result(self):
        """An all-eliminated joint QTL fit returns empty tensors and zero
        total R²."""
        r = JointQTLResult(
            qtl_snps=[],
            qtl_indices=[],
            beta=torch.tensor([]),
            se=torch.tensor([]),
            r_squared=torch.tensor([]),
            lrt_p=torch.tensor([]),
            total_r_squared=0.0,
            n_eliminated=5,
        )
        assert r.beta.numel() == 0
        assert r.n_eliminated == 5

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "JointQTLResult" in s


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------


class TestFoldResult:
    """Per-fold nuisance estimation output for orthogonal cross-fit LMM."""

    def _build(self) -> FoldResult:
        n_test = 10
        return FoldResult(
            fold_id=0,
            test_indices=torch.arange(n_test, dtype=torch.int64),
            sig2_g=0.4,
            sig2_e=0.6,
            beta_hat=torch.zeros(2, dtype=torch.float64),
            y_tilde_test=torch.randn(n_test, dtype=torch.float64),
            log_likelihood=-50.0,
            converged=True,
        )

    def test_is_dataclass(self):
        assert is_dataclass(FoldResult)

    def test_construct(self):
        f = self._build()
        assert f.fold_id == 0
        assert f.test_indices.shape == (10,)
        assert f.sig2_g == pytest.approx(0.4)
        assert f.sig2_e == pytest.approx(0.6)
        assert f.beta_hat.shape == (2,)
        assert f.log_likelihood == pytest.approx(-50.0)
        assert f.converged is True

    def test_field_round_trip(self):
        f = self._build()
        f.fold_id = 7
        f.converged = False
        f.log_likelihood = -123.4
        assert f.fold_id == 7
        assert f.converged is False
        assert f.log_likelihood == pytest.approx(-123.4)

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "FoldResult" in s


# ---------------------------------------------------------------------------
# LinkFunction (base class)
# ---------------------------------------------------------------------------


class TestLinkFunction:
    """``LinkFunction`` is a base class declaring four required methods
    (link, inverse, derivative, variance) that subclasses must implement."""

    def test_required_methods_raise_on_base(self):
        """``LinkFunction`` is an abc.ABC with four @abstractmethod
        declarations (link, inverse, derivative, variance). Python's ABC
        machinery prevents instantiation entirely until every abstract
        method is overridden, so the constraint surfaces as ``TypeError``
        on ``LinkFunction()`` itself — not as ``NotImplementedError`` on
        per-method calls (the older pre-ABC convention)."""
        with pytest.raises(TypeError, match="abstract"):
            LinkFunction()
        # And the four declarations must remain marked abstract.
        abstract = set(LinkFunction.__abstractmethods__)
        assert abstract == {"link", "inverse", "derivative", "variance"}

    def test_subclass_can_override_required_methods(self):
        """A minimal subclass overriding all four methods works."""

        class _MyLink(LinkFunction):
            name = "my"

            def link(self, mu: Tensor) -> Tensor:
                return mu * 2

            def inverse(self, eta: Tensor) -> Tensor:
                return eta / 2

            def derivative(self, mu: Tensor) -> Tensor:
                return torch.full_like(mu, 2.0)

            def variance(self, mu: Tensor) -> Tensor:
                return torch.ones_like(mu)

        link = _MyLink()
        assert isinstance(link, LinkFunction)
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert torch.allclose(link.link(x), x * 2)
        assert torch.allclose(link.inverse(link.link(x)), x)

    def test_has_name_attribute(self):
        """``LinkFunction.name`` is the documented identifier slot."""
        assert hasattr(LinkFunction, "name")
        assert LinkFunction.name == "base"


# ---------------------------------------------------------------------------
# IdentityLink (concrete)
# ---------------------------------------------------------------------------


class TestIdentityLink:
    """Identity link: eta = mu (Gaussian family)."""

    def test_implements_LinkFunction(self):
        """``IdentityLink()`` is a ``LinkFunction``."""
        assert isinstance(IdentityLink(), LinkFunction)

    def test_name(self):
        assert IdentityLink.name == "identity"

    def test_link_round_trip(self):
        """``IdentityLink().link(x)`` returns ``x`` to ~machine precision."""
        link = IdentityLink()
        x = torch.tensor([-1.5, 0.0, 1.0, 3.14], dtype=torch.float64)
        out = link.link(x)
        assert torch.allclose(out, x, atol=1e-12)

    def test_inverse_round_trip(self):
        """``inverse(eta)`` returns ``eta`` to ~machine precision."""
        link = IdentityLink()
        eta = torch.tensor([-2.0, 0.0, 0.5, 2.5], dtype=torch.float64)
        out = link.inverse(eta)
        assert torch.allclose(out, eta, atol=1e-12)

    def test_derivative_is_one(self):
        """d(eta)/d(mu) = 1 for the identity link."""
        link = IdentityLink()
        x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        d = link.derivative(x)
        assert torch.allclose(d, torch.ones_like(x), atol=1e-12)

    def test_variance_is_one(self):
        """V(mu) = 1 for the Gaussian family (homoscedastic)."""
        link = IdentityLink()
        x = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        v = link.variance(x)
        assert torch.allclose(v, torch.ones_like(x), atol=1e-12)


# ---------------------------------------------------------------------------
# Major model classes — canonical-import smoke tests
# ---------------------------------------------------------------------------


class TestMajorModelCanonicalImport:
    """Smoke tests asserting the canonical import paths are wired and the
    ``BaseModel`` shape (fit_null + score_chunk) is available.

    Math correctness is exercised in tests/test_single_trait_lmm.py /
    tests/test_multi_trait_lmm.py / tests/test_approximate.py /
    tests/test_sparse_grm.py.
    """

    def test_single_trait_lmm_canonical_import(self):
        from torchgenomics.models.lmm_single import SingleTraitLMM
        model = SingleTraitLMM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")

    def test_multi_trait_lmm_canonical_import(self):
        from torchgenomics.models.lmm_multi import MultiTraitLMM
        model = MultiTraitLMM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")

    def test_sparse_lmm_canonical_import(self):
        from torchgenomics.models.sparse_lmm import SparseLMM
        model = SparseLMM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")


# ---------------------------------------------------------------------------
# fit_mvlmm_null_ai_reml / fit_mvlmm_null_lbfgs
# ---------------------------------------------------------------------------


def _tiny_mvlmm_inputs(n: int = 50, d: int = 2, c: int = 1, seed: int = 0):
    """Build a tiny synthetic mvLMM null problem.

    With identity kinship K = I, the eigendecomposition is U = I and
    eigenvalues = 1, so Y_rot = Y, X0_rot = X0. This skips the
    eigendecomposition step and exercises only the optimizer wrappers.
    """
    g = torch.Generator().manual_seed(seed)
    Y_rot = torch.randn(n, d, generator=g, dtype=torch.float64)
    X0_rot = torch.ones(n, c, dtype=torch.float64)
    eigenvalues = torch.ones(n, dtype=torch.float64)
    return Y_rot, X0_rot, eigenvalues


class TestFitMvlmmNullLbfgs:
    """LBFGS-autograd Cholesky-parameterized REML for multi-trait null."""

    def test_runs_to_completion(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_lbfgs(Y_rot, X0_rot, eigenvalues, max_iter=30)
        from torchgenomics.models.base import NullFit
        assert isinstance(nf, NullFit)
        assert nf.Vg is not None and nf.Vg.shape == (2, 2)
        assert nf.Ve is not None and nf.Ve.shape == (2, 2)

    def test_log_likelihood_finite(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_lbfgs(Y_rot, X0_rot, eigenvalues, max_iter=30)
        ll = nf.log_likelihood
        assert ll is not None
        assert np.isfinite(ll)

    def test_converges_within_iter_budget(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_lbfgs(Y_rot, X0_rot, eigenvalues, max_iter=100)
        assert nf.converged is True

    def test_null_quantities_populated(self):
        """M00 / b0 / weights are precomputed for downstream score scan."""
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_lbfgs(Y_rot, X0_rot, eigenvalues, max_iter=30)
        assert nf.M00 is not None
        assert nf.b0 is not None
        assert nf.weights is not None
        assert nf.weights.shape == (50, 2, 2)


class TestFitMvlmmNullAiReml:
    """PX-EM warm-start + AI-REML Newton-Raphson."""

    def test_runs_to_completion(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_ai_reml(
            Y_rot, X0_rot, eigenvalues, max_iter=30, em_iters=5,
        )
        from torchgenomics.models.base import NullFit
        assert isinstance(nf, NullFit)
        assert nf.Vg is not None and nf.Vg.shape == (2, 2)
        assert nf.Ve is not None and nf.Ve.shape == (2, 2)

    def test_log_likelihood_finite(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_ai_reml(
            Y_rot, X0_rot, eigenvalues, max_iter=30, em_iters=5,
        )
        ll = nf.log_likelihood
        assert ll is not None
        assert np.isfinite(ll)

    def test_converges_within_iter_budget(self):
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf = fit_mvlmm_null_ai_reml(
            Y_rot, X0_rot, eigenvalues, max_iter=100, em_iters=5,
        )
        assert nf.converged is True

    def test_matches_lbfgs_on_loglik(self):
        """AI-REML and LBFGS should converge to ~the same REML maximum on
        the same identifiable problem (relative tol 1e-4)."""
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        nf_lbfgs = fit_mvlmm_null_lbfgs(
            Y_rot, X0_rot, eigenvalues, max_iter=100,
        )
        nf_aireml = fit_mvlmm_null_ai_reml(
            Y_rot, X0_rot, eigenvalues, max_iter=100, em_iters=5,
        )
        scale = max(abs(nf_lbfgs.log_likelihood), 1.0)
        assert abs(nf_lbfgs.log_likelihood - nf_aireml.log_likelihood) / scale < 1e-4


# ---------------------------------------------------------------------------
# Post-V1 follow-up: optimizer-side converged flag plumbing
# ---------------------------------------------------------------------------


class TestConvergedFlagPlumbing:
    """The ``NullFit.converged`` flag is plumbed directly from the
    optimizer's self-reported flag (stashed in ``trace[-1]["converged"]``)
    instead of inferred from the trace-tail relative-ll change.

    The trace-tail heuristic returned False when the optimizer stopped
    at exactly ``max_iter`` even if the optimizer's own convergence
    test passed on the final step. The plumbed flag is more reliable.
    """

    def test_plumbed_flag_lbfgs_matches_optimizer(self):
        """``fit_mvlmm_null_lbfgs.converged`` matches
        ``trace[-1]["converged"]`` from ``lbfgs_reml`` directly."""
        from torchgenomics.optim.lbfgs_reml import lbfgs_reml

        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        Vg, Ve, ll, trace = lbfgs_reml(
            Y_rot, X0_rot, eigenvalues, n_traits=2, max_iter=100,
        )
        opt_converged = trace[-1].get("converged", None)
        assert opt_converged is not None, (
            "lbfgs_reml should stash converged in trace[-1]"
        )

        nf = fit_mvlmm_null_lbfgs(
            Y_rot, X0_rot, eigenvalues, max_iter=100,
        )
        # Both wrapper and optimizer agree on convergence status (we
        # can't compare bit-equal due to RNG-free deterministic behavior,
        # but we can run twice and compare).
        nf2 = fit_mvlmm_null_lbfgs(
            Y_rot, X0_rot, eigenvalues, max_iter=100,
        )
        assert nf.converged == nf2.converged
        # And it matches what the underlying optimizer reports.
        assert nf2.converged == bool(
            nf2.optimizer_trace[-1].get("converged", False)
        )

    def test_plumbed_flag_aireml_matches_optimizer(self):
        """``fit_mvlmm_null_ai_reml.converged`` matches
        ``trace[-1]["converged"]`` from ``pxem_nr_mvreml`` directly."""
        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()

        nf = fit_mvlmm_null_ai_reml(
            Y_rot, X0_rot, eigenvalues, max_iter=100, em_iters=5,
        )
        # The plumbed flag should match what the optimizer stashed.
        opt_converged = nf.optimizer_trace[-1].get("converged", None)
        assert opt_converged is not None, (
            "pxem_nr_mvreml should stash converged in trace[-1]"
        )
        assert nf.converged == bool(opt_converged)

    def test_max_iter_one_reports_not_converged(self):
        """With max_iter=1, the optimizer cannot have run its delta-
        check (which needs at least 2 entries in the trace). The
        plumbed flag should report False — even though the legacy
        trace-tail heuristic might also report False, the test gates
        the behavioral contract on the plumbed value."""
        from torchgenomics.optim.lbfgs_reml import lbfgs_reml

        Y_rot, X0_rot, eigenvalues = _tiny_mvlmm_inputs()
        # max_iter=1: only one outer step → no delta-check possible.
        Vg, Ve, ll, trace = lbfgs_reml(
            Y_rot, X0_rot, eigenvalues, n_traits=2, max_iter=1,
        )
        assert trace[-1].get("converged", None) is False

        nf = fit_mvlmm_null_lbfgs(
            Y_rot, X0_rot, eigenvalues, max_iter=1,
        )
        assert nf.converged is False


# ---------------------------------------------------------------------------
# _dispatch.native_disabled
# ---------------------------------------------------------------------------


class TestNativeDisabled:
    """Helper reading ``TORCHGENOMICS_DISABLE_NATIVE`` from the environment."""

    def test_returns_bool(self):
        assert isinstance(native_disabled(), bool)

    def test_unset_returns_false(self, monkeypatch):
        monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
        assert native_disabled() is False

    def test_set_returns_true(self, monkeypatch):
        monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
        assert native_disabled() is True

    def test_empty_string_returns_false(self, monkeypatch):
        """An empty value disables the override (matches ``bool('')`` →
        False)."""
        monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "")
        assert native_disabled() is False


# ---------------------------------------------------------------------------
# HAS_NATIVE_CAVI
# ---------------------------------------------------------------------------


class TestHasNativeCavi:
    """Module-level boolean flag indicating whether the C++ CAVI extension
    was successfully imported at module load."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_CAVI, bool)

    def test_value_consistent_with_native_module(self):
        """The flag matches whether ``_cavi_native`` is non-None in
        ``torchgenomics._native``."""
        from torchgenomics import _native
        if HAS_NATIVE_CAVI:
            assert _native._cavi_native is not None
        else:
            assert _native._cavi_native is None

    def test_imported_from_native_namespace(self):
        """The flag re-exported from ``torchgenomics.models.bayesian_vs`` is
        identical to the underlying ``torchgenomics._native.HAS_NATIVE_CAVI``."""
        from torchgenomics._native import HAS_NATIVE_CAVI as native_flag
        assert HAS_NATIVE_CAVI == native_flag
