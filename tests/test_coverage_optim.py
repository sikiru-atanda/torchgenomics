"""Tier-1 coverage tests for 18 ``torchgwas.optim`` symbols.

Bar (Pillar A spec, Tier 1 — V1-core, Phase 13 territory):

- Result dataclasses (``REMLResult``, ``PCGResult``): construct +
  field round-trip + ``repr`` smoke.
- Helpers (``compute_sigma_inv_diag``, ``mvlmm_null_quantities``):
  shape contract + dense closed-form reference comparison to 1e-8.
- REML / PQL fit functions (14 iterative optimizers): runs-to-completion
  + finite log-likelihood / variance components on tiny synthetic
  inputs. End-to-end numerical correctness is exercised by the higher-
  level model tests (e.g. ``tests/test_single_trait_lmm.py``,
  ``tests/test_multi_trait_lmm.py``, ``tests/test_random_regression.py``).

Note on import order
--------------------
The first non-future import must be ``import torchgwas`` so the root
package init runs first. There is a pre-existing circular import:
``optim.controller`` → ``linalg.eigh.compute_weights`` → ``models``
→ ``optim.controller``. It resolves under the normal pytest collection
order because ``models.*`` is loaded first; importing
``torchgwas.optim.*`` cold otherwise breaks collection.
"""

from __future__ import annotations

import torchgwas  # noqa: F401  - resolve circular models<->optim import
import torchgwas.models  # noqa: F401  - load models *before* optim

from dataclasses import is_dataclass

import numpy as np
import pytest
import torch

from torchgwas.optim.cox_pql import cox_pql_fit
from torchgwas.optim.emma_reml import emma_reml_single, gapit_emma_remle
from torchgwas.optim.fa_lbfgs_reml import fa_lbfgs_reml
from torchgwas.optim.fisher_scoring import fisher_scoring_reml
from torchgwas.optim.multi_env_pql import multi_env_pql_fit
from torchgwas.optim.multi_kernel_met_reml import multi_kernel_met_reml
from torchgwas.optim.mvlmm_reml import (
    compute_sigma_inv_diag,
    mvlmm_null_quantities,
)
from torchgwas.optim.pcg_solver import PCGResult
from torchgwas.optim.pxem_nr_mvreml import pxem_nr_mvreml
from torchgwas.optim.reml_math import REMLResult
from torchgwas.optim.rr_reml import (
    rr_reml_diagonal,
    rr_reml_fa,
    rr_reml_unstructured,
)
from torchgwas.optim.separable_kron_reml import separable_kron_reml
from torchgwas.optim.sparse_reml import sparse_reml_fit
from torchgwas.optim.triad_reml import triad_reml


pytestmark = pytest.mark.timeout(180)


# ---------------------------------------------------------------------------
# REMLResult dataclass
# ---------------------------------------------------------------------------


class TestRemlResult:
    """Output of a single single-trait REML evaluation."""

    def _build(self) -> REMLResult:
        return REMLResult(
            log_likelihood=-50.5,
            sig2_g=0.4,
            sig2_e=0.6,
            lam=2.0 / 3.0,
        )

    def test_is_dataclass(self):
        assert is_dataclass(REMLResult)

    def test_construct(self):
        r = self._build()
        assert r.log_likelihood == pytest.approx(-50.5)
        assert r.sig2_g == pytest.approx(0.4)
        assert r.sig2_e == pytest.approx(0.6)
        assert r.lam == pytest.approx(2.0 / 3.0)
        # default-valued fields
        assert r.n_iter == 0
        assert r.converged is False

    def test_field_round_trip(self):
        r = self._build()
        r.n_iter = 7
        r.converged = True
        r.sig2_g = 0.99
        assert r.n_iter == 7
        assert r.converged is True
        assert r.sig2_g == pytest.approx(0.99)

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "REMLResult" in s


# ---------------------------------------------------------------------------
# PCGResult dataclass
# ---------------------------------------------------------------------------


class TestPcgResult:
    """Output of a Preconditioned-Conjugate-Gradient solve."""

    def _build(self) -> PCGResult:
        return PCGResult(
            x=torch.zeros(5, dtype=torch.float64),
            n_iters=10,
            converged=True,
            residual_norm=1e-9,
        )

    def test_is_dataclass(self):
        assert is_dataclass(PCGResult)

    def test_construct(self):
        r = self._build()
        assert r.x.shape == (5,)
        assert r.n_iters == 10
        assert r.converged is True
        assert r.residual_norm == pytest.approx(1e-9)

    def test_field_round_trip(self):
        r = self._build()
        r.x = torch.ones(3, dtype=torch.float64)
        r.n_iters = 0
        r.converged = False
        r.residual_norm = 1.0
        assert r.x.shape == (3,)
        assert r.n_iters == 0
        assert r.converged is False
        assert r.residual_norm == pytest.approx(1.0)

    def test_repr_does_not_crash(self):
        s = repr(self._build())
        assert isinstance(s, str)
        assert "PCGResult" in s


# ---------------------------------------------------------------------------
# compute_sigma_inv_diag
# ---------------------------------------------------------------------------


class TestComputeSigmaInvDiag:
    """KED diagonal precision via :func:`linalg.kronecker_eed.diagonal_precision`."""

    def _ked_inputs(self, d: int = 2, E: int = 2, n: int = 5):
        from torchgwas.linalg.kronecker_eed import kronecker_eed

        # Build SPD random matrices of the requested sizes via outer-product
        g = torch.Generator().manual_seed(11)
        Z_t = torch.randn(d, d, generator=g, dtype=torch.float64)
        Vg_t = Z_t @ Z_t.T + 0.5 * torch.eye(d, dtype=torch.float64)
        Z_e = torch.randn(E, E, generator=g, dtype=torch.float64)
        Vg_e = Z_e @ Z_e.T + 0.5 * torch.eye(E, dtype=torch.float64)
        Ve_t = torch.eye(d, dtype=torch.float64)
        Ve_e = torch.eye(E, dtype=torch.float64)
        ked = kronecker_eed(Vg_t, Vg_e, Ve_t, Ve_e)
        eigenvalues = torch.linspace(0.1, 2.0, n, dtype=torch.float64)
        return ked, eigenvalues, Vg_t, Vg_e

    def test_returns_correct_shape(self):
        ked, evs, _, _ = self._ked_inputs(d=2, E=3, n=7)
        W_diag, logdet = compute_sigma_inv_diag(evs, ked)
        assert W_diag.shape == (7, 2 * 3)
        assert logdet.shape == (7,)

    def test_against_dense_reference(self):
        """Compare KED diagonal precision to dense (n, dE, dE) inverse for
        separable Kronecker covariance with Ve = I_dE."""
        from torchgwas.linalg.kronecker_eed import diagonal_precision

        ked, evs, Vg_t, Vg_e = self._ked_inputs(d=2, E=2, n=4)
        W_diag, logdet = compute_sigma_inv_diag(evs, ked)
        # diagonal_precision is the underlying impl — verify the wrapper passes through
        W_ref, logdet_ref = diagonal_precision(ked, evs)
        assert torch.allclose(W_diag, W_ref, atol=1e-10)
        assert torch.allclose(logdet, logdet_ref, atol=1e-10)

        # Cross-check against dense formula for a single eigenvalue:
        # In KED basis Sigma_i = diag(lam_K[i] * lam_t[a] * lam_e[b] + 1)
        # so the diagonal of Sigma_i^{-1} equals 1 / (lam_K[i] * lam_t[a] * lam_e[b] + 1).
        i = 2
        expected_diag = 1.0 / (evs[i] * ked.lam_t.view(-1, 1) * ked.lam_e.view(1, -1) + 1.0)
        assert torch.allclose(W_diag[i], expected_diag.reshape(-1), atol=1e-10)


# ---------------------------------------------------------------------------
# mvlmm_null_quantities
# ---------------------------------------------------------------------------


class TestMvlmmNullQuantities:
    """Pre-compute null-model quantities for the multi-trait Schur-complement scan."""

    def _inputs(self, n: int = 30, d: int = 2, c: int = 1, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        Y_rot = torch.randn(n, d, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        Vg = 0.5 * torch.eye(d, dtype=torch.float64)
        Ve = 1.0 * torch.eye(d, dtype=torch.float64)
        return Vg, Ve, Y_rot, X0_rot, evs

    def test_returns_dict_with_documented_keys(self):
        Vg, Ve, Y, X0, evs = self._inputs()
        out = mvlmm_null_quantities(Vg, Ve, Y, X0, evs)
        assert isinstance(out, dict)
        for k in ("W", "B", "M00", "M00_inv", "b0", "logdet_sigma"):
            assert k in out, f"missing key {k!r}"

    def test_returns_correct_shapes(self):
        n, d, c = 30, 2, 1
        Vg, Ve, Y, X0, evs = self._inputs(n=n, d=d, c=c)
        out = mvlmm_null_quantities(Vg, Ve, Y, X0, evs)
        assert out["W"].shape == (n, d, d)
        assert out["B"].shape == (c, d)
        assert out["M00"].shape == (c * d, c * d)
        assert out["M00_inv"].shape == (c * d, c * d)
        assert out["b0"].shape == (c * d,)
        assert out["logdet_sigma"].shape == (n,)

    def test_quantities_finite(self):
        Vg, Ve, Y, X0, evs = self._inputs()
        out = mvlmm_null_quantities(Vg, Ve, Y, X0, evs)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{k} has non-finite entries"

    def test_M00_inv_is_M00_inverse(self):
        Vg, Ve, Y, X0, evs = self._inputs()
        out = mvlmm_null_quantities(Vg, Ve, Y, X0, evs)
        prod = out["M00"] @ out["M00_inv"]
        eye = torch.eye(prod.shape[0], dtype=prod.dtype)
        assert torch.allclose(prod, eye, atol=1e-8)


# ---------------------------------------------------------------------------
# Tiny single-trait inputs for EMMA / Fisher
# ---------------------------------------------------------------------------


def _tiny_single_trait_rotated(n: int = 30, c: int = 1, seed: int = 0):
    """Build (Y_rot, X0_rot, eigenvalues) with K = I (so rotation = identity)."""
    g = torch.Generator().manual_seed(seed)
    Y_rot = torch.randn(n, generator=g, dtype=torch.float64)
    X0_rot = torch.ones(n, c, dtype=torch.float64)
    eigenvalues = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    return Y_rot, X0_rot, eigenvalues


# ---------------------------------------------------------------------------
# emma_reml_single
# ---------------------------------------------------------------------------


class TestEmmaRemlSingle:
    """EMMA-style single-trait REML: grid search + Brent refinement."""

    def test_runs_to_completion(self):
        Y, X0, evs = _tiny_single_trait_rotated()
        sig2_g, sig2_e, ll, trace = emma_reml_single(
            Y, X0, evs, n_grid=20,
        )
        assert isinstance(sig2_g, float)
        assert isinstance(sig2_e, float)
        assert isinstance(ll, float)
        assert isinstance(trace, list)
        assert len(trace) > 0

    def test_returned_quantities_finite(self):
        Y, X0, evs = _tiny_single_trait_rotated()
        sig2_g, sig2_e, ll, _ = emma_reml_single(Y, X0, evs, n_grid=20)
        assert np.isfinite(sig2_g) and sig2_g >= 0
        assert np.isfinite(sig2_e) and sig2_e > 0
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# gapit_emma_remle
# ---------------------------------------------------------------------------


class TestGapitEmmaRemle:
    """GAPIT-exact EMMA REML using restricted eigendecomposition."""

    def _inputs(self, n: int = 30, c: int = 1, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        Y = torch.randn(n, generator=g, dtype=torch.float64)
        X0 = torch.ones(n, c, dtype=torch.float64)
        # Build a simple PSD K via outer product
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64)
        K = Z @ Z.T / 5.0
        K = K + 1e-3 * torch.eye(n, dtype=torch.float64)
        return Y, X0, K

    def test_runs_to_completion(self):
        Y, X0, K = self._inputs()
        sig2_g, sig2_e, ll, delta, trace = gapit_emma_remle(Y, X0, K, n_grid=20)
        assert isinstance(sig2_g, float)
        assert isinstance(sig2_e, float)
        assert isinstance(ll, float)
        assert isinstance(delta, float)
        assert isinstance(trace, list)

    def test_returned_quantities_finite(self):
        Y, X0, K = self._inputs()
        sig2_g, sig2_e, ll, delta, _ = gapit_emma_remle(Y, X0, K, n_grid=20)
        assert np.isfinite(sig2_g) and sig2_g >= 0
        assert np.isfinite(sig2_e) and sig2_e > 0
        assert np.isfinite(ll)
        assert np.isfinite(delta) and delta > 0


# ---------------------------------------------------------------------------
# fisher_scoring_reml
# ---------------------------------------------------------------------------


class TestFisherScoringReml:
    """Single-trait Fisher-scoring REML using expected information."""

    def test_runs_to_completion(self):
        Y, X0, evs = _tiny_single_trait_rotated()
        sig2_g, sig2_e, ll = fisher_scoring_reml(Y, X0, evs, max_iter=50)
        assert isinstance(sig2_g, float)
        assert isinstance(sig2_e, float)
        assert isinstance(ll, float)

    def test_returned_quantities_finite(self):
        Y, X0, evs = _tiny_single_trait_rotated()
        sig2_g, sig2_e, ll = fisher_scoring_reml(Y, X0, evs, max_iter=50)
        assert np.isfinite(sig2_g) and sig2_g >= 0
        assert np.isfinite(sig2_e) and sig2_e > 0
        assert np.isfinite(ll)

    def test_log_likelihood_close_to_emma(self):
        """Fisher scoring and EMMA grid + Brent should converge to within
        the same neighbourhood on a well-conditioned tiny problem.

        Tolerance is observed-then-floored per spec section 4.3 (Pillar A
        T7 reviewer note): on this fixture (N=30) the reproduction shows
        |ll_fs - ll_emma| ~= 2.1e-5, well below the prior >= 1.0-unit
        floor. Tightening to 1e-4 (~5x the observed gap) catches any
        future regression that loses accuracy without being so tight as
        to flake on numerical noise."""
        Y, X0, evs = _tiny_single_trait_rotated()
        _, _, ll_fs = fisher_scoring_reml(Y, X0, evs, max_iter=100, lam_init=1.0)
        _, _, ll_emma, _ = emma_reml_single(Y, X0, evs, n_grid=50)
        assert abs(ll_fs - ll_emma) < 1e-4


# ---------------------------------------------------------------------------
# fa_lbfgs_reml
# ---------------------------------------------------------------------------


class TestFaLbfgsReml:
    """FA(k) REML via LBFGS-autograd for multi-environment LMM."""

    def _inputs(self, n: int = 40, E: int = 3, c: int = 1, seed: int = 2):
        g = torch.Generator().manual_seed(seed)
        Y_rot = torch.randn(n, E, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        return Y_rot, X0_rot, evs, E

    def test_runs_to_completion(self):
        Y, X0, evs, E = self._inputs()
        Vg, Ve, ll, trace, Lambda, psi = fa_lbfgs_reml(
            Y, X0, evs, n_envs=E, fa_rank=2, max_iter=30,
        )
        assert Vg.shape == (E, E)
        assert Ve.shape == (E, E)
        assert Lambda.shape == (E, 2)
        assert psi.shape == (E,)
        assert isinstance(ll, float)
        assert len(trace) > 0

    def test_returned_quantities_finite(self):
        Y, X0, evs, E = self._inputs()
        Vg, Ve, ll, _, Lambda, psi = fa_lbfgs_reml(
            Y, X0, evs, n_envs=E, fa_rank=1, max_iter=30,
        )
        for name, t in (("Vg", Vg), ("Ve", Ve), ("Lambda", Lambda), ("psi", psi)):
            assert torch.isfinite(t).all(), f"{name} has non-finite entries"
        assert np.isfinite(ll)

    def test_fa_rank_validation(self):
        """fa_rank >= n_envs must raise."""
        Y, X0, evs, E = self._inputs(E=2)
        with pytest.raises(ValueError):
            fa_lbfgs_reml(Y, X0, evs, n_envs=E, fa_rank=2, max_iter=5)


# ---------------------------------------------------------------------------
# multi_kernel_met_reml
# ---------------------------------------------------------------------------


class TestMultiKernelMetReml:
    """Multi-kernel multi-env REML via direct V construction + LBFGS."""

    def _inputs(self, n: int = 25, E: int = 2, c: int = 1, seed: int = 3):
        g = torch.Generator().manual_seed(seed)
        Y = torch.randn(n, E, generator=g, dtype=torch.float64)
        X0 = torch.ones(n, c, dtype=torch.float64)
        # Two SPD kernels via outer products
        Z1 = torch.randn(n, 6, generator=g, dtype=torch.float64)
        K_add = Z1 @ Z1.T / 6.0 + 0.1 * torch.eye(n, dtype=torch.float64)
        Z2 = torch.randn(n, 6, generator=g, dtype=torch.float64)
        K_dom = Z2 @ Z2.T / 6.0 + 0.1 * torch.eye(n, dtype=torch.float64)
        return Y, X0, {"add": K_add, "dom": K_dom}, E

    def test_runs_to_completion(self):
        Y, X0, kernels, E = self._inputs()
        result = multi_kernel_met_reml(
            Y, X0, kernels, n_envs=E, max_iter=20,
        )
        assert isinstance(result, dict)
        for k in ("Vg_dict", "Ve", "loglik", "trace", "V_inv", "converged"):
            assert k in result

    def test_returned_quantities_finite(self):
        Y, X0, kernels, E = self._inputs()
        result = multi_kernel_met_reml(
            Y, X0, kernels, n_envs=E, max_iter=20,
        )
        assert np.isfinite(result["loglik"])
        assert torch.isfinite(result["Ve"]).all()
        for name, Vg in result["Vg_dict"].items():
            assert torch.isfinite(Vg).all(), f"Vg[{name}] has non-finite entries"


# ---------------------------------------------------------------------------
# multi_env_pql_fit
# ---------------------------------------------------------------------------


class TestMultiEnvPqlFit:
    """Multi-environment PQL for binary GLMM."""

    def _inputs(self, n: int = 40, E: int = 2, c: int = 1, seed: int = 4):
        g = torch.Generator().manual_seed(seed)
        Y = torch.randint(0, 2, (n, E), generator=g, dtype=torch.float64)
        X0 = torch.ones(n, c, dtype=torch.float64)
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64)
        K = Z @ Z.T / 5.0 + 0.1 * torch.eye(n, dtype=torch.float64)
        # Simple per-env mean-rate initialization
        mu_init = Y.mean(dim=0, keepdim=True).expand(n, E).contiguous().clone()
        mu_init = mu_init.clamp(min=0.05, max=0.95)
        return Y, X0, K, mu_init

    def test_runs_to_completion(self):
        Y, X0, K, mu = self._inputs()
        result = multi_env_pql_fit(
            Y, X0, K, mu, family="binary",
            max_outer=5, max_inner_reml=3,
        )
        assert isinstance(result, dict)
        for k in ("mu", "beta", "Sigma_g", "Sigma_e",
                  "converged", "log_likelihood", "n_outer"):
            assert k in result

    def test_returned_quantities_finite(self):
        Y, X0, K, mu = self._inputs()
        result = multi_env_pql_fit(
            Y, X0, K, mu, family="binary",
            max_outer=5, max_inner_reml=3,
        )
        assert torch.isfinite(result["mu"]).all()
        assert torch.isfinite(result["beta"]).all()
        assert torch.isfinite(result["Sigma_g"]).all()
        assert torch.isfinite(result["Sigma_e"]).all()
        assert np.isfinite(result["log_likelihood"])


# ---------------------------------------------------------------------------
# pxem_nr_mvreml
# ---------------------------------------------------------------------------


class TestPxemNrMvreml:
    """PX-EM warm-start + AI-REML Newton-Raphson for multi-trait REML."""

    def _inputs(self, n: int = 40, d: int = 2, c: int = 1, seed: int = 5):
        g = torch.Generator().manual_seed(seed)
        Y_rot = torch.randn(n, d, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        return Y_rot, X0_rot, evs, d

    def test_runs_to_completion(self):
        Y, X0, evs, d = self._inputs()
        Vg, Ve, ll, trace = pxem_nr_mvreml(
            Y, X0, evs, n_traits=d, max_iter=30, em_iters=5,
        )
        assert Vg.shape == (d, d)
        assert Ve.shape == (d, d)
        assert isinstance(ll, float)
        assert isinstance(trace, list) and len(trace) > 0

    def test_returned_quantities_finite(self):
        Y, X0, evs, d = self._inputs()
        Vg, Ve, ll, _ = pxem_nr_mvreml(
            Y, X0, evs, n_traits=d, max_iter=30, em_iters=5,
        )
        assert torch.isfinite(Vg).all()
        assert torch.isfinite(Ve).all()
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# cox_pql_fit
# ---------------------------------------------------------------------------


class TestCoxPqlFit:
    """PQL for Cox PH frailty model."""

    def _inputs(self, n: int = 50, c: int = 1, seed: int = 6):
        g = torch.Generator().manual_seed(seed)
        # Exponential follow-up times (positive)
        time = -torch.log(torch.rand(n, generator=g, dtype=torch.float64))
        # ~50% events
        event = (torch.rand(n, generator=g, dtype=torch.float64) > 0.5).to(
            torch.float64
        )
        X0 = torch.ones(n, c, dtype=torch.float64)
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64)
        K = Z @ Z.T / 5.0 + 0.1 * torch.eye(n, dtype=torch.float64)
        return time, event, X0, K

    def test_runs_to_completion(self):
        time, event, X0, K = self._inputs()
        result = cox_pql_fit(time, event, X0, K, max_outer=5)
        assert isinstance(result, dict)
        for k in ("beta", "sig2_g", "sig2_e", "converged", "log_likelihood"):
            assert k in result

    def test_returned_quantities_finite(self):
        time, event, X0, K = self._inputs()
        result = cox_pql_fit(time, event, X0, K, max_outer=5)
        assert torch.isfinite(result["beta"]).all()
        assert np.isfinite(result["sig2_g"]) and result["sig2_g"] >= 0
        assert np.isfinite(result["sig2_e"]) and result["sig2_e"] > 0
        assert np.isfinite(result["log_likelihood"])


# ---------------------------------------------------------------------------
# rr_reml_diagonal / rr_reml_fa / rr_reml_unstructured
# ---------------------------------------------------------------------------


def _rr_inputs(n: int = 40, b: int = 3, c: int = 1, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    Y_rot = torch.randn(n, b, generator=g, dtype=torch.float64)
    X0_rot = torch.ones(n, c, dtype=torch.float64)
    evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
    return Y_rot, X0_rot, evs, b


class TestRrRemlDiagonal:
    """Random regression REML — diagonal K_coef structure."""

    def test_runs_to_completion(self):
        Y, X0, evs, b = _rr_inputs()
        Vg, Ve, ll, trace = rr_reml_diagonal(
            Y, X0, evs, b=b, n_grid=20,
        )
        assert Vg.shape == (b, b)
        assert Ve.shape == (b, b)
        assert isinstance(ll, float)
        assert isinstance(trace, list) and len(trace) > 0

    def test_returned_quantities_finite(self):
        Y, X0, evs, b = _rr_inputs()
        Vg, Ve, ll, _ = rr_reml_diagonal(Y, X0, evs, b=b, n_grid=20)
        assert torch.isfinite(Vg).all()
        assert torch.isfinite(Ve).all()
        assert np.isfinite(ll)
        # Diagonal structure: off-diagonal entries are zero.
        Vg_off = Vg - torch.diag(torch.diag(Vg))
        Ve_off = Ve - torch.diag(torch.diag(Ve))
        assert torch.allclose(Vg_off, torch.zeros_like(Vg_off), atol=1e-12)
        assert torch.allclose(Ve_off, torch.zeros_like(Ve_off), atol=1e-12)


class TestRrRemlFa:
    """Random regression REML — FA(k) K_coef structure."""

    def test_runs_to_completion(self):
        Y, X0, evs, b = _rr_inputs(b=3)
        Vg, Ve, ll, trace = rr_reml_fa(
            Y, X0, evs, b=b, fa_rank=1, max_iter=20,
        )
        assert Vg.shape == (b, b)
        assert Ve.shape == (b, b)
        assert isinstance(ll, float)
        assert isinstance(trace, list)

    def test_returned_quantities_finite(self):
        Y, X0, evs, b = _rr_inputs(b=3)
        Vg, Ve, ll, _ = rr_reml_fa(
            Y, X0, evs, b=b, fa_rank=1, max_iter=20,
        )
        assert torch.isfinite(Vg).all()
        assert torch.isfinite(Ve).all()
        assert np.isfinite(ll)

    def test_fa_rank_validation(self):
        Y, X0, evs, b = _rr_inputs(b=2)
        with pytest.raises(ValueError):
            rr_reml_fa(Y, X0, evs, b=b, fa_rank=2, max_iter=5)


class TestRrRemlUnstructured:
    """Random regression REML — unstructured K_coef passthrough to LBFGS."""

    def test_runs_to_completion(self):
        Y, X0, evs, b = _rr_inputs(b=3)
        out = rr_reml_unstructured(Y, X0, evs, b=b, max_iter=20)
        # lbfgs_reml returns a tuple — first two elements are Vg, Ve
        assert isinstance(out, tuple) and len(out) >= 2
        Vg, Ve = out[0], out[1]
        assert Vg.shape == (b, b)
        assert Ve.shape == (b, b)

    def test_returned_quantities_finite(self):
        Y, X0, evs, b = _rr_inputs(b=3)
        out = rr_reml_unstructured(Y, X0, evs, b=b, max_iter=20)
        Vg, Ve = out[0], out[1]
        assert torch.isfinite(Vg).all()
        assert torch.isfinite(Ve).all()


# ---------------------------------------------------------------------------
# separable_kron_reml
# ---------------------------------------------------------------------------


class TestSeparableKronReml:
    """Separable Kronecker Vg = Vg_t kron Vg_e via LBFGS-autograd."""

    def test_runs_to_completion(self):
        n, d, E, c = 30, 2, 2, 1
        dE = d * E
        g = torch.Generator().manual_seed(8)
        Y_rot = torch.randn(n, dE, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        Vg_t, Vg_e, Ve, ll, trace = separable_kron_reml(
            Y_rot, X0_rot, evs, n_traits=d, n_envs=E, max_iter=20,
        )
        assert Vg_t.shape == (d, d)
        assert Vg_e.shape == (E, E)
        assert Ve.shape == (dE, dE)
        assert isinstance(ll, float)
        assert isinstance(trace, list) and len(trace) > 0

    def test_returned_quantities_finite(self):
        n, d, E, c = 30, 2, 2, 1
        dE = d * E
        g = torch.Generator().manual_seed(8)
        Y_rot = torch.randn(n, dE, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        Vg_t, Vg_e, Ve, ll, _ = separable_kron_reml(
            Y_rot, X0_rot, evs, n_traits=d, n_envs=E, max_iter=20,
        )
        assert torch.isfinite(Vg_t).all()
        assert torch.isfinite(Vg_e).all()
        assert torch.isfinite(Ve).all()
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# sparse_reml_fit
# ---------------------------------------------------------------------------


class TestSparseRemlFit:
    """PCG-based REML for sparse-GRM path."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    def test_runs_on_cuda_without_scipy_tensor_leak(self):
        """Regression: sparse_reml_fit on CUDA must not leak a CUDA tensor
        into scipy.optimize.minimize_scalar.

        Bug history (2026-05-13): _eval_reml_neg2ll() returned a CUDA tensor
        because logdet_V from stochastic_logdet was on CUDA. scipy then
        raised:
            TypeError: can't convert cuda:0 device type tensor to numpy.
                       Use Tensor.cpu() to copy the tensor to host memory first.

        This test runs the sparse path on CUDA inputs end-to-end and asserts
        a real NullFit is returned. Surfaced by NA3 sparse-GRM benchmark.
        """
        device = torch.device("cuda")
        n, c = 40, 1
        g = torch.Generator(device="cpu").manual_seed(9)
        Y = torch.randn(n, generator=g, dtype=torch.float64).to(device)
        X0 = torch.ones(n, c, dtype=torch.float64, device=device)
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64).to(device)
        K_full = (Z @ Z.T) / 5.0 + 0.1 * torch.eye(n, dtype=torch.float64, device=device)

        def K_matvec(x: torch.Tensor) -> torch.Tensor:
            return K_full @ x

        K_diag = K_full.diag()
        result = sparse_reml_fit(
            Y, X0, K_matvec, K_diag, n=n,
            n_probes=5, lanczos_iters=15, pcg_max_iter=200, seed=42,
        )
        from torchgwas.models.base import NullFit
        assert isinstance(result, NullFit)
        # Sanity: REML scalar estimates are finite Python floats, not tensors
        assert isinstance(result.sig2_g, float) and np.isfinite(result.sig2_g)
        assert isinstance(result.sig2_e, float) and np.isfinite(result.sig2_e)

    def test_runs_to_completion(self):
        n, c = 40, 1
        g = torch.Generator().manual_seed(9)
        Y = torch.randn(n, generator=g, dtype=torch.float64)
        X0 = torch.ones(n, c, dtype=torch.float64)
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64)
        K_full = Z @ Z.T / 5.0 + 0.1 * torch.eye(n, dtype=torch.float64)

        def K_matvec(x: torch.Tensor) -> torch.Tensor:
            return K_full @ x

        K_diag = K_full.diag()
        result = sparse_reml_fit(
            Y, X0, K_matvec, K_diag, n=n,
            n_probes=5, lanczos_iters=15, pcg_max_iter=200, seed=42,
        )
        from torchgwas.models.base import NullFit
        assert isinstance(result, NullFit)

    def test_returned_quantities_finite(self):
        n, c = 40, 1
        g = torch.Generator().manual_seed(9)
        Y = torch.randn(n, generator=g, dtype=torch.float64)
        X0 = torch.ones(n, c, dtype=torch.float64)
        Z = torch.randn(n, 5, generator=g, dtype=torch.float64)
        K_full = Z @ Z.T / 5.0 + 0.1 * torch.eye(n, dtype=torch.float64)

        def K_matvec(x: torch.Tensor) -> torch.Tensor:
            return K_full @ x

        K_diag = K_full.diag()
        result = sparse_reml_fit(
            Y, X0, K_matvec, K_diag, n=n,
            n_probes=5, lanczos_iters=15, pcg_max_iter=200, seed=42,
        )
        assert np.isfinite(result.sig2_g) and result.sig2_g >= 0
        assert np.isfinite(result.sig2_e) and result.sig2_e > 0
        assert np.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# triad_reml
# ---------------------------------------------------------------------------


class TestTriadReml:
    """Trust-Region Inexact Autograd-Differentiated REML for multi-trait."""

    def _inputs(self, n: int = 40, d: int = 2, c: int = 1, seed: int = 10):
        g = torch.Generator().manual_seed(seed)
        Y_rot = torch.randn(n, d, generator=g, dtype=torch.float64)
        X0_rot = torch.ones(n, c, dtype=torch.float64)
        evs = torch.linspace(0.5, 2.0, n, dtype=torch.float64)
        return Y_rot, X0_rot, evs, d

    def test_runs_to_completion(self):
        Y, X0, evs, d = self._inputs()
        Vg, Ve, ll, trace = triad_reml(
            Y, X0, evs, n_traits=d, max_iter=20, em_warmstart=2,
        )
        assert Vg.shape == (d, d)
        assert Ve.shape == (d, d)
        assert isinstance(ll, float)
        assert isinstance(trace, list)

    def test_returned_quantities_finite(self):
        Y, X0, evs, d = self._inputs()
        Vg, Ve, ll, _ = triad_reml(
            Y, X0, evs, n_traits=d, max_iter=20, em_warmstart=2,
        )
        assert torch.isfinite(Vg).all()
        assert torch.isfinite(Ve).all()
        assert np.isfinite(ll)
