"""Single-trait optimizer controller: automatic fallback A -> B -> D -> F.

Implements the single-trait optimizer hierarchy from charter Section 13.
Modes A (PX-EM), B (AI-REML), and D (MM) are implemented here. Multi-trait
models use their dedicated optimizers instead of this controller.
"""

from __future__ import annotations

import logging
import math
from enum import Enum, auto

import torch
from torch import Tensor

from ..config import NumericalConfig
from ..linalg.eigh import compute_weights
from ..models.base import NullFit
from .ai_reml import ai_reml_single
from .em_warmstart import px_em_warmstart
from .emma_reml import emma_reml_single
from .mm_reml import mm_reml
from .reml_math import _compute_P_quantities

logger = logging.getLogger(__name__)


class OptimizerMode(Enum):
    """Optimizer modes used by the controller and related model-specific paths."""

    PX_EM = auto()        # Mode A: warm start
    AI_REML = auto()      # Mode B: primary exact (small d)
    LBFGS_AUTOGRAD = auto()  # Mode C: primary robust (general)
    MM = auto()           # Mode D: robust fallback
    PCG = auto()          # Mode E: large-scale iterative
    RESCUE = auto()       # Mode F: derivative-free rescue


class OptimizerController:
    """Manages the optimizer stack with automatic fallback.

    1. Start with Mode A (PX-EM, 3-10 iterations).
    2. Switch to Mode B (AI-REML) if d <= 6, else Mode C (LBFGS-autograd).
    3. If primary oscillates or hits boundary 3+ times, switch to Mode D (MM).
    4. If Mode D fails, try Mode F (rescue) and flag the fit.
    5. Log the optimizer trajectory for diagnostics.
    """

    def __init__(self, config: NumericalConfig | None = None) -> None:
        self.config = config or NumericalConfig()

    def fit(
        self,
        Y_rot: Tensor,
        X0_rot: Tensor,
        eigenvalues: Tensor,
        *,
        n_traits: int = 1,
        max_iter: int = 100,
        sig2_g_init: float | None = None,
        sig2_e_init: float | None = None,
    ) -> NullFit:
        """Run the optimizer stack and return the fitted null model.

        Parameters
        ----------
        Y_rot : (n,) or (n, 1) — rotated phenotype
        X0_rot : (n, c) — rotated covariates
        eigenvalues : (n,) — GRM eigenvalues
        n_traits : int — number of traits (1 for single-trait LMM)
        max_iter : int — max iterations for primary optimizer
        sig2_g_init : float, optional
            Initial genetic variance (warm-start). If provided together with
            sig2_e_init, PX-EM warm-start is skipped and these values are
            used directly as starting point for AI-REML.
        sig2_e_init : float, optional
            Initial residual variance (warm-start).

        Returns
        -------
        NullFit with variance components, weights, log-likelihood, trace
        """
        if n_traits != 1:
            raise ValueError(
                "OptimizerController is single-trait only; use MultiTraitLMM "
                "or torchgenomics.models.lmm_multi_fit for multi-trait REML."
            )

        return self._fit_single_trait(
            Y_rot, X0_rot, eigenvalues, max_iter=max_iter,
            sig2_g_init=sig2_g_init, sig2_e_init=sig2_e_init,
        )

    def _fit_single_trait(
        self,
        Y_rot: Tensor,
        X0_rot: Tensor,
        eigenvalues: Tensor,
        *,
        max_iter: int = 100,
        sig2_g_init: float | None = None,
        sig2_e_init: float | None = None,
    ) -> NullFit:
        """Single-trait optimizer pipeline."""
        tol = self.config.reml_convergence_tol

        if self.config.reml_method == "emma" and sig2_g_init is None:
            return self._fit_emma(Y_rot, X0_rot, eigenvalues, tol=tol)

        return self._fit_ai_reml_stack(
            Y_rot, X0_rot, eigenvalues, max_iter=max_iter, tol=tol,
            sig2_g_init=sig2_g_init, sig2_e_init=sig2_e_init,
        )

    def _fit_emma(
        self,
        Y_rot: Tensor,
        X0_rot: Tensor,
        eigenvalues: Tensor,
        *,
        tol: float = 1e-6,
    ) -> NullFit:
        """EMMA grid-search REML (matches GAPIT's variance component estimation)."""
        sig2_g, sig2_e, ll, full_trace = emma_reml_single(
            Y_rot, X0_rot, eigenvalues, tol=tol,
        )
        converged = True  # EMMA grid search always finds a solution

        # Build NullFit (same as _fit_ai_reml_stack finale)
        weights = compute_weights(eigenvalues, sig2_g, sig2_e)

        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (eigenvalues * lam + 1.0)
        _, beta0 = _compute_P_quantities(Y_rot, X0_rot, H_inv)

        wX = H_inv.unsqueeze(1) * X0_rot
        M00 = X0_rot.T @ wX
        b0 = X0_rot.T @ (H_inv * Y_rot.squeeze())

        return NullFit(
            sig2_g=sig2_g,
            sig2_e=sig2_e,
            eigenvalues=eigenvalues,
            Y_rot=Y_rot,
            X0_rot=X0_rot,
            M00=M00,
            b0=b0,
            weights=weights,
            log_likelihood=ll,
            optimizer_trace=full_trace,
            converged=converged,
            device=eigenvalues.device,
        )

    def _fit_ai_reml_stack(
        self,
        Y_rot: Tensor,
        X0_rot: Tensor,
        eigenvalues: Tensor,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        sig2_g_init: float | None = None,
        sig2_e_init: float | None = None,
    ) -> NullFit:
        """PX-EM → AI-REML → MM fallback → EMMA rescue stack."""
        full_trace: list[dict] = []

        if sig2_g_init is not None and sig2_e_init is not None:
            # Warm-start: skip PX-EM, use provided variance components
            if sig2_g_init <= 0 or sig2_e_init <= 0:
                logger.warning(
                    "Non-positive warm-start values (sig2_g=%.4e, sig2_e=%.4e), "
                    "falling back to cold start.",
                    sig2_g_init, sig2_e_init,
                )
                sig2_g_init = sig2_e_init = None
            elif not (math.isfinite(sig2_g_init) and math.isfinite(sig2_e_init)):
                logger.warning(
                    "Non-finite warm-start values, falling back to cold start."
                )
                sig2_g_init = sig2_e_init = None

        if sig2_g_init is not None and sig2_e_init is not None:
            sig2_g, sig2_e = sig2_g_init, sig2_e_init
            logger.info(
                "Warm-start: skipping PX-EM, using sig2_g=%.4e, sig2_e=%.4e",
                sig2_g, sig2_e,
            )
        else:
            # Cold start: PX-EM from var_y * 0.5
            em_iters = self.config.em_warmstart_iters
            y = Y_rot.squeeze()
            var_y = torch.var(y).item()
            sig2_g, sig2_e, em_trace = px_em_warmstart(
                Y_rot, X0_rot, eigenvalues,
                n_iter=em_iters,
                sig2_g_init=var_y * 0.5,
                sig2_e_init=var_y * 0.5,
            )
            full_trace.extend(em_trace)

        # Convert to lambda for AI-REML
        lam_init = sig2_g / max(sig2_e, 1e-20)

        # Track best solution across all optimizers
        best_sig2_g, best_sig2_e, best_ll = sig2_g, sig2_e, float("-inf")
        converged = False

        # --- Mode B: AI-REML (primary) ---
        try:
            sig2_g, sig2_e, ll, ai_trace = ai_reml_single(
                Y_rot, X0_rot, eigenvalues,
                lam_init=lam_init,
                max_iter=max_iter,
                tol=tol,
            )
            full_trace.extend(ai_trace)
            converged = len(ai_trace) > 0 and (
                len(ai_trace) < max_iter or
                (len(ai_trace) >= 2 and
                 abs(ai_trace[-1]["ll"] - ai_trace[-2]["ll"]) < tol * max(abs(ai_trace[-1]["ll"]), 1.0))
            )

            best_sig2_g, best_sig2_e, best_ll = sig2_g, sig2_e, ll

            if not converged:
                raise RuntimeError("AI-REML did not converge")

        except (RuntimeError, ValueError) as exc:
            logger.warning("AI-REML failed (%s), falling back to MM.", exc)

            # Use AI-REML's best lambda as MM init (not the last oscillating value)
            ai_best_lam = sig2_g / max(sig2_e, 1e-20)

            # --- Mode D: MM fallback ---
            sig2_g_mm, sig2_e_mm, ll_mm, mm_trace = mm_reml(
                Y_rot, X0_rot, eigenvalues,
                lam_init=ai_best_lam,
                max_iter=max_iter * 2,
                tol=tol,
            )
            full_trace.extend(mm_trace)

            # Keep MM result only if it's better than AI-REML's best
            if ll_mm > best_ll:
                best_sig2_g, best_sig2_e, best_ll = sig2_g_mm, sig2_e_mm, ll_mm
                converged = len(mm_trace) > 0 and len(mm_trace) < max_iter * 2
            else:
                logger.warning(
                    "MM (ll=%.4f) worse than AI-REML best (ll=%.4f), "
                    "falling back to EMMA grid search.",
                    ll_mm, best_ll,
                )
                # --- EMMA rescue ---
                sig2_g_em, sig2_e_em, ll_em, emma_trace = emma_reml_single(
                    Y_rot, X0_rot, eigenvalues, tol=tol,
                )
                full_trace.extend(emma_trace)

                if ll_em > best_ll:
                    best_sig2_g, best_sig2_e, best_ll = sig2_g_em, sig2_e_em, ll_em
                    converged = True  # EMMA always finds a solution
                # else: keep AI-REML's best

        sig2_g, sig2_e, ll = best_sig2_g, best_sig2_e, best_ll

        # Compute weights for scan
        weights = compute_weights(eigenvalues, sig2_g, sig2_e)

        # Compute null beta (fixed effects under null)
        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (eigenvalues * lam + 1.0)
        _, beta0 = _compute_P_quantities(Y_rot, X0_rot, H_inv)

        # Build null normal equations for Schur complement scan
        wX = H_inv.unsqueeze(1) * X0_rot
        M00 = X0_rot.T @ wX
        b0 = X0_rot.T @ (H_inv * Y_rot.squeeze())

        return NullFit(
            sig2_g=sig2_g,
            sig2_e=sig2_e,
            eigenvalues=eigenvalues,
            Y_rot=Y_rot,
            X0_rot=X0_rot,
            M00=M00,
            b0=b0,
            weights=weights,
            log_likelihood=ll,
            optimizer_trace=full_trace,
            converged=converged,
            device=eigenvalues.device,
        )
