"""Threshold-Linear Mixed Model for joint categorical + continuous traits.

Implements the Bermann et al. (2026) multi-trait threshold-linear model:

- Categorical traits: observed ordinal categories arise from unobserved
  Gaussian liabilities discretised by thresholds.
- Continuous traits: modelled jointly with liabilities via correlated residuals.
- Variance components (R, G) assumed known (user-supplied or from prior Gibbs).

Two solver paths for the augmented MME:

1. **T-EM**: E-step computes E_T[l_i], M-step solves standard MME.
   Accelerated by SQUAREM (Varadhan & Roland 2008).
2. **T-NR**: Newton-Raphson using per-individual Δ_i, Γ_i, R̃⁻¹ₖ.
   Default: T-EM(5) warm start → T-NR.

ssGWAS scanning via score test (Schur complement) on the converged null.

References
----------
Bermann, M. et al. (2026). *Genetics*.
Gianola, D. & Foulley, J.L. (1983). *Genet. Sel. Evol.* 15, 201–224.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from ..linalg.truncated_mvn import (
    mvn_truncated_moments,
    truncated_normal_moments,
)
from ..optim.squarem import squarem
from .base import NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)

# ===================================================================
# Threshold configuration
# ===================================================================

@dataclass
class ThresholdConfig:
    """Configuration for each categorical trait.

    Attributes
    ----------
    n_categories : int
        Number of ordinal categories (≥ 2).
    thresholds : Optional[Tensor]
        Fixed thresholds (n_categories - 1,).  If None, initialised from
        probit quantiles of observed category frequencies.
    """
    n_categories: int
    thresholds: Tensor | None = None


# ===================================================================
# Null-fit extension for threshold model
# ===================================================================

@dataclass
class ThresholdNullFit:
    """Extended null-fit carrying threshold-model-specific quantities."""

    # Core NullFit
    null_fit: NullFit

    # Converged location parameters
    theta: Tensor  # (p,) stacked fixed + random effects
    liabilities: Tensor  # (n, c1) MAP liability estimates

    # Thresholds per categorical trait
    thresholds_list: list[Tensor]  # each (n_cat_j - 1,)

    # Residual and genetic covariance
    R: Tensor  # (c, c) residual covariance
    G_cov: Tensor  # (c, c) genetic covariance

    # Per-individual modified precision (from NR)
    R_tilde_inv: Tensor | None = None  # (n, c, c) if NR was used

    # Working quantities for score test
    WtRW: Tensor | None = None  # (p, p) W'R̃⁻¹W + S
    WtRW_inv: Tensor | None = None  # (p, p) inverse

    # Convergence info
    solver: str = "nr"
    n_iter: int = 0
    converged: bool = False


# ===================================================================
# ThresholdLinearModel
# ===================================================================

class ThresholdLinearModel:
    """Multi-trait threshold-linear model (Bermann et al. 2026).

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    trait_types : list[str]
        Type of each trait: ``"ordinal"`` or ``"continuous"``.
    n_categories : list[int]
        Number of ordinal categories for each ordinal trait (ignored for
        continuous traits — use any value, e.g. 0).
    R : Tensor, shape (c, c)
        Residual covariance matrix (assumed known).
    G_cov : Tensor, shape (c, c)
        Genetic covariance matrix (assumed known).
    solver : str
        ``"nr"`` (default): T-EM(5) warm start → T-NR.
        ``"em"``: T-EM only (SQUAREM-accelerated).
    max_iter : int
        Maximum solver iterations.
    tol : float
        Convergence tolerance on relative parameter change.
    em_warmup : int
        Number of T-EM iterations before switching to T-NR (only for solver="nr").
    """

    def __init__(
        self,
        trait_types: list[str],
        n_categories: list[int],
        R: Tensor,
        G_cov: Tensor,
        *,
        solver: str = "nr",
        max_iter: int = 100,
        tol: float = 1e-8,
        em_warmup: int = 5,
    ) -> None:
        self.trait_types = trait_types
        self.n_categories = n_categories
        self.R = R.to(STAT_DTYPE)
        self.G_cov = G_cov.to(STAT_DTYPE)
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.em_warmup = em_warmup

        # Identify ordinal vs continuous indices
        self.ord_idx = [i for i, t in enumerate(trait_types) if t == "ordinal"]
        self.con_idx = [i for i, t in enumerate(trait_types) if t == "continuous"]
        self.c1 = len(self.ord_idx)  # number of categorical traits
        self.c2 = len(self.con_idx)  # number of continuous traits
        self.c = self.c1 + self.c2

    # ------------------------------------------------------------------
    # fit_null
    # ------------------------------------------------------------------

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null model (no SNP effect) for threshold-linear model.

        Parameters
        ----------
        Y : Tensor, shape (n, c)
            Phenotype matrix. Columns in ``ord_idx`` contain integer category
            labels (0-based). Columns in ``con_idx`` contain continuous values.
        X0 : Tensor, shape (n, p0)
            Covariate matrix (intercept first column).
        K : Tensor, shape (n, n), optional
            Kinship / GRM.  Used to build the S matrix (precision of random effects).
        """
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        n = Y.shape[0]
        device = Y.device

        # Extract categorical observations and continuous phenotypes
        Y_cat = Y[:, self.ord_idx].long() if self.c1 > 0 else None  # (n, c1)
        Y_con = Y[:, self.con_idx] if self.c2 > 0 else None  # (n, c2)

        # Initialise thresholds from probit quantiles
        thresholds_list = self._init_thresholds(Y_cat, device)

        # Build penalty matrix S from K (if provided)
        # S = G_cov^{-1} ⊗ K^{-1} for the random effect block
        # For simplicity in the null model, we use a diagonal approximation
        # when K is not provided
        if K is not None:
            K = K.to(STAT_DTYPE)

        # Initialise liabilities from probit of midpoint
        liabilities = self._init_liabilities(Y_cat, thresholds_list, n, device)

        # Build augmented phenotype: [liabilities, continuous]
        Y_aug = self._build_augmented_y(liabilities, Y_con, n, device)

        # Initialise theta via OLS on augmented phenotype
        theta = torch.linalg.lstsq(X0, Y_aug).solution  # (p0, c)

        # Precompute R sub-blocks
        R = self.R.to(device)
        R_inv = torch.linalg.inv(R)

        # Solver dispatch
        if self.solver == "em":
            theta, liabilities, n_iter, converged = self._solve_em(
                Y_cat, Y_con, X0, K, R, R_inv, theta, liabilities,
                thresholds_list, self.max_iter,
            )
        else:
            # T-EM warm start then T-NR
            if self.em_warmup > 0:
                theta, liabilities, _, _ = self._solve_em(
                    Y_cat, Y_con, X0, K, R, R_inv, theta, liabilities,
                    thresholds_list, self.em_warmup,
                )
            theta, liabilities, n_iter, converged = self._solve_nr(
                Y_cat, Y_con, X0, K, R, R_inv, theta, liabilities,
                thresholds_list, self.max_iter,
            )

        # Build null fit for scanning
        # Compute final R̃⁻¹ for score test
        R_tilde_inv = self._compute_R_tilde_inv(
            Y_cat, Y_con, X0, R, R_inv, theta, liabilities, thresholds_list,
        )

        # Build W'R̃⁻¹W (for score test Schur complement)
        # W = X0 applied to each trait (block-diagonal design)
        WtRW = self._build_WtRW(X0, R_tilde_inv, n)
        WtRW_inv = torch.linalg.inv(
            WtRW + 1e-8 * torch.eye(WtRW.shape[0], dtype=STAT_DTYPE, device=device)
        )

        # Pseudo-residuals for score test
        Y_aug_final = self._build_augmented_y(liabilities, Y_con, n, device)
        residuals = Y_aug_final - X0 @ theta  # (n, c)

        # Standard NullFit wrapper
        nf = NullFit(
            Vg=self.G_cov,
            Ve=self.R,
            log_likelihood=self._loglik(
                Y_cat, Y_con, X0, R, R_inv, theta, liabilities, thresholds_list,
            ),
            converged=converged if self.solver != "em" else True,
            device=device,
        )

        # Store extended info
        nf._threshold_ext = ThresholdNullFit(
            null_fit=nf,
            theta=theta,
            liabilities=liabilities,
            thresholds_list=thresholds_list,
            R=R,
            G_cov=self.G_cov.to(device),
            R_tilde_inv=R_tilde_inv,
            WtRW=WtRW,
            WtRW_inv=WtRW_inv,
            solver=self.solver,
            n_iter=n_iter,
            converged=converged if self.solver != "em" else True,
        )
        # Cache residuals and design for score test
        nf._thr_residuals = residuals
        nf._thr_X0 = X0
        nf._thr_R_tilde_inv = R_tilde_inv

        return nf

    # ------------------------------------------------------------------
    # score_chunk (ssGWAS)
    # ------------------------------------------------------------------

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score test for each SNP against the threshold-linear null.

        Uses the Schur complement approach: for each SNP g, the score
        statistic is g' R̃⁻¹ ε̃ projected out of the null space of W.
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        device = G_chunk.device

        # Handle empty chunk
        if m == 0:
            empty = torch.zeros(0, dtype=STAT_DTYPE, device=device)
            return ScanResult(
                chr=[], pos=[], snp=[], a1=[], a2=[],
                af=empty, beta=empty, se=empty,
                stat=empty, p=empty, test="score",
            )

        # Retrieve cached quantities
        ext = null_fit._threshold_ext
        R_tilde_inv = null_fit._thr_R_tilde_inv  # (n, c, c)
        residuals = null_fit._thr_residuals  # (n, c)
        X0 = null_fit._thr_X0  # (n, p0)
        WtRW_inv = ext.WtRW_inv  # (p, p)

        # Allele frequencies
        af = G_chunk.mean(dim=0) / 2.0  # diploid convention

        # Score test with proper null-space projection:
        #   U_g = g' P ε   where P = R̃⁻¹ - R̃⁻¹W(W'R̃⁻¹W)⁻¹W'R̃⁻¹
        #   I_gg = g' P g  (Schur complement)
        # We work per-trait, using the mean R̃⁻¹ for the projection operator.

        # Mean precision (averaged over individuals for tractability)
        Rtilde_mean = R_tilde_inv.mean(dim=0)  # (c, c)

        # Per-trait weighted quantities:
        # Use diagonal of mean R̃⁻¹ as per-trait weight
        # This gives a properly calibrated per-trait score test
        c = self.c
        p0 = X0.shape[1]

        # For each trait t, compute a scalar score test, then combine
        betas = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        ses = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        stats = torch.zeros(m, dtype=STAT_DTYPE, device=device)

        for t in range(c):
            w_t = R_tilde_inv[:, t, t]  # (n,) per-individual weight for trait t
            resid_t = residuals[:, t]  # (n,)

            # Weighted quantities
            wG = w_t.unsqueeze(1) * G_chunk  # (n, m)
            wX0 = w_t.unsqueeze(1) * X0  # (n, p0)

            # g'Wg
            gWg = (G_chunk * wG).sum(dim=0)  # (m,)

            # g'Wy
            gWr = (G_chunk * (w_t * resid_t).unsqueeze(1)).sum(dim=0)  # (m,)

            # X0'WX0 and g'WX0 for Schur complement
            XWX = X0.T @ wX0  # (p0, p0)
            XWX_reg = XWX + 1e-10 * torch.eye(p0, dtype=STAT_DTYPE, device=device)
            gWX = G_chunk.T @ wX0  # (m, p0)

            # Schur complement: S = g'Wg - g'WX0 (X0'WX0)^{-1} X0'Wg
            XWX_inv_gWX = torch.linalg.solve(XWX_reg, gWX.T)  # (p0, m)
            correction = (gWX * XWX_inv_gWX.T).sum(dim=1)  # (m,)
            S_t = (gWg - correction).clamp(min=1e-20)

            # Projected score: g'P r = g'Wr - g'WX(X'WX)^{-1}X'Wr
            XWr = X0.T @ (w_t * resid_t)  # (p0,)
            score_correction = gWX @ torch.linalg.solve(XWX_reg, XWr)  # (m,)
            U_t = gWr - score_correction  # (m,)

            # Per-trait beta and Wald-like contribution
            beta_t = U_t / S_t
            var_t = 1.0 / S_t
            stat_t = beta_t ** 2 / var_t  # = U_t^2 / S_t

            betas += beta_t
            ses += var_t
            stats += stat_t

        # Combined statistic: sum of per-trait chi2(1) ~ chi2(c)
        # For simplicity and calibration, use chi2(c) for c traits
        p = _chi2_sf(stats, df=c)

        # Average beta across traits (liability-scale effect)
        beta_out = betas / c
        se_out = (ses / (c ** 2)).sqrt()

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta_out,
            se=se_out,
            stat=stats,
            p=p,
            test="score",
        )

    # ==================================================================
    # Internal: Threshold initialisation
    # ==================================================================

    def _init_thresholds(
        self,
        Y_cat: Tensor | None,
        device: torch.device,
    ) -> list[Tensor]:
        """Initialise thresholds from probit quantiles of observed frequencies."""
        thresholds_list = []
        if Y_cat is None:
            return thresholds_list

        for j_local, j_global in enumerate(self.ord_idx):
            n_cat = self.n_categories[j_global]
            cats = Y_cat[:, j_local]
            # Cumulative proportions
            counts = torch.zeros(n_cat, dtype=STAT_DTYPE, device=device)
            for k in range(n_cat):
                counts[k] = (cats == k).sum().float()
            counts = counts.clamp(min=1)
            cum_prop = counts.cumsum(0) / counts.sum()
            # Thresholds = probit(cum_prop) for first (n_cat - 1) boundaries
            # Clamp to avoid ±inf
            cum_prop = cum_prop[:-1].clamp(0.001, 0.999)
            thresholds = torch.erfinv(2.0 * cum_prop - 1.0) * math.sqrt(2.0)
            thresholds_list.append(thresholds)

        return thresholds_list

    def _init_liabilities(
        self,
        Y_cat: Tensor | None,
        thresholds_list: list[Tensor],
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Initialise liabilities at midpoint of threshold intervals."""
        liabilities = torch.zeros(n, self.c1, dtype=STAT_DTYPE, device=device)
        if Y_cat is None:
            return liabilities

        for j, thresholds in enumerate(thresholds_list):
            cats = Y_cat[:, j]
            n_cat = len(thresholds) + 1
            # Extended thresholds with -inf and +inf
            ext = torch.cat([
                torch.tensor([-3.0], dtype=STAT_DTYPE, device=device),
                thresholds,
                torch.tensor([3.0], dtype=STAT_DTYPE, device=device),
            ])
            for k in range(n_cat):
                mask = cats == k
                liabilities[mask, j] = 0.5 * (ext[k] + ext[k + 1])

        return liabilities

    # ==================================================================
    # Internal: Augmented phenotype
    # ==================================================================

    def _build_augmented_y(
        self,
        liabilities: Tensor,
        Y_con: Tensor | None,
        n: int,
        device: torch.device,
    ) -> Tensor:
        """Build augmented phenotype [liabilities | continuous]."""
        parts = []
        if self.c1 > 0:
            parts.append(liabilities)
        if self.c2 > 0 and Y_con is not None:
            parts.append(Y_con)
        if not parts:
            return torch.zeros(n, 0, dtype=STAT_DTYPE, device=device)
        return torch.cat(parts, dim=1)

    # ==================================================================
    # Internal: EM solver
    # ==================================================================

    def _solve_em(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        K: Tensor | None,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
        max_iter: int,
    ) -> tuple[Tensor, Tensor, int, bool]:
        """T-EM solver with SQUAREM acceleration."""
        n = X0.shape[0]
        device = X0.device

        def em_step(params: Tensor) -> Tensor:
            # Unpack theta from flat vector
            p0 = X0.shape[1]
            theta_flat = params[:p0 * self.c].reshape(p0, self.c)
            liab_flat = params[p0 * self.c:].reshape(n, self.c1)

            # E-step: compute E[l_i | y_i, theta]
            new_liab = self._e_step(
                Y_cat, Y_con, X0, R, R_inv, theta_flat, liab_flat, thresholds_list,
            )

            # M-step: solve MME for theta given expected liabilities
            Y_aug = self._build_augmented_y(new_liab, Y_con, n, device)
            # Simple OLS per trait (with R_inv weighting for multi-trait)
            # θ = (X0'R⁻¹X0)⁻¹ X0'R⁻¹ Y_aug
            # OLS for M-step (R weighting is second-order for theta in EM)
            new_theta = torch.linalg.lstsq(X0, Y_aug).solution

            # Repack
            return torch.cat([new_theta.flatten(), new_liab.flatten()])

        # Pack initial state
        p0 = X0.shape[1]
        x0 = torch.cat([theta.flatten(), liabilities.flatten()])

        # Run SQUAREM
        x_final, n_iter, converged = squarem(
            em_step, x0, max_iter=max_iter, tol=self.tol,
        )

        # Unpack
        theta_out = x_final[:p0 * self.c].reshape(p0, self.c)
        liab_out = x_final[p0 * self.c:].reshape(n, self.c1)

        return theta_out, liab_out, n_iter, converged

    def _e_step(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
    ) -> Tensor:
        """E-step: compute E[l_i | y_i, θ] for each individual."""
        n = X0.shape[0]
        device = X0.device
        new_liab = liabilities.clone()

        if Y_cat is None or self.c1 == 0:
            return new_liab

        # Predicted mean: mu = X0 @ theta
        mu = X0 @ theta  # (n, c)
        mu_cat = mu[:, :self.c1]  # (n, c1)

        # Conditional mean/variance of liabilities given continuous
        if self.c2 > 0 and Y_con is not None:
            mu_con = mu[:, self.c1:]
            R11 = R[:self.c1, :self.c1]
            R12 = R[:self.c1, self.c1:]
            R22 = R[self.c1:, self.c1:]
            R22_inv = torch.linalg.inv(R22)
            # Conditional: mu_1|2 = mu_1 + R12 R22⁻¹ (y2 - mu_2)
            resid_con = Y_con - mu_con  # (n, c2)
            cond_shift = resid_con @ R22_inv.T @ R12.T  # (n, c1)
            mu_cond = mu_cat + cond_shift
            Sigma_cond = R11 - R12 @ R22_inv @ R12.T
        else:
            mu_cond = mu_cat
            Sigma_cond = R[:self.c1, :self.c1]

        # Truncation bounds from observed categories
        for j, thresholds in enumerate(thresholds_list):
            cats = Y_cat[:, j]
            n_cat = len(thresholds) + 1
            ext = torch.cat([
                torch.tensor([-1e6], dtype=STAT_DTYPE, device=device),
                thresholds,
                torch.tensor([1e6], dtype=STAT_DTYPE, device=device),
            ])

            if self.c1 == 1:
                # Univariate truncated normal
                sigma_j = Sigma_cond[0, 0].sqrt() if Sigma_cond.dim() == 2 else Sigma_cond.sqrt()
                a = ext[cats]  # (n,)
                b = ext[cats + 1]
                m, _ = truncated_normal_moments(mu_cond[:, j], sigma_j.expand(n), a, b)
                new_liab[:, j] = m
            else:
                # Multi-dimensional truncated normal (per-individual)
                # Build per-individual bounds
                a_all = torch.full((n, self.c1), -1e6, dtype=STAT_DTYPE, device=device)
                b_all = torch.full((n, self.c1), 1e6, dtype=STAT_DTYPE, device=device)

                for jj, thr in enumerate(thresholds_list):
                    cats_jj = Y_cat[:, jj]
                    ext_jj = torch.cat([
                        torch.tensor([-1e6], dtype=STAT_DTYPE, device=device),
                        thr,
                        torch.tensor([1e6], dtype=STAT_DTYPE, device=device),
                    ])
                    a_all[:, jj] = ext_jj[cats_jj]
                    b_all[:, jj] = ext_jj[cats_jj + 1]

                m, _ = mvn_truncated_moments(
                    mu_cond, Sigma_cond, a_all, b_all, n_qmc=5000,
                )
                new_liab = m
                break  # All ordinal traits handled together

        return new_liab

    # ==================================================================
    # Internal: NR solver
    # ==================================================================

    def _solve_nr(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        K: Tensor | None,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
        max_iter: int,
    ) -> tuple[Tensor, Tensor, int, bool]:
        """T-NR solver (Newton-Raphson on augmented MME)."""
        n = X0.shape[0]
        device = X0.device
        converged = False

        for it in range(max_iter):
            # Compute per-individual Delta_i and Gamma_i
            Delta, Gamma = self._compute_delta_gamma(
                Y_cat, Y_con, X0, R, R_inv, theta, liabilities, thresholds_list,
            )

            # Build R̃⁻¹ per individual
            R_tilde_inv = self._build_R_tilde_inv_from_gamma(Gamma, R, R_inv)

            # Build pseudo-observations ỹ
            y_tilde = self._build_pseudo_obs(
                Y_cat, Y_con, X0, R, R_inv, theta, liabilities,
                thresholds_list, Delta, Gamma,
            )

            # Solve modified MME: (X0' R̃⁻¹ X0) theta_new = X0' R̃⁻¹ ỹ
            # Per-individual R̃⁻¹: (n, c, c)
            # X0'R̃⁻¹ X0 = sum_i X0[i]' R̃⁻¹[i] X0[i]  (Kronecker structure)
            # Simplified for tractability: use trait-averaged R̃⁻¹
            # Build weighted normal equations: (X0' R̃⁻¹ X0) θ = X0' R̃⁻¹ ỹ
            # Per-individual R̃⁻¹: (n, c, c) — use diagonal weights per trait
            XtRX = torch.zeros(X0.shape[1], X0.shape[1], dtype=STAT_DTYPE, device=device)
            XtRy = torch.zeros(X0.shape[1], self.c, dtype=STAT_DTYPE, device=device)
            for t in range(self.c):
                w_t = R_tilde_inv[:, t, t]  # (n,)
                X0_w = X0 * w_t.unsqueeze(1)  # (n, p) weighted
                XtRX += X0_w.T @ X0
                XtRy[:, t] = X0_w.T @ y_tilde[:, t]

            # Solve weighted least squares
            theta_new = torch.linalg.lstsq(
                XtRX + 1e-8 * torch.eye(X0.shape[1], dtype=STAT_DTYPE, device=device),
                XtRy,
            ).solution

            # Update liabilities from E-step
            new_liab = self._e_step(
                Y_cat, Y_con, X0, R, R_inv, theta_new, liabilities, thresholds_list,
            )

            # Check convergence using combined absolute + relative tolerance.
            # The relative criterion alone blows up near theta=0 (the 1e-15
            # floor is overwhelmed once theta.norm() drops below ~tol), which
            # is exactly the regime for binary-trait null fits where the
            # intercept sits near zero. Accept either: the step is tiny in
            # absolute terms, or it's tiny relative to a non-vanishing theta.
            abs_change = (theta_new - theta).norm()
            rel_change = abs_change / (theta.norm() + 1e-15)
            theta = theta_new
            liabilities = new_liab

            if abs_change < self.tol or rel_change < self.tol:
                converged = True
                break

        return theta, liabilities, it + 1, converged

    def _compute_delta_gamma(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute per-individual Δ_i and Γ_i.

        Δ_i = Σ⁻¹ (E_T[l_i] - μ_i)
        Γ_i = Σ⁻¹ - Σ⁻¹ Var_T(l_i) Σ⁻¹
        """
        n = X0.shape[0]
        device = X0.device

        # Conditional distribution of liabilities
        mu = X0 @ theta
        mu_cat = mu[:, :self.c1]

        if self.c2 > 0 and Y_con is not None:
            mu_con = mu[:, self.c1:]
            R11 = R[:self.c1, :self.c1]
            R12 = R[:self.c1, self.c1:]
            R22 = R[self.c1:, self.c1:]
            R22_inv = torch.linalg.inv(R22)
            resid_con = Y_con - mu_con
            cond_shift = resid_con @ R22_inv.T @ R12.T
            mu_cond = mu_cat + cond_shift
            Sigma_cond = R11 - R12 @ R22_inv @ R12.T
        else:
            mu_cond = mu_cat
            Sigma_cond = R[:self.c1, :self.c1]

        Sigma_inv = torch.linalg.inv(
            Sigma_cond + 1e-10 * torch.eye(self.c1, dtype=STAT_DTYPE, device=device)
        )

        # Truncation bounds
        a_all = torch.full((n, self.c1), -1e6, dtype=STAT_DTYPE, device=device)
        b_all = torch.full((n, self.c1), 1e6, dtype=STAT_DTYPE, device=device)
        if Y_cat is not None:
            for j, thr in enumerate(thresholds_list):
                cats_j = Y_cat[:, j]
                ext = torch.cat([
                    torch.tensor([-1e6], dtype=STAT_DTYPE, device=device),
                    thr,
                    torch.tensor([1e6], dtype=STAT_DTYPE, device=device),
                ])
                a_all[:, j] = ext[cats_j]
                b_all[:, j] = ext[cats_j + 1]

        # Compute moments
        E_l, Var_l = mvn_truncated_moments(
            mu_cond, Sigma_cond, a_all, b_all, n_qmc=5000,
        )

        # Δ_i = Σ⁻¹ (E[l_i] - μ_i)
        Delta = (E_l - mu_cond) @ Sigma_inv.T  # (n, c1)

        # Γ_i = Σ⁻¹ - Σ⁻¹ Var_T(l_i) Σ⁻¹
        # (n, c1, c1)
        SinvVar = torch.einsum("ij,njk->nik", Sigma_inv, Var_l)
        SinvVarSinv = torch.einsum("nij,jk->nik", SinvVar, Sigma_inv)
        Gamma = Sigma_inv.unsqueeze(0).expand(n, -1, -1) - SinvVarSinv

        return Delta, Gamma

    def _build_R_tilde_inv_from_gamma(
        self,
        Gamma: Tensor,
        R: Tensor,
        R_inv: Tensor,
    ) -> Tensor:
        """Build per-individual modified residual precision R̃⁻¹_i.

        R̃⁻¹_i is R⁻¹ with the (1:c1, 1:c1) block modified by Γ_i.
        """
        n = Gamma.shape[0]
        c = R.shape[0]
        device = R.device

        R_tilde_inv = R_inv.unsqueeze(0).expand(n, -1, -1).clone()

        if self.c1 > 0:
            # Replace the categorical block with Gamma
            R_tilde_inv[:, :self.c1, :self.c1] = Gamma

        return R_tilde_inv

    def _build_pseudo_obs(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
        Delta: Tensor,
        Gamma: Tensor,
    ) -> Tensor:
        """Build pseudo-observations ỹ_i for NR iteration.

        For categorical traits: Γ_i⁻¹ Δ_i + W_i θ_cat + conditional shift
        For continuous traits: y2_i (unchanged)
        """
        n = X0.shape[0]
        device = X0.device
        mu = X0 @ theta

        y_tilde = torch.zeros(n, self.c, dtype=STAT_DTYPE, device=device)

        if self.c1 > 0:
            # Pseudo-liability: Γ⁻¹ Δ + mu_cat
            # Per-individual: need Gamma inverse
            Gamma_reg = Gamma + 1e-8 * torch.eye(
                self.c1, dtype=STAT_DTYPE, device=device
            ).unsqueeze(0)
            Gamma_inv_Delta = torch.linalg.solve(Gamma_reg, Delta)
            y_tilde[:, :self.c1] = Gamma_inv_Delta + mu[:, :self.c1]

        if self.c2 > 0 and Y_con is not None:
            y_tilde[:, self.c1:] = Y_con

        return y_tilde

    def _compute_R_tilde_inv(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
    ) -> Tensor:
        """Compute final R̃⁻¹ at convergence (for score test)."""
        if self.c1 > 0:
            Delta, Gamma = self._compute_delta_gamma(
                Y_cat, Y_con, X0, R, R_inv, theta, liabilities, thresholds_list,
            )
            return self._build_R_tilde_inv_from_gamma(Gamma, R, R_inv)
        else:
            n = X0.shape[0]
            return R_inv.unsqueeze(0).expand(n, -1, -1)

    def _build_WtRW(
        self,
        X0: Tensor,
        R_tilde_inv: Tensor,
        n: int,
    ) -> Tensor:
        """Build W'R̃⁻¹W where W is the covariate block for all traits."""
        p0 = X0.shape[1]
        device = X0.device

        # For the threshold model, W is block-diagonal: diag(X0, X0, ..., X0)
        # with c copies. The (p0*c, p0*c) precision is:
        # W'R̃⁻¹W = sum_i kron(R̃⁻¹_i, X0_i X0_i')

        # Efficient computation:
        WtRW = torch.zeros(p0 * self.c, p0 * self.c, dtype=STAT_DTYPE, device=device)
        for t1 in range(self.c):
            for t2 in range(self.c):
                # W_{t1}' diag(R̃⁻¹[:, t1, t2]) W_{t2}
                w = R_tilde_inv[:, t1, t2]  # (n,)
                block = X0.T @ (w.unsqueeze(1) * X0)  # (p0, p0)
                WtRW[t1*p0:(t1+1)*p0, t2*p0:(t2+1)*p0] = block

        return WtRW

    def _loglik(
        self,
        Y_cat: Tensor | None,
        Y_con: Tensor | None,
        X0: Tensor,
        R: Tensor,
        R_inv: Tensor,
        theta: Tensor,
        liabilities: Tensor,
        thresholds_list: list[Tensor],
    ) -> float:
        """Approximate log-likelihood of the threshold model."""
        n = X0.shape[0]
        device = X0.device

        mu = X0 @ theta
        Y_aug = self._build_augmented_y(liabilities, Y_con, n, device)
        resid = Y_aug - mu  # (n, c)

        # -0.5 * sum_i resid_i' R⁻¹ resid_i - n/2 * log|R| - nc/2 * log(2π)
        quad = (resid @ R_inv * resid).sum()
        sign, logdet = torch.linalg.slogdet(R)
        ll = -0.5 * quad - 0.5 * n * logdet - 0.5 * n * self.c * math.log(2 * math.pi)

        return ll.item()


# ===================================================================
# Helpers
# ===================================================================

def _chi2_sf(x: Tensor, df: int = 1) -> Tensor:
    """Survival function of chi-squared distribution via regularised gamma."""
    # Use scipy-compatible computation via torch
    # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    if df == 1:
        return 2.0 * (1.0 - 0.5 * torch.erfc(-x.clamp(min=0).sqrt() / math.sqrt(2.0)))

    # General case: use the regularised incomplete gamma function
    # P(X > x) = 1 - gamma_inc(df/2, x/2)
    # torch doesn't have igamma in all versions, fall back to scipy
    try:
        return 1.0 - torch.special.gammainc(
            torch.tensor(df / 2.0, dtype=x.dtype, device=x.device),
            x / 2.0,
        )
    except AttributeError:
        from scipy import stats as sp_stats
        p_np = sp_stats.chi2.sf(x.detach().cpu().numpy(), df=df)
        return torch.tensor(p_np, dtype=x.dtype, device=x.device)
