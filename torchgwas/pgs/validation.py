"""PGS validation metrics.

Evaluate polygenic score predictive performance in a target sample.
Supports both continuous traits (R², incremental R², Pearson/Spearman
correlation) and binary traits (AUC, Nagelkerke R², liability-scale R²).

All metrics operate on pre-computed PGS vectors (from ``score_individuals``)
and observed phenotypes.

**Polyploid compatibility.** All metrics (R², AUC, correlations,
Nagelkerke R², liability R²) are defined on scalar PGS scores and
phenotypes, making them ploidy-agnostic. Polyploid dosages in ``[0, k]``
are already incorporated into the PGS during ``score_individuals``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PGSValidation:
    """Predictive performance metrics for a polygenic score."""

    # Continuous-trait metrics
    r2: float  # PGS-phenotype R²
    r2_incremental: float  # R² gain over covariates-only model
    pearson_r: float  # Pearson correlation
    spearman_r: float  # Spearman rank correlation
    mae: float  # mean absolute error (PGS as predictor)

    # Binary-trait metrics (None for continuous)
    auc: float | None = None
    nagelkerke_r2: float | None = None
    liability_r2: float | None = None  # Lee et al. 2012 correction

    # Sample info
    n: int = 0
    trait_type: str = "continuous"  # "continuous" or "binary"
    prevalence: float | None = None  # for liability R² correction

    def __repr__(self) -> str:
        lines = [f"PGSValidation(n={self.n}, trait_type='{self.trait_type}')"]
        lines.append(f"  R²            = {self.r2:.6f}")
        lines.append(f"  R² incremental= {self.r2_incremental:.6f}")
        lines.append(f"  Pearson r     = {self.pearson_r:.6f}")
        lines.append(f"  Spearman r    = {self.spearman_r:.6f}")
        lines.append(f"  MAE           = {self.mae:.6f}")
        if self.trait_type == "binary":
            lines.append(f"  AUC           = {self.auc:.6f}" if self.auc is not None else "  AUC           = None")
            lines.append(
                f"  Nagelkerke R² = {self.nagelkerke_r2:.6f}"
                if self.nagelkerke_r2 is not None
                else "  Nagelkerke R² = None"
            )
            if self.liability_r2 is not None:
                lines.append(f"  Liability R²  = {self.liability_r2:.6f}")
            if self.prevalence is not None:
                lines.append(f"  Prevalence    = {self.prevalence:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols_r2(y: Tensor, X: Tensor) -> float:
    """R² from OLS fit ``y ~ X`` (``X`` should include an intercept column).

    Parameters
    ----------
    y : Tensor of shape ``(n,)``
    X : Tensor of shape ``(n, p)``

    Returns
    -------
    float
        Coefficient of determination, clamped to ``[0, 1]``.
    """
    # Solve normal equations via least-squares.
    # torch.linalg.lstsq returns (solution, residuals, rank, sv).
    result = torch.linalg.lstsq(X, y.unsqueeze(-1))
    beta = result.solution.squeeze(-1)  # (p,)
    y_hat = X @ beta  # (n,)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot.item() == 0.0:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot).item()
    return max(0.0, min(1.0, r2))


def _ols_predict(y: Tensor, X: Tensor) -> Tensor:
    """OLS predicted values ``y_hat`` from ``y ~ X``.

    Parameters
    ----------
    y : Tensor of shape ``(n,)``
    X : Tensor of shape ``(n, p)``

    Returns
    -------
    Tensor of shape ``(n,)``
    """
    result = torch.linalg.lstsq(X, y.unsqueeze(-1))
    beta = result.solution.squeeze(-1)
    return X @ beta


def _logistic_fit(
    y: Tensor,
    X: Tensor,
    max_iter: int = 50,
) -> tuple[Tensor, float]:
    """Newton-Raphson logistic regression.

    Parameters
    ----------
    y : Tensor of shape ``(n,)``
        Binary outcome (0/1).
    X : Tensor of shape ``(n, p)``
        Design matrix (should include intercept column).
    max_iter : int
        Maximum Newton-Raphson iterations.

    Returns
    -------
    beta : Tensor of shape ``(p,)``
        Fitted coefficients.
    log_lik : float
        Log-likelihood at convergence.
    """
    n, p = X.shape
    beta = torch.zeros(p, dtype=X.dtype, device=X.device)

    for _ in range(max_iter):
        eta = X @ beta  # (n,)
        # Numerically stable sigmoid.
        mu = torch.sigmoid(eta)  # P(y=1|X)
        # Gradient: X' (y - mu)
        grad = X.t() @ (y - mu)  # (p,)
        # Hessian diagonal weights: mu * (1 - mu), clamped for stability.
        w = (mu * (1.0 - mu)).clamp(min=1e-12)  # (n,)
        # Hessian: -X' diag(w) X
        XtWX = X.t() @ (w.unsqueeze(-1) * X)  # (p, p)
        # Newton step: beta += (X'WX)^{-1} X'(y-mu)
        try:
            delta = torch.linalg.solve(XtWX, grad)
        except torch.linalg.LinAlgError:
            # Singular Hessian — add small ridge.
            XtWX += 1e-8 * torch.eye(p, dtype=X.dtype, device=X.device)
            delta = torch.linalg.solve(XtWX, grad)

        beta = beta + delta

        # Convergence check.
        if delta.abs().max().item() < 1e-8:
            break

    # Log-likelihood at convergence.
    eta = X @ beta
    # log-lik = sum(y*eta - log(1 + exp(eta))), using logsigmoid for stability.
    ll = (y * eta + torch.nn.functional.logsigmoid(-eta)).sum().item()
    return beta, ll


def _auc_mannwhitney(pgs: Tensor, y_binary: Tensor) -> float:
    """AUC via the Mann-Whitney U statistic.

    ``AUC = P(PGS_case > PGS_control)``, computed from ranks.

    Parameters
    ----------
    pgs : Tensor of shape ``(n,)``
    y_binary : Tensor of shape ``(n,)``
        Binary labels (0/1).

    Returns
    -------
    float
        AUC in ``[0, 1]``.
    """
    n = pgs.shape[0]
    # Rank from 1..n (average ties).
    sorted_idx = pgs.argsort()
    ranks = torch.empty(n, dtype=pgs.dtype, device=pgs.device)
    # Assign ranks, handling ties by averaging.
    ranks[sorted_idx] = torch.arange(1, n + 1, dtype=pgs.dtype, device=pgs.device)

    # Average ties: for each group of equal values, assign the mean rank.
    sorted_vals = pgs[sorted_idx]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j].item() == sorted_vals[i].item():
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0  # 1-based average
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
        i = j

    n_case = y_binary.sum().item()
    n_control = n - n_case
    if n_case == 0 or n_control == 0:
        return 0.5

    rank_sum_cases = ranks[y_binary == 1].sum().item()
    u = rank_sum_cases - n_case * (n_case + 1) / 2.0
    auc = u / (n_case * n_control)
    return max(0.0, min(1.0, auc))


def _spearman_r(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation.

    Ranks both vectors (with average tie-breaking) and computes
    Pearson correlation on the ranks.

    Parameters
    ----------
    x, y : Tensor of shape ``(n,)``

    Returns
    -------
    float
    """

    def _rank(t: Tensor) -> Tensor:
        n = t.shape[0]
        sorted_idx = t.argsort()
        r = torch.empty(n, dtype=t.dtype, device=t.device)
        r[sorted_idx] = torch.arange(1, n + 1, dtype=t.dtype, device=t.device)
        sorted_vals = t[sorted_idx]
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_vals[j].item() == sorted_vals[i].item():
                j += 1
            if j > i + 1:
                avg = (i + 1 + j) / 2.0
                for k in range(i, j):
                    r[sorted_idx[k]] = avg
            i = j
        return r

    rx = _rank(x)
    ry = _rank(y)
    return _pearson_r(rx, ry)


def _pearson_r(x: Tensor, y: Tensor) -> float:
    """Pearson correlation coefficient.

    Parameters
    ----------
    x, y : Tensor of shape ``(n,)``

    Returns
    -------
    float
    """
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    denom = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt()
    if denom.item() == 0.0:
        return 0.0
    return (num / denom).item()


def _normal_pdf(x: float) -> float:
    """Standard normal PDF evaluated at *x*."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _normal_ppf(p: float) -> float:
    """Approximate inverse standard normal CDF (probit).

    Uses the rational approximation of Abramowitz & Stegun (26.2.23)
    which is accurate to ~4.5e-4. Sufficient for the liability-scale
    correction where the threshold is derived from prevalence.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    # Use symmetry: for p > 0.5 flip.
    if p > 0.5:
        return -_normal_ppf(1.0 - p)

    # Rational approximation for 0 < p <= 0.5.
    t = sqrt(-2.0 * log(p))
    # Coefficients from Abramowitz & Stegun.
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))


def _liability_r2(r2_obs: float, prevalence: float, sample_prop: float) -> float:
    """Lee et al. 2012 liability-scale R² correction.

    Converts observed-scale R² (from logistic or linear regression on
    a case-control sample) to the liability scale using the population
    prevalence.

    .. math::

        R^2_{\\text{liability}} = R^2_{\\text{obs}}
            \\frac{K(1-K)}{z^2 \\, P(1-P)}

    where *K* is the population prevalence, *P* is the sample case
    proportion, and *z* is the standard normal PDF at the liability
    threshold ``Phi^{-1}(1 - K)``.

    Parameters
    ----------
    r2_obs : float
        Observed-scale R².
    prevalence : float
        Population prevalence *K*.
    sample_prop : float
        Sample proportion of cases *P*.

    Returns
    -------
    float
        Liability-scale R², clamped to ``[0, 1]``.
    """
    K = prevalence
    P = sample_prop
    threshold = _normal_ppf(1.0 - K)
    z = _normal_pdf(threshold)
    if z == 0.0 or P == 0.0 or P == 1.0:
        return 0.0
    r2_liab = r2_obs * K * (1.0 - K) / (z * z * P * (1.0 - P))
    return max(0.0, min(1.0, r2_liab))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_pgs(
    pgs: Tensor,
    y: Tensor,
    *,
    covariates: Tensor | None = None,
    trait_type: str = "continuous",
    prevalence: float | None = None,
) -> PGSValidation:
    """Evaluate polygenic score predictive performance.

    Parameters
    ----------
    pgs : Tensor of shape ``(n,)``
        Per-individual polygenic scores (e.g. from
        :func:`score_individuals`).
    y : Tensor of shape ``(n,)``
        Observed phenotype. For binary traits, values must be 0 or 1.
    covariates : Tensor of shape ``(n, p)``, optional
        Covariate matrix. When provided, ``r2_incremental`` measures the
        R² gain of adding PGS to the covariate model.
    trait_type : ``"continuous"`` or ``"binary"``
        Determines which metric set is computed.
    prevalence : float, optional
        Population prevalence for the Lee et al. (2012) liability-scale
        R² correction. Required only for binary traits when
        ``liability_r2`` is desired.

    Returns
    -------
    PGSValidation
        Dataclass containing all computed metrics.

    Raises
    ------
    ValueError
        If inputs fail validation (mismatched lengths, bad trait_type,
        binary phenotype not 0/1, too few samples, etc.).
    """
    # ---- input validation ------------------------------------------------
    if pgs.dim() != 1:
        raise ValueError(f"pgs must be 1-D, got shape {pgs.shape}")
    if y.dim() != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    n = pgs.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"pgs and y must have same length, got {n} vs {y.shape[0]}")
    if n < 10:
        raise ValueError(f"At least 10 samples required, got {n}")
    if trait_type not in ("continuous", "binary"):
        raise ValueError(f"trait_type must be 'continuous' or 'binary', got '{trait_type}'")
    if covariates is not None:
        if covariates.dim() != 2:
            raise ValueError(f"covariates must be 2-D, got shape {covariates.shape}")
        if covariates.shape[0] != n:
            raise ValueError(
                f"covariates must have {n} rows to match pgs, got {covariates.shape[0]}"
            )

    # Cast to float64 for numerical precision.
    pgs = pgs.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    if covariates is not None:
        covariates = covariates.to(dtype=torch.float64)

    # Binary trait checks.
    if trait_type == "binary":
        unique_vals = torch.unique(y)
        if unique_vals.numel() != 2 or not (
            torch.allclose(unique_vals, torch.tensor([0.0, 1.0], dtype=torch.float64, device=y.device))
        ):
            raise ValueError(
                f"For binary trait_type, y must contain exactly 0 and 1. "
                f"Got unique values: {unique_vals.tolist()}"
            )
        if prevalence is not None and (prevalence <= 0.0 or prevalence >= 1.0):
            raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")

    # ---- design matrices -------------------------------------------------
    ones = torch.ones(n, 1, dtype=torch.float64, device=pgs.device)
    X_pgs = torch.cat([ones, pgs.unsqueeze(-1)], dim=1)  # (n, 2)

    if covariates is not None:
        X_cov = torch.cat([ones, covariates], dim=1)  # (n, 1+p)
        X_full = torch.cat([ones, covariates, pgs.unsqueeze(-1)], dim=1)  # (n, 1+p+1)
    else:
        X_cov = None
        X_full = X_pgs

    # ---- continuous metrics (always computed) -----------------------------
    r2 = _ols_r2(y, X_pgs)

    if X_cov is not None:
        r2_cov = _ols_r2(y, X_cov)
        r2_full = _ols_r2(y, X_full)
        r2_incremental = max(0.0, r2_full - r2_cov)
    else:
        r2_incremental = r2

    pearson = _pearson_r(pgs, y)
    spearman = _spearman_r(pgs, y)

    # MAE: from OLS prediction y ~ intercept + pgs.
    y_hat = _ols_predict(y, X_pgs)
    mae = (y - y_hat).abs().mean().item()

    # ---- binary metrics --------------------------------------------------
    auc_val: float | None = None
    nagelkerke_val: float | None = None
    liability_val: float | None = None

    if trait_type == "binary":
        auc_val = _auc_mannwhitney(pgs, y)

        # Nagelkerke R²: logistic regression.
        # Null model: intercept only.
        _, ll_null = _logistic_fit(y, ones)
        # Full model: intercept + (covariates +) pgs.
        _, ll_full = _logistic_fit(y, X_full)

        # Cox-Snell R² = 1 - exp(-2/n * (ll_full - ll_null))
        cox_snell = 1.0 - exp(-2.0 / n * (ll_full - ll_null))
        # Maximum possible Cox-Snell: 1 - exp(2/n * ll_null)
        cox_snell_max = 1.0 - exp(2.0 / n * ll_null)
        if cox_snell_max > 0.0:
            nagelkerke_val = cox_snell / cox_snell_max
        else:
            nagelkerke_val = 0.0
        nagelkerke_val = max(0.0, min(1.0, nagelkerke_val))

        # Liability-scale R² (Lee et al. 2012).
        if prevalence is not None:
            sample_prop = y.mean().item()
            liability_val = _liability_r2(r2, prevalence, sample_prop)

    return PGSValidation(
        r2=r2,
        r2_incremental=r2_incremental,
        pearson_r=pearson,
        spearman_r=spearman,
        mae=mae,
        auc=auc_val,
        nagelkerke_r2=nagelkerke_val,
        liability_r2=liability_val,
        n=n,
        trait_type=trait_type,
        prevalence=prevalence,
    )
