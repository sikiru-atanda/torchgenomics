"""Cox PQL: Penalized Quasi-Likelihood for Cox PH mixed model null fitting.

Implements the Breslow-Clayton (1993) PQL iteration for a Cox proportional
hazards frailty model:

    h(t | X_i, u_i) = h0(t) * exp(X_i' beta + u_i)
    u ~ N(0, sig2_g * K)

Each PQL iteration constructs a Poisson-like working model:
    z_i = log(E_i) + (delta_i - E_i) / E_i      (working response)
    W_i = E_i                                     (working weight)

where E_i = Lambda_0(T_i) * exp(eta_i) is the expected number of events.
The weighted LMM inner solve estimates variance components and BLUP.
The Breslow baseline hazard Lambda_0 is recomputed each iteration.

References
----------
- Breslow & Clayton (1993). Approximate inference in GLMM.
- Bi et al. (2020). SPACox: fast genome-wide time-to-event analysis.
- Dey et al. (2022). GATE: efficient frailty model approach.
- He & Kulminski (2020). COXMEG: fast Cox mixed-effects GWAS.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Breslow cumulative hazard
# ---------------------------------------------------------------------------

def _breslow_cumhaz(time: Tensor, event: Tensor, eta: Tensor) -> Tensor:
    """Compute Breslow cumulative hazard Lambda_0(T_i) for each individual.

    Parameters
    ----------
    time : (n,) follow-up times (positive)
    event : (n,) event indicators (0 or 1)
    eta : (n,) linear predictor (X'beta + u)

    Returns
    -------
    cum_hazard : (n,) cumulative baseline hazard at each T_i
    """
    n = time.shape[0]
    device = time.device

    # Sort by time ascending
    order = torch.argsort(time)
    time_s = time[order]
    event_s = event[order]

    # Log-sum-exp trick for numerical stability
    eta_s = eta[order]
    eta_max = eta_s.max()
    exp_eta_s = torch.exp(eta_s - eta_max)

    # Risk set sum: sum_{j >= i} exp(eta_j - eta_max)
    # Reverse cumsum gives this
    risk_sum = torch.flip(
        torch.cumsum(torch.flip(exp_eta_s, [0]), dim=0), [0]
    )
    risk_sum = risk_sum.clamp(min=1e-20)

    # Hazard increment at each time: d_i / (risk_sum_i * exp(eta_max))
    # But Lambda_0 is the BASELINE hazard, so we divide by exp(eta_max)
    # actually: Lambda_0(t) = sum_{t_j <= t, event} 1 / sum_{k in R_j} exp(eta_k)
    # = sum_{t_j <= t, event} 1 / (risk_sum_j * exp(eta_max))
    # Wait: risk_sum_j = sum_{k>=j} exp(eta_k - eta_max)
    # So sum_{k in R_j} exp(eta_k) = exp(eta_max) * risk_sum_j

    hazard_inc = event_s / (risk_sum * math.exp(eta_max))

    # Handle ties: events at the same time should use the same risk set.
    # The Breslow approach: all tied events share the full risk set at that time.
    # Since we sorted ascending and use the cumsum from the right,
    # tied events at the same time automatically get the same risk_sum_j
    # (only if no events come between them in sort order, which is true
    # since they have the same time).

    # Cumulative hazard
    cum_hazard_sorted = torch.cumsum(hazard_inc, dim=0)

    # Unsort back to original order
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(n, device=device)
    cum_hazard = cum_hazard_sorted[inv_order]

    return cum_hazard


# ---------------------------------------------------------------------------
# Cox PH initialization (no random effects)
# ---------------------------------------------------------------------------

def _cox_init_newton(
    time: Tensor,
    event: Tensor,
    X0: Tensor,
    *,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fit Cox PH model (no random effects) via Newton-Raphson.

    Optimises the partial likelihood: PL(beta) = prod_{i: event}
    [exp(X_i beta) / sum_{j in R_i} exp(X_j beta)].

    Parameters
    ----------
    time : (n,) follow-up times
    event : (n,) event indicators (0/1)
    X0 : (n, c) covariate matrix

    Returns
    -------
    beta : (c,) estimated coefficients
    cum_hazard : (n,) Breslow cumulative hazard at each T_i
    """
    n, c = X0.shape
    device = X0.device
    beta = torch.zeros(c, dtype=STAT_DTYPE, device=device)

    # Sort ascending by time
    order = torch.argsort(time)
    time_s = time[order]
    event_s = event[order]
    X0_s = X0[order]

    for it in range(max_iter):
        eta_s = X0_s @ beta  # (n,)
        eta_max = eta_s.max()
        exp_eta_s = torch.exp(eta_s - eta_max)

        # Risk set sums (reverse cumsum)
        S0 = torch.flip(torch.cumsum(torch.flip(exp_eta_s, [0]), dim=0), [0])
        S0 = S0.clamp(min=1e-20)

        # S1: sum_{j in R_i} X_j * exp(eta_j)
        wX = exp_eta_s.unsqueeze(1) * X0_s  # (n, c)
        S1 = torch.flip(
            torch.cumsum(torch.flip(wX, [0]), dim=0), [0]
        )  # (n, c)

        # S2: sum_{j in R_i} X_j X_j' exp(eta_j) — for Hessian
        # Use outer product, reverse cumsum
        wXX = exp_eta_s.unsqueeze(1).unsqueeze(2) * (
            X0_s.unsqueeze(2) * X0_s.unsqueeze(1)
        )  # (n, c, c)
        S2 = torch.flip(
            torch.cumsum(torch.flip(wXX, [0]), dim=0), [0]
        )  # (n, c, c)

        # Gradient: sum_{i: event} [X_i - S1_i / S0_i]
        E_X = S1 / S0.unsqueeze(1)  # (n, c) expected covariate in risk set
        grad = (event_s.unsqueeze(1) * (X0_s - E_X)).sum(dim=0)  # (c,)

        # Hessian: -sum_{i: event} [S2_i/S0_i - (S1_i/S0_i)(S1_i/S0_i)']
        # Vectorized: mask by events, compute all terms at once
        ev_mask = event_s > 0.5  # (n,)
        S2_over_S0 = S2[ev_mask] / S0[ev_mask].unsqueeze(1).unsqueeze(2)  # (n_ev, c, c)
        E_X_ev = E_X[ev_mask]  # (n_ev, c)
        outer_E_X = E_X_ev.unsqueeze(2) * E_X_ev.unsqueeze(1)  # (n_ev, c, c)
        info = (S2_over_S0 - outer_E_X).sum(dim=0)  # (c, c)

        # Newton step
        try:
            info_reg = info + 1e-8 * torch.eye(c, dtype=STAT_DTYPE, device=device)
            delta = torch.linalg.solve(info_reg, grad)
        except Exception:
            logger.warning("Cox init Newton: linalg.solve failed at iter %d", it)
            break

        beta = beta + delta

        if delta.abs().max().item() < tol:
            break

    # Compute Breslow cumulative hazard
    eta = X0 @ beta
    cum_hazard = _breslow_cumhaz(time, event, eta)

    return beta, cum_hazard


# ---------------------------------------------------------------------------
# Haseman-Elston regression for variance ratio warm start
# ---------------------------------------------------------------------------

def _he_regression_init(
    residuals: Tensor,
    K: Tensor,
    min_lam: float = 1e-6,
) -> tuple[float, float]:
    """Estimate sig2_g, sig2_e from residuals via Haseman-Elston regression.

    Uses off-diagonal cross-products of residuals regressed on kinship
    to get a quick estimate of the genetic variance ratio. This provides
    a much better warm start for REML than arbitrary initialization.

    Parameters
    ----------
    residuals : (n,) martingale or working residuals
    K : (n, n) kinship/GRM
    min_lam : float
        Minimum lambda to return.

    Returns
    -------
    (sig2_g, sig2_e) warm-start estimates.
    """
    n = residuals.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=K.device)

    R_outer = residuals.unsqueeze(1) * residuals.unsqueeze(0)  # (n, n)
    R_off = R_outer[mask]
    K_off = K[mask]

    # Regression: E[R_i * R_j] = sig2_g * K_ij (off-diagonal)
    K_mean = K_off.mean()
    R_mean = R_off.mean()
    cov_rk = (R_off * K_off).mean() - R_mean * K_mean
    var_k = K_off.var()

    if var_k.item() < 1e-20:
        return 0.5, 1.0  # fallback

    sig2_g_he = max(cov_rk.item() / var_k.item(), 0.0)
    var_R = residuals.var().item()
    sig2_e_he = max(var_R - sig2_g_he * K.diag().mean().item(), 1e-10)

    # Ensure minimum lambda
    if sig2_g_he < min_lam * sig2_e_he:
        sig2_g_he = min_lam * sig2_e_he

    logger.debug(
        "HE init: sig2_g=%.4f, sig2_e=%.4f, lam=%.6f",
        sig2_g_he, sig2_e_he, sig2_g_he / max(sig2_e_he, 1e-20),
    )

    return sig2_g_he, sig2_e_he


# ---------------------------------------------------------------------------
# Main Cox PQL loop
# ---------------------------------------------------------------------------

def cox_pql_fit(
    time: Tensor,
    event: Tensor,
    X0: Tensor,
    K: Tensor,
    *,
    max_outer: int = 30,
    tol: float = 1e-4,
) -> dict:
    """Run PQL iteration to fit a Cox PH frailty model.

    Parameters
    ----------
    time : (n,) follow-up times (positive)
    event : (n,) event indicators (0/1)
    X0 : (n, c) covariate matrix
    K : (n, n) kinship / GRM
    max_outer : int
        Maximum PQL outer iterations.
    tol : float
        Convergence tolerance on max|beta_new - beta_old|.

    Returns
    -------
    dict with keys: mu (E_i), martingale, beta, sig2_g, sig2_e,
        converged, eigenvalues, eigenvectors, evals_w, evecs_w,
        log_likelihood, n_outer
    """
    time = time.to(STAT_DTYPE)
    event = event.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    K = K.to(STAT_DTYPE)

    n, c = X0.shape

    # Step 0: Eigendecompose K (cached for BLUP)
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
    eigenvectors = eigenvectors.flip(1)

    # Step 1: Initialize with Cox PH (no random effects)
    beta, cum_haz = _cox_init_newton(time, event, X0)
    u = torch.zeros(n, dtype=STAT_DTYPE, device=X0.device)

    # Step 1b: HE regression warm start for variance ratio
    # Use martingale residuals from initial Cox PH to estimate sig2_g
    eta_init = X0 @ beta
    E_init = (cum_haz * torch.exp(eta_init)).clamp(min=1e-10)
    M_init = event - E_init
    sig2_g, sig2_e = _he_regression_init(M_init, K)
    converged = False

    evals_w = None
    evecs_w = None

    for outer in range(max_outer):
        # Expected events: E_i = Lambda_0(T_i) * exp(eta_i)
        eta = X0 @ beta + u
        E = cum_haz * torch.exp(eta)
        E = E.clamp(min=1e-10)

        # Working quantities (Poisson working model)
        W = E  # working weight = V(mu) for Poisson
        log_E = torch.log(E)
        z = log_E + (event - E) / E  # working response

        # Weighted LMM inner solve
        sqrtW = torch.sqrt(W)
        z_w = sqrtW * z
        X_w = sqrtW.unsqueeze(1) * X0

        # Eigendecomposition of sqrt(W) K sqrt(W)
        # Refresh at iter 0 and periodically (every 3 iters) as weights evolve
        if outer == 0 or outer % 3 == 0:
            K_w = sqrtW.unsqueeze(1) * K * sqrtW.unsqueeze(0)
            K_w = 0.5 * (K_w + K_w.T)
            evals_w, evecs_w = torch.linalg.eigh(K_w)
            evals_w = evals_w.flip(0).clamp(min=0.0)
            evecs_w = evecs_w.flip(1)

        # Rotate to eigenbasis
        z_rot = evecs_w.T @ z_w
        X_rot = evecs_w.T @ X_w

        # Profile REML for variance components
        sig2_g_reml, sig2_e_reml = _profile_reml_vc(
            z_rot, X_rot, evals_w,
            sig2_g_init=sig2_g, sig2_e_init=sig2_e,
        )

        # If REML converges to boundary (lambda < 1e-6) but HE suggests
        # non-trivial genetic variance, use HE estimate as a floor.
        # This prevents the PQL from collapsing to no random effects when
        # the Poisson working model REML is ill-conditioned.
        lam_reml = sig2_g_reml / max(sig2_e_reml, 1e-20)
        if outer <= 2 and lam_reml < 1e-5:
            # Recompute HE on current working residuals
            resid_curr = event - E
            sig2_g_he, sig2_e_he = _he_regression_init(resid_curr, K)
            lam_he = sig2_g_he / max(sig2_e_he, 1e-20)
            if lam_he > lam_reml:
                sig2_g, sig2_e = sig2_g_he, sig2_e_he
                logger.debug(
                    "PQL iter %d: REML at boundary (lam=%.2e), "
                    "using HE estimate (lam=%.2e)",
                    outer, lam_reml, lam_he,
                )
            else:
                sig2_g, sig2_e = sig2_g_reml, sig2_e_reml
        else:
            sig2_g, sig2_e = sig2_g_reml, sig2_e_reml

        # Solve for beta
        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (evals_w * lam + 1.0)
        wX = H_inv.unsqueeze(1) * X_rot
        XtHiX = X_rot.T @ wX
        XtHiz = X_rot.T @ (H_inv * z_rot)

        try:
            beta_new = torch.linalg.solve(XtHiX, XtHiz)
        except Exception:
            logger.warning("Cox PQL: linalg.solve failed at iter %d", outer)
            break

        # BLUP for u
        z_w_curr = sqrtW * (z - X0 @ beta_new)
        z_rot_resid = evecs_w.T @ z_w_curr
        shrink = (sig2_g * evals_w) / (sig2_g * evals_w + sig2_e)
        u_rot = shrink * z_rot_resid
        u_w = evecs_w @ u_rot
        u_new = u_w / sqrtW.clamp(min=1e-10)

        # Update eta and recompute Breslow hazard
        eta_new = X0 @ beta_new + u_new
        cum_haz = _breslow_cumhaz(time, event, eta_new)

        # Convergence check
        param_change = (beta_new - beta).abs().max().item()
        beta = beta_new
        u = u_new

        if param_change < tol and outer > 0:
            converged = True
            break

    # Final expected events and martingale residuals
    eta_final = X0 @ beta + u
    E_final = cum_haz * torch.exp(eta_final)
    E_final = E_final.clamp(min=1e-10)
    martingale = event - E_final

    # Quasi-log-likelihood (Poisson deviance)
    E_c = E_final.clamp(min=1e-300)
    ll = (event * torch.log(E_c) - E_final).sum().item()
    ll -= 0.5 * n * math.log(max(sig2_g, 1e-20))

    logger.info(
        "Cox PQL: converged=%s, outer=%d, sig2_g=%.4f, sig2_e=%.4f",
        converged, outer + 1, sig2_g, sig2_e,
    )

    return {
        "mu": E_final,           # expected events
        "martingale": martingale, # delta - E
        "beta": beta,
        "sig2_g": sig2_g,
        "sig2_e": sig2_e,
        "converged": converged,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "evals_w": evals_w,
        "evecs_w": evecs_w,
        "log_likelihood": ll,
        "n_outer": outer + 1,
    }


# ---------------------------------------------------------------------------
# Variance component estimation (replicates pattern from pql.py)
# ---------------------------------------------------------------------------

def _reml_loglik_at(
    lam: float,
    z_rot: Tensor,
    X_rot: Tensor,
    evals: Tensor,
) -> tuple[float, float, float]:
    """Evaluate profile REML log-likelihood at a given lambda.

    Returns (ll, sig2_e_hat, yPy) or (-inf, nan, nan) on failure.
    """
    n = z_rot.shape[0]
    c = X_rot.shape[1]
    H_inv = 1.0 / (evals * lam + 1.0)
    wX = H_inv.unsqueeze(1) * X_rot
    XtHiX = X_rot.T @ wX
    XtHiz = X_rot.T @ (H_inv * z_rot)

    try:
        beta = torch.linalg.solve(XtHiX, XtHiz)
    except Exception:
        return float("-inf"), float("nan"), float("nan")

    resid = z_rot - X_rot @ beta
    yPy = (resid * H_inv * resid).sum().item()
    log_det_H = torch.log(evals * lam + 1.0).sum().item()
    sign, log_det_XtHiX = torch.linalg.slogdet(XtHiX)
    if sign.item() <= 0:
        return float("-inf"), float("nan"), float("nan")

    sig2_e_hat = yPy / max(n - c, 1)
    ll = -0.5 * ((n - c) * math.log(max(sig2_e_hat, 1e-20))
                  + log_det_H + log_det_XtHiX.item())
    return ll, sig2_e_hat, yPy


def _ai_reml_newton(
    lam_init: float,
    z_rot: Tensor,
    X_rot: Tensor,
    evals: Tensor,
    max_iter: int = 15,
    tol: float = 1e-4,
) -> float:
    """Refine lambda via Average Information REML Newton updates.

    The AI (Average Information) algorithm (Gilmour et al. 1995) uses
    the average of observed and expected information, yielding a simple
    and efficient Newton step for variance ratio estimation.

    Parameters
    ----------
    lam_init : float
        Initial lambda = sig2_g / sig2_e from grid search or HE.
    z_rot, X_rot, evals : rotated working model quantities.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Relative convergence tolerance on lambda.

    Returns
    -------
    Refined lambda.
    """
    n = z_rot.shape[0]
    c = X_rot.shape[1]
    lam = max(lam_init, 1e-8)

    for it in range(max_iter):
        H_inv = 1.0 / (evals * lam + 1.0)
        wX = H_inv.unsqueeze(1) * X_rot
        XtHiX = X_rot.T @ wX
        XtHiz = X_rot.T @ (H_inv * z_rot)

        try:
            beta = torch.linalg.solve(XtHiX, XtHiz)
        except Exception:
            break

        resid = z_rot - X_rot @ beta
        Py = H_inv * resid  # P y (approximate)

        yPy = (resid * Py).sum().item()
        sig2_e_hat = max(yPy / max(n - c, 1), 1e-20)

        # Py'K Py = (Py * evals * Py).sum() since we're in the eigenbasis
        PyKPy = (Py * evals * Py).sum().item()

        # tr(P K) ≈ tr(H^{-1} K) - tr(correction)
        # In eigenbasis: tr(H^{-1} diag(evals) * lam) simplification
        # tr(P K) ≈ sum_i evals_i / (evals_i * lam + 1) - correction
        PK_diag = evals * H_inv  # diagonal of H^{-1} K in eigenbasis
        # Correction: X' H^{-1} K H^{-1} X (X' H^{-1} X)^{-1}
        KHiX = (evals * H_inv).unsqueeze(1) * X_rot
        try:
            XtHiX_inv = torch.linalg.inv(
                XtHiX + 1e-10 * torch.eye(c, dtype=z_rot.dtype, device=z_rot.device)
            )
        except Exception:
            break
        correction = (KHiX * (XtHiX_inv @ (X_rot.T @ KHiX))).sum().item()
        tr_PK = PK_diag.sum().item() - correction

        # Gradient: d(-2 ll_REML) / d(lam) = tr(PK) - PyKPy / sig2_e
        grad = 0.5 * (PyKPy / sig2_e_hat - tr_PK)  # gradient of ll w.r.t. lam

        # AI: 0.5 * Py' K P K P y / sig2_e^2
        PKPy = evals * Py  # K P y in eigenbasis (approximate)
        ai = 0.5 * (PKPy * H_inv * PKPy).sum().item() / sig2_e_hat ** 2

        if abs(ai) < 1e-30:
            break

        step = grad / ai
        lam_new = max(lam + step, 1e-8)

        # Bound step to avoid wild jumps
        if lam_new > 100.0 * lam:
            lam_new = 100.0 * lam
        if lam_new < lam / 100.0:
            lam_new = lam / 100.0

        rel_change = abs(lam_new - lam) / max(lam, 1e-8)
        lam = lam_new

        if rel_change < tol:
            break

    return lam


def _profile_reml_vc(
    z_rot: Tensor,
    X_rot: Tensor,
    evals: Tensor,
    sig2_g_init: float = 0.5,
    sig2_e_init: float = 1.0,
) -> tuple[float, float]:
    """Estimate variance components via grid search + AI-REML Newton refinement.

    Uses a coarse grid to find a good starting point, then refines with
    AI-REML Newton updates (Gilmour et al. 1995) for precise estimation.
    """
    n = z_rot.shape[0]
    c = X_rot.shape[1]

    best_ll = float("-inf")
    best_lam = sig2_g_init / max(sig2_e_init, 1e-20)

    # Extended grid covering wider range (log-spaced in low range)
    grid = [
        1e-6, 1e-5, 1e-4, 5e-4, 0.001, 0.005,
        0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
        best_lam,
    ]

    for lam in grid:
        if lam <= 0:
            continue
        ll, _, _ = _reml_loglik_at(lam, z_rot, X_rot, evals)
        if ll > best_ll:
            best_ll = ll
            best_lam = lam

    # AI-REML Newton refinement from grid optimum
    best_lam = _ai_reml_newton(best_lam, z_rot, X_rot, evals)

    # Extract final components at refined lambda
    H_inv = 1.0 / (evals * best_lam + 1.0)
    wX = H_inv.unsqueeze(1) * X_rot
    XtHiX = X_rot.T @ wX
    XtHiz = X_rot.T @ (H_inv * z_rot)
    beta = torch.linalg.solve(XtHiX, XtHiz)
    resid = z_rot - X_rot @ beta
    yPy = (resid * H_inv * resid).sum().item()
    sig2_e = max(yPy / max(n - c, 1), 1e-10)
    sig2_g = max(best_lam * sig2_e, 1e-10)

    return sig2_g, sig2_e
