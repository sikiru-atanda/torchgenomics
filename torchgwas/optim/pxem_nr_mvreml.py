"""PX-EM + Newton-Raphson (Average Information) multi-trait REML.

Matches GEMMA's optimizer for multi-trait variance component estimation.
Works in the rotated eigenspace after K eigendecomposition.

Algorithm:
    1. EM warm-start (5-10 iterations) — monotone convergence
    2. AI-REML Newton-Raphson — quadratic convergence near optimum
    3. Fallback to EM if NR step fails

References
----------
Zhou & Stephens, Nat Methods 2014 (GEMMA)
Lee & van der Werf 2006 (AI-REML for multi-trait)
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .mvlmm_reml import compute_sigma_inv

logger = logging.getLogger(__name__)


def _compute_Py(
    Y_rot: Tensor,
    X0_rot: Tensor,
    W: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute P-projected residuals and GLS quantities.

    Parameters
    ----------
    Y_rot : (n, d)
    X0_rot : (n, c)
    W : (n, d, d) — per-individual Sigma^{-1}

    Returns
    -------
    Py : (n, d) — P-projected residuals
    B : (c, d) — GLS fixed-effect estimates
    XtWX : (cd, cd) — normal equation matrix
    R : (n, d) — raw residuals Y - X@B
    """
    n, d = Y_rot.shape
    c = X0_rot.shape[1]

    XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)
    XtWX = XtWX.reshape(c * d, c * d)

    WY = torch.einsum('ist,is->it', W, Y_rot)
    XtWY = torch.einsum('ia,it->at', X0_rot, WY).reshape(c * d)

    B_vec = torch.linalg.solve(XtWX, XtWY)
    B = B_vec.reshape(c, d)

    R = Y_rot - X0_rot @ B
    Py = torch.einsum('ist,is->it', W, R)  # W @ R per individual

    return Py, B, XtWX, R


def _reml_loglikelihood(
    W: Tensor,
    logdet: Tensor,
    Py: Tensor,
    R: Tensor,
    XtWX: Tensor,
    n: int,
    d: int,
    c: int,
) -> float:
    """Evaluate multi-trait REML log-likelihood."""
    df = n - c
    quad = (R * Py).sum()
    sign, logdet_XtWX = torch.linalg.slogdet(XtWX)
    logdet_XtWX_val = logdet_XtWX.item() if sign > 0 else 0.0
    neg2ll = (
        logdet.sum().item()
        + quad.item()
        + df * d * math.log(2 * math.pi)
        + logdet_XtWX_val
    )
    return -0.5 * neg2ll


def _compute_trace_terms(
    W: Tensor,
    X0_rot: Tensor,
    XtWX_inv: Tensor,
    eigenvalues: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute trace terms for EM/score: tr(P dV/dVg) and tr(P dV/dVe).

    Returns
    -------
    Tr_g : (d, d) — sum_i evals[i] * P_eff_i[s,t]
    Tr_e : (d, d) — sum_i P_eff_i[s,t]
    """
    n, d, _ = W.shape
    c = X0_rot.shape[1]

    # Part 1: sum_i f(evals[i]) * W_i[s,t]
    Tr_g_raw = torch.einsum('i,ist->st', eigenvalues, W)  # (d, d)
    Tr_e_raw = W.sum(dim=0)  # (d, d)

    # Part 2: correction from fixed effects
    # tr((XtWX)^{-1} @ Q_k) where Q_k involves W_i dV W_i terms
    # Q_g[as, bt] = sum_i evals[i] * X[i,a] * W_i[s,r] * E_rt * W_i[r,q] * X[i,b]
    # For the trace over (as,bt), we need sum_{a,s} Q_g[as,as] for each pair
    # But actually we need the full (d,d) correction matrix.

    # WX[i, s, a] = sum_t W[i, s, t] * X0_rot[i, a]
    WX = torch.einsum('ist,ia->isa', W, X0_rot)  # (n, d, c)

    # For Vg correction: need sum_i evals[i] * (WX_i^T WX_i) coupled through XtWX_inv
    # The correction to Tr_g[s,t] is:
    # sum_{a,b} (sum_i evals[i] * WX[i,s,a] * WX[i,t,b]) * XtWX_inv[a*d+s, b*d+t]
    # Wait, this isn't right because the indexing of XtWX is (a*d+s, b*d+t).
    #
    # Correct: tr(P dV/dVg_st) = Tr_raw_g[s,t] - correction_g[s,t]
    # where correction_g[s,t] = tr(XtWX^{-1} @ C_g_st) and
    # C_g_st[a*d+r, b*d+q] = sum_i evals[i] * X[i,a] * W[i,r,s] * W[i,t,q] * X[i,b]
    # tr(XtWX^{-1} @ C_g_st) = sum_{a,r,b,q} XtWX_inv[a*d+r, b*d+q] * C_g_st[a*d+r, b*d+q]

    # Build correction efficiently for each (s,t):
    # eWX_s[i, a] = evals[i] * WX[i, s, a] = evals[i] * sum_t W[i,s,t] * X[i,a]
    eWX = torch.einsum('i,isa->isa', eigenvalues, WX)  # (n, d, c)

    # For Vg correction at (s,t):
    # C_g_st reshaped as (cd, cd) has entry [(a,r), (b,q)] = sum_i evals[i] * X[i,a]*W[i,r,s]*W[i,t,q]*X[i,b]
    # tr(XtWX_inv @ C_g_st) = sum_i evals[i] * sum_{a,b,r,q} XtWX_inv[(a,r),(b,q)] * X[i,a]*W[i,r,s]*W[i,t,q]*X[i,b]
    # = sum_i evals[i] * WX[i,:,:]^T @ M_inv_block @ WX[i,:,:] extracted at (s,t)
    # where M_inv_block involves XtWX_inv.
    #
    # More concretely, define V_i[r, q] = sum_{a,b} WX[i,r,a] * XtWX_inv[(a*d+r),(b*d+q)] * WX[i,q,b]
    # But the XtWX_inv indexing couples (a,r) with (b,q), not just (a) with (b).

    # Simpler approach: compute the correction as a whole matrix.
    # Define ZW[i, s, a*d+r] = X[i,a] * W[i,r,s] -> actually this is WX reshaped.
    # Then C_g_st = sum_i evals[i] * ZW_s[i,:] @ ZW_t[i,:]^T where ZW_s is the slice.

    # ZW[i, (a,r)] = X[i,a] * W[i,r,s] for fixed s
    # But this is just WX[i, s, a] indexed differently... no, WX[i,s,a] = sum_t W[i,s,t]*X[i,a]
    # We need X[i,a]*W[i,r,s] which is X_rot[i,a] * W[i,r,s].
    # Define: XW_col_s[i, a, r] = X[i,a] * W[i,r,s]  -> (n, c, d)
    # Then: XW_col_s reshaped to (n, cd) for each s.

    # For each s: XW_s[i, a*d+r] = X[i,a] * W[i,r,s]
    # C_g_st[(a*d+r), (b*d+q)] = sum_i evals[i] * XW_s[i,a*d+r] * XW_t[i,b*d+q]

    # XW_all[i, s, a, r] = X[i,a] * W[i,r,s]
    XW_all = torch.einsum('ia,irs->isar', X0_rot, W)  # (n, d, c, d)
    # Reshape: XW_all[i, s, cd_idx] where cd_idx = a*d+r
    XW_flat = XW_all.reshape(n, d, c * d)  # (n, d, cd)

    # C_g[s,t] as a (cd, cd) matrix = sum_i evals[i] * XW_flat[i,s,:] @ XW_flat[i,t,:]^T
    # But we only need the trace against XtWX_inv:
    # corr_g[s,t] = sum_i evals[i] * XW_flat[i,s,:] @ XtWX_inv @ XW_flat[i,t,:]
    # = sum_i evals[i] * (XW_flat[i,s,:] @ XtWX_inv) . XW_flat[i,t,:]

    # Precompute: XW_M = XW_flat @ XtWX_inv -> (n, d, cd)
    XW_M = torch.einsum('isr,rq->isq', XW_flat, XtWX_inv)  # (n, d, cd)

    # corr_g[s,t] = sum_i evals[i] * sum_q XW_M[i,s,q] * XW_flat[i,t,q]
    corr_g = torch.einsum('i,isq,itq->st', eigenvalues, XW_M, XW_flat)  # (d, d)
    corr_e = torch.einsum('isq,itq->st', XW_M, XW_flat)  # (d, d)

    Tr_g = Tr_g_raw - corr_g
    Tr_e = Tr_e_raw - corr_e

    return Tr_g, Tr_e


def _em_step(
    Py: Tensor,
    eigenvalues: Tensor,
    W: Tensor,
    X0_rot: Tensor,
    XtWX_inv: Tensor,
    Vg: Tensor,
    Ve: Tensor,
    n: int,
    c: int,
) -> tuple[Tensor, Tensor]:
    """One EM update step for multi-trait Vg, Ve.

    Update: Vg_new = Vg + Vg @ (Q_g - Tr_g) @ Vg / df
            Ve_new = Ve + Ve @ (Q_e - Tr_e) @ Ve / df
    """
    d = Vg.shape[0]
    df = n - c

    # Quadratic terms from Py
    Q_g = torch.einsum('i,is,it->st', eigenvalues, Py, Py)  # (d, d)
    Q_e = torch.einsum('is,it->st', Py, Py)  # (d, d)

    # Trace terms
    Tr_g, Tr_e = _compute_trace_terms(W, X0_rot, XtWX_inv, eigenvalues)

    # EM update
    S_g = (Q_g - Tr_g) / df
    S_e = (Q_e - Tr_e) / df
    Vg_new = Vg + Vg @ S_g @ Vg
    Ve_new = Ve + Ve @ S_e @ Ve

    # Ensure symmetric
    Vg_new = 0.5 * (Vg_new + Vg_new.T)
    Ve_new = 0.5 * (Ve_new + Ve_new.T)

    # Ensure PD
    Vg_new = _ensure_pd(Vg_new)
    Ve_new = _ensure_pd(Ve_new)

    return Vg_new, Ve_new


def _ensure_pd(M: Tensor, min_eval: float = 1e-8) -> Tensor:
    """Ensure matrix is symmetric positive definite."""
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=min_eval)
    return evecs @ torch.diag(evals) @ evecs.T


def _vech_indices(d: int) -> list[tuple[int, int]]:
    """Lower-triangle indices for vech parameterization (column-major)."""
    indices = []
    for t in range(d):
        for s in range(t, d):
            indices.append((s, t))
    return indices


def _ai_step(
    Py: Tensor,
    eigenvalues: Tensor,
    W: Tensor,
    X0_rot: Tensor,
    XtWX: Tensor,
    XtWX_inv: Tensor,
    Vg: Tensor,
    Ve: Tensor,
    n: int,
    c: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One AI-REML Newton-Raphson step.

    Returns
    -------
    delta_Vg : (d, d) — proposed change in Vg
    delta_Ve : (d, d) — proposed change in Ve
    score : (p,) — score vector
    AI : (p, p) — average information matrix
    """
    d = Vg.shape[0]
    df = n - c
    device = Vg.device

    vech_idx = _vech_indices(d)
    p_half = len(vech_idx)  # d*(d+1)/2
    p = 2 * p_half

    # --- Score vector ---
    # Q_g[s,t] = sum_i evals[i] * Py[i,s] * Py[i,t]
    Q_g = torch.einsum('i,is,it->st', eigenvalues, Py, Py)
    Q_e = torch.einsum('is,it->st', Py, Py)

    Tr_g, Tr_e = _compute_trace_terms(W, X0_rot, XtWX_inv, eigenvalues)

    score = torch.zeros(p, dtype=STAT_DTYPE, device=device)
    for k, (s, t) in enumerate(vech_idx):
        factor = 1.0 if s == t else 2.0
        score[k] = -0.5 * factor * (Tr_g[s, t] - Q_g[s, t])
        score[p_half + k] = -0.5 * factor * (Tr_e[s, t] - Q_e[s, t])

    # --- AI matrix ---
    # AI[k, l] = 0.5 * Py^T dV_k P dV_l Py
    # where dV_k involves evals for Vg params, 1 for Ve params.
    #
    # For each vech param k -> (s,t), construct z_k[i,:] = f(evals[i]) * E_st @ Py[i,:]
    # Then apply P to z_k, and take inner products.

    # Build z vectors for all params: z[k, i, :] shape (p, n, d)
    # For Vg param k -> (s,t): z_k[i, r] = evals[i] * (delta_{r,s}*Py[i,t] + delta_{r,t}*Py[i,s]) / (1+delta_{s,t})
    # For Ve param k -> (s,t): same but without evals[i]

    # More efficiently: for Vg params, define Py_g[i,s] = evals[i] * Py[i,s]
    Py_g = eigenvalues.unsqueeze(1) * Py  # (n, d)
    Py_e = Py  # (n, d)

    # Apply P to Py_g and Py_e column-by-column
    # P @ v = W @ v - W @ X @ (XtWX)^{-1} @ X^T @ W @ v
    def apply_P(v):
        """Apply P operator to (n, d) matrix v."""
        Wv = torch.einsum('ist,is->it', W, v)  # (n, d)
        XtWv = torch.einsum('ia,it->at', X0_rot, Wv).reshape(c * d)
        correction = XtWX_inv @ XtWv  # (cd,)
        XB = X0_rot @ correction.reshape(c, d)  # (n, d)
        WXB = torch.einsum('ist,is->it', W, XB)  # (n, d)
        return Wv - WXB

    PPy_g = apply_P(Py_g)  # (n, d)
    PPy_e = apply_P(Py_e)  # (n, d)

    # AI matrix computation:
    # For Vg-Vg block: AI_gg[k,l] where k->(s,t), l->(u,v):
    # = 0.5 * sum_i [Py_g[i,s]*PPy_g[i,v]*delta_{t,u} + Py_g[i,s]*PPy_g[i,u]*delta_{t,v}
    #               + Py_g[i,t]*PPy_g[i,v]*delta_{s,u} + Py_g[i,t]*PPy_g[i,u]*delta_{s,v}]
    # Simplified by using: z_k = E_st @ Py_g, P(dV_l @ Py) = E_uv applied to PPy_g
    # AI[k,l] = 0.5 * sum_i z_k[i,:]^T @ P(z_l)[i,:]

    # Precompute cross products
    # G_gg[s,v] = sum_i Py_g[i,s] * PPy_g[i,v]
    G_gg = Py_g.T @ PPy_g  # (d, d)
    G_ge = Py_g.T @ PPy_e  # (d, d)
    G_eg = Py_e.T @ PPy_g  # (d, d)
    G_ee = Py_e.T @ PPy_e  # (d, d)

    AI = torch.zeros(p, p, dtype=STAT_DTYPE, device=device)
    for k, (s, t) in enumerate(vech_idx):
        for l, (u, v) in enumerate(vech_idx):
            # Vg-Vg block
            val = 0.0
            if s == t and u == v:
                val = G_gg[s, u] * G_gg[t, v]
            elif s == t:
                val = G_gg[s, u] * G_gg[t, v] + G_gg[s, v] * G_gg[t, u]
            elif u == v:
                val = G_gg[s, u] * G_gg[t, v] + G_gg[t, u] * G_gg[s, v]
            else:
                val = (G_gg[s, u] * G_gg[t, v] + G_gg[s, v] * G_gg[t, u]
                       + G_gg[t, u] * G_gg[s, v] + G_gg[t, v] * G_gg[s, u])
            # Wait, this isn't right. Let me re-derive.
            pass

    # Actually, the AI matrix for the vech parameterization is:
    # AI_gg[k,l] = 0.5 * tr(P dV_k P dV_l)  for the AI approximation
    # But the "average information" uses: 0.5 * y^T P dV_k P dV_l P y
    #
    # Since dV/dVg_st @ Py = evals * E_st @ Py (per individual),
    # and P(dV Py) involves applying P to this,
    #
    # The inner product for the AI is:
    # AI_gg[k=(s,t), l=(u,v)] = 0.5 * (dV_k Py)^T P (dV_l Py)
    #
    # (dV_k Py)[i, r] = evals[i] * (delta_rs * Py[i,t] + delta_rt * Py[i,s] - delta_st * delta_rs * Py[i,s])
    # For s != t: (dV_k Py)[i, :] has Py[i,t] at index s and Py[i,s] at index t, 0 elsewhere
    # For s == t: (dV_k Py)[i, :] has Py[i,s] at index s, 0 elsewhere
    #
    # Wait, E_st for off-diagonal is e_s e_t^T + e_t e_s^T, so E_st @ v gives:
    # result[s] = v[t], result[t] = v[s], rest = 0  (for s != t)
    # For diagonal: E_ss @ v gives result[s] = v[s], rest = 0

    # Let me just compute directly using matrix products.
    # For each parameter k, construct the z_k vector and compute inner products.

    # Reset AI
    AI.zero_()

    # z_g[k][i, r] for Vg param (s,t): evals[i] * E_st @ Py[i,:]
    # P(z_g[k]) uses apply_P but E_st applied to Py_g

    # Build all z vectors at once using tensor indexing
    # For Vg: z_g_k[i,:] = evals[i] * E_st @ Py[i,:]
    #   For s==t: z[i, s] = evals[i]*Py[i,s], rest 0
    #   For s!=t: z[i, s] = evals[i]*Py[i,t], z[i, t] = evals[i]*Py[i,s], rest 0
    # For Ve: same without evals

    # Apply P to each z and compute inner products.
    # For efficiency, note that z_k only has 1-2 nonzero columns.
    # P(z_k) = P applied to z_k, which involves the full P machinery.

    # Since P(z_k) for param (s,t) depends on z_k which is sparse (1-2 nonzero cols of Py_g),
    # we can compute P applied to each column of Py_g/Py_e once and combine.

    # PPy_g_col[s] = P @ (evals * Py[:, s]) = PPy_g[:, s] (already computed!)
    # PPy_e_col[s] = P @ Py[:, s] = PPy_e[:, s] (already computed!)

    # For Vg param (s,t) with s==t:
    #   z_k[i,:] = evals[i]*Py[i,s]*e_s, so P(z_k)[i,:] = PPy_g[i,s]*e_s + correction_from_other_traits
    #   Wait, no. P is applied to the full d-dimensional vector z_k[i,:].
    #   z_k = Py_g[:, s:s+1] padded with zeros... no, that loses the coupling in P.
    #
    # Actually, P @ z where z[i,:] = evals[i]*Py[i,s]*e_s is NOT the same as PPy_g[:,s]*e_s
    # because P couples traits through W (which is d×d per individual).
    #
    # So we need to apply P to each sparse z vector. For d parameters per vech,
    # this means p_half P applications. Each costs O(n*d^2 + n*c*d).

    # For small d (<=7), p_half <= 28. This is manageable.

    # Let me just compute it directly.
    Pz_g = torch.zeros(p_half, n, d, dtype=STAT_DTYPE, device=device)
    z_g = torch.zeros(p_half, n, d, dtype=STAT_DTYPE, device=device)
    z_e = torch.zeros(p_half, n, d, dtype=STAT_DTYPE, device=device)
    Pz_e = torch.zeros(p_half, n, d, dtype=STAT_DTYPE, device=device)

    for k, (s, t) in enumerate(vech_idx):
        z = torch.zeros(n, d, dtype=STAT_DTYPE, device=device)
        if s == t:
            z[:, s] = Py_g[:, s]
        else:
            z[:, s] = Py_g[:, t]
            z[:, t] = Py_g[:, s]
        z_g[k] = z
        Pz_g[k] = apply_P(z)

        z2 = torch.zeros(n, d, dtype=STAT_DTYPE, device=device)
        if s == t:
            z2[:, s] = Py_e[:, s]
        else:
            z2[:, s] = Py_e[:, t]
            z2[:, t] = Py_e[:, s]
        z_e[k] = z2
        Pz_e[k] = apply_P(z2)

    # AI[k, l] = 0.5 * sum_i z_k[i,:]^T @ Pz_l[i,:]
    # z_k and Pz_l are indexed by (param_type, vech_idx)
    # Vg-Vg block: AI[k, l] = 0.5 * (z_g[k] * Pz_g[l]).sum()
    for k in range(p_half):
        for l in range(k, p_half):
            val = 0.5 * (z_g[k] * Pz_g[l]).sum().item()
            AI[k, l] = val
            AI[l, k] = val

    # Ve-Ve block
    for k in range(p_half):
        for l in range(k, p_half):
            val = 0.5 * (z_e[k] * Pz_e[l]).sum().item()
            AI[p_half + k, p_half + l] = val
            AI[p_half + l, p_half + k] = val

    # Vg-Ve cross block
    for k in range(p_half):
        for l in range(p_half):
            val = 0.5 * (z_g[k] * Pz_e[l]).sum().item()
            AI[k, p_half + l] = val
            AI[p_half + l, k] = val

    # Newton step: delta_theta = AI^{-1} @ score
    # Add small ridge for stability
    AI_reg = AI + torch.eye(p, dtype=STAT_DTYPE, device=device) * 1e-6 * AI.diag().abs().mean()
    try:
        delta_theta = torch.linalg.solve(AI_reg, score)
    except Exception:
        logger.warning("AI matrix singular, falling back to steepest ascent.")
        delta_theta = score * 0.01

    # Unpack delta_theta into delta_Vg and delta_Ve
    delta_Vg = torch.zeros(d, d, dtype=STAT_DTYPE, device=device)
    delta_Ve = torch.zeros(d, d, dtype=STAT_DTYPE, device=device)
    for k, (s, t) in enumerate(vech_idx):
        delta_Vg[s, t] = delta_theta[k]
        delta_Vg[t, s] = delta_theta[k]
        delta_Ve[s, t] = delta_theta[p_half + k]
        delta_Ve[t, s] = delta_theta[p_half + k]

    return delta_Vg, delta_Ve, score, AI


def pxem_nr_mvreml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    n_traits: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    em_iters: int = 20,
    Vg_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
) -> tuple[Tensor, Tensor, float, list[dict]]:
    """PX-EM + AI-REML for multi-trait REML variance components.

    Parameters
    ----------
    Y_rot : (n, d)
    X0_rot : (n, c)
    eigenvalues : (n,)
    n_traits : number of traits d
    max_iter : total max iterations (EM + NR combined)
    tol : relative convergence tolerance
    em_iters : EM warm-start iterations before switching to NR
    Vg_init, Ve_init : (d, d) initial covariance matrices

    Returns
    -------
    Vg, Ve : (d, d) estimated covariance matrices
    ll : float — final REML log-likelihood
    trace : list[dict] — optimizer trace
    """
    n = Y_rot.shape[0]
    d = n_traits if Y_rot.ndim == 1 else Y_rot.shape[1]
    c = X0_rot.shape[1]
    device = Y_rot.device

    if Y_rot.ndim == 1:
        Y_rot = Y_rot.unsqueeze(1)

    # Initialize
    if Vg_init is not None:
        Vg = Vg_init.clone()
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Vg = torch.eye(d, dtype=STAT_DTYPE, device=device) * var_y * 0.5

    if Ve_init is not None:
        Ve = Ve_init.clone()
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Ve = torch.eye(d, dtype=STAT_DTYPE, device=device) * var_y * 0.5

    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg = Vg.clone()
    best_Ve = Ve.clone()
    converged = False  # Post-V1 plumbing: explicit optimizer-side flag

    # Phase 1: EM warm-start
    for it in range(em_iters):
        W, logdet = compute_sigma_inv(eigenvalues, Vg, Ve)
        Py, B, XtWX, R = _compute_Py(Y_rot, X0_rot, W)
        ll = _reml_loglikelihood(W, logdet, Py, R, XtWX, n, d, c)
        XtWX_inv = torch.linalg.inv(XtWX)

        trace.append({"iter": it, "mode": "EM", "ll": ll})

        if ll > best_ll:
            best_ll = ll
            best_Vg = Vg.clone()
            best_Ve = Ve.clone()

        Vg, Ve = _em_step(Py, eigenvalues, W, X0_rot, XtWX_inv, Vg, Ve, n, c)

    # Phase 2: AI-REML Newton-Raphson
    nr_fail_count = 0
    for it in range(em_iters, max_iter):
        W, logdet = compute_sigma_inv(eigenvalues, Vg, Ve)
        Py, B, XtWX, R = _compute_Py(Y_rot, X0_rot, W)
        ll = _reml_loglikelihood(W, logdet, Py, R, XtWX, n, d, c)
        XtWX_inv = torch.linalg.inv(XtWX)

        trace.append({"iter": it, "mode": "NR", "ll": ll})

        if ll > best_ll:
            best_ll = ll
            best_Vg = Vg.clone()
            best_Ve = Ve.clone()

        # Check convergence
        if it > em_iters and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "PX-EM+NR converged in %d iterations (EM=%d, NR=%d), ll=%.6f",
                it, em_iters, it - em_iters, ll,
            )
            converged = True
            break

        # Try Newton-Raphson step
        try:
            delta_Vg, delta_Ve, score, AI = _ai_step(
                Py, eigenvalues, W, X0_rot, XtWX, XtWX_inv, Vg, Ve, n, c,
            )

            # Backtracking line search
            step_accepted = False
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                Vg_trial = 0.5 * ((Vg + alpha * delta_Vg) + (Vg + alpha * delta_Vg).T)
                Ve_trial = 0.5 * ((Ve + alpha * delta_Ve) + (Ve + alpha * delta_Ve).T)

                # Project to PD if needed (small perturbation)
                try:
                    eig_g = torch.linalg.eigvalsh(Vg_trial)
                    eig_e = torch.linalg.eigvalsh(Ve_trial)
                    if eig_g.min() < 1e-10:
                        Vg_trial = _ensure_pd(Vg_trial, min_eval=1e-10)
                    if eig_e.min() < 1e-10:
                        Ve_trial = _ensure_pd(Ve_trial, min_eval=1e-10)
                    W_trial, logdet_trial = compute_sigma_inv(
                        eigenvalues, Vg_trial, Ve_trial,
                    )
                    Py_trial, _, XtWX_trial, R_trial = _compute_Py(
                        Y_rot, X0_rot, W_trial,
                    )
                    ll_trial = _reml_loglikelihood(
                        W_trial, logdet_trial, Py_trial, R_trial,
                        XtWX_trial, n, d, c,
                    )
                    if ll_trial > ll:
                        Vg = Vg_trial
                        Ve = Ve_trial
                        step_accepted = True
                        break
                except Exception:
                    continue

            if not step_accepted:
                # Fall back to EM step
                nr_fail_count += 1
                logger.debug("NR step rejected at iter %d, using EM fallback.", it)
                Vg, Ve = _em_step(
                    Py, eigenvalues, W, X0_rot, XtWX_inv, Vg, Ve, n, c,
                )
                if nr_fail_count >= 5:
                    logger.warning(
                        "NR failed %d times, switching to EM-only for remaining iters.",
                        nr_fail_count,
                    )
                    # Run remaining iterations as EM
                    for em_it in range(it + 1, max_iter):
                        W, logdet = compute_sigma_inv(eigenvalues, Vg, Ve)
                        Py, B, XtWX, R = _compute_Py(Y_rot, X0_rot, W)
                        ll = _reml_loglikelihood(
                            W, logdet, Py, R, XtWX, n, d, c,
                        )
                        XtWX_inv = torch.linalg.inv(XtWX)
                        trace.append({"iter": em_it, "mode": "EM-fallback", "ll": ll})
                        if ll > best_ll:
                            best_ll = ll
                            best_Vg = Vg.clone()
                            best_Ve = Ve.clone()
                        if (em_it > it + 1 and
                                abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0)):
                            converged = True
                            break
                        Vg, Ve = _em_step(
                            Py, eigenvalues, W, X0_rot, XtWX_inv, Vg, Ve, n, c,
                        )
                    break

        except Exception as e:
            logger.warning("NR step failed: %s, using EM fallback.", e)
            Vg, Ve = _em_step(
                Py, eigenvalues, W, X0_rot, XtWX_inv, Vg, Ve, n, c,
            )
    else:
        logger.warning("PX-EM+NR did not converge in %d iterations.", max_iter)

    # Stash the converged flag in the last trace entry so wrappers can
    # plumb it through without breaking the existing 4-tuple contract.
    if trace:
        trace[-1]["converged"] = converged
    return best_Vg.detach(), best_Ve.detach(), best_ll, trace
