"""Multi-trait REML log-likelihood with EED (Efficient Eigen-Decomposition).

In rotated eigenspace, the joint covariance decomposes into n independent
d×d blocks: Sigma_i = evals[i] * Vg + Ve.  The REML log-likelihood
reduces to sums of d-dimensional log-determinants and quadratic forms.

References
----------
Zhou & Stephens, Nat Methods 2014 — Equations (1)-(8).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_sigma_inv(
    eigenvalues: Tensor,
    Vg: Tensor,
    Ve: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute per-individual Sigma_i^{-1} and log|Sigma_i|.

    Parameters
    ----------
    eigenvalues : (n,)
    Vg : (d, d) genetic covariance
    Ve : (d, d) residual covariance

    Returns
    -------
    W : (n, d, d) — per-individual inverse covariance
    logdet : (n,) — log-determinants
    """
    n = eigenvalues.shape[0]
    d = Vg.shape[0]

    # Sigma_i = evals[i] * Vg + Ve  → (n, d, d)
    Sigma = eigenvalues.view(n, 1, 1) * Vg.unsqueeze(0) + Ve.unsqueeze(0)

    # Batched Cholesky and inverse
    L = torch.linalg.cholesky(Sigma)  # (n, d, d)
    W = torch.cholesky_inverse(L)  # (n, d, d) = Sigma^{-1}

    # Log determinant: 2 * sum(log(diag(L)))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (n,)

    return W, logdet


def compute_sigma_inv_diag(
    eigenvalues: Tensor,
    ked,
) -> tuple[Tensor, Tensor]:
    """Compute diagonal precision via KED (Kronecker Eigenspace Diag).

    For separable Vg and Ve, the per-individual covariance in the KED basis
    is diagonal. Returns (n, dE) diagonal weights instead of (n, d, d) dense.

    Parameters
    ----------
    eigenvalues : (n,)
    ked : KronEED result from kronecker_eed()

    Returns
    -------
    W_diag : (n, dE) — diagonal precision weights
    logdet : (n,) — log-determinants
    """
    from ..linalg.kronecker_eed import diagonal_precision
    return diagonal_precision(ked, eigenvalues)


def mvlmm_reml_loglikelihood(
    Vg: Tensor,
    Ve: Tensor,
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
) -> float:
    """Evaluate the multi-trait REML log-likelihood.

    Parameters
    ----------
    Vg : (d, d) — genetic covariance matrix
    Ve : (d, d) — residual covariance matrix
    Y_rot : (n, d) — rotated phenotypes
    X0_rot : (n, c) — rotated covariates
    eigenvalues : (n,) — GRM eigenvalues

    Returns
    -------
    ll : float — REML log-likelihood
    """
    n, d = Y_rot.shape
    c = X0_rot.shape[1]
    df = n - c

    W, logdet_sigma = compute_sigma_inv(eigenvalues, Vg, Ve)  # (n,d,d), (n,)

    # GLS for fixed effects: B = (X^T V^{-1} X)^{-1} X^T V^{-1} Y
    # In rotated space with per-individual blocks:
    # M00 = sum_i kron(W_i, x_i x_i^T) — but since x_i is a row vector,
    # we build M00 as (c*d, c*d) or use the vec trick.
    # Simpler: treat it as d separate weighted LS problems with coupling.

    # Build block normal equations: (cd × cd) system
    # M00[a*d+s, b*d+t] = sum_i X_rot[i,a] * W[i,s,t] * X_rot[i,b]
    # Use einsum for efficiency:
    # M00 = einsum('ia,ist,ib->asbt', X_rot, W, X_rot) reshaped to (cd, cd)
    XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)  # (c, d, c, d)
    XtWX = XtWX.permute(0, 1, 2, 3).reshape(c * d, c * d)

    # RHS: b0[a*d+t] = sum_i X_rot[i,a] * sum_s W[i,t,s] * Y_rot[i,s]
    WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, d)
    XtWY = torch.einsum('ia,it->at', X0_rot, WY)  # (c, d)
    XtWY_vec = XtWY.reshape(c * d)

    # Solve for B (vectorized)
    B_vec = torch.linalg.solve(XtWX, XtWY_vec)  # (c*d,)
    B = B_vec.reshape(c, d)  # (c, d)

    # Residuals
    R = Y_rot - X0_rot @ B  # (n, d)

    # Quadratic form: sum_i r_i^T W_i r_i
    WR = torch.einsum('ist,is->it', W, R)  # (n, d)
    quad = (R * WR).sum()

    # Log-likelihood
    # -2 ll = sum_i log|Sigma_i| + quad + df*d*log(2*pi) + log|X^T V^{-1} X|
    sum_logdet = logdet_sigma.sum()
    sign, logdet_XtWX = torch.linalg.slogdet(XtWX)
    logdet_XtWX_val = logdet_XtWX.item() if sign > 0 else 0.0

    neg2ll = sum_logdet.item() + quad.item() + df * d * math.log(2 * math.pi) + logdet_XtWX_val
    ll = -0.5 * neg2ll

    return ll


def mvlmm_null_quantities(
    Vg: Tensor,
    Ve: Tensor,
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
) -> dict:
    """Precompute null-model quantities for the Schur complement scan.

    Returns
    -------
    dict with: W, B, M00, M00_inv, b0, R (residuals)
    """
    n, d = Y_rot.shape
    c = X0_rot.shape[1]

    W, logdet_sigma = compute_sigma_inv(eigenvalues, Vg, Ve)

    # Normal equations
    XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)
    XtWX = XtWX.permute(0, 1, 2, 3).reshape(c * d, c * d)

    WY = torch.einsum('ist,is->it', W, Y_rot)
    XtWY = torch.einsum('ia,it->at', X0_rot, WY)
    XtWY_vec = XtWY.reshape(c * d)

    B_vec = torch.linalg.solve(XtWX, XtWY_vec)
    B = B_vec.reshape(c, d)

    M00_inv = torch.linalg.inv(XtWX)

    return {
        "W": W,               # (n, d, d)
        "B": B,               # (c, d)
        "M00": XtWX,          # (cd, cd)
        "M00_inv": M00_inv,   # (cd, cd)
        "b0": XtWY_vec,       # (cd,)
        "logdet_sigma": logdet_sigma,  # (n,)
    }
