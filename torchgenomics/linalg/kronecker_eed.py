"""Kronecker Eigenspace Diagonalization (KED) for separable covariance models.

Joint diagonalization of Vg and Ve via Cholesky whitening + eigendecomposition.
For separable Vg = Vg_t kron Vg_e and Ve = Ve_t kron Ve_e:
    1. Cholesky-whiten each factor: Le_t^{-1} Vg_t Le_t^{-T} → Qt Λt Qt'
    2. Transform: T_t = Le_t^{-T} Qt, T_e = Le_e^{-T} Qe
    3. In the (T_t kron T_e) basis: Vg → diag(Λt kron Λe), Ve → I
    4. Per-individual Sigma_i = lambda_K,i * diag(Λt kron Λe) + I → diagonal!

This reduces W from dense (n, dE, dE) to diagonal (n, dE).
Memory: O(n*dE) instead of O(n*dE^2). Exact — no approximation.

References:
    - Zhou & Stephens 2014 (GEMMA joint diagonalization for multi-trait)
    - Thompson et al. 2003 (separable Kronecker in mixed models)
    - Meyer 2009 (Factor Analytic in genetics)
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


class KronEED(NamedTuple):
    """Result of Kronecker eigenspace diagonalization."""

    Tt: Tensor       # (d, d) trait transform: Le_t^{-T} Qt
    Te: Tensor       # (E, E) env transform: Le_e^{-T} Qe
    lam_t: Tensor    # (d,)   eigenvalues of Le_t^{-1} Vg_t Le_t^{-T}
    lam_e: Tensor    # (E,)   eigenvalues of Le_e^{-1} Vg_e Le_e^{-T}
    d: int           # number of traits
    E: int           # number of environments


def kronecker_eed(
    Vg_t: Tensor,
    Vg_e: Tensor,
    Ve_t: Tensor,
    Ve_e: Tensor,
) -> KronEED:
    """Joint diagonalization for separable Kronecker covariance.

    Cholesky-whitens Ve from each factor of Vg, then eigendecomposes.
    In the resulting basis, Vg is diagonal and Ve = I.

    Parameters
    ----------
    Vg_t : (d, d) trait genetic covariance
    Vg_e : (E, E) environment genetic covariance
    Ve_t : (d, d) trait residual covariance
    Ve_e : (E, E) env residual covariance

    Returns
    -------
    KronEED with transforms and eigenvalues for diagonal precision.
    """
    d = Vg_t.shape[0]
    E = Vg_e.shape[0]

    Tt, lam_t = _joint_diag_factor(Vg_t, Ve_t)
    Te, lam_e = _joint_diag_factor(Vg_e, Ve_e)

    return KronEED(Tt=Tt, Te=Te, lam_t=lam_t, lam_e=lam_e, d=d, E=E)


def _joint_diag_factor(
    Vg: Tensor,
    Ve: Tensor,
) -> tuple[Tensor, Tensor]:
    """Joint-diagonalize one factor: Cholesky-whiten Ve, eigendecompose Vg.

    Returns transform T = Le^{-T} Q and eigenvalues lambda such that:
        T' Vg T = diag(lambda)
        T' Ve T = I

    Parameters
    ----------
    Vg : (p, p) genetic covariance factor
    Ve : (p, p) residual covariance factor

    Returns
    -------
    T : (p, p) joint transform
    lam : (p,) eigenvalues
    """
    Vg = Vg.to(torch.float64)
    Ve = Ve.to(torch.float64)

    # Add jitter for numerical stability
    p = Vg.shape[0]
    jitter = 1e-8 * torch.eye(p, dtype=torch.float64, device=Vg.device)
    Ve_safe = Ve + jitter

    # Cholesky of Ve
    Le = torch.linalg.cholesky(Ve_safe)  # (p, p) lower-triangular
    Le_inv = torch.linalg.inv(Le)  # Le^{-1}

    # Whiten Vg: Le^{-1} Vg Le^{-T}
    Vg_w = Le_inv @ Vg @ Le_inv.T

    # Symmetrize
    Vg_w = (Vg_w + Vg_w.T) / 2.0

    # Eigendecompose whitened Vg
    lam, Q = torch.linalg.eigh(Vg_w)
    lam = lam.flip(0).clamp(min=0.0)
    Q = Q.flip(1)

    # Transform: T = Le^{-T} Q
    T = Le_inv.T @ Q

    return T.to(STAT_DTYPE), lam.to(STAT_DTYPE)


def kronecker_eed_from_full(
    Vg: Tensor,
    Ve: Tensor,
    d: int,
    E: int,
) -> KronEED:
    """KED from full (dE, dE) covariance matrices when they are separable.

    Extracts trait and env factors from the full matrices, then applies KED.
    Used when the REML output is full Vg/Ve but the model is separable.

    Parameters
    ----------
    Vg : (dE, dE) full genetic covariance (assumed separable)
    Ve : (dE, dE) full residual covariance (assumed separable)
    d : number of traits
    E : number of environments

    Returns
    -------
    KronEED
    """
    # Extract approximate trait and env factors from full Vg
    # Vg_t[a,b] ≈ mean over e of Vg[a*E+e, b*E+e]
    # Vg_e[e,f] ≈ mean over a of Vg[a*E+e, a*E+f] / Vg_t[a,a]
    Vg_t = torch.zeros(d, d, dtype=STAT_DTYPE, device=Vg.device)
    for a in range(d):
        for b in range(d):
            vals = Vg[a * E:(a + 1) * E, b * E:(b + 1) * E]
            Vg_t[a, b] = vals.diag().mean()

    Vg_e = torch.zeros(E, E, dtype=STAT_DTYPE, device=Vg.device)
    scale = Vg_t.diag().mean().clamp(min=1e-10)
    for e in range(E):
        for f in range(E):
            vals = torch.stack([Vg[a * E + e, a * E + f] for a in range(d)])
            Vg_e[e, f] = vals.mean() / scale

    Ve_t = torch.zeros(d, d, dtype=STAT_DTYPE, device=Ve.device)
    for a in range(d):
        for b in range(d):
            vals = Ve[a * E:(a + 1) * E, b * E:(b + 1) * E]
            Ve_t[a, b] = vals.diag().mean()

    Ve_e = torch.zeros(E, E, dtype=STAT_DTYPE, device=Ve.device)
    scale_e = Ve_t.diag().mean().clamp(min=1e-10)
    for e in range(E):
        for f in range(E):
            vals = torch.stack([Ve[a * E + e, a * E + f] for a in range(d)])
            Ve_e[e, f] = vals.mean() / scale_e

    return kronecker_eed(Vg_t, Vg_e, Ve_t, Ve_e)


def diagonal_precision(
    ked: KronEED,
    eigenvalues_K: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute diagonal precision weights W and log-determinants.

    In the KED basis: Vg → diag(lam_t kron lam_e), Ve → I.
    So Sigma_i = lambda_K[i] * diag(lam_t kron lam_e) + I.

    Parameters
    ----------
    ked : KronEED result
    eigenvalues_K : (n,) GRM eigenvalues

    Returns
    -------
    W_diag : (n, dE) diagonal precision weights
    logdet : (n,) log-determinants
    """
    n = eigenvalues_K.shape[0]
    d, E = ked.d, ked.E

    # Sigma_diag[i, a*E+b] = lambda_K[i] * lam_t[a] * lam_e[b] + 1.0
    # Shape: (n, 1, 1) * (1, d, 1) * (1, 1, E) + 1.0 → (n, d, E)
    Sigma_diag = (
        eigenvalues_K.view(n, 1, 1)
        * ked.lam_t.view(1, d, 1)
        * ked.lam_e.view(1, 1, E)
        + 1.0
    ).reshape(n, d * E)

    Sigma_diag = Sigma_diag.clamp(min=1e-20)
    W_diag = 1.0 / Sigma_diag
    logdet = torch.log(Sigma_diag).sum(dim=1)

    return W_diag, logdet


def rotate_to_ked_basis(
    M: Tensor,
    Tt: Tensor,
    Te: Tensor,
    d: int,
    E: int,
) -> Tensor:
    """Rotate matrix M into KED basis without forming full Kronecker.

    Applies (Tt kron Te)^{-1} @ M = (Tt^{-1} kron Te^{-1}) @ M
    by reshaping and applying Tt^{-1}, Te^{-1} separately.
    Since T = Le^{-T} Q and T^{-1} = Q' Le, we use T^{-1} = Tt' or Te'.
    Actually for joint diag: T'Vg T = Λ, T'Ve T = I, so T^{-1} = Ve T = Le Le' T.
    Simpler: just use Tt' and Te' since T' is the left-inverse when T'Ve T = I
    i.e. T^{-1} = T' Ve. But for data rotation we want (T kron T)^{-T} basis.

    For the score test, we need data in the basis where Sigma is diagonal.
    That basis is: y_ked = (Tt kron Te)' @ y_rot  (NOT inverse).

    Parameters
    ----------
    M : (n, dE) data in trait-major order (col j = trait a, env b where j = a*E+b)
    Tt : (d, d) trait transform from KED
    Te : (E, E) env transform from KED
    d, E : dimensions

    Returns
    -------
    M_ked : (n, dE) rotated into KED basis
    """
    n = M.shape[0]
    # Reshape: (n, dE) → (n, d, E)
    M3 = M.reshape(n, d, E)

    # T' on trait axis: Tt' @ M[..., d, ...] for each n, E
    M3 = torch.einsum('td,nde->nte', Tt.T, M3)

    # T' on env axis: Te' @ M[..., E] for each n, d
    M3 = torch.einsum('ef,ndf->nde', Te.T, M3)

    return M3.reshape(n, d * E)


def inverse_rotate_from_ked(
    M_ked: Tensor,
    Tt: Tensor,
    Te: Tensor,
    d: int,
    E: int,
) -> Tensor:
    """Rotate from KED basis back to K-eigenspace: M = (Tt kron Te) @ M_ked."""
    n = M_ked.shape[0]
    M3 = M_ked.reshape(n, d, E)
    M3 = torch.einsum('td,nde->nte', Tt, M3)
    M3 = torch.einsum('ef,ndf->nde', Te, M3)
    return M3.reshape(n, d * E)


def woodbury_fa_precision(
    Lambda: Tensor,
    psi: Tensor,
    eigenvalues_K: Tensor,
    lam_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Precision weights for FA(k) environmental covariance via Woodbury.

    For Vg_env = Lambda Lambda' + diag(psi) with FA(k) structure and Ve = I,
    in the trait-whitened basis where Vg_t → diag(lam_t) and Ve_t → I:

    Sigma_i[a-block] = lambda_K[i] * lam_t[a] * (Lambda Lambda' + diag(psi)) + I_E

    The low-rank part Lambda Lambda' is handled via Woodbury:
    (D + scale * Lambda Lambda')^{-1} = D^{-1} - D^{-1}(scale*Lambda)(I + scale*Lambda'D^{-1}Lambda)^{-1}(scale*Lambda)'D^{-1}

    where D = lambda_K[i] * lam_t[a] * diag(psi) + I_E.

    Parameters
    ----------
    Lambda : (E, k) FA loading matrix
    psi : (E,) FA specific variances
    eigenvalues_K : (n,) GRM eigenvalues
    lam_t : (d,) trait eigenvalues from KED

    Returns
    -------
    W_diag : (n, dE) diagonal precision (exact for FA diagonal blocks)
    logdet : (n,) log-determinants
    """
    n = eigenvalues_K.shape[0]
    d = lam_t.shape[0]
    E = Lambda.shape[0]
    k = Lambda.shape[1]
    device = Lambda.device

    W_diag = torch.zeros(n, d * E, dtype=STAT_DTYPE, device=device)
    logdet = torch.zeros(n, dtype=STAT_DTYPE, device=device)

    for a in range(d):
        # Process in batches for memory efficiency
        for i0 in range(0, n, 512):
            i1 = min(i0 + 512, n)
            bn = i1 - i0
            lam_Ki = eigenvalues_K[i0:i1]  # (bn,)

            # scale = lambda_K[i] * lam_t[a]
            scale = lam_Ki * lam_t[a]  # (bn,)

            # D_diag = scale * psi + 1
            D_diag = scale.unsqueeze(1) * psi.unsqueeze(0) + 1.0  # (bn, E)
            D_inv = 1.0 / D_diag.clamp(min=1e-20)  # (bn, E)

            # scale * Lambda: (bn, E, k)
            sL = scale.view(bn, 1, 1) * Lambda.unsqueeze(0)

            # D^{-1} Lambda: (bn, E, k)
            DinvL = D_inv.unsqueeze(2) * Lambda.unsqueeze(0)

            # Core = I_k + scale * Lambda' D^{-1} Lambda: (bn, k, k)
            core = torch.eye(k, dtype=STAT_DTYPE, device=device).unsqueeze(0) + \
                torch.bmm(sL.transpose(1, 2), DinvL)

            try:
                core_inv = torch.linalg.inv(core)
            except Exception:
                core_inv = torch.linalg.pinv(core)

            # Correction diagonal: (D^{-1} sL core^{-1} sL' D^{-1})_diag
            DinvL_ci = torch.bmm(DinvL, core_inv)  # (bn, E, k)
            sDinvL = sL * D_inv.unsqueeze(2)  # (bn, E, k)
            corr = (DinvL_ci * sDinvL).sum(dim=2)  # (bn, E)

            W_block = D_inv - corr  # (bn, E)
            W_diag[i0:i1, a * E:(a + 1) * E] = W_block

            # logdet: log|D| + log|core|
            logdet[i0:i1] += torch.log(D_diag).sum(dim=1)
            logdet[i0:i1] += torch.linalg.slogdet(core)[1]

    return W_diag, logdet


def ked_reml_quantities(
    ked: KronEED,
    eigenvalues_K: Tensor,
    Y_rot: Tensor,
    X0_rot: Tensor,
) -> dict:
    """Precompute null-model quantities using KED diagonal precision.

    Equivalent to mvlmm_null_quantities but with diagonal W.

    Parameters
    ----------
    ked : KronEED result
    eigenvalues_K : (n,) GRM eigenvalues
    Y_rot : (n, dE) rotated phenotypes (K-eigenspace)
    X0_rot : (n, c) rotated covariates (K-eigenspace)

    Returns
    -------
    dict with: W_diag, B, XtWX_blocks, XtWX_inv_blocks, residuals_ked,
               Y_ked, logdet, Tt, Te, d, E
    """
    d, E = ked.d, ked.E
    dE = d * E
    n, c = X0_rot.shape

    # Rotate phenotypes into KED basis
    Y_ked = rotate_to_ked_basis(Y_rot, ked.Tt, ked.Te, d, E)
    # Covariates are scalar per individual — broadcast across trait-env dims
    # In KED basis, covariates don't need trait-env rotation
    X0_ked = X0_rot

    # Diagonal precision
    W_diag, logdet = diagonal_precision(ked, eigenvalues_K)  # (n, dE), (n,)

    # GLS fixed effects with diagonal W.
    # Since W is diagonal, the normal equations decouple per trait-env dim:
    # For each j in [0, dE): XtWX_j = X0' diag(W[:,j]) X0 — (c, c)
    # B_j = XtWX_j^{-1} X0' diag(W[:,j]) Y_ked[:,j]

    # XtWX_blocks[j, a, b] = sum_i X0[i,a] * W[i,j] * X0[i,b]
    # Vectorized: (n, dE, c) = W[:, :, None] * X0[:, None, :]
    # then XtWX_blocks = X0.T @ (W * X0) per j — use einsum
    # XtWX_blocks[j, a, b] = sum_i X0[i,a] * W[i,j] * X0[i,b]
    XtWX_blocks = torch.einsum('na,nj,nb->jab', X0_ked, W_diag, X0_ked)  # (dE, c, c)

    WY = W_diag * Y_ked  # (n, dE)
    XtWY = X0_ked.T @ WY  # (c, dE)
    XtWY = XtWY.T  # (dE, c)

    try:
        B_blocks = torch.linalg.solve(XtWX_blocks, XtWY.unsqueeze(2))  # (dE, c, 1)
        B = B_blocks.squeeze(2)  # (dE, c)
    except Exception:
        B = torch.zeros(dE, c, dtype=STAT_DTYPE, device=Y_rot.device)
        for j in range(dE):
            try:
                B[j] = torch.linalg.solve(XtWX_blocks[j], XtWY[j])
            except Exception:
                pass

    # Residuals in KED basis
    R_ked = Y_ked - X0_ked @ B.T  # (n, dE)

    # XtWX inverse blocks for Schur complement in score test
    try:
        XtWX_inv_blocks = torch.linalg.inv(XtWX_blocks)  # (dE, c, c)
    except Exception:
        XtWX_inv_blocks = torch.linalg.pinv(XtWX_blocks)

    return {
        "W_diag": W_diag,                    # (n, dE)
        "B": B,                              # (dE, c)
        "XtWX_blocks": XtWX_blocks,          # (dE, c, c)
        "XtWX_inv_blocks": XtWX_inv_blocks,  # (dE, c, c)
        "residuals_ked": R_ked,              # (n, dE)
        "Y_ked": Y_ked,                      # (n, dE)
        "logdet": logdet,                    # (n,)
        "Tt": ked.Tt,                        # (d, d)
        "Te": ked.Te,                        # (E, E)
        "d": d,
        "E": E,
    }
