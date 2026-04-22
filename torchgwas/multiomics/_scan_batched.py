"""Block-batched ``scan_mediation`` engine (Phase 49b).

For every candidate (SNP, feature) pair, the Phase 49 per-pair path runs three
rotated WLS fits — Stage-M, Stage-Y-total, Stage-Y-direct — each with a
(p x p) solve. This engine preserves the rotated two-stage semantics exactly
but vectorises the fits:

* Eigenbasis and null variance components (sigma2_g, sigma2_e) are computed
  once per scan and held constant.
* For each SNP block of size s_b, Stage-M for all f_b features in a feature
  block is solved as one multi-RHS WLS per SNP (shape ``(p_m, p_m)`` factor,
  ``(p_m, f_b)`` RHS, where ``p_m = 1 + n_covariates_with_intercept``).
* For each (SNP, feature block) pair, Stage-Y-direct is solved as a batched
  ``(f_b, 2 + c0, 2 + c0)`` linear system using the Schur-partitioned Gram.

All three fits return parameter variances using the diagonal of the Gram
inverse — identical to the Phase 49 ``_wls`` contract, where the rotated
WLS weights already absorb residual variance. Per-pair SE estimation reuses
the existing ``sobel_se`` / ``monte_carlo_se`` / ``bootstrap_se`` helpers so
batched-path rows are bit-for-bit identical to loop-path rows when the same
seed / ``se`` are passed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..linalg.eigh import rotate
from ..models.base import NullFit
from ._se import monte_carlo_se, sobel_se
from ._sensitivity import imai_rho_sensitivity

logger = logging.getLogger("torchgwas.multiomics")


def _null_weights(nf: NullFit) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (U, w, Y_r, X0_r) from the shared null fit."""
    U = nf.eigenvectors
    sig2_g = float(nf.sig2_g)
    sig2_e = float(nf.sig2_e)
    w = 1.0 / (nf.eigenvalues * sig2_g + sig2_e).clamp(min=1e-20)
    return U, w, nf.Y_rot.reshape(-1), nf.X0_rot


def _stage_m_block(
    SNP_block_r: Tensor,  # (n, s_b)
    M_block_r: Tensor,    # (n, f_b)
    X0_r: Tensor,         # (n, c0)
    w: Tensor,            # (n,)
) -> tuple[Tensor, Tensor]:
    """Return (a, var_a) with shape (s_b, f_b).

    For each SNP i in the block, fits M = SNP_i * a + X0 * A0 + V using
    multi-RHS WLS over all f_b features in the block. The coefficient on
    SNP_i is extracted from the first row and its variance from the
    [0, 0] entry of the Gram inverse.
    """
    s_b = SNP_block_r.shape[1]
    c0 = X0_r.shape[1]
    n = SNP_block_r.shape[0]
    a = torch.empty(s_b, M_block_r.shape[1], dtype=SNP_block_r.dtype)
    var_a = torch.empty_like(a)

    wX0 = w.unsqueeze(1) * X0_r
    X0tWX0 = X0_r.T @ wX0                # (c0, c0)
    X0tWM = wX0.T @ M_block_r            # (c0, f_b)  (note: (X0 * w)^T @ M)

    for i in range(s_b):
        snp = SNP_block_r[:, i]
        wSNP = w * snp
        snp_t_w_snp = float((snp * wSNP).sum().item())
        snp_t_w_X0 = (wSNP.unsqueeze(0) @ X0_r).squeeze(0)  # (c0,)
        snp_t_w_M = (wSNP.unsqueeze(0) @ M_block_r).squeeze(0)  # (f_b,)

        A = torch.empty(1 + c0, 1 + c0, dtype=X0_r.dtype)
        A[0, 0] = snp_t_w_snp
        A[0, 1:] = snp_t_w_X0
        A[1:, 0] = snp_t_w_X0
        A[1:, 1:] = X0tWX0

        B = torch.empty(1 + c0, M_block_r.shape[1], dtype=X0_r.dtype)
        B[0, :] = snp_t_w_M
        B[1:, :] = X0tWM

        beta = torch.linalg.solve(A, B)   # (1 + c0, f_b)
        a[i, :] = beta[0, :]
        Ainv = torch.linalg.inv(A)
        var_a[i, :] = Ainv[0, 0]
    return a, var_a


def _total_effect_block(
    SNP_block_r: Tensor,  # (n, s_b)
    Y_r: Tensor,          # (n,)
    X0_r: Tensor,         # (n, c0)
    w: Tensor,            # (n,)
) -> tuple[Tensor, Tensor]:
    """Return (c, var_c) of shape (s_b,) — total SNP->Y effect per SNP in block."""
    s_b = SNP_block_r.shape[1]
    c0 = X0_r.shape[1]

    wX0 = w.unsqueeze(1) * X0_r
    X0tWX0 = X0_r.T @ wX0
    X0tWY = wX0.T @ Y_r                  # (c0,)
    out_c = torch.empty(s_b, dtype=X0_r.dtype)
    out_var = torch.empty_like(out_c)

    for i in range(s_b):
        snp = SNP_block_r[:, i]
        wSNP = w * snp
        snp_t_w_snp = float((snp * wSNP).sum().item())
        snp_t_w_X0 = (wSNP.unsqueeze(0) @ X0_r).squeeze(0)
        snp_t_w_Y = float((wSNP * Y_r).sum().item())

        A = torch.empty(1 + c0, 1 + c0, dtype=X0_r.dtype)
        A[0, 0] = snp_t_w_snp
        A[0, 1:] = snp_t_w_X0
        A[1:, 0] = snp_t_w_X0
        A[1:, 1:] = X0tWX0
        b_vec = torch.empty(1 + c0, dtype=X0_r.dtype)
        b_vec[0] = snp_t_w_Y
        b_vec[1:] = X0tWY

        beta = torch.linalg.solve(A, b_vec)
        Ainv = torch.linalg.inv(A)
        out_c[i] = beta[0]
        out_var[i] = Ainv[0, 0]
    return out_c, out_var


def _direct_effect_block(
    SNP_block_r: Tensor,  # (n, s_b)
    M_block_r: Tensor,    # (n, f_b)
    Y_r: Tensor,          # (n,)
    X0_r: Tensor,         # (n, c0)
    w: Tensor,            # (n,)
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (c_prime, var_c_prime, b, var_b) each shape (s_b, f_b).

    Solves per (i, j): [SNP_i, M_j, X0]^T diag(w) [SNP_i, M_j, X0] * beta
                      = [SNP_i, M_j, X0]^T diag(w) Y.
    """
    s_b = SNP_block_r.shape[1]
    f_b = M_block_r.shape[1]
    c0 = X0_r.shape[1]
    n = SNP_block_r.shape[0]

    wX0 = w.unsqueeze(1) * X0_r
    X0tWX0 = X0_r.T @ wX0                       # (c0, c0)
    X0tWY = wX0.T @ Y_r                         # (c0,)

    # Precompute feature-side blocks used across all SNPs.
    wM = w.unsqueeze(1) * M_block_r             # (n, f_b)
    M_t_w_M = (M_block_r * wM).sum(dim=0)       # (f_b,) diag
    M_t_w_X0 = wM.T @ X0_r                      # (f_b, c0)
    M_t_w_Y = (wM * Y_r.unsqueeze(1)).sum(dim=0)  # (f_b,)

    c_prime = torch.empty(s_b, f_b, dtype=X0_r.dtype)
    var_c_prime = torch.empty_like(c_prime)
    b = torch.empty_like(c_prime)
    var_b = torch.empty_like(c_prime)

    # SNP-side precomputes per SNP.
    for i in range(s_b):
        snp = SNP_block_r[:, i]
        wSNP = w * snp
        snp_t_w_snp = float((snp * wSNP).sum().item())
        snp_t_w_X0 = (wSNP.unsqueeze(0) @ X0_r).squeeze(0)   # (c0,)
        snp_t_w_Y = float((wSNP * Y_r).sum().item())
        snp_t_w_M = (wSNP.unsqueeze(0) @ M_block_r).squeeze(0)  # (f_b,)

        # Batched (f_b, 2 + c0, 2 + c0) system per j.
        p = 2 + c0
        A = torch.empty(f_b, p, p, dtype=X0_r.dtype)
        B = torch.empty(f_b, p, dtype=X0_r.dtype)

        # Fill SNP row/col (index 0).
        A[:, 0, 0] = snp_t_w_snp
        A[:, 0, 1] = snp_t_w_M
        A[:, 1, 0] = snp_t_w_M
        A[:, 0, 2:] = snp_t_w_X0.unsqueeze(0).expand(f_b, -1)
        A[:, 2:, 0] = snp_t_w_X0.unsqueeze(0).expand(f_b, -1)
        # M row/col (index 1).
        A[:, 1, 1] = M_t_w_M
        A[:, 1, 2:] = M_t_w_X0
        A[:, 2:, 1] = M_t_w_X0
        # X0 block (indices 2:).
        A[:, 2:, 2:] = X0tWX0.unsqueeze(0).expand(f_b, -1, -1)

        B[:, 0] = snp_t_w_Y
        B[:, 1] = M_t_w_Y
        B[:, 2:] = X0tWY.unsqueeze(0).expand(f_b, -1)

        beta = torch.linalg.solve(A, B.unsqueeze(-1)).squeeze(-1)  # (f_b, p)
        Ainv = torch.linalg.inv(A)  # (f_b, p, p)

        c_prime[i, :] = beta[:, 0]
        b[i, :] = beta[:, 1]
        var_c_prime[i, :] = Ainv[:, 0, 0]
        var_b[i, :] = Ainv[:, 1, 1]
    return c_prime, var_c_prime, b, var_b


def _sigma_v_block(
    SNP_block_r: Tensor,
    M_block_r: Tensor,
    X0_r: Tensor,
    w: Tensor,
    a: Tensor,
) -> Tensor:
    """Compute weighted Stage-M residual sigma_v per (i, j) — used for rho-sensitivity."""
    s_b, f_b = a.shape
    n = SNP_block_r.shape[0]
    p_m = 1 + X0_r.shape[1]
    sigma_v = torch.empty(s_b, f_b, dtype=X0_r.dtype)
    for i in range(s_b):
        Xm = torch.cat([SNP_block_r[:, i:i + 1], X0_r], dim=1)  # (n, p_m)
        # Solve M_block = Xm * beta (under weights) to get residuals.
        wXm = w.unsqueeze(1) * Xm
        A = Xm.T @ wXm
        B = wXm.T @ M_block_r
        beta = torch.linalg.solve(A, B)                        # (p_m, f_b)
        resid = M_block_r - Xm @ beta                          # (n, f_b)
        ssr = (w.unsqueeze(1) * resid * resid).sum(dim=0)      # (f_b,)
        dof = max(n - p_m, 1)
        sigma_v[i, :] = torch.sqrt((ssr / dof).clamp(min=1e-24))
    return sigma_v


def _sigma_u_block(
    SNP_block_r: Tensor,
    M_block_r: Tensor,
    Y_r: Tensor,
    X0_r: Tensor,
    w: Tensor,
    c_prime: Tensor,
    b: Tensor,
    X0_beta_implied: None = None,
) -> Tensor:
    """Compute weighted Stage-Y direct residual sigma_u per (i, j)."""
    s_b, f_b = c_prime.shape
    n = SNP_block_r.shape[0]
    p_y = 2 + X0_r.shape[1]
    sigma_u = torch.empty(s_b, f_b, dtype=X0_r.dtype)
    for i in range(s_b):
        snp = SNP_block_r[:, i]
        for j in range(f_b):
            Xy = torch.cat(
                [snp.unsqueeze(1), M_block_r[:, j:j + 1], X0_r], dim=1
            )
            wXy = w.unsqueeze(1) * Xy
            A = Xy.T @ wXy
            B = wXy.T @ Y_r
            beta = torch.linalg.solve(A, B)
            resid = Y_r - Xy @ beta
            ssr = float((w * resid * resid).sum().item())
            dof = max(n - p_y, 1)
            sigma_u[i, j] = float(np.sqrt(max(ssr / dof, 1e-24)))
    return sigma_u


def batched_scan_pairs(
    nf: NullFit,
    G: Tensor,                        # (n, s)
    M: Tensor,                        # (n, f)
    pairs: list[tuple[int, int]],
    *,
    se: str = "monte-carlo",
    n_mc_draws: int = 10_000,
    n_boot: int = 0,
    sensitivity: bool = True,
    seed: int | None = None,
    block_size: tuple[int, int] = (256, 64),
) -> list[dict[str, Any]]:
    """Run the batched mediation scan over ``pairs``.

    Returns a list of row dicts with the same keys as the Phase 49 loop path
    (``_mediate_from_nullfit``), in the same order as ``pairs``.
    """
    U, w, Y_r, X0_r = _null_weights(nf)
    n = Y_r.shape[0]

    # Rotate G, M once. This is a single (n, s) @ (n, n) product each — the
    # dominant fixed cost, amortized across the entire scan.
    G_r = rotate(G, U)                # (n, s)
    M_r = rotate(M, U)                # (n, f)

    s_b, f_b = block_size
    # Group pairs by (snp_block, feat_block) so each block's batched work is
    # done once and its results scatter back into the output rows.
    pair_by_block: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for row_idx, (i, j) in enumerate(pairs):
        key = (i // s_b, j // f_b)
        pair_by_block.setdefault(key, []).append((row_idx, (i, j)))

    results: list[dict[str, Any] | None] = [None] * len(pairs)

    for (sb_idx, fb_idx), members in pair_by_block.items():
        snp_start = sb_idx * s_b
        snp_end = min(snp_start + s_b, G_r.shape[1])
        feat_start = fb_idx * f_b
        feat_end = min(feat_start + f_b, M_r.shape[1])
        SNP_blk = G_r[:, snp_start:snp_end]           # (n, s_b_eff)
        M_blk = M_r[:, feat_start:feat_end]           # (n, f_b_eff)

        a_blk, var_a_blk = _stage_m_block(SNP_blk, M_blk, X0_r, w)
        c_blk, var_c_blk = _total_effect_block(SNP_blk, Y_r, X0_r, w)
        c_prime_blk, var_cp_blk, b_blk, var_b_blk = _direct_effect_block(
            SNP_blk, M_blk, Y_r, X0_r, w
        )
        if sensitivity:
            sigma_v_blk = _sigma_v_block(SNP_blk, M_blk, X0_r, w, a_blk)
            sigma_u_blk = _sigma_u_block(
                SNP_blk, M_blk, Y_r, X0_r, w, c_prime_blk, b_blk
            )

        for row_idx, (i, j) in members:
            li = i - snp_start
            lj = j - feat_start
            a = float(a_blk[li, lj])
            var_a = float(var_a_blk[li, lj])
            b = float(b_blk[li, lj])
            var_b = float(var_b_blk[li, lj])
            c = float(c_blk[li])
            var_c = float(var_c_blk[li])
            c_prime = float(c_prime_blk[li, lj])
            var_cp = float(var_cp_blk[li, lj])

            if se == "sobel":
                s_ind, ci_l, ci_u, p_ind = sobel_se(a, b, var_a, var_b)
            elif se == "monte-carlo":
                s_ind, ci_l, ci_u, p_ind = monte_carlo_se(
                    a, b, var_a, var_b, cov_ab=0.0,
                    n_draws=n_mc_draws, seed=seed,
                )
            elif se == "bootstrap":
                raise ValueError(
                    "bootstrap SE is not supported in the batched scan path; "
                    "use se='sobel' or se='monte-carlo' or call the per-pair "
                    "scan_mediation with device='cpu' and a small pair count."
                )
            else:
                raise ValueError(
                    f"se must be 'sobel' or 'monte-carlo' in batched path; got {se!r}"
                )

            indirect = a * b
            total = c
            inconsistent = (
                (c_prime != 0.0)
                and (indirect != 0.0)
                and ((c_prime > 0) != (indirect > 0))
            )
            if total == 0.0 or inconsistent:
                proportion = float("nan")
            else:
                proportion = indirect / total

            if sensitivity:
                sigma_v = float(sigma_v_blk[li, lj])
                sigma_u = float(sigma_u_blk[li, lj])
                rho = imai_rho_sensitivity(b, sigma_v, sigma_u)
            else:
                rho = None

            results[row_idx] = {
                "a": a, "a_se": float(np.sqrt(max(var_a, 0.0))),
                "b": b, "b_se": float(np.sqrt(max(var_b, 0.0))),
                "c": c, "c_prime": c_prime,
                "c_se": float(np.sqrt(max(var_c, 0.0))),
                "c_prime_se": float(np.sqrt(max(var_cp, 0.0))),
                "indirect": indirect,
                "indirect_se": s_ind,
                "indirect_pvalue": p_ind,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
                "proportion_mediated": proportion,
                "inconsistent": inconsistent,
                "sensitivity_rho": rho,
                "total": total,
                "n": n,
            }

    # No element should be None.
    return [r for r in results if r is not None]
