"""Mode A: EM-REML warm-start (3-10 iterations) for variance components.

Monotonic convergence guaranteed — keeps parameters in valid space.
Used before AI-REML to stabilize from poor initial values.

In rotated eigenspace, the EM updates for the two variance components are:

    sig2_g_new = (u^T u + sig2_e * tr(C_uu)) / n
    sig2_e_new = (e^T e + sig2_e * tr(C_ee)) / n

where u = BLUP of random effects, e = residuals, and C_uu, C_ee are
the relevant blocks of the mixed-model coefficient matrix inverse.

For single-trait in rotated space, these simplify to O(n) sums.
"""

from __future__ import annotations

import logging

from torch import Tensor

logger = logging.getLogger(__name__)


def px_em_warmstart(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    n_iter: int = 5,
    sig2_g_init: float = 0.5,
    sig2_e_init: float = 0.5,
) -> tuple[float, float, list[dict]]:
    """EM-REML warm-start for single-trait variance components.

    Parameters
    ----------
    Y_rot : (n,) or (n, 1) — rotated phenotype
    X0_rot : (n, c) — rotated covariates
    eigenvalues : (n,) — GRM eigenvalues (descending)
    n_iter : int — number of EM iterations
    sig2_g_init, sig2_e_init : initial variance components

    Returns
    -------
    sig2_g : float — warm-started genetic variance
    sig2_e : float — warm-started residual variance
    trace : list[dict] — iteration trace
    """
    import torch

    y = Y_rot.squeeze()  # (n,)
    X = X0_rot  # (n, c)
    n = y.shape[0]
    c = X.shape[1]
    df = n - c
    evals = eigenvalues  # (n,)

    sig2_g = sig2_g_init
    sig2_e = sig2_e_init
    trace: list[dict] = []

    for it in range(n_iter):
        # H diagonal in rotated space: evals * sig2_g + sig2_e
        H_diag = evals * sig2_g + sig2_e
        H_inv = 1.0 / H_diag  # (n,)

        # Weighted LS for fixed effects: beta = (X^T H^{-1} X)^{-1} X^T H^{-1} y
        wX = H_inv.unsqueeze(1) * X  # (n, c)
        XtHiX = X.T @ wX  # (c, c)
        XtHiy = X.T @ (H_inv * y)  # (c,)
        beta = torch.linalg.solve(XtHiX, XtHiy)

        # Residuals
        resid = y - X @ beta  # (n,)

        # P operator diagonal for EM update
        # P = H^{-1} - H^{-1} X (X^T H^{-1} X)^{-1} X^T H^{-1}
        wX_XtHiXi = wX @ torch.linalg.inv(XtHiX)  # (n, c)
        P_diag = H_inv - (wX_XtHiXi * wX).sum(dim=1)  # (n,)

        # Py
        Py = H_inv * resid - wX @ (torch.linalg.inv(XtHiX) @ (X.T @ (H_inv * resid)))

        # EM updates for variance components:
        # sig2_g_new = sig2_g^2 * (Py^T evals Py + tr(P diag(evals)) * sig2_g ???)
        # Simpler: use the standard EM-REML formulas in rotated space.
        #
        # For single-trait, the EM update is:
        #   sig2_g_new = sig2_g + sig2_g^2 * [y^T P K P y - tr(P K)] / n_eff
        #   sig2_e_new = sig2_e + sig2_e^2 * [y^T P P y - tr(P)] / n_eff
        #
        # In rotated space, K = diag(evals), so:
        #   y^T P K P y = Py^T diag(evals) Py
        #   tr(P K) = sum(P_diag * evals)
        #   y^T P P y = Py^T Py
        #   tr(P) = sum(P_diag)

        yPKPy = (Py * evals * Py).sum().item()
        trPK = (P_diag * evals).sum().item()
        yPPy = (Py * Py).sum().item()
        trP = P_diag.sum().item()

        # EM update (Zhou et al. 2019 form; guaranteed non-negative)
        sig2_g_new = sig2_g ** 2 * yPKPy / df + sig2_g - sig2_g ** 2 * trPK / df
        sig2_e_new = sig2_e ** 2 * yPPy / df + sig2_e - sig2_e ** 2 * trP / df

        # Ensure positivity
        sig2_g_new = max(sig2_g_new, 1e-10)
        sig2_e_new = max(sig2_e_new, 1e-10)

        trace.append({
            "iter": it,
            "mode": "PX-EM",
            "sig2_g": sig2_g_new,
            "sig2_e": sig2_e_new,
        })

        sig2_g = sig2_g_new
        sig2_e = sig2_e_new

    logger.info(
        "PX-EM warm-start (%d iters): sig2_g=%.6e, sig2_e=%.6e",
        n_iter, sig2_g, sig2_e,
    )

    return sig2_g, sig2_e, trace
