"""Basis function evaluation for random regression models.

Implements:
- Normalized Legendre polynomials on standardized time [-1, 1] (Bonnet recurrence)
- B-spline basis via Cox-de Boor recursion
- 2D tensor-product P-spline bases for spatial smoothing
- Difference penalty matrices for P-spline regularization
- Knot placement utilities

All functions are pure-tensor (torch.float64) with no model state, making them
reusable for random regression GWAS, longitudinal random effects, and field-trial
spatial correction.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import STAT_DTYPE


# ── Time standardization ────────────────────────────────────────────


def standardize_time(
    t: Tensor,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[Tensor, float, float]:
    """Map raw time values to the interval [-1, 1].

    Used to bring time values into the natural domain of Legendre polynomials.
    The min/max are returned so that new time points can be re-standardized
    against the same reference range.

    Parameters
    ----------
    t : Tensor
        Raw time values (any shape).
    t_min, t_max : float, optional
        Reference min and max for standardization. If omitted, taken from
        ``t``.

    Returns
    -------
    t_std : Tensor
        Standardized values in [-1, 1].
    t_min : float
        Reference minimum used.
    t_max : float
        Reference maximum used.
    """
    t = t.to(STAT_DTYPE)
    if t_min is None:
        t_min = float(t.min().item())
    if t_max is None:
        t_max = float(t.max().item())
    if t_max == t_min:
        raise ValueError("standardize_time: t_max == t_min (all times identical).")
    t_std = 2.0 * (t - t_min) / (t_max - t_min) - 1.0
    return t_std, t_min, t_max


# ── Legendre polynomials ────────────────────────────────────────────


def legendre_basis(t_std: Tensor, order: int) -> Tensor:
    """Evaluate normalized Legendre polynomials of orders 0..order at t_std.

    Uses the Bonnet three-term recurrence:
        (k+1) P_{k+1}(t) = (2k+1) t P_k(t) - k P_{k-1}(t)
    Then normalizes each P_k by sqrt((2k+1)/2) so that
        ∫_{-1}^{1} P_k(t)^2 dt = 1.

    Parameters
    ----------
    t_std : Tensor
        Time values, expected in [-1, 1] (call :func:`standardize_time` first).
    order : int
        Highest polynomial order. Returns ``order + 1`` basis columns.

    Returns
    -------
    Phi : Tensor
        ``(T, order + 1)`` design matrix of normalized Legendre polynomials.
    """
    if order < 0:
        raise ValueError("legendre_basis: order must be >= 0.")

    t_std = t_std.to(STAT_DTYPE).reshape(-1)
    T = t_std.shape[0]
    b = order + 1

    Phi = torch.zeros(T, b, dtype=STAT_DTYPE, device=t_std.device)
    # Unnormalized P_0(t) = 1
    Phi[:, 0] = 1.0
    if order >= 1:
        Phi[:, 1] = t_std
    for k in range(1, order):
        # (k+1) P_{k+1} = (2k+1) t P_k - k P_{k-1}
        Phi[:, k + 1] = ((2 * k + 1) * t_std * Phi[:, k] - k * Phi[:, k - 1]) / (k + 1)

    # Normalize: ∫_{-1}^{1} P_k^2 dt = 2 / (2k+1)  ⇒  norm by sqrt((2k+1)/2)
    norms = torch.tensor(
        [((2 * k + 1) / 2.0) ** 0.5 for k in range(b)],
        dtype=STAT_DTYPE,
        device=t_std.device,
    )
    return Phi * norms.unsqueeze(0)


# ── B-spline basis ──────────────────────────────────────────────────


def place_knots(
    t: Tensor,
    n_interior: int,
    kind: str = "quantile",
    degree: int = 3,
) -> Tensor:
    """Place knots for a B-spline basis.

    Parameters
    ----------
    t : Tensor
        Time values used to anchor knot placement.
    n_interior : int
        Number of interior knots (excluding boundary).
    kind : {"quantile", "uniform", "extended"}
        - ``quantile``: equal-count placement based on data quantiles.
        - ``uniform``: equally spaced between min and max.
        - ``extended``: uniform with extra boundary knots padded by ``degree``.
    degree : int
        B-spline degree. Used for boundary padding when ``kind="extended"``.

    Returns
    -------
    knots : Tensor
        1D tensor of knot positions, strictly increasing.
    """
    t = t.to(STAT_DTYPE).reshape(-1)
    t_min = float(t.min().item())
    t_max = float(t.max().item())
    if n_interior < 0:
        raise ValueError("place_knots: n_interior must be >= 0.")

    if kind == "quantile":
        if n_interior == 0:
            interior = torch.tensor([], dtype=STAT_DTYPE, device=t.device)
        else:
            qs = torch.linspace(0.0, 1.0, n_interior + 2, dtype=STAT_DTYPE)[1:-1]
            interior = torch.quantile(t, qs)
        knots = torch.cat(
            [
                torch.tensor([t_min], dtype=STAT_DTYPE, device=t.device),
                interior.to(t.device),
                torch.tensor([t_max], dtype=STAT_DTYPE, device=t.device),
            ]
        )
    elif kind == "uniform":
        knots = torch.linspace(
            t_min, t_max, n_interior + 2, dtype=STAT_DTYPE, device=t.device
        )
    elif kind == "extended":
        spacing = (t_max - t_min) / max(n_interior + 1, 1)
        left = torch.tensor(
            [t_min - (degree - i) * spacing for i in range(degree)],
            dtype=STAT_DTYPE,
            device=t.device,
        )
        middle = torch.linspace(
            t_min, t_max, n_interior + 2, dtype=STAT_DTYPE, device=t.device
        )
        right = torch.tensor(
            [t_max + (i + 1) * spacing for i in range(degree)],
            dtype=STAT_DTYPE,
            device=t.device,
        )
        knots = torch.cat([left, middle, right])
    else:
        raise ValueError(f"place_knots: unknown kind '{kind}'.")

    # Enforce strict monotonicity (jitter ties)
    diffs = knots[1:] - knots[:-1]
    if (diffs <= 0).any():
        eps = (t_max - t_min) * 1e-9 + 1e-12
        for i in range(1, len(knots)):
            if knots[i] <= knots[i - 1]:
                knots[i] = knots[i - 1] + eps

    return knots


def bspline_basis(
    t: Tensor,
    knots: Tensor,
    degree: int = 3,
) -> Tensor:
    """Evaluate B-spline basis functions via Cox-de Boor recursion.

    Builds an open knot vector (degree+1 repeated boundary knots on each side)
    from the supplied interior+boundary knots, then evaluates each B-spline
    basis function at the query points.

    Parameters
    ----------
    t : Tensor
        Query time values.
    knots : Tensor
        Knot positions including boundaries (use :func:`place_knots`).
    degree : int
        Polynomial degree (default 3 = cubic).

    Returns
    -------
    Phi : Tensor
        ``(T, n_basis)`` matrix where ``n_basis = len(knots) + degree - 1``.
    """
    t = t.to(STAT_DTYPE).reshape(-1)
    knots = knots.to(STAT_DTYPE).to(t.device)

    # Build extended knot vector with degree+1 repetitions at the boundaries
    left_pad = knots[0].repeat(degree)
    right_pad = knots[-1].repeat(degree)
    ext = torch.cat([left_pad, knots, right_pad])  # length len(knots) + 2*degree

    n_basis = len(ext) - degree - 1  # = len(knots) + degree - 1
    T = t.shape[0]

    # Clamp t to [knots[0], knots[-1]] for evaluation safety
    t_eval = torch.clamp(t, min=float(knots[0].item()), max=float(knots[-1].item()))

    # Initial degree-0 (piecewise constant) basis functions
    # B_{i,0}(t) = 1 if ext[i] <= t < ext[i+1] else 0, with right-closure for
    # the rightmost non-degenerate interval so partition-of-unity holds at t_max.
    rightmost_knot = ext[-1]
    # Identify the last non-degenerate interval (one whose right endpoint equals
    # the rightmost knot value but whose left endpoint is strictly less).
    last_real_interval = None
    for i in range(len(ext) - 2, -1, -1):
        if ext[i + 1] == rightmost_knot and ext[i] < rightmost_knot:
            last_real_interval = i
            break

    B = torch.zeros(T, len(ext) - 1, dtype=STAT_DTYPE, device=t.device)
    for i in range(len(ext) - 1):
        left = ext[i]
        right = ext[i + 1]
        if i == last_real_interval:
            mask = (t_eval >= left) & (t_eval <= right)
        else:
            mask = (t_eval >= left) & (t_eval < right)
        B[:, i] = mask.to(STAT_DTYPE)

    # Cox-de Boor recursion
    for k in range(1, degree + 1):
        n_k = len(ext) - k - 1
        B_new = torch.zeros(T, n_k, dtype=STAT_DTYPE, device=t.device)
        for i in range(n_k):
            denom1 = ext[i + k] - ext[i]
            denom2 = ext[i + k + 1] - ext[i + 1]
            term1 = torch.zeros(T, dtype=STAT_DTYPE, device=t.device)
            term2 = torch.zeros(T, dtype=STAT_DTYPE, device=t.device)
            if denom1 > 0:
                term1 = (t_eval - ext[i]) / denom1 * B[:, i]
            if denom2 > 0:
                term2 = (ext[i + k + 1] - t_eval) / denom2 * B[:, i + 1]
            B_new[:, i] = term1 + term2
        B = B_new

    return B[:, :n_basis]


# ── Difference penalty for P-splines ────────────────────────────────


def difference_penalty(b: int, order: int = 2) -> Tensor:
    """Build the P-spline difference penalty matrix D'D.

    The order-th difference operator D applied to b coefficients yields a
    ``(b - order, b)`` matrix, and the penalty is ``D' D``.

    Parameters
    ----------
    b : int
        Number of basis functions.
    order : int
        Difference order (1 = first difference, 2 = second difference).

    Returns
    -------
    P : Tensor
        ``(b, b)`` symmetric positive semidefinite penalty matrix.
    """
    if b <= order:
        raise ValueError(f"difference_penalty: b={b} must exceed order={order}.")
    D = torch.eye(b, dtype=STAT_DTYPE)
    for _ in range(order):
        D = D[1:, :] - D[:-1, :]
    return D.T @ D


# ── 2D tensor-product P-spline ──────────────────────────────────────


def pspline_2d(
    row_coords: Tensor,
    col_coords: Tensor,
    n_knots_row: int = 8,
    n_knots_col: int = 8,
    degree: int = 3,
    penalty_order: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a 2D tensor-product P-spline design matrix and penalty matrices.

    For field-trial spatial correction the surface is modeled as
    ``f(row, col) = sum_{i,j} c_{ij} B_i(row) B_j(col)``, where the marginal
    bases ``B_i(row)`` and ``B_j(col)`` are cubic B-splines. The tensor-product
    design matrix has rows indexed by observations and columns indexed by the
    pair ``(i, j)`` flattened.

    Parameters
    ----------
    row_coords, col_coords : Tensor
        Per-observation row and column positions (same length).
    n_knots_row, n_knots_col : int
        Number of interior knots in each margin.
    degree : int
        B-spline degree.
    penalty_order : int
        Difference order for the marginal P-spline penalties.

    Returns
    -------
    Phi : Tensor
        ``(n, b_row * b_col)`` tensor-product design matrix.
    P_row : Tensor
        ``(b_row * b_col, b_row * b_col)`` penalty matrix for the row margin.
    P_col : Tensor
        ``(b_row * b_col, b_row * b_col)`` penalty matrix for the column
        margin.
    """
    row_coords = row_coords.to(STAT_DTYPE).reshape(-1)
    col_coords = col_coords.to(STAT_DTYPE).reshape(-1)
    if row_coords.shape != col_coords.shape:
        raise ValueError("pspline_2d: row_coords and col_coords must match in length.")

    knots_row = place_knots(row_coords, n_knots_row, kind="quantile", degree=degree)
    knots_col = place_knots(col_coords, n_knots_col, kind="quantile", degree=degree)

    Phi_row = bspline_basis(row_coords, knots_row, degree=degree)  # (n, b_row)
    Phi_col = bspline_basis(col_coords, knots_col, degree=degree)  # (n, b_col)

    n = row_coords.shape[0]
    b_row = Phi_row.shape[1]
    b_col = Phi_col.shape[1]

    # Row-wise tensor product: Phi[obs, i*b_col + j] = Phi_row[obs, i] * Phi_col[obs, j]
    Phi = (Phi_row.unsqueeze(2) * Phi_col.unsqueeze(1)).reshape(n, b_row * b_col)

    # Marginal penalties promoted to the joint coefficient space
    P_row_marginal = difference_penalty(b_row, order=penalty_order)  # (b_row, b_row)
    P_col_marginal = difference_penalty(b_col, order=penalty_order)  # (b_col, b_col)
    I_col = torch.eye(b_col, dtype=STAT_DTYPE)
    I_row = torch.eye(b_row, dtype=STAT_DTYPE)

    P_row = torch.kron(P_row_marginal, I_col)
    P_col = torch.kron(I_row, P_col_marginal)

    return Phi, P_row, P_col


# ── Re-evaluation at new time points ────────────────────────────────


def evaluate_basis_at(
    t_query: Tensor,
    basis_kind: str,
    basis_params: dict,
) -> Tensor:
    """Re-evaluate any basis at new time points using stored parameters.

    This is the bridge between fit-time bases (training time points) and
    scan-time per-time-point effect reconstruction.

    Parameters
    ----------
    t_query : Tensor
        Raw time values to evaluate at (same units as fit-time).
    basis_kind : {"legendre", "bspline"}
    basis_params : dict
        Parameters captured at fit time. For Legendre: ``{"order", "t_min",
        "t_max"}``. For B-spline: ``{"knots", "degree"}``.

    Returns
    -------
    Phi : Tensor
        ``(T_query, b)`` basis design matrix.
    """
    t_query = t_query.to(STAT_DTYPE)

    if basis_kind == "legendre":
        order = int(basis_params["order"])
        t_min = float(basis_params["t_min"])
        t_max = float(basis_params["t_max"])
        t_std, _, _ = standardize_time(t_query, t_min=t_min, t_max=t_max)
        return legendre_basis(t_std, order)
    elif basis_kind == "bspline":
        knots = basis_params["knots"]
        if not isinstance(knots, Tensor):
            knots = torch.as_tensor(knots, dtype=STAT_DTYPE)
        degree = int(basis_params.get("degree", 3))
        return bspline_basis(t_query, knots, degree=degree)
    else:
        raise ValueError(f"evaluate_basis_at: unknown basis_kind '{basis_kind}'.")
