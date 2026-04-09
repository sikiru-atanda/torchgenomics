"""Stochastic trace and log-determinant estimation.

Provides Hutchinson's trace estimator and stochastic Lanczos quadrature (SLQ)
for log-determinant computation. These enable REML likelihood evaluation
without eigendecomposing the full covariance matrix.

References
----------
Hutchinson (1990). "A stochastic estimator of the trace of the influence matrix."
Communications in Statistics - Simulation and Computation.

Ubaru, Chen, Saad (2017). "Fast Estimation of tr(f(A)) via Stochastic Lanczos
Quadrature." SIAM J. Matrix Anal. Appl.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def hutchinson_trace(
    A_matvec: Callable[[Tensor], Tensor],
    n: int,
    n_probes: int = 30,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Estimate tr(A) using Hutchinson's stochastic estimator.

    tr(A) approx (1/n_probes) * sum_{i=1}^{n_probes} z_i^T A z_i

    where z_i are Rademacher random vectors (+1/-1 with equal probability).
    The estimator is unbiased with variance O(||A||_F^2 / n_probes).

    Parameters
    ----------
    A_matvec : Callable
        Function computing A @ x for a vector x.
    n : int
        Matrix dimension.
    n_probes : int
        Number of random probe vectors (more = lower variance).
    seed : int, optional
        Random seed for reproducibility.
    device : torch.device, optional
        Device for probe vectors.
    dtype : torch.dtype
        Computation dtype.

    Returns
    -------
    float
        Estimated trace of A.
    """
    if device is None:
        device = torch.device("cpu")

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    trace_sum = 0.0
    for _ in range(n_probes):
        # Rademacher random vector: +1/-1 with equal probability
        z = torch.where(
            torch.rand(n, device=device, dtype=dtype, generator=gen) < 0.5,
            torch.tensor(-1.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )
        Az = A_matvec(z)
        trace_sum += (z * Az).sum().item()

    return trace_sum / n_probes


def _lanczos_tridiag(
    A_matvec: Callable[[Tensor], Tensor],
    v0: Tensor,
    k: int,
) -> Tensor:
    """Run k steps of the Lanczos algorithm to produce a tridiagonal matrix T.

    Parameters
    ----------
    A_matvec : Callable
        Matrix-vector product function.
    v0 : Tensor, shape (n,)
        Starting vector (should be normalized).
    k : int
        Number of Lanczos iterations.

    Returns
    -------
    Tensor, shape (k, k)
        Symmetric tridiagonal matrix T.
    """
    n = v0.shape[0]
    dtype = v0.dtype
    device = v0.device

    # Lanczos vectors and tridiagonal entries
    alpha = torch.zeros(k, dtype=dtype, device=device)  # diagonal
    beta = torch.zeros(k, dtype=dtype, device=device)  # sub-diagonal

    v_prev = torch.zeros(n, dtype=dtype, device=device)
    v_curr = v0.clone()

    for j in range(k):
        w = A_matvec(v_curr)
        alpha[j] = (v_curr * w).sum()

        w = w - alpha[j] * v_curr
        if j > 0:
            w = w - beta[j] * v_prev

        # Reorthogonalization (partial — sufficient for our purposes)
        beta_next = w.norm()
        if beta_next < 1e-12:
            # Invariant subspace found; break early
            alpha = alpha[: j + 1]
            beta = beta[: j + 1]
            break

        if j + 1 < k:
            beta[j + 1] = beta_next

        v_prev = v_curr
        v_curr = w / beta_next

    # Build tridiagonal matrix
    m = len(alpha)
    T = torch.diag(alpha)
    if m > 1:
        T += torch.diag(beta[1:m], diagonal=1)
        T += torch.diag(beta[1:m], diagonal=-1)

    return T


def stochastic_logdet(
    A_matvec: Callable[[Tensor], Tensor],
    n: int,
    n_probes: int = 30,
    lanczos_iters: int = 50,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Estimate log|det(A)| via stochastic Lanczos quadrature (SLQ).

    For each probe vector z:
    1. Run Lanczos iteration: A → tridiagonal T of size (k, k)
    2. Eigendecompose T: T = Q diag(theta) Q^T
    3. Approximate z^T log(A) z ≈ n * sum(q_{1,i}^2 * log(theta_i))
       where q_{1,i} is the first row of Q
    4. Average over probes: log|det(A)| ≈ (1/n_probes) * sum estimates

    The estimator uses the identity:
        log|det(A)| = tr(log(A))
    and Hutchinson's estimator for the trace.

    Parameters
    ----------
    A_matvec : Callable
        Function computing A @ x for vector x. A must be SPD.
    n : int
        Matrix dimension.
    n_probes : int
        Number of random probe vectors.
    lanczos_iters : int
        Number of Lanczos iterations per probe.
    seed : int, optional
        Random seed.
    device : torch.device, optional
        Computation device.
    dtype : torch.dtype
        Computation dtype.

    Returns
    -------
    float
        Estimated log-determinant of A.
    """
    if device is None:
        device = torch.device("cpu")

    lanczos_k = min(lanczos_iters, n)

    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    logdet_sum = 0.0

    for _ in range(n_probes):
        # Rademacher probe vector, normalized
        z = torch.where(
            torch.rand(n, device=device, dtype=dtype, generator=gen) < 0.5,
            torch.tensor(-1.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )
        z_norm = z.norm()
        v0 = z / z_norm

        # Lanczos → tridiagonal T
        T = _lanczos_tridiag(A_matvec, v0, lanczos_k)

        # Eigendecompose small tridiagonal matrix
        T = (T + T.T) / 2.0
        theta, Q = torch.linalg.eigh(T)

        # Clamp eigenvalues to avoid log of negative/zero
        theta = torch.clamp(theta, min=1e-30)

        # First row of Q (corresponds to our starting vector)
        q1 = Q[0, :]

        # Quadrature estimate: z^T log(A) z ≈ ||z||^2 * sum(q1_i^2 * log(theta_i))
        estimate = (z_norm ** 2) * (q1 ** 2 * torch.log(theta)).sum().item()
        logdet_sum += estimate

    return logdet_sum / n_probes
