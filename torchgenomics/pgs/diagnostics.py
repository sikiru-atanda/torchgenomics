"""MCMC convergence diagnostics for PGS methods.

Provides:
    - rhat(chains): Gelman-Rubin potential scale reduction factor
    - ess(chains): effective sample size via autocorrelation
    - check_convergence(result, rhat_threshold): boolean guard
"""

from __future__ import annotations

import os
from .._dispatch import native_disabled

import torch
from torch import Tensor

from .._native import HAS_NATIVE_ESS, _ess_native
from .base import PGSResult


def _native_enabled() -> bool:
    """Whether the native C++ Geyer-ESS accelerator should be used.

    Disabled when the extension was not built or when the user sets
    ``TORCHGENOMICS_DISABLE_NATIVE=1`` in the environment.
    """
    return HAS_NATIVE_ESS and not native_disabled()


def rhat(chains: Tensor) -> Tensor:
    """Gelman-Rubin R-hat (potential scale reduction factor).

    Parameters
    ----------
    chains : Tensor of shape ``(n_chains, n_iter, m)`` or ``(n_chains, n_iter)``
        Posterior samples. Per-parameter R-hat is computed over the
        ``(n_chains, n_iter)`` axes; an extra trailing parameter dim is
        preserved.

    Returns
    -------
    Tensor
        R-hat per parameter; shape ``(m,)`` for 3D input or scalar for 2D.
        ``R_hat = sqrt(((N - 1) / N) * W + B / N) / sqrt(W)`` where
        ``W`` is within-chain variance and ``B`` is between-chain variance
        scaled by ``N`` (the standard Gelman-Rubin definition).
    """
    if chains.dim() == 2:
        chains = chains.unsqueeze(-1)
        squeeze_out = True
    else:
        squeeze_out = False
    if chains.dim() != 3:
        raise ValueError(
            f"rhat expects (n_chains, n_iter[, m]); got shape {tuple(chains.shape)}."
        )
    M, N, _ = chains.shape
    if M < 2 or N < 2:
        raise ValueError(f"rhat requires >=2 chains and >=2 iterations; got M={M}, N={N}.")

    chain_means = chains.mean(dim=1)  # (M, m)
    overall_mean = chain_means.mean(dim=0)  # (m,)
    # Between-chain variance
    B = N * ((chain_means - overall_mean.unsqueeze(0)) ** 2).sum(dim=0) / (M - 1)
    # Within-chain variance
    chain_vars = chains.var(dim=1, unbiased=True)  # (M, m)
    W = chain_vars.mean(dim=0)  # (m,)
    var_hat = (N - 1) / N * W + B / N
    W_safe = W.clamp(min=1e-30)
    r = torch.sqrt(var_hat / W_safe)
    if squeeze_out:
        r = r.squeeze(-1)
    return r


def _autocorr_fft(x: Tensor) -> Tensor:
    """One-chain autocorrelation via FFT along dim=0.

    ``x`` has shape ``(N, m)`` (or ``(N,)``). Returns autocorrelation of
    lag 0..N-1 for each parameter (same shape as input).
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    N, m = x.shape
    x_c = x - x.mean(dim=0, keepdim=True)
    # Zero-pad to next power of two of 2N-1 for linear (not circular) correlation
    size = 1
    target = 2 * N - 1
    while size < target:
        size *= 2
    Xf = torch.fft.rfft(x_c, n=size, dim=0)
    power = (Xf * Xf.conj()).real
    acov = torch.fft.irfft(power, n=size, dim=0)[:N]
    # Normalize so acov[0] == var
    acov = acov / N
    var = acov[0].clamp(min=1e-30)
    acorr = acov / var
    if squeeze:
        acorr = acorr.squeeze(-1)
    return acorr


def ess(chains: Tensor) -> Tensor:
    """Effective sample size via autocorrelation (Geyer initial positive).

    Parameters
    ----------
    chains : Tensor of shape ``(n_chains, n_iter, m)`` or ``(n_chains, n_iter)``
        Posterior samples.

    Returns
    -------
    Tensor
        Per-parameter ESS; shape ``(m,)`` for 3D, scalar-shape for 2D.
    """
    if chains.dim() == 2:
        chains = chains.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    if chains.dim() != 3:
        raise ValueError(
            f"ess expects (n_chains, n_iter[, m]); got shape {tuple(chains.shape)}."
        )
    M, N, m = chains.shape
    # Stack chains -> (M*N, m); compute per-chain autocorrelation then average
    acorrs = []
    for c in range(M):
        acorrs.append(_autocorr_fft(chains[c]))  # (N, m)
    acorr_mean = torch.stack(acorrs, dim=0).mean(dim=0)  # (N, m)

    # Geyer initial positive: sum pairs (acorr[2k] + acorr[2k+1]) until negative.
    # The native C++ shortcut runs the per-parameter inner loop without the
    # per-element ``.item()`` round-trips that dominate the Python path. The
    # pure-Python loop below remains the canonical algorithmic reference and
    # is exercised when the extension is missing or
    # ``TORCHGENOMICS_DISABLE_NATIVE=1`` is set.
    if _native_enabled() and acorr_mean.device.type == "cpu":
        acorr_np = acorr_mean.detach().to(torch.float64).cpu().numpy()
        out_np = _ess_native.geyer_initial_positive_ess(acorr_np, float(M * N))
        out = torch.from_numpy(out_np).to(dtype=chains.dtype, device=chains.device)
    else:
        out = torch.empty(m, dtype=chains.dtype, device=chains.device)
        for j in range(m):
            tau = 1.0
            k = 1
            while k + 1 < N:
                pair = float(acorr_mean[k, j].item() + acorr_mean[k + 1, j].item())
                if pair <= 0:
                    break
                tau += 2.0 * pair
                k += 2
            out[j] = M * N / max(tau, 1.0)
    if squeeze:
        out = out.squeeze(-1)
    return out


def check_convergence(
    result: PGSResult,
    rhat_threshold: float = 1.1,
    min_fraction: float = 0.95,
) -> bool:
    """Return True iff at least ``min_fraction`` of per-SNP R-hats are below threshold.

    When ``result.rhat`` is None the function returns ``result.converged``.
    """
    if result.rhat is None:
        return bool(result.converged)
    below = (result.rhat < rhat_threshold).float().mean().item()
    return bool(below >= min_fraction)
