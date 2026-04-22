"""Saddlepoint approximation (SPA) for calibrated tail p-values.

Used by BinaryGLM and BinaryGLMM to correct score test p-values under
case-control imbalance (SAIGE, Zhou et al. 2018). Standard chi-square
p-values are miscalibrated in the tails when prevalence is extreme;
SPA uses the exact CGF of the score statistic for accurate tail probs.

For each SNP with |T| > threshold, computes exact p-value via
Lugannani-Rice formula on the CGF of Sum_i (Y_i - mu_i) * g_i.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _cgf(t: Tensor, mu: Tensor, g: Tensor) -> Tensor:
    """Cumulant generating function K(t) = sum_i log(q_i exp(-mu_i g_i t) + mu_i exp(q_i g_i t)).

    Parameters
    ----------
    t : scalar or (k,) — evaluation points
    mu : (n,) — fitted null probabilities
    g : (n,) — genotype dosages for one SNP

    Returns
    -------
    K(t) : same shape as t
    """
    q = 1.0 - mu
    # (n,) or (n, k) via broadcasting
    if t.dim() == 0:
        term1 = q * torch.exp(-mu * g * t)
        term2 = mu * torch.exp(q * g * t)
    else:
        # t is (k,), expand for broadcasting: mu (n,1), g (n,1), t (1,k)
        mu_ = mu.unsqueeze(1)
        q_ = q.unsqueeze(1)
        g_ = g.unsqueeze(1)
        t_ = t.unsqueeze(0)
        term1 = q_ * torch.exp(-mu_ * g_ * t_)
        term2 = mu_ * torch.exp(q_ * g_ * t_)

    log_sum = torch.log(term1 + term2)

    if t.dim() == 0:
        return log_sum.sum()
    else:
        return log_sum.sum(dim=0)  # (k,)


def _cgf_deriv1(t: float, mu: Tensor, g: Tensor) -> float:
    """First derivative K'(t) = sum_i g_i (mu_i exp(q_i g_i t) - q_i exp(-mu_i g_i t)) / denom."""
    t_t = torch.tensor(t, dtype=mu.dtype, device=mu.device)
    q = 1.0 - mu
    exp_pos = torch.exp(q * g * t_t)
    exp_neg = torch.exp(-mu * g * t_t)
    denom = q * exp_neg + mu * exp_pos
    denom = torch.clamp(denom, min=1e-30)
    numer = g * (mu * q * exp_pos - mu * q * exp_neg)
    # Simplify: K'(t) = sum_i g_i * mu_i * q_i * (exp(q_i g_i t) - exp(-mu_i g_i t)) / denom_i
    # Actually: d/dt of log(q exp(-mu g t) + mu exp(q g t))
    #   = (-mu g q exp(-mu g t) + q g mu exp(q g t)) / (q exp(-mu g t) + mu exp(q g t))
    #   = mu q g (exp(q g t) - exp(-mu g t)) / denom
    numer = mu * q * g * (exp_pos - exp_neg)
    return (numer / denom).sum().item()


def _cgf_deriv2(t: float, mu: Tensor, g: Tensor) -> float:
    """Second derivative K''(t)."""
    t_t = torch.tensor(t, dtype=mu.dtype, device=mu.device)
    q = 1.0 - mu
    exp_pos = torch.exp(q * g * t_t)
    exp_neg = torch.exp(-mu * g * t_t)
    denom = q * exp_neg + mu * exp_pos
    denom = torch.clamp(denom, min=1e-30)

    # K''(t) = sum_i [mu_i q_i g_i^2 (q_i exp(q g t) + mu_i exp(-mu g t)) / denom_i
    #           - (mu_i q_i g_i)^2 (exp(q g t) - exp(-mu g t))^2 / denom_i^2]
    g2 = g * g
    term1 = mu * q * g2 * (q * exp_pos + mu * exp_neg) / denom
    term2 = (mu * q * g) ** 2 * (exp_pos - exp_neg) ** 2 / (denom ** 2)

    return (term1 - term2).sum().item()


def _solve_saddlepoint(observed: float, mu: Tensor, g: Tensor,
                        max_iter: int = 50, tol: float = 1e-8) -> float:
    """Find saddlepoint t_hat such that K'(t_hat) = observed via Newton's method."""
    t = 0.0
    for _ in range(max_iter):
        k1 = _cgf_deriv1(t, mu, g)
        k2 = _cgf_deriv2(t, mu, g)
        if abs(k2) < 1e-30:
            break
        t_new = t + (observed - k1) / k2
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new
    return t


def saddlepoint_pvalue(
    observed_score: Tensor,
    mu: Tensor,
    G_chunk: Tensor,
    threshold: float = 2.0,
    variance: Tensor | None = None,
) -> Tensor:
    """Compute SPA-corrected p-values for binary score statistics.

    For SNPs where the chi-square statistic exceeds `threshold`, computes
    the exact p-value via saddlepoint approximation. For others, uses
    the standard chi-square approximation.

    Parameters
    ----------
    observed_score : (m,) — raw score statistics U = g'(Y - mu)
    mu : (n,) — fitted null probabilities
    G_chunk : (n, m) — genotype matrix
    threshold : float — chi2 threshold for SPA activation (default 2.0)
    variance : (m,) optional — pre-computed score variance (e.g. Schur
        complement from covariate adjustment). If None, uses simple g'Wg.

    Returns
    -------
    p : (m,) — p-values (SPA-corrected where applicable)
    """
    from .tests import chi2_sf

    m = observed_score.shape[0]
    n = mu.shape[0]

    # Compute variance of score under null for chi2 reference
    if variance is not None:
        V = variance
    else:
        W = mu * (1.0 - mu)  # (n,)
        WG = W.unsqueeze(1) * G_chunk  # (n, m)
        V = (G_chunk * WG).sum(dim=0)  # (m,) — simple variance

    V_safe = torch.clamp(V, min=1e-20)
    chi2_stat = observed_score ** 2 / V_safe

    # Standard chi2 p-values for all
    p = chi2_sf(chi2_stat, df=1)

    # SPA correction for extreme statistics
    spa_mask = chi2_stat > threshold
    spa_indices = spa_mask.nonzero(as_tuple=True)[0]

    # Native fast-path: push the entire per-extreme-SNP Newton + Lugannani-Rice
    # body down to C++. Eliminates ~200 .item() round-trips per call from the
    # Python implementation below. The pure-Python path remains intact as the
    # algorithmic spec and runs whenever the extension is missing, the data is
    # not on CPU, or TORCHGWAS_DISABLE_NATIVE=1 is set.
    from .._dispatch import native_disabled
    from .._native import HAS_NATIVE_SPA, _spa_native

    use_native_spa = (
        HAS_NATIVE_SPA and not native_disabled()
        and mu.device.type == "cpu" and mu.dtype == torch.float64
        and G_chunk.device.type == "cpu" and G_chunk.dtype == torch.float64
    )

    if use_native_spa:
        mu_np = mu.detach().contiguous().numpy()
        # Filter out monomorphic SNPs before sending to the batched kernel so
        # the C++ side never sees a zero-variance column (its Newton solve
        # would still succeed but would waste work).
        keep_j: list[int] = []
        for idx in spa_indices:
            j = int(idx.item())
            if float(G_chunk[:, j].std().item()) < 1e-10:
                continue
            keep_j.append(j)

        if keep_j:
            j_tensor = torch.tensor(keep_j, dtype=torch.long, device=G_chunk.device)
            # (n, k) column-major view; .numpy() on a contiguous transpose
            # gives a Fortran-order buffer for the C++ batched entry.
            G_ext_np = (
                G_chunk.index_select(1, j_tensor)
                .detach()
                .cpu()
                .numpy()
                .copy(order="F")
            )
            obs_np = observed_score.index_select(0, j_tensor).detach().cpu().numpy()
            p_batch = _spa_native.spa_lugannani_rice_batch(mu_np, G_ext_np, obs_np)
            for kk, j in enumerate(keep_j):
                p_val = float(p_batch[kk])
                if math.isnan(p_val):
                    continue
                p[j] = min(p_val, p[j].item())
        return p

    for idx in spa_indices:
        j = idx.item()
        obs = observed_score[j].item()
        g_j = G_chunk[:, j]

        # Skip if genotype is monomorphic
        if g_j.std() < 1e-10:
            continue

        # Find saddlepoint
        t_hat = _solve_saddlepoint(obs, mu, g_j)

        # Compute K(t_hat) and K''(t_hat)
        t_tensor = torch.tensor(t_hat, dtype=mu.dtype, device=mu.device)
        K_val = _cgf(t_tensor, mu, g_j).item()
        K2_val = _cgf_deriv2(t_hat, mu, g_j)

        if K2_val <= 0 or not math.isfinite(K_val):
            continue  # fallback to chi2

        # Lugannani-Rice formula
        w = math.copysign(1.0, t_hat) * math.sqrt(2.0 * (t_hat * obs - K_val))
        u = t_hat * math.sqrt(K2_val)

        if abs(w) < 1e-10 or abs(u) < 1e-10:
            continue

        from scipy.stats import norm
        # Two-sided p-value
        p_spa_one_tail = norm.sf(abs(w)) + norm.pdf(abs(w)) * (1.0 / abs(w) - 1.0 / abs(u))

        # Also compute for negative tail
        t_hat_neg = _solve_saddlepoint(-obs, mu, g_j)
        K_val_neg = _cgf(torch.tensor(t_hat_neg, dtype=mu.dtype, device=mu.device),
                          mu, g_j).item()
        K2_val_neg = _cgf_deriv2(t_hat_neg, mu, g_j)

        if K2_val_neg > 0 and math.isfinite(K_val_neg):
            w_neg = math.copysign(1.0, t_hat_neg) * math.sqrt(
                2.0 * (t_hat_neg * (-obs) - K_val_neg))
            u_neg = t_hat_neg * math.sqrt(K2_val_neg)

            if abs(w_neg) > 1e-10 and abs(u_neg) > 1e-10:
                p_spa_other = norm.sf(abs(w_neg)) + norm.pdf(abs(w_neg)) * (
                    1.0 / abs(w_neg) - 1.0 / abs(u_neg))
            else:
                p_spa_other = p_spa_one_tail
        else:
            p_spa_other = p_spa_one_tail

        p_spa = abs(p_spa_one_tail) + abs(p_spa_other)
        p_spa = max(min(p_spa, 1.0), 1e-300)
        # SAIGE hybrid: SPA should never be more conservative than chi2.
        # Under balanced prevalence the CGF-based SPA can overcorrect because
        # it uses raw genotype variance rather than the Schur-complement
        # (covariate-adjusted) variance.  Take the minimum of SPA and chi2.
        p[j] = min(p_spa, p[j].item())

    return p
