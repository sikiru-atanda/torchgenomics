"""GPU-accelerated maxT permutation testing for FWER control."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def permutation_maxT(
    G: Tensor,
    Y: Tensor,
    X0: Tensor,
    n_perms: int = 1000,
    seed: Optional[int] = None,
) -> Tensor:
    """GPU-batched maxT permutation test for family-wise error rate control.

    Implements the Westfall-Young maxT procedure:
    1. For each permutation, shuffle Y (preserving covariate structure via residuals)
    2. Compute test statistic for all SNPs simultaneously (batched GEMM)
    3. Record max|T| across SNPs for each permutation
    4. Adjusted p-value for SNP j = fraction of permutations where max|T| >= |T_obs_j|

    GPU acceleration: all permutations are batched into a single tensor operation,
    avoiding per-permutation Python loops.

    Parameters
    ----------
    G : (n, m) genotype dosage matrix
    Y : (n,) or (n, 1) phenotype
    X0 : (n, c) covariate matrix (including intercept)
    n_perms : number of permutations
    seed : random seed for reproducibility

    Returns
    -------
    p_adj : (m,) adjusted p-values (FWER-controlling)
    """
    if seed is not None:
        torch.manual_seed(seed)

    T_obs_abs, G_resid, Y_resid, df, GtG_safe = _residualize_and_tstat(G, Y, X0)
    n = G.shape[0]
    m = G.shape[1]
    device = G.device
    YtY = (Y_resid ** 2).sum()

    # --- Batched permutation ---
    perm_indices = torch.stack([torch.randperm(n, device=device) for _ in range(n_perms)])
    Y_perm = Y_resid[perm_indices]  # (n_perms, n)

    # Batched G^T Y_perm: (m, n) @ (n, n_perms) = (m, n_perms)
    GtY_perm = G_resid.T @ Y_perm.T

    # Per-permutation RSS and T-stats
    rss_perm = YtY - GtY_perm ** 2 / GtG_safe.unsqueeze(1)
    rss_perm = torch.clamp(rss_perm, min=1e-20)
    se2_perm = rss_perm / df
    T_perm = GtY_perm / torch.sqrt(GtG_safe.unsqueeze(1) * se2_perm)
    T_perm_abs = T_perm.abs()

    # maxT across SNPs for each permutation
    maxT_per_perm = T_perm_abs.max(dim=0).values  # (n_perms,)

    # Adjusted p-values
    exceed_count = (maxT_per_perm.unsqueeze(0) >= T_obs_abs.unsqueeze(1)).sum(dim=1).to(STAT_DTYPE)
    p_adj = (1.0 + exceed_count) / (1.0 + n_perms)

    logger.info(
        "maxT permutation: %d perms, %d SNPs, device=%s, min_p=%.2e",
        n_perms, m, device, p_adj.min().item(),
    )

    return p_adj


def adaptive_permutation_maxT(
    G: Tensor,
    Y: Tensor,
    X0: Tensor,
    n_perms_min: int = 1000,
    n_perms_max: int = 10000,
    early_stop_threshold: float = 0.01,
    early_stop_exceedances: int = 50,
    seed: Optional[int] = None,
    batch_size: int = 500,
) -> Tensor:
    """Adaptive maxT permutation test with two-phase design.

    Phase 1: Run n_perms_min permutations for all SNPs.
    Phase 2: For SNPs with estimated p < early_stop_threshold,
             run additional permutations up to n_perms_max.
    Early stopping: if a SNP accumulates >= early_stop_exceedances
             exceedances during Phase 2, it is resolved and stops.

    Parameters
    ----------
    G : (n, m) genotype dosage matrix
    Y : (n,) or (n, 1) phenotype
    X0 : (n, c) covariate matrix (including intercept)
    n_perms_min : int
        Phase 1 permutation count (applied to all SNPs).
    n_perms_max : int
        Maximum total permutations for promising SNPs.
    early_stop_threshold : float
        P-value threshold to enter Phase 2. SNPs with Phase 1
        p >= this threshold are considered resolved.
    early_stop_exceedances : int
        Stop a SNP early if its exceedance count reaches this value
        during Phase 2 (clearly non-significant at fine resolution).
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int
        Number of permutations per sub-batch in Phase 2.

    Returns
    -------
    p_adj : (m,) adjusted p-values (FWER-controlling)
    """
    if seed is not None:
        torch.manual_seed(seed)

    T_obs_abs, G_resid, Y_resid, df, GtG_safe = _residualize_and_tstat(G, Y, X0)
    n = G.shape[0]
    m = G.shape[1]
    device = G.device
    YtY = (Y_resid ** 2).sum()

    # Track per-SNP exceedance counts and total permutations used
    exceed_count = torch.zeros(m, dtype=STAT_DTYPE, device=device)
    n_perms_used = torch.zeros(m, dtype=STAT_DTYPE, device=device)

    # --- Phase 1: Run n_perms_min for all SNPs ---
    maxT_history = _run_perm_batch(
        G_resid, Y_resid, GtG_safe, YtY, df, n, n_perms_min, device,
    )  # (n_perms_min,)

    # Count exceedances for all SNPs
    exceed_count += (maxT_history.unsqueeze(0) >= T_obs_abs.unsqueeze(1)).sum(dim=1).to(STAT_DTYPE)
    n_perms_used += n_perms_min

    # Preliminary p-values
    p_prelim = (1.0 + exceed_count) / (1.0 + n_perms_used)

    # Identify promising SNPs that need more permutations
    promising = p_prelim < early_stop_threshold
    # Also exclude SNPs already clearly non-significant
    resolved = exceed_count >= early_stop_exceedances
    needs_more = promising & ~resolved

    n_promising = int(needs_more.sum().item())
    logger.info(
        "Adaptive permutation Phase 1: %d perms, %d/%d SNPs need Phase 2",
        n_perms_min, n_promising, m,
    )

    if n_promising == 0:
        p_adj = (1.0 + exceed_count) / (1.0 + n_perms_used)
        return p_adj

    # --- Phase 2: Additional permutations in batches ---
    remaining_perms = n_perms_max - n_perms_min
    perms_done = 0

    while perms_done < remaining_perms and needs_more.any():
        this_batch = min(batch_size, remaining_perms - perms_done)

        maxT_batch = _run_perm_batch(
            G_resid, Y_resid, GtG_safe, YtY, df, n, this_batch, device,
        )

        # Update exceedance counts only for SNPs that still need more
        batch_exceeds = (maxT_batch.unsqueeze(0) >= T_obs_abs.unsqueeze(1)).sum(dim=1).to(STAT_DTYPE)

        exceed_count += torch.where(needs_more, batch_exceeds, torch.zeros_like(batch_exceeds))
        n_perms_used += torch.where(needs_more, torch.tensor(float(this_batch), device=device), torch.zeros(1, device=device))

        perms_done += this_batch

        # Update resolved status
        resolved = exceed_count >= early_stop_exceedances
        needs_more = promising & ~resolved

        # Also check if we've used enough perms
        enough_perms = n_perms_used >= n_perms_max
        needs_more = needs_more & ~enough_perms

    n_phase2 = perms_done
    logger.info(
        "Adaptive permutation Phase 2: %d additional perms, %d SNPs still promising",
        n_phase2, int(needs_more.sum().item()),
    )

    # Final p-values
    p_adj = (1.0 + exceed_count) / (1.0 + n_perms_used)

    logger.info(
        "Adaptive permutation: total min_p=%.2e, device=%s",
        p_adj.min().item(), device,
    )

    return p_adj


# --- Internal helpers ---


def _residualize_and_tstat(
    G: Tensor, Y: Tensor, X0: Tensor,
) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:
    """Residualize G and Y against covariates, compute observed T-statistics.

    Returns
    -------
    T_obs_abs : (m,) absolute observed test statistics
    G_resid : (n, m) residualized genotypes
    Y_resid : (n,) residualized phenotype
    df : degrees of freedom
    GtG_safe : (m,) clamped sum of squared residualized genotypes
    """
    G = G.to(STAT_DTYPE)
    Y = Y.to(STAT_DTYPE).squeeze()
    X0 = X0.to(STAT_DTYPE)
    n, m = G.shape
    device = G.device

    # Freedman-Lane residualization
    Q, _ = torch.linalg.qr(X0)
    P_X0 = Q @ Q.T
    M_X0 = torch.eye(n, dtype=STAT_DTYPE, device=device) - P_X0

    Y_resid = M_X0 @ Y
    G_resid = M_X0 @ G

    # Observed test statistics
    df = n - X0.shape[1] - 1
    GtY = (G_resid * Y_resid.unsqueeze(1)).sum(dim=0)
    GtG = (G_resid ** 2).sum(dim=0)
    YtY = (Y_resid ** 2).sum()

    GtG_safe = torch.clamp(GtG, min=1e-20)
    rss = YtY - GtY ** 2 / GtG_safe
    rss_safe = torch.clamp(rss, min=1e-20)
    se2 = rss_safe / df
    T_obs = GtY / torch.sqrt(GtG_safe * se2)
    T_obs_abs = T_obs.abs()

    return T_obs_abs, G_resid, Y_resid, df, GtG_safe


def _run_perm_batch(
    G_resid: Tensor,
    Y_resid: Tensor,
    GtG_safe: Tensor,
    YtY: Tensor,
    df: int,
    n: int,
    n_perms: int,
    device: torch.device,
) -> Tensor:
    """Run a batch of permutations and return maxT per permutation.

    Returns
    -------
    maxT_per_perm : (n_perms,) max |T| across all SNPs for each permutation
    """
    perm_indices = torch.stack([torch.randperm(n, device=device) for _ in range(n_perms)])
    Y_perm = Y_resid[perm_indices]  # (n_perms, n)

    GtY_perm = G_resid.T @ Y_perm.T  # (m, n_perms)

    rss_perm = YtY - GtY_perm ** 2 / GtG_safe.unsqueeze(1)
    rss_perm = torch.clamp(rss_perm, min=1e-20)
    se2_perm = rss_perm / df
    T_perm = GtY_perm / torch.sqrt(GtG_safe.unsqueeze(1) * se2_perm)
    T_perm_abs = T_perm.abs()

    return T_perm_abs.max(dim=0).values  # (n_perms,)
