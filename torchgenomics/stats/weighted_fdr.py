"""Weighted BH, hierarchical FDR, and local FDR (Phase 14).

Implements three advanced FDR-controlling procedures:
- weighted_bh: Weighted Benjamini-Hochberg (wBHa, Ignatiadis et al. 2023)
- hierarchical_fdr: Two-stage gene-then-SNP FDR (Heller et al. 2009)
- local_fdr: Empirical Bayes local FDR (Efron 2004)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import torch
from torch import Tensor

from .multipletesting import _cummin_reverse, benjamini_hochberg

logger = logging.getLogger(__name__)


def weighted_bh(
    p: Tensor,
    weights: Tensor,
    q: float = 0.05,
) -> Tensor:
    """Weighted Benjamini-Hochberg (wBHa) adjusted p-values.

    Weights encode prior information (MAF, functional annotation, LD score)
    to prioritize biologically plausible SNPs. Higher weight = more power
    (lower effective threshold) for that test.

    Parameters
    ----------
    p : Tensor, shape (m,)
        Raw p-values.
    weights : Tensor, shape (m,)
        Non-negative weights. Auto-normalized so mean == 1.
    q : float
        Nominal FDR level (used for documentation; the returned adjusted
        p-values can be compared to any threshold).

    Returns
    -------
    Tensor, shape (m,)
        Adjusted p-values, clamped to [0, 1].
    """
    m = p.shape[0]
    if m == 0:
        return p.clone()

    weights = weights.to(dtype=p.dtype, device=p.device)

    # Normalize weights to mean == 1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean
    else:
        return torch.ones_like(p)

    # Compute weighted p-values: p_i / w_i
    # Zero-weight SNPs get p/w = inf (never rejected)
    pw = torch.where(
        weights > 0,
        p / weights,
        torch.tensor(float("inf"), dtype=p.dtype, device=p.device),
    )

    # Sort by weighted p-values
    sorted_pw, sort_idx = pw.sort()

    # BH formula on weighted p-values: adj = pw_sorted * m / rank
    ranks = torch.arange(1, m + 1, dtype=p.dtype, device=p.device)
    adjusted = sorted_pw * m / ranks

    # Monotonicity enforcement (cumulative min from right)
    adjusted = _cummin_reverse(adjusted)
    adjusted = torch.clamp(adjusted, min=0.0, max=1.0)

    # Unsort back to original order
    result = torch.empty_like(p)
    result[sort_idx] = adjusted
    return result


def hierarchical_fdr(
    p: Tensor,
    group_ids: list[str],
    q: float = 0.05,
    group_method: str = "simes",
) -> Tensor:
    """Two-stage hierarchical FDR controlling procedure.

    Stage 1: Compute gene/group-level p-values.
    Stage 2: Apply BH to group-level p-values at level q.
    Stage 3: Within significant groups, apply BH to SNP-level p-values.

    This is more powerful than flat BH for gene-centric analyses because
    it concentrates the testing burden on significant genes.

    Parameters
    ----------
    p : Tensor, shape (m,)
        Raw p-values.
    group_ids : list[str], length m
        Group assignment (e.g., gene name) for each test.
    q : float
        Nominal FDR level applied at both stages.
    group_method : str
        How to compute group-level p-values:
        - "simes": Simes' method — min(p_sorted * n_g / rank) within group.
        - "fisher": Fisher's method — -2 * sum(log(p)) ~ chi2(2*n_g).

    Returns
    -------
    Tensor, shape (m,)
        Per-SNP adjusted p-values, clamped to [0, 1].
        SNPs in non-significant groups get p_adj = 1.0.
    """
    m = p.shape[0]
    if m == 0:
        return p.clone()

    # Build group → indices mapping
    groups: dict[str, list[int]] = defaultdict(list)
    for i, gid in enumerate(group_ids):
        groups[gid].append(i)

    group_names = list(groups.keys())
    n_groups = len(group_names)

    # Stage 1: Compute group-level p-values
    group_p = torch.ones(n_groups, dtype=p.dtype, device=p.device)
    for g_idx, gname in enumerate(group_names):
        indices = groups[gname]
        p_group = p[indices]
        group_p[g_idx] = _group_pvalue(p_group, group_method)

    # Stage 2: Apply BH to group-level p-values
    group_p_adj = benjamini_hochberg(group_p)

    # Identify significant groups
    sig_groups = set()
    for g_idx, gname in enumerate(group_names):
        if group_p_adj[g_idx].item() < q:
            sig_groups.add(gname)

    logger.debug(
        "Hierarchical FDR: %d/%d groups significant at q=%.3f",
        len(sig_groups), n_groups, q,
    )

    # Stage 3: Per-SNP adjusted p-values
    result = torch.ones(m, dtype=p.dtype, device=p.device)

    for g_idx, gname in enumerate(group_names):
        indices = groups[gname]
        g_adj = group_p_adj[g_idx].item()

        if gname not in sig_groups:
            # Non-significant group: all SNPs get 1.0
            for i in indices:
                result[i] = 1.0
            continue

        # Within-group BH
        p_within = p[indices]
        if len(indices) == 1:
            # Singleton group: SNP-level adj is just the group adj
            result[indices[0]] = max(g_adj, p_within[0].item())
        else:
            within_adj = benjamini_hochberg(p_within)
            for local_idx, global_idx in enumerate(indices):
                # Two-stage guarantee: max of group-level and within-group
                result[global_idx] = max(g_adj, within_adj[local_idx].item())

    return torch.clamp(result, min=0.0, max=1.0)


def local_fdr(
    p: Tensor,
    pi0_lambda: float = 0.5,
    bw_method: str | float | None = None,
) -> Tensor:
    """Empirical Bayes local FDR (Efron 2004).

    Estimates the posterior probability that each test is a true null,
    computed from the p-value distribution using kernel density estimation.

    Unlike adjusted p-values, local FDR values are posterior probabilities:
    lower lfdr = more likely to be a true discovery.

    Parameters
    ----------
    p : Tensor, shape (m,)
        Raw p-values.
    pi0_lambda : float
        Threshold for Storey's pi0 estimation. P-values above this
        threshold are assumed to be from the null distribution.
    bw_method : str, float, or None
        Bandwidth for scipy KDE. None = Scott's rule (default).

    Returns
    -------
    Tensor, shape (m,)
        Local false discovery rate for each test, clamped to [0, 1].
    """
    m = p.shape[0]
    if m == 0:
        return p.clone()
    if m < 10:
        # Too few tests for reliable density estimation
        return torch.ones_like(p)

    # Clamp p-values
    p_clamped = torch.clamp(p, min=1e-300, max=1.0 - 1e-15)

    # Estimate pi0 (proportion of true nulls) via Storey's method
    n_above = (p > pi0_lambda).sum().item()
    pi0 = n_above / (m * (1.0 - pi0_lambda))
    pi0 = min(max(pi0, 0.01), 1.0)  # Clamp to [0.01, 1.0]

    # Convert p-values to z-scores: z = Phi^{-1}(1 - p)
    # Using erfinv: Phi^{-1}(x) = sqrt(2) * erfinv(2*x - 1)
    # So Phi^{-1}(1 - p) = sqrt(2) * erfinv(1 - 2*p)
    z = math.sqrt(2.0) * torch.erfinv(1.0 - 2.0 * p_clamped)

    # Clamp z to avoid extreme values that break KDE
    z = torch.clamp(z, min=-10.0, max=10.0)

    # f0(z): standard normal density (theoretical null)
    f0 = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)

    # f(z): empirical density via KDE (scipy bridge, CPU round-trip)
    from scipy.stats import gaussian_kde

    z_np = z.detach().cpu().numpy()
    try:
        kde = gaussian_kde(z_np, bw_method=bw_method)
        f_hat_np = kde(z_np)
    except Exception:
        # KDE failure (e.g., singular data) → return conservative lfdr = 1
        return torch.ones_like(p)

    f_hat = torch.tensor(f_hat_np, dtype=p.dtype, device=p.device)

    # Clamp f_hat to avoid division by zero
    f_hat = torch.clamp(f_hat, min=1e-20)

    # Local FDR: lfdr = pi0 * f0 / f_hat
    lfdr = pi0 * f0 / f_hat

    return torch.clamp(lfdr, min=0.0, max=1.0)


# --- Internal helpers ---


def _group_pvalue(p_group: Tensor, method: str) -> float:
    """Compute a group-level p-value from individual SNP p-values.

    Parameters
    ----------
    p_group : Tensor, shape (n_g,)
        P-values for SNPs in one group.
    method : str
        "simes" or "fisher".

    Returns
    -------
    float
        Group-level p-value.
    """
    n_g = p_group.shape[0]
    if n_g == 0:
        return 1.0

    if method == "simes":
        sorted_p, _ = p_group.sort()
        ranks = torch.arange(1, n_g + 1, dtype=p_group.dtype, device=p_group.device)
        simes = (sorted_p * n_g / ranks).min().item()
        return min(simes, 1.0)
    elif method == "fisher":
        # Delegate to the public Fisher's combined test
        # (torchgenomics.postgwas.fisher_combined) so the kernel lives in
        # exactly one place. Behaviour identical to the previous
        # in-line implementation: −2 Σ log(p) ~ χ²(2 n_g).
        from ..postgwas._combine import fisher_combined
        _, p_combined = fisher_combined(p_group)
        return float(p_combined)
    else:
        raise ValueError(f"Unknown group_method: {method!r}. Use 'simes' or 'fisher'.")
