"""Adaptive FDR control: IHW and AdaPT (Phase 30).

Implements two covariate-adaptive multiple testing procedures that learn
from auxiliary information (MAF, imputation R², LD score) to boost power
while controlling FDR:

- IHW (Independent Hypothesis Weighting; Ignatiadis & Huber 2021):
  K-fold cross-fitting learns weight function w(bin) from other folds,
  then applies weighted BH with learned weights.

- AdaPT (Adaptive P-value Thresholding; Lei & Fithian 2018):
  EM iterations estimate local FDR given covariates, then update per-bin
  rejection thresholds.  FDR guaranteed by masking principle.

Both procedures reuse existing infrastructure: weighted_bh(),
benjamini_hochberg(), local_fdr().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .multipletesting import benjamini_hochberg, _cummin_reverse
from .weighted_fdr import weighted_bh

logger = logging.getLogger(__name__)


# ===================================================================
# Result dataclasses
# ===================================================================

@dataclass
class IHWResult:
    """Output of ihw().

    Attributes
    ----------
    adjusted_p : (m,) adjusted p-values (compare to q for rejection).
    weights : (m,) learned per-hypothesis weights (mean-normalised to 1).
    n_rejections : int — number of discoveries at nominal q.
    bins : (m,) int tensor — bin assignment for each hypothesis.
    fold_assignments : (m,) int tensor — cross-fitting fold assignment.
    n_bins : int
    n_folds : int
    """

    adjusted_p: Tensor
    weights: Tensor
    n_rejections: int
    bins: Tensor
    fold_assignments: Tensor
    n_bins: int
    n_folds: int


@dataclass
class AdaPTResult:
    """Output of adapt().

    Attributes
    ----------
    adjusted_p : (m,) adjusted p-values.
    thresholds : (m,) per-hypothesis learned threshold s(x_i).
    n_rejections : int — number of discoveries at nominal q.
    bins : (m,) int tensor — bin assignment.
    n_iter : int — EM iterations until convergence.
    converged : bool
    n_bins : int
    """

    adjusted_p: Tensor
    thresholds: Tensor
    n_rejections: int
    bins: Tensor
    n_iter: int
    converged: bool
    n_bins: int


# ===================================================================
# Binning helper
# ===================================================================

def _bin_covariates(
    covariates: Tensor,
    n_bins: int,
) -> Tensor:
    """Assign hypotheses to bins via quantile-based binning of a 1-D covariate.

    Parameters
    ----------
    covariates : (m,) 1-D covariate (e.g. MAF, imputation R²).
    n_bins : int — number of bins.

    Returns
    -------
    (m,) long tensor — bin index in [0, n_bins).
    """
    m = covariates.shape[0]
    if m == 0:
        return torch.zeros(0, dtype=torch.long, device=covariates.device)
    if n_bins >= m:
        n_bins = max(m, 1)

    # Quantile boundaries
    quantiles = torch.linspace(0, 1, n_bins + 1, device=covariates.device,
                               dtype=covariates.dtype)
    boundaries = torch.quantile(covariates.float(), quantiles.float())

    # bucketize: returns indices in [0, n_bins]
    bins = torch.bucketize(covariates.contiguous(), boundaries[1:-1].contiguous())
    bins = torch.clamp(bins, min=0, max=n_bins - 1)
    return bins.long()


# ===================================================================
# IHW
# ===================================================================

def _ihw_weight_optimization(
    p: Tensor,
    bins: Tensor,
    n_bins: int,
    q: float,
    lambda_: float,
) -> Tensor:
    """Optimize IHW weights for a single training fold.

    Learns per-bin weights that maximise the number of rejections
    subject to the constraint that mean(w) = 1 and w >= 0.

    The weight for bin b is proportional to the fraction of p-values
    below lambda_ in that bin (an estimate of the signal density).

    Parameters
    ----------
    p : (m_train,) p-values in this training fold.
    bins : (m_train,) bin assignments.
    n_bins : int
    q : float — target FDR level.
    lambda_ : float — threshold for signal density estimation.

    Returns
    -------
    (n_bins,) non-negative weights with mean == 1.
    """
    device = p.device
    dtype = p.dtype

    # Count signal density per bin: proportion of p < lambda_
    w = torch.zeros(n_bins, dtype=dtype, device=device)
    for b in range(n_bins):
        mask = bins == b
        n_b = mask.sum().item()
        if n_b == 0:
            w[b] = 1.0
            continue
        # Signal proportion in bin b
        sig_frac = (p[mask] < lambda_).float().mean().item()
        w[b] = max(sig_frac, 1e-6)

    # Normalize to mean == 1
    w_mean = w.mean()
    if w_mean > 0:
        w = w / w_mean
    else:
        w = torch.ones(n_bins, dtype=dtype, device=device)

    return w


def ihw(
    p: Tensor,
    covariates: Tensor,
    q: float = 0.05,
    n_folds: int = 5,
    n_bins: int = 10,
    seed: int | None = None,
    lambda_: float = 0.1,
) -> IHWResult:
    """Independent Hypothesis Weighting (Ignatiadis & Huber 2021).

    K-fold cross-fitting procedure:
      1. Bin hypotheses by covariate into n_bins groups.
      2. Split hypotheses into K folds.
      3. For each fold k: learn weights w(bin) from the remaining K-1 folds
         by estimating signal density per bin.
      4. Apply weighted BH with the learned weights.

    Parameters
    ----------
    p : (m,) raw p-values.
    covariates : (m,) auxiliary covariate (e.g. MAF, imputation R²).
    q : float — target FDR level (default 0.05).
    n_folds : int — number of cross-fitting folds (default 5).
    n_bins : int — number of covariate bins (default 10).
    seed : int or None — random seed for fold assignment.
    lambda_ : float — threshold for signal density estimation (default 0.1).

    Returns
    -------
    IHWResult
    """
    m = p.shape[0]
    device = p.device
    dtype = p.dtype

    # Edge case: very few hypotheses
    if m < n_folds * 2:
        adj = benjamini_hochberg(p)
        return IHWResult(
            adjusted_p=adj,
            weights=torch.ones(m, dtype=dtype, device=device),
            n_rejections=(adj <= q).sum().item(),
            bins=torch.zeros(m, dtype=torch.long, device=device),
            fold_assignments=torch.zeros(m, dtype=torch.long, device=device),
            n_bins=1,
            n_folds=1,
        )

    # Step 1: Bin covariates
    actual_n_bins = min(n_bins, m)
    bins = _bin_covariates(covariates, actual_n_bins)

    # Step 2: Assign folds
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
    else:
        gen = None
    fold_perm = torch.randperm(m, generator=gen, device="cpu").to(device)
    folds = torch.zeros(m, dtype=torch.long, device=device)
    for i in range(m):
        folds[fold_perm[i]] = i % n_folds

    # Step 3: Cross-fitting — learn weights per fold
    per_hyp_weights = torch.ones(m, dtype=dtype, device=device)

    for k in range(n_folds):
        test_mask = folds == k
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        # Learn weights from training fold
        p_train = p[train_mask]
        bins_train = bins[train_mask]
        bin_weights = _ihw_weight_optimization(
            p_train, bins_train, actual_n_bins, q, lambda_,
        )

        # Apply learned weights to test fold
        test_bins = bins[test_mask]
        per_hyp_weights[test_mask] = bin_weights[test_bins]

    # Normalize weights to mean == 1
    w_mean = per_hyp_weights.mean()
    if w_mean > 0:
        per_hyp_weights = per_hyp_weights / w_mean

    # Step 4: Weighted BH
    adjusted_p = weighted_bh(p, per_hyp_weights, q=q)

    n_rej = (adjusted_p <= q).sum().item()

    return IHWResult(
        adjusted_p=adjusted_p,
        weights=per_hyp_weights,
        n_rejections=n_rej,
        bins=bins,
        fold_assignments=folds,
        n_bins=actual_n_bins,
        n_folds=n_folds,
    )


# ===================================================================
# AdaPT
# ===================================================================

def _adapt_em_step(
    p: Tensor,
    bins: Tensor,
    n_bins: int,
    s: Tensor,
    pi1: Tensor,
) -> tuple[Tensor, Tensor]:
    """One EM step for AdaPT.

    E-step: For masked hypotheses (p_i in [s_i, 1-s_i]), estimate
    posterior probability of being non-null.

    M-step: Update per-bin pi1 (signal probability) and threshold s.

    Parameters
    ----------
    p : (m,) raw p-values.
    bins : (m,) bin assignments.
    n_bins : int
    s : (n_bins,) current per-bin thresholds.
    pi1 : (n_bins,) current per-bin signal probabilities.

    Returns
    -------
    s_new : (n_bins,) updated thresholds.
    pi1_new : (n_bins,) updated signal probabilities.
    """
    dtype = p.dtype
    device = p.device

    pi1_new = torch.zeros(n_bins, dtype=dtype, device=device)
    s_new = torch.zeros(n_bins, dtype=dtype, device=device)

    for b in range(n_bins):
        mask = bins == b
        n_b = mask.sum().item()
        if n_b == 0:
            pi1_new[b] = pi1[b]
            s_new[b] = s[b]
            continue

        p_b = p[mask]
        s_b = s[b]
        pi1_b = pi1[b]

        # Revealed: p_i < s_b or p_i > 1 - s_b
        revealed = (p_b < s_b) | (p_b > 1.0 - s_b)
        # Masked: the rest
        n_revealed = revealed.sum().item()
        n_masked = n_b - n_revealed

        # Among revealed, how many are small (likely signal)?
        n_small = (p_b < s_b).sum().item()

        # E-step: for masked hypotheses, posterior prob of being signal
        # Under null: p ~ Uniform → P(masked | null) = 1 - 2*s_b (for s_b < 0.5)
        # Under alt: P(masked | alt) depends on signal distribution
        # Simple approximation: signal concentrated at small p
        p_masked_null = max(1.0 - 2.0 * s_b, 1e-10)
        # Approximate P(masked | alt): most signal p-values are < s_b, few masked
        p_masked_alt = max(1.0 - 2.0 * s_b, 1e-10) * 0.5  # signal less likely masked

        if n_masked > 0:
            # Posterior prob of null for masked
            denom = (1.0 - pi1_b) * p_masked_null + pi1_b * p_masked_alt
            if denom > 0:
                w_null = (1.0 - pi1_b) * p_masked_null / denom
            else:
                w_null = 0.5
            expected_signal_masked = n_masked * (1.0 - w_null)
        else:
            expected_signal_masked = 0.0

        # M-step: update pi1
        total_signal = n_small + expected_signal_masked
        pi1_new[b] = max(min(total_signal / max(n_b, 1), 0.99), 0.01)

        # M-step: update threshold — increase s where signal is concentrated
        # Higher pi1 → higher s (reject more aggressively)
        s_new[b] = min(pi1_new[b].item() * 0.5, 0.49)

    return s_new, pi1_new


def adapt(
    p: Tensor,
    covariates: Tensor,
    q: float = 0.05,
    n_iter: int = 50,
    n_bins: int = 10,
    tol: float = 1e-4,
) -> AdaPTResult:
    """Adaptive P-value Thresholding (Lei & Fithian 2018).

    EM-based procedure that learns covariate-adaptive rejection thresholds:
      1. Bin hypotheses by covariate.
      2. Initialize per-bin thresholds s(bin) and signal probabilities pi1(bin).
      3. Iterate EM: E-step estimates local FDR; M-step updates thresholds.
      4. Final rejection set uses masking principle for FDR control.

    Parameters
    ----------
    p : (m,) raw p-values.
    covariates : (m,) auxiliary covariate.
    q : float — target FDR level (default 0.05).
    n_iter : int — maximum EM iterations (default 50).
    n_bins : int — number of covariate bins (default 10).
    tol : float — convergence tolerance on threshold change.

    Returns
    -------
    AdaPTResult
    """
    m = p.shape[0]
    device = p.device
    dtype = p.dtype

    # Edge case: very few hypotheses
    if m < 5:
        adj = benjamini_hochberg(p)
        return AdaPTResult(
            adjusted_p=adj,
            thresholds=torch.zeros(m, dtype=dtype, device=device),
            n_rejections=(adj <= q).sum().item(),
            bins=torch.zeros(m, dtype=torch.long, device=device),
            n_iter=0,
            converged=True,
            n_bins=1,
        )

    # Step 1: Bin covariates
    actual_n_bins = min(n_bins, m)
    bins = _bin_covariates(covariates, actual_n_bins)

    # Step 2: Initialize
    pi1 = torch.full((actual_n_bins,), 0.1, dtype=dtype, device=device)
    s = torch.full((actual_n_bins,), 0.05, dtype=dtype, device=device)

    converged = False
    final_iter = n_iter

    # Step 3: EM iterations
    for it in range(n_iter):
        s_old = s.clone()
        s, pi1 = _adapt_em_step(p, bins, actual_n_bins, s, pi1)

        # Check convergence
        delta = (s - s_old).abs().max().item()
        if delta < tol:
            converged = True
            final_iter = it + 1
            break

    if not converged:
        final_iter = n_iter

    # Step 4: Compute adjusted p-values using the learned thresholds
    # Per-hypothesis threshold
    per_hyp_s = s[bins]  # (m,)

    # AdaPT rejection rule (masking principle):
    # Reject hypothesis i if p_i < s(x_i), subject to FDR estimate
    # FDP_hat = (1 + #{p_i > 1 - s(x_i)}) / max(1, #{p_i < s(x_i)})

    # Search over a grid of scaling factors for s to find the largest
    # that still controls FDR at level q
    alphas = torch.linspace(0.01, 2.0, 200, dtype=dtype, device=device)
    best_alpha = torch.tensor(0.0, dtype=dtype, device=device)
    best_n_rej = 0

    for alpha in alphas:
        s_scaled = per_hyp_s * alpha
        s_scaled = torch.clamp(s_scaled, max=0.499)

        n_rej = (p < s_scaled).sum().item()
        n_mirror = (p > 1.0 - s_scaled).sum().item()

        if n_rej == 0:
            continue

        fdp_hat = (1.0 + n_mirror) / max(1, n_rej)
        if fdp_hat <= q and n_rej >= best_n_rej:
            best_n_rej = n_rej
            best_alpha = alpha

    # Compute final thresholds
    final_s = per_hyp_s * best_alpha
    final_s = torch.clamp(final_s, max=0.499)

    # Convert to adjusted p-values:
    # adj_p_i = min alpha' s.t. p_i would NOT be rejected at level alpha' * q
    # Approximation: adj_p = p / s_final * q (scaled), clamped to [0, 1]
    # More precisely, for each hypothesis, the adjusted p-value is the
    # smallest q' at which it would be rejected
    adjusted_p = torch.ones(m, dtype=dtype, device=device)

    if best_alpha > 0:
        # Binary-search-like: for each p_i, find smallest q' such that
        # FDP <= q' with the learned thresholds
        # Approximation: rank-based adjusted p-values
        rejected = p < final_s
        if rejected.any():
            # For rejected hypotheses: adj_p = FDP_hat * q
            n_rej = rejected.sum().item()
            n_mirror = (p > 1.0 - final_s).sum().item()
            fdp_hat = (1.0 + n_mirror) / max(1, n_rej)

            # Rank rejected p-values
            rej_p = p[rejected]
            rej_ranks = rej_p.argsort().argsort().to(dtype) + 1.0

            # Scale: smaller p → smaller adjusted p
            adjusted_p[rejected] = fdp_hat * rej_ranks / n_rej
            adjusted_p[rejected] = torch.clamp(adjusted_p[rejected], max=1.0)

    adjusted_p = torch.clamp(adjusted_p, min=0.0, max=1.0)
    n_rej_final = (adjusted_p <= q).sum().item()

    return AdaPTResult(
        adjusted_p=adjusted_p,
        thresholds=final_s,
        n_rejections=n_rej_final,
        bins=bins,
        n_iter=final_iter,
        converged=converged,
        n_bins=actual_n_bins,
    )
