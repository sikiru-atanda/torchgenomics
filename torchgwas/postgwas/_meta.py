"""Meta-analysis methods for combining GWAS results across studies.

Implements four standard approaches:
- Fixed-effect (inverse-variance weighted, IVW)
- Random-effect (DerSimonian & Laird, 1986)
- Sample-size weighted (Stouffer's Z)
- Han-Eskin RE2 (Han & Eskin, 2011, AJHG) — modified likelihood ratio

All methods operate per-SNP across K studies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..stats.tests import chi2_sf


@dataclass
class MetaResult:
    """Result from a meta-analysis across K studies."""

    beta_meta: Tensor  # (m,) combined effect
    se_meta: Tensor  # (m,) combined SE
    p_meta: Tensor  # (m,) combined p-value
    z_meta: Tensor  # (m,) combined z-score
    q_stat: Tensor  # (m,) Cochran's Q
    i2: Tensor  # (m,) I-squared heterogeneity
    tau2: Tensor  # (m,) between-study variance (0 for fixed-effect)
    method: str


def _cochran_q(
    beta: Tensor, se: Tensor, beta_fe: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Cochran's Q, I-squared, and DerSimonian-Laird tau-squared.

    Parameters
    ----------
    beta : (m, K) per-study effects
    se : (m, K) per-study SEs
    beta_fe : (m,) fixed-effect combined beta

    Returns
    -------
    q_stat : (m,)
    i2 : (m,)
    tau2 : (m,)
    """
    w = 1.0 / (se**2)  # (m, K)
    K = beta.shape[1]

    # Q = sum w_k * (beta_k - beta_fe)^2
    resid = beta - beta_fe.unsqueeze(1)
    q_stat = (w * resid**2).sum(dim=1)

    # I^2 = max(0, (Q - df) / Q)
    df = K - 1
    i2 = torch.clamp((q_stat - df) / q_stat, min=0.0)
    i2 = torch.where(q_stat > 0, i2, torch.zeros_like(i2))

    # DL tau^2 = max(0, (Q - df) / (sum(w) - sum(w^2)/sum(w)))
    w_sum = w.sum(dim=1)
    w_sq_sum = (w**2).sum(dim=1)
    c = w_sum - w_sq_sum / w_sum
    tau2 = torch.clamp((q_stat - df) / c, min=0.0)

    return q_stat, i2, tau2


def meta_fixed_effect(
    beta: Tensor,
    se: Tensor,
) -> MetaResult:
    """Fixed-effect meta-analysis via inverse-variance weighting.

    Parameters
    ----------
    beta : (m, K) Tensor
        Per-SNP effect sizes from K studies.
    se : (m, K) Tensor
        Per-SNP standard errors from K studies.

    Returns
    -------
    MetaResult
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)

    w = 1.0 / (se**2)  # (m, K)
    w_sum = w.sum(dim=1)  # (m,)

    beta_meta = (w * beta).sum(dim=1) / w_sum
    se_meta = 1.0 / w_sum.sqrt()
    z_meta = beta_meta / se_meta
    p_meta = _z_to_p(z_meta)

    q_stat, i2, _ = _cochran_q(beta, se, beta_meta)

    return MetaResult(
        beta_meta=beta_meta,
        se_meta=se_meta,
        p_meta=p_meta,
        z_meta=z_meta,
        q_stat=q_stat,
        i2=i2,
        tau2=torch.zeros_like(beta_meta),
        method="fixed_effect",
    )


def meta_random_effect(
    beta: Tensor,
    se: Tensor,
) -> MetaResult:
    """Random-effects meta-analysis (DerSimonian & Laird, 1986).

    Adds between-study variance tau^2 to within-study variances.

    Parameters
    ----------
    beta : (m, K) Tensor
    se : (m, K) Tensor

    Returns
    -------
    MetaResult
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)

    # First get fixed-effect estimate for Q computation
    w_fe = 1.0 / (se**2)
    beta_fe = (w_fe * beta).sum(dim=1) / w_fe.sum(dim=1)

    q_stat, i2, tau2 = _cochran_q(beta, se, beta_fe)

    # Random-effects weights: w* = 1 / (se^2 + tau^2)
    w_re = 1.0 / (se**2 + tau2.unsqueeze(1))
    w_re_sum = w_re.sum(dim=1)

    beta_meta = (w_re * beta).sum(dim=1) / w_re_sum
    se_meta = 1.0 / w_re_sum.sqrt()
    z_meta = beta_meta / se_meta
    p_meta = _z_to_p(z_meta)

    return MetaResult(
        beta_meta=beta_meta,
        se_meta=se_meta,
        p_meta=p_meta,
        z_meta=z_meta,
        q_stat=q_stat,
        i2=i2,
        tau2=tau2,
        method="random_effect",
    )


def meta_sample_size(
    p: Tensor,
    n: Tensor,
    direction: Tensor | None = None,
) -> MetaResult:
    """Sample-size weighted meta-analysis (Stouffer's Z).

    Combines p-values using sample-size weighted z-scores.
    When direction of effect is unknown, uses absolute z-scores.

    Parameters
    ----------
    p : (m, K) Tensor
        Per-SNP p-values from K studies.
    n : (K,) Tensor
        Sample sizes per study.
    direction : (m, K) Tensor or None
        Sign of effect (+1 or -1). If None, uses two-sided unsigned.

    Returns
    -------
    MetaResult
    """
    p = p.to(torch.float64)
    n = n.to(torch.float64)

    # Convert p to z-scores (two-sided)
    z = _p_to_z(p)

    if direction is not None:
        z = z * direction.to(torch.float64)

    # Weights proportional to sqrt(n)
    w = n.sqrt()  # (K,)
    w = w / w.sum()  # normalize

    z_meta = (z * w.unsqueeze(0)).sum(dim=1)  # (m,)
    p_meta = _z_to_p(z_meta)

    # No meaningful beta/se for sample-size method
    beta_meta = torch.zeros_like(z_meta)
    se_meta = torch.ones_like(z_meta)

    return MetaResult(
        beta_meta=beta_meta,
        se_meta=se_meta,
        p_meta=p_meta,
        z_meta=z_meta,
        q_stat=torch.zeros_like(z_meta),
        i2=torch.zeros_like(z_meta),
        tau2=torch.zeros_like(z_meta),
        method="sample_size",
    )


def meta_han_eskin(
    beta: Tensor,
    se: Tensor,
) -> MetaResult:
    """Han-Eskin RE2 meta-analysis (Han & Eskin, 2011).

    Modified random-effects test that is more powerful than DerSimonian-Laird
    under effect heterogeneity. Uses a modified likelihood ratio statistic
    that tests H0: all beta_k = 0 vs H1: beta_k ~ N(mu, tau^2).

    The RE2 statistic is: T_RE2 = T_FE^2 * Q_tilde / (K-1+Q_tilde)
    where T_FE is the fixed-effect z-score and Q_tilde is a modified Q.

    Parameters
    ----------
    beta : (m, K) Tensor
    se : (m, K) Tensor

    Returns
    -------
    MetaResult
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)
    K = beta.shape[1]

    # Fixed-effect
    w = 1.0 / (se**2)
    w_sum = w.sum(dim=1)
    beta_fe = (w * beta).sum(dim=1) / w_sum
    se_fe = 1.0 / w_sum.sqrt()
    z_fe = beta_fe / se_fe

    # Cochran's Q
    resid = beta - beta_fe.unsqueeze(1)
    q_stat = (w * resid**2).sum(dim=1)

    # RE2 statistic: modified LRT
    # T_RE2 = z_FE^2 when Q <= K-1 (no heterogeneity detected)
    # T_RE2 = z_FE^2 * Q / (K-1) when Q > K-1
    df = K - 1
    q_ratio = torch.where(
        q_stat > df,
        q_stat / df,
        torch.ones_like(q_stat),
    )
    t_re2 = z_fe**2 * q_ratio

    # P-value from chi2(1)
    p_meta = chi2_sf(t_re2, df=1)

    # Heterogeneity metrics
    i2 = torch.clamp((q_stat - df) / q_stat, min=0.0)
    i2 = torch.where(q_stat > 0, i2, torch.zeros_like(i2))
    w_sq_sum = (w**2).sum(dim=1)
    c = w_sum - w_sq_sum / w_sum
    tau2 = torch.clamp((q_stat - df) / c, min=0.0)

    z_meta = torch.sign(z_fe) * t_re2.sqrt()

    return MetaResult(
        beta_meta=beta_fe,
        se_meta=se_fe,
        p_meta=p_meta,
        z_meta=z_meta,
        q_stat=q_stat,
        i2=i2,
        tau2=tau2,
        method="han_eskin",
    )


# ── Utility functions ──────────────────────────────────────────────

def _z_to_p(z: Tensor) -> Tensor:
    """Two-sided p-value from z-score using normal survival function."""
    # 2 * Phi(-|z|) via erfc
    p = torch.erfc(z.abs() / 2.0**0.5)
    return p.clamp(min=1e-300, max=1.0)


def _p_to_z(p: Tensor) -> Tensor:
    """Convert two-sided p-value to |z|-score (unsigned)."""
    # z = Phi^{-1}(1 - p/2) = sqrt(2) * erfinv(1 - p)
    # Clamp p to avoid infinities
    p_c = p.clamp(min=1e-300, max=1.0 - 1e-10)
    z = (2.0**0.5) * torch.erfinv(1.0 - p_c)
    return z.clamp(min=0.0)
