"""Statistical power analysis for GWAS.

Computes detection power, minimum detectable effect sizes, and required
sample sizes for genome-wide association studies using the non-centrality
parameter (NCP) approach (Sham & Purcell, 2014, Nature Reviews Genetics).

For a quantitative trait under the additive model:

    NCP = 2 * N * af * (1 - af) * beta^2
    power = Phi(sqrt(NCP) - z_alpha) + Phi(-sqrt(NCP) - z_alpha)

where ``z_alpha = Phi^{-1}(1 - alpha/2)`` for a two-sided test. The second
term is negligible for NCP > 4 and is included for completeness.

For binary traits (case-control), an effective sample size
``N_eff = 4 / (1/N_cases + 1/N_controls)`` replaces ``N``.

The ``power_curve`` function returns the minimum detectable effect size at
each allele frequency for a given sample size and power target — this is
the curve that forms the "trumpet" envelope in trumpet plots.

**Polyploid compatibility.** Power depends on allele frequency and effect
size in the *GWAS parameterization*. For polyploid organisms the per-allele
effect is already scaled by ploidy in the GWAS model, so the NCP formula
is ploidy-agnostic when fed the GWAS-reported beta and allele frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Normal helpers (pure torch, no scipy)
# ---------------------------------------------------------------------------

def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))."""
    return 0.5 * torch.erfc(-x / (2.0 ** 0.5))


def _normal_sf(x: Tensor) -> Tensor:
    """Standard normal survival: 1 - Phi(x)."""
    return 0.5 * torch.erfc(x / (2.0 ** 0.5))


def _normal_quantile(p: Tensor) -> Tensor:
    """Normal quantile (probit) via scipy (handles extreme tails)."""
    from scipy.stats import norm  # type: ignore[import-untyped]
    return torch.tensor(
        norm.ppf(p.detach().cpu().numpy()),
        dtype=p.dtype, device=p.device,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PowerResult:
    """Result from a GWAS power calculation.

    Attributes
    ----------
    power : (m,) Tensor
        Detection probability per variant.
    ncp : (m,) Tensor
        Non-centrality parameter per variant.
    alpha : float
        Significance threshold used.
    n : int or float
        Sample size.
    min_detectable_beta : (m,) Tensor
        Minimum ``|beta|`` for ``target_power`` at each allele frequency.
    """

    power: Tensor
    ncp: Tensor
    alpha: float
    n: int | float
    min_detectable_beta: Tensor


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def gwas_power(
    n: int | float,
    af: Tensor,
    beta: Tensor,
    alpha: float = 5e-8,
    target_power: float = 0.8,
) -> PowerResult:
    """Compute GWAS detection power for each variant.

    Parameters
    ----------
    n : int or float
        Sample size (or effective N for case-control).
    af : (m,) Tensor
        Allele frequencies in (0, 1).
    beta : (m,) Tensor
        Per-SNP effect sizes.
    alpha : float
        Genome-wide significance threshold (default 5e-8).
    target_power : float
        Power target for minimum detectable effect size (default 0.8).

    Returns
    -------
    PowerResult
    """
    if not isinstance(af, Tensor):
        af = torch.tensor(af, dtype=torch.float64)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=torch.float64)
    af = af.to(torch.float64)
    beta = beta.to(torch.float64)

    # NCP = 2 * N * af * (1 - af) * beta^2
    ncp = 2.0 * float(n) * af * (1.0 - af) * beta ** 2

    # z_alpha for two-sided test
    z_alpha = _normal_quantile(
        torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64, device=af.device)
    )

    sqrt_ncp = torch.sqrt(torch.clamp(ncp, min=0.0))
    # power = Phi(sqrt(NCP) - z_alpha) + Phi(-sqrt(NCP) - z_alpha)
    pwr = _normal_cdf(sqrt_ncp - z_alpha) + _normal_cdf(-sqrt_ncp - z_alpha)

    # Minimum detectable beta at target_power
    min_beta = _min_detectable_beta(n, af, alpha, target_power)

    return PowerResult(
        power=pwr,
        ncp=ncp,
        alpha=alpha,
        n=n,
        min_detectable_beta=min_beta,
    )


def power_curve(
    n: int | float,
    af_grid: Tensor,
    alpha: float = 5e-8,
    target_power: float = 0.8,
) -> Tensor:
    """Minimum detectable ``|beta|`` at each allele frequency.

    This is the envelope curve used in trumpet plots. For each AF value,
    returns the smallest ``|beta|`` that yields ``target_power`` detection
    probability at sample size ``n`` and significance level ``alpha``.

    Parameters
    ----------
    n : int or float
        Sample size.
    af_grid : (g,) Tensor
        Allele frequency values (typically a linspace or logspace grid).
    alpha : float
        Significance threshold.
    target_power : float
        Target power level (e.g. 0.8 for 80%).

    Returns
    -------
    (g,) Tensor
        Minimum detectable ``|beta|`` at each AF.
    """
    return _min_detectable_beta(n, af_grid, alpha, target_power)


def required_n(
    af: Tensor,
    beta: Tensor,
    alpha: float = 5e-8,
    target_power: float = 0.8,
) -> Tensor:
    """Sample size needed to achieve ``target_power`` for each variant.

    Parameters
    ----------
    af : (m,) Tensor
        Allele frequencies.
    beta : (m,) Tensor
        Effect sizes.
    alpha : float
        Significance threshold.
    target_power : float
        Target power level.

    Returns
    -------
    (m,) Tensor
        Required N per variant (float, may be non-integer).
    """
    if not isinstance(af, Tensor):
        af = torch.tensor(af, dtype=torch.float64)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=torch.float64)
    af = af.to(torch.float64)
    beta = beta.to(torch.float64)

    z_alpha = _normal_quantile(
        torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64, device=af.device)
    )
    z_beta = _normal_quantile(
        torch.tensor(float(target_power), dtype=torch.float64, device=af.device)
    )

    # Power = Phi(sqrt(NCP) - z_alpha). For power = target_power we need
    # sqrt(NCP) = z_alpha + z_beta, so NCP_target = (z_alpha + z_beta)^2.
    ncp_target = (z_alpha + z_beta) ** 2

    # N = NCP / (2 * af * (1-af) * beta^2)
    denom = 2.0 * af * (1.0 - af) * beta ** 2
    denom = torch.clamp(denom, min=1e-300)
    return ncp_target / denom


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _min_detectable_beta(
    n: int | float,
    af: Tensor,
    alpha: float,
    target_power: float,
) -> Tensor:
    """Closed-form minimum |beta| from the normal-approximation inversion."""
    if not isinstance(af, Tensor):
        af = torch.tensor(af, dtype=torch.float64)
    af = af.to(torch.float64)

    z_alpha = _normal_quantile(
        torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64, device=af.device)
    )
    z_beta = _normal_quantile(
        torch.tensor(float(target_power), dtype=torch.float64, device=af.device)
    )

    # NCP_target = (z_alpha - z_beta)^2
    ncp_target = (z_alpha - z_beta) ** 2

    # beta = sqrt(NCP / (2 * N * af * (1-af)))
    denom = 2.0 * float(n) * af * (1.0 - af)
    denom = torch.clamp(denom, min=1e-300)
    return torch.sqrt(ncp_target / denom)
