"""Imai/Keele/Yamamoto (2010) sensitivity analysis for unmeasured M→Y confounders.

For the linear-Gaussian mediation model, the indirect effect under residual
confounding ρ = corr(e_M, e_Y) is

    ACME(ρ) = a · (b - ρ · σ_u / σ_v)

where σ_v = sd of the Stage-M residual and σ_u = sd of the Stage-Y residual.
The minimum |ρ| that drives ACME to zero is

    ρ* = b · σ_v / σ_u

clamped to [-1, 1]. A small |ρ*| means even modest M↔Y confounding nullifies
the indirect effect; a large |ρ*| (or unattainable >1) means the conclusion
is robust.
"""

from __future__ import annotations

import math


def imai_rho_sensitivity(
    b: float,
    sigma_v: float,
    sigma_u: float,
) -> float:
    """Return the critical residual correlation ρ* that nulls the indirect effect.

    Returns NaN if either residual SD is zero.
    """
    if sigma_v <= 0.0 or sigma_u <= 0.0 or not math.isfinite(sigma_v) or not math.isfinite(sigma_u):
        return float("nan")
    rho = b * sigma_v / sigma_u
    if rho > 1.0:
        rho = 1.0
    elif rho < -1.0:
        rho = -1.0
    return float(rho)
