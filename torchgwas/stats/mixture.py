"""Chi-squared mixture distribution p-values for SKAT.

When Q ~ sum(lambda_k * chi2_1), the p-value P(Q > q) requires specialized
methods since Q does not follow a standard distribution.

Two methods:
1. Davies (1980): Numerical inversion of the characteristic function. Exact
   but can fail for extreme eigenvalue ratios.
2. Liu et al. (2009): Moment-matching to a scaled/shifted chi-squared.
   Robust fallback when Davies is unstable.

References:
    - Davies (1980). Algorithm AS 155: distribution of a positive
      combination of chi-squared random variables.
    - Liu et al. (2009). A new chi-square approximation to the distribution
      of non-negative definite quadratic forms in non-central normal variables.
    - Lee et al. (2012). Optimal unified approach for rare-variant association
      testing with application to small-sample case-control WES studies.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


def davies_pvalue(q: float, lambdas: Tensor, acc: float = 1e-6, lim: int = 10000) -> float:
    """P(Q > q) where Q ~ sum(lambda_k * chi2_1) via Davies' method.

    Uses numerical integration of the characteristic function inversion
    formula. Falls back to liu_pvalue() if integration fails.

    Parameters
    ----------
    q : float
        Observed test statistic.
    lambdas : Tensor, shape (r,)
        Positive eigenvalues (weights of chi-squared mixture).
    acc : float
        Target accuracy for numerical integration.
    lim : int
        Maximum number of subintervals.

    Returns
    -------
    float
        P-value clamped to [1e-300, 1.0].
    """
    lam = lambdas.detach().cpu().numpy().astype(np.float64)
    lam = lam[lam > 1e-15]  # filter near-zero eigenvalues

    if len(lam) == 0:
        return 1.0

    q = float(q)
    if q <= 0:
        return 1.0

    # Special case: all eigenvalues equal => Q ~ lambda * chi2(k)
    if np.allclose(lam, lam[0], rtol=1e-8):
        from scipy.stats import chi2
        k = len(lam)
        p = chi2.sf(q / lam[0], df=k)
        return float(np.clip(p, 1e-300, 1.0))

    # Davies' method via amplitude/phase separation (numerically stable).
    # P(Q > q) = 0.5 - (1/pi) * int_0^inf sin(theta(u)) / (u * R(u)) du
    # where theta(u) = 0.5 * sum(arctan(2*u*lam_k)) - 0.5*q*u
    # and   R(u) = prod((1 + 4*u^2*lam_k^2)^{1/4})
    try:
        from scipy.integrate import quad

        def _integrand(u: float) -> float:
            if u == 0:
                return 0.0
            # Phase: 0.5 * sum(arctan(2*u*lam_k)) - q*u
            theta = 0.5 * np.sum(np.arctan(2.0 * u * lam)) - q * u
            # Amplitude: prod((1 + 4*u^2*lam_k^2)^{1/4})
            log_R = 0.25 * np.sum(np.log1p(4.0 * u * u * lam * lam))
            return math.sin(theta) * math.exp(-log_R) / u

        integral, abserr = quad(_integrand, 0, np.inf, limit=lim, epsabs=acc, epsrel=acc)
        p = 0.5 + integral / math.pi

        # Detect integration failure: if the integral is near-zero relative
        # to its error estimate, or if Q >> E[Q] but p is suspiciously large,
        # the oscillatory integrand didn't converge — fall back to Liu.
        mean_Q = float(np.sum(lam))  # E[Q] under H0
        integration_unreliable = (
            abs(integral) < 10.0 * abserr  # integral is noise
            or (q > 3.0 * mean_Q and p > 0.1)  # Q is extreme but p is large
        )
        if integration_unreliable:
            logger.debug(
                "Davies integration unreliable (integral=%.4e, err=%.4e, q/E[Q]=%.1f, p=%.4e), "
                "using Liu fallback",
                integral, abserr, q / max(mean_Q, 1e-20), p,
            )
            return liu_pvalue(q, lambdas)

        if 0 < p < 1:
            return float(np.clip(p, 1e-300, 1.0))
        else:
            logger.debug("Davies integration out of range (p=%.4e), using Liu fallback", p)
            return liu_pvalue(q, lambdas)

    except Exception as e:
        logger.debug("Davies integration failed (%s), using Liu fallback", e)
        return liu_pvalue(q, lambdas)


def liu_pvalue(q: float, lambdas: Tensor) -> float:
    """Moment-matching approximation (Liu et al. 2009 / Satterthwaite).

    Matches first 3 cumulants of Q ~ sum(lambda_k * chi2_1) to a
    scaled/shifted chi-squared: (Q - mu_Q)/sigma_Q ~ (chi2(l) - l)/sqrt(2l)
    where l is chosen to match the skewness.

    Parameters
    ----------
    q : float
        Observed test statistic.
    lambdas : Tensor, shape (r,)
        Positive eigenvalues.

    Returns
    -------
    float
        Approximate p-value clamped to [1e-300, 1.0].
    """
    from scipy.stats import chi2

    lam = lambdas.detach().cpu().numpy().astype(np.float64)
    lam = lam[lam > 1e-15]

    if len(lam) == 0:
        return 1.0

    q = float(q)
    if q <= 0:
        return 1.0

    # Cumulants: c_r = 2^{r-1} * (r-1)! * sum(lambda_k^r)
    c1 = np.sum(lam)           # mean
    c2 = 2.0 * np.sum(lam**2)  # variance
    c3 = 8.0 * np.sum(lam**3)  # 3rd cumulant (for skewness)

    if c2 <= 0:
        return 1.0

    # Skewness: s = c3 / c2^{3/2}
    s = c3 / (c2 ** 1.5)

    # Match to chi2(l): skewness of chi2(l) = 2*sqrt(2/l)
    # So l = 8/s^2 (if s > 0)
    if s > 0:
        l = 8.0 / (s * s)
    else:
        # Degenerate: use normal approximation
        z = (q - c1) / math.sqrt(c2)
        from scipy.stats import norm
        p = norm.sf(z)
        return float(np.clip(p, 1e-300, 1.0))

    # Standardize: (Q - mu) / sigma ~ (chi2(l) - l) / sqrt(2l)
    # So chi2_val = (q - c1) * sqrt(2*l) / sqrt(c2) + l
    mu_Q = c1
    sigma_Q = math.sqrt(c2)
    q_standardized = (q - mu_Q) / sigma_Q * math.sqrt(2.0 * l) + l

    p = chi2.sf(q_standardized, df=l)
    return float(np.clip(p, 1e-300, 1.0))


def mixture_chi2_pvalue(
    q: float,
    lambdas: Tensor,
    method: str = "davies",
) -> float:
    """Compute p-value for chi-squared mixture distribution.

    Dispatcher: tries the specified method, falls back to Liu if needed.

    Parameters
    ----------
    q : float
        Observed test statistic.
    lambdas : Tensor, shape (r,)
        Positive eigenvalues.
    method : str
        "davies" (default) or "liu".

    Returns
    -------
    float
        P-value clamped to [1e-300, 1.0].
    """
    if method == "liu":
        return liu_pvalue(q, lambdas)
    else:
        return davies_pvalue(q, lambdas)
