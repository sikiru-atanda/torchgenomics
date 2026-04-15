"""Standard-error estimators for the indirect mediation effect ab.

All estimators return ``(se, ci_lower, ci_upper, pvalue)`` for a two-sided test
of H0: ab = 0 at 95% confidence.
"""

from __future__ import annotations

import math

import numpy as np


def sobel_se(a: float, b: float, var_a: float, var_b: float) -> tuple[float, float, float, float]:
    """Delta-method SE assuming Cov(a, b) = 0 (MacKinnon 1995)."""
    var_ab = a * a * var_b + b * b * var_a
    se = math.sqrt(max(var_ab, 0.0))
    ab = a * b
    ci_l = ab - 1.96 * se
    ci_u = ab + 1.96 * se
    if se <= 0.0:
        p = 1.0 if ab == 0.0 else 0.0
    else:
        z = ab / se
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return se, ci_l, ci_u, p


def monte_carlo_se(
    a: float,
    b: float,
    var_a: float,
    var_b: float,
    cov_ab: float = 0.0,
    n_draws: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Draw (a, b) from joint normal, summarise the empirical distribution of ab."""
    if n_draws <= 1:
        raise ValueError(f"monte_carlo_se requires n_draws >= 2; got {n_draws}.")
    rng = np.random.default_rng(seed)
    cov = np.array([[var_a, cov_ab], [cov_ab, var_b]], dtype=np.float64)
    # symmetrise + jitter for numerical PSD
    cov = 0.5 * (cov + cov.T)
    try:
        draws = rng.multivariate_normal([a, b], cov, size=n_draws, method="cholesky")
    except np.linalg.LinAlgError:
        cov = cov + 1e-12 * np.eye(2)
        draws = rng.multivariate_normal([a, b], cov, size=n_draws)
    ab = draws[:, 0] * draws[:, 1]
    se = float(ab.std(ddof=1))
    ci_l = float(np.quantile(ab, 0.025))
    ci_u = float(np.quantile(ab, 0.975))
    # two-sided p: 2 * min(P(ab >= 0), P(ab <= 0))
    p_pos = float((ab >= 0).mean())
    p_neg = float((ab <= 0).mean())
    p = 2.0 * min(p_pos, p_neg)
    p = min(max(p, 0.0), 1.0)
    return se, ci_l, ci_u, p


def bootstrap_se(
    refit_fn,
    n: int,
    n_boot: int = 1_000,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Non-parametric row-bootstrap of (a, b) via a caller-supplied refit function.

    ``refit_fn(idx: np.ndarray) -> (a, b)`` performs both stages on the
    re-sampled rows and returns the new (a, b) point estimates.
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_i, b_i = refit_fn(idx)
        samples[i] = a_i * b_i
    se = float(samples.std(ddof=1))
    ci_l = float(np.quantile(samples, 0.025))
    ci_u = float(np.quantile(samples, 0.975))
    p_pos = float((samples >= 0).mean())
    p_neg = float((samples <= 0).mean())
    p = 2.0 * min(p_pos, p_neg)
    p = min(max(p, 0.0), 1.0)
    return se, ci_l, ci_u, p
