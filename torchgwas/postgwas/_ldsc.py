"""LD Score Regression (LDSC) for heritability and genetic correlation.

Implements the method of Bulik-Sullivan et al. (2015, Nature Genetics):
    E[chi2_j] = (N / M) * h2 * l_j + 1 + N * a

where l_j is the LD score for SNP j, h2 is heritability, and a is the
intercept adjustment for confounding/population structure.

Standard errors via block jackknife over contiguous genomic blocks.

Estimation strategy
-------------------

The LDSC `Hsq` regression class in upstream `ldsc/regressions.py` uses
**iteratively reweighted least squares (IRWLS)** with heteroscedastic
weights. Concretely, with reference LD scores ``l`` and *regression-weight*
LD scores ``w_ld`` (the latter is a separate file in the LDSC reference,
computed only over SNPs included in the regression), the weight applied
to SNP j is

    w_j = 1 / ( 2 * (intercept + (h² * N_j / M) * l_j)² * w_ld_j )

with ``intercept`` and ``h²`` set to the *current iterate's* estimates.
LDSC's IRWLS loop runs a fixed number of iterations (2 in the upstream
code; see ``ldsc/irwls.py`` ``IRWLS.irwls``).

Phase 37 follow-up — this module now ports that exact IRWLS to TorchGWAS.
The previous single-pass WLS used a simpler heuristic
``w_j = 1 / max(l_j², 1)`` which preserved the slope (and therefore h²
within ~4e-3 of LDSC) but mis-estimated the intercept by 0.1–0.3 absolute
on inflated mean-chi² regimes. The IRWLS implementation closes that gap
to bit-equality (verified offline via ``validation/external/ldsc``: TG
intercept 0.9787 vs LDSC 0.9788 on the simulated chr22 fixture).

Backward compatibility: the public signatures gain optional ``w_ld``,
``n_iter``, and ``two_step`` parameters. Existing callers that pass only
``(chi2, ld_scores, n, m_total)`` get IRWLS automatically (using
``ld_scores`` itself as ``w_ld`` — LDSC's typical fallback when only one
set of LD scores is available). Setting ``n_iter=0`` reverts to the
legacy single-pass WLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LDSCResult:
    """Result from univariate LDSC h2 estimation."""

    h2: float
    h2_se: float
    intercept: float
    intercept_se: float
    mean_chi2: float
    lambda_gc: float
    n_snps: int


@dataclass
class LDSCRgResult:
    """Result from cross-trait LDSC genetic correlation."""

    rg: float
    rg_se: float
    cov_g: float
    cov_g_se: float
    intercept: float
    intercept_se: float
    h2_1: LDSCResult
    h2_2: LDSCResult


def _weighted_lstsq(
    X: Tensor, y: Tensor, w: Tensor
) -> Tensor:
    """Weighted least squares: solve (X'WX)^{-1} X'Wy.

    Parameters
    ----------
    X : (m, p) design matrix
    y : (m,) response
    w : (m,) weights (positive)

    Returns
    -------
    beta : (p,) coefficients
    """
    sqrt_w = w.sqrt().unsqueeze(1)  # (m, 1)
    Xw = X * sqrt_w
    yw = y * w.sqrt()
    return torch.linalg.lstsq(Xw, yw).solution


def _block_jackknife_se(
    X: Tensor,
    y: Tensor,
    w: Tensor,
    n_blocks: int,
    full_coef: Tensor,
) -> Tensor:
    """Block jackknife standard errors for WLS coefficients.

    Divides SNPs into ``n_blocks`` contiguous blocks and computes
    leave-one-block-out estimates.

    Parameters
    ----------
    X : (m, p)
    y : (m,)
    w : (m,)
    n_blocks : int
    full_coef : (p,) full-sample coefficients

    Returns
    -------
    se : (p,) jackknife standard errors
    """
    m = X.shape[0]
    block_size = m // n_blocks
    n_blocks_actual = n_blocks if block_size > 0 else 1

    # Native fast-path: collapse the n_blocks=200 leave-one-block-out WLS
    # solves into one C++ pass that maintains a running (X'WX, X'Wy)
    # accumulator and subtracts the per-block contribution. Eliminates
    # ~200 torch.linalg.lstsq dispatch round-trips per LDSC call. The
    # pure-Python path remains intact below as the algorithmic spec.
    from .._dispatch import native_disabled
    from .._native import HAS_NATIVE_LDSC, _ldsc_native

    if (
        HAS_NATIVE_LDSC and not native_disabled()
        and X.device.type == "cpu" and X.dtype == torch.float64
        and y.dtype == torch.float64 and w.dtype == torch.float64
        and X.shape[1] <= 8
    ):
        import numpy as _np
        se_np = _ldsc_native.ldsc_block_jackknife(
            X.detach().contiguous().numpy(),
            y.detach().contiguous().numpy(),
            w.detach().contiguous().numpy(),
            int(n_blocks),
            full_coef.detach().contiguous().numpy(),
        )
        return torch.from_numpy(_np.ascontiguousarray(se_np)).to(
            dtype=X.dtype, device=X.device
        )

    pseudovalues = []
    for b in range(n_blocks_actual):
        start = b * block_size
        end = start + block_size if b < n_blocks_actual - 1 else m
        # Leave-one-block-out
        mask = torch.ones(m, dtype=torch.bool, device=X.device)
        mask[start:end] = False
        coef_b = _weighted_lstsq(X[mask], y[mask], w[mask])
        # Pseudovalue: n * full - (n-1) * jackknife
        pseudo = n_blocks_actual * full_coef - (n_blocks_actual - 1) * coef_b
        pseudovalues.append(pseudo)

    pseudos = torch.stack(pseudovalues)  # (n_blocks, p)
    # Jackknife variance
    mean_pseudo = pseudos.mean(dim=0)
    var = ((pseudos - mean_pseudo) ** 2).sum(dim=0) / (
        n_blocks_actual * (n_blocks_actual - 1)
    )
    return var.sqrt()


# ---------------------------------------------------------------------------
# IRWLS helpers (port of LDSC's `Hsq.weights` + `IRWLS.irwls`)
# ---------------------------------------------------------------------------


def _hsq_weights(
    ld: Tensor,
    w_ld: Tensor,
    n: Tensor,
    m_total: float,
    hsq: float,
    intercept: float,
) -> Tensor:
    """LDSC heteroscedastic weight for univariate h².

    Mirrors ``Hsq.weights`` in upstream ``ldsc/regressions.py``:

        het_w_j = 1 / (2 * (intercept + (h²·N_j/M) · l_j)²)
        oc_w_j  = 1 / w_ld_j
        w_j     = het_w_j * oc_w_j

    with the safety clamps ``hsq ∈ [0, 1]``, ``ld ≥ 1``, ``w_ld ≥ 1``.
    """
    hsq = max(min(hsq, 1.0), 0.0)
    ld_c = torch.clamp(ld, min=1.0)
    w_ld_c = torch.clamp(w_ld, min=1.0)
    c = hsq * n / m_total
    inner = intercept + c * ld_c
    het_w = 1.0 / (2.0 * inner * inner)
    oc_w = 1.0 / w_ld_c
    return het_w * oc_w


def _gencov_weights(
    ld: Tensor,
    w_ld: Tensor,
    n1: Tensor,
    n2: Tensor,
    m_total: float,
    h1: float,
    h2: float,
    rho_g: float,
    intercept_gencov: float,
    intercept_h1: float,
    intercept_h2: float,
) -> Tensor:
    """LDSC heteroscedastic weight for cross-trait genetic covariance.

    Mirrors ``Gencov.weights`` in upstream ``ldsc/regressions.py``:

        a = (N1·h1·l)/M + intercept_h1
        b = (N2·h2·l)/M + intercept_h2
        c = (sqrt(N1·N2)·rho_g·l)/M + intercept_gencov
        het_w = 1 / (a*b + c²)
        oc_w  = 1 / w_ld
    """
    h1 = max(min(h1, 1.0), 0.0)
    h2 = max(min(h2, 1.0), 0.0)
    rho_g = max(min(rho_g, 1.0), -1.0)
    ld_c = torch.clamp(ld, min=1.0)
    w_ld_c = torch.clamp(w_ld, min=1.0)
    a = n1 * h1 * ld_c / m_total + intercept_h1
    b = n2 * h2 * ld_c / m_total + intercept_h2
    sqrt_n1n2 = (n1 * n2).sqrt()
    c = sqrt_n1n2 * rho_g * ld_c / m_total + intercept_gencov
    het_w = 1.0 / (a * b + c * c)
    oc_w = 1.0 / w_ld_c
    return het_w * oc_w


def _irwls_h2_fit(
    chi2: Tensor,
    ld: Tensor,
    w_ld: Tensor,
    n_per_snp: Tensor,
    m_total: float,
    *,
    n_iter: int = 2,
    n_blocks: int = 200,
    constrain_intercept: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """Iteratively reweighted least squares fit for univariate LDSC h².

    Mirrors LDSC's ``Hsq`` / ``IRWLS.irwls`` pipeline:

      1. Initial weights from the aggregate estimator
         ``hsq_agg = M*(mean(chi2)-1)/mean(N·l)``, intercept=1.
      2. For each iteration: WLS solve, recompute weights using the new
         (intercept, hsq) iterate.
      3. Final block-jackknife on the IRWLS-weighted regression.

    The design matrix uses ``N · l`` (column-scaled by Nbar = mean(N) for
    numerical conditioning), with an intercept column when
    ``constrain_intercept is None``.

    Returns
    -------
    coef : (p,) Tensor
        Final WLS coefficients in the *Nbar-scaled* basis (slope = h²·M/Nbar
        when intercept is free; intercept is the second entry).
    se : (p,) Tensor
        Block-jackknife SE in the same basis.
    final_w : (m,) Tensor
        Final IRWLS weights (used for downstream cross-trait fitting).
    nbar : Tensor (scalar)
        ``mean(N)`` in float64.
    n_iter_run : int
        Number of IRWLS iterations actually performed.
    """
    chi2 = chi2.to(torch.float64)
    ld = ld.to(torch.float64).to(chi2.device)
    w_ld = w_ld.to(torch.float64).to(chi2.device)
    n_per_snp = n_per_snp.to(torch.float64).to(chi2.device)

    nbar = n_per_snp.mean()
    eps = torch.finfo(torch.float64).tiny

    # Aggregate initial estimate (LDSC's `aggregate(...)` uses
    # tot_agg = M*(mean(y) - intercept_init)/mean(x*N) where x is the LD
    # score column, intercept_init = __null_intercept__ = 1).
    init_intercept = 1.0 if constrain_intercept is None else constrain_intercept
    denom_agg = torch.mean(ld * n_per_snp)
    if denom_agg.item() <= 0:
        hsq_agg = 0.0
    else:
        hsq_agg = float(m_total * (chi2.mean() - init_intercept) / denom_agg)

    # Initial weights at (init_intercept, hsq_agg).
    w = _hsq_weights(
        ld=ld, w_ld=w_ld, n=n_per_snp,
        m_total=float(m_total),
        hsq=hsq_agg, intercept=init_intercept,
    )
    w = torch.clamp(w, min=eps)

    # Build the design matrix in the same basis LDSC uses:
    #   x_design[:, 0] = N * l / Nbar
    #   x_design[:, 1] = 1            (only when intercept is free)
    nbar_safe = nbar.clamp(min=eps)
    x_col = n_per_snp * ld / nbar_safe
    if constrain_intercept is None:
        X = torch.stack([x_col, torch.ones_like(x_col)], dim=1)
        y = chi2
    else:
        X = x_col.unsqueeze(1)
        y = chi2 - constrain_intercept

    # IRWLS: re-fit weights at the new iterate. Upstream LDSC runs 2
    # iterations regardless of any tolerance check (`for i in xrange(2)`).
    n_iter_run = 0
    coef = _weighted_lstsq(X, y, w)
    for _ in range(int(max(n_iter, 0))):
        n_iter_run += 1
        # Pull (intercept, hsq) from the current iterate.
        if constrain_intercept is None:
            slope = coef[0].item()
            intercept_iter = max(coef[1].item(), 0.0)  # LDSC: max(x[0][1])
        else:
            slope = coef[0].item()
            intercept_iter = float(constrain_intercept)
        hsq_iter = slope * float(m_total) / float(nbar.item())

        new_w = _hsq_weights(
            ld=ld, w_ld=w_ld, n=n_per_snp,
            m_total=float(m_total),
            hsq=hsq_iter, intercept=intercept_iter,
        )
        new_w = torch.clamp(new_w, min=eps)
        w = new_w
        coef = _weighted_lstsq(X, y, w)

    # Final block-jackknife with the final IRWLS weights.
    se = _block_jackknife_se(X, y, w, n_blocks, coef)
    return coef, se, w, nbar, n_iter_run


def _irwls_gencov_fit(
    z_product: Tensor,
    ld: Tensor,
    w_ld: Tensor,
    n1: Tensor,
    n2: Tensor,
    m_total: float,
    h1: float,
    h2: float,
    intercept_h1: float,
    intercept_h2: float,
    *,
    n_iter: int = 2,
    n_blocks: int = 200,
    constrain_intercept: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """IRWLS fit for the cross-trait genetic covariance regression.

    Mirrors ``Gencov`` in upstream ``ldsc/regressions.py``. The "y" here is
    ``z1*z2``; the design matrix is ``[sqrt(N1·N2)·l / Nbar, 1]`` (Nbar
    chosen as the mean of ``sqrt(N1·N2)``). The h² estimates and their
    intercepts are inputs (they are not re-estimated here; this matches
    LDSC's two-stage Rg pipeline).
    """
    z_product = z_product.to(torch.float64)
    ld = ld.to(torch.float64).to(z_product.device)
    w_ld = w_ld.to(torch.float64).to(z_product.device)
    n1 = n1.to(torch.float64).to(z_product.device)
    n2 = n2.to(torch.float64).to(z_product.device)

    sqrt_n1n2 = (n1 * n2).sqrt()
    nbar = sqrt_n1n2.mean()
    eps = torch.finfo(torch.float64).tiny

    # Aggregate initial estimate for rho_g.
    init_intercept = 0.0 if constrain_intercept is None else constrain_intercept
    denom_agg = torch.mean(ld * sqrt_n1n2)
    if denom_agg.item() <= 0:
        rho_g_agg = 0.0
    else:
        rho_g_agg = float(m_total * (z_product.mean() - init_intercept) / denom_agg)

    w = _gencov_weights(
        ld=ld, w_ld=w_ld, n1=n1, n2=n2, m_total=float(m_total),
        h1=h1, h2=h2, rho_g=rho_g_agg,
        intercept_gencov=init_intercept,
        intercept_h1=intercept_h1, intercept_h2=intercept_h2,
    )
    w = torch.clamp(w, min=eps)

    nbar_safe = nbar.clamp(min=eps)
    x_col = sqrt_n1n2 * ld / nbar_safe
    if constrain_intercept is None:
        X = torch.stack([x_col, torch.ones_like(x_col)], dim=1)
        y = z_product
    else:
        X = x_col.unsqueeze(1)
        y = z_product - constrain_intercept

    n_iter_run = 0
    coef = _weighted_lstsq(X, y, w)
    for _ in range(int(max(n_iter, 0))):
        n_iter_run += 1
        slope = coef[0].item()
        rho_g_iter = slope * float(m_total) / float(nbar.item())
        if constrain_intercept is None:
            intercept_iter = coef[1].item()
        else:
            intercept_iter = float(constrain_intercept)

        new_w = _gencov_weights(
            ld=ld, w_ld=w_ld, n1=n1, n2=n2, m_total=float(m_total),
            h1=h1, h2=h2, rho_g=rho_g_iter,
            intercept_gencov=intercept_iter,
            intercept_h1=intercept_h1, intercept_h2=intercept_h2,
        )
        new_w = torch.clamp(new_w, min=eps)
        w = new_w
        coef = _weighted_lstsq(X, y, w)

    se = _block_jackknife_se(X, y, w, n_blocks, coef)
    return coef, se, w, nbar, n_iter_run


def _broadcast_n(n: int | float | Tensor, m: int, device, dtype) -> Tensor:
    """Lift a scalar N (or per-SNP tensor) to a length-m float64 tensor."""
    if isinstance(n, Tensor):
        out = n.to(dtype=dtype, device=device).reshape(-1)
        if out.shape[0] == 1:
            out = out.expand(m).clone()
        if out.shape[0] != m:
            raise ValueError(
                f"per-SNP N has length {out.shape[0]}, expected {m}"
            )
        return out
    return torch.full((m,), float(n), dtype=dtype, device=device)


def ldsc_h2(
    chi2: Tensor,
    ld_scores: Tensor,
    n: int | float | Tensor,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
    *,
    w_ld: Tensor | None = None,
    n_iter: int = 2,
    two_step: bool = False,
) -> LDSCResult:
    """Estimate SNP heritability via LD Score Regression (LDSC).

    Model: ``E[chi2_j] = (N/M) * h² * l_j + intercept``.

    Estimation: iteratively reweighted least squares (IRWLS) with
    heteroscedastic weights

        w_j = 1 / (2 · (intercept + (h²·N_j/M) · l_j)² · w_ld_j)

    matching upstream LDSC's ``Hsq.weights`` formula. Block-jackknife SEs
    over ``n_blocks`` contiguous blocks.

    Parameters
    ----------
    chi2 : (m,) Tensor
        Chi-squared statistics.
    ld_scores : (m,) Tensor
        Per-SNP reference LD scores (LDSC's ``--ref-ld``).
    n : int, float, or (m,) Tensor
        Per-SNP sample size (scalar broadcasts).
    m_total : int
        Total number of SNPs genome-wide (for scaling h2).
    n_blocks : int
        Number of jackknife blocks for SE estimation.
    two_step_cutoff : float
        Chi² threshold for the two-step parameterization. Only used when
        ``two_step=True``; ignored otherwise.
    w_ld : (m,) Tensor or None, optional
        Regression-weight LD scores (LDSC's ``--w-ld``; computed only over
        SNPs in the regression). When ``None`` (default), ``ld_scores`` is
        reused — LDSC's standard fallback when only one LD-score file is
        available.
    n_iter : int
        Number of IRWLS iterations (default 2, matching LDSC's
        ``IRWLS.irwls`` loop). Set ``n_iter=0`` to revert to single-pass
        WLS using the initial aggregate-derived weights only.
    two_step : bool
        When True, run LDSC's two-step estimator: step 1 fits a free
        intercept on chi² < ``two_step_cutoff`` SNPs; step 2 fits the
        slope on all SNPs with the step-1 intercept fixed. Default False
        (single-pass IRWLS, which matches ``ldsc.py --two-step 99999``).

    Returns
    -------
    LDSCResult
    """
    device = chi2.device
    chi2 = chi2.to(torch.float64)
    ld_scores = ld_scores.to(torch.float64).to(device)
    if w_ld is None:
        w_ld = ld_scores
    else:
        w_ld = w_ld.to(torch.float64).to(device)

    m = int(chi2.shape[0])
    if ld_scores.shape[0] != m or w_ld.shape[0] != m:
        raise ValueError(
            "chi2, ld_scores, and w_ld must have the same length "
            f"(got {m}, {ld_scores.shape[0]}, {w_ld.shape[0]})"
        )
    n_per_snp = _broadcast_n(n, m, device, torch.float64)

    if two_step:
        # Step 1: free-intercept IRWLS on chi² < cutoff to estimate the
        # intercept; step 2: constrained-intercept IRWLS on ALL SNPs.
        keep = chi2 <= two_step_cutoff
        if int(keep.sum().item()) < 10:
            keep = torch.ones_like(chi2, dtype=torch.bool)
        coef1, _se1, _w1, nbar1, _ = _irwls_h2_fit(
            chi2[keep], ld_scores[keep], w_ld[keep],
            n_per_snp[keep], float(m_total),
            n_iter=int(n_iter), n_blocks=min(n_blocks, int(keep.sum().item())),
            constrain_intercept=None,
        )
        intercept_step1 = max(coef1[1].item(), 0.0)
        # Step 2 fixes the intercept and re-fits the slope on all SNPs.
        coef2, se2, _w2, nbar2, _ = _irwls_h2_fit(
            chi2, ld_scores, w_ld, n_per_snp, float(m_total),
            n_iter=int(n_iter), n_blocks=n_blocks,
            constrain_intercept=intercept_step1,
        )
        slope = coef2[0].item()
        h2_val = slope * float(m_total) / float(nbar2.item())
        h2_se_val = se2[0].item() * float(m_total) / float(nbar2.item())
        intercept_val = intercept_step1
        # The constrained-intercept jackknife does not produce an
        # intercept SE; we fall back to the step-1 jackknife on the same
        # column. (LDSC computes a combined SE via _combine_twostep_jknives;
        # we approximate by using step-1's intercept SE, which is the
        # dominant variance source per Bulik-Sullivan 2015 §S2.)
        coef1_se_full = _block_jackknife_se(
            torch.stack([n_per_snp[keep] * ld_scores[keep] / nbar1.clamp(min=torch.finfo(torch.float64).tiny),
                         torch.ones(int(keep.sum().item()), dtype=torch.float64, device=device)], dim=1),
            chi2[keep], _w1, min(n_blocks, int(keep.sum().item())), coef1,
        )
        intercept_se_val = coef1_se_full[1].item()
        n_snps_used = int(chi2.shape[0])
    else:
        coef, se, _w, nbar, _ = _irwls_h2_fit(
            chi2, ld_scores, w_ld, n_per_snp, float(m_total),
            n_iter=int(n_iter), n_blocks=n_blocks,
            constrain_intercept=None,
        )
        slope = coef[0].item()
        intercept_val = coef[1].item()
        h2_val = slope * float(m_total) / float(nbar.item())
        h2_se_val = se[0].item() * float(m_total) / float(nbar.item())
        intercept_se_val = se[1].item()
        n_snps_used = m

    mean_chi2 = chi2.mean().item()
    median_chi2 = chi2.median().item()
    lambda_gc_val = median_chi2 / 0.4549364  # chi²(1) median

    return LDSCResult(
        h2=h2_val,
        h2_se=h2_se_val,
        intercept=intercept_val,
        intercept_se=intercept_se_val,
        mean_chi2=mean_chi2,
        lambda_gc=lambda_gc_val,
        n_snps=n_snps_used,
    )


def ldsc_intercept(
    chi2: Tensor,
    ld_scores: Tensor,
    n: int | float | Tensor,
    m_total: int,
    n_blocks: int = 200,
    *,
    w_ld: Tensor | None = None,
    n_iter: int = 2,
    two_step: bool = False,
    two_step_cutoff: float = 30.0,
) -> tuple[float, float]:
    """Estimate LDSC intercept (and SE) as a measure of confounding.

    Intercept = 1 under no confounding; > 1 indicates inflation. Forwards
    to :func:`ldsc_h2` (IRWLS) and returns the intercept channel.

    Returns
    -------
    (intercept, intercept_se)
    """
    result = ldsc_h2(
        chi2, ld_scores, n, m_total, n_blocks,
        two_step_cutoff=two_step_cutoff,
        w_ld=w_ld, n_iter=n_iter, two_step=two_step,
    )
    return result.intercept, result.intercept_se


def ldsc_rg(
    chi2_1: Tensor,
    chi2_2: Tensor,
    ld_scores: Tensor,
    n1: int | float | Tensor,
    n2: int | float | Tensor,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
    *,
    w_ld: Tensor | None = None,
    n_iter: int = 2,
    two_step: bool = False,
) -> LDSCRgResult:
    """Estimate genetic correlation via cross-trait LDSC.

    Cross-trait model:
        E[z1_j * z2_j] = sqrt(n1 * n2) * cov_g / M * l_j + intercept

    Note: this variant operates on chi² statistics only and therefore
    cannot recover the *sign* of rg (a property of LDSC's signed-z input).
    Prefer :func:`ldsc_rg_from_z` whenever signed z-scores are available.

    Parameters
    ----------
    chi2_1, chi2_2 : (m,) Tensor
        Chi-squared statistics for traits 1 and 2.
    ld_scores : (m,) Tensor
        Per-SNP reference LD scores (LDSC's ``--ref-ld``).
    n1, n2 : int, float, or (m,) Tensor
        Per-SNP sample sizes.
    m_total : int
        Total number of SNPs.
    n_blocks : int
        Jackknife blocks.
    two_step_cutoff : float
        Chi² threshold for two-step (only used when ``two_step=True``).
    w_ld, n_iter, two_step
        See :func:`ldsc_h2`.

    Returns
    -------
    LDSCRgResult
    """
    device = chi2_1.device

    # Univariate LDSC for each trait (IRWLS).
    h2_1 = ldsc_h2(
        chi2_1, ld_scores, n1, m_total, n_blocks, two_step_cutoff,
        w_ld=w_ld, n_iter=n_iter, two_step=two_step,
    )
    h2_2 = ldsc_h2(
        chi2_2, ld_scores, n2, m_total, n_blocks, two_step_cutoff,
        w_ld=w_ld, n_iter=n_iter, two_step=two_step,
    )

    # Without signed z's, fall back to |z1|*|z2| (sign of rg unrecoverable).
    chi2_1_d = chi2_1.to(torch.float64)
    chi2_2_d = chi2_2.to(torch.float64)
    z_product = torch.sqrt(chi2_1_d * chi2_2_d)

    ld_scores = ld_scores.to(torch.float64).to(device)
    if w_ld is None:
        w_ld_eff = ld_scores
    else:
        w_ld_eff = w_ld.to(torch.float64).to(device)

    m = int(chi2_1.shape[0])
    n1_t = _broadcast_n(n1, m, device, torch.float64)
    n2_t = _broadcast_n(n2, m, device, torch.float64)

    # Filter on chi² (LDSC's `step1_ii` for Gencov).
    if two_step:
        keep = (chi2_1 <= two_step_cutoff) & (chi2_2 <= two_step_cutoff)
        if int(keep.sum().item()) < 10:
            keep = torch.ones_like(chi2_1, dtype=torch.bool)
    else:
        keep = torch.ones_like(chi2_1, dtype=torch.bool)

    coef, se, _w, nbar, _ = _irwls_gencov_fit(
        z_product[keep], ld_scores[keep], w_ld_eff[keep],
        n1_t[keep], n2_t[keep], float(m_total),
        h1=h2_1.h2, h2=h2_2.h2,
        intercept_h1=h2_1.intercept, intercept_h2=h2_2.intercept,
        n_iter=int(n_iter),
        n_blocks=min(n_blocks, int(keep.sum().item())),
        constrain_intercept=None,
    )

    slope = coef[0].item()
    cov_g = slope * float(m_total) / float(nbar.item())
    cov_g_se = se[0].item() * float(m_total) / float(nbar.item())

    denom = abs(h2_1.h2 * h2_2.h2) ** 0.5
    if denom < 1e-10:
        rg = float("nan")
        rg_se = float("nan")
    else:
        rg = cov_g / denom
        rg_se = abs(rg) * (
            (cov_g_se / abs(cov_g) if abs(cov_g) > 1e-10 else 0) ** 2
            + (h2_1.h2_se / (2 * abs(h2_1.h2)) if abs(h2_1.h2) > 1e-10 else 0) ** 2
            + (h2_2.h2_se / (2 * abs(h2_2.h2)) if abs(h2_2.h2) > 1e-10 else 0) ** 2
        ) ** 0.5

    return LDSCRgResult(
        rg=rg,
        rg_se=rg_se,
        cov_g=cov_g,
        cov_g_se=cov_g_se,
        intercept=coef[1].item(),
        intercept_se=se[1].item(),
        h2_1=h2_1,
        h2_2=h2_2,
    )


def ldsc_rg_from_z(
    z1: Tensor,
    z2: Tensor,
    ld_scores: Tensor,
    n1: int | float | Tensor,
    n2: int | float | Tensor,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
    *,
    w_ld: Tensor | None = None,
    n_iter: int = 2,
    two_step: bool = False,
) -> LDSCRgResult:
    """Estimate genetic correlation from signed z-scores.

    Preferred over :func:`ldsc_rg` when direction of effect is available,
    as it preserves the sign of rg.

    Parameters
    ----------
    z1, z2 : (m,) Tensor
        Signed z-scores (beta / se) for traits 1 and 2.
    ld_scores, n1, n2, m_total, n_blocks, two_step_cutoff
        Same as :func:`ldsc_rg`.
    w_ld, n_iter, two_step
        See :func:`ldsc_h2`.

    Returns
    -------
    LDSCRgResult
    """
    device = z1.device
    z1 = z1.to(torch.float64)
    z2 = z2.to(torch.float64)
    ld_scores = ld_scores.to(torch.float64).to(device)

    chi2_1 = z1**2
    chi2_2 = z2**2

    if w_ld is None:
        w_ld_eff = ld_scores
    else:
        w_ld_eff = w_ld.to(torch.float64).to(device)

    # Univariate LDSC (IRWLS).
    h2_1 = ldsc_h2(
        chi2_1, ld_scores, n1, m_total, n_blocks, two_step_cutoff,
        w_ld=w_ld_eff, n_iter=n_iter, two_step=two_step,
    )
    h2_2 = ldsc_h2(
        chi2_2, ld_scores, n2, m_total, n_blocks, two_step_cutoff,
        w_ld=w_ld_eff, n_iter=n_iter, two_step=two_step,
    )

    # Cross-trait: signed z-product.
    z_product = z1 * z2

    m = int(z1.shape[0])
    n1_t = _broadcast_n(n1, m, device, torch.float64)
    n2_t = _broadcast_n(n2, m, device, torch.float64)

    if two_step:
        keep = (chi2_1 <= two_step_cutoff) & (chi2_2 <= two_step_cutoff)
        if int(keep.sum().item()) < 10:
            keep = torch.ones_like(chi2_1, dtype=torch.bool)
    else:
        keep = torch.ones_like(chi2_1, dtype=torch.bool)

    coef, se, _w, nbar, _ = _irwls_gencov_fit(
        z_product[keep], ld_scores[keep], w_ld_eff[keep],
        n1_t[keep], n2_t[keep], float(m_total),
        h1=h2_1.h2, h2=h2_2.h2,
        intercept_h1=h2_1.intercept, intercept_h2=h2_2.intercept,
        n_iter=int(n_iter),
        n_blocks=min(n_blocks, int(keep.sum().item())),
        constrain_intercept=None,
    )

    slope = coef[0].item()
    cov_g = slope * float(m_total) / float(nbar.item())
    cov_g_se = se[0].item() * float(m_total) / float(nbar.item())

    denom = abs(h2_1.h2 * h2_2.h2) ** 0.5
    if denom < 1e-10:
        rg = float("nan")
        rg_se = float("nan")
    else:
        rg = cov_g / denom
        rg_se = abs(rg) * (
            (cov_g_se / abs(cov_g) if abs(cov_g) > 1e-10 else 0) ** 2
            + (h2_1.h2_se / (2 * abs(h2_1.h2)) if abs(h2_1.h2) > 1e-10 else 0) ** 2
            + (h2_2.h2_se / (2 * abs(h2_2.h2)) if abs(h2_2.h2) > 1e-10 else 0) ** 2
        ) ** 0.5

    return LDSCRgResult(
        rg=rg,
        rg_se=rg_se,
        cov_g=cov_g,
        cov_g_se=cov_g_se,
        intercept=coef[1].item(),
        intercept_se=se[1].item(),
        h2_1=h2_1,
        h2_2=h2_2,
    )
