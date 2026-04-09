"""Spatio-temporal random regression GWAS (Phase 38, Step 8).

Combines a temporal random-regression LMM (Legendre / B-spline basis on
``time_values``) with a fixed-effect 2D tensor-product P-spline spatial
smoother on per-observation field-trial coordinates ``(row, col)``.

Method: two-step deconfounding
------------------------------
The spatial nuisance surface is removed *before* the temporal RR fit by
solving a penalized least-squares problem on the long-format phenotype:

    c_S = argmin_c  ||Y_long - Phi_S c||² + λ_r c' P_row c + λ_c c' P_col c

with ``Phi_S`` the 2D P-spline tensor-product design matrix and
``(P_row, P_col)`` the marginal difference penalties from
:func:`torchgwas.linalg.basis.pspline_2d`. The cleaned phenotype
``Y_clean = Y_long - Phi_S c_S`` is then handed off to a standard
:class:`RandomRegressionLMM`.

This is the simplest of three possible designs (the others are
joint REML and a random-effect smoother) and matches the previous
agent's spec: 2D P-spline ships as a *fixed-effect* smoother in
Phase 38, with the random-effect path deferred to a follow-up phase.

References
----------
- Eilers, Marx (1996) — Flexible smoothing with B-splines and penalties.
- Lee, Durbán (2011) — P-spline ANOVA-type interaction models.
- Rodríguez-Álvarez et al. (2018) — SpATS field-trial spatial correction.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.basis import pspline_2d
from .base import NullFit, VariantMeta
from .rr_lmm import RandomRegressionLMM, RRScanResult

logger = logging.getLogger(__name__)


def fit_spatial_pspline(
    Y_long: Tensor,
    row_coords: Tensor,
    col_coords: Tensor,
    *,
    n_knots_row: int = 8,
    n_knots_col: int = 8,
    degree: int = 3,
    penalty_order: int = 2,
    lambda_row: float = 1.0,
    lambda_col: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Penalized least-squares fit of a 2D tensor-product P-spline surface.

    Returns
    -------
    Y_clean : Tensor
        ``(N,)`` phenotype with the fitted spatial surface subtracted.
    spatial_coef : Tensor
        ``(b_row * b_col,)`` fitted P-spline coefficients.
    Phi_S : Tensor
        ``(N, b_row * b_col)`` tensor-product design matrix.
    """
    Y_long = Y_long.to(STAT_DTYPE).reshape(-1)
    row_coords = row_coords.to(STAT_DTYPE).reshape(-1)
    col_coords = col_coords.to(STAT_DTYPE).reshape(-1)
    if Y_long.shape != row_coords.shape or Y_long.shape != col_coords.shape:
        raise ValueError(
            "fit_spatial_pspline: Y_long, row_coords, col_coords must share length."
        )

    Phi_S, P_row, P_col = pspline_2d(
        row_coords, col_coords,
        n_knots_row=n_knots_row, n_knots_col=n_knots_col,
        degree=degree, penalty_order=penalty_order,
    )
    PtP = Phi_S.T @ Phi_S
    PtY = Phi_S.T @ Y_long
    A = PtP + float(lambda_row) * P_row + float(lambda_col) * P_col
    # Tiny ridge for numerical safety
    A = A + 1e-8 * torch.eye(A.shape[0], dtype=STAT_DTYPE, device=A.device)
    spatial_coef = torch.linalg.solve(A, PtY)
    Y_fit = Phi_S @ spatial_coef
    Y_clean = Y_long - Y_fit
    return Y_clean, spatial_coef, Phi_S


class SpatioTemporalRR:
    """Random regression LMM with a 2D P-spline spatial smoother.

    Wraps :class:`RandomRegressionLMM` with a pre-fit spatial deconfounding
    step. ``fit_null`` removes a fitted P-spline field surface from the
    long-format phenotype, then runs the temporal random-regression null
    fit on the cleaned data. ``score_chunk`` is delegated to the inner
    :class:`RandomRegressionLMM`.

    Parameters
    ----------
    basis, order, n_interior_knots, degree, knot_kind,
    k_coef_structure, include_pe, mode, config :
        Forwarded to :class:`RandomRegressionLMM`.
    n_knots_row, n_knots_col :
        Interior-knot counts for each spatial margin.
    spatial_degree :
        B-spline degree for the spatial smoother.
    spatial_penalty_order :
        Difference order for the marginal P-spline penalties.
    lambda_row, lambda_col :
        Smoothing parameters for the row / column margins. Higher values
        produce smoother surfaces. Reasonable defaults of 1.0 work for
        unit-scaled coordinates; tune via cross-validation in practice.
    """

    def __init__(
        self,
        basis: str = "legendre",
        order: int = 3,
        n_interior_knots: int = 4,
        degree: int = 3,
        knot_kind: str = "quantile",
        k_coef_structure: str = "unstructured",
        include_pe: bool = False,
        mode: str = "projection",
        config: Optional[NumericalConfig] = None,
        *,
        n_knots_row: int = 8,
        n_knots_col: int = 8,
        spatial_degree: int = 3,
        spatial_penalty_order: int = 2,
        lambda_row: float = 1.0,
        lambda_col: float = 1.0,
    ) -> None:
        self.inner = RandomRegressionLMM(
            basis=basis,
            order=order,
            n_interior_knots=n_interior_knots,
            degree=degree,
            knot_kind=knot_kind,
            k_coef_structure=k_coef_structure,
            include_pe=include_pe,
            mode=mode,
            config=config,
        )
        self.n_knots_row = n_knots_row
        self.n_knots_col = n_knots_col
        self.spatial_degree = spatial_degree
        self.spatial_penalty_order = spatial_penalty_order
        self.lambda_row = lambda_row
        self.lambda_col = lambda_col

    def fit_null(
        self,
        Y_long: Tensor,
        X0: Tensor,
        K: Tensor,
        *,
        sample_ids: Tensor,
        time_values: Tensor,
        row_coords: Tensor,
        col_coords: Tensor,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Pre-fit spatial smoother, deconfound Y, then fit temporal RR null.

        Parameters
        ----------
        Y_long : (N,) long-format phenotype.
        X0, K : per-individual covariates and GRM (forwarded to inner).
        sample_ids, time_values : (N,) per-observation index/time vectors.
        row_coords, col_coords : (N,) per-observation field-trial layout.
        t_min, t_max : optional time-standardization range.

        Returns
        -------
        NullFit
            With extra spatial-smoother metadata: ``spatial_coef``,
            ``spatial_lambda_row``, ``spatial_lambda_col``,
            ``Y_long_cleaned``.
        """
        Y_clean, spatial_coef, Phi_S = fit_spatial_pspline(
            Y_long, row_coords, col_coords,
            n_knots_row=self.n_knots_row,
            n_knots_col=self.n_knots_col,
            degree=self.spatial_degree,
            penalty_order=self.spatial_penalty_order,
            lambda_row=self.lambda_row,
            lambda_col=self.lambda_col,
        )
        null_fit = self.inner.fit_null(
            Y_clean, X0, K,
            sample_ids=sample_ids,
            time_values=time_values,
            t_min=t_min, t_max=t_max,
        )
        # Stash spatial metadata for downstream inspection
        null_fit.spatial_coef = spatial_coef
        null_fit.spatial_lambda_row = float(self.lambda_row)
        null_fit.spatial_lambda_col = float(self.lambda_col)
        null_fit.Y_long_cleaned = Y_clean
        logger.info(
            "SpatioTemporalRR null fit: spatial b=%d, lambda_row=%.3g, lambda_col=%.3g",
            int(Phi_S.shape[1]), self.lambda_row, self.lambda_col,
        )
        return null_fit

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
        **kwargs: Any,
    ) -> RRScanResult:
        """Delegate to the inner RandomRegressionLMM.score_chunk."""
        return self.inner.score_chunk(
            G_chunk, null_fit, variant_meta, test=test, **kwargs
        )
