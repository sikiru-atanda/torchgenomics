"""Genotype-Uncertainty LMM: dosage-variance–corrected association tests.

Most GWAS tools treat imputed/low-depth dosages as exact point estimates,
ignoring the per-sample genotype probability (GP) uncertainty.  This deflates
the information matrix, biases standard errors, and creates calibration
artifacts — especially when uncertainty varies by MAF or imputation panel.

GU-LMM corrects the score test denominator using dosage variance from
genotype probabilities:

    Standard:   T = (g^T P y)^2 / (σ²_e · g^T P g)
    Corrected:  T = (g^T P y)^2 / (σ²_e · (g^T P g + Var_g^T diag(P)))

where diag(P) is the diagonal of the LMM projection matrix, cached once
from the null model.  The correction adds the expected "missing information"
due to genotype uncertainty.

To our knowledge, this is the first GWAS engine that natively supports
GP-aware inference on GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import rotate
from ..optim.reml_math import _compute_P_quantities
from .base import NullFit, ScanResult, VariantMeta
from .single_trait_lmm import SingleTraitLMM, _f_sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GUResult:
    """Extended result from GU-LMM with uncertainty diagnostics."""

    scan: ScanResult
    mean_correction: Tensor   # (m,) mean correction fraction per SNP
    dosage_rsq: Tensor        # (m,) imputation quality per marker


# ---------------------------------------------------------------------------
# Helper: compute diag(P) in original (unrotated) space
# ---------------------------------------------------------------------------

def _compute_diag_P(null_fit: NullFit) -> Tensor:
    """Compute diag(P) where P = V^{-1} - V^{-1}X(X^T V^{-1}X)^{-1}X^T V^{-1}.

    In the eigenspace, V = U diag(evals*lam + 1) U^T, so V^{-1} = U diag(H_inv) U^T.
    The diagonal of P in original space is:

        diag(P)[i] = sum_k U[i,k]^2 * H_inv[k]
                     - sum over (row i of U @ wX @ M00_inv @ wX^T @ U^T)

    Computed once at O(n^2) cost (same order as eigendecomposition).

    Parameters
    ----------
    null_fit : NullFit with eigenvalues, eigenvectors, M00, sig2_g, sig2_e, X0_rot

    Returns
    -------
    diag_P : (n,) float64 — diagonal of P in original sample space
    """
    U = null_fit.eigenvectors       # (n, n) or (n, k)
    evals = null_fit.eigenvalues    # (n,) or (k,)
    X0_rot = null_fit.X0_rot        # (n, c) or (k, c)
    M00 = null_fit.M00              # (c, c)
    sig2_g = null_fit.sig2_g
    sig2_e = null_fit.sig2_e

    lam = sig2_g / max(sig2_e, 1e-20)
    H_inv = 1.0 / (evals * lam + 1.0)  # (n,)

    # Term 1: diag(U @ diag(H_inv) @ U^T) = (U^2) @ H_inv
    term1 = (U ** 2) @ H_inv  # (n,)

    # Term 2: diag(U @ wX @ M00_inv @ wX^T @ U^T)
    # where wX = diag(H_inv) @ X0_rot
    wX = H_inv.unsqueeze(1) * X0_rot  # (n, c)
    M00_inv = torch.linalg.inv(M00)   # (c, c)
    # U @ wX = (n, c), then @ M00_inv = (n, c)
    UwX = U @ wX                       # (n, c)
    UwX_M = UwX @ M00_inv             # (n, c)
    # diag of (UwX_M @ UwX^T) = row-wise dot product
    term2 = (UwX_M * UwX).sum(dim=1)  # (n,)

    diag_P = term1 - term2
    return diag_P


# ---------------------------------------------------------------------------
# Main class: GULM
# ---------------------------------------------------------------------------

class GULM:
    """Genotype-Uncertainty LMM with dosage-variance corrected score test.

    Conforms to BaseModel protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    config : NumericalConfig, optional
    p3d : bool
        Use P3D approximation (default True).
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        p3d: bool = True,
    ) -> None:
        self.config = config or NumericalConfig()
        self.p3d = p3d
        self._lmm = SingleTraitLMM(config=self.config, p3d=p3d)

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model and cache diag(P) for uncertainty correction.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates
        K : (n, n) — GRM

        Returns
        -------
        NullFit with additional ``diag_P`` attribute (n,).
        """
        null_fit = self._lmm.fit_null(Y, X0, K=K, **kwargs)

        # Cache diag(P) for uncertainty correction
        null_fit.diag_P = _compute_diag_P(null_fit)

        logger.info(
            "GU-LMM null fit: sig2_g=%.6e, sig2_e=%.6e, h2=%.4f, "
            "mean(diag_P)=%.6f",
            null_fit.sig2_g, null_fit.sig2_e,
            null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
            null_fit.diag_P.mean().item(),
        )

        return null_fit

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
        dosage_var: Tensor | None = None,
    ) -> ScanResult:
        """Score genotype chunk with optional uncertainty correction.

        Parameters
        ----------
        G_chunk : (n, m) — expected dosage
        null_fit : output of fit_null (must have diag_P)
        variant_meta : marker metadata
        test : "score" (uncertainty-corrected) or "wald"/"lrt" (delegated)
        dosage_var : (n, m) optional — per-sample dosage variance from GP.
            If None, falls back to standard (uncorrected) test.

        Returns
        -------
        ScanResult with corrected test statistics when dosage_var is provided.
        """
        # For non-score tests, delegate to standard LMM
        if test != "score":
            return self._lmm.score_chunk(G_chunk, null_fit, variant_meta, test)

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        # Rotate genotypes
        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)

        # Allele frequencies
        af = G_chunk.mean(dim=0) / 2.0

        # Standard score test quantities
        Y_rot = null_fit.Y_rot.squeeze()
        X0_rot = null_fit.X0_rot
        sig2_e = null_fit.sig2_e
        sig2_g = null_fit.sig2_g
        evals = null_fit.eigenvalues

        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (evals * lam + 1.0)
        Py, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv)

        # g^T P y
        gPy = (G_rot * Py.unsqueeze(1)).sum(dim=0)  # (m,)

        # g^T P g
        wG = H_inv.unsqueeze(1) * G_rot
        XtWG = X0_rot.T @ wG
        M00_inv_XtWG = torch.linalg.solve(null_fit.M00, XtWG)
        PG = wG - (H_inv.unsqueeze(1) * X0_rot) @ M00_inv_XtWG
        gPg = (G_rot * PG).sum(dim=0)  # (m,)

        # Apply uncertainty correction if dosage_var is provided
        if dosage_var is not None and hasattr(null_fit, "diag_P"):
            dosage_var = dosage_var.to(STAT_DTYPE)
            diag_P = null_fit.diag_P  # (n,)
            # Correction: Var_g^T @ diag(P) for each variant
            correction = (dosage_var * diag_P.unsqueeze(1)).sum(dim=0)  # (m,)
            gPg_corrected = gPg + correction
        else:
            correction = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)
            gPg_corrected = gPg

        gPg_safe = torch.clamp(gPg_corrected, min=1e-20)

        # Score statistic
        stat = gPy ** 2 / (sig2_e * gPg_safe)

        # P-values
        c = X0_rot.shape[1]
        df2 = n - c - 1
        p = _f_sf(stat, df1=1, df2=df2)

        nan_m = torch.full((m,), float("nan"), dtype=STAT_DTYPE)

        return ScanResult(
            chr=variant_meta.chr, pos=variant_meta.pos, snp=variant_meta.snp,
            a1=variant_meta.a1, a2=variant_meta.a2, af=af,
            beta=nan_m, se=nan_m, stat=stat, p=p, test="score",
        )

    def run(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor,
        gp_probs: Tensor,
        variant_meta: VariantMeta,
        ploidy: int = 2,
    ) -> GUResult:
        """Convenience method for full GP-aware scan.

        Parameters
        ----------
        Y : (n,) or (n,1) — phenotype
        X0 : (n, c) — covariates
        K : (n, n) — GRM
        gp_probs : (n, m, k+1) — genotype probabilities
        variant_meta : VariantMeta
        ploidy : int (default 2)

        Returns
        -------
        GUResult with scan results, correction diagnostics, and dosage R².
        """
        from ..preprocess.dosage_uncertainty import (
            dosage_rsq,
            dosage_variance,
            expected_dosage,
        )

        # Compute expected dosage and variance from GP
        G = expected_dosage(gp_probs, ploidy)       # (n, m)
        dvar = dosage_variance(gp_probs, ploidy)     # (n, m)
        rsq = dosage_rsq(gp_probs, ploidy)           # (m,)

        device = K.device
        G = G.to(device)
        dvar = dvar.to(device)

        # Fit null
        null_fit = self.fit_null(Y, X0, K=K)

        # Scan with correction
        result = self.score_chunk(G, null_fit, variant_meta,
                                  test="score", dosage_var=dvar)

        # Compute correction fraction: correction / (gPg + correction)
        # Re-derive for diagnostics
        diag_P = null_fit.diag_P
        correction = (dvar * diag_P.unsqueeze(1)).sum(dim=0)

        U = null_fit.eigenvectors
        G_rot = rotate(G, U)
        lam = null_fit.sig2_g / max(null_fit.sig2_e, 1e-20)
        H_inv = 1.0 / (null_fit.eigenvalues * lam + 1.0)
        wG = H_inv.unsqueeze(1) * G_rot
        gPg_raw = (G_rot * wG).sum(dim=0)
        mean_correction = correction / (gPg_raw + correction).clamp(min=1e-20)

        return GUResult(
            scan=result,
            mean_correction=mean_correction,
            dosage_rsq=rsq.to(device),
        )
