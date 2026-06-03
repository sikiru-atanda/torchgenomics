"""Multi-QTL joint model fitting with backward elimination (charter Section 9g).

Fits a joint LMM with all QTL simultaneously as fixed effects, reports
per-QTL R², LRT p-value, and effect size. Backward elimination iteratively
drops non-significant QTL. Equivalent to GWASpoly's ``fit.QTL``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import NumericalConfig
from .single_trait_lmm import SingleTraitLMM

logger = logging.getLogger(__name__)


@dataclass
class JointQTLResult:
    """Results from joint multi-QTL model."""

    qtl_snps: list[str]
    qtl_indices: list[int]
    beta: Tensor  # (q,) effect sizes for each QTL
    se: Tensor  # (q,) standard errors
    r_squared: Tensor  # (q,) per-QTL proportion of variance explained
    lrt_p: Tensor  # (q,) LRT p-value (drop-one test)
    total_r_squared: float
    n_eliminated: int  # number of QTL dropped by backward elimination


def fit_joint_qtl(
    Y: Tensor,
    X0: Tensor,
    G: Tensor,
    K: Tensor,
    qtl_indices: list[int],
    qtl_snps: list[str] | None = None,
    backward_elimination: bool = True,
    elimination_threshold: float = 0.05,
    config: NumericalConfig | None = None,
) -> JointQTLResult:
    """Fit LMM with multiple QTL as fixed effects.

    Uses SingleTraitLMM with expanded X = [X0, G[:, qtl_indices]].

    For each QTL, computes:
    - Beta and SE from the full model Wald test
    - R² = (SS_reduced - SS_full) / SS_reduced for each QTL
    - LRT p-value by comparing full vs drop-one model

    Parameters
    ----------
    Y : Tensor, shape (n,)
        Phenotype vector.
    X0 : Tensor, shape (n, c)
        Covariate matrix (intercept + covariates).
    G : Tensor, shape (n, m)
        Full genotype matrix.
    K : Tensor, shape (n, n)
        Kinship matrix.
    qtl_indices : list[int]
        Column indices in G for the QTL to fit jointly.
    qtl_snps : list[str] | None
        SNP names for the QTL. If None, uses index strings.
    backward_elimination : bool
        If True, iteratively drop QTL with LRT p > elimination_threshold.
    elimination_threshold : float
        P-value threshold for backward elimination.
    config : NumericalConfig | None
        REML configuration.

    Returns
    -------
    JointQTLResult
    """
    from scipy.stats import chi2

    if config is None:
        config = NumericalConfig(reml_method="emma")

    if qtl_snps is None:
        qtl_snps = [str(i) for i in qtl_indices]

    current_indices = list(qtl_indices)
    current_snps = list(qtl_snps)
    n_eliminated = 0

    while True:
        q = len(current_indices)
        if q == 0:
            return JointQTLResult(
                qtl_snps=[], qtl_indices=[], beta=torch.tensor([]),
                se=torch.tensor([]), r_squared=torch.tensor([]),
                lrt_p=torch.tensor([]), total_r_squared=0.0,
                n_eliminated=n_eliminated,
            )

        # Build expanded covariate matrix: [X0, G_qtl]
        G_qtl = G[:, current_indices]  # (n, q)
        X_full = torch.cat([X0, G_qtl], dim=1)  # (n, c+q)

        # Fit full model
        lmm = SingleTraitLMM(config=config)
        nf_full = lmm.fit_null(Y, X_full, K=K)

        # Fit base model (no QTL)
        lmm_base = SingleTraitLMM(config=config)
        nf_base = lmm_base.fit_null(Y, X0, K=K)

        # Total R²: 1 - SS_full / SS_base
        ss_full = nf_full.sig2_e * Y.shape[0] if nf_full.sig2_e else 1.0
        ss_base = nf_base.sig2_e * Y.shape[0] if nf_base.sig2_e else 1.0
        total_r2 = max(0.0, 1.0 - ss_full / ss_base) if ss_base > 0 else 0.0

        # Per-QTL: drop-one LRT
        c = X0.shape[1]
        betas = torch.zeros(q, dtype=torch.float64)
        ses = torch.zeros(q, dtype=torch.float64)
        r_squareds = torch.zeros(q, dtype=torch.float64)
        lrt_ps = torch.zeros(q, dtype=torch.float64)

        ll_full = nf_full.log_likelihood if nf_full.log_likelihood else 0.0

        for j in range(q):
            # Extract beta and SE for QTL j from full model
            # QTL j is at column c + j in X_full
            if nf_full.b0 is not None and len(nf_full.b0) > c + j:
                betas[j] = nf_full.b0[c + j]
                # SE from M00 inverse diagonal
                if nf_full.M00 is not None:
                    try:
                        M00_inv = torch.linalg.inv(nf_full.M00)
                        var_b = M00_inv[c + j, c + j] * nf_full.sig2_e
                        ses[j] = torch.sqrt(torch.clamp(var_b, min=1e-20))
                    except Exception:
                        ses[j] = float("nan")

            # Drop-one model
            keep = [k for k in range(q) if k != j]
            if keep:
                X_drop = torch.cat([X0, G_qtl[:, keep]], dim=1)
            else:
                X_drop = X0

            lmm_drop = SingleTraitLMM(config=config)
            nf_drop = lmm_drop.fit_null(Y, X_drop, K=K)

            ll_drop = nf_drop.log_likelihood if nf_drop.log_likelihood else 0.0

            # LRT statistic
            lrt_stat = max(0.0, 2.0 * (ll_full - ll_drop))
            lrt_ps[j] = float(chi2.sf(lrt_stat, df=1))

            # Per-QTL R²
            ss_drop = nf_drop.sig2_e * Y.shape[0] if nf_drop.sig2_e else 1.0
            r_squareds[j] = max(0.0, 1.0 - ss_full / ss_drop) if ss_drop > 0 else 0.0

        # Backward elimination
        if backward_elimination and q > 1:
            worst_idx = lrt_ps.argmax().item()
            if lrt_ps[worst_idx] > elimination_threshold:
                logger.info(
                    "Backward elimination: dropping %s (LRT p=%.4g)",
                    current_snps[worst_idx], lrt_ps[worst_idx].item(),
                )
                del current_indices[worst_idx]
                del current_snps[worst_idx]
                n_eliminated += 1
                continue  # re-fit without this QTL

        # No more elimination needed
        return JointQTLResult(
            qtl_snps=current_snps,
            qtl_indices=current_indices,
            beta=betas,
            se=ses,
            r_squared=r_squareds,
            lrt_p=lrt_ps,
            total_r_squared=total_r2,
            n_eliminated=n_eliminated,
        )
