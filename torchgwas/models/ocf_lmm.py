"""Orthogonal Cross-Fit LMM (OCF-LMM).

Implements Double/Debiased ML (Chernozhukov et al. 2018) for mixed-model GWAS.
Cross-fits polygenic nuisance on K-1 folds, tests on held-out fold.
The scan is simple OLS on cross-fit residuals — no per-SNP eigendecomposition.

Model:
    y = alpha_j * g_j + f(W) + u + epsilon,  u ~ N(0, sig2_g * K)

Cross-fitting ensures the nuisance estimation error (from f(W) + u) is
independent of the test statistic for alpha_j, yielding valid chi-squared(1)
null even when the nuisance model converges slowly.

References:
    - Chernozhukov et al. (2018, Econometrica): Double/Debiased ML
    - Mbatchou et al. (2021, Nat Genet): REGENIE (approximate cross-fitting)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import scipy.stats as sp_stats
import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import eigendecompose, rotate, compute_weights
from ..optim.controller import OptimizerController
from ..optim.reml_math import _compute_P_quantities
from .base import BaseModel, NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Per-fold nuisance estimation output."""

    fold_id: int
    test_indices: Tensor        # (n_test,) int64 indices into original sample
    sig2_g: float
    sig2_e: float
    beta_hat: Tensor            # (c,) fixed effect estimates from train
    y_tilde_test: Tensor        # (n_test,) cross-fit residuals on held-out
    log_likelihood: float
    converged: bool


@dataclass
class OCFNullFit:
    """Cross-fit null model: stores per-fold variance components and
    assembled cross-fit residuals.

    Not a NullFit subclass, but provides property forwarding for
    compatibility with UnifiedScanner.
    """

    # Assembled cross-fit residuals
    y_tilde: Tensor             # (n,) cross-fit outcome residuals
    fold_ids: Tensor            # (n,) integer fold assignment per sample
    X0: Tensor                  # (n, c) covariates for genotype projection

    # Per-fold diagnostics
    fold_results: list[FoldResult]
    n_folds: int

    # Aggregated variance components (mean across folds)
    mean_sig2_g: float
    mean_sig2_e: float
    mean_h2: float

    # Convergence: all folds converged
    converged: bool
    device: Optional[torch.device] = None

    # DML scan options (stored at fit time, used at scan time)
    variance_type: str = "HC"
    project_genotype: bool = True

    @property
    def sig2_g(self) -> float:
        return self.mean_sig2_g

    @property
    def sig2_e(self) -> float:
        return self.mean_sig2_e

    @property
    def log_likelihood(self) -> Optional[float]:
        lls = [fr.log_likelihood for fr in self.fold_results]
        return sum(lls) / len(lls) if lls else None


# ---------------------------------------------------------------------------
# Fold creation
# ---------------------------------------------------------------------------

def _create_folds(
    n: int,
    n_folds: int,
    seed: Optional[int] = None,
) -> list[Tensor]:
    """Create K random fold assignments.

    Returns list of K tensors, each containing sample indices for that fold.
    Folds are approximately equal size (differ by at most 1).
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
    else:
        perm = torch.randperm(n)

    return list(torch.tensor_split(perm, n_folds))


# ---------------------------------------------------------------------------
# Per-fold REML + BLUP
# ---------------------------------------------------------------------------

def _fit_fold(
    Y: Tensor,
    X0: Tensor,
    K: Tensor,
    test_indices: Tensor,
    train_indices: Tensor,
    config: NumericalConfig,
    sig2_g_init: Optional[float] = None,
    sig2_e_init: Optional[float] = None,
    fold_id: int = 0,
) -> FoldResult:
    """Fit nuisance on train set, BLUP predict on test set, compute residual.

    Steps:
    1. Extract K_train, eigendecompose
    2. Rotate train data, REML via OptimizerController (warm-start)
    3. BLUP for test: u_hat = sig2_g * K_{test,train} @ V_train^{-1} @ r_train
    4. y_tilde_test = Y_test - X0_test @ beta_hat - u_hat_test
    """
    device = Y.device
    n_train = len(train_indices)
    n_test = len(test_indices)

    # Extract train subsets
    Y_train = Y[train_indices]
    X0_train = X0[train_indices]
    K_train = K[train_indices][:, train_indices]

    # Eigendecompose K_train
    ed = eigendecompose(K_train, eigenvalue_floor=config.eigenvalue_floor)
    Y_train_rot = rotate(Y_train, ed.eigenvectors)
    X0_train_rot = rotate(X0_train, ed.eigenvectors)

    # REML on train set
    controller = OptimizerController(config)
    nf_train = controller.fit(
        Y_train_rot, X0_train_rot, ed.eigenvalues,
        n_traits=1,
        max_iter=config.reml_max_iter,
        sig2_g_init=sig2_g_init,
        sig2_e_init=sig2_e_init,
    )

    sig2_g = nf_train.sig2_g
    sig2_e = nf_train.sig2_e
    # NullFit.b0 = X_rot^T W Y_rot (RHS of normal equations), not beta_hat.
    # Solve: beta_hat = (X^T W X)^{-1} X^T W Y = M00^{-1} b0
    beta_hat = torch.linalg.solve(nf_train.M00, nf_train.b0)  # (c,)

    # BLUP for test samples:
    # V_train_inv @ r_train, where V_train = sig2_g*K_train + sig2_e*I
    # In eigenbasis: V_inv_diag = 1 / (evals * sig2_g + sig2_e)
    r_train = Y_train - X0_train @ beta_hat  # (n_train,)

    V_inv_diag = 1.0 / (ed.eigenvalues * sig2_g + sig2_e)  # (n_train,)
    # V_train_inv @ r_train = U @ diag(V_inv_diag) @ U^T @ r_train
    r_train_rot = ed.eigenvectors.T @ r_train  # (n_train,)
    V_inv_r = ed.eigenvectors @ (V_inv_diag * r_train_rot)  # (n_train,)

    # Cross-kernel: K_{test, train}
    K_test_train = K[test_indices][:, train_indices]  # (n_test, n_train)
    u_hat_test = sig2_g * (K_test_train @ V_inv_r)  # (n_test,)

    # Cross-fit residual
    Y_test = Y[test_indices]
    X0_test = X0[test_indices]
    y_tilde_test = Y_test - X0_test @ beta_hat - u_hat_test

    return FoldResult(
        fold_id=fold_id,
        test_indices=test_indices,
        sig2_g=sig2_g,
        sig2_e=sig2_e,
        beta_hat=beta_hat,
        y_tilde_test=y_tilde_test,
        log_likelihood=nf_train.log_likelihood,
        converged=nf_train.converged,
    )


# ---------------------------------------------------------------------------
# Assemble cross-fit residuals
# ---------------------------------------------------------------------------

def _assemble_residuals(
    fold_results: list[FoldResult],
    n: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Assemble full-sample y_tilde and fold_ids from per-fold results.

    Returns
    -------
    y_tilde : (n,) cross-fit residuals in original sample order
    fold_ids : (n,) integer fold assignment for each sample
    """
    y_tilde = torch.zeros(n, dtype=STAT_DTYPE, device=device)
    fold_ids = torch.zeros(n, dtype=torch.long, device=device)

    for fr in fold_results:
        y_tilde[fr.test_indices] = fr.y_tilde_test
        fold_ids[fr.test_indices] = fr.fold_id

    return y_tilde, fold_ids


# ---------------------------------------------------------------------------
# DML scan algebra
# ---------------------------------------------------------------------------

def _dml_score_batch(
    G_chunk: Tensor,
    y_tilde: Tensor,
    X0: Tensor,
    variance_type: str = "HC",
    project_genotype: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """DML scan: OLS of y_tilde on (optionally projected) genotypes.

    Parameters
    ----------
    G_chunk : (n, m)
    y_tilde : (n,)
    X0 : (n, c) covariates for genotype projection
    variance_type : "HC" (sandwich) or "homoskedastic"
    project_genotype : whether to project genotypes onto covariate space

    Returns
    -------
    beta : (m,)
    se : (m,)
    stat : (m,) chi2(1)
    p : (m,)
    """
    n, m = G_chunk.shape
    device = G_chunk.device

    # Genotype nuisance: project out covariates from genotypes
    if project_genotype and X0.shape[1] > 0:
        XtX = X0.T @ X0  # (c, c)
        XtG = X0.T @ G_chunk  # (c, m)
        proj = torch.linalg.solve(XtX, XtG)  # (c, m)
        g_tilde = G_chunk - X0 @ proj  # (n, m)
    else:
        g_tilde = G_chunk

    # OLS: alpha_hat = (g_tilde^T y_tilde) / (g_tilde^T g_tilde)
    gTg = (g_tilde ** 2).sum(dim=0)  # (m,)
    gTy = (g_tilde * y_tilde.unsqueeze(1)).sum(dim=0)  # (m,)

    gTg_safe = gTg.clamp(min=1e-20)
    beta = gTy / gTg_safe  # (m,)

    # Residuals
    eps = y_tilde.unsqueeze(1) - beta.unsqueeze(0) * g_tilde  # (n, m)

    # Variance estimation
    if variance_type == "HC":
        # HC (sandwich) estimator: robust to heteroskedasticity
        # V = sum(g_tilde^2 * eps^2) / (sum(g_tilde^2))^2
        numerator = (g_tilde ** 2 * eps ** 2).sum(dim=0)  # (m,)
        var_beta = numerator / (gTg_safe ** 2)
    else:
        # Homoskedastic: V = sum(eps^2) / (n-1) / sum(g_tilde^2)
        sig2_eps = (eps ** 2).sum(dim=0) / max(n - 1, 1)  # (m,)
        var_beta = sig2_eps / gTg_safe

    var_safe = var_beta.clamp(min=1e-30)
    se = torch.sqrt(var_safe)  # (m,)

    stat = (beta ** 2) / var_safe  # (m,) chi2(1)
    stat = stat.clamp(min=0.0)

    # P-values
    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=1)
    p_np = np.clip(p_np, 1e-300, 1.0)
    p = torch.tensor(p_np, dtype=STAT_DTYPE, device=device)

    return beta, se, stat, p


# ---------------------------------------------------------------------------
# OCFLMM model class
# ---------------------------------------------------------------------------

class OCFLMM:
    """Orthogonal Cross-Fit LMM.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    config : NumericalConfig, optional
    n_folds : int
        Number of cross-fitting folds (default 5).
    seed : int, optional
        Random seed for fold assignment (for reproducibility).
    variance_type : str
        ``"HC"`` (default) — heteroskedasticity-consistent sandwich estimator.
        ``"homoskedastic"`` — assumes constant residual variance.
    project_genotype : bool
        If True (default), project genotypes onto covariate space before testing.
        This is the genotype nuisance l(W) = E[g|W] approximation.
    """

    def __init__(
        self,
        config: Optional[NumericalConfig] = None,
        n_folds: int = 5,
        seed: Optional[int] = None,
        variance_type: str = "HC",
        project_genotype: bool = True,
    ) -> None:
        self.config = config or NumericalConfig()
        self.n_folds = n_folds
        self.seed = seed
        self.variance_type = variance_type
        self.project_genotype = project_genotype

    # ------------------------------------------------------------------
    # fit_null
    # ------------------------------------------------------------------

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> OCFNullFit:
        """Cross-fit null model: K-fold nuisance estimation.

        For each fold k:
        1. Eigendecompose K_train
        2. REML on rotated train data → (sig2_g, sig2_e, beta_hat)
        3. BLUP predict on held-out fold
        4. y_tilde_test = Y_test - X_test @ beta - u_hat_test

        Parameters
        ----------
        Y : (n,) or (n, 1) phenotype vector.
        X0 : (n, c) covariates (intercept as first column).
        K : (n, n) GRM / kinship matrix (required).
        """
        if K is None:
            raise ValueError("OCFLMM requires a kinship matrix K.")

        Y = Y.to(STAT_DTYPE).squeeze()
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        n = Y.shape[0]
        device = Y.device

        if Y.ndim != 1:
            raise ValueError(f"Y must be 1D (single trait), got shape {Y.shape}")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.n_folds > n:
            raise ValueError(f"n_folds ({self.n_folds}) > n ({n})")

        # --- Quick full-sample REML for warm-start ---
        logger.info("OCF-LMM: full-sample REML for warm-start...")
        ed_full = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)
        Y_rot_full = rotate(Y, ed_full.eigenvectors)
        X0_rot_full = rotate(X0, ed_full.eigenvectors)

        controller_full = OptimizerController(self.config)
        nf_full = controller_full.fit(
            Y_rot_full, X0_rot_full, ed_full.eigenvalues,
            n_traits=1,
            max_iter=self.config.reml_max_iter,
        )
        sig2_g_init = nf_full.sig2_g
        sig2_e_init = nf_full.sig2_e
        logger.info(
            "OCF-LMM warm-start: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
            sig2_g_init, sig2_e_init,
            sig2_g_init / max(sig2_g_init + sig2_e_init, 1e-20),
        )

        # --- Create folds ---
        fold_index_lists = _create_folds(n, self.n_folds, self.seed)

        # --- Per-fold cross-fitting ---
        fold_results: list[FoldResult] = []
        all_indices = torch.arange(n, device=device)

        for k in range(self.n_folds):
            test_idx = fold_index_lists[k].to(device)
            # Train = all samples NOT in fold k
            train_mask = torch.ones(n, dtype=torch.bool, device=device)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]

            logger.info(
                "OCF-LMM fold %d/%d: train=%d, test=%d",
                k + 1, self.n_folds, len(train_idx), len(test_idx),
            )

            fr = _fit_fold(
                Y, X0, K,
                test_indices=test_idx,
                train_indices=train_idx,
                config=self.config,
                sig2_g_init=sig2_g_init,
                sig2_e_init=sig2_e_init,
                fold_id=k,
            )
            fold_results.append(fr)

            logger.info(
                "  Fold %d: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f, ll=%.4f, conv=%s",
                k + 1, fr.sig2_g, fr.sig2_e,
                fr.sig2_g / max(fr.sig2_g + fr.sig2_e, 1e-20),
                fr.log_likelihood, fr.converged,
            )

        # --- Assemble cross-fit residuals ---
        y_tilde, fold_ids = _assemble_residuals(fold_results, n, device)

        # Aggregated diagnostics
        mean_sig2_g = sum(fr.sig2_g for fr in fold_results) / self.n_folds
        mean_sig2_e = sum(fr.sig2_e for fr in fold_results) / self.n_folds
        mean_h2 = mean_sig2_g / max(mean_sig2_g + mean_sig2_e, 1e-20)
        all_converged = all(fr.converged for fr in fold_results)

        logger.info(
            "OCF-LMM null fit: n_folds=%d, mean_h2=%.4f, all_converged=%s, "
            "mean(y_tilde)=%.4e",
            self.n_folds, mean_h2, all_converged,
            y_tilde.mean().item(),
        )

        return OCFNullFit(
            y_tilde=y_tilde,
            fold_ids=fold_ids,
            X0=X0,
            fold_results=fold_results,
            n_folds=self.n_folds,
            mean_sig2_g=mean_sig2_g,
            mean_sig2_e=mean_sig2_e,
            mean_h2=mean_h2,
            converged=all_converged,
            device=device,
            variance_type=self.variance_type,
            project_genotype=self.project_genotype,
        )

    # ------------------------------------------------------------------
    # score_chunk
    # ------------------------------------------------------------------

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: OCFNullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """DML scan: OLS of cross-fit residuals on genotypes.

        Returns ScanResult with inference_type="cross_fit".
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        # Handle empty chunk
        if m == 0:
            empty = torch.zeros(0, dtype=STAT_DTYPE, device=G_chunk.device)
            return ScanResult(
                chr=[], pos=[], snp=[], a1=[], a2=[],
                af=empty, beta=empty, se=empty,
                stat=empty, p=empty,
                test="wald",
                inference_type="cross_fit",
            )

        # Allele frequency
        af = G_chunk.mean(dim=0) / 2.0

        # DML score
        beta, se, stat, p = _dml_score_batch(
            G_chunk,
            null_fit.y_tilde,
            null_fit.X0,
            variance_type=null_fit.variance_type,
            project_genotype=null_fit.project_genotype,
        )

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,
            se=se,
            stat=stat,
            p=p,
            test="wald",
            inference_type="cross_fit",
        )
