"""Sparse LMM: score-test-only model for sparse GRM path.

Uses PCG-based REML (no eigendecomposition) and score tests for per-SNP
association. This is the model used when ``--approx-method sparse`` is
specified, following the fastGWA approach.

Only the score test is supported because Wald/LRT would require a PCG solve
per SNP (prohibitively expensive at biobank scale). The score test needs only
P @ y (precomputed) and g^T P g (cheap via sparse operations).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.sparse_grm import make_sparse_matvec
from ..optim.pcg_solver import diagonal_preconditioner, pcg_solve
from ..optim.sparse_reml import sparse_reml_fit
from .base import NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class SparseLMM:
    """Score-test-only LMM for sparse GRM.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.

    The sparse path stores P @ y in ``NullFit.Y_rot`` and the sparse
    GRM + variance components for on-the-fly P @ g computation.
    """

    def __init__(self, config: NumericalConfig | None = None) -> None:
        self.config = config or NumericalConfig()
        self._K_sparse: Tensor | None = None
        self._K_matvec: Callable | None = None

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model via PCG-based REML with sparse GRM.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates
        K : Tensor (sparse COO) — sparse GRM

        Additional kwargs
        -----------------
        K_diag : Tensor (n,) — diagonal of K for preconditioner
        n_probes : int — stochastic probe count
        lanczos_iters : int — Lanczos iterations for logdet

        Returns
        -------
        NullFit with PCG-solved quantities for score test.
        """
        if K is None:
            raise ValueError("SparseLMM requires a sparse kinship matrix K.")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)

        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n = Y.shape[0]

        # Store sparse K for score_chunk
        self._K_sparse = K

        # Extract diagonal for preconditioner
        K_diag = kwargs.get("K_diag")
        if K_diag is None:
            # Extract diagonal from sparse tensor
            K_dense_diag = torch.sparse.sum(
                K * torch.eye(n, dtype=K.dtype, device=K.device).to_sparse_coo(),
                dim=1,
            ).to_dense()
            K_diag = K_dense_diag
        K_diag = K_diag.to(STAT_DTYPE)

        # Sparse matvec for K
        def _K_mv(x: Tensor) -> Tensor:
            if x.ndim == 1:
                return torch.sparse.mm(K, x.unsqueeze(1)).squeeze(1)
            return torch.sparse.mm(K, x)

        self._K_matvec = _K_mv

        # Fit via PCG-REML
        null_fit = sparse_reml_fit(
            Y, X0, _K_mv, K_diag, n,
            config=self.config,
            n_probes=kwargs.get("n_probes", self.config.approx_stochastic_probes),
            lanczos_iters=kwargs.get("lanczos_iters", self.config.approx_lanczos_iters),
            seed=kwargs.get("seed"),
        )

        # Store K_sparse reference and V_matvec for score_chunk
        null_fit.eigenvectors = None  # No eigenvectors in sparse path

        logger.info(
            "SparseLMM null fit: sig2_g=%.6e, sig2_e=%.6e, h2=%.4f",
            null_fit.sig2_g, null_fit.sig2_e,
            null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e)
            if (null_fit.sig2_g + null_fit.sig2_e) > 0 else 0.0,
        )

        return null_fit

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score test for a genotype chunk against sparse null.

        T_score = (g^T P y)^2 / (sig2_e * g^T P g)

        where P y is precomputed (stored in null_fit.Y_rot) and
        P g is computed via sparse operations.

        Parameters
        ----------
        G_chunk : (n, m) — genotype dosage
        null_fit : output of fit_null
        variant_meta : marker metadata
        test : must be "score" (only supported test for sparse path)
        """
        if test != "score":
            raise ValueError(
                f"SparseLMM only supports score test, got '{test}'. "
                "Wald and LRT require eigendecomposition — use exact or "
                "randomized_svd/nystrom approx_method instead."
            )

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        sig2_g = null_fit.sig2_g
        sig2_e = null_fit.sig2_e

        # P y was stored in Y_rot by sparse_reml_fit
        Py = null_fit.Y_rot  # (n,)

        # Allele frequencies
        af = G_chunk.mean(dim=0) / 2.0

        # g^T P y for each SNP
        gPy = G_chunk.T @ Py  # (m,)

        # g^T P g: P g = V^{-1} g - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} g
        # For efficiency, compute V^{-1} g via PCG for each SNP in the chunk
        # Then project out covariates
        Vinv_X = null_fit.X0_rot  # V^{-1} X stored by sparse_reml_fit
        X0 = Vinv_X  # This is actually V^{-1} X, we need original X0 too

        # Reconstruct V matvec
        lam = sig2_g / max(sig2_e, 1e-20)
        V_matvec = make_sparse_matvec(self._K_sparse, sig2_g, sig2_e)
        V_diag = sig2_g * torch.sparse.sum(
            self._K_sparse * torch.eye(n, dtype=STAT_DTYPE, device=G_chunk.device).to_sparse_coo(),
            dim=1,
        ).to_dense() + sig2_e
        precond = diagonal_preconditioner(V_diag)

        # Batch PCG for all SNPs in chunk
        gPg = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)

        # For efficiency, compute V^{-1} G as a batch
        # Then gPg = diag(G^T P G) where P = V^{-1} - V^{-1}X(X^TVX)^{-1}X^TV^{-1}
        Vinv_G = torch.zeros_like(G_chunk)
        for j in range(m):
            res = pcg_solve(V_matvec, G_chunk[:, j], precond=precond,
                            tol=1e-6, max_iter=200)
            Vinv_G[:, j] = res.x

        # X^T V^{-1} X (recompute from stored Vinv_X)
        # null_fit.X0_rot = V^{-1} X, but we need original X0
        # Actually sparse_reml stores Vinv_X in X0_rot
        # We need X0 to compute X^T V^{-1} G
        # Reconstruct: if b0 = (X^T V^{-1} X)^{-1} X^T V^{-1} y, and we have Vinv_X
        # X^T V^{-1} G = (V^{-1} X)^T G ... but V^{-1} X is not X^T V^{-1}
        # Actually (V^{-1})^T = V^{-1} since V is symmetric, so:
        # X^T V^{-1} G = (Vinv_X)^T @ G ... no, Vinv_X = V^{-1} X which is (n, c)
        # X^T V^{-1} G needs X^T @ Vinv_G = X's original @ Vinv_G
        # We don't have original X0 stored directly in NullFit...
        # But we can recover: V @ Vinv_X = X0, so X0 = V_matvec(Vinv_X)

        # Recover original X0
        Vinv_X_stored = null_fit.X0_rot  # (n, c)
        c = Vinv_X_stored.shape[1]
        X0_orig = torch.zeros_like(Vinv_X_stored)
        for j in range(c):
            X0_orig[:, j] = V_matvec(Vinv_X_stored[:, j])

        XtVinvX = X0_orig.T @ Vinv_X_stored  # (c, c)
        XtVinvG = X0_orig.T @ Vinv_G  # (c, m)

        try:
            M_inv_XtVinvG = torch.linalg.solve(XtVinvX, XtVinvG)  # (c, m)
        except torch.linalg.LinAlgError:
            M_inv_XtVinvG = torch.zeros_like(XtVinvG)

        # P G = V^{-1} G - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} G
        PG = Vinv_G - Vinv_X_stored @ M_inv_XtVinvG  # (n, m)

        gPg = (G_chunk * PG).sum(dim=0)  # (m,)
        gPg = torch.clamp(gPg, min=1e-20)

        # Score statistic
        stat = gPy ** 2 / (sig2_e * gPg)

        # P-values from chi2(1) (score test uses chi-squared, not F)
        import numpy as np
        import scipy.stats as sp_stats
        stat_np = stat.detach().cpu().numpy().astype(np.float64)
        p_np = sp_stats.chi2.sf(stat_np, df=1)
        p_np = np.clip(p_np, 1e-300, 1.0)
        p = torch.tensor(p_np, dtype=STAT_DTYPE, device=G_chunk.device)

        nan_m = torch.full((m,), float("nan"), dtype=STAT_DTYPE)

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=nan_m,
            se=nan_m,
            stat=stat,
            p=p,
            test="score",
        )
