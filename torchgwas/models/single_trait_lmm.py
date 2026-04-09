"""Single-trait Linear Mixed Model: y = Xb + u + e.

Implements the eigendecomposition trick from GEMMA (Zhou & Stephens, 2012):
1. Eigendecompose K once: K = U diag(evals) U^T
2. Rotate all data: y_rot = U^T y, X_rot = U^T X, g_rot = U^T g
3. In rotated space, covariance is diagonal → per-SNP cost is O(n)

Supports Wald, LRT, and Score tests.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import EigenDecomp, eigendecompose, rotate
from ..optim.controller import OptimizerController
from ..optim.emma_reml import gapit_emma_remle
from ..optim.reml_math import _compute_P_quantities, reml_loglikelihood
from .base import BaseModel, NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class SingleTraitLMM:
    """Single-trait LMM with eigendecomposition trick.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(
        self,
        config: Optional[NumericalConfig] = None,
        p3d: bool = True,
    ) -> None:
        self.config = config or NumericalConfig()
        self.p3d = p3d

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null model (no SNP effect).

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates (intercept as first column)
        K : (n, n) — GRM / kinship matrix

        Keyword Arguments
        -----------------
        approx_method : str, optional
            Approximate eigendecomposition method. One of:
            ``"randomized_svd"``, ``"nystrom"``, ``"lobpcg"``, or None (exact).
            The ``"sparse"`` path uses SparseLMM instead.
        approx_config : dict, optional
            Parameters for the approximate method (n_components, etc.).

        Returns
        -------
        NullFit with cached rotated quantities for per-SNP scan.
        """
        approx_method = kwargs.pop("approx_method", None)
        approx_config = kwargs.pop("approx_config", None) or {}

        if K is None:
            raise ValueError("SingleTraitLMM requires a kinship matrix K.")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        n = Y.shape[0]
        if Y.ndim == 2:
            if Y.shape[1] != 1:
                raise ValueError(
                    f"SingleTraitLMM expects 1 trait, got {Y.shape[1]}. "
                    "Use MultiTraitLMM for d > 1."
                )
            Y = Y.squeeze(1)

        # --- Step 1: Eigendecompose K (exact or approximate) ---
        if approx_method is None:
            ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)
        elif approx_method == "randomized_svd":
            from ..linalg.randomized import randomized_svd
            ed = randomized_svd(
                K,
                n_components=approx_config.get("n_components", self.config.approx_n_components),
                n_oversamples=approx_config.get("n_oversamples", self.config.approx_n_oversamples),
                n_power_iters=approx_config.get("n_power_iters", self.config.approx_n_power_iters),
                seed=approx_config.get("seed", None),
            )
            logger.info("Using randomized SVD with k=%d components.", ed.eigenvalues.shape[0])
        elif approx_method == "lobpcg":
            from ..linalg.randomized import lobpcg_decompose
            ed = lobpcg_decompose(
                K,
                n_components=approx_config.get("n_components", self.config.approx_n_components),
                seed=approx_config.get("seed", None),
            )
            logger.info("Using LOBPCG with k=%d components.", ed.eigenvalues.shape[0])
        elif approx_method == "nystrom":
            from ..linalg.nystrom import nystrom_approximate
            chunk_iter = approx_config.get("chunk_iter", None)
            if chunk_iter is None:
                raise ValueError(
                    "Nystrom approx_method requires 'chunk_iter' in approx_config "
                    "(a genotype chunk iterator for streaming GRM construction)."
                )
            ed, _ = nystrom_approximate(
                chunk_iter,
                n_samples=n,
                n_landmarks=approx_config.get("n_landmarks", self.config.approx_n_landmarks),
                ploidy=approx_config.get("ploidy", 2),
                seed=approx_config.get("seed", None),
            )
            logger.info("Using Nystrom with %d landmarks, %d components.",
                       approx_config.get("n_landmarks", self.config.approx_n_landmarks),
                       ed.eigenvalues.shape[0])
        else:
            raise ValueError(
                f"Unknown approx_method '{approx_method}'. "
                "Use 'randomized_svd', 'lobpcg', 'nystrom', or None."
            )

        # --- Step 2: Rotate data ---
        Y_rot = rotate(Y, ed.eigenvectors)  # (k,) or (n,)
        X0_rot = rotate(X0, ed.eigenvectors)  # (k, c) or (n, c)

        # --- Step 3: Fit variance components ---
        if self.config.reml_method == "emma" and approx_method is None:
            # Use GAPIT-exact restricted eigendecomposition REML
            # This matches GAPIT's GAPIT.emma.REMLE + emma.eigen.R.wo.Z
            sig2_g, sig2_e, ll, delta, trace = gapit_emma_remle(Y, X0, K)

            # Build NullFit using the L-eigendecomposition (for scan)
            # GAPIT uses eig.L for the scan, eig.R for REML estimation
            from ..linalg.eigh import compute_weights
            lam = sig2_g / max(sig2_e, 1e-20)
            H_inv = 1.0 / (ed.eigenvalues * lam + 1.0)
            _, beta0 = _compute_P_quantities(Y_rot, X0_rot, H_inv)

            wX = H_inv.unsqueeze(1) * X0_rot
            M00 = X0_rot.T @ wX

            null_fit = NullFit(
                sig2_g=sig2_g,
                sig2_e=sig2_e,
                eigenvalues=ed.eigenvalues,
                Y_rot=Y_rot,
                X0_rot=X0_rot,
                M00=M00,
                b0=beta0,
                weights=compute_weights(ed.eigenvalues, sig2_g, sig2_e),
                log_likelihood=ll,
                optimizer_trace=trace,
                converged=True,
                device=Y.device,
            )
        else:
            # Use the optimizer stack (AI-REML with PX-EM warmstart)
            # Works for both exact and approximate eigenpairs
            controller = OptimizerController(config=self.config)
            null_fit = controller.fit(
                Y_rot, X0_rot, ed.eigenvalues,
                n_traits=1,
                max_iter=self.config.reml_max_iter,
            )

        # Store eigenvectors for rotating SNPs during scan
        null_fit.eigenvectors = ed.eigenvectors

        # Mark approximate status
        if approx_method is not None:
            null_fit.approximate = True
            null_fit.approx_method = approx_method
            null_fit.approx_config = approx_config

        logger.info(
            "Null fit: sig2_g=%.6e, sig2_e=%.6e, h2=%.4f, ll=%.4f, converged=%s%s",
            null_fit.sig2_g, null_fit.sig2_e,
            null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
            null_fit.log_likelihood, null_fit.converged,
            f" [APPROXIMATE: {approx_method}]" if approx_method else "",
        )

        return null_fit

    def update_null(self, null_fit: NullFit, *, max_iter: int = 100) -> NullFit:
        """Resume optimization from a previous NullFit (ASReml-R update style).

        Reuses cached eigendecomposition and rotated data, warm-starts the
        optimizer from the previous variance components, and runs up to
        ``max_iter`` additional iterations.

        Parameters
        ----------
        null_fit : NullFit
            Previous (possibly non-converged) null model fit.
        max_iter : int
            Maximum iterations for the resumed run.

        Returns
        -------
        NullFit with refined variance components and accumulated trace.
        """
        from .base import update_null
        return update_null(null_fit, max_iter=max_iter, config=self.config)

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score a genotype chunk against the cached null.

        Parameters
        ----------
        G_chunk : (n, m) — genotype dosage for m SNPs
        null_fit : output of fit_null
        variant_meta : marker metadata
        test : "wald", "lrt", or "score"

        Returns
        -------
        ScanResult with beta, SE, test statistic, and p-values.
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        m = G_chunk.shape[1]

        # P3D=FALSE: refit variance components per SNP
        if not self.p3d:
            return self._score_chunk_refit(G_chunk, null_fit, variant_meta, test)

        # Rotate genotypes into eigenspace
        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)

        # Allele frequencies for output
        af = G_chunk.mean(dim=0) / 2.0  # diploid convention

        if test == "wald":
            return self._wald_test(G_rot, null_fit, variant_meta, af)
        elif test == "lrt":
            return self._lrt_test(G_rot, null_fit, variant_meta, af)
        elif test == "score":
            return self._score_test(G_rot, null_fit, variant_meta, af)
        else:
            raise ValueError(f"Unknown test '{test}'. Use 'wald', 'lrt', or 'score'.")

    # ------------------------------------------------------------------
    # Wald test (vectorized over m SNPs)
    # ------------------------------------------------------------------

    def _wald_test(
        self,
        G_rot: Tensor,
        nf: NullFit,
        vmeta: VariantMeta,
        af: Tensor,
    ) -> ScanResult:
        """Wald test: beta_g^2 / Var(beta_g) ~ chi2(1).

        In rotated space with diagonal covariance, this is a simple
        weighted least squares regression for each SNP.

        Uses H_inv = 1/(evals*lambda + 1) consistently for all weighted
        quantities and the Schur complement.
        """
        n, m = G_rot.shape
        Y_rot = nf.Y_rot.squeeze()  # (n,)
        X0_rot = nf.X0_rot  # (n, c)
        sig2_e = nf.sig2_e

        # Use H_inv consistently (M00 was also built with H_inv)
        lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
        H_inv = 1.0 / (nf.eigenvalues * lam + 1.0)
        Py, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv)

        # Weighted quantities using H_inv (consistent with M00)
        wG = H_inv.unsqueeze(1) * G_rot  # (n, m)
        gWg = (G_rot * wG).sum(dim=0)  # (m,)
        gWX = G_rot.T @ (H_inv.unsqueeze(1) * X0_rot)  # (m, c)

        # M00 was computed with H_inv in the controller
        M00 = nf.M00  # (c, c)
        M00_inv_gWXt = torch.linalg.solve(M00, gWX.T)  # (c, m)

        # Schur complement: S = gWg - gWX @ M00^{-1} @ gWX^T
        S = gWg - (gWX * M00_inv_gWXt.T).sum(dim=1)  # (m,)

        # g^T P y
        gPy = (G_rot * Py.unsqueeze(1)).sum(dim=0)  # (m,)

        # Beta and variance (V = sig2_e * H, so Var(beta) = sig2_e / S)
        S_safe = torch.clamp(S, min=1e-20)
        beta = gPy / S_safe  # (m,)
        var_beta = sig2_e / S_safe  # (m,)
        se = torch.sqrt(var_beta)  # (m,)

        # Wald statistic = beta^2 / var_beta ~ F(1, n-c-1)
        stat = beta ** 2 / var_beta  # (m,)

        # P-values from F(1, n-c-1) distribution (matches GAPIT MLM)
        c = X0_rot.shape[1]
        df2 = n - c - 1
        p = _f_sf(stat, df1=1, df2=df2)

        return ScanResult(
            chr=vmeta.chr,
            pos=vmeta.pos,
            snp=vmeta.snp,
            a1=vmeta.a1,
            a2=vmeta.a2,
            af=af,
            beta=beta,
            se=se,
            stat=stat,
            p=p,
            test="wald",
        )

    # ------------------------------------------------------------------
    # LRT (fixed-VC fast scan)
    # ------------------------------------------------------------------

    def _lrt_test(
        self,
        G_rot: Tensor,
        nf: NullFit,
        vmeta: VariantMeta,
        af: Tensor,
    ) -> ScanResult:
        """Likelihood Ratio Test: 2(ll_full - ll_null) ~ chi2(1).

        Uses fixed variance components from the null model (fast scan).
        For each SNP, compute the full-model log-likelihood with the SNP
        added as a fixed effect, then compare to the null ll.
        """
        n, m = G_rot.shape
        Y_rot = nf.Y_rot.squeeze()
        X0_rot = nf.X0_rot
        sig2_g = nf.sig2_g
        sig2_e = nf.sig2_e
        evals = nf.eigenvalues
        ll_null = nf.log_likelihood
        c = X0_rot.shape[1]

        lam = sig2_g / max(sig2_e, 1e-20)
        H_diag = evals * lam + 1.0
        H_inv = 1.0 / H_diag

        # For each SNP, form X_full = [X0, g] and compute REML ll
        # With fixed VCs, the ll difference simplifies to:
        # 2 * (ll_full - ll_null) = (beta_g^2 * S) / sig2_e
        # which is identical to the Wald statistic when VCs are fixed.
        # So LRT with fixed VCs = Wald test. This is standard in GEMMA.

        # We compute it through the full formula for correctness:
        w = H_inv

        # Null: y^T P0 y (already computed during null fit)
        Py0, _ = _compute_P_quantities(Y_rot, X0_rot, w)
        yP0y = (Y_rot.squeeze() * Py0).sum()

        # Get Wald quantities for the LRT statistic
        wG = w.unsqueeze(1) * G_rot
        gWg = (G_rot * wG).sum(dim=0)
        gWX = G_rot.T @ (w.unsqueeze(1) * X0_rot)
        M00 = nf.M00
        M00_inv_gWXt = torch.linalg.solve(M00, gWX.T)
        S = gWg - (gWX * M00_inv_gWXt.T).sum(dim=1)
        gPy = (G_rot * Py0.unsqueeze(1)).sum(dim=0)

        S_safe = torch.clamp(S, min=1e-20)
        beta = gPy / S_safe
        se = torch.sqrt(sig2_e / S_safe)

        # LRT statistic = beta^2 * S / sig2_e (= Wald stat for fixed VCs)
        stat = beta ** 2 * S_safe / sig2_e

        # P-values from F(1, n-c-1)
        c = X0_rot.shape[1]
        df2 = n - c - 1
        p = _f_sf(stat, df1=1, df2=df2)

        return ScanResult(
            chr=vmeta.chr, pos=vmeta.pos, snp=vmeta.snp,
            a1=vmeta.a1, a2=vmeta.a2, af=af,
            beta=beta, se=se, stat=stat, p=p, test="lrt",
        )

    # ------------------------------------------------------------------
    # Score test (no per-SNP refitting)
    # ------------------------------------------------------------------

    def _score_test(
        self,
        G_rot: Tensor,
        nf: NullFit,
        vmeta: VariantMeta,
        af: Tensor,
    ) -> ScanResult:
        """Score test: gradient under the null, no refitting.

        T_score = (g^T P y)^2 / (sig2_e * g^T P g) ~ chi2(1)

        Fastest scan — no beta/SE computed.
        """
        n, m = G_rot.shape
        Y_rot = nf.Y_rot.squeeze()
        X0_rot = nf.X0_rot
        sig2_e = nf.sig2_e
        sig2_g = nf.sig2_g
        evals = nf.eigenvalues

        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (evals * lam + 1.0)
        Py, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv)

        # g^T P y
        gPy = (G_rot * Py.unsqueeze(1)).sum(dim=0)  # (m,)

        # g^T P g (need P applied to each column of G_rot)
        # P g_j = H^{-1} g_j - H^{-1} X (X^T H^{-1} X)^{-1} X^T H^{-1} g_j
        wG = H_inv.unsqueeze(1) * G_rot
        XtWG = X0_rot.T @ wG  # (c, m)
        M00_inv_XtWG = torch.linalg.solve(nf.M00, XtWG)  # (c, m)
        PG = wG - (H_inv.unsqueeze(1) * X0_rot) @ M00_inv_XtWG  # (n, m)

        gPg = (G_rot * PG).sum(dim=0)  # (m,)
        gPg_safe = torch.clamp(gPg, min=1e-20)

        # Score statistic
        stat = gPy ** 2 / (sig2_e * gPg_safe)

        # P-values from F(1, n-c-1)
        c = X0_rot.shape[1]
        df2 = n - c - 1
        p = _f_sf(stat, df1=1, df2=df2)

        # Score test does not produce beta/SE
        nan_m = torch.full((m,), float("nan"), dtype=STAT_DTYPE)

        return ScanResult(
            chr=vmeta.chr, pos=vmeta.pos, snp=vmeta.snp,
            a1=vmeta.a1, a2=vmeta.a2, af=af,
            beta=nan_m, se=nan_m, stat=stat, p=p, test="score",
        )

    # ------------------------------------------------------------------
    # P3D=FALSE: per-marker variance component re-estimation
    # ------------------------------------------------------------------

    def _score_chunk_refit(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str,
    ) -> ScanResult:
        """Per-SNP REML refit (P3D=FALSE). Re-estimates variance components
        for each marker with the SNP included as a fixed effect.

        This is expensive (one REML fit per SNP) but more accurate for
        oligogenic traits where individual markers explain large fractions
        of variance. Equivalent to GWASpoly's P3D=FALSE mode.

        Uses the eigendecomposition trick: rotate everything, then run
        REML optimization per SNP in the rotated space via OptimizerController.
        """
        n, m = G_chunk.shape
        U = null_fit.eigenvectors
        X0_rot = null_fit.X0_rot  # already rotated
        Y_rot = null_fit.Y_rot.squeeze()
        K_evals = null_fit.eigenvalues

        af = G_chunk.mean(dim=0) / 2.0
        betas = torch.zeros(m, dtype=STAT_DTYPE)
        ses = torch.zeros(m, dtype=STAT_DTYPE)
        stats = torch.zeros(m, dtype=STAT_DTYPE)
        p_vals = torch.zeros(m, dtype=STAT_DTYPE)

        c = X0_rot.shape[1]

        for j in range(m):
            g_j = G_chunk[:, j:j + 1]
            g_rot = rotate(g_j, U)  # (n, 1)
            X_full_rot = torch.cat([X0_rot, g_rot], dim=1)  # (n, c+1)

            try:
                # Re-estimate variance components via optimizer stack
                controller = OptimizerController(config=self.config)
                nf_j = controller.fit(Y_rot, X_full_rot, K_evals)

                sig2_g = nf_j.sig2_g
                sig2_e = nf_j.sig2_e

                lam = sig2_g / max(sig2_e, 1e-20)
                H_inv = 1.0 / (K_evals * lam + 1.0)

                # Compute beta for the SNP from the full model
                # X_full = [X0, g], beta_full = (X^T W X)^-1 X^T W y
                wX_full = H_inv.unsqueeze(1) * X_full_rot
                M_full = X_full_rot.T @ wX_full  # (c+1, c+1)
                XWy = X_full_rot.T @ (H_inv * Y_rot)  # (c+1,)
                beta_full = torch.linalg.solve(M_full, XWy)  # (c+1,)

                # SNP beta is the last element
                beta_snp = beta_full[-1]

                # SE from diagonal of (X^T W X)^-1 * sig2_e
                M_full_inv = torch.linalg.inv(M_full)
                var_beta_snp = sig2_e * M_full_inv[-1, -1]
                se_snp = torch.sqrt(torch.clamp(var_beta_snp, min=1e-20))

                stat_j = (beta_snp ** 2 / var_beta_snp).item()

                betas[j] = beta_snp
                ses[j] = se_snp
                stats[j] = stat_j
                df2 = n - c - 2  # extra df for the SNP
                p_vals[j] = _f_sf(
                    torch.tensor([stat_j], dtype=STAT_DTYPE),
                    df1=1, df2=max(df2, 1),
                )[0]
            except Exception:
                betas[j] = float("nan")
                ses[j] = float("nan")
                stats[j] = float("nan")
                p_vals[j] = float("nan")

        return ScanResult(
            chr=variant_meta.chr, pos=variant_meta.pos, snp=variant_meta.snp,
            a1=variant_meta.a1, a2=variant_meta.a2, af=af,
            beta=betas, se=ses, stat=stats, p=p_vals, test=test,
        )


# ---------------------------------------------------------------------------
# Chi-squared survival function (p-value from test statistic)
# ---------------------------------------------------------------------------

def _f_sf(stat: Tensor, df1: int = 1, df2: int = 1) -> Tensor:
    """P-values from F-distribution (matches GAPIT's MLM test)."""
    import scipy.stats as sp_stats
    import numpy as np

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.f.sf(stat_np, dfn=df1, dfd=df2)
    p_np = np.clip(p_np, 1e-300, 1.0)

    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
