"""TractorLMM: ancestry-specific mixed-model GWAS for admixed cohorts.

Reference: Tan et al., "Extending GWAS to admixed cohorts with high degrees
of relatedness" (Tractor-Mix), Nat. Genet. 2026. Builds on the local-ancestry
deconvolution association test of Atkinson et al., "Tractor uses local
ancestry to enable the inclusion of admixed individuals in GWAS and to
reveal distinct signals of selection", Nat. Genet. 2021, and on the
GMMAT mixed-model score-test framework of Chen et al., "Control for
population structure and relatedness for binary traits in genetic
association studies via logistic mixed models", Am. J. Hum. Genet. 2016.

This module implements the continuous (Gaussian) null-model fit and the
reusable GMMAT-style projection operator

    V = sigma^2 I + tau^2 K
    P = V^{-1} - V^{-1} X0 (X0^T V^{-1} X0)^{-1} X0^T V^{-1}

which residualizes any vector or matrix against the fixed-effect design
X0 under the estimated null covariance V. The per-variant efficient score
test itself (T = g^T P y, Var = g^T P g, one score per ancestry-specific
dosage column) is implemented in a follow-up unit; this module only
produces the ``TractorNullFit`` that the score test consumes.

Variance components (tau2 = additive genetic variance multiplying the
GRM K, sigma2 = residual variance multiplying the identity) are obtained
by delegating to the existing GAPIT-exact EMMA REML solver
(:func:`torchgenomics.optim.emma_reml.gapit_emma_remle`), which is the
same variance-component estimator already used by
:class:`torchgenomics.models.single_trait_lmm.SingleTraitLMM`. No new
REML solver is introduced here (DRY: Python-as-spec convention — the
existing solver is the single source of truth for null variance
components across all LMM-family models in this codebase).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..optim.emma_reml import gapit_emma_remle


@dataclass
class TractorNullFit:
    """Cached null-model quantities for the Tractor-Mix continuous model.

    All tensors are FP64. ``Py`` / ``PX`` apply the GMMAT projection
    ``P = V^{-1} - V^{-1} X0 (X0^T V^{-1} X0)^{-1} X0^T V^{-1}`` without
    ever materializing the dense ``n x n`` matrix ``P`` itself; instead
    ``V^{-1}`` is applied via the eigendecomposition of ``K`` (see
    Chen et al. 2016, GMMAT, eq. 4-6, and Zhou et al. 2018, SAIGE, for the
    analogous rotation-based construction).
    """

    U: Tensor            # (n, n) eigenvectors of K
    d: Tensor            # (n,) eigenvalues of K
    sigma2: float        # residual variance (multiplies I)
    tau2: float          # additive genetic variance (multiplies K)
    X0: Tensor           # (n, c) fixed-effect design
    resid: Tensor        # (n,) P @ Y
    n: int
    c: int
    _XtVi: Tensor        # (c, n) = X0^T V^{-1}
    _XtViX_inv: Tensor   # (c, c) = (X0^T V^{-1} X0)^{-1}

    def _Vi(self, M: Tensor) -> Tensor:
        """Apply V^{-1} to ``M`` via the eigendecomposition of K.

        V^{-1} = U diag(1 / (tau2 * d + sigma2)) U^T
        """
        w = 1.0 / (self.tau2 * self.d + self.sigma2)
        return self.U @ (w.unsqueeze(1) * (self.U.T @ M))

    def Py(self, M: Tensor) -> Tensor:
        """Apply the GMMAT projection P to a column vector or matrix ``M``.

        ``P M = V^{-1} M - V^{-1} X0 (X0^T V^{-1} X0)^{-1} X0^T V^{-1} M``.

        Because P already residualizes against the fixed-effect design X0,
        no separate centering of a genotype/dosage vector against
        covariates is needed before computing the efficient score
        ``g^T P y`` (Chen et al. 2016, GMMAT §2.2).
        """
        ViM = self._Vi(M)
        return ViM - self._Vi(self.X0) @ (self._XtViX_inv @ (self._XtVi @ M))

    def PX(self, M: Tensor) -> Tensor:
        """Alias of :meth:`Py` — apply P to an arbitrary (n, k) matrix."""
        return self.Py(M)


class TractorLMM:
    """Ancestry-specific mixed-model association test (Tractor-Mix).

    Continuous (Gaussian) null fit + reusable GMMAT projection P. The
    per-variant local-ancestry-specific score test (Task 4 of this unit)
    consumes the ``TractorNullFit`` produced by :meth:`fit_null`.

    Parameters
    ----------
    family : {"gaussian", "binary"}
        Trait family. Only "gaussian" is implemented by :meth:`fit_null`
        in this module; "binary" (PQL null, GMMAT-style) is deferred to
        a follow-up unit.
    test : {"score", "wald"}
        Association test to use in the (not-yet-implemented) scan step.
    min_allele_count : int
        Minimum ancestry-specific allele count for a variant to be tested
        (Tractor convention, Atkinson et al. 2021).
    use_spa : bool
        Whether to apply saddlepoint approximation for rare-variant score
        calibration (binary trait only).
    spa_threshold : float
        |Z| threshold above which SPA is triggered.
    ancestry_names : list[str] | None
        Optional labels for the ancestry-specific dosage columns (e.g.
        ["AFR", "EUR"]); defaults to "anc0", "anc1", ... when None. Not
        used by the continuous null fit itself but threaded through so
        the scan adapter can label per-ancestry effect columns.
    """

    def __init__(
        self,
        family: str = "gaussian",
        test: str = "score",
        min_allele_count: int = 50,
        use_spa: bool = False,
        spa_threshold: float = 2.0,
        ancestry_names: list[str] | None = None,
    ) -> None:
        if family not in ("gaussian", "binary"):
            raise ValueError(f"family must be gaussian|binary, got {family}")
        if test not in ("score", "wald"):
            raise ValueError(f"test must be score|wald, got {test}")
        self.family = family
        self.test = test
        self.min_allele_count = min_allele_count
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold
        # None -> scan adapter defaults to anc0/anc1/... labels
        self._ancestry_names = ancestry_names

    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor) -> TractorNullFit:
        """Fit the continuous (Gaussian) null model and build P.

        Estimates ``tau2`` (additive genetic variance, multiplies K) and
        ``sigma2`` (residual variance, multiplies I) via the existing
        GAPIT-exact EMMA REML solver
        (:func:`torchgenomics.optim.emma_reml.gapit_emma_remle`), then
        builds ``V^{-1}`` from a direct eigendecomposition of K so that
        the returned :class:`TractorNullFit` can apply the GMMAT
        projection P to any vector/matrix without re-forming ``V`` or
        ``P`` densely.

        Parameters
        ----------
        Y : (n,) or (n, 1) — continuous phenotype
        X0 : (n, c) — fixed-effect covariate design (including intercept)
        K : (n, n) — kinship/GRM matrix

        Returns
        -------
        TractorNullFit
        """
        if self.family != "gaussian":
            raise NotImplementedError(
                "TractorLMM.fit_null currently implements only the "
                "continuous (gaussian) null fit; binary PQL null is a "
                "follow-up unit."
            )
        Y = Y.to(torch.float64).reshape(-1)
        X0 = X0.to(torch.float64)
        K = K.to(torch.float64)
        n, c = X0.shape

        # Variance components via the existing GAPIT-exact EMMA REML
        # solver (same estimator SingleTraitLMM.fit_null already uses).
        sig2_g, sig2_e, _ll, _delta, _trace = gapit_emma_remle(Y, X0, K)
        tau2 = float(sig2_g)
        sigma2 = float(sig2_e)

        # Direct eigendecomposition of K (not the restricted eigendecomposition
        # internal to gapit_emma_remle) so V^{-1} can be applied to arbitrary
        # future vectors (e.g. per-variant dosage columns in the score test).
        d, U = torch.linalg.eigh(K)

        w = 1.0 / (tau2 * d + sigma2)
        Vi = U @ (w.unsqueeze(1) * U.T)
        XtVi = X0.T @ Vi
        XtViX_inv = torch.linalg.inv(XtVi @ X0)

        nf = TractorNullFit(
            U=U,
            d=d,
            sigma2=sigma2,
            tau2=tau2,
            X0=X0,
            resid=torch.zeros(n, dtype=torch.float64, device=Y.device),
            n=n,
            c=c,
            _XtVi=XtVi,
            _XtViX_inv=XtViX_inv,
        )
        nf.resid = nf.Py(Y.unsqueeze(1)).squeeze(1)
        return nf
