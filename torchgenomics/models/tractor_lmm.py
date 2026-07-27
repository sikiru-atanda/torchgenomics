"""TractorLMM: ancestry-specific mixed-model GWAS for admixed cohorts.

Reference: Tan et al., "Extending GWAS to admixed cohorts with high degrees
of relatedness" (Tractor-Mix), Nat. Genet. 2026. Builds on the local-ancestry
deconvolution association test of Atkinson et al., "Tractor uses local
ancestry to enable the inclusion of admixed individuals in GWAS and to
reveal distinct signals of selection", Nat. Genet. 2021, and on the
GMMAT mixed-model score-test framework of Chen et al., "Control for
population structure and relatedness for binary traits in genetic
association studies via logistic mixed models", Am. J. Hum. Genet. 2016.

This module implements the continuous (Gaussian) null-model fit, the
binary (logistic GLMM, PQL null; delegating to ``BinaryGLMM``) null-model
fit, and the reusable GMMAT-style projection operator

    V = sigma^2 I + tau^2 K
    P = V^{-1} - V^{-1} X0 (X0^T V^{-1} X0)^{-1} X0^T V^{-1}

which residualizes any vector or matrix against the fixed-effect design
X0 under the estimated null covariance V. The per-variant efficient score
test itself (T = g^T P y, Var = g^T P g, one score per ancestry-specific
dosage column) is implemented by :meth:`TractorLMM.score_chunk` /
:meth:`TractorLMM._score_stats` in this same module, which consume the
``TractorNullFit`` produced by :meth:`TractorLMM.fit_null`.

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
from .base import VariantMeta


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

    # --- family dispatch + binary (logistic GLMM, PQL null) fields ---------
    # "gaussian" (default) uses the GMMAT projection P below; "binary" uses
    # the SAIGE/GMMAT logistic score-test working covariance built from the
    # PQL-fitted mean ``mu`` and working weights ``W = mu(1-mu)`` (Chen et al.
    # 2016, GMMAT eq. 4; Zhou et al. 2018, SAIGE). The binary path never
    # forms P and leaves the eigendecomposition fields (U/d/_XtVi/...) as
    # empty placeholders.
    family: str = "gaussian"
    _W: Tensor | None = None       # (n,) PQL working weights mu(1-mu)  [binary]
    _M00: Tensor | None = None     # (c, c) = (X0^T W X0)^{-1}          [binary]
    _binary_nf: object = None      # cached BinaryGLMM NullFit          [binary]

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

    def score_moments(self, Gv: Tensor) -> tuple[Tensor, Tensor]:
        """Efficient score ``T`` and its covariance ``Var`` for one variant.

        Shared entry point for both trait families so the downstream
        allele-count filter / rank-df / pseudo-inverse-effect logic in
        :meth:`TractorLMM.score_chunk` is written once (DRY). ``Gv`` is the
        ``(n, K)`` matrix of ancestry-specific dosage columns for a single
        variant, used **raw** (unrotated) in both families — matching the
        basis in which the null quantities were fitted.

        Gaussian (GMMAT continuous score test; Chen et al. 2016 §2.2)::

            T   = Gv^T (P y)            # self.resid == P y
            Var = Gv^T (P Gv)

        Binary (logistic GLMM PQL score test — BinaryGLMM's single-SNP test
        generalized from a scalar SNP to a K-column dosage matrix; Chen et
        al. 2016 GMMAT eq. 4-7, Zhou et al. 2018 SAIGE)::

            resid = y - mu                          # self.resid
            T     = Gv^T resid
            Var   = Gv^T W Gv - (X0^T W Gv)^T M00 (X0^T W Gv)

        where ``W = diag(mu(1-mu))`` are the PQL working weights and
        ``M00 = (X0^T W X0)^{-1}`` residualizes the score against the
        fixed-effect design (both cached on this null fit). For ``K = 1``
        this reduces exactly to ``BinaryGLMM.score_chunk``'s 1-df test
        (``U^2 / V``).
        """
        T = Gv.T @ self.resid                      # (K,)
        if self.family == "binary":
            WG = self._W.unsqueeze(1) * Gv          # (n, K)
            gWg = Gv.T @ WG                          # (K, K)
            X0tWG = self.X0.T @ WG                   # (c, K)
            Var = gWg - X0tWG.T @ (self._M00 @ X0tWG)
        else:
            Var = Gv.T @ self.Py(Gv)                # (K, K) == Gv^T P Gv
        return T, Var


@dataclass
class TractorScanResult:
    """Per-variant joint K-df score test + ancestry-specific effects.

    Distinct from the shared :class:`torchgenomics.models.base.ScanResult`
    schema because the Tractor-Mix association test produces *per-ancestry*
    effect estimates (one ``beta`` / ``se`` / ``p`` triple per local-ancestry
    dosage track) in addition to a single joint statistic per variant —
    a shape the shared schema (one ``beta``/``se`` column per variant) cannot
    represent (Atkinson et al. 2021, Tractor; Tan et al. 2026, Tractor-Mix).

    Attributes
    ----------
    snp, chr, pos : list — variant identifiers (length m)
    ancestry_names : list[str] — labels for the K ancestry dosage tracks
    joint_stat : (m,) — Rao score joint chi2_K statistic per variant
    joint_p : (m,) — chi2_K survival-function p-value per variant
    beta : (m, K) — per-ancestry effect estimate (Var^{-1} T)
    se : (m, K) — per-ancestry standard error (sqrt of diag(Var^{-1}))
    p_anc : (m, K) — per-ancestry two-sided normal p-value
    test : str — always "score" for this model
    """

    snp: list[str]
    chr: list[str]
    pos: list[int]
    ancestry_names: list[str]
    joint_stat: Tensor       # (m,)
    joint_p: Tensor          # (m,)
    beta: Tensor             # (m, K)
    se: Tensor               # (m, K)
    p_anc: Tensor            # (m, K)
    test: str = "score"

    def __len__(self) -> int:
        return len(self.snp)

    def to_dataframe(self):
        """Flatten to a long-form DataFrame with one row per variant.

        Per-ancestry columns are suffixed with the ancestry name, e.g.
        ``beta_AFR``, ``se_AFR``, ``p_AFR``.
        """
        import pandas as pd

        d = {
            "snp": self.snp,
            "chr": self.chr,
            "pos": self.pos,
            "joint_p": self.joint_p.detach().cpu().numpy(),
        }
        for k, nm in enumerate(self.ancestry_names):
            d[f"beta_{nm}"] = self.beta[:, k].detach().cpu().numpy()
            d[f"se_{nm}"] = self.se[:, k].detach().cpu().numpy()
            d[f"p_{nm}"] = self.p_anc[:, k].detach().cpu().numpy()
        return pd.DataFrame(d)


class TractorLMM:
    """Ancestry-specific mixed-model association test (Tractor-Mix).

    Continuous (Gaussian) null fit + reusable GMMAT projection P. The
    per-variant local-ancestry-specific score test (:meth:`score_chunk`)
    consumes the ``TractorNullFit`` produced by :meth:`fit_null`.

    Parameters
    ----------
    family : {"gaussian", "binary"}
        Trait family. "gaussian" uses the GMMAT continuous score test;
        "binary" fits a logistic GLMM null via PQL (delegating to the
        validated :class:`~torchgenomics.models.binary_glmm.BinaryGLMM`)
        and runs the SAIGE/GMMAT logistic score test, generalized from a
        scalar SNP to K ancestry-specific dosage columns.
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
        Y = Y.to(torch.float64).reshape(-1)
        X0 = X0.to(torch.float64)
        K = K.to(torch.float64)
        n, c = X0.shape

        if self.family == "binary":
            return self._fit_null_binary(Y, X0, K, n, c)

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

    def _fit_null_binary(
        self, Y: Tensor, X0: Tensor, K: Tensor, n: int, c: int
    ) -> TractorNullFit:
        """Fit the binary (logistic GLMM) null via PQL, GMMAT/SAIGE-style.

        Delegates the penalized quasi-likelihood null fit to the existing,
        validated :class:`torchgenomics.models.binary_glmm.BinaryGLMM`
        (Python-as-spec / DRY: the PQL solver is the single source of truth
        for the logistic-mixed-model null across the codebase), then caches
        exactly the quantities BinaryGLMM's own score test consumes:

        - ``mu`` — PQL-fitted null mean (``NullFit._glm_mu``);
        - ``W = diag(mu(1-mu))`` — working weights (``NullFit._glm_W``);
        - ``M00 = (X0^T W X0)^{-1}`` — covariate-adjustment matrix
          (``NullFit.M00``);

        so that :meth:`TractorNullFit.score_moments` reproduces
        ``BinaryGLMM.score_chunk``'s score ``U = g^T (y - mu)`` and variance
        ``V = g^T W g - (X0^T W g)^T M00 (X0^T W g)`` per ancestry column,
        in the same raw (unrotated) sample basis (``BinaryGLMM`` stores
        ``Y_rot = Y`` and ``X0_rot = X0`` and scores ``G`` raw). See Chen et
        al. 2016 (GMMAT, eq. 4-7) and Zhou et al. 2018 (SAIGE).

        The score residual is stored as ``resid = y - mu`` so the shared
        downstream (filter / rank-df / effect) logic in :meth:`score_chunk`
        is reused unchanged.
        """
        from .binary_glmm import BinaryGLMM

        # PQL null (SPA irrelevant here: only mu / W / M00 are consumed).
        binary_nf = BinaryGLMM(use_spa=False).fit_null(Y, X0, K)
        mu = binary_nf._glm_mu.to(torch.float64).reshape(-1)   # (n,)
        W = binary_nf._glm_W.to(torch.float64).reshape(-1)      # (n,)
        M00 = binary_nf.M00.to(torch.float64)                   # (c, c)
        resid = Y - mu                                          # y - mu, (n,)

        empty = torch.empty(0, dtype=torch.float64, device=Y.device)
        nf = TractorNullFit(
            U=empty,
            d=empty,
            sigma2=float(binary_nf.sig2_e),
            tau2=float(binary_nf.sig2_g),
            X0=X0,
            resid=resid,
            n=n,
            c=c,
            _XtVi=empty,
            _XtViX_inv=empty,
            family="binary",
            _W=W,
            _M00=M00,
            _binary_nf=binary_nf,
        )
        return nf

    def _score_stats(
        self, null: TractorNullFit, Gv: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """Per-variant joint Rao score statistic against K ancestry dosages.

        Parameters
        ----------
        null : TractorNullFit — cached null-model projection P
        Gv : (n, K) — ancestry-specific dosage columns for one variant

        Returns
        -------
        T : (K,) — efficient score ``Gv^T P y``
        Var : (K, K) — score covariance ``Gv^T P Gv``
        Vinv : (K, K) — Moore-Penrose pseudo-inverse of Var (robust to
            rank-deficient ancestry tracks, e.g. an ancestry absent from
            the local cohort at this locus)
        joint : float — joint statistic ``T^T Vinv T ~ chi2_K`` under the
            null (Chen et al. 2016, GMMAT eq. 7; Atkinson et al. 2021,
            Tractor, joint test extension in Tan et al. 2026, Tractor-Mix)
        """
        T, Var = null.score_moments(Gv)   # family-dispatched (gaussian|binary)
        Vinv = torch.linalg.pinv(Var)
        joint = float(T @ Vinv @ T)
        return T, Var, Vinv, joint

    def score_chunk(
        self,
        null: TractorNullFit,
        G_chunk: Tensor,
        meta: VariantMeta,
        L_chunk: Tensor | None = None,
    ) -> TractorScanResult:
        """Joint K-df Rao score test + ancestry-specific effects, per variant.

        For each variant ``j`` with ancestry-dosage matrix ``Gv = G_chunk[:, :, j].T``
        (n, K), computes the efficient score ``T = Gv^T P y``, its covariance
        ``Var = Gv^T P Gv`` under the fitted null model, and:

        - the joint test ``stat = T^T Var^-1 T ~ chi2_df`` (Rao score test
          against the null that *all* surviving ancestry-specific effects
          are zero; Chen et al. 2016, GMMAT §2.2; Atkinson et al. 2021,
          Tractor §Methods; Tan et al. 2026, Tractor-Mix, joint
          local-ancestry test), with p-value from the chi2_df survival
          function;
        - per-ancestry effect estimates ``beta = Var^-1 T``, standard errors
          ``se = sqrt(diag(Var^-1))``, and two-sided normal p-values
          ``p_a = 2 * (1 - Phi(|beta_a / se_a|))``.

        **Allele-count threshold (Tractor convention).** Before building
        ``Gv``, each ancestry ``a`` is tested for minimum evidence at this
        locus: its summed dosage (``Gv[:, a].sum()``, the ancestry-specific
        allele count) must be ``>= self.min_allele_count`` (default 50,
        Atkinson et al. 2021, Tractor §Methods — ancestry tracks with too
        few copies of the ancestry-of-origin at a locus give unstable
        effect estimates and are excluded rather than reported). Dropped
        ancestries receive ``se = nan`` / ``p_anc = nan`` in the returned
        result; the joint test is computed over the *surviving* ancestries
        only. If every ancestry is dropped at a variant, no test can be
        formed and ``joint_stat`` / ``joint_p`` are ``nan``.

        **Degrees of freedom = rank, not ancestry count.** Even after the
        allele-count filter, the surviving ``Var = Gv_keep^T P Gv_keep``
        can still be rank-deficient — e.g. two surviving ancestry dosage
        columns that are collinear at this locus, or a column effectively
        annihilated by the projection P (zero variance after removing the
        fixed-effect design). Using ``df = (number of surviving
        ancestries)`` in that case assumes full rank and understates
        significance (the chi2 tail with too many df is stochastically
        larger than the correct one). This implementation instead uses
        ``df = torch.linalg.matrix_rank(Var)`` — the true number of
        identifiable degrees of freedom in the surviving score statistic
        — for the chi2 survival-function p-value; if ``df == 0`` (the
        surviving Var is exactly singular with no identifiable direction),
        ``joint_p = nan``.

        ``Var^-1`` is computed via Moore-Penrose pseudo-inverse rather than
        a plain inverse so that ancestry tracks with (near-)zero local
        dosage variance at a given locus (e.g. an ancestry not observed in
        the cohort at that position) degrade gracefully instead of raising.

        Where both ``beta_a`` and ``se_a`` are (numerically) zero for a
        surviving ancestry — the pinv-degenerate case where that ancestry's
        column contributes nothing identifiable to the score, distinct from
        being dropped outright by the allele-count filter — the raw ratio
        ``beta_a / se_a`` is the indeterminate form 0/0 (``nan`` in IEEE
        arithmetic). This is guarded so the division is well-defined
        (``p_a = 1.0``, i.e. "no signal" — the same conclusion a defined
        z-statistic of 0 would give) rather than silently propagating
        ``nan`` for a case that is not actually a dropped ancestry.

        Parameters
        ----------
        null : TractorNullFit — cached null-model fit from :meth:`fit_null`
        G_chunk : (K, n, c) — ancestry-specific dosage tensor for c variants
        meta : VariantMeta — variant identifiers (snp/chr/pos), length c
        L_chunk : optional — reserved for a future local-ancestry-probability
            weighted variant of the test; unused by the continuous model

        Returns
        -------
        TractorScanResult
        """
        from scipy.stats import chi2, norm

        n_anc, n, c = G_chunk.shape
        names = self._ancestry_names or [f"anc{k}" for k in range(n_anc)]
        device = G_chunk.device

        joint_stat = torch.full((c,), float("nan"), dtype=torch.float64, device=device)
        joint_p = torch.full((c,), float("nan"), dtype=torch.float64, device=device)
        beta = torch.zeros(c, n_anc, dtype=torch.float64, device=device)
        se = torch.full((c, n_anc), float("nan"), dtype=torch.float64, device=device)
        p_anc = torch.full((c, n_anc), float("nan"), dtype=torch.float64, device=device)

        zero_tol = 1e-12
        for j in range(c):
            full = G_chunk[:, :, j].to(torch.float64)   # (K, n)
            ac = full.sum(dim=1)                          # (K,) ancestry allele count
            keep = (ac >= self.min_allele_count).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                # Every ancestry below the allele-count threshold at this
                # locus: no joint test can be formed and no per-ancestry
                # effect is estimable. All outputs stay at their nan/zero
                # init (joint_stat/joint_p/se/p_anc already nan; beta 0).
                continue

            Gv = full[keep].T  # (n, K_keep)
            T, Var, Vinv, joint = self._score_stats(null, Gv)
            rank = int(torch.linalg.matrix_rank(Var))
            if rank == 0:
                # Surviving Var is exactly singular with no identifiable
                # direction -- no valid chi2 df for a joint test.
                joint_p[j] = float("nan")
            else:
                joint_stat[j] = joint
                joint_p[j] = float(chi2.sf(joint, df=rank))

            b = Vinv @ T
            s = torch.sqrt(torch.clamp(torch.diag(Vinv), min=0.0))

            # 0/0 guard: when both beta and se are numerically zero for a
            # surviving ancestry, beta/se is the indeterminate 0/0 rather
            # than a genuine large |z|; report p=1.0 ("no signal") instead
            # of letting it fall through as nan.
            zero_mask = (s.abs() < zero_tol) & (b.abs() < zero_tol)
            s_safe = torch.where(zero_mask, torch.ones_like(s), s)
            z = b / s_safe
            p_kept = torch.as_tensor(
                2.0 * norm.sf(torch.abs(z).detach().cpu().numpy()),
                dtype=torch.float64,
                device=device,
            )
            p_kept = torch.where(zero_mask, torch.ones_like(p_kept), p_kept)

            for slot, k in enumerate(keep.tolist()):
                beta[j, k] = b[slot]
                se[j, k] = s[slot]
                p_anc[j, k] = p_kept[slot]
            # Dropped ancestries stay nan (already the init default; set
            # explicitly for robustness against future default changes).
            dropped = set(range(n_anc)) - set(keep.tolist())
            for k in dropped:
                se[j, k] = float("nan")
                p_anc[j, k] = float("nan")

        return TractorScanResult(
            snp=list(meta.snp),
            chr=list(meta.chr),
            pos=list(meta.pos),
            ancestry_names=list(names),
            joint_stat=joint_stat,
            joint_p=joint_p,
            beta=beta,
            se=se,
            p_anc=p_anc,
        )
