"""Multi-Environment and Multi-Trait Haplotype GWAS (Phase 47).

Extends Phase 46's single-trait haplotype GWAS to:

1. **HaplotypeMultiEnvGWAS** — haplotype association across E discrete
   environments, delegating null fitting to :class:`MultiEnvLMM`.
2. **HaplotypeMultiTraitGWAS** — haplotype association across d continuous
   traits, delegating null fitting to :class:`MultiTraitLMM`.
3. **HaplotypeMTMETGWAS** — haplotype association across d traits x E
   environments, delegating null fitting to :class:`MultiTraitMultiEnvLMM`.

All three classes use a thin composition pattern: they reuse the existing
null model fitting and GLS machinery, but replace the per-SNP scan with a
per-block GLS Wald test on the haplotype dosage matrix.

References
----------
- Zaykin et al. (2002). Testing association of statistically inferred
  haplotypes with discrete and continuous traits. Human Heredity.
- Zhou & Stephens (2014). Efficient mvLMM (GEMMA).
- Smith et al. (2001). FA models for multi-environment trials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import rotate
from .base import NullFit
from .haplotype_gwas import HaplotypeBlock, HaplotypeGWAS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# chi2 p-value helper (local, avoids circular import)
# ---------------------------------------------------------------------------

def _chi2_sf(stat: float, df: int) -> float:
    """Single chi2 survival probability."""
    from scipy.stats import chi2 as chi2_dist
    if df <= 0:
        return 1.0
    p = float(chi2_dist.sf(stat, df=df))
    return max(p, 1e-300)


def _chi2_sf_tensor(stat: Tensor, df: int) -> Tensor:
    """Vectorized chi2 survival probability."""
    from scipy.stats import chi2 as chi2_dist
    if df <= 0:
        return torch.ones_like(stat)
    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = chi2_dist.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HaplotypeMultiEnvResult:
    """Results from multi-environment haplotype GWAS."""

    block_id: list[str] = field(default_factory=list)
    chr: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    n_haplotypes: list[int] = field(default_factory=list)

    # Per-block effects: list of (H-1, E) tensors
    beta: list[Tensor] = field(default_factory=list)
    se: list[Tensor] = field(default_factory=list)

    # Joint test: all (H-1)*E betas = 0
    stat_joint: Tensor = field(default_factory=lambda: torch.tensor([]))
    p_joint: Tensor = field(default_factory=lambda: torch.tensor([]))

    # Per-haplotype across envs: chi2(E) each
    stat_per_hap: list[Tensor] = field(default_factory=list)
    p_per_hap: list[Tensor] = field(default_factory=list)

    # Per-env across haplotypes: chi2(H-1) each
    stat_per_env: list[Tensor] = field(default_factory=list)
    p_per_env: list[Tensor] = field(default_factory=list)

    # Homogeneity per haplotype: chi2(E-1) each
    stat_homogeneity: list[Tensor] = field(default_factory=list)
    p_homogeneity: list[Tensor] = field(default_factory=list)

    # Haplotype-GxE interaction: chi2((H-1)*(E-1))
    stat_hap_gxe: Tensor = field(default_factory=lambda: torch.tensor([]))
    p_hap_gxe: Tensor = field(default_factory=lambda: torch.tensor([]))

    haplotype_labels: list[list[str]] = field(default_factory=list)
    env_names: Optional[list[str]] = None
    n_obs: int = 0


@dataclass
class HaplotypeMultiTraitResult:
    """Results from multi-trait haplotype GWAS."""

    block_id: list[str] = field(default_factory=list)
    chr: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    n_haplotypes: list[int] = field(default_factory=list)

    # Per-block effects: list of (H-1, d) tensors
    beta: list[Tensor] = field(default_factory=list)
    se: list[Tensor] = field(default_factory=list)

    # Joint test: all (H-1)*d betas = 0
    stat_joint: Tensor = field(default_factory=lambda: torch.tensor([]))
    p_joint: Tensor = field(default_factory=lambda: torch.tensor([]))

    # Per-haplotype across traits: chi2(d) each
    stat_per_hap: list[Tensor] = field(default_factory=list)
    p_per_hap: list[Tensor] = field(default_factory=list)

    # Per-trait across haplotypes: chi2(H-1) each
    stat_per_trait: list[Tensor] = field(default_factory=list)
    p_per_trait: list[Tensor] = field(default_factory=list)

    haplotype_labels: list[list[str]] = field(default_factory=list)
    trait_names: Optional[list[str]] = None
    n_obs: int = 0


@dataclass
class HaplotypeMTMETResult:
    """Results from multi-trait multi-environment haplotype GWAS."""

    block_id: list[str] = field(default_factory=list)
    chr: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    n_haplotypes: list[int] = field(default_factory=list)

    # Per-block effects: list of (H-1, d, E) tensors
    beta: list[Tensor] = field(default_factory=list)
    se: list[Tensor] = field(default_factory=list)

    # Joint test: all (H-1)*d*E betas = 0
    stat_joint: Tensor = field(default_factory=lambda: torch.tensor([]))
    p_joint: Tensor = field(default_factory=lambda: torch.tensor([]))

    # Per-haplotype across all trait-envs: chi2(dE) each
    stat_per_hap: list[Tensor] = field(default_factory=list)
    p_per_hap: list[Tensor] = field(default_factory=list)

    # Per-trait across envs and haplotypes: chi2((H-1)*E) each
    stat_per_trait: list[Tensor] = field(default_factory=list)
    p_per_trait: list[Tensor] = field(default_factory=list)

    # Per-env across traits and haplotypes: chi2((H-1)*d) each
    stat_per_env: list[Tensor] = field(default_factory=list)
    p_per_env: list[Tensor] = field(default_factory=list)

    # Per-haplotype GxE: chi2(E-1) each
    stat_hap_gxe: list[Tensor] = field(default_factory=list)
    p_hap_gxe: list[Tensor] = field(default_factory=list)

    haplotype_labels: list[list[str]] = field(default_factory=list)
    trait_names: Optional[list[str]] = None
    env_names: Optional[list[str]] = None
    n_obs: int = 0


# ---------------------------------------------------------------------------
# Core GLS Wald helper for multi-column haplotype dosage blocks
# ---------------------------------------------------------------------------

def _haplotype_gls_wald(
    D_rot: Tensor,
    null_fit: NullFit,
    p: int,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """GLS Wald test for a single block with (H-1) haplotype regressors
    and p pseudo-traits (E, d, or dE).

    Builds the full-system ``(c*p + h*p) x (c*p + h*p)`` GLS and solves
    once per block, then extracts the haplotype sub-block of Var(beta).

    Parameters
    ----------
    D_rot : (n, h) rotated haplotype dosage (reference dropped), h = H-1.
    null_fit : NullFit from MultiEnvLMM / MultiTraitLMM / MultiTraitMultiEnvLMM.
        Must contain: weights (n, p, p), Y_rot (n, p), X0_rot (n, c),
        M00 (c*p, c*p), b0 (c*p,).
    p : number of pseudo-traits.

    Returns
    -------
    beta_hap : (h, p) haplotype effect estimates.
    se_hap : (h, p) standard errors.
    Var_hap : (h*p, h*p) variance-covariance of the haplotype block.
    stat_joint : float, Wald chi2(h*p) joint statistic.
    """
    n, h = D_rot.shape
    device = D_rot.device

    W = null_fit.weights       # (n, p, p)
    Y_rot = null_fit.Y_rot     # (n, p)
    X0_rot = null_fit.X0_rot   # (n, c)
    M00 = null_fit.M00         # (c*p, c*p)
    b0 = null_fit.b0           # (c*p,)
    c = X0_rot.shape[1]

    # Weighted phenotype: WY[i, t] = sum_s W[i, t, s] * Y[i, s]
    WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, p)

    # M11: (h, p, h, p) block — D^T W D per haplotype pair
    # For h haplotype columns, M11[a, s, b, t] = sum_i D[i,a] * W[i,s,t] * D[i,b]
    # Reshape to (h*p, h*p)
    # Use outer product of D columns with weights
    M11_4d = torch.einsum('na,nst,nb->asbt', D_rot, W, D_rot)  # (h, p, h, p)
    M11 = M11_4d.reshape(h * p, h * p)

    # M01: (c*p, h*p) block — X0^T W D per (covariate, haplotype) pair
    # M01_raw[a, s, b, t] = sum_i X0[i,a] * W[i,s,t] * D[i,b]
    M01_4d = torch.einsum('na,nst,nb->asbt', X0_rot, W, D_rot)  # (c, p, h, p)
    M01 = M01_4d.reshape(c * p, h * p)

    # b1: (h*p,) — D^T W Y per (haplotype, pseudo-trait)
    # b1_2d[a, s] = sum_i D[i,a] * WY[i,s]
    b1_2d = torch.einsum('na,ns->as', D_rot, WY)  # (h, p)
    b1 = b1_2d.reshape(h * p)

    # Build full (c*p + h*p) x (c*p + h*p) system
    dim_cov = c * p
    dim_hap = h * p
    dim_full = dim_cov + dim_hap

    M_full = torch.zeros(dim_full, dim_full, dtype=STAT_DTYPE, device=device)
    M_full[:dim_cov, :dim_cov] = M00
    M_full[:dim_cov, dim_cov:] = M01
    M_full[dim_cov:, :dim_cov] = M01.T
    M_full[dim_cov:, dim_cov:] = M11

    rhs = torch.zeros(dim_full, dtype=STAT_DTYPE, device=device)
    rhs[:dim_cov] = b0
    rhs[dim_cov:] = b1

    # Jitter for numerical stability
    diag_min = M_full.diag().abs().min().item()
    if diag_min < 1e-10:
        M_full = M_full + torch.eye(dim_full, dtype=STAT_DTYPE, device=device) * 1e-10

    # Solve for coefficients
    coef = torch.linalg.solve(M_full, rhs)  # (dim_full,)
    beta_flat = coef[dim_cov:]  # (h*p,)

    # Var(beta): extract haplotype sub-block of M_full^{-1}
    rhs_var = torch.zeros(dim_full, dim_hap, dtype=STAT_DTYPE, device=device)
    rhs_var[dim_cov:, :] = torch.eye(dim_hap, dtype=STAT_DTYPE, device=device)
    Minv_cols = torch.linalg.solve(M_full, rhs_var)  # (dim_full, dim_hap)
    Var_hap = Minv_cols[dim_cov:, :]  # (dim_hap, dim_hap)

    # Reshape beta and SE
    beta_hap = beta_flat.reshape(h, p)
    se_hap = torch.sqrt(
        torch.clamp(Var_hap.diag().reshape(h, p), min=1e-30)
    )

    # Joint Wald: beta^T Var^{-1} beta ~ chi2(h*p)
    Var_inv_beta = torch.linalg.solve(Var_hap, beta_flat.unsqueeze(-1)).squeeze(-1)
    stat_joint = max(float((beta_flat * Var_inv_beta).sum()), 0.0)

    return beta_hap, se_hap, Var_hap, stat_joint


def _wald_subblock(
    beta_flat: Tensor,
    Var_full: Tensor,
    indices: list[int],
) -> tuple[float, float]:
    """Wald chi2 test on a sub-block of the haplotype beta vector.

    Parameters
    ----------
    beta_flat : (h*p,) flattened haplotype effects.
    Var_full : (h*p, h*p) variance-covariance.
    indices : integer indices into the h*p dimension.

    Returns
    -------
    stat, p_value
    """
    idx = torch.tensor(indices, dtype=torch.long, device=beta_flat.device)
    beta_sub = beta_flat[idx]
    Var_sub = Var_full[idx][:, idx]

    Var_inv_b = torch.linalg.solve(Var_sub, beta_sub.unsqueeze(-1)).squeeze(-1)
    stat = max(float((beta_sub * Var_inv_b).sum()), 0.0)
    p = _chi2_sf(stat, df=len(indices))
    return stat, p


def _wald_contrast(
    beta_flat: Tensor,
    Var_full: Tensor,
    indices: list[int],
    C: Tensor,
) -> tuple[float, float]:
    """Wald test with contrast matrix C on a sub-block.

    Parameters
    ----------
    beta_flat : (h*p,) flattened haplotype effects.
    Var_full : (h*p, h*p) variance-covariance.
    indices : integer indices into the h*p dimension for the sub-block.
    C : (k, len(indices)) contrast matrix.

    Returns
    -------
    stat, p_value
    """
    idx = torch.tensor(indices, dtype=torch.long, device=beta_flat.device)
    beta_sub = beta_flat[idx]
    Var_sub = Var_full[idx][:, idx]

    Cb = C @ beta_sub  # (k,)
    CVCt = C @ Var_sub @ C.T  # (k, k)
    CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
    stat = max(float((Cb * CVCt_inv_Cb).sum()), 0.0)
    p = _chi2_sf(stat, df=C.shape[0])
    return stat, p


def _build_contrast_matrix(size: int, device: torch.device) -> Tensor:
    """Successive-differences contrast matrix (size-1, size)."""
    C = torch.zeros(size - 1, size, dtype=STAT_DTYPE, device=device)
    for i in range(size - 1):
        C[i, i] = 1.0
        C[i, i + 1] = -1.0
    return C


# ---------------------------------------------------------------------------
# Shared scan infrastructure
# ---------------------------------------------------------------------------

def _construct_and_prepare_blocks(
    scanner: HaplotypeGWAS,
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    blocks,
    haplotypes: Tensor | None,
    **block_kwargs,
) -> list[HaplotypeBlock]:
    """Construct haplotype blocks using the shared HaplotypeGWAS infrastructure."""
    return scanner.construct_haplotypes(
        G, variant_pos, variant_chr,
        blocks=blocks, haplotypes=haplotypes,
        **block_kwargs,
    )


def _drop_reference(hb: HaplotypeBlock) -> tuple[Tensor, list[str]]:
    """Drop reference haplotype and return test dosage + labels."""
    H = len(hb.haplotypes)
    ref_idx = hb.frequencies.argmax().item()
    test_idx = [j for j in range(H) if j != ref_idx]
    D = hb.dosage[:, test_idx].to(STAT_DTYPE)
    test_labels = [hb.haplotypes[j] for j in test_idx]
    return D, test_labels


# ---------------------------------------------------------------------------
# Class 1: HaplotypeMultiEnvGWAS
# ---------------------------------------------------------------------------

class HaplotypeMultiEnvGWAS:
    """Haplotype association testing across E discrete environments.

    Delegates null model fitting to :class:`MultiEnvLMM`, then runs
    per-block GLS Wald tests on haplotype dosage matrices.

    Parameters
    ----------
    method : str
        ``"block"`` or ``"window"``.
    window_size : int
        SNPs per window (for ``method="window"``).
    step : int
        Window step size.
    block_method : str
        LD block detection method.
    min_hap_freq : float
        Minimum haplotype frequency.
    max_haplotypes : int
        Maximum haplotypes per block.
    em_max_iter, em_tol : int, float
        EM parameters for unphased inference.
    vg_structure : str
        ``"unstructured"`` or ``"fa(k)"`` for the genetic covariance.
    ploidy : int
        Ploidy level.
    """

    def __init__(
        self,
        method: str = "block",
        window_size: int = 5,
        step: int = 1,
        block_method: str = "gabriel",
        min_hap_freq: float = 0.01,
        max_haplotypes: int = 20,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        vg_structure: str = "unstructured",
        ploidy: int = 2,
    ):
        self.method = method
        self.window_size = window_size
        self.step = step
        self.block_method = block_method
        self.min_hap_freq = min_hap_freq
        self.max_haplotypes = max_haplotypes
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.vg_structure = vg_structure
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor,
        *,
        env_names: Optional[list[str]] = None,
        **kwargs,
    ) -> NullFit:
        """Fit multi-environment null model.

        Parameters
        ----------
        Y : (n, E) phenotype matrix, one column per environment.
        X0 : (n, c) covariates.
        K : (n, n) kinship matrix.
        env_names : optional environment names.

        Returns
        -------
        NullFit with cached GLS quantities.
        """
        from .multi_env_lmm import MultiEnvLMM

        met = MultiEnvLMM(
            config=NumericalConfig(),
            vg_structure=self.vg_structure,
        )
        nf = met.fit_null(Y, X0, K, env_names=env_names, **kwargs)
        return nf

    def scan(
        self,
        G: Tensor,
        null_fit: NullFit,
        variant_pos: list[int] | None = None,
        variant_chr: list[str] | None = None,
        blocks=None,
        haplotypes: Tensor | None = None,
        **block_kwargs,
    ) -> HaplotypeMultiEnvResult:
        """Run multi-environment haplotype scan.

        Parameters
        ----------
        G : (n, m) genotype dosage matrix.
        null_fit : output of ``fit_null()``.
        variant_pos, variant_chr : per-variant metadata.
        blocks : pre-computed LD blocks.
        haplotypes : (n, ploidy, m) phased data.

        Returns
        -------
        HaplotypeMultiEnvResult
        """
        n, m = G.shape
        E = null_fit.n_env
        U = null_fit.eigenvectors

        if variant_pos is None:
            variant_pos = list(range(m))
        if variant_chr is None:
            variant_chr = ["1"] * m

        # Build haplotype scanner for block construction
        scanner = HaplotypeGWAS(
            method=self.method,
            test="f_test",
            window_size=self.window_size,
            step=self.step,
            block_method=self.block_method,
            min_hap_freq=self.min_hap_freq,
            max_haplotypes=self.max_haplotypes,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            ploidy=self.ploidy,
        )
        hap_blocks = _construct_and_prepare_blocks(
            scanner, G, variant_pos, variant_chr,
            blocks, haplotypes, **block_kwargs,
        )

        # Accumulators
        block_ids = []
        chrs = []
        starts = []
        ends = []
        n_variants_list = []
        n_haps_list = []
        beta_list = []
        se_list = []
        stat_joint_list = []
        p_joint_list = []
        stat_per_hap_list = []
        p_per_hap_list = []
        stat_per_env_list = []
        p_per_env_list = []
        stat_homog_list = []
        p_homog_list = []
        stat_gxe_list = []
        p_gxe_list = []
        labels_list = []

        device = G.device

        for hb in hap_blocks:
            H = len(hb.haplotypes)
            block_ids.append(hb.block_id)
            chrs.append(hb.chr)
            starts.append(hb.start)
            ends.append(hb.end)
            n_variants_list.append(len(hb.variant_indices))
            n_haps_list.append(H)

            if H < 2:
                # Degenerate block
                beta_list.append(torch.tensor([]))
                se_list.append(torch.tensor([]))
                stat_joint_list.append(0.0)
                p_joint_list.append(1.0)
                stat_per_hap_list.append(torch.tensor([]))
                p_per_hap_list.append(torch.tensor([]))
                stat_per_env_list.append(torch.ones(E))
                p_per_env_list.append(torch.ones(E))
                stat_homog_list.append(torch.tensor([]))
                p_homog_list.append(torch.tensor([]))
                stat_gxe_list.append(0.0)
                p_gxe_list.append(1.0)
                labels_list.append([])
                continue

            D, test_labels = _drop_reference(hb)
            h = D.shape[1]  # H - 1
            labels_list.append(test_labels)

            # Rotate dosage into eigenspace
            D_rot = rotate(D, U)

            # GLS Wald
            beta_hap, se_hap, Var_hap, stat_j = _haplotype_gls_wald(
                D_rot, null_fit, E,
            )
            beta_list.append(beta_hap)
            se_list.append(se_hap)
            stat_joint_list.append(stat_j)
            p_joint_list.append(_chi2_sf(stat_j, df=h * E))

            beta_flat = beta_hap.reshape(-1)  # (h*E,)

            # Per-haplotype across envs: chi2(E)
            # beta is ordered as [hap0_env0, hap0_env1, ..., hap1_env0, ...]
            s_ph = torch.zeros(h)
            p_ph = torch.ones(h)
            for a in range(h):
                idx = list(range(a * E, (a + 1) * E))
                s_ph[a], p_ph[a] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_hap_list.append(s_ph)
            p_per_hap_list.append(p_ph)

            # Per-env across haplotypes: chi2(H-1)
            s_pe = torch.zeros(E)
            p_pe = torch.ones(E)
            for e in range(E):
                idx = [a * E + e for a in range(h)]
                s_pe[e], p_pe[e] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_env_list.append(s_pe)
            p_per_env_list.append(p_pe)

            # Homogeneity per haplotype: chi2(E-1) — does the haplotype effect
            # vary across environments?
            if E > 1:
                C_env = _build_contrast_matrix(E, device)  # (E-1, E)
                s_hom = torch.zeros(h)
                p_hom = torch.ones(h)
                for a in range(h):
                    idx = list(range(a * E, (a + 1) * E))
                    s_hom[a], p_hom[a] = _wald_contrast(
                        beta_flat, Var_hap, idx, C_env,
                    )
                stat_homog_list.append(s_hom)
                p_homog_list.append(p_hom)
            else:
                stat_homog_list.append(torch.zeros(h))
                p_homog_list.append(torch.ones(h))

            # Haplotype-GxE interaction: chi2((H-1)*(E-1))
            if E > 1:
                # Build block-diagonal contrast for all haplotypes
                # For each haplotype, apply the E-1 contrast; stack h blocks
                C_env = _build_contrast_matrix(E, device)
                gxe_dim = h * (E - 1)
                C_gxe = torch.zeros(gxe_dim, h * E, dtype=STAT_DTYPE, device=device)
                for a in range(h):
                    row_start = a * (E - 1)
                    col_start = a * E
                    C_gxe[row_start:row_start + (E - 1), col_start:col_start + E] = C_env
                Cb = C_gxe @ beta_flat
                CVCt = C_gxe @ Var_hap @ C_gxe.T
                CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
                stat_gxe = max(float((Cb * CVCt_inv_Cb).sum()), 0.0)
                p_gxe = _chi2_sf(stat_gxe, df=gxe_dim)
                stat_gxe_list.append(stat_gxe)
                p_gxe_list.append(p_gxe)
            else:
                stat_gxe_list.append(0.0)
                p_gxe_list.append(1.0)

        return HaplotypeMultiEnvResult(
            block_id=block_ids,
            chr=chrs,
            start=starts,
            end=ends,
            n_variants=n_variants_list,
            n_haplotypes=n_haps_list,
            beta=beta_list,
            se=se_list,
            stat_joint=torch.tensor(stat_joint_list, dtype=STAT_DTYPE),
            p_joint=torch.tensor(p_joint_list, dtype=STAT_DTYPE),
            stat_per_hap=stat_per_hap_list,
            p_per_hap=p_per_hap_list,
            stat_per_env=stat_per_env_list,
            p_per_env=p_per_env_list,
            stat_homogeneity=stat_homog_list,
            p_homogeneity=p_homog_list,
            stat_hap_gxe=torch.tensor(stat_gxe_list, dtype=STAT_DTYPE),
            p_hap_gxe=torch.tensor(p_gxe_list, dtype=STAT_DTYPE),
            haplotype_labels=labels_list,
            env_names=null_fit.env_names,
            n_obs=n,
        )


# ---------------------------------------------------------------------------
# Class 2: HaplotypeMultiTraitGWAS
# ---------------------------------------------------------------------------

class HaplotypeMultiTraitGWAS:
    """Haplotype association testing across d continuous traits.

    Delegates null model fitting to :class:`MultiTraitLMM`, then runs
    per-block GLS Wald tests on haplotype dosage matrices.
    """

    def __init__(
        self,
        method: str = "block",
        window_size: int = 5,
        step: int = 1,
        block_method: str = "gabriel",
        min_hap_freq: float = 0.01,
        max_haplotypes: int = 20,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        ploidy: int = 2,
    ):
        self.method = method
        self.window_size = window_size
        self.step = step
        self.block_method = block_method
        self.min_hap_freq = min_hap_freq
        self.max_haplotypes = max_haplotypes
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor,
        **kwargs,
    ) -> NullFit:
        """Fit multi-trait null model.

        Parameters
        ----------
        Y : (n, d) phenotype matrix (d >= 2 traits).
        X0 : (n, c) covariates.
        K : (n, n) kinship matrix.

        Returns
        -------
        NullFit with cached GLS quantities.
        """
        from .multi_trait_lmm import MultiTraitLMM

        mt = MultiTraitLMM(config=NumericalConfig())
        nf = mt.fit_null(Y, X0, K, **kwargs)
        d = Y.shape[1]
        nf.n_traits = d
        return nf

    def scan(
        self,
        G: Tensor,
        null_fit: NullFit,
        variant_pos: list[int] | None = None,
        variant_chr: list[str] | None = None,
        blocks=None,
        haplotypes: Tensor | None = None,
        trait_names: Optional[list[str]] = None,
        **block_kwargs,
    ) -> HaplotypeMultiTraitResult:
        """Run multi-trait haplotype scan.

        Parameters
        ----------
        G : (n, m) genotype dosage.
        null_fit : output of ``fit_null()``.
        variant_pos, variant_chr : per-variant metadata.
        blocks : pre-computed LD blocks.
        haplotypes : (n, ploidy, m) phased data.
        trait_names : optional trait labels.

        Returns
        -------
        HaplotypeMultiTraitResult
        """
        n, m = G.shape
        d = null_fit.n_traits
        U = null_fit.eigenvectors

        if variant_pos is None:
            variant_pos = list(range(m))
        if variant_chr is None:
            variant_chr = ["1"] * m

        scanner = HaplotypeGWAS(
            method=self.method,
            test="f_test",
            window_size=self.window_size,
            step=self.step,
            block_method=self.block_method,
            min_hap_freq=self.min_hap_freq,
            max_haplotypes=self.max_haplotypes,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            ploidy=self.ploidy,
        )
        hap_blocks = _construct_and_prepare_blocks(
            scanner, G, variant_pos, variant_chr,
            blocks, haplotypes, **block_kwargs,
        )

        block_ids = []
        chrs = []
        starts = []
        ends = []
        n_variants_list = []
        n_haps_list = []
        beta_list = []
        se_list = []
        stat_joint_list = []
        p_joint_list = []
        stat_per_hap_list = []
        p_per_hap_list = []
        stat_per_trait_list = []
        p_per_trait_list = []
        labels_list = []

        for hb in hap_blocks:
            H = len(hb.haplotypes)
            block_ids.append(hb.block_id)
            chrs.append(hb.chr)
            starts.append(hb.start)
            ends.append(hb.end)
            n_variants_list.append(len(hb.variant_indices))
            n_haps_list.append(H)

            if H < 2:
                beta_list.append(torch.tensor([]))
                se_list.append(torch.tensor([]))
                stat_joint_list.append(0.0)
                p_joint_list.append(1.0)
                stat_per_hap_list.append(torch.tensor([]))
                p_per_hap_list.append(torch.tensor([]))
                stat_per_trait_list.append(torch.ones(d))
                p_per_trait_list.append(torch.ones(d))
                labels_list.append([])
                continue

            D, test_labels = _drop_reference(hb)
            h = D.shape[1]
            labels_list.append(test_labels)

            D_rot = rotate(D, U)

            beta_hap, se_hap, Var_hap, stat_j = _haplotype_gls_wald(
                D_rot, null_fit, d,
            )
            beta_list.append(beta_hap)
            se_list.append(se_hap)
            stat_joint_list.append(stat_j)
            p_joint_list.append(_chi2_sf(stat_j, df=h * d))

            beta_flat = beta_hap.reshape(-1)  # (h*d,)

            # Per-haplotype across traits: chi2(d)
            s_ph = torch.zeros(h)
            p_ph = torch.ones(h)
            for a in range(h):
                idx = list(range(a * d, (a + 1) * d))
                s_ph[a], p_ph[a] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_hap_list.append(s_ph)
            p_per_hap_list.append(p_ph)

            # Per-trait across haplotypes: chi2(H-1)
            s_pt = torch.zeros(d)
            p_pt = torch.ones(d)
            for t in range(d):
                idx = [a * d + t for a in range(h)]
                s_pt[t], p_pt[t] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_trait_list.append(s_pt)
            p_per_trait_list.append(p_pt)

        return HaplotypeMultiTraitResult(
            block_id=block_ids,
            chr=chrs,
            start=starts,
            end=ends,
            n_variants=n_variants_list,
            n_haplotypes=n_haps_list,
            beta=beta_list,
            se=se_list,
            stat_joint=torch.tensor(stat_joint_list, dtype=STAT_DTYPE),
            p_joint=torch.tensor(p_joint_list, dtype=STAT_DTYPE),
            stat_per_hap=stat_per_hap_list,
            p_per_hap=p_per_hap_list,
            stat_per_trait=stat_per_trait_list,
            p_per_trait=p_per_trait_list,
            haplotype_labels=labels_list,
            trait_names=trait_names or [f"trait_{i}" for i in range(d)],
            n_obs=n,
        )


# ---------------------------------------------------------------------------
# Class 3: HaplotypeMTMETGWAS
# ---------------------------------------------------------------------------

class HaplotypeMTMETGWAS:
    """Haplotype association testing across d traits x E environments.

    Delegates null model fitting to :class:`MultiTraitMultiEnvLMM`, then
    runs per-block GLS Wald tests on haplotype dosage matrices.

    Column ordering is trait-major: column index = t*E + e, matching
    the ``MultiTraitMultiEnvLMM`` convention.
    """

    def __init__(
        self,
        method: str = "block",
        window_size: int = 5,
        step: int = 1,
        block_method: str = "gabriel",
        min_hap_freq: float = 0.01,
        max_haplotypes: int = 20,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        vg_structure: str = "separable",
        ploidy: int = 2,
    ):
        self.method = method
        self.window_size = window_size
        self.step = step
        self.block_method = block_method
        self.min_hap_freq = min_hap_freq
        self.max_haplotypes = max_haplotypes
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.vg_structure = vg_structure
        self.ploidy = ploidy

    def fit_null(
        self,
        Y_wide: Tensor,
        X0: Tensor,
        K: Tensor,
        d: int,
        E: int,
        *,
        trait_names: Optional[list[str]] = None,
        env_names: Optional[list[str]] = None,
        **kwargs,
    ) -> NullFit:
        """Fit MT-MET null model.

        Parameters
        ----------
        Y_wide : (n, d*E) phenotype matrix, trait-major column order.
        X0 : (n, c) covariates.
        K : (n, n) kinship matrix.
        d : number of traits.
        E : number of environments.
        trait_names, env_names : optional names.

        Returns
        -------
        NullFit with cached GLS quantities.
        """
        from .multi_trait_multi_env_lmm import MultiTraitMultiEnvLMM

        mtmet = MultiTraitMultiEnvLMM(
            config=NumericalConfig(),
            vg_structure=self.vg_structure,
        )
        nf = mtmet.fit_null(Y_wide, X0, K, n_traits=d, n_envs=E, **kwargs)
        nf.n_traits = d
        nf.n_env = E
        nf.trait_names = trait_names or [f"trait_{i}" for i in range(d)]
        nf.env_names = env_names or [f"env_{i}" for i in range(E)]
        return nf

    def scan(
        self,
        G: Tensor,
        null_fit: NullFit,
        variant_pos: list[int] | None = None,
        variant_chr: list[str] | None = None,
        blocks=None,
        haplotypes: Tensor | None = None,
        **block_kwargs,
    ) -> HaplotypeMTMETResult:
        """Run MT-MET haplotype scan.

        Parameters
        ----------
        G : (n, m) genotype dosage.
        null_fit : output of ``fit_null()``.
        variant_pos, variant_chr : per-variant metadata.
        blocks : pre-computed LD blocks.
        haplotypes : (n, ploidy, m) phased data.

        Returns
        -------
        HaplotypeMTMETResult
        """
        n, m = G.shape
        d = null_fit.n_traits
        E = null_fit.n_env
        dE = d * E
        U = null_fit.eigenvectors

        if variant_pos is None:
            variant_pos = list(range(m))
        if variant_chr is None:
            variant_chr = ["1"] * m

        scanner = HaplotypeGWAS(
            method=self.method,
            test="f_test",
            window_size=self.window_size,
            step=self.step,
            block_method=self.block_method,
            min_hap_freq=self.min_hap_freq,
            max_haplotypes=self.max_haplotypes,
            em_max_iter=self.em_max_iter,
            em_tol=self.em_tol,
            ploidy=self.ploidy,
        )
        hap_blocks = _construct_and_prepare_blocks(
            scanner, G, variant_pos, variant_chr,
            blocks, haplotypes, **block_kwargs,
        )

        block_ids = []
        chrs_list = []
        starts = []
        ends = []
        n_variants_list = []
        n_haps_list = []
        beta_list = []
        se_list = []
        stat_joint_list = []
        p_joint_list = []
        stat_per_hap_list = []
        p_per_hap_list = []
        stat_per_trait_list = []
        p_per_trait_list = []
        stat_per_env_list = []
        p_per_env_list = []
        stat_hap_gxe_list = []
        p_hap_gxe_list = []
        labels_list = []

        device = G.device

        for hb in hap_blocks:
            H = len(hb.haplotypes)
            block_ids.append(hb.block_id)
            chrs_list.append(hb.chr)
            starts.append(hb.start)
            ends.append(hb.end)
            n_variants_list.append(len(hb.variant_indices))
            n_haps_list.append(H)

            if H < 2:
                beta_list.append(torch.tensor([]))
                se_list.append(torch.tensor([]))
                stat_joint_list.append(0.0)
                p_joint_list.append(1.0)
                stat_per_hap_list.append(torch.tensor([]))
                p_per_hap_list.append(torch.tensor([]))
                stat_per_trait_list.append(torch.ones(d))
                p_per_trait_list.append(torch.ones(d))
                stat_per_env_list.append(torch.ones(E))
                p_per_env_list.append(torch.ones(E))
                stat_hap_gxe_list.append(torch.tensor([]))
                p_hap_gxe_list.append(torch.tensor([]))
                labels_list.append([])
                continue

            D, test_labels = _drop_reference(hb)
            h = D.shape[1]
            labels_list.append(test_labels)

            D_rot = rotate(D, U)

            beta_hap, se_hap, Var_hap, stat_j = _haplotype_gls_wald(
                D_rot, null_fit, dE,
            )
            # beta_hap is (h, dE), reshape to (h, d, E)
            beta_shaped = beta_hap.reshape(h, d, E)
            se_shaped = se_hap.reshape(h, d, E)
            beta_list.append(beta_shaped)
            se_list.append(se_shaped)
            stat_joint_list.append(stat_j)
            p_joint_list.append(_chi2_sf(stat_j, df=h * dE))

            beta_flat = beta_hap.reshape(-1)  # (h*dE,)

            # Per-haplotype across all trait-envs: chi2(dE)
            s_ph = torch.zeros(h)
            p_ph = torch.ones(h)
            for a in range(h):
                idx = list(range(a * dE, (a + 1) * dE))
                s_ph[a], p_ph[a] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_hap_list.append(s_ph)
            p_per_hap_list.append(p_ph)

            # Per-trait across envs and haplotypes: chi2((H-1)*E)
            # For trait t, indices are: a*dE + t*E + e for a in range(h), e in range(E)
            s_pt = torch.zeros(d)
            p_pt = torch.ones(d)
            for t in range(d):
                idx = []
                for a in range(h):
                    for e in range(E):
                        idx.append(a * dE + t * E + e)
                s_pt[t], p_pt[t] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_trait_list.append(s_pt)
            p_per_trait_list.append(p_pt)

            # Per-env across traits and haplotypes: chi2((H-1)*d)
            # For env e, indices are: a*dE + t*E + e for a in range(h), t in range(d)
            s_pe = torch.zeros(E)
            p_pe = torch.ones(E)
            for e in range(E):
                idx = []
                for a in range(h):
                    for t in range(d):
                        idx.append(a * dE + t * E + e)
                s_pe[e], p_pe[e] = _wald_subblock(beta_flat, Var_hap, idx)
            stat_per_env_list.append(s_pe)
            p_per_env_list.append(p_pe)

            # Per-haplotype GxE: chi2(E-1) for each haplotype
            if E > 1:
                C_env = _build_contrast_matrix(E, device)
                s_gxe = torch.zeros(h)
                p_gxe = torch.ones(h)
                for a in range(h):
                    # For haplotype a, across all traits, test env homogeneity
                    # Build block-diagonal contrast: d blocks of (E-1, E)
                    gxe_rows = d * (E - 1)
                    C_block = torch.zeros(gxe_rows, dE, dtype=STAT_DTYPE, device=device)
                    for t in range(d):
                        r_start = t * (E - 1)
                        c_start = t * E
                        C_block[r_start:r_start + (E - 1), c_start:c_start + E] = C_env

                    # Extract this haplotype's dE-dimensional sub-block
                    hap_idx = list(range(a * dE, (a + 1) * dE))
                    idx_t = torch.tensor(hap_idx, dtype=torch.long, device=device)
                    beta_sub = beta_flat[idx_t]
                    Var_sub = Var_hap[idx_t][:, idx_t]

                    Cb = C_block @ beta_sub
                    CVCt = C_block @ Var_sub @ C_block.T
                    CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
                    s_gxe[a] = max(float((Cb * CVCt_inv_Cb).sum()), 0.0)
                    p_gxe[a] = _chi2_sf(s_gxe[a].item(), df=gxe_rows)
                stat_hap_gxe_list.append(s_gxe)
                p_hap_gxe_list.append(p_gxe)
            else:
                stat_hap_gxe_list.append(torch.zeros(h))
                p_hap_gxe_list.append(torch.ones(h))

        return HaplotypeMTMETResult(
            block_id=block_ids,
            chr=chrs_list,
            start=starts,
            end=ends,
            n_variants=n_variants_list,
            n_haplotypes=n_haps_list,
            beta=beta_list,
            se=se_list,
            stat_joint=torch.tensor(stat_joint_list, dtype=STAT_DTYPE),
            p_joint=torch.tensor(p_joint_list, dtype=STAT_DTYPE),
            stat_per_hap=stat_per_hap_list,
            p_per_hap=p_per_hap_list,
            stat_per_trait=stat_per_trait_list,
            p_per_trait=p_per_trait_list,
            stat_per_env=stat_per_env_list,
            p_per_env=p_per_env_list,
            stat_hap_gxe=stat_hap_gxe_list,
            p_hap_gxe=p_hap_gxe_list,
            haplotype_labels=labels_list,
            trait_names=getattr(null_fit, 'trait_names', None),
            env_names=getattr(null_fit, 'env_names', None),
            n_obs=n,
        )
