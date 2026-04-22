"""Within-Family / Family-Aware LMM for confounding diagnostics.

Implements the within-family demeaning approach (Young et al. 2022, Nat Genet):
1. Subtract family means from phenotypes, covariates, and genotypes.
2. Run standard LMM on original data and within-family LMM on demeaned data.
3. Compute per-SNP attenuation metrics: β_within / β_standard.

SNPs with large attenuation (ratio ≪ 1) are flagged as confounded by
population structure or indirect genetic effects.  This provides the
strongest causal credibility assessment for genetic associations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from .base import NullFit, ScanResult, VariantMeta
from .single_trait_lmm import SingleTraitLMM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WithinFamilyNullFit — caches both standard and within-family null fits
# ---------------------------------------------------------------------------

@dataclass
class WithinFamilyNullFit:
    """Cached null-model quantities for dual standard + within-family scan."""

    standard_null_fit: NullFit
    within_null_fit: NullFit

    # Family structure (for demeaning genotypes during scan)
    family_indices: list[Tensor]  # per-family sample index lists
    family_sizes: Tensor  # (n_families,) number of members per family
    n_families: int
    n_valid_samples: int  # samples in families with size >= min_family_size

    # Forward key fields from standard null fit for NullFit compatibility
    @property
    def sig2_g(self) -> float:
        return self.standard_null_fit.sig2_g

    @property
    def sig2_e(self) -> float:
        return self.standard_null_fit.sig2_e

    @property
    def log_likelihood(self) -> float:
        return self.standard_null_fit.log_likelihood

    @property
    def converged(self) -> bool:
        return (self.standard_null_fit.converged
                and self.within_null_fit.converged)


# ---------------------------------------------------------------------------
# Helpers: family parsing and demeaning
# ---------------------------------------------------------------------------

def _parse_family_ids(
    family_ids: Tensor,
    min_family_size: int = 2,
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Group samples by family and filter small families.

    Parameters
    ----------
    family_ids : (n,) integer tensor of family assignments.
    min_family_size : minimum members for inclusion.

    Returns
    -------
    family_indices : list of (n_f,) index tensors, one per valid family.
    family_sizes : (n_valid_families,) tensor.
    valid_mask : (n,) boolean mask of samples in valid families.
    """
    unique_fids, inverse = torch.unique(family_ids, return_inverse=True)
    n_families_total = unique_fids.shape[0]

    family_indices = []
    family_sizes = []
    valid_mask = torch.zeros(family_ids.shape[0], dtype=torch.bool)

    for k in range(n_families_total):
        idx = (inverse == k).nonzero(as_tuple=True)[0]
        if idx.shape[0] >= min_family_size:
            family_indices.append(idx)
            family_sizes.append(idx.shape[0])
            valid_mask[idx] = True

    if len(family_indices) < 2:
        raise ValueError(
            f"Need at least 2 families with >= {min_family_size} members, "
            f"but found {len(family_indices)}."
        )

    return (
        family_indices,
        torch.tensor(family_sizes, dtype=torch.long),
        valid_mask,
    )


def _demean_within_families(
    X: Tensor,
    family_indices: list[Tensor],
) -> Tensor:
    """Subtract per-family means from X.

    Parameters
    ----------
    X : (n, ...) tensor to demean.
    family_indices : list of index tensors, one per family.

    Returns
    -------
    X_demean : (n, ...) demeaned tensor. Samples not in any family
        retain their original values (should be masked out later).
    """
    X_demean = X.clone()
    for idx in family_indices:
        family_mean = X[idx].mean(dim=0)
        X_demean[idx] = X[idx] - family_mean
    return X_demean


# ---------------------------------------------------------------------------
# WithinFamilyLMM model
# ---------------------------------------------------------------------------

class WithinFamilyLMM:
    """Within-family / family-aware LMM for confounding diagnostics.

    Runs two parallel LMM scans — standard and within-family (demeaned) — and
    computes per-SNP attenuation metrics.  Conforms to the BaseModel protocol.

    Parameters
    ----------
    config : NumericalConfig, optional
        Numerical configuration for REML fitting.
    min_family_size : int
        Minimum number of members per family for inclusion in the
        within-family scan.  Families below this are excluded from the
        within-family analysis (but included in the standard scan).
    p3d : bool
        If True (default), use P3D=TRUE mode (fixed variance components
        across all SNPs).
    confound_threshold : float
        Absolute deviation of attenuation from 1.0 that triggers the
        confounding flag.  Default 0.5.
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        min_family_size: int = 2,
        p3d: bool = True,
        confound_threshold: float = 0.5,
    ) -> None:
        self.config = config or NumericalConfig()
        self.min_family_size = min_family_size
        self.p3d = p3d
        self.confound_threshold = confound_threshold
        self._standard_lmm = SingleTraitLMM(config=self.config, p3d=p3d)
        self._within_lmm = SingleTraitLMM(config=self.config, p3d=p3d)

    # ------------------------------------------------------------------
    # fit_null
    # ------------------------------------------------------------------

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        *,
        family_ids: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null models for both standard and within-family scans.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype.
        X0 : (n, c) — covariates (intercept as first column).
        K : (n, n) — GRM / kinship matrix.
        family_ids : (n,) integer tensor — family assignments per sample.

        Returns
        -------
        NullFit (actually WithinFamilyNullFit) with both cached null fits.
        """
        if K is None:
            raise ValueError("WithinFamilyLMM requires a kinship matrix K.")
        if family_ids is None:
            raise ValueError(
                "WithinFamilyLMM requires family_ids. "
                "Pass a (n,) integer tensor of family assignments."
            )

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        family_ids = family_ids.long()

        n = Y.shape[0]
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        # --- Parse families ---
        family_indices, family_sizes, valid_mask = _parse_family_ids(
            family_ids, self.min_family_size,
        )
        n_valid = valid_mask.sum().item()
        n_families = len(family_indices)
        logger.info(
            "Family structure: %d families (min_size=%d), %d/%d samples valid",
            n_families, self.min_family_size, n_valid, n,
        )

        # --- Standard null fit (all samples) ---
        logger.info("Fitting standard null model...")
        standard_nf = self._standard_lmm.fit_null(Y, X0, K=K, **kwargs)

        # --- Within-family null fit (demeaned, valid samples only) ---
        Y_demean = _demean_within_families(Y.unsqueeze(1), family_indices).squeeze(1)
        X0_demean = _demean_within_families(X0, family_indices)

        # Drop intercept column (all-ones becomes all-zeros after demeaning).
        # If X0 has more than 1 column, keep the remaining covariates.
        if X0.shape[1] > 1:
            # Check which columns became near-zero (i.e., were constant)
            # Use only valid samples (in families) for variance check
            col_var = X0_demean[valid_mask].var(dim=0)
            keep_cols = col_var > 1e-10
            if keep_cols.sum() == 0:
                # All covariates were constant → use dummy column
                X0_wf = torch.ones(n, 1, dtype=STAT_DTYPE, device=Y.device)
            else:
                X0_wf = X0_demean[:, keep_cols]
        else:
            # Only intercept column → after demeaning it's zeros → use dummy
            X0_wf = torch.ones(n, 1, dtype=STAT_DTYPE, device=Y.device)

        logger.info("Fitting within-family null model (%d covariates)...", X0_wf.shape[1])
        within_nf = self._within_lmm.fit_null(Y_demean, X0_wf, K=K, **kwargs)

        wf_null = WithinFamilyNullFit(
            standard_null_fit=standard_nf,
            within_null_fit=within_nf,
            family_indices=family_indices,
            family_sizes=family_sizes,
            n_families=n_families,
            n_valid_samples=n_valid,
        )

        logger.info(
            "Standard: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f | "
            "Within-family: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
            standard_nf.sig2_g, standard_nf.sig2_e,
            standard_nf.sig2_g / (standard_nf.sig2_g + standard_nf.sig2_e),
            within_nf.sig2_g, within_nf.sig2_e,
            within_nf.sig2_g / (within_nf.sig2_g + within_nf.sig2_e),
        )

        return wf_null

    # ------------------------------------------------------------------
    # score_chunk
    # ------------------------------------------------------------------

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score a genotype chunk, returning standard + within-family results.

        The returned ScanResult has the standard (population-level) results
        in its primary fields (beta, se, stat, p).  Within-family results
        and attenuation metrics are stored as extra attributes:
        ``_wf_beta``, ``_wf_se``, ``_wf_stat``, ``_wf_p``,
        ``_attenuation``, ``_confound_flag``.
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        # Handle empty chunk
        if m == 0:
            empty = torch.zeros(0, dtype=STAT_DTYPE)
            result = ScanResult(
                chr=[], pos=[], snp=[], a1=[], a2=[],
                af=empty, beta=empty, se=empty,
                stat=empty, p=empty, test=test,
            )
            result._wf_beta = empty
            result._wf_se = empty
            result._wf_stat = empty
            result._wf_p = empty
            result._attenuation = empty
            result._confound_flag = torch.zeros(0, dtype=torch.bool)
            return result

        wf_null: WithinFamilyNullFit = null_fit

        # --- Standard scan ---
        std_result = self._standard_lmm.score_chunk(
            G_chunk, wf_null.standard_null_fit, variant_meta, test,
        )

        # --- Within-family scan (demean genotypes) ---
        G_demean = _demean_within_families(G_chunk, wf_null.family_indices)
        wf_result = self._within_lmm.score_chunk(
            G_demean, wf_null.within_null_fit, variant_meta, test,
        )

        # --- Attenuation metric ---
        # attenuation = β_within / β_standard
        eps = 1e-10
        beta_std = std_result.beta
        beta_wf = wf_result.beta
        attenuation = torch.where(
            beta_std.abs() > eps,
            beta_wf / beta_std,
            torch.ones_like(beta_std),  # undefined → 1.0 (no evidence)
        )

        confound_flag = (attenuation - 1.0).abs() > self.confound_threshold

        # --- Build result (standard results as primary) ---
        result = ScanResult(
            chr=std_result.chr,
            pos=std_result.pos,
            snp=std_result.snp,
            a1=std_result.a1,
            a2=std_result.a2,
            af=std_result.af,
            beta=std_result.beta,
            se=std_result.se,
            stat=std_result.stat,
            p=std_result.p,
            test=std_result.test,
            n_obs=std_result.n_obs,
            inference_type="marginal",
        )

        # Attach within-family results as extra attributes
        result._wf_beta = beta_wf
        result._wf_se = wf_result.se
        result._wf_stat = wf_result.stat
        result._wf_p = wf_result.p
        result._attenuation = attenuation
        result._confound_flag = confound_flag

        return result
