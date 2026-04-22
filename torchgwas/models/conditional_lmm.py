"""LD-Conditional Score GWAS: block-aware conditional analysis.

Implements GCTA-COJO-style conditional testing within LD blocks:
1. Run marginal LMM scan (standard Wald/score test)
2. Detect LD blocks via existing 13-method infrastructure
3. Within each block: condition on lead SNP(s) by augmenting X0
4. Compute conditional p-values and persistence metrics

Conditional persistence flags SNPs that retain signal after conditioning,
distinguishing likely causal variants from LD proxies (Yang et al. 2012).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import rotate
from .base import NullFit, ScanResult, VariantMeta
from .single_trait_lmm import SingleTraitLMM, _f_sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConditionalScanResult
# ---------------------------------------------------------------------------

@dataclass
class ConditionalScanResult:
    """Extended scan result with both marginal and conditional statistics."""

    marginal: ScanResult  # standard marginal results

    # Conditional statistics (same shape as marginal, m SNPs)
    conditional_beta: Tensor  # (m,)
    conditional_se: Tensor  # (m,)
    conditional_stat: Tensor  # (m,)
    conditional_p: Tensor  # (m,)

    # Diagnostics
    persistence: Tensor  # (m,) bool — retains signal after conditioning
    ld_block_id: list[str]  # (m,) block assignment ("none" if unblocked)
    r2_to_lead: Tensor  # (m,) r² to block lead SNP (0 for unblocked)
    lead_snp: list[str]  # (m,) lead SNP ID for this SNP's block

    def __len__(self) -> int:
        return len(self.marginal)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conditional_wald_batch(
    G_rot: Tensor,
    Y_rot: Tensor,
    X0_rot: Tensor,
    evals: Tensor,
    sig2_g: float,
    sig2_e: float,
    test_indices: list[int],
    cond_indices: list[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute conditional Wald test for test_indices, conditioning on cond_indices.

    Parameters
    ----------
    G_rot : (n, m_block) rotated genotypes for the block.
    Y_rot : (n,) rotated phenotype.
    X0_rot : (n, c) rotated null covariates.
    evals : (n,) eigenvalues.
    sig2_g, sig2_e : variance components.
    test_indices : local indices in G_rot to test.
    cond_indices : local indices in G_rot to condition on.

    Returns
    -------
    beta, se, stat, p : each (len(test_indices),)
    """
    n = G_rot.shape[0]
    device = G_rot.device

    lam = sig2_g / max(sig2_e, 1e-20)
    H_inv = 1.0 / (evals * lam + 1.0)

    # Augmented covariate matrix: [X0, conditioning SNPs]
    G_cond = G_rot[:, cond_indices]  # (n, n_cond)
    X_aug = torch.cat([X0_rot, G_cond], dim=1)  # (n, c + n_cond)

    # Weighted augmented normal equations
    wX_aug = H_inv.unsqueeze(1) * X_aug  # (n, c+n_cond)
    M_aug = X_aug.T @ wX_aug  # (c+n_cond, c+n_cond)

    # Regularize for numerical stability (collinear conditioning SNPs)
    M_aug = M_aug + 1e-8 * torch.eye(M_aug.shape[0], dtype=M_aug.dtype, device=device)

    # P_aug y = H_inv y - H_inv X_aug M_aug^{-1} X_aug' H_inv y
    wY = H_inv * Y_rot
    XtwY = X_aug.T @ wY
    M_inv_XtwY = torch.linalg.solve(M_aug, XtwY)
    Py = wY - wX_aug @ M_inv_XtwY  # (n,)

    # Test SNPs
    G_test = G_rot[:, test_indices]  # (n, n_test)

    # g' P_aug y
    gPy = (G_test * Py.unsqueeze(1)).sum(dim=0)  # (n_test,)

    # g' P_aug g via Schur complement
    wG_test = H_inv.unsqueeze(1) * G_test
    gWg = (G_test * wG_test).sum(dim=0)  # (n_test,)
    gWX_aug = G_test.T @ wX_aug  # (n_test, c+n_cond)
    M_inv_gWXt = torch.linalg.solve(M_aug, gWX_aug.T)  # (c+n_cond, n_test)
    S = gWg - (gWX_aug * M_inv_gWXt.T).sum(dim=1)  # (n_test,)
    S_safe = S.clamp(min=1e-20)

    beta = gPy / S_safe
    var_beta = sig2_e / S_safe
    se = var_beta.sqrt()
    stat = beta ** 2 / var_beta

    c_aug = X_aug.shape[1]
    df2 = n - c_aug - 1
    p = _f_sf(stat, df1=1, df2=max(df2, 1))

    return beta, se, stat, p


def _stepwise_leads(
    marginal_p: Tensor,
    block_indices: list[int],
    max_leads: int,
    sig_threshold: float,
) -> list[int]:
    """Select lead SNPs in a block via greedy stepwise (GCTA-COJO style).

    Returns local indices (within block_indices) of lead SNPs.
    """
    p_block = marginal_p[block_indices]

    leads = []
    remaining = list(range(len(block_indices)))

    for _ in range(max_leads):
        if not remaining:
            break
        # Find most significant remaining SNP
        p_remaining = p_block[remaining]
        best_local = remaining[p_remaining.argmin().item()]
        if p_block[best_local] >= sig_threshold:
            break
        leads.append(best_local)
        remaining.remove(best_local)

    return leads


# ---------------------------------------------------------------------------
# ConditionalLMM
# ---------------------------------------------------------------------------

class ConditionalLMM:
    """LD-block-conditional LMM for distinguishing causal from proxy SNPs.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    config : NumericalConfig, optional
    ld_method : str
        LD block detection method (default "r2", fast and simple).
    max_kb : float
        Max distance for LD block detection.
    persistence_ratio : float
        A SNP is persistent if its conditional -log10(p) exceeds this
        fraction of its marginal -log10(p).
    sig_threshold : float
        Only evaluate persistence for SNPs below this marginal p-value.
    max_conditioning : int
        Max lead SNPs to condition on per block.
    p3d : bool
        P3D mode for the underlying LMM.
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        ld_method: str = "r2",
        max_kb: float = 200.0,
        persistence_ratio: float = 0.5,
        sig_threshold: float = 5e-8,
        max_conditioning: int = 5,
        p3d: bool = True,
    ) -> None:
        self.config = config or NumericalConfig()
        self.ld_method = ld_method
        self.max_kb = max_kb
        self.persistence_ratio = persistence_ratio
        self.sig_threshold = sig_threshold
        self.max_conditioning = max_conditioning
        self._lmm = SingleTraitLMM(config=self.config, p3d=p3d)

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model (delegates to SingleTraitLMM)."""
        return self._lmm.fit_null(Y, X0, K=K, **kwargs)

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score a genotype chunk with marginal + conditional tests.

        Returns a standard ScanResult (marginal) with a ``_conditional``
        attribute holding the full ConditionalScanResult.
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        device = G_chunk.device

        # Handle empty chunk
        if m == 0:
            empty = torch.zeros(0, dtype=STAT_DTYPE, device=device)
            result = ScanResult(
                chr=[], pos=[], snp=[], a1=[], a2=[],
                af=empty, beta=empty, se=empty,
                stat=empty, p=empty, test=test,
            )
            result._conditional = ConditionalScanResult(
                marginal=result,
                conditional_beta=empty, conditional_se=empty,
                conditional_stat=empty, conditional_p=empty,
                persistence=torch.zeros(0, dtype=torch.bool, device=device),
                ld_block_id=[], r2_to_lead=empty, lead_snp=[],
            )
            return result

        # --- 1. Marginal scan ---
        marginal = self._lmm.score_chunk(G_chunk, null_fit, variant_meta, test)

        # --- 2. Detect LD blocks ---
        from ..ld import detect_blocks
        try:
            blocks = detect_blocks(
                G_chunk,
                variant_pos=variant_meta.pos,
                variant_chr=variant_meta.chr,
                variant_ids=variant_meta.snp,
                method=self.ld_method,
                max_kb=self.max_kb,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("LD block detection failed (%s), skipping conditioning: %s",
                          self.ld_method, e)
            blocks = []

        # --- 3. Prepare conditional results ---
        cond_beta = marginal.beta.clone()
        cond_se = marginal.se.clone()
        cond_stat = marginal.stat.clone()
        cond_p = marginal.p.clone()
        block_ids = ["none"] * m
        r2_to_lead = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        lead_snps = ["none"] * m

        # Rotation setup
        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)
        Y_rot = null_fit.Y_rot.squeeze()
        X0_rot = null_fit.X0_rot
        evals = null_fit.eigenvalues
        sig2_g = null_fit.sig2_g
        sig2_e = null_fit.sig2_e

        # --- 4. Per-block conditional testing ---
        from ..ld._pairwise import compute_r2_matrix

        for block in blocks:
            bidx = block.variant_indices
            if len(bidx) < 2:
                # Single-SNP blocks: no conditioning needed
                for i in bidx:
                    block_ids[i] = block.region.region_id
                continue

            # Select lead SNPs
            leads_local = _stepwise_leads(
                marginal.p, bidx, self.max_conditioning, self.sig_threshold,
            )

            if not leads_local:
                # No significant leads — mark block but keep marginal results
                for i in bidx:
                    block_ids[i] = block.region.region_id
                continue

            # Lead indices in the full G_chunk
            lead_global = [bidx[j] for j in leads_local]
            lead_snp_id = variant_meta.snp[lead_global[0]]

            # Block genotype submatrix for r² computation
            G_block = G_chunk[:, bidx]
            r2_mat = compute_r2_matrix(G_block)

            # Conditional Wald test for non-lead SNPs
            # Map: local indices within block
            non_lead_local = [j for j in range(len(bidx)) if j not in leads_local]

            if non_lead_local:
                beta_c, se_c, stat_c, p_c = _conditional_wald_batch(
                    G_rot[:, bidx],
                    Y_rot, X0_rot, evals, sig2_g, sig2_e,
                    test_indices=non_lead_local,
                    cond_indices=leads_local,
                )

                for k, local_j in enumerate(non_lead_local):
                    global_j = bidx[local_j]
                    cond_beta[global_j] = beta_c[k]
                    cond_se[global_j] = se_c[k]
                    cond_stat[global_j] = stat_c[k]
                    cond_p[global_j] = p_c[k]

            # Assign block metadata
            for j, global_idx in enumerate(bidx):
                block_ids[global_idx] = block.region.region_id
                lead_snps[global_idx] = lead_snp_id
                # r² to primary lead (leads_local[0])
                r2_to_lead[global_idx] = r2_mat[j, leads_local[0]].item()

        # --- 5. Persistence metric ---
        marginal_logp = -torch.log10(marginal.p.clamp(min=1e-300))
        cond_logp = -torch.log10(cond_p.clamp(min=1e-300))

        is_significant = marginal.p < self.sig_threshold
        retains_signal = cond_logp > self.persistence_ratio * marginal_logp
        persistence = is_significant & retains_signal

        # --- 6. Build result ---
        # Return marginal as primary ScanResult (compatible with UnifiedScanner)
        result = ScanResult(
            chr=marginal.chr, pos=marginal.pos, snp=marginal.snp,
            a1=marginal.a1, a2=marginal.a2, af=marginal.af,
            beta=marginal.beta, se=marginal.se,
            stat=marginal.stat, p=marginal.p,
            test=marginal.test, n_obs=marginal.n_obs,
            inference_type="marginal",
        )

        result._conditional = ConditionalScanResult(
            marginal=marginal,
            conditional_beta=cond_beta,
            conditional_se=cond_se,
            conditional_stat=cond_stat,
            conditional_p=cond_p,
            persistence=persistence,
            ld_block_id=block_ids,
            r2_to_lead=r2_to_lead,
            lead_snp=lead_snps,
        )

        return result
