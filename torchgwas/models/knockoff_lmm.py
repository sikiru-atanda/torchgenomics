"""Knockoff Mixed Model: FDR-controlled GWAS via group knockoffs.

Implements the knockoff filter (Sesia et al. 2020 JASA; KnockoffGWAS)
under the LMM framework.  Provides explicit FDR control at the locus
(LD block) level, unlike standard p-value thresholding + clumping.

Pipeline:
  1. Partition variants into LD blocks (any of 13 methods)
  2. Generate knockoff genotypes per block (equicorrelated method)
  3. Fit LMM null model once, scan originals and knockoffs
  4. Compute block-level importance statistics W_b
  5. Apply knockoff+ filter at target FDR
  6. Report selected blocks and per-SNP annotations

This is a whole-genome procedure — requires all genotypes at once
(NOT compatible with chunk-by-chunk UnifiedScanner streaming).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KnockoffResult:
    """Output of KnockoffLMM.run().

    Contains per-block importance statistics, the knockoff+ filter results,
    and per-SNP association statistics from both original and knockoff scans.
    """

    # Per-block results
    block_indices: list[list[int]]  # variant indices per block
    W_stat: Tensor                  # (B,) importance statistics
    selected_blocks: list[int]      # indices of blocks passing filter
    threshold: float                # knockoff+ threshold tau

    # Per-SNP (all m variants)
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor              # (m,)
    beta: Tensor            # (m,) original scan effect sizes
    se: Tensor              # (m,) original scan SEs
    stat: Tensor            # (m,) original scan chi-sq statistics
    p: Tensor               # (m,) original scan p-values
    beta_knockoff: Tensor   # (m,) knockoff scan effects
    stat_knockoff: Tensor   # (m,) knockoff scan statistics
    is_selected: Tensor     # (m,) bool — in a selected block

    # Metadata
    target_fdr: float
    n_blocks: int
    n_selected: int
    ld_method: str
    knockoff_method: str
    aggregation: str        # "max_stat" or "sum_sq"


# ---------------------------------------------------------------------------
# Helper: correlation matrix
# ---------------------------------------------------------------------------

def _compute_correlation_matrix(G_block: Tensor) -> Tensor:
    """Pearson correlation matrix for a genotype block.

    Parameters
    ----------
    G_block : (n, p) float64

    Returns
    -------
    Sigma : (p, p) float64 — correlation matrix with unit diagonal
    """
    G = G_block.to(STAT_DTYPE)
    n, p = G.shape

    # Center columns
    G_centered = G - G.mean(dim=0, keepdim=True)

    # Standard deviation per column
    std = G_centered.std(dim=0, correction=1)
    # Guard against zero-variance columns
    std = std.clamp(min=1e-12)

    # Standardize
    G_std = G_centered / std.unsqueeze(0)

    # Correlation = (1/(n-1)) * G_std^T @ G_std
    Sigma = (G_std.T @ G_std) / (n - 1)

    # Force exact symmetry and unit diagonal
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma.fill_diagonal_(1.0)

    return Sigma


# ---------------------------------------------------------------------------
# Helper: equicorrelated knockoff generation
# ---------------------------------------------------------------------------

def _generate_knockoffs_equicorrelated(
    G_block: Tensor,
    Sigma: Tensor,
    seed: int | None = None,
) -> Tensor:
    """Generate knockoff genotypes using the equicorrelated method.

    For block with p variants:
      s = min(2 * lambda_min(Sigma), 1)
      G_tilde = G - G @ Sigma_inv @ diag(s) + C @ Z
      where C = chol(2*s*I - s^2 * Sigma_inv)

    Parameters
    ----------
    G_block : (n, p) genotype submatrix
    Sigma : (p, p) correlation matrix
    seed : optional random seed

    Returns
    -------
    G_tilde : (n, p) knockoff genotypes
    """
    G = G_block.to(STAT_DTYPE)
    n, p = G.shape
    device = G.device

    if p == 1:
        # Single-SNP block: knockoff is just noise with same mean/var
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        mu = G.mean(dim=0, keepdim=True)
        std = G.std(dim=0, keepdim=True).clamp(min=1e-12)
        return mu + std * torch.randn(n, 1, dtype=STAT_DTYPE, device=device,
                                      generator=gen)

    # Add jitter for numerical stability
    jitter = 1e-6 * torch.eye(p, dtype=STAT_DTYPE, device=device)
    Sigma_reg = Sigma + jitter

    # Compute minimum eigenvalue
    eigvals = torch.linalg.eigvalsh(Sigma_reg)
    lambda_min = eigvals[0].item()

    # Equicorrelated parameter: s = min(2 * lambda_min, 1)
    s = min(2.0 * max(lambda_min, 0.0), 1.0)
    # Ensure s > 0 for valid knockoffs
    s = max(s, 1e-6)

    # Sigma_inv
    Sigma_inv = torch.linalg.inv(Sigma_reg)

    # G_tilde = G - G @ Sigma_inv @ diag(s) + C @ Z
    # Since s is scalar (equicorrelated): diag(s) = s * I
    # Term 1: G - s * G @ Sigma_inv
    mean_shift = G - s * (G @ Sigma_inv)

    # C = chol(2*s*I - s^2 * Sigma_inv)
    # This is the covariance of the knockoff noise
    cov_noise = 2.0 * s * torch.eye(p, dtype=STAT_DTYPE, device=device) - s**2 * Sigma_inv
    # Ensure PD with small jitter
    cov_noise = (cov_noise + cov_noise.T) / 2.0
    cov_noise += 1e-6 * torch.eye(p, dtype=STAT_DTYPE, device=device)

    # Cholesky for sampling
    try:
        C = torch.linalg.cholesky(cov_noise)
    except torch.linalg.LinAlgError:
        # Fallback: eigenclamp to make PD
        eigvals_c, eigvecs_c = torch.linalg.eigh(cov_noise)
        eigvals_c = eigvals_c.clamp(min=1e-6)
        cov_noise = eigvecs_c @ torch.diag(eigvals_c) @ eigvecs_c.T
        C = torch.linalg.cholesky(cov_noise)

    # Sample noise
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    Z = torch.randn(n, p, dtype=STAT_DTYPE, device=device, generator=gen)

    G_tilde = mean_shift + Z @ C.T

    return G_tilde


# ---------------------------------------------------------------------------
# Helper: block-level importance statistics
# ---------------------------------------------------------------------------

def _knockoff_importance(
    stat_orig: Tensor,
    stat_knock: Tensor,
    beta_orig: Tensor,
    beta_knock: Tensor,
    block_indices: list[list[int]],
    aggregation: str = "max_stat",
) -> Tensor:
    """Compute block-level importance statistics W_b.

    For each block b:
      Z_j = sign(beta_j) * sqrt(stat_j)  (signed Z-score)
      T(Z) depends on aggregation method
      W_b = T(Z_orig_b) - T(Z_knock_b)

    Parameters
    ----------
    stat_orig : (m,) original scan chi-sq statistics
    stat_knock : (m,) knockoff scan chi-sq statistics
    beta_orig : (m,) original scan effect sizes
    beta_knock : (m,) knockoff scan effect sizes
    block_indices : list of lists of variant indices per block
    aggregation : "max_stat" or "sum_sq"

    Returns
    -------
    W : (B,) importance statistics per block
    """
    B = len(block_indices)
    device = stat_orig.device
    W = torch.zeros(B, dtype=STAT_DTYPE, device=device)

    # Signed Z-scores
    Z_orig = torch.sign(beta_orig) * torch.sqrt(stat_orig.clamp(min=0.0))
    Z_knock = torch.sign(beta_knock) * torch.sqrt(stat_knock.clamp(min=0.0))

    for b, indices in enumerate(block_indices):
        if len(indices) == 0:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        z_o = Z_orig[idx]
        z_k = Z_knock[idx]

        if aggregation == "max_stat":
            # T(Z) = max(|Z_j|)
            T_orig = z_o.abs().max().item()
            T_knock = z_k.abs().max().item()
        elif aggregation == "sum_sq":
            # T(Z) = sum(Z_j^2)
            T_orig = (z_o**2).sum().item()
            T_knock = (z_k**2).sum().item()
        else:
            raise ValueError(f"Unknown aggregation '{aggregation}'. "
                             "Use 'max_stat' or 'sum_sq'.")

        W[b] = T_orig - T_knock

    return W


# ---------------------------------------------------------------------------
# Helper: knockoff+ filter
# ---------------------------------------------------------------------------

def _knockoff_plus_filter(
    W: Tensor,
    target_fdr: float,
) -> tuple[float, list[int]]:
    """Apply the knockoff+ filter.

    tau = min{t > 0 : (1 + #{W <= -t}) / max(1, #{W >= t}) <= q}

    Parameters
    ----------
    W : (B,) importance statistics
    target_fdr : target FDR level q

    Returns
    -------
    threshold : float — the knockoff+ threshold tau (inf if no selections)
    selected : list[int] — indices of blocks with W >= tau
    """
    # Candidate thresholds: sorted unique absolute values of W
    W_abs = W.abs()
    # Only consider positive W values as candidate thresholds
    W_pos = W[W > 0]
    if W_pos.numel() == 0:
        return float("inf"), []

    candidates = torch.sort(W_pos)[0].unique()

    threshold = float("inf")
    for t in candidates:
        t_val = t.item()
        n_above = (W >= t_val).sum().item()
        n_below = (W <= -t_val).sum().item()
        ratio = (1.0 + n_below) / max(1.0, n_above)
        if ratio <= target_fdr:
            threshold = t_val
            break

    if threshold == float("inf"):
        return threshold, []

    selected = [b for b in range(len(W)) if W[b].item() >= threshold]
    return threshold, selected


# ---------------------------------------------------------------------------
# Main class: KnockoffLMM
# ---------------------------------------------------------------------------

class KnockoffLMM:
    """Knockoff Mixed Model for FDR-controlled GWAS.

    Combines LD block detection, knockoff generation, LMM scanning,
    and the knockoff+ filter into a single ``run()`` method.

    This is NOT a BaseModel — it does not conform to the fit_null/score_chunk
    protocol because it requires all genotypes at once.

    Parameters
    ----------
    config : NumericalConfig, optional
        Numerical configuration for the underlying LMM.
    target_fdr : float
        Target FDR level (default 0.05).
    ld_method : str
        LD block detection method (default "gabriel").
    knockoff_method : str
        Knockoff generation method (default "equicorrelated").
    aggregation : str
        Block importance aggregation: "max_stat" or "sum_sq" (default "max_stat").
    seed : int, optional
        Random seed for knockoff generation.
    **ld_kwargs
        Additional keyword arguments passed to detect_blocks().
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        target_fdr: float = 0.05,
        ld_method: str = "gabriel",
        knockoff_method: str = "equicorrelated",
        aggregation: str = "max_stat",
        seed: int | None = None,
        **ld_kwargs,
    ) -> None:
        self.config = config or NumericalConfig()
        self.target_fdr = target_fdr
        self.ld_method = ld_method
        self.knockoff_method = knockoff_method
        self.aggregation = aggregation
        self.seed = seed
        self.ld_kwargs = ld_kwargs

    def run(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor,
        G: Tensor,
        variant_meta,
        variant_pos: list[int],
        variant_chr: list[str],
    ) -> KnockoffResult:
        """Full knockoff GWAS pipeline.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates (intercept as first column)
        K : (n, n) — GRM / kinship matrix
        G : (n, m) — full genotype matrix (all variants)
        variant_meta : VariantMeta — per-variant annotation
        variant_pos : list[int] — base-pair positions
        variant_chr : list[str] — chromosome labels

        Returns
        -------
        KnockoffResult
        """
        from .single_trait_lmm import SingleTraitLMM
        from ..ld import detect_blocks

        device = G.device
        n, m = G.shape
        G = G.to(STAT_DTYPE)

        if Y.ndim == 2:
            Y = Y.squeeze(1)

        # ── Step 1: Detect LD blocks ──────────────────────────────
        logger.info("Step 1: Detecting LD blocks (method=%s)...", self.ld_method)
        blocks = detect_blocks(
            G, variant_pos, variant_chr,
            method=self.ld_method,
            **self.ld_kwargs,
        )

        # Extract block indices; also collect unblocked singletons
        blocked_set: set[int] = set()
        block_indices: list[list[int]] = []
        for blk in blocks:
            block_indices.append(blk.variant_indices)
            blocked_set.update(blk.variant_indices)

        # Add singleton blocks for unblocked variants
        for j in range(m):
            if j not in blocked_set:
                block_indices.append([j])

        B = len(block_indices)
        logger.info("  %d blocks (%d from LD + %d singletons)",
                     B, len(blocks), B - len(blocks))

        # ── Step 2: Generate knockoff genotypes ───────────────────
        logger.info("Step 2: Generating knockoff genotypes...")
        G_knockoff = torch.zeros_like(G)

        for b, indices in enumerate(block_indices):
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            G_block = G[:, idx]
            p_b = len(indices)

            # Per-block seed for reproducibility
            block_seed = None
            if self.seed is not None:
                block_seed = self.seed + b

            if p_b == 1:
                G_knockoff[:, idx] = _generate_knockoffs_equicorrelated(
                    G_block,
                    torch.ones(1, 1, dtype=STAT_DTYPE, device=device),
                    seed=block_seed,
                )
            else:
                Sigma_b = _compute_correlation_matrix(G_block)
                G_knockoff[:, idx] = _generate_knockoffs_equicorrelated(
                    G_block, Sigma_b, seed=block_seed,
                )

        # ── Step 3: Fit LMM null + scan original & knockoff ──────
        logger.info("Step 3: Fitting LMM null model...")
        lmm = SingleTraitLMM(config=self.config)
        null_fit = lmm.fit_null(Y, X0, K=K)

        logger.info("Step 3a: Scanning original genotypes...")
        result_orig = lmm.score_chunk(G, null_fit, variant_meta, test="wald")

        logger.info("Step 3b: Scanning knockoff genotypes...")
        # Build knockoff variant_meta (same structure, just for scan)
        result_knock = lmm.score_chunk(G_knockoff, null_fit, variant_meta,
                                       test="wald")

        # ── Step 4: Block-level importance statistics ─────────────
        logger.info("Step 4: Computing importance statistics (aggregation=%s)...",
                     self.aggregation)
        W = _knockoff_importance(
            result_orig.stat, result_knock.stat,
            result_orig.beta, result_knock.beta,
            block_indices, self.aggregation,
        )

        # ── Step 5: Knockoff+ filter ─────────────────────────────
        logger.info("Step 5: Applying knockoff+ filter (FDR=%.3f)...",
                     self.target_fdr)
        threshold, selected = _knockoff_plus_filter(W, self.target_fdr)

        # Build per-SNP is_selected mask
        is_selected = torch.zeros(m, dtype=torch.bool, device=device)
        for b in selected:
            for j in block_indices[b]:
                is_selected[j] = True

        n_selected = len(selected)
        n_snps_selected = is_selected.sum().item()
        logger.info("  threshold=%.4f, %d blocks selected (%d SNPs)",
                     threshold if threshold != float("inf") else 0.0,
                     n_selected, n_snps_selected)

        return KnockoffResult(
            block_indices=block_indices,
            W_stat=W,
            selected_blocks=selected,
            threshold=threshold,
            chr=result_orig.chr,
            pos=result_orig.pos,
            snp=result_orig.snp,
            a1=result_orig.a1,
            a2=result_orig.a2,
            af=result_orig.af,
            beta=result_orig.beta,
            se=result_orig.se,
            stat=result_orig.stat,
            p=result_orig.p,
            beta_knockoff=result_knock.beta,
            stat_knockoff=result_knock.stat,
            is_selected=is_selected,
            target_fdr=self.target_fdr,
            n_blocks=B,
            n_selected=n_selected,
            ld_method=self.ld_method,
            knockoff_method=self.knockoff_method,
            aggregation=self.aggregation,
        )
