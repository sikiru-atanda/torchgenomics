"""Leave-Region-Out LMM: block-level LOCO for fine-grained proximal decontamination.

Standard LOCO (Leave-One-Chromosome-Out) excludes entire chromosomes from the
GRM when testing SNPs on that chromosome.  This is coarse — in crops and dense
sequence data, proximal contamination (where a causal variant's signal is absorbed
into the random polygenic effect) is often local rather than chromosome-wide.

LRO-LMM refines LOCO to the LD-block level:

    K_{-b} = K_full - K_b

where K_b is the GRM contribution of block b's variants.  For each block,
eigendecompose K_{-b}, fit a fresh null model, and scan only that block's SNPs.

This is a whole-genome procedure requiring all genotypes at once.

Reuses: SingleTraitLMM for per-block null fitting + scanning,
detect_blocks() for block partitioning, grm_vanraden() for GRM computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LROResult:
    """Output of LROLMM.run().

    Wraps a merged ScanResult with block-level diagnostics.
    """

    # Per-SNP results (all m variants)
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor          # (m,)
    beta: Tensor        # (m,)
    se: Tensor          # (m,)
    stat: Tensor        # (m,)
    p: Tensor           # (m,)
    test: str

    # Block diagnostics
    n_blocks: int
    block_sizes: list[int]
    block_method: str


# ---------------------------------------------------------------------------
# Helper: compute a block's GRM contribution
# ---------------------------------------------------------------------------

def _block_grm_contribution(
    G_block: Tensor,
    ploidy: int,
    normalizer: float,
) -> Tensor:
    """Compute block b's contribution to the full GRM.

    K_b = X_b_centered @ X_b_centered^T / normalizer

    Uses the SAME normalizer as the full GRM (sum over ALL variants),
    so K_full = sum_b K_b exactly.

    Parameters
    ----------
    G_block : (n, p_b) genotype submatrix
    ploidy : int
    normalizer : float — full-genome normalizer from grm_vanraden

    Returns
    -------
    K_b : (n, n) block GRM contribution
    """
    from ..linalg.kinship import compute_allele_frequencies

    G_block = G_block.to(STAT_DTYPE)
    af = compute_allele_frequencies(G_block, ploidy=ploidy)
    means = af * ploidy
    X_centered = G_block - means.unsqueeze(0)

    K_b = (X_centered @ X_centered.T) / normalizer
    return K_b


# ---------------------------------------------------------------------------
# Main class: LROLMM
# ---------------------------------------------------------------------------

class LROLMM:
    """Leave-Region-Out LMM for block-level proximal decontamination.

    Parameters
    ----------
    config : NumericalConfig, optional
    ld_method : str
        LD block detection method (default "r2" for speed).
    ploidy : int
        Organism ploidy level (default 2).
    **ld_kwargs
        Additional keyword arguments for detect_blocks().
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        ld_method: str = "r2",
        ploidy: int = 2,
        **ld_kwargs,
    ) -> None:
        self.config = config or NumericalConfig()
        self.ld_method = ld_method
        self.ploidy = ploidy
        self.ld_kwargs = ld_kwargs

    def run(
        self,
        Y: Tensor,
        X0: Tensor,
        G: Tensor,
        variant_meta,
        variant_pos: list[int],
        variant_chr: list[str],
        test: str = "wald",
        *,
        K_full: Tensor | None = None,
        normalizer: float | None = None,
    ) -> LROResult:
        """Full leave-region-out GWAS pipeline.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates
        G : (n, m) — genotype matrix. Either the full genome (legacy use) or
            a single chromosome's slice (streaming use). When a chromosome
            slice is passed, ``K_full`` and ``normalizer`` MUST be supplied
            externally so that K_full reflects all chromosomes; the
            per-block ``K_b = G_block @ G_block^T / normalizer`` is then
            subtracted from the genome-wide K_full to give the correct
            leave-this-block-out kinship.
        variant_meta : VariantMeta — for the SNPs in ``G``.
        variant_pos : list[int] — for the SNPs in ``G``.
        variant_chr : list[str] — for the SNPs in ``G``.
        test : "wald", "score", or "lrt"
        K_full : (n, n) Tensor, optional
            Pre-computed genome-wide GRM. If None, computed from ``G`` in
            the legacy single-call code path.
        normalizer : float, optional
            VanRaden normalizer for ``K_full`` (sum 2pq across genome).
            Required when K_full is supplied; otherwise computed from G.

        Returns
        -------
        LROResult
        """
        from ..ld import detect_blocks
        from ..linalg.kinship import grm_vanraden
        from .base import VariantMeta
        from .single_trait_lmm import SingleTraitLMM

        G = G.to(STAT_DTYPE)
        device = G.device
        n, m = G.shape

        if Y.ndim == 2:
            Y = Y.squeeze(1)
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)

        # ── Step 1: Compute full GRM and its normalizer ──────────
        if K_full is None or normalizer is None:
            logger.info("Step 1: Computing full GRM (VanRaden)...")
            K_full, grm_meta = grm_vanraden(G, ploidy=self.ploidy)
            normalizer = grm_meta.normalizer
        else:
            logger.info(
                "Step 1: Using caller-supplied genome-wide GRM "
                "(streaming variant; normalizer=%.4g).",
                float(normalizer),
            )

        # ── Step 2: Detect LD blocks ─────────────────────────────
        logger.info("Step 2: Detecting LD blocks (method=%s)...", self.ld_method)
        blocks = detect_blocks(
            G, variant_pos, variant_chr,
            method=self.ld_method,
            **self.ld_kwargs,
        )

        # Build block list including singletons
        blocked_set: set[int] = set()
        block_indices: list[list[int]] = []
        for blk in blocks:
            block_indices.append(blk.variant_indices)
            blocked_set.update(blk.variant_indices)

        for j in range(m):
            if j not in blocked_set:
                block_indices.append([j])

        B = len(block_indices)
        block_sizes = [len(b) for b in block_indices]
        logger.info("  %d blocks (avg %.1f SNPs/block)", B,
                     sum(block_sizes) / max(B, 1))

        # ── Step 3: Per-block LRO scan ───────────────────────────
        logger.info("Step 3: Per-block leave-region-out scan (test=%s)...", test)

        # Pre-allocate output tensors
        all_beta = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        all_se = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        all_stat = torch.zeros(m, dtype=STAT_DTYPE, device=device)
        all_p = torch.ones(m, dtype=STAT_DTYPE, device=device)
        all_af = torch.zeros(m, dtype=STAT_DTYPE, device=device)

        lmm = SingleTraitLMM(config=self.config)

        for b_idx, indices in enumerate(block_indices):
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            p_b = len(indices)

            # Compute K_{-b} = K_full - K_b
            G_block = G[:, idx]
            K_b = _block_grm_contribution(G_block, self.ploidy, normalizer)
            K_minus_b = K_full - K_b

            # Ensure PD (small jitter)
            K_minus_b = K_minus_b + 1e-6 * torch.eye(n, dtype=STAT_DTYPE,
                                                       device=device)

            # Fit null on K_{-b}
            null_fit = lmm.fit_null(Y, X0, K=K_minus_b)

            # Build per-block VariantMeta
            vmeta_block = VariantMeta(
                snp=[variant_meta.snp[j] for j in indices],
                chr=[variant_meta.chr[j] for j in indices],
                pos=[variant_meta.pos[j] for j in indices],
                a1=[variant_meta.a1[j] for j in indices],
                a2=[variant_meta.a2[j] for j in indices],
            )

            # Scan this block's SNPs against the block-excluded null
            result_b = lmm.score_chunk(G_block, null_fit, vmeta_block, test=test)

            # Store results
            all_beta[idx] = result_b.beta
            all_se[idx] = result_b.se
            all_stat[idx] = result_b.stat
            all_p[idx] = result_b.p
            all_af[idx] = result_b.af

            if (b_idx + 1) % 50 == 0 or b_idx == B - 1:
                logger.info("  Block %d/%d done", b_idx + 1, B)

        return LROResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=all_af,
            beta=all_beta,
            se=all_se,
            stat=all_stat,
            p=all_p,
            test=test,
            n_blocks=B,
            block_sizes=block_sizes,
            block_method=self.ld_method,
        )
