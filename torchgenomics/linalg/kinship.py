"""GRM builders: streaming GPU GEMM, VanRaden (diploid + polyploid)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from ..preprocess.standardize import compute_allele_frequencies

logger = logging.getLogger(__name__)


@dataclass
class GRMMetadata:
    """Provenance record for a GRM (charter Section 6a-0: Internal Data Contract).

    Tracks how the kinship matrix was constructed so downstream code and
    audit logs can verify consistency.
    """

    method: str  # "vanraden", "zhang", "vanraden_streaming"
    n_samples: int
    n_snps_used: int  # SNPs that contributed (after filtering invariant, etc.)
    ploidy: int
    standardization: str  # "center_scale" for VanRaden, "center_only" for Zhang
    loco_chr: str | None = None  # None = genome-wide; "1" = LOCO excluding chr 1
    gemm_dtype: str | None = None  # e.g. "float32" for streaming GRM
    normalizer: float = 0.0  # sum(k*p*(1-p)) or equivalent


def grm_vanraden(G: Tensor, ploidy: int = 2) -> tuple[Tensor, GRMMetadata]:
    """Compute VanRaden-style GRM: (X - kP)(X - kP)^T / normalizer.

    Matches GEMMA's "centered relatedness matrix" for diploid (k=2).
    For polyploid, generalizes per charter Section 9d:

        K = (X - kP)(X - kP)^T / sum(k/2 * 2 * p_j * (1 - p_j))

    where P is the allele frequency matrix broadcast to (n, m),
    and the denominator normalizes by expected variance under HWE.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (NaN-free; impute first).
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
        Genomic relationship matrix (float64) and provenance metadata.
    """
    G = G.to(torch.float64)
    n, m = G.shape

    # Guard: GRM requires NaN-free input
    if torch.isnan(G).any():
        raise ValueError(
            "GRM input contains NaN values. Impute missing genotypes before "
            "computing the GRM (e.g., torchgenomics.preprocess.impute.impute_mean)."
        )

    # Allele frequencies: mean(dosage) / ploidy
    af = compute_allele_frequencies(G, ploidy=ploidy)  # (m,)

    # Center: X - k*p
    means = af * ploidy  # (m,) per-SNP mean dosage
    X_centered = G - means.unsqueeze(0)  # (n, m)

    # Normalizer: sum over SNPs of k/2 * 2 * p * (1-p) = sum(k * p * (1-p))
    normalizer = (ploidy * af * (1.0 - af)).sum()

    if normalizer < 1e-10:
        logger.warning("GRM normalizer near zero — all SNPs may be monomorphic.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    # K = X_centered @ X_centered.T / normalizer
    K = (X_centered @ X_centered.T) / normalizer

    meta = GRMMetadata(
        method="vanraden",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=float(normalizer),
    )

    return K, meta


def grm_zhang(G: Tensor) -> tuple[Tensor, GRMMetadata]:
    """Compute Zhang (2010) kinship matrix — matches GAPIT's default.

    Algorithm (from GAPIT source ``GAPIT.kinship.Zhang``):
      1. Remove invariant SNPs (MAF = 0 or 1)
      2. Compute heterozygosity and inbreeding coefficient
      3. Center genotypes by subtracting column means
      4. Cross-product: K_raw = X_centered @ X_centered^T
      5. Three-step adjustment:
         a) Normalize by diagonal range: K = top*(K - floor)/(DU - floor)
         b) Adjust diagonal if min(diag) < 1
         c) Adjust off-diagonal if max(off-diag) > top

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix in {0, 1, 2} (NaN-free; impute first).

    Returns
    -------
    (Tensor, GRMMetadata) — Zhang kinship matrix (float64) and provenance metadata.
    """
    G = G.to(torch.float64)
    n, m_orig = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    # Step 1: Remove invariant SNPs
    fa = G.sum(dim=0) / (2 * n)  # allele frequency
    keep = (fa > 0) & (fa < 1)
    G = G[:, keep]
    m = G.shape[1]

    if m == 0:
        logger.warning("Zhang kinship: no polymorphic SNPs remaining.")
        meta = GRMMetadata(
            method="zhang", n_samples=n, n_snps_used=0, ploidy=2,
            standardization="center_only",
        )
        return torch.eye(n, dtype=torch.float64, device=G.device), meta

    # Step 2: Heterozygosity and inbreeding
    het = 1.0 - (G - 1.0).abs()  # het=1 for heterozygous (dosage=1)
    ind_sum = het.sum(dim=1)      # per-individual heterozygosity count
    fi = ind_sum / (2 * m)        # per-individual inbreeding proxy
    inbreeding = 1.0 - fi.min().item()

    # Step 3: Center by column mean
    snp_mean = G.mean(dim=0)  # (m,)
    G_centered = G - snp_mean.unsqueeze(0)  # (n, m)

    # Step 4: Cross-product
    K = G_centered @ G_centered.T  # (n, n)

    # Step 5a: Normalize by diagonal range
    d = K.diag()
    DL = d.min().item()
    DU = d.max().item()
    floor_val = K.min().item()
    top = 1.0 + inbreeding

    if abs(DU - floor_val) < 1e-20:
        logger.warning("Zhang kinship: degenerate diagonal range.")
        meta = GRMMetadata(
            method="zhang",
            n_samples=n,
            n_snps_used=m,
            ploidy=2,
            standardization="center_only",
        )
        return torch.eye(n, dtype=torch.float64, device=G.device), meta

    K = top * (K - floor_val) / (DU - floor_val)
    Dmin = top * (DL - floor_val) / (DU - floor_val)

    # Step 5b: Adjust diagonal if minimum diagonal < 1
    if Dmin < 1.0:
        diag_idx = torch.arange(n, device=G.device)
        # Off-diagonal scaling
        mask_diag = torch.zeros(n, n, dtype=torch.bool, device=G.device)
        mask_diag[diag_idx, diag_idx] = True

        off_diag = K.clone()
        off_diag[mask_diag] = 0
        off_diag = off_diag * (1.0 / Dmin)

        new_diag = (K.diag() - Dmin + 1.0) / ((top + 1.0 - Dmin) * 0.5)

        K = off_diag
        K[diag_idx, diag_idx] = new_diag

    # Step 5c: Adjust off-diagonal if max > top
    diag_idx = torch.arange(n, device=G.device)
    mask_diag = torch.zeros(n, n, dtype=torch.bool, device=G.device)
    mask_diag[diag_idx, diag_idx] = True
    off_diag_vals = K[~mask_diag]
    Omax = off_diag_vals.max().item()
    if Omax > top:
        K[~mask_diag] = K[~mask_diag] * (top / Omax)

    meta = GRMMetadata(
        method="zhang",
        n_samples=n,
        n_snps_used=m,
        ploidy=2,
        standardization="center_only",
    )

    return K, meta


def grm_vanraden_streaming(
    chunk_iter: Iterator[tuple[Tensor, object]],
    n_samples: int,
    ploidy: int = 2,
    gemm_dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[Tensor, GRMMetadata]:
    """Streaming GRM: FP32 GEMM with FP64 accumulator, chunk by chunk.

    Per charter: matrix multiply uses *gemm_dtype* (FP32 by default for speed)
    but the running sum K_accum is always FP64 for numerical fidelity.

    Parameters
    ----------
    chunk_iter : Iterator
        Yields (G_chunk, variant_meta) tuples. G_chunk is (n, m_chunk).
    n_samples : int
        Total number of samples.
    ploidy : int
        Organism ploidy level.
    gemm_dtype : torch.dtype
        Dtype for the per-chunk matrix multiply (default FP32 for speed).
        The accumulator is always FP64.
    device : torch.device, optional
        Device for computation.

    Returns
    -------
    (Tensor, GRMMetadata)
        Genomic relationship matrix (float64) and provenance metadata.
    """
    if device is None:
        device = torch.device("cpu")

    accum_dtype = torch.float64
    K_accum = torch.zeros(n_samples, n_samples, dtype=accum_dtype, device=device)
    normalizer_accum = torch.tensor(0.0, dtype=accum_dtype, device=device)
    total_snps = 0

    for G_chunk, _ in chunk_iter:
        # Allele frequencies computed in FP64 for accuracy
        G_f64 = G_chunk.to(dtype=accum_dtype, device=device)
        af = compute_allele_frequencies(G_f64, ploidy=ploidy)  # (m_chunk,)

        # Center in FP64, then cast to gemm_dtype for the matmul
        means = af * ploidy
        X_c = G_f64 - means.unsqueeze(0)
        X_c_gemm = X_c.to(gemm_dtype)

        # GEMM (FP32 or AMP FP16/BF16), then accumulate into FP64
        if amp_enabled and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                K_chunk = X_c_gemm @ X_c_gemm.T
            K_accum += K_chunk.to(accum_dtype)
        else:
            K_chunk = (X_c_gemm @ X_c_gemm.T).to(accum_dtype)
            K_accum += K_chunk

        # Normalizer always in FP64
        normalizer_accum += (ploidy * af * (1.0 - af)).sum()
        total_snps += G_chunk.shape[1]

    if normalizer_accum < 1e-10:
        logger.warning("GRM normalizer near zero after streaming all chunks.")
        normalizer_accum = torch.tensor(1.0, dtype=accum_dtype, device=device)

    K_accum /= normalizer_accum

    meta = GRMMetadata(
        method="vanraden_streaming",
        n_samples=n_samples,
        n_snps_used=total_snps,
        ploidy=ploidy,
        standardization="center_scale",
        gemm_dtype=str(gemm_dtype),
        normalizer=float(normalizer_accum),
    )

    return K_accum, meta
