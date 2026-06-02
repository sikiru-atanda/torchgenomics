"""Streaming construction of additive + dominance + epistatic kernels (F3).

Used by ``mklmm-scan`` to avoid materializing the full ``(n, m)`` genotype
matrix when building the kernel set for :class:`MultiKernelLMM`.

The streaming insight: every kernel here is ``(n, n)`` — small. Each base
kernel (additive VanRaden, Vitezica dominance) is a sum of per-SNP
contributions; we accumulate them chunk-by-chunk in ``float64`` and free
each chunk before reading the next. Epistatic kernels are Hadamard
products of the now-built ``(n, n)`` base kernels, so they cost nothing
extra in terms of G residency.

Memory: ``O(n² × n_kernels × 8 B)`` (same as the materialized path) plus
``O(n × chunk_size × 8 B)`` per chunk during the accumulator pass. The win
is dropping the ``n × m × 8 B`` full-G allocation, which at biobank scale
(~500K × 10M float64) is ~40 TB → 0.

Behavioral parity vs the eager :func:`build_multi_kernels`: float32-GEMM
streaming slack ~1e-7 on the additive path (matches
:func:`grm_vanraden_streaming`'s documented slack); the dominance path
runs the GEMM in ``float64`` to match :func:`grm_vitezica_dominance`
exactly (the dominance encoding has small absolute values and benefits
from full-precision accumulation).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import Tensor

from ..preprocess.standardize import compute_allele_frequencies

logger = logging.getLogger(__name__)


def _vitezica_dominance_encoding(
    G_chunk_f64: Tensor,
    af: Tensor,
    ploidy: int,
) -> Tensor:
    """Dominance encoding matching :func:`grm_vitezica_dominance`.

    For diploid (``ploidy == 2``):
        d = -2p²       if dose == 0
        d =  2pq       if dose == 1
        d = -2q²       if dose == 2

    For polyploid, intermediate dosage classes are treated as
    heterozygous, with coding extended proportionally — preserves the
    eager polyploid extension exactly.

    Parameters
    ----------
    G_chunk_f64 : (n, m_chunk) float64 dosage chunk (NaN-free).
    af : (m_chunk,) per-SNP allele frequencies.
    ploidy : organism ploidy level.

    Returns
    -------
    D_chunk : (n, m_chunk) dominance-encoded chunk in float64.
    """
    p_flat = af.unsqueeze(0).expand_as(G_chunk_f64)
    q_flat = 1.0 - p_flat

    if ploidy == 2:
        D = torch.zeros_like(G_chunk_f64)
        mask0 = G_chunk_f64 == 0
        mask1 = G_chunk_f64 == 1
        mask2 = G_chunk_f64 == 2
        D[mask0] = -(2.0 * p_flat[mask0] ** 2)
        D[mask1] = 2.0 * p_flat[mask1] * q_flat[mask1]
        D[mask2] = -(2.0 * q_flat[mask2] ** 2)
        return D

    # Polyploid: heterozygosity-proportional coding.
    het_prop = G_chunk_f64 * (ploidy - G_chunk_f64) / ((ploidy / 2.0) ** 2)
    expected_het = 2.0 * p_flat * q_flat
    return het_prop * expected_het - (1.0 - het_prop) * 2.0 * p_flat * q_flat


def build_multi_kernels_streaming(
    chunk_iter: Iterator[tuple[Tensor, object]],
    n_samples: int,
    ploidy: int = 2,
    *,
    include_dominance: bool = True,
    include_epistatic: bool = True,
    device: torch.device | None = None,
    additive_gemm_dtype: torch.dtype = torch.float32,
) -> tuple[list[Tensor], list[str]]:
    """Stream-build additive + dominance + epistatic kernels chunk-by-chunk.

    Drop-in streaming counterpart of
    :func:`torchgenomics.models.multi_kernel_lmm.build_multi_kernels`. The
    returned ``(kernels, names)`` tuple matches the eager builder's
    contract exactly so :class:`MultiKernelLMM.fit_null` can consume
    them unchanged.

    Parameters
    ----------
    chunk_iter : Iterator
        Yields ``(G_chunk, variant_meta)`` tuples. ``G_chunk`` is
        ``(n, m_chunk)``; assumed NaN-free (mean-impute via
        ``_impute_chunk_iter`` upstream).
    n_samples : int
        Total number of samples (rows in each ``G_chunk``).
    ploidy : int
        Organism ploidy level.
    include_dominance : bool
        Include Vitezica dominance GRM.
    include_epistatic : bool
        Include additive × additive epistatic GRM (and additive ×
        dominance, dominance × dominance when dominance is also on).
    device : torch.device, optional
        Device for the per-chunk computation; accumulators always
        live in ``float64`` on this device.
    additive_gemm_dtype : torch.dtype
        Dtype for the per-chunk additive cross-product (default
        ``float32`` for speed; matches
        :func:`grm_vanraden_streaming`'s default). The dominance
        cross-product always runs in ``float64`` because the
        dominance encoding has small absolute values.

    Returns
    -------
    kernels : list of (n, n) ``float64`` kernel matrices.
    names : list of kernel names matching the eager builder
        (``"additive"`` / ``"dominance"`` / ``"epistatic_aa"``).
    """
    if device is None:
        device = torch.device("cpu")

    accum_dtype = torch.float64
    K_add_accum = torch.zeros(n_samples, n_samples, dtype=accum_dtype, device=device)
    norm_add = torch.tensor(0.0, dtype=accum_dtype, device=device)

    if include_dominance:
        K_dom_accum = torch.zeros(n_samples, n_samples, dtype=accum_dtype, device=device)
        norm_dom = torch.tensor(0.0, dtype=accum_dtype, device=device)
    else:
        K_dom_accum = None
        norm_dom = None

    total_snps = 0

    for G_chunk, _ in chunk_iter:
        G_f64 = G_chunk.to(dtype=accum_dtype, device=device)
        af = compute_allele_frequencies(G_f64, ploidy=ploidy)  # (m_chunk,)

        # --- Additive accumulator (VanRaden centering + scale via global norm) ---
        means = af * ploidy
        X_c = G_f64 - means.unsqueeze(0)
        X_c_gemm = X_c.to(additive_gemm_dtype)
        K_chunk = (X_c_gemm @ X_c_gemm.T).to(accum_dtype)
        K_add_accum += K_chunk
        norm_add = norm_add + (ploidy * af * (1.0 - af)).sum()

        # --- Dominance accumulator (Vitezica encoding + global norm) ---
        if include_dominance:
            D_chunk = _vitezica_dominance_encoding(G_f64, af, ploidy=ploidy)
            K_dom_accum += D_chunk @ D_chunk.T
            af_sq = af ** 2
            norm_dom = norm_dom + (2.0 * af_sq * (1.0 - af) ** 2).sum()

        total_snps += G_f64.shape[1]

    # --- Normalizers and final kernel build ---
    if norm_add < 1e-10:
        logger.warning("Streaming additive normalizer near zero.")
        norm_add = torch.tensor(1.0, dtype=accum_dtype, device=device)
    K_add = K_add_accum / norm_add

    kernels: list[Tensor] = [K_add]
    names: list[str] = ["additive"]

    if include_dominance:
        if norm_dom < 1e-10:
            logger.warning("Streaming dominance normalizer near zero.")
            norm_dom = torch.tensor(1.0, dtype=accum_dtype, device=device)
        K_dom = K_dom_accum / norm_dom
        kernels.append(K_dom)
        names.append("dominance")

        if include_epistatic:
            kernels.append(K_add * K_add)
            names.append("epistatic_aa")
    elif include_epistatic:
        kernels.append(K_add * K_add)
        names.append("epistatic_aa")

    logger.info(
        "Streaming multi-kernel build: %d kernels (%s), %d SNPs streamed.",
        len(kernels), ", ".join(names), total_snps,
    )

    return kernels, names
