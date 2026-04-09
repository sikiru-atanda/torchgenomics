"""Polyploid-specific GRM: per-gene-action-model kinship, LOCO, ASV, epistatic."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ..preprocess.polyploid import recode_gene_action
from .kinship import GRMMetadata, grm_vanraden

logger = logging.getLogger(__name__)


def grm_polyploid_gene_action(G: Tensor, model: str, ploidy: int) -> tuple[Tensor, GRMMetadata]:
    """Compute GRM after recoding G under a specific gene-action model.

    For dominant/overdominant/diplo-additive models, the genotype matrix
    is recoded via the gene-action function before GRM construction.
    This produces a model-specific relatedness matrix (charter Section 9d).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (values in [0, k], NaN-free).
    model : str
        Gene-action model name (additive, 1-dom, diplo-additive, etc.).
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
        Gene-action-model-specific GRM and provenance metadata.
    """
    if model == "general":
        # General model produces (n, m, k-1) tensor — use additive GRM
        logger.warning(
            "General model produces multi-column encoding. "
            "Using additive GRM instead."
        )
        return grm_vanraden(G, ploidy=ploidy)

    # Recode genotypes under the gene-action model
    G_recoded = recode_gene_action(G, model, ploidy)

    # For recoded data, effective ploidy for GRM normalization:
    # - additive: ploidy as-is
    # - binary models (j-dom, overdominant): max dosage = 1, treat as diploid-like
    # - diplo-additive: max dosage = k/2
    if model in ("additive",):
        effective_ploidy = ploidy
    elif model == "diplo-additive":
        effective_ploidy = ploidy // 2
    else:
        # Binary models (j-dom, overdominant): values in {0, 1}
        effective_ploidy = 1

    return grm_vanraden(G_recoded, ploidy=effective_ploidy)


def grm_loco(
    G: Tensor,
    chr_labels: list[str],
    exclude_chr: str,
    ploidy: int = 2,
) -> tuple[Tensor, GRMMetadata]:
    """Leave-One-Chromosome-Out GRM: exclude variants on the test chromosome.

    Essential for avoiding proximal contamination — when the test SNP
    is included in the GRM, it inflates the kinship signal and reduces
    power / biases p-values.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Full dosage matrix (NaN-free).
    chr_labels : list[str]
        Per-variant chromosome labels, length m.
    exclude_chr : str
        Chromosome to exclude.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    Tensor, shape (n, n)
        GRM computed from all chromosomes except *exclude_chr*.
    """
    if len(chr_labels) != G.shape[1]:
        raise ValueError(
            f"chr_labels length ({len(chr_labels)}) does not match "
            f"number of variants ({G.shape[1]})."
        )

    # Build column mask: True for variants NOT on excluded chromosome
    keep_mask = torch.tensor(
        [c != exclude_chr for c in chr_labels],
        dtype=torch.bool,
    )

    n_keep = keep_mask.sum().item()
    if n_keep == 0:
        raise ValueError(
            f"No variants remain after excluding chromosome '{exclude_chr}'. "
            f"All {G.shape[1]} variants are on this chromosome."
        )

    n_excluded = G.shape[1] - n_keep
    logger.info(
        "LOCO GRM: excluding chr %s (%d variants), using %d variants.",
        exclude_chr, n_excluded, n_keep,
    )

    G_loco = G[:, keep_mask]
    K, meta = grm_vanraden(G_loco, ploidy=ploidy)
    meta.loco_chr = exclude_chr
    return K, meta


def grm_asv_transform(K: Tensor) -> Tensor:
    """Normalize GRM to ASV (Additive Standardized Variance) form.

    Scales K so that average diagonal equals 1:
        K_asv = K / (trace(K) / (n - 1))

    Recommended by Feldmann et al. (2022) for comparability across
    GRM methods and for diagonal elements to approximate individual
    inbreeding coefficients.

    Parameters
    ----------
    K : Tensor, shape (n, n)
        Genomic relationship matrix.

    Returns
    -------
    Tensor, shape (n, n)
        Scaled GRM with average diagonal ≈ 1.
    """
    n = K.shape[0]
    if n <= 1:
        return K.clone()
    scale = torch.trace(K) / (n - 1)
    if scale.abs() < 1e-10:
        logger.warning("ASV scale factor near zero — returning unscaled GRM.")
        return K.clone()
    return K / scale


def grm_epistatic_hadamard(
    K_add: Tensor,
    K_dom: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute epistatic GRMs via Hadamard (element-wise) products.

    Following Muñoz et al. (2014) and Su et al. (2012), epistatic
    relationship matrices are element-wise products of additive (G)
    and dominance (D) GRMs:

    - K_aa = K_add ⊙ K_add  (additive × additive epistasis)
    - K_dd = K_dom ⊙ K_dom  (dominance × dominance, if K_dom provided)
    - K_ad = K_add ⊙ K_dom  (additive × dominance, if K_dom provided)

    Parameters
    ----------
    K_add : Tensor, shape (n, n)
        Additive GRM.
    K_dom : Tensor, shape (n, n), optional
        Dominance GRM. If None, only K_aa is returned.

    Returns
    -------
    dict[str, Tensor]
        Epistatic GRMs keyed by "K_aa", "K_dd", "K_ad".
    """
    result = {"K_aa": K_add * K_add}

    if K_dom is not None:
        if K_dom.shape != K_add.shape:
            raise ValueError(
                f"K_dom shape {K_dom.shape} does not match K_add shape {K_add.shape}."
            )
        result["K_dd"] = K_dom * K_dom
        result["K_ad"] = K_add * K_dom

    return result
