"""Advanced GRM methods: Slater, Endelman, Vitezica, Su, Yang, pseudo-diploid, weighted.

References:
- Slater et al. (2016) doi:10.1007/s00122-016-2761-3
- Endelman & Jannink (2012) doi:10.2135/cropsci2011.08.0451
- Vitezica et al. (2013) doi:10.1534/genetics.113.155176
- Su et al. (2012) doi:10.1371/journal.pone.0045293
- Yang et al. (2010) doi:10.1038/ng.608
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from ..preprocess.standardize import compute_allele_frequencies
from .kinship import GRMMetadata, grm_vanraden

logger = logging.getLogger(__name__)


def grm_slater(G: Tensor, ploidy: int = 4) -> tuple[Tensor, GRMMetadata]:
    """Slater et al. (2016) polyploid-specific GRM with modified centering.

    K = (G - M)(G - M)^T / sum_j(mean_j * (k - mean_j))

    where M is the column-mean matrix broadcast to (n, m), and k is the ploidy.
    This differs from VanRaden in that the normalizer uses raw mean dosage
    rather than allele-frequency-based 2p(1-p).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix in [0, k], NaN-free.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    n, m = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    col_means = G.mean(dim=0)  # (m,)
    X_centered = G - col_means.unsqueeze(0)

    # Normalizer: sum_j mean_j * (k - mean_j)
    normalizer = (col_means * (ploidy - col_means)).sum()
    if normalizer.abs() < 1e-10:
        logger.warning("Slater normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    K = X_centered @ X_centered.T / normalizer

    meta = GRMMetadata(
        method="slater",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=normalizer.item(),
    )
    return K, meta


def grm_endelman_digenic(G: Tensor) -> tuple[Tensor, GRMMetadata]:
    """Endelman & Jannink (2012) digenic interaction GRM for tetraploids.

    Captures non-additive (digenic) relationships via:
        X_recoded[i,j] = 6*p_j^2 - 3*p_j*d[i,j] + 0.5*d[i,j]*(d[i,j] - 1)
        K = X_recoded @ X_recoded^T / sum_j(6 * p_j^2 * q_j^2)

    Restricted to ploidy = 4 (tetraploid).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix in [0, 4], NaN-free.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    n, m = G.shape
    ploidy = 4

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    max_dose = G.max().item()
    if max_dose > 4 + 0.5:
        raise ValueError(
            f"Endelman digenic GRM is tetraploid-only (ploidy=4). "
            f"Max dosage {max_dose} suggests higher ploidy."
        )

    af = compute_allele_frequencies(G, ploidy=ploidy)  # (m,)
    p = af
    q = 1.0 - af

    # Recode: 6p² - 3pd + 0.5*d*(d-1)
    p_exp = p.unsqueeze(0).expand_as(G)
    X_recoded = 6.0 * p_exp**2 - 3.0 * p_exp * G + 0.5 * G * (G - 1.0)

    # Normalizer: sum(6 * p^2 * q^2)
    normalizer = (6.0 * p**2 * q**2).sum()
    if normalizer.abs() < 1e-10:
        logger.warning("Endelman normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    K = X_recoded @ X_recoded.T / normalizer

    meta = GRMMetadata(
        method="endelman_digenic",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=normalizer.item(),
    )
    return K, meta


def grm_vitezica_dominance(G: Tensor, ploidy: int = 2) -> tuple[Tensor, GRMMetadata]:
    """Vitezica et al. (2013) genomic dominance relationship matrix.

    Orthogonal partition of genetic variance into additive and dominance.
    For diploid genotypes (0, 1, 2):
        d_ij = 2*p*q for het (dose=1), -(2p²) for dose=0, -(2q²) for dose=2
        K_dom = D @ D^T / sum(2*p²*q²) for each SNP j

    For polyploid, intermediate dosage classes (0 < d < k) are treated as
    heterozygous, with coding extended proportionally.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix, NaN-free.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    n, m = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    af = compute_allele_frequencies(G, ploidy=ploidy)
    p = af.unsqueeze(0).expand_as(G)
    q = 1.0 - p

    if ploidy == 2:
        # Exact diploid coding
        D = torch.zeros_like(G)
        D[G == 0] = -(2.0 * p[G == 0] ** 2)
        D[G == 1] = 2.0 * p[G == 1] * q[G == 1]
        D[G == 2] = -(2.0 * q[G == 2] ** 2)
    else:
        # Polyploid extension: heterozygosity-proportional coding
        # het_prop = d*(k-d) / (k/2)^2 gives 0 for hom, max for d=k/2
        het_prop = G * (ploidy - G) / ((ploidy / 2.0) ** 2)
        # Scale: het -> 2pq, hom -> -2p² or -2q²
        p_flat = af.unsqueeze(0).expand_as(G)
        q_flat = 1.0 - p_flat
        expected_het = 2.0 * p_flat * q_flat
        D = het_prop * expected_het - (1.0 - het_prop) * 2.0 * p_flat * q_flat

    # Normalizer: sum over SNPs of (2 * p_j² * q_j²)
    af_sq = af**2
    normalizer = (2.0 * af_sq * (1.0 - af) ** 2).sum()
    if normalizer.abs() < 1e-10:
        logger.warning("Vitezica normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    K = D @ D.T / normalizer

    meta = GRMMetadata(
        method="vitezica_dominance",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=normalizer.item(),
    )
    return K, meta


def grm_su_dominance(G: Tensor, ploidy: int = 2) -> tuple[Tensor, GRMMetadata]:
    """Su et al. (2012) alternative dominance GRM.

    For diploid:
        h_ij = (1 - 2*p*q) for het (dose=1), (0 - 2*p*q) for hom
        K = H @ H^T / sum(p*q*(1 - p*q))

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix, NaN-free.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    n, m = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    af = compute_allele_frequencies(G, ploidy=ploidy)
    p = af.unsqueeze(0).expand_as(G)
    q = 1.0 - p
    two_pq = 2.0 * p * q

    if ploidy == 2:
        is_het = (G == 1)
        H = torch.where(is_het, 1.0 - two_pq, -two_pq)
    else:
        # Polyploid: use het proportion as in Vitezica
        het_prop = G * (ploidy - G) / ((ploidy / 2.0) ** 2)
        het_prop = torch.clamp(het_prop, 0.0, 1.0)
        H = het_prop * (1.0 - two_pq) + (1.0 - het_prop) * (-two_pq)

    # Normalizer: sum(p*q*(1 - p*q))
    pq = af * (1.0 - af)
    normalizer = (pq * (1.0 - pq)).sum()
    if normalizer.abs() < 1e-10:
        logger.warning("Su normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    K = H @ H.T / normalizer

    meta = GRMMetadata(
        method="su_dominance",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=normalizer.item(),
    )
    return K, meta


def grm_yang_gcta(G: Tensor, ploidy: int = 2) -> tuple[Tensor, GRMMetadata]:
    """Yang et al. (2010) GCTA-style GRM with per-SNP normalization.

    K[i,j] = (1/m) sum_l (x_il - 2p_l)(x_jl - 2p_l) / (2p_l(1-p_l))

    Each SNP is individually standardized before the cross-product,
    so the diagonal directly estimates 1 + F (inbreeding coefficient).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix, NaN-free.
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    n, m = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    af = compute_allele_frequencies(G, ploidy=ploidy)

    # Per-SNP variance: k * p * (1-p) for polyploid
    per_snp_var = ploidy * af * (1.0 - af)  # (m,)

    # Filter monomorphic SNPs
    polymorphic = per_snp_var > 1e-10
    n_poly = int(polymorphic.sum().item())

    if n_poly == 0:
        logger.warning("No polymorphic SNPs for Yang GRM.")
        K = torch.eye(n, dtype=torch.float64, device=G.device)
        meta = GRMMetadata(
            method="yang_gcta", n_samples=n, n_snps_used=0,
            ploidy=ploidy, standardization="per_snp_scale",
        )
        return K, meta

    G_poly = G[:, polymorphic]
    af_poly = af[polymorphic]
    var_poly = per_snp_var[polymorphic]

    # Center and per-SNP scale
    means = af_poly * ploidy
    Z = (G_poly - means.unsqueeze(0)) / torch.sqrt(var_poly).unsqueeze(0)

    K = Z @ Z.T / n_poly

    meta = GRMMetadata(
        method="yang_gcta",
        n_samples=n,
        n_snps_used=n_poly,
        ploidy=ploidy,
        standardization="per_snp_scale",
    )
    return K, meta


def grm_pseudo_diploid(G: Tensor, ploidy: int = 4) -> tuple[Tensor, GRMMetadata]:
    """Pseudo-diploid GRM: convert polyploid dosage to diploid scale then VanRaden.

    Maps dosage from [0, k] to [0, 2] via rounding: round(dose * 2 / k).
    Useful as a fallback when polyploid-specific methods are unstable.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix in [0, k], NaN-free.
    ploidy : int
        Original organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")

    G_diploid = torch.round(G * 2.0 / ploidy).clamp(0.0, 2.0)
    K, meta = grm_vanraden(G_diploid, ploidy=2)
    meta.method = "pseudo_diploid"
    meta.ploidy = ploidy
    return K, meta


def grm_weighted(
    G: Tensor,
    weights: Tensor,
    ploidy: int = 2,
) -> tuple[Tensor, GRMMetadata]:
    """Weighted GRM: K = X @ diag(w) @ X^T / sum(w * k*p*(1-p)).

    Supports LD-weighted or GWAS-informed kinship computation
    (e.g., for iterative GWAS like FarmCPU in polyploid context).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix, NaN-free.
    weights : Tensor, shape (m,)
        Per-SNP weights (non-negative).
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    (Tensor, GRMMetadata)
    """
    G = G.to(torch.float64)
    weights = weights.to(torch.float64)
    n, m = G.shape

    if torch.isnan(G).any():
        raise ValueError("GRM input contains NaN. Impute first.")
    if weights.shape[0] != m:
        raise ValueError(f"weights length {weights.shape[0]} != number of SNPs {m}")
    if (weights < 0).any():
        raise ValueError("Weights must be non-negative.")

    af = compute_allele_frequencies(G, ploidy=ploidy)
    means = af * ploidy
    X_centered = G - means.unsqueeze(0)

    # Weighted cross-product: X @ diag(w) @ X^T
    X_weighted = X_centered * torch.sqrt(weights).unsqueeze(0)
    K_unnorm = X_weighted @ X_weighted.T

    # Normalizer: sum(w * k * p * (1-p))
    normalizer = (weights * ploidy * af * (1.0 - af)).sum()
    if normalizer.abs() < 1e-10:
        logger.warning("Weighted GRM normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=torch.float64, device=G.device)

    K = K_unnorm / normalizer

    meta = GRMMetadata(
        method="weighted",
        n_samples=n,
        n_snps_used=m,
        ploidy=ploidy,
        standardization="center_scale",
        normalizer=normalizer.item(),
    )
    return K, meta
