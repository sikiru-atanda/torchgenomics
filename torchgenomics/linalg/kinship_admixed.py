"""Admixture-aware GRM/PCs: KING-robust -> LD-prune -> PC-AiR -> PC-Relate.

Reference chain (Tan et al. 2026, Tractor-Mix, Extended Data Fig 1):
- KING-robust kinship: Manichaikul et al. 2010, Bioinformatics 26:2867.
- PC-AiR (relatedness-robust PCs): Conomos et al. 2015, Genet. Epidemiol. 39:276.
- PC-Relate (ancestry-adjusted kinship): Conomos et al. 2016, AJHG 98:127.
- Reference implementation: GENESIS (Gogarten et al. 2019, Bioinformatics 35:5346).

All estimators run in FP64. KING-robust and PC-Relate stream over marker chunks.
"""
from __future__ import annotations

import torch
from torch import Tensor


def king_robust_kinship(G: Tensor, chunk_size: int = 2000) -> Tensor:
    """KING-robust between-family kinship estimator.

    Implements the "between-family" (population-structure-robust) kinship
    estimator of Manichaikul et al. 2010, *Bioinformatics* 26:2867, eq. 11:

        phi_ij = (N_AaAa - 2 * N_AAaa) / (N_Aa^i + N_Aa^j)

    where, over the set of markers scored for the pair (i, j):

    - ``N_AaAa``  = number of markers at which *both* individuals are
      heterozygous (dosage == 1 for both).
    - ``N_AAaa``  = number of markers at which the two individuals carry
      *opposite* homozygous genotypes (one is dosage 0 and the other is
      dosage 2). This is the classical IBS0 count.
    - ``N_Aa^i``  = number of heterozygous markers in individual ``i`` alone
      (i.e. the individual's own het count, not restricted to markers shared
      with ``j``).

    The diagonal is fixed at 0.5 (self-kinship), matching the KING
    convention (the formula above is undefined for i == j since it would
    divide by 2 * N_Aa^i and evaluate to -0.5, which is not meaningful as a
    self-kinship coefficient).

    This is the *between-family* estimator (eq. 11 in the paper), chosen
    because — unlike the "population-specific"/homogeneous estimator (eq.
    9) which assumes all individuals are drawn from a single, unstructured
    population — it remains well-behaved for pairs of individuals drawn
    from *different* subpopulations (e.g. an admixed cohort), which is the
    intended use case here (Task 2 of Phase 57 Unit A, upstream of
    admixture-aware PC-AiR / PC-Relate).

    Scope note (important — do not overclaim): this function is an
    implementation of the published *formula*. It has been pinned against a
    hand-computed numerical example (see
    ``tests/test_kinship_admixed.py::test_king_robust_hand_computed``) and
    against directional sanity properties on a synthetic admixed+related
    fixture (parent-offspring vs. cross-population-unrelated pairs). It has
    **not** been validated against the reference KING software or
    SNPRelate's ``snpgdsIBDKING`` in this task — that external reference-tool
    comparison is Task 6 of this Unit. Do not read the tests in this module
    as reference-equivalence evidence; they are internal-consistency checks
    only.

    Missing data (NaN dosages): a marker contributes to the counts for a
    pair (i, j) only if *neither* individual has a NaN dosage at that
    marker. This applies independently to every pairwise count
    (N_AaAa, N_AAaa) and to each individual's own het count N_Aa^i (which
    only counts that individual's non-missing heterozygous markers — see
    below for why the per-individual het denominator is *not* restricted to
    markers observed in the partner, matching the paper's definition of
    N_Aa(i) as a per-individual quantity).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Diploid dosage matrix, values in {0, 1, 2} (float), NaN for missing.
    chunk_size : int, default 2000
        Number of markers processed per streaming chunk. The (n, m) matrix
        is never fully materialized as an (n, n, m) pairwise tensor; only
        (n, chunk_size) slices and (n, n) accumulators are held at once.

    Returns
    -------
    Tensor, shape (n, n)
        Symmetric KING-robust kinship matrix, diagonal == 0.5.

    References
    ----------
    Manichaikul, A., Mychaleckyj, J.C., Rich, S.S., Daly, K., Sale, M., and
    Chen, W.-M. (2010). Robust relationship inference in genome-wide
    association studies. Bioinformatics 26(22), 2867-2873. Eq. 11.
    """
    G = G.to(torch.float64)
    n, m = G.shape
    device = G.device

    N_AaAa = torch.zeros(n, n, dtype=torch.float64, device=device)
    N_AAaa = torch.zeros(n, n, dtype=torch.float64, device=device)
    het_count = torch.zeros(n, dtype=torch.float64, device=device)  # N_Aa^i

    for s in range(0, m, chunk_size):
        gc = G[:, s:s + chunk_size]                      # (n, c)
        valid = ~torch.isnan(gc)                          # (n, c)
        gc_safe = torch.where(valid, gc, torch.zeros_like(gc))

        het = (gc_safe == 1.0) & valid                     # (n, c)
        hom0 = (gc_safe == 0.0) & valid                     # (n, c)
        hom2 = (gc_safe == 2.0) & valid                     # (n, c)

        het_f = het.to(torch.float64)
        hom0_f = hom0.to(torch.float64)
        hom2_f = hom2.to(torch.float64)

        # Per-individual heterozygous-marker count (own missingness only).
        het_count += het_f.sum(dim=1)

        # Both-heterozygous count restricted to co-observed markers: since
        # het_f is already zeroed at missing positions (valid=False forces
        # gc_safe=0 -> het=False there), het_f @ het_f.T automatically
        # excludes any marker where either individual is missing.
        N_AaAa += het_f @ het_f.T

        # Opposite-homozygote count: same reasoning — hom0_f/hom2_f are zero
        # at missing positions, so the product implicitly requires both
        # individuals observed.
        N_AAaa += hom0_f @ hom2_f.T + hom2_f @ hom0_f.T

    denom = het_count.unsqueeze(1) + het_count.unsqueeze(0)
    # Guard against div-by-zero for pairs where both individuals are fully
    # homozygous (het_count == 0 for both) or have no shared data; such
    # pairs get phi = 0 rather than NaN/inf.
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    phi = (N_AaAa - 2.0 * N_AAaa) / safe_denom
    phi = torch.where(denom > 0, phi, torch.zeros_like(phi))

    phi.fill_diagonal_(0.5)
    phi = 0.5 * (phi + phi.T)
    phi.fill_diagonal_(0.5)
    return phi
