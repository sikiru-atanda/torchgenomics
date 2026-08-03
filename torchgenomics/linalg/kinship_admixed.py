"""Admixture-aware GRM/PCs: KING-robust -> LD-prune -> PC-AiR -> PC-Relate.

Reference chain (Tan et al. 2026, Tractor-Mix, Extended Data Fig 1):
- KING-robust kinship: Manichaikul et al. 2010, Bioinformatics 26:2867.
- PC-AiR (relatedness-robust PCs): Conomos et al. 2015, Genet. Epidemiol. 39:276.
- PC-Relate (ancestry-adjusted kinship): Conomos et al. 2016, AJHG 98:127.
- Reference implementation: GENESIS (Gogarten et al. 2019, Bioinformatics 35:5346).

All estimators run in FP64. KING-robust and PC-Relate stream over marker chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from ..ld._pairwise import compute_r2_matrix
from .eigh import eigendecompose

logger = logging.getLogger(__name__)


def king_robust_kinship(G: Tensor, chunk_size: int = 2000) -> Tensor:
    """KING-robust between-family kinship estimator.

    Implements the "between-family" (population-structure-robust) kinship
    estimator of Manichaikul et al. 2010, *Bioinformatics* 26:2867 -- the
    same estimator computed by the reference KING software, PLINK2
    ``--make-king``, and SNPRelate's ``snpgdsIBDKING(type="KING-robust")``:

        phi_ij = 0.5 - Sd_ij / (4 * min(Nhet_i, Nhet_j))

    where, over the **pairwise-complete** set of markers scored for the
    pair (i, j) -- i.e. markers at which *both* ``i`` and ``j`` are
    observed (non-NaN dosage):

    - ``Sd_ij``   = ``sum_l (g_il - g_jl)^2``, the sum of squared genotype
      differences over the co-observed markers ``l``. A marker contributes
      0 when both individuals share the same dosage (including both het),
      1 when one is heterozygous and the other homozygous, and 4 when the
      two individuals carry *opposite* homozygous genotypes (dosage 0 vs.
      dosage 2, the classical IBS0 case).
    - ``Nhet_i``  = number of heterozygous markers in individual ``i``,
      restricted to the pairwise-complete marker set for the pair (i, j)
      (i.e. markers where ``i`` is heterozygous *and* ``j`` is observed
      there) -- and likewise for ``Nhet_j``. ``min(Nhet_i, Nhet_j)`` is the
      **elementwise minimum** of the two pairwise-complete het counts, not
      their sum.

    Validation (Phase 57 Unit A, Task 6): verified **machine-exact** against
    SNPRelate's ``snpgdsIBDKING(type="KING-robust")`` on a synthetic
    admixed+related fixture (n=120, m=2000, 2 populations, 15 injected
    parent-offspring pairs) -- **max-abs-diff = 4.996e-16, Pearson r = 1.0**
    over all off-diagonal pairs. This is REFERENCE-EQUIVALENT: the residual
    difference is at the FP64 machine-epsilon floor, not a numerical
    approximation. See
    ``validation/external/genesis/{run_king_snprelate.R,compare.py,VALIDATION_RESULTS.md}``
    and
    ``tests/test_kinship_admixed.py::test_king_robust_hand_computed`` /
    ``test_king_robust_unequal_het_counts_distinguishes_min_from_sum``. A
    prior version of this function used an incorrect
    ``(N_AaAa - 2*N_AAaa) / (Nhet_i + Nhet_j)`` formula (a SUM rather than
    MIN denominator, and one that silently dropped every het/homozygote
    single-mismatch marker (``|g_i - g_j| == 1``) from the numerator
    entirely) -- that formula is not the Manichaikul et al. 2010 KING-robust
    estimator and disagreed with SNPRelate. Do not reintroduce a sum-based
    denominator or a numerator that omits diff==1 markers.

    The diagonal is fixed at 0.5 (self-kinship) explicitly, rather than
    relying on the raw formula evaluated at i == j (which is well-defined
    there -- Sd == 0 and min(Nhet_i, Nhet_i) == Nhet_i when i == j, so it
    reduces to 0.5 - 0/(4*Nhet_i) = 0.5 -- but setting it explicitly is more
    robust and sidesteps the Nhet_i == 0 edge case, where the raw formula
    would divide by zero).

    This is the *between-family* estimator, chosen because — unlike the
    "population-specific"/homogeneous estimator which assumes all
    individuals are drawn from a single, unstructured population — it
    remains well-behaved for pairs of individuals drawn from *different*
    subpopulations (e.g. an admixed cohort), which is the intended use case
    here (Task 2 of Phase 57 Unit A, upstream of admixture-aware PC-AiR /
    PC-Relate).

    Missing data (NaN dosages): a marker contributes to *every* term for a
    pair (i, j) -- ``Sd_ij`` and both denominator terms ``Nhet_i``,
    ``Nhet_j`` -- only if *neither* individual has a NaN dosage at that
    marker. In other words, the whole estimator for pair (i, j) is computed
    over the pairwise-complete marker set. This means ``Nhet_i`` is, in
    general, pair-specific (it can differ across the row/column of the
    kinship matrix for a fixed i), because it excludes markers where the
    *other* member of the pair happens to be missing, even though i itself
    is observed there.

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
    association studies. Bioinformatics 26(22), 2867-2873.
    """
    G = G.to(torch.float64)
    n, m = G.shape
    device = G.device

    # Sd[i, j] accumulates sum_l valid_i_l * valid_j_l * (g_il - g_jl)^2 over
    # marker chunks. Expanding the square:
    #   valid_i*valid_j*(g_i - g_j)^2
    #     = valid_i*valid_j*g_i^2 - 2*valid_i*valid_j*g_i*g_j + valid_i*valid_j*g_j^2
    # Let gc_safe = g with NaN positions zeroed (valid ? g : 0), and
    # sq = gc_safe**2 (== valid*g^2 elementwise, since valid is 0/1). Then,
    # summed over l:
    #   term1(i,j) = sum_l valid_i_l * (valid_j_l * g_il^2)      = sq  @ valid_f.T
    #   term3(i,j) = sum_l valid_i_l * (valid_j_l * g_jl^2)      = valid_f @ sq.T
    #   term2(i,j) = sum_l (valid_i_l*g_il) * (valid_j_l*g_jl)   = gc_safe @ gc_safe.T
    # (term2 holds because valid is 0/1-valued, so
    # valid_i*valid_j*g_i*g_j == (valid_i*g_i)*(valid_j*g_j).) This lets the
    # whole (n, n) Sd accumulator be built from three matmuls per chunk,
    # with no (n, n, m) pairwise tensor ever materialized.
    Sd = torch.zeros(n, n, dtype=torch.float64, device=device)
    # Nhet_pair[i, j] = # markers where i is het AND j is observed (pairwise-
    # complete restriction of individual i's own het count to the marker set
    # shared with j). Accumulated in the same streaming chunk loop as Sd so
    # numerator and denominator always share one marker set.
    Nhet_pair = torch.zeros(n, n, dtype=torch.float64, device=device)

    for s in range(0, m, chunk_size):
        gc = G[:, s:s + chunk_size]                      # (n, c)
        valid = ~torch.isnan(gc)                          # (n, c)
        valid_f = valid.to(torch.float64)                  # observed (any genotype)
        gc_safe = torch.where(valid, gc, torch.zeros_like(gc))
        sq = gc_safe * gc_safe                              # valid * g^2

        Sd += (sq @ valid_f.T) + (valid_f @ sq.T) - 2.0 * (gc_safe @ gc_safe.T)

        het = (gc_safe == 1.0) & valid                     # (n, c)
        het_f = het.to(torch.float64)                       # het AND observed

        # Pairwise-complete per-individual het count: [i, j] = # markers
        # where i is het (and, via het_f, i observed) AND j is observed.
        Nhet_pair += het_f @ valid_f.T

    # min(Nhet_i, Nhet_j) -- the KING-robust denominator per Manichaikul
    # et al. 2010 / SNPRelate's snpgdsIBDKING(type="KING-robust"), NOT the
    # sum Nhet_i + Nhet_j (that was the pre-fix bug; see docstring above).
    min_het = torch.minimum(Nhet_pair, Nhet_pair.T)
    # Guard against div-by-zero for pairs with no shared heterozygosity
    # (e.g. both individuals fully homozygous, or no co-observed markers);
    # such pairs get phi = 0 rather than NaN/inf (documented convention,
    # matching the prior implementation's zero-denominator handling).
    safe_min = torch.where(min_het > 0, min_het, torch.ones_like(min_het))
    phi = 0.5 - Sd / (4.0 * safe_min)
    phi = torch.where(min_het > 0, phi, torch.zeros_like(phi))

    # Sd and min_het are symmetric by construction (each is built from
    # A @ B.T + B @ A.T - 2*(C @ C.T) or elementwise min of M and M.T over
    # the same per-marker vectors), so this symmetrization is defensive/a
    # no-op guarding against floating-point non-associativity.
    phi = 0.5 * (phi + phi.T)
    phi.fill_diagonal_(0.5)
    return phi


def ld_prune_independent(
    G: Tensor,
    r2_threshold: float = 0.1,
    window: int = 500,
) -> Tensor:
    """Greedy LD pruning to a mutually-independent SNP subset.

    Sequentially scans SNPs left-to-right and keeps a SNP only if its
    pairwise r-squared against *every already-kept SNP within the trailing
    ``window``* is strictly below ``r2_threshold``. This is the same greedy
    "keep first, drop correlated followers" strategy used by PLINK
    ``--indep-pairwise`` and is the LD-pruning step feeding PC-AiR: Conomos
    et al. 2015, *Genet. Epidemiol.* 39:276, recommend pruning input markers
    to r2 < 0.1 before computing relatedness-robust principal components
    (Methods, "Estimation of Ancestry via PC-AiR"), so that the KING-robust
    kinship + PCA step is not dominated by a handful of tightly-linked
    genomic regions. This module's ``king_robust_kinship`` (Task 2) and this
    function together form the pre-PC-AiR pipeline stages of Tan et al.
    2026 (Tractor-Mix, Extended Data Fig. 1).

    Because pruning is greedy and order-dependent, the *specific* SNP kept
    out of a correlated cluster is whichever one is encountered first
    (lowest column index) -- this matches PLINK's convention and is
    sufficient for the intended downstream use (PC-AiR only needs *an*
    approximately-independent marker panel, not a canonical/optimal one).
    The property this function guarantees, and that is verified below, is
    that the *returned* kept set is mutually below the threshold: for every
    pair (j, k) of kept SNPs with k within `window` positions of j among the
    kept set, r2(j, k) < r2_threshold.

    Complexity note: `compute_r2_matrix` materializes the full (m, m) r2
    matrix up front, which is O(m^2) memory -- appropriate for the typical
    PC-AiR use case (LD-pruning a marker panel of up to a few hundred
    thousand SNPs on a single chromosome/chunk at a time), not for a
    genome-wide, unchunked panel of millions of SNPs. Chunked/streaming
    computation of `compute_r2_matrix` itself is out of scope for this task
    (Task 3 of Phase 57 Unit A); callers pruning a full genome should
    pre-chunk by chromosome or LD block.

    Missing-data / monomorphic-SNP handling
    ----------------------------------------
    `compute_r2_matrix` is not NaN-safe for genotypes that are themselves
    NaN (missing dosages): callers must impute or otherwise resolve missing
    genotypes before calling this function, exactly as required upstream
    for `king_robust_kinship`'s pairwise-complete NaN handling not applying
    here.

    A **monomorphic** SNP (zero variance across all individuals) is *not*
    a missing-data problem, though, and is handled gracefully:
    `compute_r2_matrix` centers by the column mean and divides by
    ``clamp(std, min=1e-10)``, so a zero-variance column produces an
    all-zero normalized column (0 / 1e-10 == 0) rather than a 0/0 NaN. Its
    r2 against every other SNP therefore evaluates to exactly 0.0, i.e. a
    monomorphic SNP is treated as "uncorrelated with everything": it is
    always kept (nothing can ever exceed the threshold *against* it) and it
    never blocks any other SNP from being kept. As a defense-in-depth
    measure against future changes to `compute_r2_matrix` (or an r2 input
    fed in some other way that *does* contain NaN), any NaN entry
    encountered during the greedy scan is treated as "not correlated"
    (does not block a keep) rather than raised or silently treated as
    correlated -- pruning must never spuriously *drop* a SNP because of an
    undefined comparison.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Genotype dosage matrix (diploid dosages in {0, 1, 2}, or polyploid
        dosages in [0, k]); float, no missing values.
    r2_threshold : float, default 0.1
        Maximum allowed pairwise r-squared between any two kept SNPs.
        Conomos et al. 2015 use r2 < 0.1 for PC-AiR input.
    window : int, default 500
        Only the trailing ``window`` already-kept SNPs are checked against
        each candidate (a local/banded approximation to full all-pairs
        pruning, matching PLINK's sliding-window convention and keeping the
        per-candidate cost bounded independent of how many SNPs have
        already been kept).

    Returns
    -------
    Tensor, shape (k,), dtype torch.long
        Column indices (into `G`) of the kept, mutually-independent SNPs,
        in increasing order.

    References
    ----------
    Conomos, M.P., Miller, M.B., and Thornton, T.A. (2015). Robust
    inference of population structure for ancestry prediction and
    correction of stratification in the presence of relatedness. Genetic
    Epidemiology 39(4), 276-293. ("Estimation of Ancestry via PC-AiR":
    LD-prune markers to r2 < 0.1 before PCA.)
    """
    G = G.to(torch.float64)
    m = G.shape[1]
    r2 = compute_r2_matrix(G)                # (m, m), float64, symmetric
    kept: list[int] = []
    for j in range(m):
        ok = True
        for k in kept[-window:]:
            val = r2[j, k]
            if not torch.isnan(val) and float(val) >= r2_threshold:
                ok = False
                break
        if ok:
            kept.append(j)
    return torch.tensor(kept, dtype=torch.long, device=G.device)


def pcair_partition(
    king_kinship: Tensor,
    kin_threshold: float = 0.025,
    div_threshold: float = -0.025,
) -> tuple[Tensor, Tensor]:
    """Partition individuals into an ancestry-representative *unrelated* set
    and a *related* set for PC-AiR (Conomos et al. 2015, *Genet. Epidemiol.*
    39:276).

    A pair ``(i, j)`` is "related" when ``king_kinship[i, j] > kin_threshold``.
    The unrelated set is built as a greedy Maximal-Independent-Set of the
    relatedness graph: individuals are considered in *descending order of
    ancestry-informativeness* and each is admitted to the unrelated set unless
    it is related to an individual already admitted; admitting an individual
    blocks all of its relatives from later admission. The related set is the
    complement.

    "Ancestry-informativeness" is scored by the KING *divergence* signal that
    Conomos et al. use to prioritise which member of a related group to retain:
    strongly *negative* KING-robust estimates flag pairs drawn from *different*
    ancestral populations (population structure depresses shared
    heterozygosity relative to IBS0). An individual with many partners below
    ``div_threshold`` is divergent from many others, i.e. ancestry-representative,
    and is therefore preferred when seeding the unrelated set so that each
    ancestral component is represented in the PCA basis. This ordering rule is
    a defensible torch realisation of the GENESIS strategy; the *exact*
    GENESIS ``pcair`` partition (and its handling of higher-degree relative
    graphs) is verified against the reference implementation in Task 6 of this
    Unit -- do not read this as reference-equivalence.

    Parameters
    ----------
    king_kinship : Tensor, shape (n, n)
        Symmetric KING-robust kinship matrix (e.g. from
        :func:`king_robust_kinship`). Only off-diagonal entries are used.
    kin_threshold : float, default 0.025
        Pairs with kinship strictly above this are "related". 0.025 sits
        between 3rd-degree (~0.0625) and unrelated (0.0) on the KING scale,
        the conventional cut for declaring a pair related.
    div_threshold : float, default -0.025
        Partners with kinship below this count toward an individual's
        ancestry-informativeness (divergence) score.

    Returns
    -------
    (unrel_mask, rel_mask) : tuple[Tensor, Tensor]
        Boolean masks of shape (n,). ``rel_mask == ~unrel_mask`` exactly. The
        unrelated set is guaranteed to contain no within-set pair with
        kinship above ``kin_threshold``.

    References
    ----------
    Conomos, M.P., Miller, M.B., and Thornton, T.A. (2015). Robust inference
    of population structure for ancestry prediction and correction of
    stratification in the presence of relatedness. Genetic Epidemiology
    39(4), 276-293.
    """
    K = king_kinship.to(torch.float64)
    n = K.shape[0]
    device = K.device

    related = K > kin_threshold
    related = related.clone()
    related.fill_diagonal_(False)

    # Ancestry-informativeness: number of strongly-divergent partners.
    informative = (K < div_threshold).sum(dim=1)
    # Descending order; stable so ties resolve by original index (determinism).
    order = torch.argsort(informative, descending=True, stable=True)

    unrel_mask = torch.zeros(n, dtype=torch.bool, device=device)
    blocked = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in order.tolist():
        if bool(blocked[idx]):
            continue
        unrel_mask[idx] = True
        # Everyone related to this newly-admitted member is now barred from the
        # unrelated set (guarantees the returned set is mutually unrelated).
        blocked = blocked | related[idx]

    rel_mask = ~unrel_mask
    return unrel_mask, rel_mask


def pc_air(
    G_pruned: Tensor,
    king_kinship: Tensor,
    n_pcs: int = 10,
    kin_threshold: float = 0.025,
    div_threshold: float = -0.025,
    return_internals: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """PC-AiR: relatedness-robust principal components (Conomos et al. 2015).

    Principal Components Analysis in Related samples (PC-AiR) recovers
    ancestry axes that are *not* distorted by relatedness. Ordinary PCA on a
    cohort containing families is pulled toward whichever cluster carries the
    most relatives (close relatives inflate that group's apparent variance and
    can hijack the leading PCs), so the top PCs mix ancestry with family
    structure. PC-AiR avoids this by (i) computing the PCA basis on a
    carefully-chosen *unrelated* subset that still spans the ancestral
    variation, then (ii) *projecting* the related individuals onto that basis
    via SNP loadings. The result is a set of PCs whose leading axes track
    ancestry regardless of how relatives are distributed across the cohort.

    Algorithm (Conomos et al. 2015, "Estimation of Ancestry via PC-AiR"):

    1. **Partition** (:func:`pcair_partition`): split the sample into an
       ancestry-representative unrelated set and its related complement using
       a greedy MIS over the relatedness graph, preferring
       ancestry-informative (KING-divergent) individuals.
    2. **PCA on the unrelated set**: standardize ``G_pruned`` columns by the
       *unrelated-set* mean/sd, form the unrelated GRM
       ``Zu Zu^T / m``, and eigendecompose. The top ``n_pcs`` eigenvectors
       ``U`` (unit-norm columns) are the unrelated-set PC scores.
    3. **Project the related set**: standardize the related individuals by the
       *same* unrelated-set mean/sd (``Zr``) and predict their scores with the
       out-of-sample PCA-projection formula

           scores_rel = Zr @ Zu^T @ U / (m * lambda)

       where ``lambda`` are the unrelated-set eigenvalues. This formula is the
       exact out-of-sample extension of PCA: applied to the unrelated
       individuals themselves it reproduces ``U`` identically
       (``Zu Zu^T U / (m*lambda) = GRM_u U / lambda = U``), so related and
       unrelated scores live on one common scale and one common sign
       convention -- essential for the assembled PC matrix to be usable as a
       single set of ancestry axes. (This differs from, and corrects, a
       per-component ``1/sqrt(lambda)`` mis-scaling; the projection was chosen
       for internal scale-consistency. Exact scaling/sign parity with GENESIS
       ``pcair`` is verified in Task 6 by absolute correlation, since PC signs
       are arbitrary.)
    4. **Assemble** the full ``(n, n_pcs)`` matrix in original sample order.

    Input markers should be LD-pruned (see :func:`ld_prune_independent`;
    Conomos et al. recommend r2 < 0.1) so the PCA basis is not dominated by a
    few tightly-linked regions.

    Correctness / validation status
    -------------------------------
    The partition-validity, ancestry-separation, relatedness-robustness, and
    projection-sanity tests in ``tests/test_kinship_admixed.py`` are Tier-1
    internal-correctness gates on synthetic admixed+related fixtures; they
    check internal consistency, not agreement with an external reference tool.

    Validation (Phase 57 Unit A, Task 6): compared against GENESIS ``pcair``
    on two synthetic admixed+related fixtures (see
    ``validation/external/genesis/VALIDATION_RESULTS.md`` for the full
    write-up):

    - **2-population fixture** (n=120, m=2000, 1 real ancestry axis): PC1
      ``|r| = 0.9998`` vs. GENESIS ``pcair`` PC1.
    - **3-population fixture** (n=135, m=2500, 2 real ancestry axes, via
      ``export_fixture_3pop.py`` / ``run_pcair_3pop.R``): PC1 ``|r| = 0.9992``,
      PC2 ``|r| = 0.9990`` vs. the corresponding GENESIS ``pcair`` axes.

    Every REAL ancestry axis (i.e. every PC up to the number of source
    populations minus 1) matches GENESIS to ``|r| ~ 0.999`` -- this is
    **REFERENCE-CONCORDANT**. Sub-dominant PCs beyond the number of real
    ancestry dimensions (pure sampling/LD noise eigenvectors with no
    population-structure signal to anchor them) are expected to diverge
    between independent eigensolver implementations and should not be
    interpreted as ancestry axes by callers; this divergence is correct
    behavior, not a defect. PC signs are arbitrary and are compared by
    absolute correlation throughout.

    Parameters
    ----------
    G_pruned : Tensor, shape (n, m)
        LD-pruned genotype dosage matrix (diploid {0,1,2} or polyploid [0,k]);
        float, no missing values. Cast to FP64 internally.
    king_kinship : Tensor, shape (n, n)
        KING-robust kinship matrix used only for the relatedness partition
        (:func:`king_robust_kinship`).
    n_pcs : int, default 10
        Number of principal components to return. Truncated to the unrelated
        set size if smaller; any remaining columns are zero-padded.
    kin_threshold : float, default 0.025
        Relatedness cut for the partition (see :func:`pcair_partition`).
    div_threshold : float, default -0.025
        Divergence cut for ancestry-informativeness (see
        :func:`pcair_partition`).
    return_internals : bool, default False
        If True, also return a dict of the internal unrelated-set quantities
        used to build the projection (``Zu``, ``U``, ``lam``, ``mu``, ``sd``)
        -- primarily for tests that need to verify the out-of-sample
        projection formula reproduces these values exactly. Does not change
        the default (``False``) return shape/type.

    Returns
    -------
    Tensor, shape (n, n_pcs), dtype float64
        Relatedness-robust PC scores in original sample order. PC signs are
        arbitrary (an eigenvector and its negation are equivalent).
        If ``return_internals=True``, returns a tuple ``(pcs, internals)``
        where ``internals`` is a dict with keys ``Zu`` (n_u, m) the
        standardized unrelated-set genotypes, ``U`` (n_u, k) the unrelated-set
        PCA eigenvectors, ``lam`` (k,) the corresponding eigenvalues, and
        ``mu``/``sd`` (m,) the unrelated-set column mean/sd used for
        standardization.

    References
    ----------
    Conomos, M.P., Miller, M.B., and Thornton, T.A. (2015). Robust inference
    of population structure for ancestry prediction and correction of
    stratification in the presence of relatedness. Genetic Epidemiology
    39(4), 276-293.
    """
    G_pruned = G_pruned.to(torch.float64)
    n, m = G_pruned.shape
    device = G_pruned.device

    unrel_mask, rel_mask = pcair_partition(
        king_kinship, kin_threshold=kin_threshold, div_threshold=div_threshold
    )

    # --- PCA on the unrelated set ---
    Gu = G_pruned[unrel_mask]                       # (n_u, m)
    mu = Gu.mean(dim=0)
    sd = Gu.std(dim=0).clamp(min=1e-8)
    Zu = (Gu - mu) / sd                              # (n_u, m)
    grm_u = (Zu @ Zu.T) / m                          # (n_u, n_u)

    k = min(n_pcs, grm_u.shape[0])
    ed = eigendecompose(grm_u, n_components=k)
    U = ed.eigenvectors                              # (n_u, k), unit-norm cols
    lam = ed.eigenvalues                             # (k,), descending

    pcs = torch.zeros(n, k, dtype=torch.float64, device=device)
    unrel_idx = unrel_mask.nonzero(as_tuple=True)[0]
    pcs[unrel_idx] = U

    # --- Project the related set onto the unrelated-set axes ---
    if bool(rel_mask.any()):
        Zr = (G_pruned[rel_mask] - mu) / sd          # (n_r, m), same mu/sd
        lam_safe = lam.clamp(min=1e-12)
        # Out-of-sample PCA projection: reproduces U exactly on the unrelated
        # set, so all samples share one scale and sign convention.
        proj = (Zr @ Zu.T) @ U                        # (n_r, k)
        proj = proj / (m * lam_safe).unsqueeze(0)
        rel_idx = rel_mask.nonzero(as_tuple=True)[0]
        pcs[rel_idx] = proj

    # Zero-pad if the unrelated set was smaller than the requested n_pcs.
    if k < n_pcs:
        logger.warning(
            "pc_air: requested n_pcs=%d but the unrelated set has only "
            "%d individuals (n_unrelated=%d), so only %d PCs are computable; "
            "the remaining %d columns are zero-padded. Callers must not "
            "treat these structurally-zero columns as informative "
            "PC-Relate/association covariates.",
            n_pcs, k, int(unrel_mask.sum()), k, n_pcs - k,
        )
        pad = torch.zeros(n, n_pcs - k, dtype=torch.float64, device=device)
        pcs = torch.cat([pcs, pad], dim=1)

    if return_internals:
        return pcs, {"Zu": Zu, "U": U, "lam": lam, "mu": mu, "sd": sd}

    return pcs


def pc_relate(
    G_pruned: Tensor,
    pcs: Tensor,
    n_pcs_adjust: int = 3,
    maf_min: float = 0.01,
    training_set: Tensor | None = None,
    chunk_size: int = 2000,
) -> Tensor:
    """PC-Relate ancestry-adjusted kinship estimator (Conomos et al. 2016).

    Estimates pairwise kinship coefficients that are unbiased in the presence
    of population structure / admixture by first modelling each individual's
    *individual-specific allele frequency* as a linear function of ancestry
    principal components, then forming a method-of-moments kinship estimator
    from the ancestry-adjusted genotype residuals. Because the residuals are
    taken relative to each individual's own ancestry-predicted allele
    frequency, allele-frequency divergence between subpopulations no longer
    inflates the apparent kinship of unrelated cross-population pairs (which
    is the failure mode of a naive GRM that uses a single global mean
    frequency per SNP).

    Algorithm (Conomos et al. 2016, *AJHG* 98:127, eqs. 3-5)
    -------------------------------------------------------
    1. **Individual-specific allele frequencies.** Regress each SNP's genotype
       vector ``g_.l`` (over individuals) on the design matrix
       ``D = [1 | pcs[:, :n_pcs_adjust]]`` by ordinary least squares, giving
       fitted values ``\\hat{g}_il``. The individual-specific allele frequency
       is ``mu_il = \\hat{g}_il / 2`` (diploid), clamped to
       ``[maf_min, 1 - maf_min]`` so the binomial variance term
       ``mu(1-mu)`` stays strictly positive.

       By default (``training_set=None``) the regression betas are fit using
       *all* ``n`` individuals -- the OLS hat matrix
       ``H = D (D^T D)^{-1} D^T`` (shape ``(n, n)``, independent of the number
       of markers) is formed once and applied per marker chunk, so the full
       ``(n, m)`` individual-specific-frequency matrix is never materialized.
       With ``n_pcs_adjust = 0`` the design collapses to an intercept only and
       ``mu_il`` reduces to the single global mean frequency of SNP ``l`` for
       every individual -- i.e. the *unadjusted* estimator, exposed here only
       so tests can demonstrate that the PC-adjustment removes cross-ancestry
       inflation.

       When ``training_set`` is given, the betas are instead fit using
       *only* the training-set rows -- ``beta_l = (D_train^T D_train)^{-1}
       D_train^T g_train,l`` (via ``pinv`` for rank-deficiency robustness) --
       and then *applied to every individual*, ``\\hat{g}_.l = D @ beta_l``.
       This is the GENESIS ``pcrelate`` convention (Conomos et al. 2016,
       Methods: "Estimating Individual-Specific Allele Frequencies"): fitting
       the regression on an unrelated training subset (e.g. the PC-AiR
       unrelated partition) avoids the betas themselves being biased by
       relatedness in the sample, while the fitted individual-specific
       frequencies are still produced for the full cohort. Internally this
       is implemented as a single ``(n, n_train)`` projection matrix
       ``P = D @ pinv(D_train)`` formed once and applied per marker chunk
       (``fitted = P @ g_train_chunk``), so peak memory is unaffected by the
       training-set restriction.
    2. **Kinship moment estimator.** With genotype residuals
       ``r_il = g_il - 2 mu_il`` and per-individual binomial-variance weights
       ``w_il = sqrt(mu_il (1 - mu_il))``,

           phi_ij = ( sum_l r_il r_jl )
                    / ( 4 sum_l w_il w_jl )

       where both sums run over all markers (accumulated chunk-by-chunk). This
       is eq. 5 of Conomos et al. 2016 for the off-diagonal (i != j) kinship.
    3. **Streaming.** The numerator ``sum_l r_il r_jl`` and denominator
       ``sum_l w_il w_jl`` are ``(n, n)`` accumulators built one marker chunk
       at a time; the marker-loop peak memory is ``O(n * chunk_size + n^2)``
       and does not scale with the total marker count ``m``.

    Self-kinship (diagonal)
    -----------------------
    The diagonal is set explicitly to ``0.5`` (the identity-by-descent
    self-kinship of a non-inbred individual). Conomos et al. 2016 define a
    separate self-kinship / inbreeding estimator (their eq. 6, using the
    homozygosity residual ``(g_il - 2 mu_il)^2 - ...``); implementing and
    validating that exact self-estimator is deferred to Task 6 of this Unit
    together with the GENESIS reference comparison. For the intended use here
    (a random-effect covariance whose diagonal is conventionally standardized
    to the non-inbred value) the fixed 0.5 diagonal is the documented,
    conservative choice.

    Correctness / validation status
    -------------------------------
    This function implements the published Conomos 2016 *formula*. The tests in
    ``tests/test_kinship_admixed.py`` are Tier-1 internal-correctness gates:
    (i) pedigree recovery -- injected parent-offspring pairs (true kinship
    0.25) estimate markedly above unrelated cross-population pairs (~0), and in
    the right ballpark; (ii) ancestry-adjustment direction -- the PC-adjusted
    estimator gives lower spurious cross-ancestry kinship than the unadjusted
    (``n_pcs_adjust=0``) estimator. These check internal consistency only.

    Validation (Phase 57 Unit A, Task 6) against actual GENESIS ``pcrelate``
    output (using GENESIS's own ``pcair`` PCs as input, so this isolates the
    PC-Relate step itself) on the ``validation/external/genesis`` fixture --
    see ``VALIDATION_RESULTS.md`` for the full write-up -- gives, in order of
    successive refinement:

    - r = 0.849 -- AF regression fit on *all* individuals (``training_set=None``).
    - r = 0.913 -- AF regression restricted to the unrelated training set via
      ``training_set`` (max-abs-diff 0.045). **This is the shipped default
      recommendation** (pass the PC-AiR unrelated-partition mask).
    - r = 0.937 -- using GENESIS's *own exact* unrelated training set
      (``pca$unrels``, exported via ``export_unrels.R``) rather than this
      module's independently-computed partition, isolating the AF-regression
      step to the residual formula/normalization difference alone.

    r = 0.937 is the empirically-determined **ceiling** of this
    moment-estimator implementation: additionally matching GENESIS's
    per-pair SNP filtering (per-pair MAF/missingness exclusions) was tried
    and verified to NOT close the remaining gap. The residual ~6% divergence
    is therefore attributed to GENESIS ``pcrelate``'s internal small-sample
    bias-correction / normalization machinery, which goes beyond the
    published Conomos et al. 2016 moment-estimator formula (eq. 5) implemented
    here (e.g. iterative re-weighting and the exact self-kinship/inbreeding
    estimator, eq. 6, which this function does not implement -- see
    "Self-kinship (diagonal)" above).

    **Verdict: STRONGLY CONCORDANT with GENESIS ``pcrelate`` (r ~ 0.94), NOT
    reference-equivalent.** This is an honest, documented limitation, not a
    claim of reference-equivalence. Callers who require exact numerical
    equivalence with GENESIS ``pcrelate`` (e.g. for a publication claiming
    tool-parity) should use GENESIS ``pcrelate`` directly rather than this
    function; this function is appropriate where a fast, GPU-portable,
    moment-consistent ancestry-adjusted kinship estimate is sufficient.

    Parameters
    ----------
    G_pruned : Tensor, shape (n, m)
        LD-pruned diploid dosage matrix, values in ``[0, 2]`` (float), no
        missing values. Missing values (NaN) are NOT supported and will corrupt the result — a single NaN genotype NaN-poisons entire rows/columns of the kinship accumulators; inputs must be complete (imputed) diploid dosages. Cast to FP64 internally. Column slices ``[:, s:e]`` are
        the only indexing performed, so a width-tracking wrapper exposing
        ``shape`` and 2-D column slicing may be passed in place of a raw tensor
        (used by the streaming-memory regression test).
    pcs : Tensor, shape (n, p) with p >= n_pcs_adjust
        Ancestry principal components (e.g. from :func:`pc_air`). Only the
        first ``n_pcs_adjust`` columns are used.
    n_pcs_adjust : int, default 3
        Number of leading PCs used to model individual-specific allele
        frequencies. ``0`` yields the unadjusted (single global mean freq)
        estimator.
    maf_min : float, default 0.01
        Individual-specific allele frequencies are clamped to
        ``[maf_min, 1 - maf_min]`` to keep the binomial variance positive.
    training_set : Tensor or None, default None
        Selects the individuals used to *fit* the allele-frequency regression
        betas (the GENESIS ``pcrelate`` "unrelated training set" convention;
        Conomos et al. 2016). Either a boolean mask of shape ``(n,)`` or a
        ``torch.long`` tensor of row indices. If ``None`` (default), the
        regression is fit on *all* individuals -- this is the original,
        backward-compatible behavior. When given, the fitted individual-
        specific allele frequencies ``mu_il`` are still produced for *every*
        individual (only the betas are restricted to the training rows).
        Typical usage: pass the PC-AiR unrelated-partition mask (see
        :func:`pcair_partition`), as done by :func:`admixed_grm`.
    chunk_size : int, default 2000
        Number of markers processed per streaming chunk.

    Returns
    -------
    Tensor, shape (n, n), dtype float64
        Symmetric ancestry-adjusted kinship matrix, diagonal == 0.5.

    References
    ----------
    Conomos, M.P., Reiner, A.P., Weir, B.S., and Thornton, T.A. (2016).
    Model-free estimation of recent genetic relatedness. American Journal of
    Human Genetics 98(1), 127-148.
    """
    n, m = G_pruned.shape
    # Device is derived from G_pruned (the large (n, m) marker matrix), NOT
    # from pcs (a small (n, p) side input). If a caller passes a CUDA
    # G_pruned with CPU-resident pcs (e.g. an ancestry-PC file loaded
    # separately from the genotype tensor), deriving device from pcs would
    # silently host-move every (n, c) marker chunk each iteration of the
    # streaming loop below -- the exact silent-perf-cliff class this
    # codebase treats as a bug (see CLAUDE.md "GPU kernels follow the same
    # dispatch discipline" / "never proactively move a CUDA tensor to
    # host"). Move the small tensor (pcs) to the big tensor's device instead.
    device = G_pruned.device
    pcs = pcs.to(device=device, dtype=torch.float64)

    # --- OLS design (formed once; independent of m) ---
    ones = torch.ones(n, 1, dtype=torch.float64, device=device)
    if n_pcs_adjust > 0:
        p_use = min(n_pcs_adjust, pcs.shape[1])
        D = torch.cat([ones, pcs[:, :p_use]], dim=1)     # (n, 1 + p_use)
    else:
        D = ones                                          # (n, 1) intercept only

    train_idx: Tensor | None = None
    if training_set is None:
        # Backward-compatible default: fit betas on ALL individuals via the
        # OLS hat matrix H = D (D^T D)^{-1} D^T (pinv for rank-deficiency
        # robustness), applied per marker chunk.
        proj = D @ torch.linalg.pinv(D)                    # (n, n)
    else:
        ts = training_set
        if ts.dtype == torch.bool:
            train_idx = ts.nonzero(as_tuple=True)[0].to(device)
        else:
            train_idx = ts.to(device=device, dtype=torch.long)
        D_train = D[train_idx]                              # (n_train, 1+p_use)
        # beta_l = pinv(D_train) @ g_train,l  (== (D_train^T D_train)^{-1}
        # D_train^T g_train,l for full column rank D_train; pinv is used for
        # rank-deficiency robustness). Folding beta and the "apply to all"
        # step D @ beta into one (n, n_train) projection lets the per-chunk
        # loop below stay a single matmul, exactly as in the training_set=None
        # branch above.
        proj = D @ torch.linalg.pinv(D_train)               # (n, n_train)

    num = torch.zeros(n, n, dtype=torch.float64, device=device)
    den = torch.zeros(n, n, dtype=torch.float64, device=device)

    for s in range(0, m, chunk_size):
        e = min(s + chunk_size, m)
        gc = G_pruned[:, s:e].to(torch.float64).to(device)   # (n, c)
        gc_fit = gc if train_idx is None else gc[train_idx]   # (n or n_train, c)
        fitted = proj @ gc_fit                                # (n, c); betas
        # fit on the training rows only (if given), applied to every
        # individual -- Conomos et al. 2016 / GENESIS pcrelate convention.
        mu = (fitted / 2.0).clamp(maf_min, 1.0 - maf_min)     # (n, c)
        r = gc - 2.0 * mu                                     # residuals (n, c)
        w = torch.sqrt(mu * (1.0 - mu))                       # (n, c)
        num += r @ r.T
        den += w @ w.T

    safe_den = torch.where(den > 0, den, torch.ones_like(den))
    phi = num / (4.0 * safe_den)
    phi = torch.where(den > 0, phi, torch.zeros_like(phi))
    # Symmetric by construction (A @ A.T); symmetrize defensively.
    phi = 0.5 * (phi + phi.T)
    phi.fill_diagonal_(0.5)
    return phi


@dataclass
class AdmixedGRMResult:
    """Result of the admixture-aware GRM pipeline (:func:`admixed_grm`).

    Attributes
    ----------
    pcs : Tensor, shape (n, n_pcs)
        Relatedness-robust ancestry principal components (PC-AiR).
    grm : Tensor, shape (n, n)
        Dense genomic relationship matrix ``2 * pc_relate_kinship`` (diagonal
        1.0 for non-inbred individuals).
    sparse_grm : Tensor (sparse COO), shape (n, n)
        Thresholded/sparsified GRM: off-diagonal entries with magnitude below
        ``sparsify_threshold`` are zeroed and the diagonal is set to 1.0
        (Tan et al. 2026 sparsification rule).
    king_kinship : Tensor, shape (n, n)
        KING-robust kinship matrix used for the PC-AiR partition.
    pruned_idx : Tensor, shape (k,), dtype long
        Column indices of the LD-pruned markers used for all downstream
        estimators (KING / PC-AiR / PC-Relate).
    """

    pcs: Tensor
    grm: Tensor
    sparse_grm: Tensor
    king_kinship: Tensor
    pruned_idx: Tensor


def admixed_grm(
    G: Tensor,
    n_pcs: int = 10,
    r2_threshold: float = 0.1,
    sparsify_threshold: float = 0.05,
    chunk_size: int = 2000,
    n_pcs_adjust: int = 3,
    kin_threshold: float = 0.025,
    div_threshold: float = -0.025,
    maf_min: float = 0.01,
) -> AdmixedGRMResult:
    """Admixture-aware GRM orchestrator: LD-prune -> KING -> PC-AiR -> PC-Relate.

    Composes the four Unit-A estimators into a single relatedness- and
    ancestry-aware genomic relationship matrix suitable as a GWAS random-effect
    covariance for structured / admixed cohorts (Tan et al. 2026, Tractor-Mix,
    Extended Data Fig. 1):

    1. **LD-prune** the markers to an approximately-independent panel
       (:func:`ld_prune_independent`, r2 < ``r2_threshold``).
    2. **KING-robust kinship** on the pruned panel
       (:func:`king_robust_kinship`) for the relatedness partition.
    3. **PC-AiR** relatedness-robust ancestry PCs (:func:`pc_air`).
    4. **PC-Relate** ancestry-adjusted kinship (:func:`pc_relate`), using the
       first ``n_pcs_adjust`` PC-AiR PCs to model individual-specific allele
       frequencies. The allele-frequency regression betas are fit on the
       PC-AiR *unrelated* partition only (recomputed here via
       :func:`pcair_partition` on the same ``king``/``kin_threshold``/
       ``div_threshold`` that ``pc_air`` used internally, so it is exactly
       the partition ``pc_air`` based its PCA on) and then applied to every
       individual, per the GENESIS ``pcrelate`` convention (Conomos et al.
       2016) -- see :func:`pc_relate`'s ``training_set`` parameter.

    The dense GRM is ``2 * kinship`` (diagonal 1.0). A sparse companion is
    produced by the fastGWA / Tan-2026 sparsification rule: off-diagonal
    entries with ``|entry| < sparsify_threshold`` are set to 0, the diagonal is
    set to 1.0, and the result is stored as a ``torch.sparse_coo_tensor`` for
    memory-efficient downstream matrix-vector products.

    Validation status
    -----------------
    Internal-correctness only (shape coherence + pedigree recovery via the
    PC-Relate Tier-1 gate). Reference-equivalence to GENESIS is Task 6.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Diploid dosage matrix, values in ``[0, 2]`` (float), no missing values.
    n_pcs : int, default 10
        Number of PC-AiR ancestry PCs to compute/return.
    r2_threshold : float, default 0.1
        LD-pruning r-squared threshold.
    sparsify_threshold : float, default 0.05
        Off-diagonal GRM entries below this magnitude are zeroed in
        ``sparse_grm``.
    chunk_size : int, default 2000
        Marker-chunk size for the streaming KING / PC-Relate estimators.
    n_pcs_adjust : int, default 3
        Number of leading PC-AiR PCs used by PC-Relate for the
        individual-specific-frequency adjustment.
    kin_threshold, div_threshold : float
        Relatedness / divergence cuts for the PC-AiR partition.
    maf_min : float, default 0.01
        Individual-specific allele-frequency clamp for PC-Relate.

    Returns
    -------
    AdmixedGRMResult
        ``pcs`` (n, n_pcs), dense ``grm`` (n, n), ``sparse_grm`` (n, n sparse
        COO), ``king_kinship`` (n, n), and ``pruned_idx`` (k,).

    References
    ----------
    Tan et al. (2026). Tractor-Mix (Extended Data Fig. 1).
    Conomos et al. (2015, 2016); Manichaikul et al. (2010).
    """
    G = G.to(torch.float64)
    n = G.shape[0]
    device = G.device

    keep = ld_prune_independent(G, r2_threshold=r2_threshold)
    Gp = G[:, keep]
    king = king_robust_kinship(Gp, chunk_size=chunk_size)
    pcs = pc_air(
        Gp, king, n_pcs=n_pcs,
        kin_threshold=kin_threshold, div_threshold=div_threshold,
    )
    # pc_air does not itself return the unrelated-set mask it used, so
    # recompute it here from the same king_kinship / kin_threshold /
    # div_threshold -- pcair_partition is a deterministic function of those
    # three inputs, so this reproduces exactly the partition pc_air based its
    # PCA basis on. Threading it through as PC-Relate's training_set fits the
    # allele-frequency regression on the unrelated individuals only (GENESIS
    # pcrelate convention) while still producing fitted frequencies -- and
    # kinship estimates -- for every individual.
    unrel_mask, _ = pcair_partition(
        king, kin_threshold=kin_threshold, div_threshold=div_threshold,
    )
    kin = pc_relate(
        Gp, pcs, n_pcs_adjust=n_pcs_adjust, maf_min=maf_min,
        training_set=unrel_mask, chunk_size=chunk_size,
    )

    grm = 2.0 * kin                                     # dense; diagonal 1.0

    # --- Sparsify: |off-diag| < threshold -> 0, diagonal -> 1.0 ---
    grm_sp = grm.clone()
    off_mask = grm_sp.abs() < sparsify_threshold
    grm_sp[off_mask] = 0.0
    diag_idx = torch.arange(n, device=device)
    grm_sp[diag_idx, diag_idx] = 1.0
    sparse_grm = grm_sp.to_sparse_coo()

    return AdmixedGRMResult(
        pcs=pcs,
        grm=grm,
        sparse_grm=sparse_grm,
        king_kinship=king,
        pruned_idx=keep,
    )
