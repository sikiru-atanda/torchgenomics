"""Test suite for synthetic 2-population admixed+related fixture (Task 1)
and for torchgenomics.linalg.kinship_admixed.king_robust_kinship (Task 2).

King-robust reference: Manichaikul et al. 2010, Bioinformatics 26:2867,
eq. 11 (KING-robust between-family kinship estimator).
"""
from __future__ import annotations
import torch
from tests.fixtures.admixed.make_admixed import make_admixed


def test_make_admixed_shapes_and_structure():
    d = make_admixed(n_per_pop=50, n_related_pairs=10, m=400, seed=3)
    n = d["G"].shape[0]
    assert d["G"].shape == (n, 400)
    assert d["G"].dtype == torch.float64
    assert d["G"].min() >= 0 and d["G"].max() <= 2.0
    # true_kinship symmetric, self-kinship 0.5 on diagonal
    K = d["true_kinship"]
    assert torch.allclose(K, K.T, atol=1e-9)
    assert torch.allclose(torch.diag(K), torch.full((n,), 0.5, dtype=torch.float64), atol=1e-9)
    # related pairs have elevated kinship vs a random unrelated pair
    i, j, _ = d["related_pairs"][0]
    assert K[i, j] >= 0.2
    # determinism
    d2 = make_admixed(n_per_pop=50, n_related_pairs=10, m=400, seed=3)
    assert torch.equal(d["G"], d2["G"])


def test_make_admixed_po_transmission_is_mendelian():
    """Guards Fix 2: parent-offspring genotype similarity must be markedly
    higher than an unrelated cross-population pair, consistent with the
    labeled true_kinship=0.25. Under the old torch.binomial(clamp(G,max=1),
    0.5) transmission, a homozygous-alt parent (dosage 2) was wrongly given
    only a 50% chance of transmitting the alt allele, which pulls the
    empirical parent-child correlation below what a true kinship of 0.25
    implies. This test computes the per-pair genotype correlation (a proxy
    for the KING-style similarity metrics validated elsewhere) for every
    parent-offspring pair and for a random unrelated cross-population pair,
    and asserts PO pairs are markedly more similar.
    """
    d = make_admixed(n_per_pop=50, n_related_pairs=10, m=400, seed=3)
    G = d["G"]
    n = G.shape[0]

    def _pair_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = a.norm() * b.norm()
        if float(denom) == 0.0:
            return 0.0
        return float((a @ b) / denom)

    po_corrs = [_pair_corr(G[i], G[j]) for (i, j, _) in d["related_pairs"]]
    mean_po_corr = sum(po_corrs) / len(po_corrs)

    # unrelated cross-population pairs (population 0 index 0..49, population
    # 1 index 50..99): none of these indices appear in related_pairs.
    unrel_pairs = [(0, n - 1), (1, n - 2), (2, n - 3)]
    unrel_corrs = [_pair_corr(G[i], G[j]) for (i, j) in unrel_pairs]
    mean_unrel_corr = sum(unrel_corrs) / len(unrel_corrs)

    assert mean_po_corr > mean_unrel_corr + 0.15, (
        f"PO genotype correlation {mean_po_corr} not markedly above "
        f"unrelated-pair correlation {mean_unrel_corr}"
    )
    # KING-style het/opposite-homozygote counts: PO pairs should show far
    # fewer opposite-homozygote markers than an unrelated pair, since a
    # homozygous-alt (or homozygous-ref) parent now ALWAYS transmits the
    # matching allele and can never produce an opposite-homozygote child.
    def _n_opp_hom(a: torch.Tensor, b: torch.Tensor) -> int:
        return int((((a == 0) & (b == 2)) | ((a == 2) & (b == 0))).sum())

    po_opp_hom = [_n_opp_hom(G[i], G[j]) for (i, j, _) in d["related_pairs"]]
    unrel_opp_hom = [_n_opp_hom(G[i], G[j]) for (i, j) in unrel_pairs]
    assert (sum(po_opp_hom) / len(po_opp_hom)) < (sum(unrel_opp_hom) / len(unrel_opp_hom))


def test_make_admixed_rejects_too_many_related_pairs():
    """Guards the n_related_pairs <= n_per_pop // 2 assertion (Fix 3 minor):
    the parent=r, child=n_per_pop-1-r scheme collides once n_related_pairs
    exceeds half the population.
    """
    import pytest
    with pytest.raises(AssertionError):
        make_admixed(n_per_pop=10, n_related_pairs=6, m=20, seed=0)


def test_king_robust_hand_computed():
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship
    # 3 individuals, 6 markers, diploid dosages
    # ind0: 1 1 0 2 1 0   ind1: 1 1 2 0 1 2   ind2: 1 1 1 1 0 0
    G = torch.tensor([
        [1, 1, 0, 2, 1, 0],
        [1, 1, 2, 0, 1, 2],
        [1, 1, 1, 1, 0, 0],
    ], dtype=torch.float64)
    phi = king_robust_kinship(G)
    assert phi.shape == (3, 3)
    assert torch.allclose(torch.diag(phi), torch.full((3,), 0.5, dtype=torch.float64))
    # Pair (0,1): both-het markers = {0,1,4} -> N_AaAa=3; opposite-hom markers:
    # marker 3 (ind0=2, ind1=0 -> opp), marker 2 (ind0=0, ind1=2 -> opp),
    # marker 5 (ind0=0, ind1=2 -> opp) -> N_AAaa=3
    # N_Aa^0 = #het in ind0 = markers {0,1,4} = 3 ; N_Aa^1 = {0,1,4} = 3
    # phi_01 = (3 - 2*3) / (3 + 3) = -3/6 = -0.5
    expected_01 = (3 - 2 * 3) / (3 + 3)
    assert abs(float(phi[0, 1]) - expected_01) < 1e-9, f"got {float(phi[0,1])}, want {expected_01}"
    # symmetry
    assert torch.allclose(phi, phi.T)


def test_king_robust_fixture_parent_offspring_vs_unrelated():
    """Tier-1 correctness check beyond the hand-computed pin.

    Uses the Task-1 synthetic admixed+related fixture
    (tests/fixtures/admixed/make_admixed.py) to check a directional
    property of the KING-robust estimator that a transposed or miscounted
    implementation would violate: parent-offspring pairs can *never* be
    opposite homozygotes at any marker (Mendelian transmission guarantees
    at least one shared allele), so N_AAaa == 0 for every PO pair and their
    KING-robust kinship must sit well above the kinship of random
    cross-population "unrelated" pairs, which is expected to trend
    negative/near-zero (the defining robustness property of the estimator
    under population structure per Manichaikul et al. 2010, Section 2.2).

    This is an internal-consistency / sanity check, NOT a reference-tool
    comparison (that is Task 6, against SNPRelate's snpgdsIBDKING).
    """
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship

    fixture = make_admixed(n_per_pop=60, n_related_pairs=10, m=2000, fst=0.15, seed=1)
    G = fixture["G"]
    pop_label = fixture["pop_label"]
    related_pairs = fixture["related_pairs"]

    phi = king_robust_kinship(G, chunk_size=500)

    po_kinships = torch.tensor([float(phi[p, c]) for p, c, _deg in related_pairs])
    assert po_kinships.numel() > 0

    # Cross-population pairs: population 0 is [0, n_per_pop), population 1
    # is [n_per_pop, 2*n_per_pop) by construction of make_admixed. None of
    # these pairs were injected as related.
    pop0_idx = (pop_label == 0).nonzero(as_tuple=True)[0]
    pop1_idx = (pop_label == 1).nonzero(as_tuple=True)[0]
    cross_kinships = phi[pop0_idx][:, pop1_idx].reshape(-1)

    # Parent-offspring kinship (true value 0.25) must be markedly higher
    # than the mean of cross-population unrelated-pair kinship.
    assert po_kinships.mean() > cross_kinships.mean() + 0.1, (
        f"PO mean {po_kinships.mean():.4f} not markedly above "
        f"cross-pop mean {cross_kinships.mean():.4f}"
    )
    # Every individual PO pair should exceed every cross-population pair's
    # mean by a wide margin -- guards against a transposed/miscounted
    # N_AAaa (which would instead push PO pairs toward zero/negative).
    assert po_kinships.min() > cross_kinships.mean(), (
        f"weakest PO pair {po_kinships.min():.4f} does not exceed "
        f"cross-pop mean {cross_kinships.mean():.4f}"
    )
    # KING-robust's defining property: cross-population unrelated pairs
    # trend negative/near-zero (population structure inflates IBS0 without
    # inflating shared heterozygosity), rather than the positive value a
    # naive homogeneous-population estimator would report given Fst > 0.
    assert cross_kinships.mean() < 0.05, (
        f"cross-pop mean kinship {cross_kinships.mean():.4f} unexpectedly high "
        f"(sanity check on KING-robust's structure-robustness property)"
    )


def test_king_robust_nan_handling():
    """NaN (missing) genotypes must be excluded per pair-marker, not crash
    or propagate to NaN kinship.

    Construct two individuals identical except that a block of markers is
    set to NaN in one of them. Those markers must simply drop out of both
    the pairwise counts and (for the NaN-carrying individual) its own
    N_Aa^i denominator -- i.e. results must match manually recomputing the
    formula on the observed (non-NaN) markers only.
    """
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship

    # ind0: all heterozygous; ind1: heterozygous except last 2 markers are
    # opposite-homozygote mismatches, one of which is later corrupted to NaN.
    ind0 = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float64)
    ind1 = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2], dtype=torch.float64)
    G_full = torch.stack([ind0, ind1])
    phi_full = king_robust_kinship(G_full)
    # N_AaAa = 6, N_AAaa = 2 (markers 6,7), N_Aa^0 = 6, N_Aa^1 = 6
    expected_full = (6 - 2 * 2) / (6 + 6)
    assert abs(float(phi_full[0, 1]) - expected_full) < 1e-9

    # Now corrupt marker index 6 to NaN in ind1: it must be dropped from
    # N_AAaa (pairwise) and must not crash the computation.
    ind1_nan = ind1.clone()
    ind1_nan[6] = float("nan")
    G_nan = torch.stack([ind0, ind1_nan])
    phi_nan = king_robust_kinship(G_nan)
    assert not torch.isnan(phi_nan).any(), "kinship must not contain NaN even with missing genotypes"
    # N_AaAa unaffected (marker 6 wasn't a het,het marker either way) = 6
    # N_AAaa drops from 2 to 1 (marker 6 dropped, marker 7 remains an
    # opposite-homozygote match) -> phi = (6 - 2*1) / (6 + 6) = 4/12
    expected_nan = (6 - 2 * 1) / (6 + 6)
    assert abs(float(phi_nan[0, 1]) - expected_nan) < 1e-9, (
        f"got {float(phi_nan[0,1])}, want {expected_nan}"
    )

    # Also verify a het marker turned NaN drops out of the NaN-carrier's own
    # N_Aa^i count and out of N_AaAa for the pair -- AND (pairwise-complete
    # fix) drops out of the *other* individual's N_Aa^j count too, since
    # marker 0 is no longer co-observed for the pair.
    ind1_nan2 = ind1.clone()
    ind1_nan2[0] = float("nan")  # marker 0 was a shared het marker
    G_nan2 = torch.stack([ind0, ind1_nan2])
    phi_nan2 = king_robust_kinship(G_nan2)
    # N_AaAa drops to 5 (marker 0 excluded, both were het there).
    # N_AAaa unaffected = 2 (markers 6,7 opposite-homozygote; marker 0 was
    # never part of N_AAaa).
    # Pairwise-complete denominator (Fix 1): N_Aa^0(pairwise) = # markers
    # where ind0 is het AND ind1 is observed = {1,2,3,4,5} = 5 (marker 0
    # excluded because ind1 is NaN there, even though ind0 itself is
    # observed at marker 0 -- this is the behavior change from the fix).
    # N_Aa^1(pairwise) = # markers where ind1 is het AND ind0 is observed =
    # {1,2,3,4,5} = 5 (marker 0 excluded because ind1 itself is NaN there).
    # denom = 5 + 5 = 10 (was 6 + 5 = 11 under the old own-count denominator,
    # which wrongly included ind0's marker-0 het count even though marker 0
    # is not part of the pairwise-complete marker set for this pair).
    expected_nan2 = (5 - 2 * 2) / (5 + 5)
    assert abs(float(phi_nan2[0, 1]) - expected_nan2) < 1e-9, (
        f"got {float(phi_nan2[0,1])}, want {expected_nan2}"
    )
    assert not torch.isnan(phi_nan2).any()


def test_king_robust_pairwise_complete_denominator_excludes_partner_missing():
    """Directly proves Fix 1: the denominator N_Aa^i must be restricted to
    the pairwise-complete marker set (markers where BOTH i and j are
    observed), not individual i's own non-missing markers.

    Construct 2 individuals, 3 markers:
        ind0: 1 (het), 1 (het), 0 (hom0)
        ind1: 1 (het), NaN,     2 (hom2)

    Marker 1 is a het marker for ind0 but NaN for ind1. Under the
    pairwise-complete convention, marker 1 must be EXCLUDED from N_Aa^0 in
    the denominator (since ind1 is not observed there), even though ind0
    itself has a valid, heterozygous genotype at that marker.

    Hand-derivation:
    - N_AaAa (both-het, co-observed): marker 0 only -> 1.
      (marker 1 excluded because ind1 is NaN there.)
    - N_AAaa (opposite-homozygote, co-observed): marker 2 (ind0=0, ind1=2)
      -> 1.
    - N_Aa^0(pairwise) = # markers where ind0 het AND ind1 observed
      = {marker 0} = 1 (marker 1 dropped: ind0 is het there, but ind1 is
      NaN, so it must NOT count toward the denominator).
    - N_Aa^1(pairwise) = # markers where ind1 het AND ind0 observed
      = {marker 0} = 1 (marker 1: ind1 itself is NaN there, so it was
      never counted as a het marker for ind1 either).
    - denom = 1 + 1 = 2.
    - phi_01 = (N_AaAa - 2*N_AAaa) / denom = (1 - 2*1) / 2 = -0.5.

    Contrast: a (WRONG) naive own-count denominator would instead use
    N_Aa^0(own) = 2 (markers 0 and 1, ind0's own het count, ignoring ind1's
    missingness at marker 1) and N_Aa^1(own) = 1, giving denom = 3 and
    phi_01 = (1 - 2) / 3 = -1/3 != -0.5. The assertion below pins the
    pairwise-complete value (-0.5), which would fail under the old
    (pre-fix) implementation.
    """
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship

    G = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, float("nan"), 2.0],
    ], dtype=torch.float64)
    phi = king_robust_kinship(G)
    expected = (1 - 2 * 1) / (1 + 1)
    assert abs(expected - (-0.5)) < 1e-12  # sanity on the hand derivation itself
    assert abs(float(phi[0, 1]) - expected) < 1e-9, (
        f"got {float(phi[0,1])}, want {expected} (pairwise-complete); "
        f"a naive own-count denominator would give {(1 - 2) / 3:.6f} instead"
    )
    assert not torch.isnan(phi).any()


def test_ld_prune_drops_correlated():
    """Task 3 brief test: perfectly-correlated duplicate SNP must be dropped,
    and the surviving kept set must be mutually below the r2 threshold."""
    from torchgenomics.linalg.kinship_admixed import ld_prune_independent
    torch.manual_seed(0)
    base = torch.randint(0, 3, (60, 20)).to(torch.float64)   # 60 indiv, 20 SNPs
    # duplicate SNP 0 into SNP 1 (perfectly correlated) -> one must be pruned
    base[:, 1] = base[:, 0]
    keep = ld_prune_independent(base, r2_threshold=0.1)
    assert keep.dtype == torch.long
    # not both 0 and 1 survive
    ks = set(keep.tolist())
    assert not (0 in ks and 1 in ks)
    # kept SNPs are mutually below r2 threshold
    from torchgenomics.ld._pairwise import compute_r2_matrix
    r2 = compute_r2_matrix(base[:, keep])
    off = r2 - torch.diag(torch.diag(r2))
    assert float(off.abs().max()) < 0.1 + 1e-6


def test_ld_prune_block_of_three_collapses_to_one():
    """A block of 3 mutually-correlated SNPs (all identical dosage columns,
    hence pairwise r2 == 1.0 among them) must collapse to exactly 1 survivor
    under greedy pruning, while genuinely independent SNPs elsewhere all
    survive. This exercises the "mutual" independence property beyond the
    brief's pairwise-duplicate case: greedy pruning must correctly propagate
    the exclusion transitively across a whole correlated block, not just a
    single pair.
    """
    from torchgenomics.linalg.kinship_admixed import ld_prune_independent
    torch.manual_seed(1)
    n = 80
    # 5 independent SNPs drawn i.i.d.
    indep = torch.randint(0, 3, (n, 5)).to(torch.float64)
    # a block of 3 SNPs that are all identical (perfectly correlated with
    # each other and with nothing else)
    block_base = torch.randint(0, 3, (n, 1)).to(torch.float64)
    block = block_base.repeat(1, 3)
    G = torch.cat([indep, block], dim=1)  # columns 0-4 independent, 5-7 the block
    keep = ld_prune_independent(G, r2_threshold=0.1)
    ks = set(keep.tolist())
    # exactly one of {5, 6, 7} survives
    assert len(ks & {5, 6, 7}) == 1
    # all 5 independent SNPs survive (they are i.i.d. random, so with n=80
    # and 5 SNPs the odds of a spurious r2>=0.1 collision are negligible,
    # and none of them share the block's genotype pattern)
    assert {0, 1, 2, 3, 4}.issubset(ks)
    # mutual independence of the final kept set
    from torchgenomics.ld._pairwise import compute_r2_matrix
    r2 = compute_r2_matrix(G[:, keep])
    off = r2 - torch.diag(torch.diag(r2))
    assert float(off.abs().max()) < 0.1 + 1e-6


def test_ld_prune_monomorphic_snp_is_kept_and_does_not_block():
    """A monomorphic SNP (zero variance across all individuals) has an
    undefined correlation with every other SNP in the strict mathematical
    sense (0/0), but torchgenomics.ld._pairwise.compute_r2_matrix resolves
    this via its variance floor (clamp(std, min=_EPS)) to r2 == 0 rather than
    NaN -- i.e. a monomorphic SNP is treated as "uncorrelated with
    everything". This test pins that a monomorphic SNP therefore (a) always
    survives pruning itself (nothing can exceed the threshold against it),
    and (b) never blocks any other SNP from being kept.
    """
    from torchgenomics.linalg.kinship_admixed import ld_prune_independent
    torch.manual_seed(2)
    n = 50
    mono = torch.full((n, 1), 1.0, dtype=torch.float64)  # constant column
    others = torch.randint(0, 3, (n, 4)).to(torch.float64)
    G = torch.cat([mono, others], dim=1)  # column 0 is monomorphic
    keep = ld_prune_independent(G, r2_threshold=0.1)
    ks = set(keep.tolist())
    assert 0 in ks
    # no NaN ever leaks into the kept-set r2 check
    from torchgenomics.ld._pairwise import compute_r2_matrix
    r2 = compute_r2_matrix(G[:, keep])
    assert not torch.isnan(r2).any()


# ---------------------------------------------------------------------------
# Task 4: PC-AiR (relatedness-robust principal components; Conomos 2015)
# ---------------------------------------------------------------------------
def _sep_ratio(pc1: torch.Tensor, pop: torch.Tensor) -> float:
    """Sign-invariant ancestry-separation ratio on a single PC axis:
    |mean_pop0 - mean_pop1| / pooled within-population std. Larger = cleaner
    ancestry separation. Sign of the axis is irrelevant (absolute gap)."""
    a = pc1[pop == 0]
    b = pc1[pop == 1]
    gap = abs(float(a.mean() - b.mean()))
    pooled = float(torch.sqrt(0.5 * (a.var(unbiased=False) + b.var(unbiased=False))).clamp(min=1e-12))
    return gap / pooled


def _plain_pca_scores(G: torch.Tensor, n_pcs: int) -> torch.Tensor:
    """Standard PCA on the FULL sample (no relatedness correction): standardize
    all columns by full-sample mean/sd, form the GRM, return top-n_pcs
    eigenvectors as scores. This is the baseline PC-AiR must beat."""
    from torchgenomics.linalg.eigh import eigendecompose
    G = G.to(torch.float64)
    mu = G.mean(dim=0)
    sd = G.std(dim=0).clamp(min=1e-8)
    Z = (G - mu) / sd
    grm = (Z @ Z.T) / G.shape[1]
    ed = eigendecompose(grm, n_components=n_pcs)
    return ed.eigenvectors


def test_pc_air_separates_ancestry_and_partitions():
    from torchgenomics.linalg.kinship_admixed import (
        king_robust_kinship, ld_prune_independent, pc_air,
    )
    d = make_admixed(n_per_pop=60, n_related_pairs=15, m=800, fst=0.15, seed=7)
    G = d["G"]
    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)
    pcs = pc_air(Gp, phi, n_pcs=5)
    assert pcs.shape == (G.shape[0], 5)
    assert pcs.dtype == torch.float64
    pop = d["pop_label"]
    pc1 = pcs[:, 0]
    sep = abs(float(pc1[pop == 0].mean() - pc1[pop == 1].mean()))
    within = float(pc1.std())
    assert sep > within, f"PC1 ancestry separation {sep} not > within-scatter {within}"


def test_pc_air_partition_is_a_valid_unrelated_set():
    """CORRECTNESS BAR 1: the selected 'unrelated' set must contain NO pair
    whose KING kinship exceeds kin_threshold, and the related set must be
    exactly its complement."""
    from torchgenomics.linalg.kinship_admixed import (
        king_robust_kinship, ld_prune_independent, pcair_partition,
    )
    d = make_admixed(n_per_pop=60, n_related_pairs=25, m=800, fst=0.12, seed=11)
    G = d["G"]
    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)
    kin_threshold = 0.025
    unrel_mask, rel_mask = pcair_partition(phi, kin_threshold=kin_threshold)
    n = G.shape[0]
    assert unrel_mask.shape == (n,) and rel_mask.shape == (n,)
    # partition: related is exactly the complement of unrelated
    assert bool((rel_mask == ~unrel_mask).all())
    assert not bool((unrel_mask & rel_mask).any())
    assert int(unrel_mask.sum()) + int(rel_mask.sum()) == n
    # the unrelated set is genuinely unrelated: no within-set pair > threshold
    u_idx = unrel_mask.nonzero(as_tuple=True)[0]
    sub = phi[u_idx][:, u_idx].clone()
    sub.fill_diagonal_(0.0)
    assert float(sub.max()) <= kin_threshold, (
        f"unrelated set has a pair with kinship {float(sub.max())} "
        f"> kin_threshold {kin_threshold}"
    )
    # at least the injected parents/children cannot ALL be in the unrelated
    # set: each PO pair (0.25 >> 0.025) forces at least one member out.
    assert int(rel_mask.sum()) >= 1


def test_pc_air_is_robust_to_concentrated_relatedness():
    """THE defining PC-AiR property (CORRECTNESS BAR 3): relatedness must not
    distort the ancestry axis.

    We build a cohort in which a large, tightly-related family is concentrated
    in ONE population: a founder in population 0 is copied into a block of
    near-clonal relatives (KING kinship ~0.5 among them, well above threshold).
    On such a cohort ordinary PCA-on-everyone is pulled toward the family
    cluster -- its leading PC becomes a *family* axis, and ancestry leaks onto
    a later PC. PC-AiR, by building its basis on the unrelated subset (which
    retains only ONE member of the family), keeps ancestry on PC1.

    We assert (a) PC-AiR's PC1 cleanly separates ancestry, (b) it does so
    STRICTLY BETTER than plain PCA's PC1, and (c) plain PCA is genuinely
    contaminated -- ancestry separates better on some *later* plain PC than on
    its PC1 (proof its PC1 was hijacked, not merely weaker). All comparisons
    use the sign-invariant absolute between-population gap over pooled
    within-population scatter (see _sep_ratio), never raw signs.
    """
    from torchgenomics.linalg.kinship_admixed import (
        king_robust_kinship, ld_prune_independent, pc_air,
    )
    # Clean 2-population base (no injected pairs), modest Fst so ancestry is
    # real but not overwhelming.
    d = make_admixed(n_per_pop=60, n_related_pairs=0, m=1000, fst=0.06, seed=5)
    G = d["G"].clone()
    pop = d["pop_label"]
    # Inject a large near-clonal family into population 0: 30 of its 60 members
    # become near-copies of the founder (index 0), each with a small
    # per-marker mutation rate so they are near-identical but not degenerate.
    gen = torch.Generator().manual_seed(123)
    fam_size, mut_rate = 30, 0.03
    founder = G[0].clone()
    for i in range(1, fam_size):
        child = founder.clone()
        mut = torch.rand(G.shape[1], generator=gen) < mut_rate
        n_mut = int(mut.sum())
        child[mut] = torch.randint(0, 3, (n_mut,), generator=gen).to(torch.float64)
        G[i] = child

    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)

    pcair_pcs = pc_air(Gp, phi, n_pcs=5)
    plain_pcs = _plain_pca_scores(Gp, n_pcs=5)

    pcair_sep = _sep_ratio(pcair_pcs[:, 0], pop)
    plain_pc1_sep = _sep_ratio(plain_pcs[:, 0], pop)
    plain_best_later = max(_sep_ratio(plain_pcs[:, j], pop) for j in range(1, 5))

    # (a) PC-AiR PC1 itself cleanly separates ancestry.
    assert pcair_sep > 1.0, f"PC-AiR PC1 fails to separate ancestry (ratio {pcair_sep:.3f})"
    # (b) strictly better than plain PCA's (hijacked) PC1 -- the whole point.
    assert pcair_sep > plain_pc1_sep, (
        f"PC-AiR PC1 ancestry separation {pcair_sep:.3f} not better than plain "
        f"PCA PC1 {plain_pc1_sep:.3f} -- PC-AiR failed its defining property"
    )
    # (c) plain PCA's PC1 was genuinely hijacked by relatedness: ancestry
    # separates better on a later plain PC than on plain PC1.
    assert plain_best_later > plain_pc1_sep, (
        f"expected plain PCA PC1 to be hijacked by the family cluster "
        f"(best later plain PC {plain_best_later:.3f} vs plain PC1 "
        f"{plain_pc1_sep:.3f})"
    )


def test_pc_air_related_individuals_project_to_correct_population():
    """CORRECTNESS BAR 4 (projection sanity): related individuals (all in
    population 0 by fixture construction) must project onto population 0's
    side of PC1 -- i.e. cluster with the UNRELATED population-0 members and
    away from population 1, not collapse to 0 or land on the wrong side."""
    from torchgenomics.linalg.kinship_admixed import (
        king_robust_kinship, ld_prune_independent, pc_air, pcair_partition,
    )
    d = make_admixed(n_per_pop=60, n_related_pairs=25, m=1000, fst=0.12, seed=9)
    G = d["G"]
    pop = d["pop_label"]
    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)
    pcs = pc_air(Gp, phi, n_pcs=5)
    unrel_mask, rel_mask = pcair_partition(phi)

    pc1 = pcs[:, 0]
    # reference population-1 (unrelated ancestry) centre
    pop1_center = float(pc1[pop == 1].mean())
    unrel_pop0 = (unrel_mask & (pop == 0))
    rel_pop0 = (rel_mask & (pop == 0))
    assert bool(rel_pop0.any()), "fixture should place related individuals in pop 0"
    unrel_pop0_center = float(pc1[unrel_pop0].mean())
    rel_pop0_center = float(pc1[rel_pop0].mean())

    # related pop-0 individuals sit on the SAME side of the pop-1 centre as
    # the unrelated pop-0 individuals (sign-invariant: compare displacement
    # direction relative to the pop-1 reference).
    assert (unrel_pop0_center - pop1_center) * (rel_pop0_center - pop1_center) > 0, (
        f"related pop-0 projected to the wrong side: rel {rel_pop0_center:.4f}, "
        f"unrel {unrel_pop0_center:.4f}, pop1 {pop1_center:.4f}"
    )
    # and they do not collapse to ~0: their distance from the pop-1 cluster is
    # a substantial fraction of the unrelated pop-0 distance (>= 40%).
    unrel_dist = abs(unrel_pop0_center - pop1_center)
    rel_dist = abs(rel_pop0_center - pop1_center)
    assert rel_dist > 0.4 * unrel_dist, (
        f"related pop-0 collapsed toward the population boundary: "
        f"rel_dist {rel_dist:.4f} vs unrel_dist {unrel_dist:.4f}"
    )
