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
    # N_Aa^i count and out of N_AaAa for the pair.
    ind1_nan2 = ind1.clone()
    ind1_nan2[0] = float("nan")  # marker 0 was a shared het marker
    G_nan2 = torch.stack([ind0, ind1_nan2])
    phi_nan2 = king_robust_kinship(G_nan2)
    # N_AaAa drops to 5 (marker 0 excluded); N_Aa^1 drops to 5 (ind1's own
    # het count no longer includes the NaN'd marker 0); N_Aa^0 stays 6
    # (ind0's own dosage at marker 0 is still 1, unaffected by ind1's NaN).
    # N_AAaa unaffected = 2.
    expected_nan2 = (5 - 2 * 2) / (6 + 5)
    assert abs(float(phi_nan2[0, 1]) - expected_nan2) < 1e-9, (
        f"got {float(phi_nan2[0,1])}, want {expected_nan2}"
    )
    assert not torch.isnan(phi_nan2).any()
