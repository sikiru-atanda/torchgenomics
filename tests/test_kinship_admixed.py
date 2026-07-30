"""Tests for torchgenomics.linalg.kinship_admixed.king_robust_kinship.

Reference: Manichaikul et al. 2010, Bioinformatics 26:2867, eq. 11
(KING-robust between-family kinship estimator).
"""
from __future__ import annotations
import torch


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
    from tests.fixtures.admixed.make_admixed import make_admixed
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship

    fixture = make_admixed(n_per_pop=60, n_related_pairs=10, m=2000, fst=0.15, seed=1)
    G = fixture["G"]
    pop_label = fixture["pop_label"]
    related_pairs = fixture["related_pairs"]
    n_per_pop = 60

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

    # ind0: all heterozygous; ind1: heterozygous except last 4 markers are
    # opposite-homozygote mismatches, of which 2 are corrupted to NaN in ind1.
    ind0 = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float64)
    ind1 = torch.tensor([1, 1, 1, 1, 1, 1, 2, 2], dtype=torch.float64)
    G_full = torch.stack([ind0, ind1])
    phi_full = king_robust_kinship(G_full)
    # N_AaAa = 6, N_AAaa = 2 (markers 6,7), N_Aa^0 = 6, N_Aa^1 = 6
    expected_full = (6 - 2 * 2) / (6 + 6)
    assert abs(float(phi_full[0, 1]) - expected_full) < 1e-9

    # Now corrupt marker index 6 to NaN in ind1: it must be dropped from
    # N_AAaa (pairwise) AND from ind1's own het count is unaffected (marker
    # 6 wasn't heterozygous in ind1 anyway) but must not contribute to
    # N_AAaa nor crash the computation.
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
