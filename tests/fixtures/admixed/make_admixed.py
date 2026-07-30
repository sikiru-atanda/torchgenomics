"""Synthetic 2-population admixed cohort with pedigree-induced relatedness.

Two source populations with Balding-Nichols allele frequencies (Fst), plus a
set of parent-offspring / sib pairs injected so the true kinship matrix has
known off-diagonal structure. Sufficient for unit-testing KING-robust /
PC-AiR / PC-Relate; NOT a full coalescent simulation.
"""
from __future__ import annotations
import numpy as np
import torch


def make_admixed(n_per_pop: int = 100, n_related_pairs: int = 20, m: int = 1000,
                 fst: float = 0.1, seed: int = 0) -> dict:
    """Generate synthetic 2-population admixed+related genotype fixture.

    Parameters
    ----------
    n_per_pop : int, default 100
        Samples per population (total n = 2 * n_per_pop).
    n_related_pairs : int, default 20
        Number of parent-offspring pairs to inject. Must be <= n_per_pop // 2:
        the `parent=r`, `child=n_per_pop-1-r` assignment scheme below collides
        (parent index == child index for some r) once n_related_pairs exceeds
        half the population, so this is asserted at entry.
    m : int, default 1000
        Number of variants (SNPs).
    fst : float, default 0.1
        Balding-Nichols Fst parameter for population differentiation.
    seed : int, default 0
        Random seed for deterministic reproducibility.

    Returns
    -------
    dict
        Keys: 'G' (n, m) FP64 diploid dosages [0, 2]
              'pop_label' (n,) ancestry labels {0, 1}
              'related_pairs' list of (parent_idx, child_idx, degree) tuples
              'true_kinship' (n, n) FP64 true kinship matrix
              'p0' (m,) allele frequencies in population 0
              'p1' (m,) allele frequencies in population 1

    Notes
    -----
    Population allele frequencies follow the Balding & Nichols (1995)
    *Genetics* 121:635 Fst-parameterized Beta model:
    freq_pop ~ Beta(p_anc * (1-Fst)/Fst, (1-p_anc) * (1-Fst)/Fst). The
    diagonal (self) kinship of 0.5 and off-diagonal parent-offspring kinship
    of 0.25 are the standard identity-by-descent coefficients of Lynch &
    Walsh (1998), *Genetics and Analysis of Quantitative Traits*, Sinauer
    Associates, Ch. 7.

    Determinism: `torch.distributions.Beta.sample()` has no `generator=`
    argument, so the Beta allele-frequency draws are made via a seeded numpy
    `default_rng(seed)` instead of a global `torch.manual_seed()` call. All
    torch draws (rand / binomial) use a scoped `torch.Generator().manual_seed
    (seed)`. This function does not mutate any global RNG state (torch or
    numpy) — safe to call repeatedly inside a larger test session without
    disturbing unrelated seeded code.
    """
    assert n_related_pairs <= n_per_pop // 2, (
        f"n_related_pairs ({n_related_pairs}) must be <= n_per_pop // 2 "
        f"({n_per_pop // 2}) to avoid parent/child index collision under the "
        f"parent=r, child=n_per_pop-1-r assignment scheme"
    )
    g = torch.Generator().manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    dtype = torch.float64
    n = 2 * n_per_pop
    # ancestral + population-specific frequencies (Balding-Nichols)
    p_anc = 0.1 + 0.8 * torch.rand(m, generator=g, dtype=dtype)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    # Beta sampling via a seeded numpy generator: torch.distributions.Beta has
    # no `generator=` kwarg, so we avoid the global-RNG-mutating
    # torch.manual_seed(seed) and instead draw from numpy (mirrors the
    # sibling Unit B fixture's approach for its Dirichlet draws).
    p0 = torch.from_numpy(np_rng.beta(a.numpy(), b.numpy())).to(dtype)
    p1 = torch.from_numpy(np_rng.beta(a.numpy(), b.numpy())).to(dtype)
    pop_label = torch.cat([torch.zeros(n_per_pop), torch.ones(n_per_pop)]).to(dtype)
    freqs = torch.where(pop_label.unsqueeze(1) == 0, p0.unsqueeze(0), p1.unsqueeze(0))  # (n,m)
    # founders: G ~ Binomial(2, freq) — use float32 for binomial, convert result to float64
    G = torch.binomial(torch.full((n, m), 2.0, dtype=torch.float32), freqs.to(torch.float32), generator=g)
    G = G.to(dtype)
    kinship = torch.zeros(n, n, dtype=dtype)
    kinship.fill_diagonal_(0.5)
    # inject parent-offspring pairs within a population: offspring inherits
    # exactly one Mendelian allele from the parent, the other from the
    # population allele frequency
    related_pairs = []
    for r in range(min(n_related_pairs, n_per_pop - 1)):
        parent = r
        child = n_per_pop - 1 - r  # distinct index, same population
        if child <= parent:
            break
        # Mendelian transmission: a heterozygous parent (dosage 1) transmits
        # either allele with probability 0.5; a homozygous-alt parent
        # (dosage 2) ALWAYS transmits the alt allele (transmit=1); a
        # homozygous-ref parent (dosage 0) ALWAYS transmits the ref allele
        # (transmit=0). The previous torch.binomial(clamp(G,max=1), 0.5)
        # implementation incorrectly coin-flipped even hom-alt parents,
        # understating the true parent-child genotype correlation below the
        # labeled true_kinship=0.25 (Fix per code review).
        ones = torch.ones(m, dtype=dtype)
        transmit = torch.where(
            G[parent] == 1.0,
            torch.binomial(ones, 0.5 * ones, generator=g),   # het parent: 50/50
            (G[parent] == 2.0).to(dtype),                    # hom-alt: always 1; hom-ref: always 0
        )
        # The child's other (non-transmitted) allele is drawn from the
        # population-0 allele frequency p0. This is valid because related
        # pairs are constructed exclusively among indices [0, n_per_pop),
        # i.e. all related pairs live within population 0 by construction
        # (see pop_label / freqs above) — so p0 is always the correct source
        # frequency for the untransmitted allele.
        other = torch.binomial(ones, p0, generator=g)
        # transmit, other in {0,1}; their sum is already in [0,2] (no clamp
        # needed — kept implicit rather than a redundant torch.clamp call).
        G[child] = transmit + other
        kinship[parent, child] = kinship[child, parent] = 0.25
        related_pairs.append((parent, child, "po"))
    return {"G": G, "pop_label": pop_label, "related_pairs": related_pairs,
            "true_kinship": kinship, "p0": p0, "p1": p1}
