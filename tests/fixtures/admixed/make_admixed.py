"""Synthetic 2-population admixed cohort with pedigree-induced relatedness.

Two source populations with Balding-Nichols allele frequencies (Fst), plus a
set of parent-offspring / sib pairs injected so the true kinship matrix has
known off-diagonal structure. Sufficient for unit-testing KING-robust /
PC-AiR / PC-Relate; NOT a full coalescent simulation.
"""
from __future__ import annotations
import torch


def make_admixed(n_per_pop: int = 100, n_related_pairs: int = 20, m: int = 1000,
                 fst: float = 0.1, seed: int = 0) -> dict:
    """Generate synthetic 2-population admixed+related genotype fixture.

    Parameters
    ----------
    n_per_pop : int, default 100
        Samples per population (total n = 2 * n_per_pop).
    n_related_pairs : int, default 20
        Number of parent-offspring pairs to inject.
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
    Deterministic given seed via scoped generator for binomial operations and
    global seed for Beta distribution sampling. Genotypes are FP64 for statistical
    inference. Self-kinship set to 0.5 (diploid), parent-offspring kinship set to 0.25.
    """
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    n = 2 * n_per_pop
    # ancestral + population-specific frequencies (Balding-Nichols)
    p_anc = 0.1 + 0.8 * torch.rand(m, generator=g, dtype=dtype)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    # Beta sampling requires global seed for determinism (torch.distributions doesn't support generator param)
    torch.manual_seed(seed)
    p0 = torch.distributions.Beta(a, b).sample()
    p1 = torch.distributions.Beta(a, b).sample()
    pop_label = torch.cat([torch.zeros(n_per_pop), torch.ones(n_per_pop)]).to(dtype)
    freqs = torch.where(pop_label.unsqueeze(1) == 0, p0.unsqueeze(0), p1.unsqueeze(0))  # (n,m)
    # founders: G ~ Binomial(2, freq) — use float32 for binomial, convert result to float64
    G = torch.binomial(torch.full((n, m), 2.0, dtype=torch.float32), freqs.to(torch.float32), generator=g)
    G = G.to(dtype)
    kinship = torch.zeros(n, n, dtype=dtype)
    kinship.fill_diagonal_(0.5)
    # inject parent-offspring pairs within a population: offspring = avg-ish of parent alleles
    related_pairs = []
    for r in range(min(n_related_pairs, n_per_pop - 1)):
        parent = r
        child = n_per_pop - 1 - r  # distinct index, same population
        if child <= parent:
            break
        # child inherits one allele from parent (transmit ~half), other from pop freq
        transmit = torch.binomial(torch.clamp(G[parent], max=1.0).to(torch.float32), torch.full((m,), 0.5, dtype=torch.float32), generator=g)
        other = torch.binomial(torch.ones(m, dtype=torch.float32), p0.to(torch.float32), generator=g)
        G[child] = torch.clamp(transmit + other, max=2.0).to(dtype)
        kinship[parent, child] = kinship[child, parent] = 0.25
        related_pairs.append((parent, child, "po"))
    return {"G": G, "pop_label": pop_label, "related_pairs": related_pairs,
            "true_kinship": kinship, "p0": p0, "p1": p1}
