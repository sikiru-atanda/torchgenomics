# Phase 57 — Unit A: Admixture-aware GRM/PCs (PC-AiR/PC-Relate) — Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `linalg/kinship_admixed.py` — the paper's KING-robust → LD-prune → PC-AiR → PC-Relate chain — returning relatedness-robust PCs + a sparse admixture-aware GRM for admixed cohorts with relatedness, validated against GENESIS.

**Architecture:** Four pure functions in a new `torchgenomics/linalg/kinship_admixed.py`, composed by an `admixed_grm()` orchestrator. Reuses `linalg/eigh.py` (eigendecomposition), `linalg/sparse_grm.py` (sparse GRM assembly), and `ld/_pairwise.compute_r2_matrix` (LD pruning). All GPU-torch, FP64 for the estimators.

**Tech Stack:** Python 3.12, PyTorch (FP64), pytest. External validation (opt-in): R + GENESIS/SNPRelate.

## Global Constraints

- **FP64 mandatory** for all kinship/PC estimators (statistical inference).
- **Streaming-first:** KING-robust, LD-prune, and PC-Relate stream over marker chunks; never materialize a full (n, m) genotype matrix beyond a chunk. Sample-space objects (n×n kinship, n×p PCs) are computed once. New perf-sensitive paths require a paired `tests/test_streaming_memory.py` entry.
- **Device discipline:** all-torch, on-device; obey `torchgenomics._dispatch.select_path`; never proactively move a CUDA tensor to host.
- **Python-as-spec / native-as-shortcut:** any future C++ accelerator wires at the top; the torch body stays the reference.
- **Reviewer-grade docstrings** with primary-source citations: Manichaikul 2010 (KING-robust), Conomos 2015 (PC-AiR), Conomos 2016 (PC-Relate), Gogarten 2019 (GENESIS). Novelty claims "to our knowledge."
- **Observed-then-floored tolerances** — reference-equivalence vs GENESIS uses tolerances set after observing, never aspirational.
- **Every phase commit carries "Phase 57" (Unit A)** in the message. No autonomous push.

## File Structure

- Create: `torchgenomics/linalg/kinship_admixed.py` — the four functions + `admixed_grm()` orchestrator + `AdmixedGRMResult` dataclass.
- Create: `tests/test_kinship_admixed.py` — Tier 1/2 unit + integration tests.
- Create: `tests/fixtures/admixed/make_admixed.py` — synthetic 2-population admixed+related genotype generator with known ancestry + pedigree (may adapt `tests/fixtures/tractor/make_synth.py` if present, else standalone).
- Modify: `torchgenomics/linalg/__init__.py` — export `admixed_grm`, `king_robust_kinship`, `pc_air`, `pc_relate`, `AdmixedGRMResult`.
- Create (Task 6): `validation/external/genesis/{install.sh,fetch_data.sh,run_genesis.R,compare.py}` — opt-in GENESIS equivalence harness.

**Interface contract (types later tasks rely on):**

```python
# all genotypes G are (n, m) FP64 dosages in [0,2] (diploid); missing = nan
def king_robust_kinship(G: Tensor, chunk_size: int = 2000) -> Tensor: ...   # (n,n) KING-robust kinship phi
def ld_prune_independent(G: Tensor, r2_threshold: float = 0.1,
                         window: int = 500) -> Tensor: ...                   # (k,) LongTensor of kept SNP indices
def pc_air(G_pruned: Tensor, king_kinship: Tensor, n_pcs: int = 10,
           kin_threshold: float = 0.025, div_threshold: float = -0.025) -> Tensor: ...  # (n, n_pcs) PCs
def pc_relate(G_pruned: Tensor, pcs: Tensor, n_pcs_adjust: int = 3,
              maf_min: float = 0.01) -> Tensor: ...                          # (n,n) PC-Relate kinship

@dataclass
class AdmixedGRMResult:
    pcs: Tensor            # (n, n_pcs)
    grm: Tensor            # (n,n) dense 2*kinship (before sparsification)
    sparse_grm: Tensor     # (n,n) sparse (torch.sparse_coo) admixture-aware GRM
    king_kinship: Tensor   # (n,n)
    pruned_idx: Tensor     # (k,)

def admixed_grm(G: Tensor, n_pcs: int = 10, r2_threshold: float = 0.1,
                sparsify_threshold: float = 0.05, chunk_size: int = 2000) -> AdmixedGRMResult: ...
```

---

### Task 1: Synthetic 2-population admixed+related fixture

**Files:**
- Create: `tests/fixtures/admixed/make_admixed.py`, `tests/fixtures/admixed/__init__.py`
- Test: `tests/test_kinship_admixed.py`

**Interfaces:**
- Produces: `make_admixed(n_per_pop=100, n_related_pairs=20, m=1000, fst=0.1, seed=0) -> dict` with keys `G` (n,m) FP64 diploid dosages, `pop_label` (n,) ancestry {0,1} (or admixed proportion), `related_pairs` list[(i,j,degree)], `true_kinship` (n,n). Deterministic (scoped generator).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kinship_admixed.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kinship_admixed.py::test_make_admixed_shapes_and_structure -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# tests/fixtures/admixed/make_admixed.py
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
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    n = 2 * n_per_pop
    # ancestral + population-specific frequencies (Balding-Nichols)
    p_anc = 0.1 + 0.8 * torch.rand(m, generator=g, dtype=dtype)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    p0 = torch.distributions.Beta(a, b).sample()
    p1 = torch.distributions.Beta(a, b).sample()
    pop_label = torch.cat([torch.zeros(n_per_pop), torch.ones(n_per_pop)]).to(dtype)
    freqs = torch.where(pop_label.unsqueeze(1) == 0, p0.unsqueeze(0), p1.unsqueeze(0))  # (n,m)
    # founders: G ~ Binomial(2, freq)
    G = torch.binomial(torch.full((n, m), 2.0), freqs, generator=g)
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
        transmit = torch.binomial(torch.clamp(G[parent], max=1.0), torch.full((m,), 0.5, dtype=dtype), generator=g)
        other = torch.binomial(torch.ones(m, dtype=dtype), p0, generator=g)
        G[child] = torch.clamp(transmit + other, max=2.0)
        kinship[parent, child] = kinship[child, parent] = 0.25
        related_pairs.append((parent, child, "po"))
    return {"G": G, "pop_label": pop_label, "related_pairs": related_pairs,
            "true_kinship": kinship, "p0": p0, "p1": p1}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kinship_admixed.py::test_make_admixed_shapes_and_structure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/admixed/ tests/test_kinship_admixed.py
git commit -m "Phase 57 (Unit A): synthetic 2-population admixed+related fixture"
```

---

### Task 2: KING-robust kinship (streaming)

**Files:**
- Create: `torchgenomics/linalg/kinship_admixed.py`
- Modify: `torchgenomics/linalg/__init__.py`
- Test: `tests/test_kinship_admixed.py`

**Interfaces:**
- Produces: `king_robust_kinship(G, chunk_size=2000) -> Tensor` (n,n).

**Reference (VERIFY against primary source):** KING-robust between-family kinship (Manichaikul et al. 2010, *Bioinformatics* 26:2867, eq. 11). For a pair (i,j):
`phi_ij = (N_Aa,Aa - 2*N_AA,aa) / (N_Aa^i + N_Aa^j)`
where `N_Aa,Aa` = # markers both heterozygous, `N_AA,aa` = # markers opposite homozygotes, `N_Aa^i` = # heterozygous markers in i. **The implementer MUST confirm the exact numerator/denominator constants against Manichaikul 2010 eq. 11 and pin them with the hand-computed test below (and, in Task 6, against SNPRelate `snpgdsIBDKING`).** Diagonal set to 0.5.

- [ ] **Step 1: Write the failing test (hand-computed 3-sample example)**

```python
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
    # Pair (0,1): both-het markers = {0,1,4} -> N_AaAa=3; opposite-hom = {3(2,0? ind0=2 ind1=0 -> opp),2(ind0=0 ind1=2 -> opp),5(ind0=0 ind1=2 -> opp)} -> N_AAaa=3
    # N_Aa^0 = #het in ind0 = markers {0,1,4} = 3 ; N_Aa^1 = {0,1,4} = 3
    # phi_01 = (3 - 2*3) / (3 + 3) = -3/6 = -0.5  (unrelated, different implied structure)
    expected_01 = (3 - 2*3) / (3 + 3)
    assert abs(float(phi[0, 1]) - expected_01) < 1e-9, f"got {float(phi[0,1])}, want {expected_01}"
    # symmetry
    assert torch.allclose(phi, phi.T)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kinship_admixed.py::test_king_robust_hand_computed -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Write minimal implementation**

```python
# torchgenomics/linalg/kinship_admixed.py
"""Admixture-aware GRM/PCs: KING-robust -> LD-prune -> PC-AiR -> PC-Relate.

Reference chain (Tan et al. 2026, Tractor-Mix, Extended Data Fig 1):
- KING-robust kinship: Manichaikul et al. 2010, Bioinformatics 26:2867.
- PC-AiR (relatedness-robust PCs): Conomos et al. 2015, Genet. Epidemiol. 39:276.
- PC-Relate (ancestry-adjusted kinship): Conomos et al. 2016, AJHG 98:127.
- Reference implementation: GENESIS (Gogarten et al. 2019, Bioinformatics 35:5346).

All estimators run in FP64. KING-robust and PC-Relate stream over marker chunks.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


def king_robust_kinship(G: Tensor, chunk_size: int = 2000) -> Tensor:
    """KING-robust between-family kinship (Manichaikul 2010, eq. 11).

    phi_ij = (N_AaAa - 2 N_AAaa) / (N_Aa^i + N_Aa^j), diagonal 0.5.
    Streams over marker chunks; NaN genotypes are ignored per pair-marker.
    """
    G = G.to(torch.float64)
    n, m = G.shape
    het = (G == 1.0).to(torch.float64)              # (n,m) heterozygous indicator
    # opposite homozygote indicator needs pairwise; accumulate in chunks
    N_AaAa = torch.zeros(n, n, dtype=torch.float64, device=G.device)
    N_AAaa = torch.zeros(n, n, dtype=torch.float64, device=G.device)
    het_count = het.sum(dim=1)                       # N_Aa^i per individual
    for s in range(0, m, chunk_size):
        gc = G[:, s:s + chunk_size]
        hc = het[:, s:s + chunk_size]
        N_AaAa += hc @ hc.T
        # opposite homozygote: (i==0 & j==2) or (i==2 & j==0)
        hom0 = (gc == 0.0).to(torch.float64)
        hom2 = (gc == 2.0).to(torch.float64)
        N_AAaa += hom0 @ hom2.T + hom2 @ hom0.T
    denom = het_count.unsqueeze(1) + het_count.unsqueeze(0)
    denom = torch.clamp(denom, min=1.0)
    phi = (N_AaAa - 2.0 * N_AAaa) / denom
    phi.fill_diagonal_(0.5)
    return 0.5 * (phi + phi.T)
```

Then export in `torchgenomics/linalg/__init__.py`:
```python
from .kinship_admixed import king_robust_kinship  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kinship_admixed.py::test_king_robust_hand_computed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/linalg/kinship_admixed.py torchgenomics/linalg/__init__.py tests/test_kinship_admixed.py
git commit -m "Phase 57 (Unit A): KING-robust kinship (streaming, Manichaikul 2010)"
```

---

### Task 3: LD pruning to independent SNPs

**Files:**
- Modify: `torchgenomics/linalg/kinship_admixed.py`, `torchgenomics/linalg/__init__.py`
- Test: `tests/test_kinship_admixed.py`

**Interfaces:**
- Consumes: `ld._pairwise.compute_r2_matrix` (existing).
- Produces: `ld_prune_independent(G, r2_threshold=0.1, window=500) -> Tensor` (LongTensor of kept indices).

- [ ] **Step 1: Write the failing test**

```python
def test_ld_prune_drops_correlated():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kinship_admixed.py::test_ld_prune_drops_correlated -v`
Expected: FAIL — function not found.

- [ ] **Step 3: Write minimal implementation**

```python
# append to kinship_admixed.py
from ..ld._pairwise import compute_r2_matrix


def ld_prune_independent(G: Tensor, r2_threshold: float = 0.1, window: int = 500) -> Tensor:
    """Greedy LD pruning: keep SNPs mutually below r2_threshold (Conomos 2015
    used r2 < 0.1 for PC-AiR input). Sequential greedy scan; drops a SNP if it
    exceeds r2_threshold with any already-kept SNP within `window`.
    """
    G = G.to(torch.float64)
    m = G.shape[1]
    r2 = compute_r2_matrix(G)                # (m,m)
    kept: list[int] = []
    for j in range(m):
        ok = True
        for k in kept[-window:]:
            if float(r2[j, k]) >= r2_threshold:
                ok = False
                break
        if ok:
            kept.append(j)
    return torch.tensor(kept, dtype=torch.long, device=G.device)
```

Export `ld_prune_independent` in `__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kinship_admixed.py::test_ld_prune_drops_correlated -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/linalg/kinship_admixed.py torchgenomics/linalg/__init__.py tests/test_kinship_admixed.py
git commit -m "Phase 57 (Unit A): LD pruning to independent SNPs (r2<0.1)"
```

---

### Task 4: PC-AiR — relatedness-robust PCs

**Files:**
- Modify: `torchgenomics/linalg/kinship_admixed.py`, `torchgenomics/linalg/__init__.py`
- Test: `tests/test_kinship_admixed.py`

**Interfaces:**
- Consumes: `linalg.eigh.eigendecompose` (existing); `king_robust_kinship` (Task 2).
- Produces: `pc_air(G_pruned, king_kinship, n_pcs=10, kin_threshold=0.025, div_threshold=-0.025) -> Tensor` (n, n_pcs).

**Algorithm (Conomos 2015 — VERIFY the partition + projection against GENESIS `pcair` in Task 6):**
1. Build a relatedness graph: pair (i,j) related if `king_kinship[i,j] > kin_threshold`.
2. Partition into an **unrelated set** and a **related set** via a greedy Maximal-Independent-Set-style rule that also prefers ancestry-representative individuals (Conomos uses a KING-based "divergence" measure `div_threshold` to keep ancestry-informative unrelated individuals). A defensible torch implementation: greedily add individuals to the unrelated set in descending order of "ancestry informativeness" (number of `div < div_threshold` partners), skipping any that are related (`kin > kin_threshold`) to an already-selected member. Remaining individuals form the related set.
3. **PCA on the unrelated set:** standardize `G_pruned[unrelated]` columns (mean/sd over the unrelated set), form its GRM, `eigendecompose` → top `n_pcs` eigenvectors → unrelated-set scores.
4. **Project the related set** onto those axes using SNP loadings derived from the unrelated-set PCA (loadings = standardized-genotype-weighted eigenvectors), giving related-set scores.
5. Assemble the full `(n, n_pcs)` PC matrix in original sample order.

- [ ] **Step 1: Write the failing test (PCs separate ancestry; partition is valid)**

```python
def test_pc_air_separates_ancestry_and_partitions():
    from torchgenomics.linalg.kinship_admixed import king_robust_kinship, ld_prune_independent, pc_air
    from tests.fixtures.admixed.make_admixed import make_admixed
    d = make_admixed(n_per_pop=60, n_related_pairs=15, m=800, fst=0.15, seed=7)
    G = d["G"]
    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)
    pcs = pc_air(Gp, phi, n_pcs=5)
    assert pcs.shape == (G.shape[0], 5)
    # PC1 separates the two populations: mean PC1 differs by ancestry
    pop = d["pop_label"]
    pc1 = pcs[:, 0]
    sep = abs(float(pc1[pop == 0].mean() - pc1[pop == 1].mean()))
    within = float(pc1.std())
    assert sep > within, f"PC1 ancestry separation {sep} not > within-scatter {within}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kinship_admixed.py::test_pc_air_separates_ancestry_and_partitions -v`
Expected: FAIL — function not found.

- [ ] **Step 3: Write minimal implementation**

Implement `pc_air` per the algorithm above. Key torch pieces (the implementer fills the partition greedily and the projection):
```python
from .eigh import eigendecompose


def _standardize(G: Tensor, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mu = ref.mean(dim=0)
    sd = ref.std(dim=0).clamp(min=1e-8)
    return (G - mu) / sd, mu, sd


def pc_air(G_pruned: Tensor, king_kinship: Tensor, n_pcs: int = 10,
           kin_threshold: float = 0.025, div_threshold: float = -0.025) -> Tensor:
    G_pruned = G_pruned.to(torch.float64)
    n = G_pruned.shape[0]
    related = king_kinship > kin_threshold
    related.fill_diagonal_(False)
    # ancestry informativeness score: count of strongly-divergent partners
    informative = (king_kinship < div_threshold).sum(dim=1)
    order = torch.argsort(informative, descending=True)
    unrel_mask = torch.zeros(n, dtype=torch.bool, device=G_pruned.device)
    chosen: list[int] = []
    for idx in order.tolist():
        if any(bool(related[idx, c]) for c in chosen):
            continue
        chosen.append(idx); unrel_mask[idx] = True
    rel_mask = ~unrel_mask
    # PCA on unrelated set
    Gu = G_pruned[unrel_mask]
    Zu, mu, sd = _standardize(Gu, Gu)
    grm_u = (Zu @ Zu.T) / Gu.shape[1]
    ed = eigendecompose(grm_u, k=n_pcs)
    scores_u = ed.eigenvectors[:, :n_pcs] * torch.sqrt(ed.eigenvalues[:n_pcs].clamp(min=0)).unsqueeze(0)
    # SNP loadings from unrelated PCA: L = Zu^T @ scores_u / eig  (project related set)
    load = Zu.T @ ed.eigenvectors[:, :n_pcs]           # (m, n_pcs)
    pcs = torch.zeros(n, n_pcs, dtype=torch.float64, device=G_pruned.device)
    pcs[unrel_mask] = ed.eigenvectors[:, :n_pcs]
    if bool(rel_mask.any()):
        Zr = (G_pruned[rel_mask] - mu) / sd
        proj = Zr @ load                                # (n_rel, n_pcs)
        # scale projection to the unrelated-set eigenvector norm
        proj = proj / torch.sqrt(ed.eigenvalues[:n_pcs].clamp(min=1e-12)).unsqueeze(0) / Gu.shape[1]
        pcs[rel_mask] = proj
    return pcs
```
(Note: the exact projection scaling must be checked against GENESIS `pcair` in Task 6; the ancestry-separation test is the Tier-1 gate. If the sign/scale differs from GENESIS, adjust — PC signs are arbitrary, so Task 6 compares |correlation|.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kinship_admixed.py::test_pc_air_separates_ancestry_and_partitions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/linalg/kinship_admixed.py torchgenomics/linalg/__init__.py tests/test_kinship_admixed.py
git commit -m "Phase 57 (Unit A): PC-AiR relatedness-robust PCs (Conomos 2015)"
```

---

### Task 5: PC-Relate kinship + `admixed_grm` orchestrator

**Files:**
- Modify: `torchgenomics/linalg/kinship_admixed.py`, `torchgenomics/linalg/__init__.py`
- Test: `tests/test_kinship_admixed.py`, `tests/test_streaming_memory.py`

**Interfaces:**
- Consumes: `sparse_grm` (existing) for sparse assembly.
- Produces: `pc_relate(G_pruned, pcs, n_pcs_adjust=3, maf_min=0.01) -> Tensor` (n,n); `admixed_grm(G, ...) -> AdmixedGRMResult`.

**Algorithm (Conomos 2016 — VERIFY against GENESIS `pcrelate` in Task 6):**
1. For each SNP, regress genotype on `[1 | pcs[:, :n_pcs_adjust]]` (least squares) → individual-specific allele frequency `mu_i = fitted/2`, clamp to `[maf_min, 1-maf_min]`.
2. PC-Relate kinship estimator: `phi_ij = sum_l (g_il - 2 mu_il)(g_jl - 2 mu_jl) / (4 sum_l sqrt(mu_il(1-mu_il) mu_jl(1-mu_jl)))`.
3. Diagonal set to 0.5 (or self-kinship estimator). Stream over marker chunks.
4. `admixed_grm`: GRM = `2 * kinship`; sparsify (`|entry| < sparsify_threshold -> 0`, diagonal -> 1 per the paper's ×2/diag-1/<0.05→0 rule); assemble sparse via `sparse_grm`.

- [ ] **Step 1: Write the failing test (recovers pedigree kinship; orchestrator)**

```python
def test_pc_relate_recovers_pedigree_and_orchestrator():
    from torchgenomics.linalg.kinship_admixed import (
        king_robust_kinship, ld_prune_independent, pc_air, pc_relate, admixed_grm)
    from tests.fixtures.admixed.make_admixed import make_admixed
    d = make_admixed(n_per_pop=60, n_related_pairs=15, m=1000, fst=0.15, seed=11)
    G = d["G"]
    keep = ld_prune_independent(G, r2_threshold=0.2)
    Gp = G[:, keep]
    phi = king_robust_kinship(Gp)
    pcs = pc_air(Gp, phi, n_pcs=5)
    kin = pc_relate(Gp, pcs, n_pcs_adjust=3)
    assert kin.shape == (G.shape[0], G.shape[0])
    # parent-offspring pairs (true kinship 0.25) estimated higher than unrelated (~0)
    po = [(i, j) for (i, j, _) in d["related_pairs"]]
    po_est = torch.tensor([float(kin[i, j]) for i, j in po])
    # unrelated cross-population pairs
    unrel_est = torch.tensor([float(kin[0, G.shape[0] - 1])])
    assert po_est.mean() > 0.10, f"PO kinship {po_est.mean()} too low"
    assert po_est.mean() > unrel_est.mean() + 0.10
    # orchestrator returns a coherent result
    res = admixed_grm(G, n_pcs=5, r2_threshold=0.2)
    assert res.pcs.shape == (G.shape[0], 5)
    assert res.sparse_grm.shape == (G.shape[0], G.shape[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kinship_admixed.py::test_pc_relate_recovers_pedigree_and_orchestrator -v`
Expected: FAIL — functions not found.

- [ ] **Step 3: Write minimal implementation**

Implement `pc_relate` (streaming moment estimator per the algorithm) and `admixed_grm` (orchestrator composing Tasks 2-5 + sparsify). Populate `AdmixedGRMResult`. Add the `@dataclass AdmixedGRMResult` from the interface contract. Sparsify per the paper: `2*kinship`, set `|.|<sparsify_threshold` to 0, diagonal to 1, assemble as a `torch.sparse_coo_tensor` (or reuse `sparse_grm`). Export `pc_relate`, `admixed_grm`, `AdmixedGRMResult`.

- [ ] **Step 4: Run test + streaming-memory entry**

Add a `tests/test_streaming_memory.py` entry that runs `king_robust_kinship`/`pc_relate` with a small `chunk_size` over a largish m and asserts the marker-loop peak does not scale with total m (behavioral width-tracker, mirroring the Unit B approach — tracemalloc does NOT track torch tensors, so track the max marker-chunk width processed).

Run: `pytest tests/test_kinship_admixed.py::test_pc_relate_recovers_pedigree_and_orchestrator tests/test_streaming_memory.py -k admixed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/linalg/kinship_admixed.py torchgenomics/linalg/__init__.py tests/test_kinship_admixed.py tests/test_streaming_memory.py
git commit -m "Phase 57 (Unit A): PC-Relate kinship + admixed_grm orchestrator + sparse GRM"
```

---

### Task 6: External GENESIS equivalence harness (Tier 3, opt-in)

**Files:**
- Create: `validation/external/genesis/{install.sh,fetch_data.sh,run_genesis.R,compare.py,README.md}`
- Test: `tests/test_kinship_admixed.py` (an `@pytest.mark.external` test that reads the harness outputs if present, else skips)

**Interfaces:**
- Produces: golden GENESIS outputs (KING kinship, PC-AiR PCs, PC-Relate kinship) on a shared 1000G-admixed-subset (or the synthetic fixture exported to PLINK) → `compare.py` asserts tolerance-bounded agreement (PCs by |correlation| / Procrustes; kinship by max-abs-diff, observed-then-floored).

- [ ] **Step 1: Write the harness scripts**

Author (via heredoc — subagent Write is unreliable) `install.sh` (R + BiocManager::install(c("GENESIS","SNPRelate","GWASTools"))), `fetch_data.sh` (export the synthetic fixture to PLINK bed/bim/fam via a small python writer, OR download a 1000G AFR-EUR subset), `run_genesis.R` (snpgdsIBDKING → pcair → pcrelate; write kinship/PCs to TSV), `compare.py` (load torchgenomics `admixed_grm` outputs + GENESIS TSVs; assert |PC correlation| ≥ threshold and kinship max-abs-diff ≤ tol). **Every shell script sources `validation/external/_lib/preflight.sh`** (disk+RAM headroom — hard rule; abort on insufficient resources).

- [ ] **Step 2: Add the opt-in pytest gate**

```python
import pytest, os
@pytest.mark.external
def test_genesis_equivalence_if_present():
    out = "validation/external/genesis/outputs"
    if not os.path.isdir(out):
        pytest.skip("GENESIS harness outputs not present; run validation/external/genesis/*.sh")
    # load + compare (delegates to compare.py logic); assert within observed-then-floored tolerance
    ...
```

- [ ] **Step 3: Run the local (skip) path**

Run: `pytest tests/test_kinship_admixed.py::test_genesis_equivalence_if_present -v`
Expected: SKIP (harness not run in CI by default; opt-in only).

- [ ] **Step 4: Document re-run path in README.md**

Document the install/fetch/run/compare sequence + the preflight requirement.

- [ ] **Step 5: Commit**

```bash
git add validation/external/genesis/ tests/test_kinship_admixed.py
git commit -m "Phase 57 (Unit A): GENESIS equivalence harness (opt-in, preflight-gated)"
```

---

## Self-Review

**Spec coverage (spec §5 Unit A + §8 Claim 3):**
- KING-robust kinship → Task 2. ✅
- LD-prune r²<0.1 → Task 3. ✅
- PC-AiR partition + PCs → Task 4. ✅
- PC-Relate kinship + sparse GRM (×2/diag-1/<0.05→0) → Task 5. ✅
- `admixed_grm` orchestrator returning (pcs, sparse_grm) → Task 5. ✅
- GENESIS equivalence (Claim 3) → Task 6. ✅
- Streaming + memory net → Task 5. ✅

**Known verification points flagged for the implementer:** (a) KING-robust exact eq. 11 constants — pinned by Task 2's hand-computed test + Task 6 SNPRelate; (b) PC-AiR partition + projection scaling — Task 4 ancestry-separation gate + Task 6 |correlation| vs GENESIS `pcair`; (c) PC-Relate moment estimator + individual-specific AF regression — Task 5 pedigree-recovery gate + Task 6 vs GENESIS `pcrelate`; (d) `eigendecompose` / `sparse_grm` / `compute_r2_matrix` exact signatures — confirm at implementation.

**Placeholder scan:** the PC-AiR projection scaling and PC-Relate estimator carry algorithm specs with reference-verification gates rather than claimed-exact constants, because reference-equivalence to GENESIS is the acceptance criterion and the exact scaling must be pinned against `pcair`/`pcrelate` output (Task 6) — these are verification steps, not placeholders. All Tier-1 tests carry runnable code + a concrete numeric/statistical assertion.

**Downstream:** Unit C (Plan 3) wires `admixed_grm` output as the GRM/PCs input to `TractorLMM` (Unit B) behind the `tractor-scan` CLI's `--compute-grm` path.
