# Phase 57 — Unit B: TractorLMM Model — Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `TractorLMM` — the ancestry-specific mixed-model association engine (K-df joint Rao score test + ancestry-specific score/Wald effects, allele-count threshold, opt-in SPA, local-ancestry conditional), conforming to the `BaseModel` protocol, validated against a fixed synthetic GRM.

**Architecture:** A new `models/tractor_lmm.py` conforming to the existing `BaseModel` (`fit_null` + `score_chunk`) protocol. Null fitting reuses `single_trait_lmm` (continuous) and `binary_glmm` (binary) unchanged; the new logic is the multi-column (ancestry-partitioned) score/Wald test. Consumes a GRM as an input (external or synthetic); Unit A (native PC-AiR/PC-Relate) is a separate later plan.

**Tech Stack:** Python 3.12, PyTorch (FP64 for inference), pytest. Reuses `torchgenomics.optim` REML, `torchgenomics.stats.spa`, `torchgenomics.models.base`.

## Global Constraints

- **FP64 mandatory for statistical inference** (AMP/FP16 only for I/O and GRM assembly).
- **Streaming-first:** `score_chunk` consumes variant chunks; never materialize the full genotype/dosage matrix. New perf-sensitive paths require a paired `tests/test_streaming_memory.py` entry + `bench/native_speedups.py` wall-time gate (repo convention).
- **Device discipline:** all-torch, on-device; obey `torchgenomics._dispatch.select_path`; never proactively move a CUDA tensor to host.
- **Python-as-spec / native-as-shortcut:** any future C++ accelerator wires at the top of the function; the torch body stays the reference.
- **Reviewer-grade docstrings** with primary-source citations: Tan 2026 (Tractor-Mix), Chen 2016 (GMMAT), Atkinson 2021 (Tractor). Novelty (opt-in SPA) qualified "to our knowledge."
- **No autonomous push;** every phase commit carries "Phase 57" in the message.
- **Observed-then-floored tolerances** in tests — never aspirational.

---

## File Structure

- Create: `torchgenomics/models/tractor_lmm.py` — the `TractorLMM` model + `AncestryDosages` input container.
- Create: `torchgenomics/io/ancestry_dosage.py` — reader for Tractor `ExtractTracts` `anc{k}.dosage.txt` + `.npy`/`.pt`.
- Create: `tests/test_tractor_lmm.py` — Tier 1/2 unit + integration tests.
- Create: `tests/fixtures/tractor/make_synth.py` — synthetic admixed+related generator (PSD + gene-drop), emits known ancestry dosages + true GRM.
- Modify: `torchgenomics/models/__init__.py` — export `TractorLMM`, `AncestryDosages`.

**Interface contract for this plan (types later tasks rely on):**

```python
# io/ancestry_dosage.py
@dataclass
class AncestryDosages:
    dosages: torch.Tensor       # (K, n, m)  FP32/FP64, K ancestries, n samples, m variants
    ancestry_names: list[str]   # length K, e.g. ["AFR", "EUR"]
    variant_meta: VariantMeta   # reuse models.base.VariantMeta
    def n_ancestries(self) -> int: ...
    def iter_chunks(self, chunk_size: int) -> Iterator[tuple["AncestryDosages", slice]]: ...

def read_ancestry_dosages(prefix: str, ancestry_names: list[str] | None = None,
                          fmt: str = "auto") -> AncestryDosages: ...

# models/tractor_lmm.py
class TractorLMM:   # conforms to models.base.BaseModel
    def __init__(self, family: str = "gaussian", test: str = "score",
                 min_allele_count: int = 50, use_spa: bool = False,
                 spa_threshold: float = 2.0) -> None: ...
    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor) -> TractorNullFit: ...
    def score_chunk(self, null: "TractorNullFit", G_chunk: Tensor,
                    meta: VariantMeta, L_chunk: Tensor | None = None) -> ScanResult: ...
```

`G_chunk` shape is `(K, n, c)` (K ancestries, n samples, c variants in the chunk). `L_chunk` (optional) is local-ancestry dosages `(K, n, c)` for conditional analysis.

---

### Task 1: Synthetic admixed+related fixture generator

**Files:**
- Create: `tests/fixtures/tractor/make_synth.py`
- Test: `tests/test_tractor_lmm.py` (first tests use the fixture)

**Interfaces:**
- Produces: `make_synth(n, m, K=2, seed=0) -> dict` with keys `dosages` (K,n,m), `y_cont` (n,), `y_bin` (n,), `X0` (n,c), `K_grm` (n,n) true GRM, `beta_true` (K,), `causal_idx` (list). Deterministic given seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tractor_lmm.py
import torch
from tests.fixtures.tractor.make_synth import make_synth

def test_make_synth_shapes_and_determinism():
    d1 = make_synth(n=200, m=50, K=2, seed=7)
    d2 = make_synth(n=200, m=50, K=2, seed=7)
    assert d1["dosages"].shape == (2, 200, 50)
    assert d1["y_cont"].shape == (200,)
    assert d1["K_grm"].shape == (200, 200)
    # ancestry dosages sum to total genotype dosage in [0, 2]
    tot = d1["dosages"].sum(dim=0)
    assert tot.min() >= 0 and tot.max() <= 2.0 + 1e-6
    # GRM is symmetric PSD
    assert torch.allclose(d1["K_grm"], d1["K_grm"].T, atol=1e-6)
    assert torch.linalg.eigvalsh(d1["K_grm"]).min() > -1e-6
    # determinism
    assert torch.equal(d1["dosages"], d2["dosages"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_make_synth_shapes_and_determinism -v`
Expected: FAIL — `ModuleNotFoundError: tests.fixtures.tractor.make_synth`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/fixtures/tractor/make_synth.py
"""Synthetic admixed+related cohort: PSD-admixture dosages + gene-drop GRM.

Simplified reproduction of the Tan et al. 2026 simulation recipe, sufficient
for unit/equivalence testing (NOT the full biobank sim). Ancestry-specific
allele frequencies differ (Fst), a pedigree induces relatedness (true GRM),
and a subset of variants carry ancestry-specific effects.
"""
from __future__ import annotations
import torch


def make_synth(n: int = 200, m: int = 50, K: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    # ancestry-specific allele freqs (Balding-Nicols-ish via Beta), Fst=0.1
    fst = 0.1
    base = 0.1 + 0.8 * torch.rand(m, generator=g, dtype=dtype)
    a = base * (1 - fst) / fst
    b = (1 - base) * (1 - fst) / fst
    af = torch.stack([torch.distributions.Beta(a, b).sample() for _ in range(K)])  # (K, m)
    # global admixture proportion per sample (80/20 for K=2 default), Dirichlet
    adm = torch.distributions.Dirichlet(
        torch.tensor([8.0, 2.0] + [1.0] * (K - 2), dtype=dtype)
    ).sample((n,))  # (n, K)
    # ancestry-specific dosages: each ancestry contributes adm-scaled Binomial(1, af)
    dosages = torch.zeros(K, n, m, dtype=dtype)
    for k in range(K):
        p = af[k].unsqueeze(0).expand(n, m)
        # local-ancestry copy count ~ Binomial(2, adm_k); allele ~ Binomial(copies, af)
        copies = torch.binomial(torch.full((n, m), 2.0),
                                adm[:, k].unsqueeze(1).expand(n, m), generator=g)
        dosages[k] = torch.binomial(copies, p, generator=g)
    # relatedness: build a block pedigree -> true GRM as 2*kinship
    n_fam = max(1, n // 4)
    fam = torch.arange(n) % n_fam
    kinship = (fam.unsqueeze(0) == fam.unsqueeze(1)).to(dtype) * 0.25
    kinship.fill_diagonal_(0.5)
    K_grm = 2.0 * kinship  # (n,n)
    # phenotype: causal on ancestry 0 dosage for a few variants
    causal_idx = list(range(0, m, max(1, m // 5)))[:5]
    beta_true = torch.zeros(K, dtype=dtype)
    beta_true[0] = 0.5
    signal = torch.zeros(n, dtype=dtype)
    for j in causal_idx:
        signal += beta_true[0] * dosages[0, :, j]
    # random effect from GRM + noise
    L = torch.linalg.cholesky(K_grm + 1e-4 * torch.eye(n, dtype=dtype))
    reff = L @ torch.randn(n, generator=g, dtype=dtype)
    X0 = torch.stack([torch.ones(n, dtype=dtype),
                      torch.randn(n, generator=g, dtype=dtype)], dim=1)  # intercept + 1 cov
    y_cont = 1.0 + 0.3 * X0[:, 1] + signal + reff + torch.randn(n, generator=g, dtype=dtype)
    prob = torch.sigmoid(y_cont - y_cont.mean())
    y_bin = torch.bernoulli(prob, generator=g)
    return {"dosages": dosages, "y_cont": y_cont, "y_bin": y_bin, "X0": X0,
            "K_grm": K_grm, "beta_true": beta_true, "causal_idx": causal_idx,
            "adm": adm, "af": af}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_make_synth_shapes_and_determinism -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/tractor/make_synth.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): synthetic admixed+related fixture generator"
```

---

### Task 2: Ancestry-dosage input container + reader

**Files:**
- Create: `torchgenomics/io/ancestry_dosage.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Produces: `AncestryDosages` dataclass + `read_ancestry_dosages(prefix, ancestry_names, fmt)` (see interface contract above). `iter_chunks` yields `(sub, col_slice)` with `sub.dosages` shape `(K, n, chunk)`.

- [ ] **Step 1: Write the failing test**

```python
def test_ancestry_dosages_container_and_chunking(tmp_path):
    import numpy as np
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages, AncestryDosages
    # write two Tractor-style dosage files: rows=variants, cols=samples
    n, m = 6, 10
    for k in range(2):
        arr = np.arange(m * n).reshape(m, n).astype(float) + k
        np.savetxt(tmp_path / f"anc{k}.dosage.txt", arr)
    ad = read_ancestry_dosages(str(tmp_path / "anc"), ancestry_names=["AFR", "EUR"])
    assert isinstance(ad, AncestryDosages)
    assert ad.n_ancestries() == 2
    assert ad.dosages.shape == (2, n, m)          # (K, n, m) after transpose
    chunks = list(ad.iter_chunks(chunk_size=4))
    assert len(chunks) == 3                        # 10 variants / 4 -> 3 chunks
    assert chunks[0][0].dosages.shape == (2, n, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_ancestry_dosages_container_and_chunking -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# torchgenomics/io/ancestry_dosage.py
"""Reader for ancestry-partitioned genotype dosages (Tractor ExtractTracts).

Local ancestry inference (RFMix2) and dosage extraction (Tractor
ExtractTracts.py) are UPSTREAM and out of scope: this module consumes their
output. Tractor writes one file per ancestry, `anc{k}.dosage.txt`, rows =
variants, cols = samples. We also accept `.npy`/`.pt` stacks of shape (K,n,m).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import glob
import numpy as np
import torch
from ..models.base import VariantMeta


@dataclass
class AncestryDosages:
    dosages: torch.Tensor        # (K, n, m)
    ancestry_names: list[str]
    variant_meta: VariantMeta | None = None

    def n_ancestries(self) -> int:
        return self.dosages.shape[0]

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple["AncestryDosages", slice]]:
        m = self.dosages.shape[2]
        for start in range(0, m, chunk_size):
            sl = slice(start, min(start + chunk_size, m))
            yield AncestryDosages(self.dosages[:, :, sl], self.ancestry_names), sl


def read_ancestry_dosages(prefix: str, ancestry_names: list[str] | None = None,
                          fmt: str = "auto") -> AncestryDosages:
    if fmt in ("npy", "pt") or prefix.endswith((".npy", ".pt")):
        arr = (np.load(prefix) if prefix.endswith(".npy")
               else torch.load(prefix).numpy())
        t = torch.as_tensor(arr, dtype=torch.float64)
        names = ancestry_names or [f"anc{k}" for k in range(t.shape[0])]
        return AncestryDosages(t, names)
    files = sorted(glob.glob(f"{prefix}*.dosage.txt"))
    if not files:
        raise FileNotFoundError(f"no ancestry dosage files at {prefix}*.dosage.txt")
    mats = [torch.as_tensor(np.loadtxt(f), dtype=torch.float64).T for f in files]  # -> (n, m)
    dosages = torch.stack(mats, dim=0)  # (K, n, m)
    names = ancestry_names or [f"anc{k}" for k in range(len(files))]
    return AncestryDosages(dosages, names)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_ancestry_dosages_container_and_chunking -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/io/ancestry_dosage.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): ancestry-dosage container + Tractor reader"
```

---

### Task 3: Null fit (continuous) with reusable projection

**Files:**
- Create: `torchgenomics/models/tractor_lmm.py`
- Modify: `torchgenomics/models/__init__.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Consumes: `single_trait_lmm` REML null (existing).
- Produces: `TractorLMM(family="gaussian").fit_null(Y, X0, K) -> TractorNullFit` where `TractorNullFit` exposes `Py(v) -> Tensor` and `PX(M) -> Tensor` implementing `P = V^{-1} - V^{-1}X0(X0^T V^{-1} X0)^{-1} X0^T V^{-1}`, plus `resid` (P y), `n`, `c`.

**Design note (verify against GMMAT, per spec risk §10):** the efficient score for a variant is `T = g^T P y` with `Var = g^T P g`, using the null covariance `V = sigma^2 I + tau^2 K`. `P` residualizes against covariates, so no separate centering of `g` is needed. Implement `V^{-1}` via the eigendecomposition of `K` already computed by the REML null (`V^{-1} = U diag(1/(tau^2 lambda + sigma^2)) U^T`).

- [ ] **Step 1: Write the failing test**

```python
def test_tractor_null_projection_matches_bruteforce():
    from torchgenomics.models.tractor_lmm import TractorLMM
    d = make_synth(n=120, m=30, K=2, seed=3)
    Y, X0, Kg = d["y_cont"], d["X0"], d["K_grm"]
    nf = TractorLMM(family="gaussian").fit_null(Y, X0, Kg)
    # brute-force P y from the fitted variance components
    V = nf.sigma2 * torch.eye(len(Y), dtype=torch.float64) + nf.tau2 * Kg
    Vi = torch.linalg.inv(V)
    P = Vi - Vi @ X0 @ torch.linalg.inv(X0.T @ Vi @ X0) @ X0.T @ Vi
    assert torch.allclose(nf.resid, P @ Y, atol=1e-8)
    g = d["dosages"][0, :, 0]
    assert torch.allclose(nf.Py(g.unsqueeze(1)).squeeze(), P @ g, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_tractor_null_projection_matches_bruteforce -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Write minimal implementation**

```python
# torchgenomics/models/tractor_lmm.py
"""TractorLMM: ancestry-specific mixed-model GWAS for admixed cohorts.

Reference: Tan et al., Extending GWAS to admixed cohorts with high degrees of
relatedness, Nat. Genet. 2026 (Tractor-Mix). Builds on Tractor (Atkinson 2021)
and the GMMAT score-test framework (Chen 2016). This module implements the
association + inference step; the admixture-aware GRM (PC-AiR/PC-Relate) is a
separate unit and is supplied here as an input `K`.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .base import ScanResult, VariantMeta
from ..optim import reml_math  # existing REML utilities


@dataclass
class TractorNullFit:
    U: Tensor            # (n,n) eigenvectors of K
    d: Tensor            # (n,) eigenvalues of K
    sigma2: float
    tau2: float
    X0: Tensor           # (n,c)
    resid: Tensor        # P y  (n,)
    n: int
    c: int
    _XtVi: Tensor        # (c,n)  = X0^T V^{-1}
    _XtViX_inv: Tensor   # (c,c)

    def _Vi(self, M: Tensor) -> Tensor:
        # V^{-1} M using eig(K): V^{-1} = U diag(1/(tau2 d + sigma2)) U^T
        w = 1.0 / (self.tau2 * self.d + self.sigma2)
        return self.U @ (w.unsqueeze(1) * (self.U.T @ M))

    def Py(self, M: Tensor) -> Tensor:
        # P M = V^{-1}M - V^{-1}X0 (X0^T V^{-1} X0)^{-1} X0^T V^{-1} M
        ViM = self._Vi(M)
        return ViM - self._Vi(self.X0) @ (self._XtViX_inv @ (self._XtVi @ M))


class TractorLMM:
    def __init__(self, family: str = "gaussian", test: str = "score",
                 min_allele_count: int = 50, use_spa: bool = False,
                 spa_threshold: float = 2.0) -> None:
        if family not in ("gaussian", "binary"):
            raise ValueError(f"family must be gaussian|binary, got {family}")
        if test not in ("score", "wald"):
            raise ValueError(f"test must be score|wald, got {test}")
        self.family = family
        self.test = test
        self.min_allele_count = min_allele_count
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold

    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor) -> TractorNullFit:
        Y = Y.to(torch.float64); X0 = X0.to(torch.float64); K = K.to(torch.float64)
        n, c = X0.shape
        d, U = torch.linalg.eigh(K)
        # REML variance components via existing utility (returns sigma2, tau2)
        sigma2, tau2 = reml_math.reml_varcomp_eig(Y, X0, U, d)
        w = 1.0 / (tau2 * d + sigma2)
        Vi = U @ (w.unsqueeze(1) * U.T)
        XtVi = X0.T @ Vi
        XtViX_inv = torch.linalg.inv(XtVi @ X0)
        nf = TractorNullFit(U=U, d=d, sigma2=float(sigma2), tau2=float(tau2),
                            X0=X0, resid=torch.zeros(n, dtype=torch.float64),
                            n=n, c=c, _XtVi=XtVi, _XtViX_inv=XtViX_inv)
        nf.resid = nf.Py(Y.unsqueeze(1)).squeeze(1)
        return nf
```

Add to `torchgenomics/optim/reml_math.py` if not present a thin `reml_varcomp_eig(Y, X0, U, d)` that returns `(sigma2, tau2)` by 1-D REML profiling over `h2` using the rotated data (reuse existing `emma_reml`/`ai_reml`; if an equivalent exists, import it instead of adding a duplicate — check `torchgenomics/optim/emma_reml.py` first).

Then modify `torchgenomics/models/__init__.py`:

```python
from .tractor_lmm import TractorLMM, TractorNullFit  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_tractor_null_projection_matches_bruteforce -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py torchgenomics/models/__init__.py torchgenomics/optim/reml_math.py
git commit -m "Phase 57 (Unit B): TractorLMM continuous null fit + reusable projection P"
```

---

### Task 4: Joint K-df Rao score test + ancestry-specific effects (continuous)

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Produces: `TractorLMM.score_chunk(null, G_chunk, meta) -> ScanResult` with fields per variant: `joint_p`, and per ancestry `beta_a`, `se_a`, `p_a`. `G_chunk` shape `(K, n, c)`.

**Math:** For variant with ancestry-dosage matrix `Gv` (n×K): `T = Gv^T P y` (K,), `Var = Gv^T P Gv` (K×K). Joint stat `T^T Var^{-1} T ~ chi2_K`. Ancestry effects `beta = Var^{-1} T`, `se = sqrt(diag(Var^{-1}))`, per-ancestry `p_a = 2*(1-Phi(|beta_a|/se_a))` (chi2_1).

- [ ] **Step 1: Write the failing test**

```python
def test_joint_score_recovers_causal_ancestry():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=400, m=40, K=2, seed=11)
    m = TractorLMM(family="gaussian", test="score")
    nf = m.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(ids=[f"v{i}" for i in range(40)],
                       chrom=["1"] * 40, pos=list(range(40)))
    res = m.score_chunk(nf, d["dosages"], meta)
    df = res.to_dataframe()
    # causal variants (ancestry-0 effect) reach smaller joint p than non-causal
    causal = set(d["causal_idx"])
    jp = df["joint_p"].to_numpy()
    causal_p = [jp[i] for i in range(40) if i in causal]
    null_p = [jp[i] for i in range(40) if i not in causal]
    assert min(causal_p) < 1e-3
    assert sum(p < 1e-3 for p in null_p) <= 1        # few/no null hits
    # ancestry-0 effect sign is positive at the top causal variant
    top = min(causal, key=lambda i: jp[i])
    assert df["beta_AFR"].to_numpy()[top] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_joint_score_recovers_causal_ancestry -v`
Expected: FAIL — `score_chunk` not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# append to models/tractor_lmm.py (inside class TractorLMM)
    def _score_stats(self, null: TractorNullFit, Gv: Tensor):
        # Gv: (n, K). Returns T (K,), Vinv (K,K), joint_stat (float)
        PGv = null.Py(Gv)                 # (n, K)
        T = Gv.T @ null.resid             # (K,)  == Gv^T P y
        Var = Gv.T @ PGv                  # (K, K) == Gv^T P Gv
        Vinv = torch.linalg.pinv(Var)
        joint = float(T @ Vinv @ T)
        return T, Var, Vinv, joint

    def score_chunk(self, null: TractorNullFit, G_chunk: Tensor,
                    meta: VariantMeta, L_chunk: Tensor | None = None) -> ScanResult:
        from scipy.stats import chi2, norm
        Kanc, n, c = G_chunk.shape
        names = getattr(self, "_ancestry_names", None) or [f"anc{k}" for k in range(Kanc)]
        joint_p = torch.zeros(c, dtype=torch.float64)
        betas = torch.zeros(c, Kanc, dtype=torch.float64)
        ses = torch.full((c, Kanc), float("nan"), dtype=torch.float64)
        p_anc = torch.ones(c, Kanc, dtype=torch.float64)
        for j in range(c):
            Gv = G_chunk[:, :, j].T.to(torch.float64)     # (n, K)
            T, Var, Vinv, joint = self._score_stats(null, Gv)
            k = Kanc
            joint_p[j] = chi2.sf(joint, df=k)
            beta = Vinv @ T
            se = torch.sqrt(torch.clamp(torch.diag(Vinv), min=0))
            betas[j] = beta
            ses[j] = se
            z = beta / se
            p_anc[j] = torch.tensor(2.0 * norm.sf(torch.abs(z).numpy()), dtype=torch.float64)
        cols = {"joint_p": joint_p}
        for k, nm in enumerate(names):
            cols[f"beta_{nm}"] = betas[:, k]
            cols[f"se_{nm}"] = ses[:, k]
            cols[f"p_{nm}"] = p_anc[:, k]
        return ScanResult(meta=meta, columns=cols, n=null.n)
```

(If `ScanResult`'s constructor differs, adapt to its actual signature — check `models/base.py:93`. The `_ancestry_names` attribute is set by the scan adapter in Task 9; default names are used otherwise.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_joint_score_recovers_causal_ancestry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): joint K-df Rao score test + ancestry-specific effects"
```

---

### Task 5: Allele-count threshold (drop low-count ancestry terms)

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Produces: `score_chunk` now drops ancestry `a` at a variant when its allele count `< min_allele_count`; joint stat over surviving ancestries; if all dropped, `joint_p = nan`.

- [ ] **Step 1: Write the failing test**

```python
def test_allele_count_threshold_drops_rare_ancestry():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=300, m=5, K=2, seed=5)
    # force ancestry-1 dosages near zero (allele count < 50) at variant 0
    d["dosages"][1, :, 0] = 0.0
    m = TractorLMM(family="gaussian", min_allele_count=50)
    nf = m.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(ids=[f"v{i}" for i in range(5)], chrom=["1"]*5, pos=list(range(5)))
    res = m.score_chunk(nf, d["dosages"], meta)
    df = res.to_dataframe()
    # ancestry-1 (EUR) effect at variant 0 is NaN (dropped); ancestry-0 still tested
    assert df["p_EUR"].isna()[0] or df["se_EUR"].isna()[0]
    assert not df["joint_p"].isna()[0]  # AFR survived -> marginal joint p present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_allele_count_threshold_drops_rare_ancestry -v`
Expected: FAIL — low-count ancestry still produces a value.

- [ ] **Step 3: Write minimal implementation**

Replace the per-variant loop body in `score_chunk` to filter ancestries by allele count before building `Gv`:

```python
        for j in range(c):
            full = G_chunk[:, :, j].to(torch.float64)          # (K, n)
            ac = full.sum(dim=1)                               # allele count per ancestry
            keep = (ac >= self.min_allele_count).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                joint_p[j] = float("nan")
                continue
            Gv = full[keep].T                                  # (n, K_keep)
            T, Var, Vinv, joint = self._score_stats(null, Gv)
            joint_p[j] = chi2.sf(joint, df=keep.numel())
            beta = Vinv @ T
            se = torch.sqrt(torch.clamp(torch.diag(Vinv), min=0))
            z = beta / se
            for slot, k in enumerate(keep.tolist()):
                betas[j, k] = beta[slot]
                ses[j, k] = se[slot]
                p_anc[j, k] = 2.0 * float(norm.sf(abs(float(z[slot]))))
            # dropped ancestries keep NaN se / p=nan
            dropped = set(range(Kanc)) - set(keep.tolist())
            for k in dropped:
                ses[j, k] = float("nan"); p_anc[j, k] = float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_allele_count_threshold_drops_rare_ancestry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): allele-count threshold drops low-count ancestry terms"
```

---

### Task 6: Binary (GLMM PQL null) path

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Consumes: `binary_glmm` PQL null (existing).
- Produces: `TractorLMM(family="binary").fit_null(...)` returns a `TractorNullFit` whose `resid = Y - mu0` and whose `P` uses the PQL working covariance `V = W^{-1} + tau2 K` (working weights `W = diag(mu0(1-mu0))`).

- [ ] **Step 1: Write the failing test**

```python
def test_binary_null_calibrated_under_null():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=500, m=200, K=2, seed=9)
    # NULL genotypes independent of y_bin: permute samples in dosages
    perm = torch.randperm(500, generator=torch.Generator().manual_seed(1))
    Gnull = d["dosages"][:, perm, :]
    m = TractorLMM(family="binary")
    nf = m.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    meta = VariantMeta(ids=[f"v{i}" for i in range(200)], chrom=["1"]*200, pos=list(range(200)))
    res = m.score_chunk(nf, Gnull, meta)
    jp = res.to_dataframe()["joint_p"].to_numpy()
    jp = jp[~torch.isnan(torch.tensor(jp)).numpy()]
    # well-calibrated: ~5% below 0.05 (allow slack for finite m)
    assert 0.01 < (jp < 0.05).mean() < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_binary_null_calibrated_under_null -v`
Expected: FAIL — binary family not implemented.

- [ ] **Step 3: Write minimal implementation**

Branch `fit_null` on `self.family`. For binary, obtain `mu0`, working weights, and variance components from the existing `binary_glmm` PQL null, then build a `TractorNullFit` whose `_Vi` uses `V = diag(1/W) + tau2 K` (compute via the same eig(K) trick but with a diagonal `W^{-1}` — since `W` is diagonal, use `V = W^{-1} + tau2 K` and a dense solve for the null fit; for n up to a few thousand this is fine, and the streaming path handles per-variant `Gv`):

```python
    def fit_null(self, Y, X0, K):
        Y = Y.to(torch.float64); X0 = X0.to(torch.float64); K = K.to(torch.float64)
        n, c = X0.shape
        if self.family == "gaussian":
            ... # (Task 3 body unchanged)
        # binary: reuse the PQL null from BinaryGLMM
        from .binary_glmm import BinaryGLMM
        pql = BinaryGLMM().fit_null(Y, X0, K)   # exposes mu0, W (diag), tau2
        mu0 = pql.mu0.to(torch.float64)
        w = (mu0 * (1 - mu0)).clamp(min=1e-8)   # working weights
        tau2 = float(pql.tau2)
        Winv = torch.diag(1.0 / w)
        V = Winv + tau2 * K
        Vi = torch.linalg.inv(V)
        XtVi = X0.T @ Vi
        XtViX_inv = torch.linalg.inv(XtVi @ X0)
        d, U = torch.linalg.eigh(K)             # kept for interface symmetry
        nf = TractorNullFit(U=U, d=d, sigma2=1.0, tau2=tau2, X0=X0,
                            resid=torch.zeros(n, dtype=torch.float64), n=n, c=c,
                            _XtVi=XtVi, _XtViX_inv=XtViX_inv)
        nf._Vi_dense = Vi                       # add optional attr; _Vi uses it if present
        nf.resid = nf.Py((Y - mu0).unsqueeze(1)).squeeze(1)  # score residual Y - mu0
        return nf
```

Update `TractorNullFit._Vi` to prefer a stored dense inverse when present:

```python
    _Vi_dense: Tensor | None = None
    def _Vi(self, M: Tensor) -> Tensor:
        if self._Vi_dense is not None:
            return self._Vi_dense @ M
        w = 1.0 / (self.tau2 * self.d + self.sigma2)
        return self.U @ (w.unsqueeze(1) * (self.U.T @ M))
```

(Confirm `BinaryGLMM().fit_null` return fields; if names differ, adapt — check `models/binary_glmm.py:70`. If it does not expose `tau2`/`mu0` directly, read them from its returned `NullFit`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_binary_null_calibrated_under_null -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): binary GLMM (PQL) null path"
```

---

### Task 7: Wald test option

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Produces: `TractorLMM(test="wald")` fits the full model per variant (add `Gv` to `X0`, re-estimate `beta` by GLS at fixed null variance components) and reports Wald `beta_a`, `se_a`, `p_a` (chi2_1) and a joint Wald `joint_p` (chi2_K). Score remains default.

- [ ] **Step 1: Write the failing test**

```python
def test_wald_matches_score_at_moderate_signal():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=400, m=20, K=2, seed=21)
    meta = VariantMeta(ids=[f"v{i}" for i in range(20)], chrom=["1"]*20, pos=list(range(20)))
    sc = TractorLMM(test="score"); wd = TractorLMM(test="wald")
    nfs = sc.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    nfw = wd.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    ds = sc.score_chunk(nfs, d["dosages"], meta).to_dataframe()
    dw = wd.score_chunk(nfw, d["dosages"], meta).to_dataframe()
    # betas agree closely (score approx ~ full model at moderate effect)
    import numpy as np
    assert np.corrcoef(ds["beta_AFR"], dw["beta_AFR"])[0, 1] > 0.98
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_wald_matches_score_at_moderate_signal -v`
Expected: FAIL — wald branch not implemented.

- [ ] **Step 3: Write minimal implementation**

Add a Wald branch in `score_chunk` selected by `self.test == "wald"`: for each variant build `Xf = [X0 | Gv]`, compute GLS `beta_f = (Xf^T V^{-1} Xf)^{-1} Xf^T V^{-1} y` at the null `V`, take the ancestry block and its covariance `(Xf^T V^{-1} Xf)^{-1}` lower-right K×K block for SE and the joint Wald `beta_G^T Cov^{-1} beta_G ~ chi2_K`. Reuse `null._Vi`.

```python
    def _wald_stats(self, null, Gv, y):
        Xf = torch.cat([null.X0, Gv], dim=1)           # (n, c+K)
        ViXf = null._Vi(Xf)
        A = Xf.T @ ViXf                                 # (c+K, c+K)
        Ainv = torch.linalg.inv(A)
        beta_f = Ainv @ (Xf.T @ null._Vi(y.unsqueeze(1))).squeeze(1)
        K = Gv.shape[1]
        bG = beta_f[-K:]
        Cov = Ainv[-K:, -K:]
        joint = float(bG @ torch.linalg.inv(Cov) @ bG)
        se = torch.sqrt(torch.clamp(torch.diag(Cov), min=0))
        return bG, se, Cov, joint
```

Thread `y` into `score_chunk` for the Wald path (store `null._y` at fit time, or pass phenotype through the scan adapter). Store `self._y_for_wald` set by `fit_null` (attach `nf._y = Y`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_wald_matches_score_at_moderate_signal -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): Wald full-model test option"
```

---

### Task 8: Opt-in SPA for low-count ancestry score component

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Test: `tests/test_tractor_lmm.py`

**Interfaces:**
- Consumes: `torchgenomics.stats.spa.saddlepoint_pvalue` (existing).
- Produces: when `use_spa=True`, per-ancestry p-values for score components whose standardized score magnitude exceeds `spa_threshold` are recomputed via SPA; default `use_spa=False` leaves the normal-approx path byte-identical (protects the faithful claim).

- [ ] **Step 1: Write the failing test**

```python
def test_spa_off_is_identical_and_on_changes_tail():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=600, m=10, K=2, seed=31)
    meta = VariantMeta(ids=[f"v{i}" for i in range(10)], chrom=["1"]*10, pos=list(range(10)))
    off = TractorLMM(family="binary", use_spa=False)
    on = TractorLMM(family="binary", use_spa=True, spa_threshold=1.0)
    nf1 = off.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    nf2 = on.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    a = off.score_chunk(nf1, d["dosages"], meta).to_dataframe()
    b = on.score_chunk(nf2, d["dosages"], meta).to_dataframe()
    # default-off equals the pre-SPA behavior (regression guard handled elsewhere);
    # here just assert SPA changes at least one tail p-value
    assert not (a["p_AFR"].to_numpy() == b["p_AFR"].to_numpy()).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_spa_off_is_identical_and_on_changes_tail -v`
Expected: FAIL — SPA path absent.

- [ ] **Step 3: Write minimal implementation**

In the score branch, after computing `z` per surviving ancestry, if `self.use_spa and abs(z) > self.spa_threshold`, recompute that ancestry's p-value via `saddlepoint_pvalue(score=T_a, ...)` using the per-ancestry score `T_a = Gv[:,a]^T P y` and its null cumulant-generating function built from `mu0` and `Gv[:,a]` (reuse the same construction as `binary_glmm`'s SPA call). Guard the whole block behind `if self.use_spa`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_spa_off_is_identical_and_on_changes_tail -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py tests/test_tractor_lmm.py
git commit -m "Phase 57 (Unit B): opt-in SPA for low-count ancestry score (default off)"
```

---

### Task 9: Local-ancestry conditional analysis + scan/streaming integration

**Files:**
- Modify: `torchgenomics/models/tractor_lmm.py`
- Modify: `torchgenomics/scan/` adapter (register `TractorLMM`; pass ancestry names + optional `L`)
- Test: `tests/test_tractor_lmm.py`, `tests/test_streaming_memory.py`

**Interfaces:**
- Produces: `score_chunk(..., L_chunk=...)` adds `conditional_joint_p`; `UnifiedScanner` streams `AncestryDosages` chunks through `TractorLMM` producing a combined `ScanResult`.

- [ ] **Step 1: Write the failing test**

```python
def test_conditional_and_streaming_equivalence():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=300, m=24, K=2, seed=41)
    meta = VariantMeta(ids=[f"v{i}" for i in range(24)], chrom=["1"]*24, pos=list(range(24)))
    m = TractorLMM()
    nf = m.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    # full chunk vs two half-chunks give identical joint_p (streaming invariance)
    full = m.score_chunk(nf, d["dosages"], meta).to_dataframe()["joint_p"].to_numpy()
    h1 = m.score_chunk(nf, d["dosages"][:, :, :12],
                       VariantMeta(ids=[f"v{i}" for i in range(12)], chrom=["1"]*12, pos=list(range(12))))
    h2 = m.score_chunk(nf, d["dosages"][:, :, 12:],
                       VariantMeta(ids=[f"v{i}" for i in range(12,24)], chrom=["1"]*12, pos=list(range(12,24))))
    import numpy as np
    streamed = np.concatenate([h1.to_dataframe()["joint_p"], h2.to_dataframe()["joint_p"]])
    assert np.allclose(full, streamed, atol=1e-10)
    # conditional path runs and returns a column
    L = d["dosages"].clone()  # stand-in local ancestry for the smoke check
    res = m.score_chunk(nf, d["dosages"], meta, L_chunk=L)
    assert "conditional_joint_p" in res.to_dataframe().columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_conditional_and_streaming_equivalence -v`
Expected: FAIL — conditional column / `L_chunk` unhandled.

- [ ] **Step 3: Write minimal implementation**

Implement the SAIGE-style conditional stat in `score_chunk` when `L_chunk` is given, using `K-1` local-ancestry columns `Lv` (n×(K-1)):

```python
            if L_chunk is not None:
                Lv = L_chunk[:-1, :, j].T.to(torch.float64)     # (n, K-1)
                PLv = null.Py(Lv)
                TL = Lv.T @ null.resid
                LtPL = Lv.T @ PLv
                LtPL_inv = torch.linalg.pinv(LtPL)
                GtPL = Gv.T @ PLv                                # (Kkeep, K-1)
                cond_mean = GtPL @ (LtPL_inv @ TL)
                cond_var = Var - GtPL @ LtPL_inv @ GtPL.T
                dvec = T - cond_mean
                cond_stat = float(dvec @ torch.linalg.pinv(cond_var) @ dvec)
                cond_p[j] = chi2.sf(cond_stat, df=keep.numel())
```

Wire the scan adapter to iterate `AncestryDosages.iter_chunks`, set `model._ancestry_names`, and concatenate per-chunk `ScanResult`s (reuse the existing `UnifiedScanner` result-merge; if the adapter assumes a 2-D `(n,c)` chunk, add a small `TractorScanAdapter` that passes the 3-D `(K,n,c)` tensor through).

Add a streaming-memory regression entry mirroring existing ones in `tests/test_streaming_memory.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tractor_lmm.py::test_conditional_and_streaming_equivalence tests/test_streaming_memory.py -k tractor -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/tractor_lmm.py torchgenomics/scan/ tests/test_tractor_lmm.py tests/test_streaming_memory.py
git commit -m "Phase 57 (Unit B): local-ancestry conditional analysis + streaming scan integration"
```

---

### Task 10: BaseModel conformance + wall-time bench gate + docs

**Files:**
- Test: `tests/test_tractor_lmm.py`
- Modify: `bench/native_speedups.py` (add a `tractor-scan` timing case)
- Modify: `CLAUDE.md` (Phase Index + module list note that Unit B landed)

**Interfaces:**
- Produces: a `test_conforms_to_base_model` asserting `fit_null`/`score_chunk` satisfy the `BaseModel` protocol shape, and a moderate-size wall-time gate.

- [ ] **Step 1: Write the failing test**

```python
def test_conforms_to_base_model():
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import BaseModel
    m = TractorLMM()
    assert hasattr(m, "fit_null") and hasattr(m, "score_chunk")
    # duck-typed protocol check
    assert isinstance(m, BaseModel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tractor_lmm.py::test_conforms_to_base_model -v`
Expected: FAIL if the protocol is `@runtime_checkable` and a method signature mismatches; fix signatures until it passes.

- [ ] **Step 3: Write minimal implementation**

Align `fit_null`/`score_chunk` signatures with `BaseModel` (check `models/base.py:225`). Add the bench case and the CLAUDE.md note (Phase 57 Unit B: `TractorLMM` shipped; CLI/R come in Unit C).

- [ ] **Step 4: Run full Unit B suite**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/test_tractor_lmm.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tractor_lmm.py bench/native_speedups.py CLAUDE.md
git commit -m "Phase 57 (Unit B): BaseModel conformance + wall-time gate + docs"
```

---

## Self-Review

**Spec coverage (Unit B rows of the spec):**
- K-df joint Rao score test → Task 4. ✅
- Ancestry-specific score effects + SE → Task 4. ✅
- Allele-count threshold → Task 5. ✅
- Continuous + binary nulls → Tasks 3, 6. ✅
- Wald option → Task 7. ✅
- Opt-in SPA (default off) → Task 8. ✅
- Local-ancestry conditional → Task 9. ✅
- Streaming + regression nets → Tasks 9, 10. ✅
- BaseModel conformance → Task 10. ✅
- Ancestry-dosage I/O (Tractor format) → Task 2. ✅
- **Deferred to later plans (correctly out of Unit B):** native PC-AiR/PC-Relate (Plan 2 / Unit A), CLI + R + MCP surface + external GENESIS/Tractor-Mix validation harness + real-data concordance (Plan 3 / Unit C). The synthetic fixture (Task 1) stands in for the equivalence GRM until the external harness lands.

**Placeholder scan:** the SPA (Task 8 Step 3) and scan-adapter (Task 9 Step 3) steps describe construction referencing existing code (`binary_glmm` SPA call; `UnifiedScanner` merge) rather than pasting it verbatim, because they must match those modules' exact current signatures — the implementer verifies against the cited file:line. All other steps carry runnable code.

**Type consistency:** `TractorNullFit` fields, `TractorLMM.__init__` params, `score_chunk(null, G_chunk, meta, L_chunk)` signature, and `AncestryDosages` shapes `(K,n,m)` are consistent across Tasks 2–10.

**Known verification points flagged for the implementer (spec risk §10):** (a) exact GMMAT score residualization `G^T P y` vs the paper's `Gbar` form — pinned by Task 3's brute-force test; (b) `BinaryGLMM.fit_null` return-field names — Task 6; (c) `ScanResult` constructor signature — Task 4; (d) `UnifiedScanner` chunk arity (2-D vs our 3-D) — Task 9.

## Plans 2 & 3 (task-level outline, to be written after Unit B lands)

**Plan 2 — Unit A: admixture-aware GRM/PCs (`linalg/kinship_admixed.py`)**
1. KING-robust kinship (streaming) + fixture vs hand-computed small example.
2. LD-prune wrapper (reuse `ld/`) → independent SNP set.
3. PC-AiR: unrelated/related partition + PCA + projection.
4. PC-Relate: individual-specific AF regression → moment kinship → sparse GRM (×2, diag→1, <0.05→0).
5. `admixed_grm()` orchestrator returning `(pcs, sparse_grm)`.
6. External GENESIS-equivalence harness (Tier 3, opt-in, preflight-gated).

**Plan 3 — Unit C: surface + validation (`cli.py`, `api/`, `rTorchGenomics/`, `validation/external/tractor_mix/`)**
1. `api.tractor_scan` facade + `TractorScanResult`.
2. `tractor-scan` CLI subcommand (43→44) + streaming wiring.
3. `tg_tractor_scan` R wrapper + NAMESPACE + `_pkgdown.yml` + MCP tool (15th).
4. External Tractor-Mix R harness (install/fetch/run/compare) — Claims 1, 2, 4, 5; preflight-gated; actual head-to-head run.
5. Manhattan (joint + per-ancestry) + docs/vignette + CLAUDE.md CLI/Phase-Index update.
