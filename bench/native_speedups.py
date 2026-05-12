"""Benchmark every TorchGWAS native C++ accelerator against its pure-Python
reference and emit a markdown table or machine-readable JSON.

Why this exists
---------------
By the end of Phase 41ab the project ships 28 native extensions under
``torchgwas._native``. Most were added on the assumption that the C++
kernel beats the pure-Python path on realistic inputs — but until now
that assumption was never measured end-to-end. This script:

1. Picks one realistic input size per kernel (small enough to run in
   seconds, large enough that the per-call pybind11 / GIL-release
   overhead is amortized).
2. Runs the kernel under both ``TORCHGWAS_DISABLE_NATIVE=1`` and unset.
3. Reports the median wall time over a small number of repeats and the
   resulting speedup.
4. Writes the table to ``bench/native_speedups.md`` and stdout — or, in
   ``--output json`` mode, dumps a structured per-row record that the
   ``bench/diff_perf.py`` regression-CI script consumes (Efficiency E5).

The default markdown output is *informational*. It does not gate any
test. Numbers vary across machines but the *ordering* of speedups is
what informs which kernels are worth further optimization (e.g. OpenMP)
and which are launch-overhead-bound at typical sizes.

The JSON output, however, is what the perf regression CI gates on.
``bench/diff_perf.py`` reads two JSON files (master baseline + PR head)
and flags any kernel whose p50 / p95 wall-time has regressed beyond the
configured threshold.

Run with::

    # Human-driven informational sweep (markdown, all kernels):
    python bench/native_speedups.py

    # Fast CI sweep (JSON, subset of high-signal kernels):
    python bench/native_speedups.py \\
        --output json \\
        --output-path benchmarks.json \\
        --kernel-subset gabriel_blocks,pelt,cc_graph,impute_knn,...

Set ``TORCHGWAS_BENCH_QUICK=1`` to use ``n_repeats=2`` for a fast
sanity check.

Set ``TORCHGWAS_BENCH_REALISTIC=1`` to scale each kernel's input 2–4×
beyond the small defaults (capped per-kernel so the pure-Python path
still finishes within a few minutes). This validates that the Phase 41
speedups do not regress at sizes closer to what real users hit, which
is how the Spine regression in Phase 41ad was ultimately diagnosed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

# Ensure the repo root is on sys.path when invoked as ``python
# bench/native_speedups.py`` from the repo root — Python adds the
# script's directory (``bench/``) to ``sys.path[0]``, which would shadow
# the in-tree ``torchgwas/`` package. Inserting the parent (the repo
# root) first makes ``import torchgwas...`` resolve to the editable
# checkout regardless of whether the package is also pip-installed.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch


# ---------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------

QUICK = bool(os.environ.get("TORCHGWAS_BENCH_QUICK"))
REALISTIC = bool(os.environ.get("TORCHGWAS_BENCH_REALISTIC"))
# CI mode: input sizes shrunk so the *python* reference path finishes
# under ~5 s per kernel even in the unrolled-loop kernels (Gabriel,
# PELT, Big-LD, etc). The native path is still measured at full
# accuracy — we're not skipping it, we're just not asking the python
# reference to run a 90 s O(m⁴) scan inside a 6-min CI budget. The
# regression test itself only cares that the native p50 doesn't
# regress; the CI-mode python timing is purely a sanity sentinel.
CI = bool(os.environ.get("TORCHGWAS_BENCH_CI"))
DEFAULT_REPEATS = 2 if (QUICK or CI) else 5


def _pick(small, realistic, ci=None):
    """Return the size for the active mode.

    Order of precedence: CI > realistic > small. The ``ci`` arg is
    optional — if a kernel doesn't supply one, we fall back to ``small``
    (which is already conservative for most kernels).
    """
    if CI:
        return ci if ci is not None else small
    return realistic if REALISTIC else small


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches ``numpy.percentile``).

    We avoid pulling numpy in here because the bench module must remain
    importable in environments where numpy is mocked or pinned to an
    unusual version (the regression CI installs a torch-pinned set).
    ``pct`` is 0.0 … 1.0 (so ``_percentile(xs, 0.95)`` is the 95th).
    """
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    sorted_v = sorted(values)
    rank = pct * (len(sorted_v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = rank - lo
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac


def _time_call(fn: Callable, args: tuple, n_repeats: int) -> tuple[float, float, float]:
    """Return (median, p50, p95) wall time of ``fn(*args)`` over ``n_repeats`` runs.

    Always does one warmup call before measurement. ``median`` is the
    statistics.median (used by the existing markdown table for
    backward-compat with prior runs); ``p50`` and ``p95`` are
    linear-interpolation percentiles consumed by the JSON output and
    by ``bench/diff_perf.py``. For ``n_repeats >= 5`` the p50 and the
    median agree exactly when the sample size is odd; we keep both
    forms in the schema so callers don't have to special-case.
    """
    fn(*args)  # warmup
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return (
        statistics.median(times),
        _percentile(times, 0.5),
        _percentile(times, 0.95),
    )


@dataclass
class Bench:
    name: str
    group: str
    builder: Callable[[], tuple]  # returns (args,) for runner
    runner: Callable[..., Any]
    repeats: int = DEFAULT_REPEATS
    slug: str = ""  # short identifier for --kernel-subset (filled in registry)


@dataclass
class BenchResult:
    """One bench × one mode (python | native) timing record.

    Keeping the percentile fields here lets the JSON output schema and
    the markdown table share a single source of truth.
    """

    median_seconds: float
    p50_seconds: float
    p95_seconds: float
    n_repeats: int


def run_bench(b: Bench) -> tuple[BenchResult, BenchResult]:
    """Time the builder's args under both paths.

    Returns (python_result, native_result). The builder is invoked
    twice — once per path — so the timings cannot share mutated input
    state across paths.
    """
    # Python path
    os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    args_py = b.builder()
    py_med, py_p50, py_p95 = _time_call(b.runner, args_py, b.repeats)

    # Native path
    os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    args_nat = b.builder()
    nat_med, nat_p50, nat_p95 = _time_call(b.runner, args_nat, b.repeats)

    return (
        BenchResult(py_med, py_p50, py_p95, b.repeats),
        BenchResult(nat_med, nat_p50, nat_p95, b.repeats),
    )


# =====================================================================
# Bench definitions
# =====================================================================
#
# Each builder returns a tuple of positional args; each runner takes
# those args and runs the kernel under whatever path the env var picks.
# Inputs are intentionally modest so the script finishes in a few
# minutes on a laptop.

# ---------- PGS: PRS-CS / LDpred2 Gibbs samplers --------------------

def _build_prscs_block():
    rng = torch.Generator().manual_seed(0)
    m = _pick(200, 400, ci=80)
    n_iter = _pick(200, 300, ci=60)
    R = torch.randn(m, m, dtype=torch.float64, generator=rng)
    R = (R + R.T) / 2.0
    R += m * torch.eye(m, dtype=torch.float64)
    R = R / R.diag().max()
    beta_std = torch.randn(m, dtype=torch.float64, generator=rng) * 0.05
    return (beta_std, R, 10000.0, 1e-2, 1.0, 0.5, n_iter, 50, rng)


def _run_prscs_block(beta_std, R, n_eff, phi, a, b, n_iter, n_burnin, rng):
    from torchgwas.pgs.prscs import _prscs_gibbs_block_dispatch
    rng2 = torch.Generator().manual_seed(0)
    return _prscs_gibbs_block_dispatch(
        beta_std, R, n_eff, phi, a, b, n_iter, n_burnin, rng2,
    )


def _build_ldpred2_block():
    rng = torch.Generator().manual_seed(0)
    m = _pick(200, 400, ci=80)
    n_iter = _pick(200, 300, ci=60)
    R = torch.randn(m, m, dtype=torch.float64, generator=rng)
    R = (R + R.T) / 2.0
    R += m * torch.eye(m, dtype=torch.float64)
    R = R / R.diag().max()
    beta_std = torch.randn(m, dtype=torch.float64, generator=rng) * 0.05
    return (beta_std, R, 0.01, 0.3, 10000.0, n_iter, 50, False)


def _run_ldpred2_block(beta_std, R, p_causal, h2, n_eff, n_iter, n_burnin, sparse):
    from torchgwas.pgs.ldpred2 import _ldpred2_gibbs_block_dispatch
    rng = torch.Generator().manual_seed(0)
    return _ldpred2_gibbs_block_dispatch(
        beta_std, R, p_causal, h2, n_eff, n_iter, n_burnin, sparse, rng,
    )


# ---------- PGS: C+T clumping ----------------------------------------

def _build_ct():
    from torchgwas.pgs.base import LDReference
    rng = torch.Generator().manual_seed(0)
    m = _pick(1500, 3000)
    R = torch.eye(m, dtype=torch.float64) + 0.05 * torch.randn(
        m, m, dtype=torch.float64, generator=rng,
    )
    R = (R + R.T) / 2.0
    p = torch.rand(m, dtype=torch.float64, generator=rng) * 1e-3
    chr_labels = ["1"] * m
    pos = list(range(0, m * 1000, 1000))
    snp_ids = [f"rs{i}" for i in range(m)]
    a1 = ["A"] * m
    a2 = ["T"] * m
    af = torch.full((m,), 0.3, dtype=torch.float64)
    ld_ref = LDReference(
        snp=snp_ids, chr=chr_labels, pos=pos, a1=a1, a2=a2, af=af,
        mode="full", R_full=R, n_ref=200,
    )
    return (p, ld_ref, chr_labels, pos, 1e-3, 0.1, 250.0)


def _run_ct(p, ld_ref, chr_labels, pos, p_thr, r2_thr, window_kb):
    from torchgwas.pgs.ct import _clump_with_ld_reference_dispatch
    return _clump_with_ld_reference_dispatch(
        p, ld_ref, chr_labels, pos,
        p_threshold=p_thr, r2_threshold=r2_thr, window_kb=window_kb,
    )


# ---------- LD: PELT change-point ------------------------------------

def _build_pelt():
    # NOTE: realistic mode keeps n=5000 because the small-bench Python
    # path is already ~700s; doubling n would need ~45 min per run.
    # CI mode uses n=400 — the python DP is O(n²), so 400² = 160k ops
    # vs 5000² = 25M; that puts python under ~1 s.
    rng = np.random.default_rng(0)
    n = _pick(5000, 5000, ci=400)
    sig = np.concatenate([
        rng.normal(0.0, 1.0, n // 4),
        rng.normal(2.0, 1.0, n // 4),
        rng.normal(0.0, 1.0, n // 4),
        rng.normal(-1.5, 1.0, n - 3 * (n // 4)),
    ])
    return (torch.tensor(sig, dtype=torch.float64), 5.0)


def _run_pelt(signal, penalty):
    from torchgwas.ld._changepoint import dp_changepoint
    return dp_changepoint(signal, penalty, cost_fn="gaussian")


# ---------- PGS: ESS Geyer ------------------------------------------

def _build_ess():
    rng = torch.Generator().manual_seed(0)
    n_chains = 4
    n_samples = _pick(8000, 16000)
    n_params = 50
    chains = torch.randn(n_chains, n_samples, n_params, generator=rng,
                         dtype=torch.float64)
    return (chains,)


def _run_ess(chains):
    from torchgwas.pgs.diagnostics import ess
    return ess(chains)


# ---------- LD: connected components --------------------------------

def _build_cc():
    rng = torch.Generator().manual_seed(0)
    n = _pick(800, 1500)
    A = torch.rand(n, n, dtype=torch.float64, generator=rng)
    A = (A + A.T) / 2.0
    return (A, 0.95)


def _run_cc(A, t):
    from torchgwas.ld._graph_utils import connected_components
    return connected_components(A, threshold=t)


# ---------- LD: Gabriel haplotype blocks ----------------------------

def _build_gabriel():
    # NOTE: realistic mode keeps m=300 because the small-bench Python
    # path is already ~120s (O(m⁴) Python scan); doubling m → 1000s+.
    # CI mode uses m=80 to keep python under ~3 s.
    from torchgwas.ld import compute_pairwise_ld
    rng = torch.Generator().manual_seed(0)
    n = 400
    m = _pick(300, 300, ci=80)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)
    return (pld, pos, chrs)


def _run_gabriel(pld, pos, chrs):
    from torchgwas.ld._blocks import detect_blocks_gabriel
    return detect_blocks_gabriel(pld, pos, chrs)


# ---------- LD: Big-LD ----------------------------------------------

def _build_big_ld():
    rng = torch.Generator().manual_seed(0)
    n, m = 400, _pick(600, 1000)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    return (G, pos, chrs)


def _run_big_ld(G, pos, chrs):
    from torchgwas.ld._blocks_literature import detect_blocks_big_ld
    return detect_blocks_big_ld(G, pos, chrs, r2_threshold=0.2)


# ---------- LD: DP-optimize -----------------------------------------

def _build_dp_opt():
    rng = torch.Generator().manual_seed(0)
    n, m = 300, _pick(250, 350)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    return (G, pos, chrs)


def _run_dp_opt(G, pos, chrs):
    from torchgwas.ld._blocks_literature import detect_blocks_dp_optimize
    return detect_blocks_dp_optimize(G, pos, chrs, objective="haplotype_diversity")


# ---------- LD: CC-graph --------------------------------------------

def _build_cc_graph():
    rng = torch.Generator().manual_seed(0)
    n, m = 300, _pick(500, 900, ci=150)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    return (G, pos, chrs)


def _run_cc_graph(G, pos, chrs):
    from torchgwas.ld._blocks_literature import detect_blocks_cc_graph
    return detect_blocks_cc_graph(G, pos, chrs, r2_threshold=0.3, window=80)


# ---------- LD: GWAS-aligned -----------------------------------------

def _build_gwas_aligned():
    rng = torch.Generator().manual_seed(0)
    n, m = 300, _pick(200, 350)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    return (G, pos, chrs)


def _run_gwas_aligned(G, pos, chrs):
    from torchgwas.ld._blocks_novel import detect_blocks_gwas_aligned
    return detect_blocks_gwas_aligned(G, pos, chrs, max_block_snps=20)


# ---------- LD: spine ------------------------------------------------

def _build_spine():
    from torchgwas.ld import compute_pairwise_ld
    rng = torch.Generator().manual_seed(0)
    n, m = 400, _pick(250, 500)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)
    return (pld, pos, chrs)


def _run_spine(pld, pos, chrs):
    from torchgwas.ld._blocks import detect_blocks_spine
    return detect_blocks_spine(pld, pos, chrs, d_prime_threshold=0.5)


# ---------- LD: ld_decay_signal --------------------------------------

def _build_ld_decay():
    rng = torch.Generator().manual_seed(0)
    n_var = _pick(2000, 4000)
    n_pairs = _pick(30000, 80000)
    idx_i = torch.randint(0, n_var, (n_pairs,), generator=rng)
    idx_j = torch.randint(0, n_var, (n_pairs,), generator=rng)
    r2 = torch.rand(n_pairs, dtype=torch.float64, generator=rng)
    return (r2, idx_i, idx_j, n_var, 10)


def _run_ld_decay(r2, idx_i, idx_j, n_var, k):
    from torchgwas.ld._changepoint import ld_decay_signal
    return ld_decay_signal(r2, idx_i, idx_j, n_var, k_neighbors=k)


# ---------- LD: greedy MWIS ------------------------------------------

def _build_mwis():
    rng = np.random.default_rng(0)
    n = _pick(5000, 15000)
    starts = rng.integers(0, 100000, size=n)
    lens = rng.integers(50, 500, size=n)
    weights = rng.uniform(0.1, 5.0, size=n)
    return ([(int(s), int(s + l), float(w)) for s, l, w in zip(starts, lens, weights)],)


def _run_mwis(intervals):
    from torchgwas.ld._graph_utils import greedy_mwis
    return greedy_mwis(intervals)


# ---------- LD: Wall-Pritchard permutation ---------------------------

def _build_wall_pritchard():
    from torchgwas.ld import compute_pairwise_ld
    rng = torch.Generator().manual_seed(0)
    n, m = 200, _pick(150, 250)
    n_perm = _pick(200, 300)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)
    return (pld, pos, chrs, n_perm)


def _run_wall_pritchard(pld, pos, chrs, n_perm):
    from torchgwas.ld._blocks_diagnostics import compute_wall_pritchard_diagnostics
    return compute_wall_pritchard_diagnostics(
        pld, pos, chrs, n_permutations=n_perm,
    )


# ---------- LD: uncertainty blocks -----------------------------------

def _build_uncertainty():
    rng = torch.Generator().manual_seed(0)
    n, m = 300, _pick(200, 350)
    G = torch.randint(0, 3, (n, m), generator=rng).to(torch.float64)
    # Synthetic high-confidence GP probs
    gp = torch.zeros(n, m, 3, dtype=torch.float64)
    for i in range(n):
        for j in range(m):
            gp[i, j, int(G[i, j].item())] = 0.95
            for k in range(3):
                if k != int(G[i, j].item()):
                    gp[i, j, k] = 0.025
    pos = list(range(0, m * 1000, 1000))
    chrs = ["1"] * m
    return (G, gp, 2, pos, chrs)


def _run_uncertainty(G, gp, ploidy, pos, chrs):
    from torchgwas.ld._blocks_novel import detect_blocks_uncertainty
    return detect_blocks_uncertainty(G, gp, ploidy, pos, chrs)


# ---------- LD: cross-pop stability ----------------------------------

def _build_cross_pop():
    rng = torch.Generator().manual_seed(0)
    n_per_pop = _pick(80, 150)
    m = _pick(80, 180)
    G = torch.randint(0, 3, (3 * n_per_pop, m), generator=rng).to(torch.float64)
    pop_ids = ["A"] * n_per_pop + ["B"] * n_per_pop + ["C"] * n_per_pop
    pos = list(range(m))
    chrs = ["1"] * m
    return (G, pop_ids, None, pos, chrs)


def _run_cross_pop(G, pop_ids, weights, pos, chrs):
    from torchgwas.ld._blocks_novel import detect_blocks_cross_pop
    return detect_blocks_cross_pop(
        G, pop_ids, weights, pos, chrs,
        n_blocks_hint=4, stability_threshold=0.0, min_block_snps=2,
    )


# ---------- preprocess: imputation -----------------------------------

def _build_impute_mode():
    rng = np.random.default_rng(0)
    n = _pick(1500, 3000, ci=400)
    m = _pick(800, 1500, ci=200)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    miss = rng.random(size=(n, m)) < 0.1
    G[miss] = np.nan
    return (torch.from_numpy(G),)


def _run_impute_mode(G):
    from torchgwas.preprocess.impute import impute_mode
    return impute_mode(G)


def _build_impute_knn():
    rng = np.random.default_rng(0)
    n = _pick(400, 700, ci=150)
    m = _pick(200, 350, ci=80)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    miss = rng.random(size=(n, m)) < 0.1
    G[miss] = np.nan
    G_t = torch.from_numpy(G)
    # GRM-like similarity from filled matrix
    G_filled = torch.from_numpy(np.where(miss, 1.0, G))
    K = (G_filled @ G_filled.T) / m
    return (G_t, K, 5)


def _run_impute_knn(G, K, k):
    from torchgwas.preprocess.impute import impute_knn
    return impute_knn(G, K, k=k)


def _build_impute_ld():
    # NOTE: pure-Python path is O(n·m·w) and already ~20s at 400×400.
    # Realistic mode scales n and m modestly to keep Python under ~90s.
    rng = np.random.default_rng(0)
    n, m = _pick(400, 600), _pick(400, 600)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    miss = rng.random(size=(n, m)) < 0.1
    G[miss] = np.nan
    return (torch.from_numpy(G), 30)


def _run_impute_ld(G, w):
    from torchgwas.preprocess.impute import impute_ld
    return impute_ld(G, window_size=w)


# ---------- preprocess: HWE ------------------------------------------

def _build_hwe():
    rng = np.random.default_rng(0)
    n = _pick(1500, 2500, ci=500)
    m = _pick(5000, 10000, ci=1500)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    af = G.mean(axis=0) / 2.0
    return (torch.from_numpy(G), torch.from_numpy(af), 2)


def _run_hwe(G, af, ploidy):
    from torchgwas.preprocess.qc import _compute_hwe_pvalue
    return _compute_hwe_pvalue(G, af, ploidy=ploidy)


def _build_hwe_dr():
    rng = np.random.default_rng(0)
    n, m = _pick(1000, 1800), _pick(1500, 3000)
    # Tetraploid: dosages 0..4
    G = rng.integers(0, 5, size=(n, m)).astype(np.float64)
    af = G.mean(axis=0) / 4.0
    return (torch.from_numpy(G), torch.from_numpy(af), 4)


def _run_hwe_dr(G, af, ploidy):
    from torchgwas.preprocess.qc import _compute_hwe_double_reduction
    return _compute_hwe_double_reduction(G, af, ploidy)


# ---------- stats: SPA ------------------------------------------------

def _build_spa():
    rng = torch.Generator().manual_seed(0)
    n = _pick(1500, 2500, ci=500)
    m = _pick(1000, 2000, ci=300)
    mu = torch.rand(n, generator=rng, dtype=torch.float64).clamp(0.05, 0.95)
    G = torch.rand(n, m, generator=rng, dtype=torch.float64) * 2.0
    Y = (torch.rand(n, generator=rng, dtype=torch.float64) < mu).to(torch.float64)
    score = (G * (Y - mu).unsqueeze(1)).sum(dim=0)
    return (score, mu, G, 2.0)


def _run_spa(score, mu, G, thr):
    from torchgwas.stats.spa import saddlepoint_pvalue
    return saddlepoint_pvalue(score, mu, G, threshold=thr)


# ---------- post-GWAS: LDSC h² jackknife -----------------------------

def _build_ldsc():
    rng = torch.Generator().manual_seed(0)
    m = _pick(8000, 40000, ci=4000)
    ld = torch.rand(m, generator=rng, dtype=torch.float64) * 50 + 5
    n_eff = 100000
    h2_true = 0.3
    chi2 = 1.0 + (n_eff * h2_true / m) * ld + torch.randn(
        m, generator=rng, dtype=torch.float64,
    )
    chi2 = chi2.clamp(min=0.01)
    return (chi2, ld, n_eff, m)


def _run_ldsc(chi2, ld, n_eff, m):
    from torchgwas.postgwas._ldsc import ldsc_h2
    return ldsc_h2(chi2, ld, n_eff, m_total=m, n_blocks=200)


# ---------- linalg: streaming GRM (post-streaming hot path) ----------

def _build_grm_streaming():
    """Build a chunked-iter for ``grm_vanraden_streaming``.

    The streaming GRM is the post-Efficiency-E1/E4 hot path for every
    LMM-style scan (lmm-scan, mvlmm-scan, glm-scan PC route,
    conditional-scan, mtmet-scan, ocf-scan, family-scan, plus the
    pipeline LMM/mvLMM branches). It accumulates ``n × n`` per-chunk
    GEMMs in FP64 — both the GEMM and the accumulator are fair-game
    for native acceleration. Picking ``n=600, m=8000`` keeps the pure
    -Python path under ~2 s while still exercising a non-trivial
    accumulator load.
    """
    rng = torch.Generator().manual_seed(0)
    n = _pick(600, 1200, ci=300)
    m = _pick(8000, 16000, ci=2000)
    chunk = _pick(800, 1600, ci=500)
    G = (torch.rand(n, m, generator=rng, dtype=torch.float64) * 2.0).round()
    chunks: list[tuple[torch.Tensor, None]] = []
    for s in range(0, m, chunk):
        chunks.append((G[:, s : s + chunk].clone(), None))
    return (chunks, n)


def _run_grm_streaming(chunks, n):
    from torchgwas.linalg.kinship import grm_vanraden_streaming
    return grm_vanraden_streaming(iter(chunks), n_samples=n, ploidy=2)


# ---------- linalg: full-batch GRM (pre-streaming path) -------------

def _build_grm_vanraden():
    """Allocate a single ``(n, m)`` tensor and call ``grm_vanraden``.

    Same ploidy/scale envelope as the streaming bench, but exercises
    the original full-batch helper that several non-streaming entry
    points still rely on (zarr / hapmap / direct dosage CSV ingest).
    Listed in the CI subset because it's the "before" reference for
    every streaming-vs-materialized regression discussion.
    """
    rng = torch.Generator().manual_seed(0)
    n = _pick(600, 1200, ci=300)
    m = _pick(4000, 8000, ci=1500)
    G = (torch.rand(n, m, generator=rng, dtype=torch.float64) * 2.0).round()
    return (G,)


def _run_grm_vanraden(G):
    from torchgwas.linalg.kinship import grm_vanraden
    return grm_vanraden(G, ploidy=2)


# ---------- NA1: bayes-scan-rss SuSiE-RSS ---------------------------

def bench_bayes_scan_rss(p: int = 1000, n: int = 5000) -> dict[str, Any]:
    """Wall-time benchmark for ``bayes-scan-rss`` on a synthetic single locus.

    Pair with the memory regression net in
    ``tests/test_streaming_memory.py::TestBayesScanRssMemory`` (per
    ``feedback_regression_nets``: every perf-sensitive path needs paired
    memory + wall-time guards). The memory test asserts the
    ``O(p_block_max^2 + L * p_total)`` bound from NA1 design spec
    section 4.5; this function records the end-to-end IBSS wall time so
    a follow-on diff against a baseline (mirroring the
    ``bench/diff_perf.py`` flow for the BENCHES registry) can flag
    regressions in the SuSiE-RSS hot path.

    Standalone callable (not registered in ``BENCHES``) because
    ``BayesianVSRss.fit_rss`` does not have a native-vs-python toggle —
    it's pure-torch — so the ``run_bench`` python/native split would be
    redundant. Callers (CI / ad-hoc) invoke it directly with the
    ``p`` and ``n`` of interest.

    Args:
        p: number of variants in the locus (default 1000).
        n: GWAS sample size used inside the SuSiE-RSS likelihood
            (default 5000). Only enters ``BayesianVSRss.fit_rss`` as a
            scalar; doesn't allocate.

    Returns:
        ``dict`` with keys ``name``, ``p``, ``n``, ``wall_time_sec``,
        ``n_iter``, ``converged`` — same schema as one row of the JSON
        emitted by the main ``bench/native_speedups.py`` flow, so it can
        be appended to a regression record.
    """
    from torchgwas.models.bayesian_vs_rss import BayesianVSRss

    rng = torch.Generator()
    rng.manual_seed(0)
    z = torch.randn(p, generator=rng, dtype=torch.float64)
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=10, max_iter=50)

    start = time.perf_counter()
    result = model.fit_rss(z=z, R=R, n=n)
    elapsed = time.perf_counter() - start

    return {
        "name": "bayes-scan-rss",
        "p": p,
        "n": n,
        "wall_time_sec": elapsed,
        "n_iter": result.n_iter,
        "converged": result.converged,
    }


# =====================================================================
# Bench registry
# =====================================================================

BENCHES: list[Bench] = [
    Bench("PRS-CS Gibbs (m=200, iter=200)", "PGS",
          _build_prscs_block, _run_prscs_block, slug="prscs_gibbs"),
    Bench("LDpred2 Gibbs (m=200, iter=200)", "PGS",
          _build_ldpred2_block, _run_ldpred2_block, slug="ldpred2_gibbs"),
    Bench("C+T clumping (m=1500)", "PGS",
          _build_ct, _run_ct, slug="ct_clumping"),
    Bench("ESS Geyer (4×8000×50)", "PGS",
          _build_ess, _run_ess, slug="ess_geyer"),

    Bench("PELT (n=5000, gaussian)", "LD",
          _build_pelt, _run_pelt, slug="pelt"),
    Bench("connected_components (n=800)", "LD",
          _build_cc, _run_cc, slug="connected_components"),
    Bench("Gabriel blocks (m=300)", "LD",
          _build_gabriel, _run_gabriel, slug="gabriel_blocks"),
    Bench("Big-LD blocks (m=600)", "LD",
          _build_big_ld, _run_big_ld, slug="big_ld_blocks"),
    Bench("DP-optimize blocks (m=250)", "LD",
          _build_dp_opt, _run_dp_opt, slug="dp_optimize_blocks"),
    Bench("CC-graph blocks (m=500)", "LD",
          _build_cc_graph, _run_cc_graph, slug="cc_graph_blocks"),
    Bench("GWAS-aligned blocks (m=200)", "LD",
          _build_gwas_aligned, _run_gwas_aligned, slug="gwas_aligned_blocks"),
    Bench("Spine blocks (m=250)", "LD",
          _build_spine, _run_spine, slug="spine_blocks"),
    Bench("ld_decay_signal (n_pairs=30000)", "LD",
          _build_ld_decay, _run_ld_decay, slug="ld_decay_signal"),
    Bench("greedy_mwis (n=5000)", "LD",
          _build_mwis, _run_mwis, slug="greedy_mwis"),
    Bench("Wall-Pritchard (m=150, n_perm=200)", "LD",
          _build_wall_pritchard, _run_wall_pritchard, slug="wall_pritchard"),
    Bench("uncertainty blocks (m=200)", "LD",
          _build_uncertainty, _run_uncertainty, slug="uncertainty_blocks"),
    Bench("cross-pop stability (3 pops, m=80)", "LD",
          _build_cross_pop, _run_cross_pop, slug="cross_pop_stability"),

    Bench("impute_mode (1500×800)", "preprocess",
          _build_impute_mode, _run_impute_mode, slug="impute_mode"),
    Bench("impute_knn (400×200, k=5)", "preprocess",
          _build_impute_knn, _run_impute_knn, slug="impute_knn"),
    Bench("impute_ld (400×400, w=30)", "preprocess",
          _build_impute_ld, _run_impute_ld, slug="impute_ld"),
    Bench("HWE (1500×5000 diploid)", "preprocess",
          _build_hwe, _run_hwe, slug="hwe_diploid"),
    Bench("HWE-DR (1000×1500 tetraploid)", "preprocess",
          _build_hwe_dr, _run_hwe_dr, slug="hwe_tetraploid"),

    Bench("SPA (1500 samples × 1000 SNPs)", "stats",
          _build_spa, _run_spa, slug="spa"),

    Bench("LDSC h² jackknife (m=8000, blocks=200)", "post-GWAS",
          _build_ldsc, _run_ldsc, slug="ldsc_jackknife"),

    Bench("GRM streaming vanraden (n=600, m=8000)", "linalg",
          _build_grm_streaming, _run_grm_streaming, slug="grm_streaming"),
    Bench("GRM full-batch vanraden (n=600, m=4000)", "linalg",
          _build_grm_vanraden, _run_grm_vanraden, slug="grm_vanraden"),
]


# ---------------------------------------------------------------------
# CI subset — kernels selected for the ``perf.yml`` GitHub workflow.
#
# Goal: cover the full speedup distribution (huge / strong / modest /
# launch-bound) across every group while keeping the wall-time budget
# under ~3 minutes per checkout (master + PR ⇒ ~6 min total). The list
# below is the recommended default; CI passes ``--kernel-subset
# <comma-separated slugs>`` so it can be overridden without touching
# this file.
#
# Coverage (per Efficiency E5 spec):
#   - 3 LD kernels (Gabriel, PELT, CC-graph)
#   - 2 imputation kernels (KNN, mode)
#   - 2 PGS kernels (LDpred2 Gibbs, PRS-CS Gibbs)
#   - 1 HWE kernel (diploid, the one most users hit)
#   - 1 SPA kernel
#   - 1 stats / post-GWAS kernel (LDSC h² jackknife)
#   - 2 linalg / streaming GRM kernels (the post-streaming hot path)
#
# Total: 12 kernels.
# ---------------------------------------------------------------------

CI_SUBSET_SLUGS: list[str] = [
    # LD (3)
    "gabriel_blocks",
    "pelt",
    "cc_graph_blocks",
    # imputation (2)
    "impute_knn",
    "impute_mode",
    # PGS (2)
    "ldpred2_gibbs",
    "prscs_gibbs",
    # preprocess HWE (1)
    "hwe_diploid",
    # stats / SPA (1)
    "spa",
    # post-GWAS (1)
    "ldsc_jackknife",
    # linalg / streaming GRM (2)
    "grm_streaming",
    "grm_vanraden",
]


# =====================================================================
# Main
# =====================================================================

def _format_seconds(s: float) -> str:
    if s >= 1.0:
        return f"{s:6.2f} s "
    if s >= 1e-3:
        return f"{s * 1e3:6.2f} ms"
    return f"{s * 1e6:6.2f} µs"


def _select_benches(slug_filter: list[str] | None) -> list[Bench]:
    """Return the registry, optionally filtered to ``slug_filter``.

    Raises ``ValueError`` on unknown slugs so a typo in the workflow
    fails fast rather than silently running zero kernels.
    """
    if not slug_filter:
        return list(BENCHES)
    known = {b.slug: b for b in BENCHES if b.slug}
    missing = [s for s in slug_filter if s not in known]
    if missing:
        valid = ", ".join(sorted(known))
        raise ValueError(
            f"Unknown kernel slug(s): {missing}. Known: {valid}",
        )
    # Preserve user-requested order.
    return [known[s] for s in slug_filter]


def _row_record(b: Bench, py: BenchResult, nat: BenchResult) -> dict[str, Any]:
    """Single-kernel JSON record (the schema the CI diff script consumes).

    Two records are emitted per kernel — one with ``mode == "python"``,
    one with ``mode == "native"`` — so a downstream consumer can filter
    on either path. ``speedup_vs_python`` is repeated on both rows for
    convenience but only the native row's value is meaningful (the
    python row's value is always 1.0).
    """
    if nat.p50_seconds > 0:
        speedup_p50 = py.p50_seconds / nat.p50_seconds
    else:
        speedup_p50 = float("inf")
    return {
        "kernel": b.name,
        "slug": b.slug,
        "group": b.group,
        "size_label": b.name.split("(")[-1].rstrip(")") if "(" in b.name else "",
        "n_repeats": b.repeats,
        "speedup_vs_python_p50": speedup_p50,
        "python": {
            "wall_seconds_p50": py.p50_seconds,
            "wall_seconds_p95": py.p95_seconds,
            "wall_seconds_median": py.median_seconds,
        },
        "native": {
            "wall_seconds_p50": nat.p50_seconds,
            "wall_seconds_p95": nat.p95_seconds,
            "wall_seconds_median": nat.median_seconds,
        },
    }


def _flatten_for_diff(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a per-kernel record into one row per (kernel, mode).

    This is the format ``bench/diff_perf.py`` reads. Each row carries
    the documented schema:
        kernel, slug, group, size_label, mode, n_repeats,
        wall_seconds_p50, wall_seconds_p95, speedup_vs_python.
    """
    out: list[dict[str, Any]] = []
    for mode in ("python", "native"):
        sub = record[mode]
        out.append(
            {
                "kernel": record["kernel"],
                "slug": record["slug"],
                "group": record["group"],
                "size_label": record["size_label"],
                "mode": mode,
                "n_repeats": record["n_repeats"],
                "wall_seconds_p50": sub["wall_seconds_p50"],
                "wall_seconds_p95": sub["wall_seconds_p95"],
                "speedup_vs_python": (
                    1.0
                    if mode == "python"
                    else record["speedup_vs_python_p50"]
                ),
            }
        )
    return out


def _emit_markdown(rows: list[tuple], out_path: str) -> str:
    """Write the legacy markdown table (unchanged from pre-E5 format)."""
    title = (
        "# Native C++ acceleration benchmark (realistic sizes)"
        if REALISTIC
        else "# Native C++ acceleration benchmark"
    )
    lines = [
        title,
        "",
        f"Repeats per kernel: **{DEFAULT_REPEATS}** (median wall time). "
        f"Mode: **{'realistic' if REALISTIC else 'small'}**.",
        "",
        f"Generated by `bench/native_speedups.py`. Numbers vary across "
        f"machines but the *relative ordering* is what informs which "
        f"kernels deserve further optimization (e.g. OpenMP).",
        "",
        "| Group | Kernel | Python | Native | Speedup | Verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for group, name, py_t, nat_t, sp, verdict in rows:
        if py_t != py_t:  # NaN
            lines.append(f"| {group} | {name} | — | — | — | {verdict} |")
        else:
            lines.append(
                f"| {group} | {name} | {_format_seconds(py_t)} | "
                f"{_format_seconds(nat_t)} | {sp:6.1f}× | {verdict} |"
            )
    md = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    return md


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark TorchGWAS native C++ kernels vs pure-Python "
            "reference. Default: write a markdown table to "
            "bench/native_speedups[_realistic].md. Use --output json "
            "for the JSON schema consumed by bench/diff_perf.py."
        ),
    )
    p.add_argument(
        "--output",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format. 'markdown' is the legacy human-readable "
        "table; 'json' is the per-kernel record consumed by "
        "bench/diff_perf.py; 'both' writes both.",
    )
    p.add_argument(
        "--output-path",
        default=None,
        help="Output path. For --output markdown, defaults to "
        "bench/native_speedups[_realistic].md. For --output json, "
        "defaults to bench/native_speedups.json. Required for "
        "--output both (used as the JSON path; markdown uses default).",
    )
    p.add_argument(
        "--kernel-subset",
        default=None,
        help="Comma-separated list of kernel slugs to run. Defaults "
        "to the full registry. Pass 'ci' to use the curated CI subset "
        "(see CI_SUBSET_SLUGS in the source).",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Override per-kernel n_repeats. Defaults to 5 (or 2 if "
        "TORCHGWAS_BENCH_QUICK=1).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.kernel_subset:
        if args.kernel_subset == "ci":
            slugs = list(CI_SUBSET_SLUGS)
        else:
            slugs = [s.strip() for s in args.kernel_subset.split(",") if s.strip()]
        benches = _select_benches(slugs)
    else:
        benches = _select_benches(None)

    if args.n_repeats is not None:
        for b in benches:
            b.repeats = args.n_repeats

    print(f"# TorchGWAS native acceleration benchmark")
    print(
        f"# repeats={benches[0].repeats if benches else DEFAULT_REPEATS}, "
        f"quick={QUICK}, realistic={REALISTIC}, "
        f"n_kernels={len(benches)}, output={args.output}"
    )
    print()

    rows: list[tuple[str, str, float, float, float, str]] = []
    json_records: list[dict[str, Any]] = []

    for b in benches:
        print(f"  running: {b.name} ...", flush=True)
        try:
            py_res, nat_res = run_bench(b)
        except Exception as e:  # noqa: BLE001
            print(f"    ! failed: {e}")
            rows.append(
                (b.group, b.name, float("nan"), float("nan"),
                 float("nan"), f"FAILED: {e}")
            )
            continue
        py_t = py_res.median_seconds
        nat_t = nat_res.median_seconds
        speedup = py_t / nat_t if nat_t > 0 else float("inf")
        verdict = (
            "launch-bound" if speedup < 1.5
            else ("modest" if speedup < 5 else
                  ("strong" if speedup < 50 else "huge"))
        )
        rows.append((b.group, b.name, py_t, nat_t, speedup, verdict))

        record = _row_record(b, py_res, nat_res)
        json_records.extend(_flatten_for_diff(record))

    # Sort markdown rows by speedup descending (failures last).
    rows.sort(key=lambda r: (-r[4] if r[4] == r[4] else float("inf")))

    here = os.path.dirname(os.path.abspath(__file__))

    if args.output in ("markdown", "both"):
        md_path = args.output_path if args.output == "markdown" else None
        if md_path is None:
            md_name = (
                "native_speedups_realistic.md" if REALISTIC
                else "native_speedups.md"
            )
            md_path = os.path.join(here, md_name)
        md = _emit_markdown(rows, md_path)
        print()
        print(md)
        print(f"wrote {md_path}")

    if args.output in ("json", "both"):
        json_path = args.output_path
        if json_path is None or args.output == "both":
            # For --output both, --output-path is the JSON path.
            json_path = (
                args.output_path
                if args.output == "both" and args.output_path
                else os.path.join(here, "native_speedups.json")
            )
        payload = {
            "schema_version": 1,
            "realistic": REALISTIC,
            "quick": QUICK,
            "n_kernels": len(benches),
            "rows": json_records,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
