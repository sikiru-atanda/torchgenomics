"""Benchmark every TorchGWAS native C++ accelerator against its pure-Python
reference and emit a markdown table.

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
4. Writes the table to ``bench/native_speedups.md`` and stdout.

The output is *informational*. It does not gate any test. Numbers vary
across machines but the *ordering* of speedups is what informs which
kernels are worth further optimization (e.g. OpenMP) and which are
launch-overhead-bound at typical sizes.

Run with::

    python bench/native_speedups.py

Set ``TORCHGWAS_BENCH_QUICK=1`` to use ``n_repeats=2`` for a fast
sanity check.

Set ``TORCHGWAS_BENCH_REALISTIC=1`` to scale each kernel's input 2–4×
beyond the small defaults (capped per-kernel so the pure-Python path
still finishes within a few minutes). This validates that the Phase 41
speedups do not regress at sizes closer to what real users hit, which
is how the Spine regression in Phase 41ad was ultimately diagnosed.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


# ---------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------

QUICK = bool(os.environ.get("TORCHGWAS_BENCH_QUICK"))
REALISTIC = bool(os.environ.get("TORCHGWAS_BENCH_REALISTIC"))
DEFAULT_REPEATS = 2 if QUICK else 5


def _pick(small, realistic):
    """Return ``realistic`` when TORCHGWAS_BENCH_REALISTIC=1, else ``small``.

    Used inside each ``_build_*`` helper so the realistic-mode scale
    factors are visible right next to the default sizes.
    """
    return realistic if REALISTIC else small


def _time_call(fn: Callable, args: tuple, n_repeats: int) -> float:
    """Return median wall time of ``fn(*args)`` over ``n_repeats`` runs.

    Always does one warmup call before measurement.
    """
    fn(*args)  # warmup
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


@dataclass
class Bench:
    name: str
    group: str
    builder: Callable[[], tuple]  # returns (args,) for runner
    runner: Callable[..., Any]
    repeats: int = DEFAULT_REPEATS


def run_bench(b: Bench) -> tuple[float, float]:
    """Time the builder's args under both paths.

    Returns (python_seconds, native_seconds). The builder is invoked
    twice — once per path — so the timings cannot share mutated input
    state across paths.
    """
    # Python path
    os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    args_py = b.builder()
    py_time = _time_call(b.runner, args_py, b.repeats)

    # Native path
    os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    args_nat = b.builder()
    nat_time = _time_call(b.runner, args_nat, b.repeats)

    return py_time, nat_time


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
    m = _pick(200, 400)
    n_iter = _pick(200, 300)
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
    m = _pick(200, 400)
    n_iter = _pick(200, 300)
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
    rng = np.random.default_rng(0)
    n = 5000
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
    from torchgwas.ld import compute_pairwise_ld
    rng = torch.Generator().manual_seed(0)
    n, m = 400, 300
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
    n, m = 300, _pick(500, 900)
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
    n, m = _pick(1500, 3000), _pick(800, 1500)
    G = rng.integers(0, 3, size=(n, m)).astype(np.float64)
    miss = rng.random(size=(n, m)) < 0.1
    G[miss] = np.nan
    return (torch.from_numpy(G),)


def _run_impute_mode(G):
    from torchgwas.preprocess.impute import impute_mode
    return impute_mode(G)


def _build_impute_knn():
    rng = np.random.default_rng(0)
    n, m = _pick(400, 700), _pick(200, 350)
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
    n, m = _pick(1500, 2500), _pick(5000, 10000)
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
    n, m = _pick(1500, 2500), _pick(1000, 2000)
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
    m = _pick(8000, 40000)
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


# =====================================================================
# Bench registry
# =====================================================================

BENCHES: list[Bench] = [
    Bench("PRS-CS Gibbs (m=200, iter=200)", "PGS",
          _build_prscs_block, _run_prscs_block),
    Bench("LDpred2 Gibbs (m=200, iter=200)", "PGS",
          _build_ldpred2_block, _run_ldpred2_block),
    Bench("C+T clumping (m=1500)", "PGS",
          _build_ct, _run_ct),
    Bench("ESS Geyer (4×8000×50)", "PGS",
          _build_ess, _run_ess),

    Bench("PELT (n=5000, gaussian)", "LD",
          _build_pelt, _run_pelt),
    Bench("connected_components (n=800)", "LD",
          _build_cc, _run_cc),
    Bench("Gabriel blocks (m=300)", "LD",
          _build_gabriel, _run_gabriel),
    Bench("Big-LD blocks (m=600)", "LD",
          _build_big_ld, _run_big_ld),
    Bench("DP-optimize blocks (m=250)", "LD",
          _build_dp_opt, _run_dp_opt),
    Bench("CC-graph blocks (m=500)", "LD",
          _build_cc_graph, _run_cc_graph),
    Bench("GWAS-aligned blocks (m=200)", "LD",
          _build_gwas_aligned, _run_gwas_aligned),
    Bench("Spine blocks (m=250)", "LD",
          _build_spine, _run_spine),
    Bench("ld_decay_signal (n_pairs=30000)", "LD",
          _build_ld_decay, _run_ld_decay),
    Bench("greedy_mwis (n=5000)", "LD",
          _build_mwis, _run_mwis),
    Bench("Wall-Pritchard (m=150, n_perm=200)", "LD",
          _build_wall_pritchard, _run_wall_pritchard),
    Bench("uncertainty blocks (m=200)", "LD",
          _build_uncertainty, _run_uncertainty),
    Bench("cross-pop stability (3 pops, m=80)", "LD",
          _build_cross_pop, _run_cross_pop),

    Bench("impute_mode (1500×800)", "preprocess",
          _build_impute_mode, _run_impute_mode),
    Bench("impute_knn (400×200, k=5)", "preprocess",
          _build_impute_knn, _run_impute_knn),
    Bench("impute_ld (400×400, w=30)", "preprocess",
          _build_impute_ld, _run_impute_ld),
    Bench("HWE (1500×5000 diploid)", "preprocess",
          _build_hwe, _run_hwe),
    Bench("HWE-DR (1000×1500 tetraploid)", "preprocess",
          _build_hwe_dr, _run_hwe_dr),

    Bench("SPA (1500 samples × 1000 SNPs)", "stats",
          _build_spa, _run_spa),

    Bench("LDSC h² jackknife (m=8000, blocks=200)", "post-GWAS",
          _build_ldsc, _run_ldsc),
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


def main() -> None:
    print(f"# TorchGWAS native acceleration benchmark")
    print(f"# repeats={DEFAULT_REPEATS}, quick={QUICK}, realistic={REALISTIC}")
    print()

    rows: list[tuple[str, str, float, float, float, str]] = []
    for b in BENCHES:
        print(f"  running: {b.name} ...", flush=True)
        try:
            py_t, nat_t = run_bench(b)
        except Exception as e:  # noqa: BLE001
            print(f"    ! failed: {e}")
            rows.append((b.group, b.name, float("nan"), float("nan"),
                         float("nan"), f"FAILED: {e}"))
            continue
        speedup = py_t / nat_t if nat_t > 0 else float("inf")
        verdict = (
            "launch-bound" if speedup < 1.5
            else ("modest" if speedup < 5 else
                  ("strong" if speedup < 50 else "huge"))
        )
        rows.append((b.group, b.name, py_t, nat_t, speedup, verdict))

    # Sort by speedup descending (failures last)
    rows.sort(key=lambda r: (-r[4] if r[4] == r[4] else float("inf")))

    # Build markdown table
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
    out_name = "native_speedups_realistic.md" if REALISTIC else "native_speedups.md"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print()
    print(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
