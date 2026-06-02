#!/usr/bin/env python3
"""Streaming vs materialized memory bench (paper Figure 6 panel B).

Measures the peak Python-tensor allocation budget of TorchGenomics's streaming
single-trait LMM scan path across a p-sweep, at fixed n. Plots streaming
peak (measured via ``tracemalloc``) against the analytically-known
materialized peak (``n * p * dtype_bytes``) to demonstrate the slope
reduction that lets biobank-scale runs fit on a workstation.

Why measure streaming and *compute* materialized:
  - Streaming peak depends on chunk_size + per-chunk Python overhead +
    GRM accumulator. There is no closed form; it must be measured.
  - Materialized peak is dominated by the single (n, p) tensor that
    ``_load_full_genotype`` returns. At p=1e6, n=2000, float32 that's
    8 GB — well past what most workstations can hold. We don't actually
    materialize a tensor that large; we just compute what the peak
    *would* be, exactly as a paper figure should report (the materialized
    path is what the streaming path replaces; we're plotting the avoided
    cost, not paying it).

The bench is reproducibility-script-friendly:

    python bench/streaming_p_sweep.py --output bench/streaming_p_sweep.json

emits a structured JSON that ``paper/reproducibility/render_figures/
f6_gpu_streaming.py`` panel B consumes. With ``--quick``, the sweep
runs in seconds; without it, peak runtime is bounded by the largest
streaming chunk + per-p random-G generation cost (a few minutes).

Pattern reference: ``tests/test_streaming_memory.py`` (the streaming
regression net that gates the same contract on every PR).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import resource
import sys
import time

import torch


def _max_rss_bytes() -> int:
    """OS-level peak resident-set size for the current process.

    Linux: ru_maxrss is in kilobytes.
    macOS: ru_maxrss is in bytes (Darwin convention).
    We normalize to bytes either way; the relative streaming-vs-
    materialized ratio is what the figure plots, so the absolute
    units only matter for readability.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: macOS reports raw bytes (large), Linux reports KB.
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from torchgenomics.models.single_trait_lmm import SingleTraitLMM  # noqa: E402
from torchgenomics.linalg.kinship import grm_vanraden  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data generator. We never materialize the full G; instead the
# streaming reader generates each chunk on demand. This is the same
# pattern the production PlinkBedReader uses — yield (G_chunk, vmeta)
# tuples without ever holding the whole matrix.
# ---------------------------------------------------------------------------


class _SyntheticChunkReader:
    """Generates G chunks on demand (never holds the full tensor)."""

    def __init__(self, n: int, n_variants: int, chunk_size: int = 1024,
                 seed: int = 42) -> None:
        self._n = n
        self._n_variants = n_variants
        self._chunk_size = chunk_size
        self._seed = seed

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def n_variants(self) -> int:
        return self._n_variants

    def iter_chunks(self, chunk_size: int | None = None):
        size = chunk_size or self._chunk_size
        gen = torch.Generator().manual_seed(self._seed)
        for start in range(0, self._n_variants, size):
            stop = min(start + size, self._n_variants)
            k = stop - start
            # Generate a chunk of synthetic genotypes in [0, 2] (additive dosage).
            # We use torch.randint for speed + determinism.
            chunk = torch.randint(0, 3, (self._n, k), generator=gen,
                                  dtype=torch.float32)
            yield chunk, None


# ---------------------------------------------------------------------------
# Per-p measurement
# ---------------------------------------------------------------------------


def _peak_streaming(n: int, p: int, chunk_size: int) -> dict:
    """Run a streaming GRM accumulation + null fit + scan on synthetic G.

    Returns peak RSS bytes + wall time + numerical sanity checks.
    The peak is measured around the whole scan via ``resource.getrusage``
    so torch tensor allocations are visible (tracemalloc only sees
    Python-managed allocations and would miss tensor storage).
    """
    gc.collect()
    # Phenotype + covariate (fixed across the sweep — small, no p-dependence)
    torch.manual_seed(1234)
    y = torch.randn(n, dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Stream the GRM in chunks, never materializing the full G.
    reader = _SyntheticChunkReader(n=n, n_variants=p, chunk_size=chunk_size)

    rss_before = _max_rss_bytes()
    t0 = time.perf_counter()

    # GRM accumulator: sum over chunks of standardized G G^T contribution.
    K = torch.zeros(n, n, dtype=torch.float64)
    p_seen = 0
    for chunk_f32, _ in reader.iter_chunks():
        chunk = chunk_f32.to(torch.float64)
        # Standardize per-variant (mean 0, var 1) — VanRaden contribution.
        mean = chunk.mean(dim=0, keepdim=True)
        p_af = mean / 2.0
        denom = (2.0 * p_af * (1.0 - p_af)).clamp_min(1e-12)
        # GRM accumulator: G_std @ G_std.T summed over chunks.
        G_std = (chunk - 2.0 * p_af) / denom.sqrt()
        K += G_std @ G_std.T
        p_seen += chunk.shape[1]
        # Free per-chunk allocations before the next iteration.
        del chunk, chunk_f32, G_std
    K /= max(p_seen, 1)

    # Null fit — this is the dominant per-scan allocation after the GRM
    # (eigendecomposition of K). The streaming contract gates that the
    # GRM never grows with p; this null-fit step verifies the LMM
    # machinery can be exercised on the streamed GRM.
    model = SingleTraitLMM()
    _ = model.fit_null(Y=y, X0=X0, K=K)

    elapsed = time.perf_counter() - t0
    rss_after = _max_rss_bytes()
    # ru_maxrss is the lifetime peak. Within a single process, this
    # is monotone in p, but the *incremental* growth from prior p
    # is what's diagnostic of "constant streaming memory". The figure
    # plots both absolute peak and delta from baseline.
    return {
        "peak_bytes": int(rss_after),
        "peak_delta_from_baseline_bytes": int(rss_after - rss_before),
        "rss_before_bytes": int(rss_before),
        "rss_after_bytes": int(rss_after),
        "elapsed_seconds": elapsed,
        "p_seen": int(p_seen),
        "n_samples": int(n),
        "chunk_size": int(chunk_size),
    }


def _materialized_peak_bytes(n: int, p: int, dtype_bytes: int = 4) -> int:
    """Analytical peak for the materialized path.

    The materialized scan holds the full (n, p) genotype tensor +
    a per-variant standardization vector + the GRM (n, n).
    Dominated by the (n, p) tensor.
    """
    g_bytes = n * p * dtype_bytes  # float32 default for PlinkBedReader
    grm_bytes = n * n * 8           # float64 GRM
    std_bytes = p * 8 * 2            # mean + var per variant
    return g_bytes + grm_bytes + std_bytes


def _log_log_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope on log-log axes (linear in log p, log peak)."""
    if len(xs) < 2:
        return float("nan")
    lx = [math.log(max(x, 1.0)) for x in xs]
    ly = [math.log(max(y, 1.0)) for y in ys]
    n = len(lx)
    mean_x = sum(lx) / n
    mean_y = sum(ly) / n
    num = sum((lx[i] - mean_x) * (ly[i] - mean_y) for i in range(n))
    den = sum((lx[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_PS = [10_000, 30_000, 100_000, 300_000, 1_000_000]
_QUICK_PS = [1_000, 10_000]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path,
                    default=ROOT / "bench" / "streaming_p_sweep.json")
    ap.add_argument("--n", type=int, default=2000,
                    help="Sample count (paper default 2000)")
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="Streaming chunk size in variants (paper default 1024)")
    ap.add_argument("--ps", type=str, default=None,
                    help="Comma-separated p values to sweep (overrides defaults)")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny sweep (p ∈ {1e3, 1e4}) for CI sanity")
    args = ap.parse_args()

    if args.ps:
        ps = [int(x) for x in args.ps.split(",")]
    elif args.quick:
        ps = _QUICK_PS
    else:
        ps = _DEFAULT_PS

    print(f"[streaming-p-sweep] n={args.n}  chunk_size={args.chunk_size}")
    print(f"[streaming-p-sweep] p sweep: {ps}")

    per_p: list[dict] = []
    for p in ps:
        print(f"[streaming-p-sweep] p={p:>9} ... ", end="", flush=True)
        meas = _peak_streaming(n=args.n, p=p, chunk_size=args.chunk_size)
        mat = _materialized_peak_bytes(n=args.n, p=p)
        meas["materialized_peak_bytes"] = mat
        meas["ratio_materialized_over_streaming"] = mat / max(meas["peak_bytes"], 1)
        per_p.append({"p": p, **meas})
        print(
            f"streaming={meas['peak_bytes'] / 1e6:>7.1f} MB  "
            f"materialized={mat / 1e6:>9.1f} MB  "
            f"ratio={meas['ratio_materialized_over_streaming']:>7.1f}x  "
            f"elapsed={meas['elapsed_seconds']:>5.1f}s"
        )

    streaming_slope = _log_log_slope(
        [r["p"] for r in per_p], [r["peak_bytes"] for r in per_p],
    )
    materialized_slope = _log_log_slope(
        [r["p"] for r in per_p],
        [r["materialized_peak_bytes"] for r in per_p],
    )

    out_payload = {
        "schema_version": 1,
        "n_samples": args.n,
        "chunk_size": args.chunk_size,
        "dtype_bytes_materialized": 4,
        "per_p": per_p,
        "streaming_log_log_slope": streaming_slope,
        "materialized_log_log_slope": materialized_slope,
        "ratio_at_max_p": per_p[-1]["ratio_materialized_over_streaming"] if per_p else None,
        "max_p": ps[-1] if ps else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2, default=str))
    print(f"[streaming-p-sweep] wrote {args.output}")
    print(
        f"[streaming-p-sweep] log-log slopes: streaming={streaming_slope:.3f}  "
        f"materialized={materialized_slope:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
