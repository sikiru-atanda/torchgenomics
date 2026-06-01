"""F6 -- GPU acceleration and biobank-scale streaming.

Four-panel figure summarizing the efficiency story:

  A. Native C++ kernel speedup distribution from
     ``bench/native_speedups_realistic.md`` -- violin + jittered scatter
     of log10(speedup), partitioned by verdict (huge / strong / modest
     / launch-bound).
  B. Streaming-vs-materialized memory slope vs. variant count p.
     Materialized: ``n * p * 8 B``. Streaming: bounded by chunk_size.
     ~19x ratio at p=1e6 per ``docs/efficiency/streaming_audit.md``.
     SCAFFOLD until ``bench/streaming_p_sweep.py`` lands in Tier 4 D2.8.
  C. Sparse-GRM PCG-REML 1.8x speedup vs. dense eigendecomposition at
     n=500K. SCAFFOLD -- requires NA3 UKB-harness GPU-node run.
  D. CPU<->GPU parity scatter at FP64 noise floor. SCAFFOLD -- GPU CI
     artifacts (``.github/workflows/gpu.yml``) are not committed; we
     report the gated GPU test count instead.

Tier 4 D2.6. Mirrors ``f2_equivalence_grid.py``.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

NATIVE_SPEEDUPS_MD = REPO_ROOT / "bench" / "native_speedups_realistic.md"
STREAMING_AUDIT_MD = REPO_ROOT / "docs" / "efficiency" / "streaming_audit.md"
GPU_TEST_FILES = [
    REPO_ROOT / "tests" / "test_gpu.py",
    REPO_ROOT / "tests" / "test_gpu_model_parity.py",
    REPO_ROOT / "tests" / "test_impute_gpu.py",
    REPO_ROOT / "tests" / "test_impute_gpu_kernels.py",
]

SPEC_SLOPE_RATIO = 19.0
SPEC_PCG_SPEEDUP = 1.8
SLOPE_N = 500_000

VERDICT_COLORS = dict(huge="#1f77b4", strong="#2ca02c", modest="#ff7f0e")
VERDICT_COLORS["launch-bound"] = "#7f7f7f"


_SPEEDUP_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|.*\|\s*([0-9.]+)\s*[x\xd7]\s*\|\s*([a-z\-]+)\s*\|\s*$",
    re.IGNORECASE,
)


def _parse_native_speedups(path):
    if not path.exists():
        return []
    rows = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        if not line.startswith("|") or line.startswith("|---") or "Group" in line:
            continue
        m = _SPEEDUP_RE.match(line)
        if not m:
            continue
        group, kernel, speedup, verdict = m.groups()
        try:
            ratio = float(speedup)
        except ValueError:
            continue
        rows.append(dict(
            group=group.strip(),
            kernel=kernel.strip(),
            speedup=ratio,
            verdict=verdict.strip(),
        ))
    return rows


def _plot_panel_a(ax):
    rows = _parse_native_speedups(NATIVE_SPEEDUPS_MD)
    if not rows:
        ax.text(0.5, 0.5, "Panel A scaffold -- no native_speedups data",
                ha="center", va="center", transform=ax.transAxes, color="#888888")
        ax.set_title("A . native kernel speedups (scaffold)", fontsize=10)
        return dict(n_kernels=0, speedup_range=[None, None], scaffold=True)

    speedups = np.array([r["speedup"] for r in rows], dtype=float)
    log_speedups = np.log10(speedups)
    verdicts = [r["verdict"] for r in rows]

    parts = ax.violinplot([log_speedups], positions=[0], widths=0.8,
                          showmeans=False, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor("#cccccc")
        body.set_alpha(0.5)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("#555555")

    rng = np.random.default_rng(seed=0)
    jitter = rng.uniform(-0.18, 0.18, size=len(rows))
    for i, (v, lv) in enumerate(zip(verdicts, log_speedups)):
        ax.scatter(jitter[i], lv, color=VERDICT_COLORS.get(v, "#000000"),
                   s=28, alpha=0.85, edgecolor="white", linewidth=0.5,
                   zorder=3)

    ymin = float(np.floor(log_speedups.min()))
    ymax = float(np.ceil(log_speedups.max()))
    ax.set_yticks(np.arange(ymin, ymax + 1))
    ax.set_yticklabels([f"{10**y:g}x" for y in np.arange(ymin, ymax + 1)])
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("native-vs-Python speedup")
    ax.set_title(f"A . {len(rows)}-kernel native speedup distribution", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=7, label=v)
        for v, c in VERDICT_COLORS.items()
        if v in set(verdicts)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.9)

    return dict(
        n_kernels=len(rows),
        speedup_range=[float(speedups.min()), float(speedups.max())],
        speedup_median=float(np.median(speedups)),
        scaffold=False,
        source=str(NATIVE_SPEEDUPS_MD.relative_to(REPO_ROOT)),
    )


def _plot_panel_b(ax):
    """Panel B: streaming vs materialized peak memory across a p-sweep.

    Reads ``bench/streaming_p_sweep.json`` when present (Tier 4 D2.8 output)
    and renders the measured curve. Falls back to the analytical scaffold
    if the bench has not run yet.
    """
    GB = 1024 ** 3
    bench_json = REPO_ROOT / "bench" / "streaming_p_sweep.json"

    if bench_json.exists():
        payload = json.loads(bench_json.read_text())
        per_p = payload.get("per_p", [])
        if per_p:
            p_values = np.array([row["p"] for row in per_p])
            mem_streaming = np.array([row["peak_bytes"] for row in per_p])
            mem_materialized = np.array(
                [row["materialized_peak_bytes"] for row in per_p],
            )
            ratio_at_max = float(payload.get("ratio_at_max_p", float("nan")))
            n = int(payload.get("n_samples", SLOPE_N))
            chunk_size = int(payload.get("chunk_size", 1024))
            streaming_slope = float(payload.get("streaming_log_log_slope", float("nan")))
            materialized_slope = float(payload.get("materialized_log_log_slope", float("nan")))
            measured = True
        else:
            measured = False
    else:
        measured = False

    if not measured:
        p_values = np.array([1e4, 3e4, 1e5, 3e5, 1e6])
        n = SLOPE_N
        chunk_size = 1024
        bytes_per_elem = 8
        mem_materialized = n * p_values * bytes_per_elem
        accumulator_per_variant = 80
        mem_streaming = n * chunk_size * bytes_per_elem + p_values * accumulator_per_variant
        ratio_at_max = float(mem_materialized[-1] / mem_streaming[-1])
        streaming_slope = float("nan")
        materialized_slope = float("nan")

    ax.loglog(p_values, mem_materialized / GB, marker="s", color="#d62728",
              label="materialized (n * p * 4B)", linewidth=2)
    ax.loglog(p_values, mem_streaming / GB, marker="o", color="#1f77b4",
              label="streaming (chunked)", linewidth=2)

    ax.annotate(
        f"{ratio_at_max:.1f}x at p={int(p_values[-1]):,}",
        xy=(p_values[-1], mem_materialized[-1] / GB),
        xytext=(p_values[-1] * 0.25, mem_materialized[-1] / GB * 1.5),
        fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
    )

    label = "B . streaming-vs-materialized memory"
    label += f" . n={n:,}"
    if measured:
        label += " (measured)"
    else:
        label += " (scaffold)"
    ax.set_xlabel("variants (p)")
    ax.set_ylabel("peak memory (GB)")
    ax.set_title(label, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    return dict(
        slope_ratio=float(ratio_at_max),
        spec_headline_ratio=SPEC_SLOPE_RATIO,
        scaffold=not measured,
        scaffold_reason=(
            None if measured
            else "definitive measurement requires bench/streaming_p_sweep.py output"
        ),
        p_sweep=[float(p) for p in p_values],
        peak_streaming_bytes=[float(x) for x in mem_streaming.tolist()],
        peak_materialized_bytes=[float(x) for x in mem_materialized.tolist()],
        streaming_log_log_slope=streaming_slope,
        materialized_log_log_slope=materialized_slope,
        n_samples=n,
        chunk_size=chunk_size,
        source=str(bench_json.relative_to(REPO_ROOT)) if measured else str(STREAMING_AUDIT_MD.relative_to(REPO_ROOT)),
    )


def _plot_panel_c(ax):
    methods = ["dense eig\n(GEMMA-style)", "sparse-GRM\nPCG-REML"]
    times_relative = [1.0, 1.0 / SPEC_PCG_SPEEDUP]
    colors = ["#d62728", "#2ca02c"]

    bars = ax.bar(methods, times_relative, color=colors, edgecolor="white",
                  linewidth=0.8)
    for bar, t in zip(bars, times_relative):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 0.02,
                f"{t:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.25)
    ax.set_ylabel("relative wall time (lower is faster)")
    ax.set_title(f"C . sparse-GRM PCG-REML {SPEC_PCG_SPEEDUP}x speedup . n=500K (scaffold)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax.annotate(
        f"{SPEC_PCG_SPEEDUP}x faster",
        xy=(1, times_relative[1]),
        xytext=(0.5, 0.65),
        ha="center", fontsize=9, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
    )

    return dict(
        speedup_x=SPEC_PCG_SPEEDUP,
        n_samples=500_000,
        scaffold=True,
        scaffold_reason="definitive measurement requires NA3 UKB harness GPU-node run; no committed artifacts yet",
    )


def _count_gpu_tests(files):
    total = 0
    pat = re.compile(r"^\s*(?:async\s+)?def\s+test_")
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if pat.match(line):
                total += 1
    return total


def _plot_panel_d(ax):
    n_gpu_tests = _count_gpu_tests(GPU_TEST_FILES)

    rng = np.random.default_rng(seed=42)
    n_points = max(n_gpu_tests, 1)
    cpu_log_p = rng.uniform(-12, -1, size=n_points)
    gpu_log_p = cpu_log_p + rng.normal(0, 1e-12, size=cpu_log_p.shape)

    ax.scatter(cpu_log_p, gpu_log_p, s=18, color="#1f77b4", alpha=0.7,
               edgecolor="white", linewidth=0.4)
    lo = float(min(cpu_log_p.min(), gpu_log_p.min()))
    hi = float(max(cpu_log_p.max(), gpu_log_p.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#555555", linewidth=1,
            label="y = x")
    ax.set_xlabel("CPU -log10(p)")
    ax.set_ylabel("CUDA -log10(p)")
    ax.set_title(
        f"D . CPU<->GPU parity . {n_gpu_tests} GPU tests gated (scaffold)",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    return dict(
        n_gpu_tests=n_gpu_tests,
        n_gpu_test_files=sum(1 for p in GPU_TEST_FILES if p.exists()),
        scaffold=True,
        scaffold_reason="GPU CI artifacts (.github/workflows/gpu.yml JUnit XML) are not committed to the repo; parity scatter shown at FP64 noise floor",
    )


@register("F6")
def render_f6(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_a = _plot_panel_a(axes[0, 0])
    panel_b = _plot_panel_b(axes[0, 1])
    panel_c = _plot_panel_c(axes[1, 0])
    panel_d = _plot_panel_d(axes[1, 1])

    fig.suptitle(
        "F6 -- GPU acceleration and biobank-scale streaming\n"
        "Genome Biology Methods Figure 6",
        fontsize=11,
    )

    panels_by_id = dict(A=panel_a, B=panel_b, C=panel_c, D=panel_d)
    scaffold_panels = [k for k, v in panels_by_id.items() if v.get("scaffold")]
    measured_panels = [k for k in "ABCD" if k not in scaffold_panels]
    footer = (
        "measured: " + (", ".join(measured_panels) or "(none)")
        + "    .    scaffold: " + (", ".join(scaffold_panels) or "(none)")
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="#555555")

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    out_path = output_dir / "F6.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = dict(panels=panels_by_id)
    return out_path, manifest
