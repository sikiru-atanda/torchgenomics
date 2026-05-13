"""Diff two benchmark JSON files and emit a perf regression verdict.

This is the gate for the Efficiency E5 perf regression CI. Reads
``master.json`` (baseline) and ``pr.json`` (PR head), computes the
percentage delta in p50 / p95 wall time per kernel × mode, and:

- Writes a markdown table (suitable for PR-comment posting).
- Returns an exit code:
    0 — all kernels OK (within ``warn_pct`` for both p50 and p95).
    1 — at least one kernel regressed beyond ``regression_pct``
        on p50 (the gate).
    2 — only ``warn_pct < x ≤ regression_pct`` warnings on p50.

The default thresholds match the Efficiency E5 spec: warn at +5 %
slower, fail at +10 % slower. Both apply to the **native** mode of
each kernel (the python column is informational only and not gated;
its wall time is measured but its purpose in the JSON is to compute
``speedup_vs_python``, which is what shows up in the markdown table).

If a kernel exists only in master (deleted in the PR) or only in the
PR (added by the PR), it is reported in the table but does NOT
contribute to the exit code. This keeps PRs that intentionally retire
or add a kernel from being mis-flagged.

Usage::

    python bench/diff_perf.py master.json pr.json \\
        --markdown /tmp/perf_diff.md \\
        [--regression-pct 10.0] [--warn-pct 5.0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Verdict thresholds (matches docs/efficiency/perf_regression.md).
# ---------------------------------------------------------------------

DEFAULT_REGRESSION_PCT = 10.0  # > 10 % slower => exit 1 (hard fail)
DEFAULT_WARN_PCT = 5.0  # > 5 % but <= 10 % => exit 2 (warning only)


# ---------------------------------------------------------------------
# Schema-aware row indexing
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RowKey:
    """Composite key matching benchmark rows across the two JSON files.

    We key on (slug, mode) rather than the human-readable kernel name
    so a cosmetic rename of a kernel (e.g. updating the size label in
    the docstring) doesn't artificially break the diff.
    """

    slug: str
    mode: str  # "python" | "native"


def _index_rows(payload: dict[str, Any]) -> dict[RowKey, dict[str, Any]]:
    """Map ``RowKey`` → row dict. Tolerant to old-shape payloads."""
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out: dict[RowKey, dict[str, Any]] = {}
    for r in rows:
        slug = r.get("slug") or r.get("kernel", "")
        mode = r.get("mode", "native")
        out[RowKey(slug=slug, mode=mode)] = r
    return out


# ---------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------


@dataclass
class KernelDiff:
    """Pair-wise diff for one (kernel, mode) row."""

    slug: str
    kernel: str
    group: str
    mode: str
    master_p50: float | None
    pr_p50: float | None
    master_p95: float | None
    pr_p95: float | None
    delta_p50_pct: float | None
    delta_p95_pct: float | None
    verdict: str  # "OK" | "WARNING" | "REGRESSION" | "NEW" | "REMOVED"


def _pct_delta(master: float, pr: float) -> float | None:
    """Return ``(pr - master) / master * 100``. ``None`` if master ~ 0.

    Sign convention: positive = PR is **slower** than master (bad).
    """
    if master is None or pr is None:
        return None
    if not math.isfinite(master) or not math.isfinite(pr):
        return None
    if master <= 0.0:
        return None
    return (pr - master) / master * 100.0


def _classify(
    delta_p50_pct: float | None,
    regression_pct: float,
    warn_pct: float,
    is_native: bool,
) -> str:
    """Map a delta to a verdict string.

    Only the native mode is gated. The python row's verdict is always
    "OK" — the python path is the algorithmic reference, and its wall
    time on the runner reflects baseline noise rather than something
    the PR is responsible for.
    """
    if delta_p50_pct is None:
        return "OK"
    if not is_native:
        return "OK"
    if delta_p50_pct > regression_pct:
        return "REGRESSION"
    if delta_p50_pct > warn_pct:
        return "WARNING"
    return "OK"


def diff_payloads(
    master: dict[str, Any],
    pr: dict[str, Any],
    *,
    regression_pct: float = DEFAULT_REGRESSION_PCT,
    warn_pct: float = DEFAULT_WARN_PCT,
) -> list[KernelDiff]:
    """Compute per-row diffs. Stable order: by group, then by slug, native first."""
    m_idx = _index_rows(master)
    p_idx = _index_rows(pr)
    all_keys = sorted(
        set(m_idx) | set(p_idx),
        key=lambda k: (
            (m_idx.get(k) or p_idx.get(k) or {}).get("group", ""),
            k.slug,
            0 if k.mode == "native" else 1,  # native row first
        ),
    )

    diffs: list[KernelDiff] = []
    for k in all_keys:
        m_row = m_idx.get(k)
        p_row = p_idx.get(k)
        kernel = (p_row or m_row or {}).get("kernel", k.slug)
        group = (p_row or m_row or {}).get("group", "")

        if m_row is None and p_row is not None:
            diffs.append(
                KernelDiff(
                    slug=k.slug, kernel=kernel, group=group, mode=k.mode,
                    master_p50=None, pr_p50=p_row.get("wall_seconds_p50"),
                    master_p95=None, pr_p95=p_row.get("wall_seconds_p95"),
                    delta_p50_pct=None, delta_p95_pct=None,
                    verdict="NEW",
                )
            )
            continue
        if p_row is None and m_row is not None:
            diffs.append(
                KernelDiff(
                    slug=k.slug, kernel=kernel, group=group, mode=k.mode,
                    master_p50=m_row.get("wall_seconds_p50"), pr_p50=None,
                    master_p95=m_row.get("wall_seconds_p95"), pr_p95=None,
                    delta_p50_pct=None, delta_p95_pct=None,
                    verdict="REMOVED",
                )
            )
            continue
        # Both rows exist.
        assert m_row is not None and p_row is not None
        m_p50 = m_row.get("wall_seconds_p50")
        p_p50 = p_row.get("wall_seconds_p50")
        m_p95 = m_row.get("wall_seconds_p95")
        p_p95 = p_row.get("wall_seconds_p95")
        d50 = _pct_delta(m_p50, p_p50)
        d95 = _pct_delta(m_p95, p_p95)
        verdict = _classify(
            d50, regression_pct=regression_pct, warn_pct=warn_pct,
            is_native=(k.mode == "native"),
        )
        diffs.append(
            KernelDiff(
                slug=k.slug, kernel=kernel, group=group, mode=k.mode,
                master_p50=m_p50, pr_p50=p_p50,
                master_p95=m_p95, pr_p95=p_p95,
                delta_p50_pct=d50, delta_p95_pct=d95,
                verdict=verdict,
            )
        )
    return diffs


# ---------------------------------------------------------------------
# Markdown rendering (PR comment payload)
# ---------------------------------------------------------------------


def _fmt_seconds(s: float | None) -> str:
    if s is None or not math.isfinite(s):
        return "—"
    if s >= 1.0:
        return f"{s:6.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:6.2f} ms"
    return f"{s * 1e6:6.2f} µs"


def _fmt_pct(p: float | None) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def render_markdown(
    diffs: list[KernelDiff],
    *,
    regression_pct: float = DEFAULT_REGRESSION_PCT,
    warn_pct: float = DEFAULT_WARN_PCT,
) -> str:
    """Markdown body suitable for posting as a PR comment."""
    n_reg = sum(1 for d in diffs if d.verdict == "REGRESSION")
    n_warn = sum(1 for d in diffs if d.verdict == "WARNING")
    n_new = sum(1 for d in diffs if d.verdict == "NEW")
    n_removed = sum(1 for d in diffs if d.verdict == "REMOVED")

    if n_reg > 0:
        headline = (
            f"### Performance regression CI: REGRESSION ({n_reg} kernel(s) "
            f"slower than {regression_pct:.0f}%)"
        )
    elif n_warn > 0:
        headline = (
            f"### Performance regression CI: WARNING ({n_warn} kernel(s) "
            f"slower than {warn_pct:.0f}%)"
        )
    else:
        headline = "### Performance regression CI: OK"

    lines: list[str] = [
        headline,
        "",
        f"Threshold: regression > {regression_pct:.0f}% slower (p50, "
        f"native), warn > {warn_pct:.0f}%. ``REGRESSION`` rows fail "
        f"the workflow (exit 1); ``WARNING`` rows post a comment but "
        f"don't block (exit 2). ``NEW``/``REMOVED`` rows are reported "
        f"but ungated.",
        "",
        f"Kernels compared: **{len(diffs)}**  ·  "
        f"REGRESSION: **{n_reg}**  ·  WARNING: **{n_warn}**  ·  "
        f"NEW: **{n_new}**  ·  REMOVED: **{n_removed}**",
        "",
        "| Kernel | Mode | master p50 | PR p50 | %Δ p50 | %Δ p95 | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for d in diffs:
        lines.append(
            f"| {d.kernel} | {d.mode} | "
            f"{_fmt_seconds(d.master_p50)} | "
            f"{_fmt_seconds(d.pr_p50)} | "
            f"{_fmt_pct(d.delta_p50_pct)} | "
            f"{_fmt_pct(d.delta_p95_pct)} | "
            f"{d.verdict} |"
        )
    lines.append("")
    lines.append(
        f"<sub>Generated by `bench/diff_perf.py`. See "
        f"`docs/efficiency/perf_regression.md` for how to interpret "
        f"and update the gate thresholds.</sub>"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Exit-code policy
# ---------------------------------------------------------------------


def exit_code_for(diffs: list[KernelDiff]) -> int:
    """0 if clean, 1 if any REGRESSION, 2 if only WARNINGs."""
    if any(d.verdict == "REGRESSION" for d in diffs):
        return 1
    if any(d.verdict == "WARNING" for d in diffs):
        return 2
    return 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diff two benchmark JSON files (master baseline vs PR head) "
            "and emit a perf regression verdict + markdown PR comment."
        ),
    )
    p.add_argument("master", type=Path, help="Master baseline JSON.")
    p.add_argument("pr", type=Path, help="PR head JSON.")
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Write the markdown PR-comment body to this path.",
    )
    p.add_argument(
        "--regression-pct",
        type=float,
        default=DEFAULT_REGRESSION_PCT,
        help=f"Hard-fail threshold (%% slower). Default {DEFAULT_REGRESSION_PCT}.",
    )
    p.add_argument(
        "--warn-pct",
        type=float,
        default=DEFAULT_WARN_PCT,
        help=f"Soft-warn threshold (%% slower). Default {DEFAULT_WARN_PCT}.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress markdown to stdout (still writes --markdown if set).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.master.exists():
        print(
            f"ERROR: master JSON not found at {args.master} — "
            f"first PR after enabling the workflow always exits 0; "
            f"subsequent PRs will diff against the master artifact.",
            file=sys.stderr,
        )
        # Soft-fall: no baseline, no diff possible. Emit empty markdown.
        if args.markdown is not None:
            args.markdown.write_text(
                "### Performance regression CI: SKIPPED\n\n"
                "No master baseline JSON was available — first run, or "
                "the master artifact has expired. This PR's run will "
                "become the baseline for the next push to master.\n",
                encoding="utf-8",
            )
        return 0

    if not args.pr.exists():
        print(f"ERROR: PR JSON not found at {args.pr}", file=sys.stderr)
        return 1

    master = json.loads(args.master.read_text(encoding="utf-8"))
    pr = json.loads(args.pr.read_text(encoding="utf-8"))

    diffs = diff_payloads(
        master, pr,
        regression_pct=args.regression_pct,
        warn_pct=args.warn_pct,
    )
    md = render_markdown(
        diffs,
        regression_pct=args.regression_pct,
        warn_pct=args.warn_pct,
    )

    if args.markdown is not None:
        args.markdown.write_text(md, encoding="utf-8")

    if not args.quiet:
        print(md)

    return exit_code_for(diffs)


if __name__ == "__main__":
    raise SystemExit(main())
