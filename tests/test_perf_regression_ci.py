"""Tests for the Efficiency E5 perf regression CI tooling.

Covers:

1. ``bench/native_speedups.py --output json`` produces the documented
   schema (slug, mode, n_repeats, wall_seconds_p50/p95,
   speedup_vs_python).
2. ``bench/diff_perf.py`` returns exit 1 on > 10% p50 regression,
   exit 2 on a 5–10% warning-only delta, and exit 0 otherwise.
3. The markdown PR-comment payload contains the documented verdict
   strings (``REGRESSION``, ``WARNING``).

These tests are fast (the JSON-schema test runs the cheapest kernel —
``greedy_mwis`` — at ``--n-repeats 1``; the diff tests synthesize
JSON files in tmp).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_RUNNER = REPO_ROOT / "bench" / "native_speedups.py"
DIFF_SCRIPT = REPO_ROOT / "bench" / "diff_perf.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(
    *,
    slug: str = "demo_kernel",
    kernel: str = "Demo kernel (m=100)",
    group: str = "demo",
    python_p50: float = 1.0,
    python_p95: float = 1.05,
    native_p50: float = 0.1,
    native_p95: float = 0.105,
) -> dict:
    """Build a minimal benchmark JSON payload with one kernel × two modes.

    Mirrors the schema emitted by ``bench/native_speedups.py --output
    json``. The diff script only reads ``slug`` / ``mode`` /
    ``wall_seconds_p50`` / ``wall_seconds_p95`` / ``kernel`` / ``group``
    so we omit the other top-level metadata for brevity.
    """
    return {
        "schema_version": 1,
        "n_kernels": 1,
        "rows": [
            {
                "kernel": kernel,
                "slug": slug,
                "group": group,
                "size_label": "m=100",
                "mode": "python",
                "n_repeats": 5,
                "wall_seconds_p50": python_p50,
                "wall_seconds_p95": python_p95,
                "speedup_vs_python": 1.0,
            },
            {
                "kernel": kernel,
                "slug": slug,
                "group": group,
                "size_label": "m=100",
                "mode": "native",
                "n_repeats": 5,
                "wall_seconds_p50": native_p50,
                "wall_seconds_p95": native_p95,
                "speedup_vs_python": python_p50 / native_p50,
            },
        ],
    }


def _run_diff(master_path: Path, pr_path: Path, md_path: Path) -> tuple[int, str]:
    """Invoke the diff script as a subprocess. Returns (exit_code, stdout)."""
    proc = subprocess.run(
        [
            sys.executable, str(DIFF_SCRIPT),
            str(master_path), str(pr_path),
            "--markdown", str(md_path),
            "--quiet",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return proc.returncode, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 1. JSON schema test
# ---------------------------------------------------------------------------


def test_native_speedups_json_schema_valid(tmp_path: Path) -> None:
    """Smoke-test the runner's JSON output on the cheapest kernel.

    ``greedy_mwis`` at n=5000 with n_repeats=1 takes < 100 ms total —
    cheap enough to gate every CI run without slowing it down.
    """
    out = tmp_path / "bench.json"
    proc = subprocess.run(
        [
            sys.executable, str(BENCH_RUNNER),
            "--kernel-subset", "greedy_mwis",
            "--n-repeats", "1",
            "--output", "json",
            "--output-path", str(out),
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}: {proc.stdout}\n{proc.stderr}"
    )
    assert out.exists(), "JSON output file was not written"
    payload = json.loads(out.read_text(encoding="utf-8"))

    # Top-level shape.
    assert payload["schema_version"] == 1
    assert payload["n_kernels"] == 1
    rows = payload["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 2  # one python + one native row per kernel.

    # Per-row schema (the contract the diff script reads).
    required_keys = {
        "kernel", "slug", "group", "size_label", "mode", "n_repeats",
        "wall_seconds_p50", "wall_seconds_p95", "speedup_vs_python",
    }
    for row in rows:
        missing = required_keys - set(row)
        assert not missing, f"row missing keys: {missing} (row={row})"
        assert row["slug"] == "greedy_mwis"
        assert row["mode"] in {"python", "native"}
        assert row["n_repeats"] == 1
        assert isinstance(row["wall_seconds_p50"], (int, float))
        assert isinstance(row["wall_seconds_p95"], (int, float))
        assert row["wall_seconds_p50"] >= 0.0
        assert row["wall_seconds_p95"] >= row["wall_seconds_p50"] - 1e-9


# ---------------------------------------------------------------------------
# 2. Diff-script gating tests
# ---------------------------------------------------------------------------


def test_diff_perf_flags_regression(tmp_path: Path) -> None:
    """20% slower on native p50 → exit 1 + markdown contains 'REGRESSION'."""
    master = tmp_path / "master.json"
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    master.write_text(json.dumps(_make_payload(native_p50=1.0, native_p95=1.05)))
    pr.write_text(json.dumps(_make_payload(native_p50=1.2, native_p95=1.25)))

    code, _ = _run_diff(master, pr, md)
    assert code == 1, f"expected exit 1, got {code}"
    body = md.read_text(encoding="utf-8")
    assert "REGRESSION" in body
    # The verdict cell on the native row must read REGRESSION (not just
    # the headline). We check the cell explicitly.
    native_lines = [
        ln for ln in body.splitlines() if "native" in ln and "|" in ln
    ]
    assert any("REGRESSION" in ln for ln in native_lines), native_lines


def _verdict_cells(body: str) -> list[str]:
    """Extract the ``Verdict`` column value from each table row.

    The diff markdown deliberately uses words like ``REGRESSION``
    in its explanatory header text, so a naive ``"REGRESSION" in body``
    check leaks across rows. We pull the verdict cell explicitly by
    splitting on the table pipe and reading the last non-empty column.
    """
    cells: list[str] = []
    for ln in body.splitlines():
        # Verdict rows are formatted ``| ... | ... | VERDICT |``.
        if "|" not in ln:
            continue
        if ln.lstrip().startswith("|---"):
            continue
        if "Verdict" in ln and "Kernel" in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        # ``"| a | b | c |".split("|") -> ["", "a", "b", "c", ""]``
        parts = [p for p in parts if p]
        if not parts:
            continue
        last = parts[-1]
        if last in {"OK", "WARNING", "REGRESSION", "NEW", "REMOVED"}:
            cells.append(last)
    return cells


def test_diff_perf_flags_warning_only(tmp_path: Path) -> None:
    """7% slower on native p50 → exit 2 + markdown WARNING verdict cell."""
    master = tmp_path / "master.json"
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    master.write_text(json.dumps(_make_payload(native_p50=1.0, native_p95=1.05)))
    pr.write_text(json.dumps(_make_payload(native_p50=1.07, native_p95=1.12)))

    code, _ = _run_diff(master, pr, md)
    assert code == 2, f"expected exit 2, got {code}"
    body = md.read_text(encoding="utf-8")
    cells = _verdict_cells(body)
    assert "WARNING" in cells, cells
    assert "REGRESSION" not in cells, (
        f"warning-only diff should not flag any REGRESSION verdict; cells={cells}"
    )


def test_diff_perf_passes_no_regression(tmp_path: Path) -> None:
    """Identical wall times → exit 0 + every verdict cell is 'OK'."""
    master = tmp_path / "master.json"
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    payload = _make_payload(native_p50=1.0, native_p95=1.05)
    master.write_text(json.dumps(payload))
    pr.write_text(json.dumps(payload))

    code, _ = _run_diff(master, pr, md)
    assert code == 0, f"expected exit 0, got {code}"
    body = md.read_text(encoding="utf-8")
    cells = _verdict_cells(body)
    assert cells, "no verdict rows parsed"
    assert all(c == "OK" for c in cells), cells


def test_diff_perf_python_row_is_ungated(tmp_path: Path) -> None:
    """A 50% python-mode regression should NOT fail the build.

    Only the native mode is gated. The python row's verdict is always
    OK because the python wall time is the algorithmic reference and
    drifts purely with runner noise.
    """
    master = tmp_path / "master.json"
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    master.write_text(json.dumps(_make_payload(python_p50=1.0, python_p95=1.05)))
    pr.write_text(json.dumps(_make_payload(python_p50=1.5, python_p95=1.55)))

    code, _ = _run_diff(master, pr, md)
    assert code == 0, f"expected exit 0 (python row ungated), got {code}"


def test_diff_perf_handles_missing_master(tmp_path: Path) -> None:
    """First-run-after-enabling: missing master JSON → exit 0, SKIPPED note."""
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    pr.write_text(json.dumps(_make_payload()))

    code, _ = _run_diff(tmp_path / "does-not-exist.json", pr, md)
    assert code == 0, f"missing-master should soft-pass; got {code}"
    body = md.read_text(encoding="utf-8")
    assert "SKIPPED" in body or "first run" in body.lower()


def test_diff_perf_added_kernel_does_not_fail(tmp_path: Path) -> None:
    """A PR that adds a new kernel reports it but does not fail the gate."""
    master = tmp_path / "master.json"
    pr = tmp_path / "pr.json"
    md = tmp_path / "diff.md"

    master.write_text(json.dumps(_make_payload(slug="kernel_a")))
    pr_payload = _make_payload(slug="kernel_a")
    new_kernel = _make_payload(slug="kernel_b", kernel="New kernel B")
    pr_payload["rows"].extend(new_kernel["rows"])
    pr.write_text(json.dumps(pr_payload))

    code, _ = _run_diff(master, pr, md)
    assert code == 0, f"adding a kernel should not fail; got {code}"
    body = md.read_text(encoding="utf-8")
    assert "NEW" in body
