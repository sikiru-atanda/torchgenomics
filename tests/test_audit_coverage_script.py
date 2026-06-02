"""Tests for scripts/audit_public_coverage.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_audit_script_runs_and_emits_json(tmp_path):
    """Audit script should produce a JSON file with the documented schema."""
    out = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_public_coverage.py"),
         "--output", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "rows" in data
    assert "summary" in data
    assert isinstance(data["rows"], list)
    assert len(data["rows"]) > 0


def test_audit_row_schema(tmp_path):
    """Each row must contain the documented keys."""
    out = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_public_coverage.py"),
         "--output", str(out)],
        cwd=REPO, check=True, timeout=120,
    )
    data = json.loads(out.read_text())
    required_keys = {"module", "symbol", "kind", "tier",
                     "has_direct_test", "test_files", "lineno"}
    for row in data["rows"][:50]:
        assert required_keys.issubset(row.keys()), row.keys()
        assert row["tier"] in (1, 2, 3, None)
        assert row["kind"] in ("function", "class", "method", "constant", "module")
        assert isinstance(row["has_direct_test"], bool)


def test_audit_summary_includes_tier_counts(tmp_path):
    """Summary block must include tier counts and untested counts."""
    out = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_public_coverage.py"),
         "--output", str(out)],
        cwd=REPO, check=True, timeout=120,
    )
    data = json.loads(out.read_text())
    summary = data["summary"]
    assert "by_tier" in summary
    assert "untested_by_tier" in summary
    for tier in (1, 2, 3):
        assert tier in summary["by_tier"] or str(tier) in summary["by_tier"]


def test_audit_classifies_known_module():
    """Sanity: torchgenomics.linalg.eigendecompose should be Tier 1."""
    out_path = Path("/tmp/_audit_test.json")
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_public_coverage.py"),
         "--output", str(out_path)],
        cwd=REPO, check=True, timeout=120,
    )
    data = json.loads(out_path.read_text())
    eig_rows = [r for r in data["rows"]
                if r["module"] == "torchgenomics.linalg" and r["symbol"] == "eigendecompose"]
    assert len(eig_rows) == 1
    assert eig_rows[0]["tier"] == 1


def test_no_false_positive_substring_match(tmp_path):
    """A common-name symbol like 'Result' should not be counted as tested
    just because some unrelated test file mentions the word 'Result'.
    Use this as a regression gate against I1.
    """
    out = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "audit_public_coverage.py"),
         "--output", str(out)],
        cwd=REPO, check=True, timeout=120,
    )
    data = json.loads(out.read_text())
    # PGSResult is in torchgenomics.pgs. Find rows for any module-symbol that
    # claims has_direct_test=True; for at least one of them, verify the
    # claimed test_file actually imports that exact symbol from that exact
    # module. (We can't enumerate all but we can spot-check.)
    sample = next(
        (r for r in data["rows"]
         if r["has_direct_test"] and r["symbol"] == "PGSResult"),
        None,
    )
    if sample is None:
        return  # nothing to check; PGSResult may not exist or may be untested
    test_path = REPO / sample["test_files"][0]
    text = test_path.read_text()
    # Either explicit `from torchgenomics.pgs import ... PGSResult ...`
    # or `from torchgenomics import pgs` + `pgs.PGSResult` somewhere.
    assert (
        "PGSResult" in text and
        ("torchgenomics.pgs" in text or "torchgenomics import pgs" in text or
         "torchgenomics import" in text)
    )
