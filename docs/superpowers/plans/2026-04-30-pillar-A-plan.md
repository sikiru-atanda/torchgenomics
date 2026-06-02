# Pillar A — Coverage Audit + Tiered Fill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive every public function in `torchgenomics/` through tier-appropriate validation (math correctness for Tier 1, behavioral for Tier 2, smoke for Tier 3), per the campaign spec at `docs/superpowers/specs/2026-04-30-validation-campaign-design.md` (commit `dc6654e`).

**Architecture:** Setup the campaign infrastructure (branch, audit script, reviewer prompts, findings ledger), run an audit producing JSON of `{module.symbol → has_direct_test}`, then close every `has_direct_test=false` row with a tier-appropriate test. Two-track reviewer agent dispatch at the end of each tier.

**Tech Stack:** Python 3.10+, `pytest`, `numpy`, `scipy`, `torch`, `pandas`, ruff, mypy. No new runtime dependencies introduced by the plan itself.

---

## File Structure

**New files (created by this plan):**

| Path | Responsibility |
|---|---|
| `scripts/audit_public_coverage.py` | Walks `torchgenomics/`, emits `coverage_audit.json` |
| `docs/validation_findings/coverage_audit.json` | Audit output; key input to Tier 1/2/3 work |
| `docs/validation_findings.md` | Master findings ledger (append-only) |
| `docs/superpowers/reviewer_prompts/code_review.md` | Track-1 reviewer prompt template |
| `docs/superpowers/reviewer_prompts/rerun_verification.md` | Track-2 reviewer prompt template |
| `tests/test_coverage_<module>.py` | New tests added per untested public symbol, one file per module |
| `tests/conftest_coverage.py` | Shared fixtures + helpers for tiered tests (small synthetic genotype, kinship, phenotype) |

**Modified files:**

| Path | Modification |
|---|---|
| `pyproject.toml` | Register `external` pytest marker for forward compatibility (Pillar B uses it) |
| `CLAUDE.md` | Mention validation campaign and `pytest -m external` once Pillar A closes |

**Branching:** All Pillar A work lands on `validation/pillar-A-coverage` branched off latest `master`. PR opened against `master` at C2 checkpoint.

---

## Task 0: Branch Setup

**Files:**
- No new files. Creates the working branch.

- [ ] **Step 0.1: Verify clean working tree**

```bash
cd /home/sikiru.atanda/Documents/GWAS_Expert
git status --short
```

Expected: a small list of pre-existing modified files (CLAUDE.md, bench/, etc., from earlier session) — fine. No staged changes.

- [ ] **Step 0.2: Stash unrelated changes if any**

```bash
git stash push -m "pre-pillar-A snapshot" -u || echo "nothing to stash"
```

Expected: either a stash entry created, or "nothing to stash".

- [ ] **Step 0.3: Create the Pillar A branch off master**

```bash
git checkout -b validation/pillar-A-coverage master
git status --short
```

Expected: on branch `validation/pillar-A-coverage`, clean working tree.

- [ ] **Step 0.4: Confirm branch is fresh from master**

```bash
git log --oneline master..HEAD
```

Expected: empty output (no commits ahead yet).

---

## Task 1: Reviewer Prompt Templates

**Files:**
- Create: `docs/superpowers/reviewer_prompts/code_review.md`
- Create: `docs/superpowers/reviewer_prompts/rerun_verification.md`

These templates are filled in per-tier by the executor; they are committed once and then referenced.

- [ ] **Step 1.1: Write the code-review template**

Create `docs/superpowers/reviewer_prompts/code_review.md` with content:

````markdown
# Code-Review Reviewer Prompt — Pillar {{PILLAR}} Tier {{TIER}}

You are reviewing tier {{TIER}} of pillar {{PILLAR}} of the TorchGenomics validation campaign.

## What was just done

The campaign spec is at `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`. The pillar plan is at `docs/superpowers/plans/2026-04-30-pillar-{{PILLAR}}-plan.md`.

This tier's bar (from §4.1 of the spec):

{{TIER_BAR_QUOTE}}

The diff under review is:

```bash
git diff {{TIER_START_SHA}}..HEAD -- tests/test_coverage_*.py scripts/ pyproject.toml
```

## What I want you to verify

1. **Bar conformance** — every new test in this tier meets the tier's bar. Tier 1 tests must verify math correctness (closed-form / scipy / Monte-Carlo). Tier 2 tests must cover golden + edge + error. Tier 3 tests must be a happy-path smoke check.
2. **Tolerance contract** — tolerances in new tests match `docs/validation.md` Section 16 where applicable. New tolerances introduced are justified (in a comment) by an observed-then-floored measurement, not aspirational.
3. **F3 severity classification** — every commit that fixes production code is correctly classified per the F3 table in §9 of the spec. V1-core fixes are real; post-V1 should be xfail'd or ledgered, not silently fixed.
4. **Test isolation** — tests do not depend on global state; fixtures use `conftest_coverage.py`; no committed-data dependencies beyond what the spec allows.
5. **Findings ledger** — every divergence found is a row in `docs/validation_findings.md` with the documented schema.

## What I do NOT want you to do

- Do not propose refactors to TorchGenomics production code beyond what F3 already drove.
- Do not propose adding test cases beyond the tier's bar.
- Do not propose new dependencies.

## Output

Return three sections:

- **Blockers** — issues that must be fixed before the tier closes.
- **Nits** — non-blocking improvements.
- **Out-of-scope** — items that should be deferred to later pillars.

Under 500 words.
````

- [ ] **Step 1.2: Write the re-run verification template**

Create `docs/superpowers/reviewer_prompts/rerun_verification.md` with content:

````markdown
# Re-Run Reviewer Prompt — Pillar {{PILLAR}} Tier {{TIER}}

You are an independent verifier for pillar {{PILLAR}} tier {{TIER}} of the TorchGenomics validation campaign. You have NOT seen the conversation that produced these tests.

## Context

TorchGenomics commit under verification: {{COMMIT_SHA}}. Branch: `validation/pillar-{{PILLAR}}-coverage`.

The campaign claims this tier validates {{TIER_CLAIM}}. The spec is at `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`. Don't take its word for the result — independently verify.

## Your task

1. **Memory pre-flight (mandatory).** Before any download or run, assert:
   - `df --output=avail /home` ≥ 5 GB.
   - `free -g` available column ≥ 8 GB.
   If either fails, abort and report. Do not partially execute.

2. **Reproduce the verification.** {{REPRO_INSTRUCTIONS}}

3. **Compare.** Run the new tests committed in this tier:
   ```bash
   pytest {{TIER_TESTS_GLOB}} -v
   ```
   Then independently re-derive the expected values for {{SAMPLE_TESTS}} using {{INDEPENDENT_METHOD}}. Compare to {{TOLERANCE}}.

4. **Classify.** For each test sampled:
   - **agree** — within tolerance.
   - **divergence** — beyond tolerance; report the function, dataset, observed Δ, tolerance, and which F3 class applies.
   - **infra-blocker** — could not run; explain.

## Output

Plain markdown, under 400 words:

- **Pre-flight result** — pass / abort.
- **Sampled tests** — list with classification.
- **Findings** — to add to `docs/validation_findings.md`. Use the schema in §11 of the spec.
- **Recommendation** — proceed to next tier / fix-now / halt.

## What you cannot do

- Do not commit. Do not push. Do not modify the repo.
- Do not skip pre-flight.
- Do not approve a tier with any V1-core divergence — that's a halt-and-flag.
````

- [ ] **Step 1.3: Verify both files render cleanly**

```bash
ls -la docs/superpowers/reviewer_prompts/
wc -l docs/superpowers/reviewer_prompts/*.md
```

Expected: two files, each ~50–80 lines.

- [ ] **Step 1.4: Commit**

```bash
git add docs/superpowers/reviewer_prompts/
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A: reviewer prompt templates

Two templates committed up front so every tier's review is consistent:
code_review.md (Track 1, superpowers:code-reviewer) and
rerun_verification.md (Track 2, general-purpose with mandatory memory
pre-flight).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Findings Ledger Skeleton

**Files:**
- Create: `docs/validation_findings.md`
- Create: `docs/validation_findings/.gitkeep`

- [ ] **Step 2.1: Write the ledger header**

Create `docs/validation_findings.md` with content:

```markdown
# TorchGenomics Validation Findings Ledger

Append-only ledger of every divergence found during the validation
campaign. Schema and policy: spec §11 + §9.

Classifications per F3:
- **V1-core / fix-now** — Phases 0–13 math error; commit fix in same tier.
- **V1-core / regression** — golden test that previously passed; halt pillar.
- **post-V1 / documented** — Phase ≥14 divergence; xfail with reason.
- **post-V1 / open issue** — divergence exposes missing-feature charter claim.
- **infra-blocker** — install / data / env failure; documented; comparison skipped.

| Date | Pillar | Tier | Module / Function | Reference | Dataset | Δ observed | Tolerance | F3 class | Resolution |
|------|--------|------|-------------------|-----------|---------|------------|-----------|----------|------------|
| _no entries yet_ | | | | | | | | | |
```

- [ ] **Step 2.2: Add a placeholder under `docs/validation_findings/`**

```bash
touch docs/validation_findings/.gitkeep
```

This directory will hold the audit JSON and any per-pillar artifacts.

- [ ] **Step 2.3: Commit**

```bash
git add docs/validation_findings.md docs/validation_findings/.gitkeep
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A: findings ledger skeleton

Append-only ledger committed up front. Every divergence found during
the campaign gets a row per the spec §11 schema, classified per F3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Coverage Auditor — TDD

**Files:**
- Create: `scripts/audit_public_coverage.py`
- Create: `tests/test_audit_coverage_script.py`

The auditor is itself a tested script.

- [ ] **Step 3.1: Write failing test for the audit data model**

Create `tests/test_audit_coverage_script.py`:

```python
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
```

- [ ] **Step 3.2: Run tests to confirm they fail (script doesn't exist yet)**

```bash
pytest tests/test_audit_coverage_script.py -v
```

Expected: 4 tests fail with `FileNotFoundError` or `subprocess.CalledProcessError`.

- [ ] **Step 3.3: Write the audit script**

Create `scripts/audit_public_coverage.py`:

```python
#!/usr/bin/env python3
"""Audit public-symbol test coverage in torchgenomics/.

Walks every module under torchgenomics/, enumerates public symbols (via __all__
when present, else top-level non-underscore names), and for each symbol
checks whether at least one file under tests/ directly imports or invokes it.

Outputs a JSON file with a `rows` array (one row per public symbol) and a
`summary` block with tier-level counts.

Tier mapping (from spec §4.1):
  Tier 1 (math): models, linalg, stats, optim, scan
  Tier 2 (behavioral): io, preprocess, ld, pgs, postgwas, multiomics
  Tier 3 (smoke): viz, annotate, cli, results
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO / "torchgenomics"
TESTS_ROOT = REPO / "tests"

TIER_MAP = {
    "torchgenomics.models": 1,
    "torchgenomics.linalg": 1,
    "torchgenomics.stats": 1,
    "torchgenomics.optim": 1,
    "torchgenomics.scan": 1,
    "torchgenomics.io": 2,
    "torchgenomics.preprocess": 2,
    "torchgenomics.ld": 2,
    "torchgenomics.pgs": 2,
    "torchgenomics.postgwas": 2,
    "torchgenomics.multiomics": 2,
    "torchgenomics.viz": 3,
    "torchgenomics.annotate": 3,
    "torchgenomics.results": 3,
    "torchgenomics.cli": 3,
}


def module_tier(modname: str) -> int | None:
    for prefix, tier in TIER_MAP.items():
        if modname == prefix or modname.startswith(prefix + "."):
            return tier
    return None


def public_symbols(mod):
    if hasattr(mod, "__all__"):
        names = list(mod.__all__)
    else:
        names = [n for n in dir(mod) if not n.startswith("_")]
    out = []
    for name in names:
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue
        # Skip re-exported submodules and external imports.
        if inspect.ismodule(obj):
            continue
        # Only count things actually defined in the package.
        defining_module = getattr(obj, "__module__", None)
        if defining_module is not None and not defining_module.startswith("torchgenomics"):
            continue
        out.append((name, obj))
    return out


def kind_of(obj):
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.ismethod(obj):
        return "method"
    if inspect.ismodule(obj):
        return "module"
    return "constant"


def lineno_of(obj):
    try:
        return inspect.getsourcelines(obj)[1]
    except (TypeError, OSError):
        return None


def find_test_files_for_symbol(symbol: str) -> list[str]:
    """Conservative match: explicit `<symbol>` token in tests/."""
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    matches: list[str] = []
    for p in TESTS_ROOT.rglob("test_*.py"):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if pattern.search(text):
            matches.append(str(p.relative_to(REPO)))
    return matches


def walk_package(root: Path):
    """Yield (module_name, module_obj) for every submodule under torchgenomics/."""
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        rel = path.relative_to(REPO).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modname = ".".join(parts)
        if not modname.startswith("torchgenomics"):
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            print(f"WARN: skipping {modname}: {exc}", file=sys.stderr)
            continue
        yield modname, mod


def build_rows():
    rows = []
    seen: set[tuple[str, str]] = set()
    for modname, mod in walk_package(PKG_ROOT):
        tier = module_tier(modname)
        for symname, obj in public_symbols(mod):
            key = (modname, symname)
            if key in seen:
                continue
            seen.add(key)
            test_files = find_test_files_for_symbol(symname)
            rows.append({
                "module": modname,
                "symbol": symname,
                "kind": kind_of(obj),
                "tier": tier,
                "has_direct_test": len(test_files) > 0,
                "test_files": test_files,
                "lineno": lineno_of(obj),
            })
    return rows


def build_summary(rows):
    by_tier: dict[int, int] = {1: 0, 2: 0, 3: 0}
    untested_by_tier: dict[int, int] = {1: 0, 2: 0, 3: 0}
    untested_no_tier = 0
    for r in rows:
        t = r["tier"]
        if t in by_tier:
            by_tier[t] += 1
            if not r["has_direct_test"]:
                untested_by_tier[t] += 1
        elif not r["has_direct_test"]:
            untested_no_tier += 1
    return {
        "total_symbols": len(rows),
        "by_tier": by_tier,
        "untested_by_tier": untested_by_tier,
        "untested_no_tier": untested_no_tier,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="docs/validation_findings/coverage_audit.json")
    args = p.parse_args(argv)

    rows = build_rows()
    summary = build_summary(rows)

    out = {"rows": rows, "summary": summary}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
```

Make it executable:

```bash
chmod +x scripts/audit_public_coverage.py
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
pytest tests/test_audit_coverage_script.py -v
```

Expected: all 4 pass.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/audit_public_coverage.py tests/test_audit_coverage_script.py
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A: coverage auditor

Walks torchgenomics/, identifies public symbols, classifies by tier
(1 math / 2 behavioral / 3 smoke), and conservatively flags whether
each symbol is directly tested by any file under tests/.

Output: docs/validation_findings/coverage_audit.json with rows + summary.
The summary's untested_by_tier counts drive the rest of the pillar.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Run the Audit

**Files:**
- Modify (create): `docs/validation_findings/coverage_audit.json`

- [ ] **Step 4.1: Run the auditor**

```bash
python3 scripts/audit_public_coverage.py
```

Expected: stdout reports `Wrote N rows to docs/validation_findings/coverage_audit.json` and a summary line.

- [ ] **Step 4.2: Inspect the summary**

```bash
python3 -c "
import json
data = json.load(open('docs/validation_findings/coverage_audit.json'))
print(json.dumps(data['summary'], indent=2))
"
```

Expected: counts per tier and untested-per-tier counts.

- [ ] **Step 4.3: Print the actionable list — Tier 1 untested symbols**

```bash
python3 -c "
import json
rows = json.load(open('docs/validation_findings/coverage_audit.json'))['rows']
untested = [r for r in rows if r['tier'] == 1 and not r['has_direct_test']]
print(f'Tier 1 untested: {len(untested)}')
for r in untested:
    print(f\"  {r['module']}.{r['symbol']} ({r['kind']})\")
" | head -100
```

This is the worklist for Task 5.

- [ ] **Step 4.4: Commit the audit JSON**

```bash
git add docs/validation_findings/coverage_audit.json
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A: coverage audit baseline

Run scripts/audit_public_coverage.py against current master. JSON
committed as the canonical worklist for Tier 1/2/3 fill tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Tier 1 — Math Correctness Tests (Worked Example: `linalg`)

This task demonstrates the test-writing pattern with a concrete worked example. After this task, **Task 6** applies the same pattern to every remaining Tier-1 untested symbol, parameterized over the audit JSON.

**Files:**
- Create: `tests/conftest_coverage.py`
- Create: `tests/test_coverage_linalg.py`

- [ ] **Step 5.1: Write shared fixtures for tiered coverage tests**

Create `tests/conftest_coverage.py`:

```python
"""Shared fixtures for tier-N coverage tests.

These fixtures produce the smallest valid inputs that exercise the function
under test with a known-truth answer. Tests in tests/test_coverage_*.py
import them via pytest collection.

Sizes are deliberately tiny: every Tier-1 test must complete in < 1 second
so the full coverage suite stays under the 5-minute CI budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_genotype_diploid():
    """100 samples × 50 SNPs, MAF ~0.3, no missing."""
    rng = np.random.default_rng(42)
    G = rng.binomial(2, 0.3, size=(100, 50)).astype(np.float64)
    return torch.from_numpy(G)


@pytest.fixture(scope="module")
def tiny_genotype_tetraploid():
    """100 samples × 50 SNPs, ploidy=4, no missing."""
    rng = np.random.default_rng(43)
    G = rng.binomial(4, 0.3, size=(100, 50)).astype(np.float64)
    return torch.from_numpy(G)


@pytest.fixture(scope="module")
def tiny_phenotype():
    """Single quantitative trait, n=100, normal."""
    rng = np.random.default_rng(44)
    return torch.from_numpy(rng.normal(0, 1, size=(100, 1)).astype(np.float64))


@pytest.fixture(scope="module")
def tiny_kinship():
    """Symmetric PSD kinship from tiny diploid genotype."""
    rng = np.random.default_rng(42)
    G = rng.binomial(2, 0.3, size=(100, 50)).astype(np.float64)
    Gc = G - G.mean(axis=0, keepdims=True)
    K = Gc @ Gc.T / 50.0
    K = K + np.eye(100) * 1e-6
    return torch.from_numpy(K)


@pytest.fixture(scope="module")
def tiny_covariates():
    """Intercept + one continuous covariate, n=100."""
    rng = np.random.default_rng(45)
    X = np.column_stack([np.ones(100), rng.normal(0, 1, 100)]).astype(np.float64)
    return torch.from_numpy(X)


@pytest.fixture(scope="module")
def stat_dtype():
    """The statistical-inference dtype TorchGenomics uses."""
    return torch.float64
```

- [ ] **Step 5.2: Write a worked Tier-1 test for `linalg.eigendecompose`**

Create `tests/test_coverage_linalg.py`:

```python
"""Tier-1 math-correctness coverage tests for torchgenomics.linalg.

Bar (spec §4.3 Tier 1):
- Closed-form / scipy / Monte-Carlo verification per public function.
- One function in isolation per test. No compositional tests.
- Tolerance ≤ 1e-10 absolute for closed-form; documented otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from torchgenomics.linalg import eigendecompose


pytestmark = pytest.mark.timeout(30)


class TestEigendecompose:
    """`eigendecompose` should return eigenvalues / eigenvectors of a symmetric
    PSD matrix, matching scipy.linalg.eigh to machine precision."""

    def test_matches_scipy_eigh(self, tiny_kinship):
        """Output eigenvalues and reconstructed matrix agree with scipy."""
        K = tiny_kinship
        out = eigendecompose(K)
        eigvals = out.eigenvalues if hasattr(out, "eigenvalues") else out[0]
        eigvecs = out.eigenvectors if hasattr(out, "eigenvectors") else out[1]

        eigvals_np = eigvals.cpu().numpy()
        eigvecs_np = eigvecs.cpu().numpy()
        K_np = K.cpu().numpy()

        ref_vals, ref_vecs = sla.eigh(K_np)

        # Eigenvalues must agree (sort-order invariant).
        np.testing.assert_allclose(np.sort(eigvals_np), np.sort(ref_vals),
                                    rtol=1e-10, atol=1e-10)

        # Reconstruction K ≈ V Λ V^T.
        K_recon = eigvecs_np @ np.diag(eigvals_np) @ eigvecs_np.T
        np.testing.assert_allclose(K_recon, K_np, rtol=1e-8, atol=1e-8)

    def test_eigenvectors_orthonormal(self, tiny_kinship):
        """Eigenvector matrix must satisfy V^T V = I to machine precision."""
        out = eigendecompose(tiny_kinship)
        V = (out.eigenvectors if hasattr(out, "eigenvectors") else out[1]).cpu().numpy()
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[0]),
                                    rtol=1e-10, atol=1e-10)
```

- [ ] **Step 5.3: Run the test**

```bash
pytest tests/test_coverage_linalg.py -v
```

Expected: both tests pass. If `eigendecompose` returns a different type than expected (tuple vs object with attributes), the test handles both via `hasattr`.

- [ ] **Step 5.4: If the test reveals a bug, classify per F3**

If a test fails:
1. Inspect: is this a V1-core math error (linalg is V1)?
2. If yes: this is **fix-now** per F3. Diagnose, fix in `torchgenomics/linalg/`, commit fix + the test, append findings ledger row.
3. If the failure is a tolerance miss only (e.g., `1e-9` vs `1e-10`), tighten or loosen the test tolerance to the observed level *with a comment explaining the floor*, per the spec's "tolerances calibrated to observed reality" rule.

If the test passes (expected case for `eigendecompose`), continue.

- [ ] **Step 5.5: Commit**

```bash
git add tests/conftest_coverage.py tests/test_coverage_linalg.py
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A Tier 1: linalg.eigendecompose math-correctness test

Worked example demonstrating the Tier-1 bar: scipy comparison +
orthonormality + reconstruction, all to ≤1e-8 absolute. Establishes
the conftest_coverage.py fixture set used by every coverage_*.py file.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Tier 1 — Audit-Driven Fill (All Remaining Untested Tier-1 Symbols)

This task iterates over every untested Tier-1 symbol from the audit JSON. For each, write a test using the pattern from Task 5.

**Files:**
- Modify: `tests/test_coverage_linalg.py` (add tests for remaining `linalg` untested symbols)
- Create: `tests/test_coverage_models.py`
- Create: `tests/test_coverage_stats.py`
- Create: `tests/test_coverage_optim.py` (if non-trivial untested optim entry points)
- Create: `tests/test_coverage_scan.py` (if non-trivial untested scan entry points)

The audit JSON is the worklist. Below is the per-symbol test-writing template.

### 6.1 Per-symbol template

For each untested Tier-1 symbol `module.symbol`:

- [ ] **Determine the math-correctness reference.** Pick the most appropriate of:
  - Closed-form expression (e.g., `bonferroni` is `min(1, p × m)`).
  - Standard-library reference (`scipy.stats.<dist>`, `scipy.linalg`, `numpy.linalg`).
  - Monte-Carlo expectation (≥ 10 000 reps; assert empirical mean / variance / coverage).
  - For an existing accelerator: parity with the pure-torch reference under `TORCHGENOMICS_DISABLE_NATIVE=1`.

- [ ] **Write the test in `tests/test_coverage_<module>.py`** under a `class Test<Symbol>:` block. At minimum:

```python
class TestSymbolName:
    """One-line description of what bar this fulfills."""

    def test_<verb>(self, <fixtures>):
        """Specific assertion, with the reference identified inline."""
        # Build input from fixtures or rng-seeded synthetic.
        # Call torchgenomics function.
        # Compute reference value via scipy / numpy / closed-form.
        # np.testing.assert_allclose(actual, ref, rtol=…, atol=…)
```

- [ ] **Run the test:** `pytest tests/test_coverage_<module>.py::TestSymbolName -v`. Expected: pass.

- [ ] **If the test fails:** apply F3 logic from spec §9.

- [ ] **Commit per logical unit** (one commit per ~5 related symbols, not per individual test, to keep history readable):

```bash
git add tests/test_coverage_<module>.py
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 1: <module> coverage tests for <symbol-list>"
```

### 6.2 Concrete worked second example: `stats.benjamini_hochberg`

Add to `tests/test_coverage_stats.py`:

```python
"""Tier-1 math-correctness coverage tests for torchgenomics.stats."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import false_discovery_control

from torchgenomics.stats import benjamini_hochberg


class TestBenjaminiHochberg:
    """BH step-up: q_(i) = min over j>=i of (m × p_(j)) / j, monotonized."""

    def test_matches_scipy_bh(self):
        """Should agree with scipy.stats.false_discovery_control to machine precision."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=200).astype(np.float64)
        actual = benjamini_hochberg(torch.from_numpy(p)).cpu().numpy()
        expected = false_discovery_control(p, method="bh")
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_handles_all_ones(self):
        """All p=1 → all q=1."""
        p = torch.ones(50, dtype=torch.float64)
        q = benjamini_hochberg(p).cpu().numpy()
        np.testing.assert_allclose(q, np.ones(50))

    def test_monotone_after_sort(self):
        """BH-adjusted p values are monotone non-decreasing in original p order."""
        rng = np.random.default_rng(1)
        p = np.sort(rng.uniform(0, 1, size=100).astype(np.float64))
        q = benjamini_hochberg(torch.from_numpy(p)).cpu().numpy()
        assert np.all(np.diff(q) >= -1e-12)
```

- [ ] **Run + commit**:

```bash
pytest tests/test_coverage_stats.py -v
git add tests/test_coverage_stats.py
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 1: stats.benjamini_hochberg math-correctness tests"
```

### 6.3 Walking the audit list

- [ ] **Generate the worklist:**

```bash
python3 -c "
import json
rows = json.load(open('docs/validation_findings/coverage_audit.json'))['rows']
untested = [r for r in rows if r['tier'] == 1 and not r['has_direct_test']]
by_module = {}
for r in untested:
    by_module.setdefault(r['module'], []).append(r['symbol'])
for m, syms in sorted(by_module.items()):
    print(f'{m}: {len(syms)}')
    for s in syms:
        print(f'  - {s}')
" > /tmp/tier1_worklist.txt
wc -l /tmp/tier1_worklist.txt
cat /tmp/tier1_worklist.txt
```

- [ ] **Walk module-by-module.** For each module in the worklist, open the corresponding `tests/test_coverage_<module>.py` (creating it if absent) and add a `class Test<Symbol>:` block per untested symbol following 6.1. Commit per module with a message listing the symbols added.

- [ ] **After every module's worth of new tests, run that module's tests:**

```bash
pytest tests/test_coverage_<module>.py -v --tb=short
```

Address failures per F3. V1-core failures get fixes in the same tier; post-V1 failures get xfail markers with reason and a findings-ledger row.

- [ ] **At Tier-1 close, re-run the audit to verify zero remaining untested Tier-1 symbols:**

```bash
python3 scripts/audit_public_coverage.py
python3 -c "
import json
data = json.load(open('docs/validation_findings/coverage_audit.json'))
untested = data['summary']['untested_by_tier']
print('Tier 1 untested:', untested.get(1, untested.get('1', 0)))
assert untested.get(1, untested.get('1', 0)) == 0
"
```

Expected: Tier-1 untested = 0.

- [ ] **Commit the updated audit JSON:**

```bash
git add docs/validation_findings/coverage_audit.json
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 1: re-audit shows 0 untested Tier-1 symbols"
```

---

## Task 7: Tier 1 — Two-Track Reviewer Dispatch

**Files:**
- Read-only: `docs/superpowers/reviewer_prompts/code_review.md` and `rerun_verification.md` (filled in)
- Modify: `docs/validation_findings.md` (append findings)

- [ ] **Step 7.1: Capture the tier start SHA**

```bash
git log --oneline | head -20
TIER1_START=$(git log --oneline | grep "Pillar A: coverage audit baseline" | awk '{print $1}')
echo "Tier 1 start SHA: $TIER1_START"
```

- [ ] **Step 7.2: Dispatch Track 1 (code-reviewer)**

Use the Agent tool with `subagent_type=superpowers:code-reviewer`. Fill in the `code_review.md` template with `{{PILLAR}}=A`, `{{TIER}}=1`, `{{TIER_START_SHA}}=$TIER1_START`, `{{TIER_BAR_QUOTE}}` = the Tier 1 row of the §4.1 table.

Address every blocker. Re-run tests. Commit fixes.

- [ ] **Step 7.3: Dispatch Track 2 (general-purpose, fresh-env re-run)**

Use the Agent tool with `subagent_type=general-purpose`. Fill in the `rerun_verification.md` template with `{{PILLAR}}=A`, `{{TIER}}=1`, `{{COMMIT_SHA}}=$(git rev-parse HEAD)`, `{{TIER_CLAIM}}` = "math correctness for models / linalg / stats / optim / scan public functions", `{{REPRO_INSTRUCTIONS}}` = "Sample 5 tests at random from `tests/test_coverage_{linalg,stats,models,optim,scan}.py`. For each, independently re-derive the expected value using scipy or numpy, comparing to the test's stated tolerance. Report whether each agrees or diverges.", `{{TIER_TESTS_GLOB}}` = `tests/test_coverage_{linalg,stats,models,optim,scan}.py`, `{{SAMPLE_TESTS}}` = "5 random Tier-1 coverage tests", `{{INDEPENDENT_METHOD}}` = "scipy / numpy from a clean Python env", `{{TOLERANCE}}` = "the tolerance written in each test".

The agent's output goes into the conversation. Append any findings to `docs/validation_findings.md`.

- [ ] **Step 7.4: Append Track-2 findings to ledger and commit**

For each finding the agent reports:

```markdown
| 2026-MM-DD | A | 1 | <module>.<symbol> | <reference> | <dataset> | <Δ> | <tol> | <F3 class> | <link or "open"> |
```

Edit `docs/validation_findings.md`, add the rows, then:

```bash
git add docs/validation_findings.md
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 1: re-run reviewer findings"
```

If Track-2 reports a V1-core divergence the local tests didn't catch: **C4 emergency-stop**. Post finding to user; do not start Tier 2. Wait for guidance.

---

## Task 8: Tier 2 — Behavioral Tests (Audit-Driven)

Same execution shape as Task 6, with a different bar.

**Files:**
- Create: `tests/test_coverage_io.py`, `tests/test_coverage_preprocess.py`, `tests/test_coverage_ld.py`, `tests/test_coverage_pgs.py`, `tests/test_coverage_postgwas.py`, `tests/test_coverage_multiomics.py` (only as needed by the audit).

### 8.1 Tier 2 bar (per spec §4.3)

For each untested Tier-2 symbol, three test methods under one `Test<Symbol>` class:

```python
class TestSymbolName:
    """Behavioral coverage: golden / edge / error."""

    def test_golden_path(self, ...):
        """Typical valid input; assert documented contract (shape, key invariants)."""
        ...

    def test_edge_case(self, ...):
        """One of: empty, single-sample, all-missing, near-singular, ploidy=1, …"""
        ...

    def test_error_path(self, ...):
        """One invalid input → expected exception with message substring."""
        with pytest.raises(ExpectedExceptionType, match="<substring>"):
            symbol(invalid_input)
```

### 8.2 Worked example: `io.detect_format`

```python
"""Tier-2 behavioral coverage tests for torchgenomics.io."""

from __future__ import annotations

from pathlib import Path

import pytest

from torchgenomics.io import detect_format


class TestDetectFormat:
    """Behavioral coverage: golden / edge / error."""

    def test_golden_bed(self, tmp_path):
        """A .bed/.bim/.fam triple should be detected as 'bed'."""
        for ext in ("bed", "bim", "fam"):
            (tmp_path / f"x.{ext}").write_bytes(b"\x6c\x1b\x01" if ext == "bed" else b"")
        assert detect_format(str(tmp_path / "x.bed")) == "bed"

    def test_edge_extensionless_path(self, tmp_path):
        """Path with no recognizable extension should raise or return a documented sentinel."""
        empty = tmp_path / "noext"
        empty.write_text("")
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(str(empty))

    def test_error_nonexistent_path(self, tmp_path):
        """Missing path should raise FileNotFoundError or ValueError."""
        with pytest.raises((FileNotFoundError, ValueError)):
            detect_format(str(tmp_path / "definitely_does_not_exist.bed"))
```

### 8.3 Walk Tier 2

- [ ] **Generate Tier-2 worklist** (analogous to Step 6.3):

```bash
python3 -c "
import json
rows = json.load(open('docs/validation_findings/coverage_audit.json'))['rows']
untested = [r for r in rows if r['tier'] == 2 and not r['has_direct_test']]
by_module = {}
for r in untested:
    by_module.setdefault(r['module'], []).append(r['symbol'])
for m, syms in sorted(by_module.items()):
    print(f'{m}: {len(syms)}')
    for s in syms:
        print(f'  - {s}')
" > /tmp/tier2_worklist.txt
cat /tmp/tier2_worklist.txt
```

- [ ] **Walk module-by-module.** For each row, write a `Test<Symbol>` class with golden + edge + error methods per 8.1. F3 logic on failure.

- [ ] **Re-audit at Tier-2 close:**

```bash
python3 scripts/audit_public_coverage.py
python3 -c "
import json
data = json.load(open('docs/validation_findings/coverage_audit.json'))
u = data['summary']['untested_by_tier']
assert u.get(2, u.get('2', 0)) == 0, 'Tier 2 still has untested symbols'
"
```

- [ ] **Commit the audit refresh:**

```bash
git add docs/validation_findings/coverage_audit.json
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 2: re-audit shows 0 untested Tier-2 symbols"
```

---

## Task 9: Tier 2 — Two-Track Reviewer Dispatch

Same shape as Task 7 with `{{TIER}}=2`. Capture `TIER2_START` as the SHA of "Pillar A Tier 1: re-audit shows 0 untested Tier-1 symbols". Track-2 sample = "5 random Tier-2 coverage tests; for each, run the test once with the documented golden input and once with a hand-perturbed valid input that should still pass; verify the contract holds."

C4 emergency-stop if any V1-core regression surfaces — but Tier 2 is mostly post-V1 modules, so most divergences here are documented-and-continue.

---

## Task 10: Tier 3 — Smoke Tests (Audit-Driven)

**Files:**
- Create: `tests/test_coverage_viz.py`, `tests/test_coverage_annotate.py`, `tests/test_coverage_results.py`, `tests/test_coverage_cli.py`.

### 10.1 Tier 3 bar (per spec §4.3)

One test per public function:

```python
def test_<symbol>_smoke(self, <fixtures>):
    """Imports + invokes with simplest valid input; no raise; correct return type."""
    result = symbol(simple_input)
    assert result is not None
    # For viz: assert isinstance(result, matplotlib.axes.Axes) and result.has_data()
    # For cli: assert exit_code == 0 and at least one output file
    # For annotate (network-required): pytest.mark.skipif(no_network)
```

### 10.2 Worked example: `viz.qq_plot`

```python
"""Tier-3 smoke coverage tests for torchgenomics.viz."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from torchgenomics.viz import qq_plot


class TestQQPlot:
    def test_smoke(self):
        """qq_plot should return a matplotlib Axes with at least one artist."""
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=1000)
        ax = qq_plot(p)
        assert ax is not None
        assert len(ax.lines) + len(ax.collections) > 0
```

### 10.3 Walk Tier 3

- [ ] **Generate Tier-3 worklist:**

```bash
python3 -c "
import json
rows = json.load(open('docs/validation_findings/coverage_audit.json'))['rows']
untested = [r for r in rows if r['tier'] == 3 and not r['has_direct_test']]
by_module = {}
for r in untested:
    by_module.setdefault(r['module'], []).append(r['symbol'])
for m, syms in sorted(by_module.items()):
    print(f'{m}: {len(syms)}')
    for s in syms:
        print(f'  - {s}')
" > /tmp/tier3_worklist.txt
cat /tmp/tier3_worklist.txt
```

- [ ] **Walk module-by-module.** Smoke test per row. Network-required functions (`annotate.*`) get `@pytest.mark.skipif(os.environ.get("CI") or not _network_ok(), reason="network required")`.

- [ ] **For CLI subcommands:** smoke test invokes the subcommand via `subprocess.run([sys.executable, "-m", "torchgenomics.cli", "<subcommand>", "--help"])` and asserts exit code 0 + non-empty stdout. Full end-to-end CLI runs are Pillar C.

- [ ] **Re-audit at Tier-3 close:**

```bash
python3 scripts/audit_public_coverage.py
python3 -c "
import json
data = json.load(open('docs/validation_findings/coverage_audit.json'))
u = data['summary']['untested_by_tier']
total_untested = sum(u.values()) if isinstance(list(u.keys())[0], int) else sum(int(v) for v in u.values())
assert total_untested == 0, f'Still untested: {u}'
"
```

- [ ] **Commit:**

```bash
git add docs/validation_findings/coverage_audit.json
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "Pillar A Tier 3: re-audit shows 0 untested symbols across all tiers"
```

---

## Task 11: Tier 3 — Two-Track Reviewer Dispatch

Same shape as Tasks 7 and 9 with `{{TIER}}=3`. Track-2 sample = "3 random Tier-3 smoke tests; for each, verify the test imports the symbol from the correct module path and the assertion is non-trivial (not e.g. `assert True`)".

---

## Task 12: Pillar A Close — PR + Spec Update + CLAUDE.md

**Files:**
- Modify: `docs/superpowers/specs/2026-04-30-validation-campaign-design.md` (flesh out Pillar B detail)
- Modify: `CLAUDE.md` (mention validation campaign + how to re-run any pillar)
- Modify: `pyproject.toml` (register `external` marker for forward compatibility)

- [ ] **Step 12.1: Register the `external` pytest marker**

Edit `pyproject.toml` to add `"external"` under `[tool.pytest.ini_options].markers`:

```toml
markers = [
    "golden: golden reference tests against GEMMA/GAPIT/GWASpoly",
    "slow: tests that take more than 30 seconds",
    "gpu: tests that require a CUDA device",
    "external: tests that install + run external reference tools (Pillar B)",
]
```

Run a quick sanity:

```bash
pytest --markers | grep external
```

Expected: `@pytest.mark.external: tests that install + run external reference tools (Pillar B)`.

- [ ] **Step 12.2: Update CLAUDE.md**

Add a section under "Development workflow" or near the testing block:

```markdown
## Validation campaign

A function-by-function validation campaign runs out of `docs/superpowers/`:

- Spec: `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`
- Per-pillar plans: `docs/superpowers/plans/2026-04-30-pillar-{A,B,C,D}-plan.md`
- Findings ledger: `docs/validation_findings.md`

Re-run any pillar's coverage audit with:

```bash
python3 scripts/audit_public_coverage.py
```

Run Pillar B's external-reference comparisons (when wired):

```bash
pytest -m external
```
```

- [ ] **Step 12.3: Flesh out Pillar B detail in the spec**

Open `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`. Section 5 currently has the Pillar B table. Append a new subsection §5.4 "Pillar B execution order" with concrete first-tool, dataset, and expected timeline based on what Pillar A revealed. (Specifics filled in at execution time, not in this plan.)

- [ ] **Step 12.4: Final test pass**

```bash
pytest tests/ -v --tb=short
```

Expected: full test suite passes. The new coverage tests are part of the default run.

- [ ] **Step 12.5: Run lint + type check**

```bash
ruff check torchgenomics tests scripts
mypy torchgenomics
```

Address blockers. ruff format if needed: `ruff format torchgenomics tests scripts`.

- [ ] **Step 12.6: Final commit**

```bash
git add CLAUDE.md pyproject.toml docs/superpowers/specs/2026-04-30-validation-campaign-design.md
git -c user.email=sikiruandfriends@gmail.com -c user.name="Sikiru Atanda" commit -m "$(cat <<'EOF'
Pillar A close: register external marker, update CLAUDE.md, flesh out Pillar B

Pillar A complete:
- All in-tier public symbols have tier-appropriate tests.
- Coverage audit JSON shows 0 untested Tier 1/2/3 symbols.
- Two-track reviewer agents signed off on each tier.
- Findings ledger current.
- pyproject.toml registers `external` marker for Pillar B.
- CLAUDE.md mentions the campaign and re-run paths.
- Spec §5 expanded with execution-order detail for Pillar B.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 12.7: Open the PR**

```bash
git push -u origin validation/pillar-A-coverage
gh pr create --title "Validation campaign — Pillar A: coverage audit + tiered fill" --body "$(cat <<'EOF'
## Summary

Pillar A of the TorchGenomics validation campaign per
`docs/superpowers/specs/2026-04-30-validation-campaign-design.md`.

- New auditor: `scripts/audit_public_coverage.py` enumerates public
  symbols across `torchgenomics/`, classifies them by tier, and conservatively
  flags whether each has a direct test.
- Tier 1 (math correctness): scipy / closed-form / Monte-Carlo tests
  for every public symbol in `models/`, `linalg/`, `stats/`, plus
  internal-but-math-heavy `optim/` and `scan/`.
- Tier 2 (behavioral): golden + edge + error tests for every public
  symbol in `io/`, `preprocess/`, `ld/`, `pgs/`, `postgwas/`, `multiomics/`.
- Tier 3 (smoke): import + happy-path invocation tests for every
  public symbol in `viz/`, `annotate/`, `cli/`, `results/`.
- Two-track reviewer agents signed off on each tier
  (`docs/superpowers/reviewer_prompts/`).
- Findings ledger at `docs/validation_findings.md` records all
  divergences classified per F3 severity.
- `pyproject.toml` registers the `external` marker for Pillar B.
- `CLAUDE.md` updated with campaign pointers.

## Test plan

- [ ] `pytest tests/ -v` — full suite green.
- [ ] `python3 scripts/audit_public_coverage.py` shows 0 untested symbols
      at every tier.
- [ ] Findings ledger has a Resolution entry for every row.
- [ ] Spec §5 (Pillar B detail) updated for the next pillar.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

If `gh` is not authenticated or fails, post the diff to the user with `git diff master..HEAD --stat` and ask.

- [ ] **Step 12.8: Post the C2 checkpoint summary**

To the user, in the conversation:

```
Pillar A complete.

- Total tier-1 symbols: <N1>; new tests: <T1>; fix-now commits: <F1>.
- Total tier-2 symbols: <N2>; new tests: <T2>; xfail-and-continue: <X2>.
- Total tier-3 symbols: <N3>; new tests: <T3>.
- Findings ledger entries: <L>.
- PR: <URL>.

Awaiting your approval before opening Pillar B branch.
```

---

## Self-Review (run by author after writing this plan)

Cross-checked against `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`:

| Spec section | Plan task | Status |
|---|---|---|
| §3 — branch + ledger artifacts | Tasks 0, 2 | ✓ |
| §4.1 — tiered bar | Tasks 5, 6, 8, 10 (per-tier) | ✓ |
| §4.2 — coverage auditor | Task 3 | ✓ |
| §4.3 — test-writing rules | Tasks 6.1, 8.1, 10.1 (templates) | ✓ |
| §4.4 — Pillar A exit criteria | Task 12 (final audit, ledger, PR) | ✓ |
| §8 — reviewer loop (R4) | Tasks 7, 9, 11 | ✓ |
| §9 — F3 severity | Tasks 5.4, 6.1, 8.3, 10.3 (per-task application) | ✓ |
| §10 — checkpoints (C2/C4) | Tasks 7, 9, 11 (C4 triggers); Task 12.8 (C2) | ✓ |
| §11 — findings ledger schema | Task 2 (skeleton); Tasks 7, 9, 11 (rows) | ✓ |
| §12 — public datasets / memory pre-flight | Pillar B (deferred); referenced in reviewer prompt §1.2 | ✓ |
| §13 — campaign exit (Pillar A subset) | Task 12 (CLAUDE.md, marker, Pillar B detail) | ✓ |

No placeholders left in the plan: every step has runnable code, exact paths, exact commands, and concrete templates. The audit-driven task bodies (6.3, 8.3, 10.3) are intentionally templated because the worklist is data — but the per-symbol pattern is fully specified with worked examples (5.2, 6.2, 8.2, 10.2).

Type / signature consistency: `eigendecompose` accessed via both attribute (`out.eigenvalues`) and tuple (`out[0]`) in Task 5.2 to defend against either contract; this is intentional.
