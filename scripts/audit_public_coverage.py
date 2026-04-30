#!/usr/bin/env python3
"""Audit public-symbol test coverage in torchgwas/.

Walks every module under torchgwas/, enumerates public symbols (via __all__
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
PKG_ROOT = REPO / "torchgwas"
TESTS_ROOT = REPO / "tests"

# Make `torchgwas` importable when the package is not pip-installed
# (script may be invoked from any cwd; Python only auto-adds the script's
# own directory to sys.path, which is `scripts/`, not the repo root).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TIER_MAP = {
    "torchgwas.models": 1,
    "torchgwas.linalg": 1,
    "torchgwas.stats": 1,
    "torchgwas.optim": 1,
    "torchgwas.scan": 1,
    "torchgwas.io": 2,
    "torchgwas.preprocess": 2,
    "torchgwas.ld": 2,
    "torchgwas.pgs": 2,
    "torchgwas.postgwas": 2,
    "torchgwas.multiomics": 2,
    "torchgwas.viz": 3,
    "torchgwas.annotate": 3,
    "torchgwas.results": 3,
    "torchgwas.cli": 3,
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
        if defining_module is not None and not defining_module.startswith("torchgwas"):
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
    """Yield (module_name, module_obj) for every submodule under torchgwas/."""
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_") and path.name != "__init__.py":
            continue
        rel = path.relative_to(REPO).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modname = ".".join(parts)
        if not modname.startswith("torchgwas"):
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
