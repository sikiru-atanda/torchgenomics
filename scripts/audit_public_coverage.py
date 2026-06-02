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
import ast
import importlib
import inspect
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO / "torchgenomics"
TESTS_ROOT = REPO / "tests"

# Make `torchgenomics` importable when the package is not pip-installed
# (script may be invoked from any cwd; Python only auto-adds the script's
# own directory to sys.path, which is `scripts/`, not the repo root).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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


class _TestFileIndex:
    """Pre-parsed index of a single test file's imports and attribute accesses.

    Built once per file (in `_build_test_index`) and queried per symbol
    via `covers(module, symbol)`. The naive approach of re-parsing every
    test file for every public symbol scales as O(symbols * files) AST
    walks (~6 minutes on this repo); caching brings it to O(files).
    """

    __slots__ = (
        "from_imports",   # dict[module_str, set[real_symbol_name]]
        "module_aliases", # dict[module_str, set[local_binding_name]]
        "parent_aliases", # dict[(parent, short), set[local_binding_name]]
        "attr_accesses",  # dict[attr_name, set[base_local_name]]
        "dotted_attrs",   # set[tuple[str, ...]] of full attribute chains
    )

    def __init__(self):
        self.from_imports: dict[str, set[str]] = {}
        self.module_aliases: dict[str, set[str]] = {}
        self.parent_aliases: dict[tuple[str, str], set[str]] = {}
        self.attr_accesses: dict[str, set[str]] = {}
        self.dotted_attrs: set[tuple[str, ...]] = set()

    def covers(self, module: str, symbol: str) -> bool:
        # Case 1: `from <module> import ... <symbol> ...`
        if symbol in self.from_imports.get(module, ()):
            return True

        # Build the set of local names through which <module> is reachable.
        module_local_names = set(self.module_aliases.get(module, ()))
        if "." in module:
            parent, short = module.rsplit(".", 1)
            module_local_names |= self.parent_aliases.get((parent, short), set())

        # Case 2 / 3: `<local_name>.<symbol>` access.
        if module_local_names:
            base_names = self.attr_accesses.get(symbol, set())
            if base_names & module_local_names:
                return True

        # Case 2 (dotted): `import torchgenomics.linalg` (no alias) + a dotted
        # `torchgenomics.linalg.<symbol>` access — recorded as a chain.
        target = tuple(module.split(".") + [symbol])
        if target in self.dotted_attrs:
            return True

        return False


def _build_test_index(tree: ast.AST) -> _TestFileIndex:
    """Walk the AST once and record everything we'll need for symbol lookup."""
    idx = _TestFileIndex()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod is None:
                continue
            for alias in node.names:
                # `from <mod> import <name>` — store the real name (alias.name);
                # an `as` rename does not change which symbol is being imported.
                idx.from_imports.setdefault(mod, set()).add(alias.name)
                # Also track `from <parent> import <short> [as alias]` so we
                # can resolve `<short>.symbol` accesses against module
                # `<parent>.<short>`.
                idx.parent_aliases.setdefault(
                    (mod, alias.name), set()
                ).add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # `import <fullmod> [as alias]`: the full dotted path resolves
                # via the local binding (alias.asname when present, else the
                # top-level package name; we store both the full module path
                # and a dotted-chain entry below).
                full = alias.name
                bind = alias.asname or alias.name
                idx.module_aliases.setdefault(full, set()).add(bind)
        elif isinstance(node, ast.Attribute):
            # Record `<base>.<attr>` for fast attribute-access lookup.
            value = node.value
            if isinstance(value, ast.Name):
                idx.attr_accesses.setdefault(node.attr, set()).add(value.id)
            # Also record full dotted chains rooted in a Name, so that
            # `import torchgenomics.linalg` + `torchgenomics.linalg.eig(...)` works
            # even though the local binding is only `torchgenomics`.
            parts: list[str] = [node.attr]
            cur: ast.AST = value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                parts.reverse()
                idx.dotted_attrs.add(tuple(parts))
    return idx


def _build_all_test_indices() -> list[tuple[str, _TestFileIndex]]:
    """Parse every test file once and return (rel_path, index) pairs."""
    out: list[tuple[str, _TestFileIndex]] = []
    for p in TESTS_ROOT.rglob("test_*.py"):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(p))
        except SyntaxError:
            # A test file with a syntax error can't reliably be analyzed; skip.
            continue
        out.append((str(p.relative_to(REPO)), _build_test_index(tree)))
    return out


# Lazily populated; built once per process.
_TEST_INDICES: list[tuple[str, _TestFileIndex]] | None = None


def find_test_files_for_symbol(module: str, symbol: str) -> list[str]:
    """Return tests/ files that genuinely import or invoke `module.symbol`.

    A test file is counted as covering `module.symbol` only if at least one of:

      1. `from <module> import ... <symbol> ...` — exact ImportFrom match,
         where the symbol appears among the imported aliases (under its real
         name, not an `as` alias's local name).
      2. `import <module>` (or `import <module> as <alias>`) AND a separate
         `<module>.<symbol>` (or `<alias>.<symbol>`) attribute access.
      3. `from <module_parent> import ... <module_short_name> ...` AND
         `<module_short_name>.<symbol>` (or aliased) attribute access
         (e.g. `from torchgenomics import linalg` + `linalg.eigendecompose(...)`).

    Uses AST analysis (built once per file and cached) instead of substring
    or word-boundary regex matching, to avoid false positives on common
    names like ``Result`` / ``fit`` / ``score`` that may appear in
    docstrings or be imported from an unrelated module.
    """
    global _TEST_INDICES
    if _TEST_INDICES is None:
        _TEST_INDICES = _build_all_test_indices()
    return [rel for rel, idx in _TEST_INDICES if idx.covers(module, symbol)]


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
        except (ImportError, ModuleNotFoundError, OSError, AttributeError) as exc:
            print(f"WARN: skipping {modname}: {type(exc).__name__}: {exc}", file=sys.stderr)
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
            test_files = find_test_files_for_symbol(modname, symname)
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
