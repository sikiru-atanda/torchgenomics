#!/usr/bin/env python3
"""Build a deduped, per-package worklist of untested public symbols
from docs/validation_findings/coverage_audit.json.

Dedup rule: a symbol is keyed by its canonical defining location
(`obj.__module__`, `obj.__qualname__`). The audit lists it under every
namespace path that re-exports it; we collapse those to one row per
canonical key, choosing the SHORTEST module path as the public-API
import location (since that's what users will actually `from <X> import`).

Usage:
    python3 scripts/build_tier_worklist.py --tier 1 --package linalg
    python3 scripts/build_tier_worklist.py --tier 1 --package models --output /tmp/worklist.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def canonical_key(modname: str, symbol: str):
    """Return (canonical_module, qualname) for the symbol object, or None
    if it cannot be resolved (e.g., import failure)."""
    try:
        mod = importlib.import_module(modname)
        obj = getattr(mod, symbol, None)
    except Exception:
        return None
    if obj is None:
        return None
    canonical_mod = getattr(obj, "__module__", modname) or modname
    qualname = getattr(obj, "__qualname__", symbol) or symbol
    return canonical_mod, qualname


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", default="docs/validation_findings/coverage_audit.json")
    p.add_argument("--tier", type=int, required=True)
    p.add_argument("--package", required=True,
                   help="Top-level package name (e.g., 'linalg' for torchgwas.linalg)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    audit_path = REPO / args.audit
    data = json.loads(audit_path.read_text())
    pkg_prefix = f"torchgwas.{args.package}"

    # Two-pass: first compute canonical key + OR'd has_direct_test across
    # ALL audit rows (any path), then filter to (tier, package, untested).
    # This avoids treating a namespace re-export as untested when the
    # canonical defining module has a direct test.
    canonical_tested: dict[tuple, bool] = {}
    for r in data["rows"]:
        key = canonical_key(r["module"], r["symbol"]) or (r["module"], r["symbol"])
        canonical_tested[key] = canonical_tested.get(key, False) or r["has_direct_test"]

    candidates = [
        r for r in data["rows"]
        if r["tier"] == args.tier
        and (r["module"] == pkg_prefix or r["module"].startswith(pkg_prefix + "."))
    ]

    # Dedup by canonical (module, qualname). Keep the row with the SHORTEST
    # module path as the "public" import location. Override has_direct_test
    # with the OR'd value across all re-export paths.
    by_key: dict[tuple, dict] = {}
    for r in candidates:
        key = canonical_key(r["module"], r["symbol"])
        if key is None:
            key = (r["module"], r["symbol"])  # fallback
        if key not in by_key or len(r["module"]) < len(by_key[key]["module"]):
            r2 = dict(r)
            r2["canonical_module"] = key[0]
            r2["canonical_qualname"] = key[1]
            r2["has_direct_test"] = canonical_tested.get(key, r["has_direct_test"])
            by_key[key] = r2

    # Drop tested-via-any-path rows now that we have the OR'd flag.
    deduped = sorted(
        (r for r in by_key.values() if not r["has_direct_test"]),
        key=lambda r: (r["module"], r["symbol"]),
    )

    out = {
        "tier": args.tier,
        "package": args.package,
        "total_raw_rows": len(candidates),
        "deduped_rows": len(deduped),
        "rows": deduped,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {len(deduped)} deduped rows ({len(candidates)} raw) to {args.output}")
    else:
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
