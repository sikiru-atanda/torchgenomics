#!/usr/bin/env python3
"""Render F1-F7 from the harness outputs into paper/reproducibility/output/.

Discovers figure-render functions registered via
paper.reproducibility.render_figures.register("F<N>"). Each render_fn
receives the OUTPUT directory and returns ``(output_path, numeric_dict)``.
The numeric_dict is aggregated into ``manifest.json`` which is the
authoritative record of every number cited in the paper.

Tier 4 D2 authors one renderer per figure (F1-F7) under
paper/reproducibility/render_figures/. This driver is the registry sink.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
OUTPUT_DIR = REPRO_ROOT / "output"
MANIFEST = REPRO_ROOT / "manifest.json"

# Ensure the render_figures package is importable when run from the repo root.
REPO_ROOT = REPRO_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        render_figures = importlib.import_module(
            "paper.reproducibility.render_figures",
        )
    except ImportError as exc:
        print(f"[render] ERROR: cannot import render_figures package: {exc}")
        print("[render] Tier 4 D2 must author the render functions first.")
        return 1

    registry = getattr(render_figures, "REGISTRY", {})
    if not registry:
        print("[render] No figure renderers registered.")
        print("[render] Tier 4 D2 must call @register(\"F<N>\") on each renderer.")
        return 1

    manifest: dict = {}
    for fig_id in sorted(registry):
        render_fn = registry[fig_id]
        print(f"[render] ==> {fig_id}")
        try:
            out_path, numeric_result = render_fn(OUTPUT_DIR)
        except Exception as exc:  # noqa: BLE001 — surface any renderer failure
            print(f"[render] {fig_id} FAILED: {exc}")
            manifest[fig_id] = {"error": str(exc)}
            continue
        manifest[fig_id] = {
            "file": str(out_path.relative_to(REPRO_ROOT))
            if isinstance(out_path, pathlib.Path)
            else str(out_path),
            "numbers": numeric_result,
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[render] wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
