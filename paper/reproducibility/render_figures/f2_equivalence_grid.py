"""F2 — Reference-tool numerical equivalence grid.

Reads ``results/agreement.json`` from every Tier 1 reference-tool harness
and every Tier 3 specialty-model harness on the paper branch, plots a
per-tool × per-check pass/fail grid annotated with observed agreement
values. The figure is the headline numerical-evidence panel of the
paper (Genome Biology Methods §3 "Reference-tool equivalence").

Each cell encodes:
  - color = pass (green) / fail (red) per the harness's own observed-
    then-floored gate.
  - text  = the observed value at significant-figure precision.

Per the spec §10.4 observed-then-floored protocol, an agreement that
passes is annotated with its observed value; an agreement that fails
is annotated and flagged for F3 follow-up in
``docs/validation_findings.md``.

Tier 4 D2.2.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; reproducible regardless of $DISPLAY
import matplotlib.pyplot as plt
import numpy as np

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


# Order is the F2 row order: Tier 1 reference-tool harnesses first
# (alphabetical within tier), then Tier 3 specialty harnesses.
HARNESSES: list[tuple[str, pathlib.Path]] = [
    ("MetaXcan (TWAS)",    REPO_ROOT / "validation/external/metaxcan/results/agreement.json"),
    ("SMR + HEIDI",        REPO_ROOT / "validation/external/smr/results/agreement.json"),
    ("coloc R",            REPO_ROOT / "validation/external/coloc/results/agreement.json"),
    ("hyprcoloc R",        REPO_ROOT / "validation/external/hyprcoloc/results/agreement.json"),
    ("haplo.stats (hapref)", REPO_ROOT / "validation/external/hapref/results/agreement.json"),
    ("Cox PH frailty (coxme)", REPO_ROOT / "validation/specialty/survival/results/agreement.json"),
    ("Random regression (lme4)", REPO_ROOT / "validation/specialty/rr/results/agreement.json"),
    ("Within-family (paper-OLS)", REPO_ROOT / "validation/specialty/family/results/agreement.json"),
    ("Threshold-linear (BLUPF90+)", REPO_ROOT / "validation/specialty/threshold/results/agreement.json"),
    ("Knockoff FDR (R)",   REPO_ROOT / "validation/specialty/knockoff/results/agreement.json"),
    ("OCF DML",            REPO_ROOT / "validation/specialty/ocf/results/agreement.json"),
]


def _load_one(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": None, "checks": []}
    return json.loads(path.read_text())


def _fmt_observed(value: Any) -> str:
    """Pretty-print an observed value for grid annotation."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "NaN"
        absv = abs(value)
        if absv >= 1.0:
            return f"{value:.3g}"
        if absv >= 1e-2:
            return f"{value:.3g}"
        if absv == 0.0:
            return "0"
        return f"{value:.1e}"
    s = str(value)
    if len(s) > 18:
        return s[:15] + "..."
    return s


@register("F2")
def render_f2(output_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for label, path in HARNESSES:
        rows.append((label, _load_one(path)))

    # Build a (harness × check) matrix. The set of check names varies
    # per harness — we keep each harness's own check list in row order
    # and pad shorter rows with "—".
    max_checks = max(len(r[1].get("checks", [])) for r in rows)
    if max_checks == 0:
        max_checks = 1

    fig_height = max(4.0, 0.45 * len(rows) + 1.5)
    fig_width = max(9.0, 1.6 * max_checks + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_colors = np.full((len(rows), max_checks), np.nan)
    cell_texts: list[list[str]] = []
    check_names_per_row: list[list[str]] = []

    for i, (label, payload) in enumerate(rows):
        checks = payload.get("checks", []) or []
        row_text: list[str] = []
        row_names: list[str] = []
        for j in range(max_checks):
            if j < len(checks):
                c = checks[j]
                # Color: green=pass (1.0), red=fail (0.0)
                cell_colors[i, j] = 1.0 if c.get("passed") else 0.0
                row_text.append(
                    f"{c.get('name', '')}\n{_fmt_observed(c.get('observed'))}"
                )
                row_names.append(str(c.get("name", "")))
            else:
                cell_colors[i, j] = float("nan")
                row_text.append("—")
                row_names.append("—")
        cell_texts.append(row_text)
        check_names_per_row.append(row_names)

    # Colormap: NaN = grey, 0 = red, 1 = green
    cmap = matplotlib.colors.ListedColormap(["#d9534f", "#5cb85c"])
    cmap.set_bad("#e8e8e8")
    masked = np.ma.masked_invalid(cell_colors)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(rows)):
        for j in range(max_checks):
            txt = cell_texts[i][j]
            color = "white" if not np.isnan(cell_colors[i, j]) else "#555555"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=7, color=color, linespacing=1.1,
            )

    # Row labels
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xticks(range(max_checks))
    ax.set_xticklabels([f"check {j+1}" for j in range(max_checks)], fontsize=8)
    ax.set_title(
        "F2 — Reference-tool numerical equivalence (Tier 1 + Tier 3)\n"
        "green = PASS at observed-then-floored gate · red = FAIL (F3 documented)",
        fontsize=10,
    )

    # Footer with summary
    n_total = sum(len(r[1].get("checks", []) or []) for r in rows)
    n_pass = sum(
        sum(1 for c in (r[1].get("checks", []) or []) if c.get("passed"))
        for r in rows
    )
    fig.text(
        0.5, 0.01,
        f"{n_pass} / {n_total} checks pass · {len(rows)} harnesses · "
        f"Genome Biology Methods Figure 2",
        ha="center", fontsize=8, color="#555555",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = output_dir / "F2.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Build the manifest entry — exact numbers from each harness, so the
    # paper can quote them directly.
    manifest: dict[str, Any] = {
        "n_harnesses": len(rows),
        "n_total_checks": n_total,
        "n_pass_checks": n_pass,
        "per_harness": {},
    }
    for label, payload in rows:
        manifest["per_harness"][label] = {
            "passed_overall": payload.get("passed"),
            "checks": [
                {
                    "name": c.get("name"),
                    "observed": c.get("observed"),
                    "threshold": c.get("threshold"),
                    "passed": c.get("passed"),
                }
                for c in (payload.get("checks", []) or [])
            ],
        }
    return out_path, manifest
