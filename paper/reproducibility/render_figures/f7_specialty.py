"""F7 — Specialty models (eight panels: six external-reference + two internal-only).

Reads ``results/agreement.json`` from each of the six Tier 3 specialty
fixtures (survival, random regression, within-family, threshold-linear,
knockoff, OCF) and renders a 4x2 grid of panels:

  - **C1 Survival** (Cox PH frailty vs R ``coxme``): observed beta Pearson
    correlation, Spearman, causal-detection counts.
  - **C2 Random regression** (TG ``RandomRegressionLMM`` vs lme4):
    7 longitudinal checks, all PASS at observed-then-floored gates;
    beta_g_P1 agrees to 10 significant figures.
  - **C3 Within-family** (TG ``WithinFamilyLMM`` vs paper-OLS, Young 2022):
    median |delta beta_direct| at 1.7e-10 on planted causal SNPs.
  - **C4 Threshold-linear** (TG ``ThresholdLinearModel`` vs BLUPF90+
    ``gibbsf90+``): 4 checks at observed-then-floored gates.
  - **C5 Knockoff FDR** (TG ``KnockoffLMM`` vs R ``knockoff`` package):
    both methods individually control FDR at target 0.2; TG block FDR =
    0.0, R per-SNP FDR = 0.1548.
  - **C6 OCF DML** (TG ``OCFLMM`` vs hand-coded DML2 reference,
    Chernozhukov 2018): coverage 0.41 (TG) vs 0.91 (reference). This
    is the known F3 #4 divergence; both checks fail their thresholds.

Plus two internal-only panels (no external reference; scaffold-only,
flagged with a distinct dashed orange border per spec section 10.6):

  - **GU** Genotype-Uncertainty LMM (Phase 28).
  - **LRO** Leave-Region-Out LMM (Phase 29).

Tier 4 D2.7.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; reproducible regardless of display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]



EXTERNAL_PANELS: list[tuple[str, str, str, pathlib.Path]] = [
    (
        "C1",
        "Survival (Cox PH frailty)",
        "TG SurvivalGLMM PQL+score vs R coxme Wald",
        REPO_ROOT / "validation/specialty/survival/results/agreement.json",
    ),
    (
        "C2",
        "Random regression (longitudinal)",
        "TG RandomRegressionLMM vs lme4",
        REPO_ROOT / "validation/specialty/rr/results/agreement.json",
    ),
    (
        "C3",
        "Within-family (sibling)",
        "TG WithinFamilyLMM vs paper-OLS (Young 2022)",
        REPO_ROOT / "validation/specialty/family/results/agreement.json",
    ),
    (
        "C4",
        "Threshold-linear (ordinal)",
        "TG ThresholdLinearModel vs BLUPF90+ gibbsf90+",
        REPO_ROOT / "validation/specialty/threshold/results/agreement.json",
    ),
    (
        "C5",
        "Knockoff FDR",
        "TG KnockoffLMM vs R knockoff package",
        REPO_ROOT / "validation/specialty/knockoff/results/agreement.json",
    ),
    (
        "C6",
        "OCF (DML cross-fit)",
        "TG OCFLMM vs hand-coded DML2 reference",
        REPO_ROOT / "validation/specialty/ocf/results/agreement.json",
    ),
]


INTERNAL_PANELS: list[tuple[str, str, str, str]] = [
    (
        "GU",
        "Genotype-Uncertainty LMM",
        "Phase 28",
        "Dosage-variance-corrected score test.",
    ),
    (
        "LRO",
        "Leave-Region-Out LMM",
        "Phase 29",
        "Block-level LOCO.",
    ),
]

COLOR_PASS = "#5cb85c"
COLOR_FAIL = "#d9534f"
COLOR_NEUTRAL = "#777777"
COLOR_INTERNAL_BORDER = "#e8871a"



def _load_one(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": None, "checks": [], "extras": {}}
    return json.loads(path.read_text())


def _fmt_observed(value: Any) -> str:
    """Pretty-print an observed value for panel annotation."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "NaN"
        absv = abs(value)
        if absv == 0.0:
            return "0"
        if absv >= 1.0:
            return f"{value:.3g}"
        if absv >= 1e-2:
            return f"{value:.3g}"
        return f"{value:.2e}"
    s = str(value)
    if len(s) > 18:
        return s[:15] + "..."
    return s



def _draw_external_panel(
    ax,
    panel_id: str,
    title: str,
    subtitle: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Draw one external-reference panel: pass/fail check stack + observed values."""
    checks = payload.get("checks", []) or []
    n_pass = sum(1 for c in checks if c.get("passed"))
    n_total = len(checks)
    overall_passed = bool(payload.get("passed"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    title_color = COLOR_PASS if overall_passed else COLOR_FAIL
    ax.text(
        0.5, 0.96,
        f"{panel_id}  {title}",
        ha="center", va="top",
        fontsize=10, fontweight="bold",
    )
    ax.text(
        0.5, 0.90,
        subtitle,
        ha="center", va="top",
        fontsize=7, color=COLOR_NEUTRAL, style="italic",
    )
    ax.text(
        0.5, 0.84,
        f"{n_pass}/{n_total} checks pass",
        ha="center", va="top",
        fontsize=9, color=title_color, fontweight="bold",
    )


    if checks:
        n_rows = max(1, len(checks))
        row_y = np.linspace(0.74, 0.12, n_rows)
        for c, y in zip(checks, row_y):
            passed = bool(c.get("passed"))
            color = COLOR_PASS if passed else COLOR_FAIL
            marker = "PASS" if passed else "FAIL"
            name = str(c.get("name", ""))
            if len(name) > 38:
                name = name[:35] + "..."
            observed = _fmt_observed(c.get("observed"))
            ax.text(
                0.04, y, marker,
                ha="left", va="center",
                fontsize=6.5, fontweight="bold", color=color,
            )
            ax.text(
                0.17, y, name,
                ha="left", va="center",
                fontsize=6.5, color="#333333",
            )
            ax.text(
                0.96, y, observed,
                ha="right", va="center",
                fontsize=6.5, color=color, fontweight="bold",
                family="monospace",
            )

    if panel_id == "C6" and not overall_passed:
        ax.text(
            0.5, 0.04,
            "F3 #4: TG coverage 0.41 < ref 0.91 (documented)",
            ha="center", va="bottom",
            fontsize=6.5, color=COLOR_FAIL, style="italic",
        )
    else:
        ax.text(
            0.5, 0.04,
            "observed-then-floored gates",
            ha="center", va="bottom",
            fontsize=6.5, color=COLOR_NEUTRAL, style="italic",
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(1.5)

    return {
        "passed_overall": payload.get("passed"),
        "n_checks": n_total,
        "n_pass": n_pass,
        "checks": [
            {
                "name": c.get("name"),
                "observed": c.get("observed"),
                "threshold": c.get("threshold"),
                "passed": c.get("passed"),
            }
            for c in checks
        ],
    }



def _draw_internal_panel(
    ax,
    panel_id: str,
    title: str,
    phase: str,
    body: str,
) -> dict[str, Any]:
    """Draw one internal-only panel: placeholder with distinct dashed border."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.5, 0.94,
        f"{panel_id}  {title}",
        ha="center", va="top",
        fontsize=10, fontweight="bold",
    )
    ax.text(
        0.5, 0.86,
        phase,
        ha="center", va="top",
        fontsize=7, color=COLOR_NEUTRAL, style="italic",
    )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.10, 0.30), 0.80, 0.42,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=COLOR_INTERNAL_BORDER,
            facecolor="#fff7ee",
        )
    )
    ax.text(
        0.5, 0.61,
        "Internal validation",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=COLOR_INTERNAL_BORDER,
    )
    ax.text(
        0.5, 0.51,
        "(no external reference)",
        ha="center", va="center",
        fontsize=8, color=COLOR_INTERNAL_BORDER, style="italic",
    )
    ax.text(
        0.5, 0.40,
        body,
        ha="center", va="center",
        fontsize=7, color="#555555",
    )

    ax.text(
        0.5, 0.20,
        "scaffold-only; see Discussion section 10.6",
        ha="center", va="top",
        fontsize=6.5, color=COLOR_NEUTRAL, style="italic",
    )
    ax.text(
        0.5, 0.10,
        "internal-consistency simulator pending",
        ha="center", va="top",
        fontsize=6.5, color=COLOR_NEUTRAL,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_INTERNAL_BORDER)
        spine.set_linewidth(1.5)
        spine.set_linestyle("--")

    return {
        "internal_only": True,
        "phase": phase,
        "external_reference": None,
        "note": "scaffold; no external reference (spec section 10.6)",
    }



@register("F7")
def render_f7(output_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    flat_axes = axes.flatten()

    manifest: dict[str, Any] = {
        "panels": {},
        "n_external_panels": len(EXTERNAL_PANELS),
        "n_internal_panels": len(INTERNAL_PANELS),
        "n_total_panels": len(EXTERNAL_PANELS) + len(INTERNAL_PANELS),
    }

    for i, (panel_id, title, subtitle, path) in enumerate(EXTERNAL_PANELS):
        ax = flat_axes[i]
        payload = _load_one(path)
        manifest["panels"][panel_id] = _draw_external_panel(
            ax, panel_id, title, subtitle, payload,
        )

    for j, (panel_id, title, phase, body) in enumerate(INTERNAL_PANELS):
        ax = flat_axes[len(EXTERNAL_PANELS) + j]
        manifest["panels"][panel_id] = _draw_internal_panel(
            ax, panel_id, title, phase, body,
        )

    fig.suptitle(
        "F7 - Specialty models: six external-reference panels + "
        "two internal-only (dashed-orange border)",
        fontsize=12, y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "green = PASS . red = FAIL (F3 documented) . dashed orange = internal only "
        "(no external reference, scaffold; spec section 10.6) . Genome Biology Methods Figure 7",
        ha="center", fontsize=7.5, color="#555555",
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path = output_dir / "F7.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    total_checks = sum(
        m.get("n_checks", 0)
        for m in manifest["panels"].values()
        if not m.get("internal_only")
    )
    total_pass = sum(
        m.get("n_pass", 0)
        for m in manifest["panels"].values()
        if not m.get("internal_only")
    )
    manifest["external_total_checks"] = total_checks
    manifest["external_total_pass"] = total_pass

    return out_path, manifest
