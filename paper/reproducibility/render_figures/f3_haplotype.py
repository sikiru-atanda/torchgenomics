"""F3 -- Haplotype layer (LD-block design + haplotype GWAS + novel hit detail).

Three-panel summary of the TorchGWAS haplotype stack for the Genome
Biology Methods manuscript (Section 3, Results: Haplotype layer):

  Panel A -- 13 LD-block-design methods on MDP vs PLINK 1.9 --blocks
            (Gabriel).  Source: validation/external/plink2/ once the
            PLINK Gabriel reference and per-method block counts land;
            the method list is sourced from torchgwas.ld (4 classical
            + 5 novel + 3 literature + 1 diagnostic).

  Panel B -- 9 haplotype GWAS methods on SoyMD.  HTR / window / block /
            SKAT from HaplotypeGWAS plus the 5 novel methods PCHT,
            HHCT, HSKAT, HapGxE, BayesHap from
            torchgwas.models.haplotype_novel.  The haplo.stats
            agreement at validation/external/hapref/results/agreement.json
            is the only concrete row available at D2.3 authoring time;
            the other eight rows are scaffold placeholders pending
            Tier 1 harnesses.

  Panel C -- Per-haplotype beta detail at the real-data hit (haplo.stats
            chr1 5-SNP MDP window -- the same locus the F2 hapref row
            passes on).  When BayesHap / PCHT / HHCT per-haplotype PIPs
            land in a future harness, they replace this panel bar chart;
            until then we render the haplo.stats vs TorchGWAS per-
            haplotype beta alignment as the concrete signal.

Panels A and B are *scaffold-flagged* -- they render layout, method
names, and the single concrete agreement value (hapref) so reviewers
can audit the structure, but the per-method block-count and per-method
p-value columns are marked "scaffold pending" until the corresponding
harness data lands.  This is the spec section 10.4 observed-then-floored
protocol applied to figure assembly: never claim a number we do not
have, but render the layout so the missing data is visible.

Tier 4 D2.3.
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


# Panel A 13 LD-block-design methods (torchgwas.ld).
LD_BLOCK_METHODS = [
    ("gabriel",         "classical"),
    ("four_gamete",     "classical"),
    ("spine",           "classical"),
    ("r2",              "classical"),
    ("gwas_aligned",    "novel"),
    ("uncertainty",     "novel"),
    ("cross_pop",       "novel"),
    ("graphical",       "novel"),
    ("changepoint",     "novel"),
    ("big_ld",          "literature"),
    ("cc_graph",        "literature"),
    ("dp_optimize",     "literature"),
    ("wall_pritchard",  "diagnostic"),
]

CATEGORY_COLORS = {
    "classical":  "#4a7ab8",
    "novel":      "#d97a2c",
    "literature": "#5cb85c",
    "diagnostic": "#9b59b6",
}

PLINK_BLOCKS_AGREEMENT = REPO_ROOT / "validation/external/plink2/results/agreement.json"


# Panel B 9 haplotype GWAS methods.
HAPLOTYPE_GWAS_METHODS = [
    ("HTR-block (F-test)",   "core",  REPO_ROOT / "validation/external/hapref/results/agreement.json"),
    ("HTR-block (LRT)",      "core",  None),
    ("HTR-window (F-test)",  "core",  None),
    ("SKAT-block",           "core",  None),
    ("PCHT",                 "novel", None),
    ("HHCT",                 "novel", None),
    ("HSKAT",                "novel", None),
    ("HapGxE",               "novel", None),
    ("BayesHap",             "novel", None),
]

FAMILY_COLORS = {
    "core":  "#4a7ab8",
    "novel": "#d97a2c",
}

HAPREF_AGREEMENT = REPO_ROOT / "validation/external/hapref/results/agreement.json"


# Helpers

def _load_json_or_none(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _fmt_observed(value):
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
    return s if len(s) <= 18 else s[:15] + "..."


def _render_panel_a(ax):
    """Panel A: 13 LD-block-design methods on MDP vs PLINK Gabriel."""
    payload = _load_json_or_none(PLINK_BLOCKS_AGREEMENT)
    per_method = {}
    scaffold_only = payload is None

    n_methods = len(LD_BLOCK_METHODS)
    y_pos = np.arange(n_methods)

    widths = np.full(n_methods, 0.5, dtype=float)
    annotations = []
    bar_colors = []

    for i, (slug, cat) in enumerate(LD_BLOCK_METHODS):
        bar_colors.append(CATEGORY_COLORS[cat])
        entry = {"category": cat}
        if payload is not None and isinstance(payload.get("per_method"), dict):
            row = payload["per_method"].get(slug)
            if isinstance(row, dict):
                obs = row.get("agreement")
                entry["observed"] = obs
                entry["passed"] = bool(row.get("passed"))
                if isinstance(obs, (int, float)) and math.isfinite(obs):
                    widths[i] = float(obs)
                annotations.append(_fmt_observed(obs))
            else:
                annotations.append("scaffold pending")
                entry["scaffold"] = True
        else:
            annotations.append("scaffold pending")
            entry["scaffold"] = True
        per_method[slug] = entry

    ax.barh(y_pos, widths, color=bar_colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([slug for slug, _ in LD_BLOCK_METHODS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("agreement vs PLINK 1.9 Gabriel\n(1.0 = perfect; scaffold = layout only)", fontsize=8)
    title_suffix = " (scaffold pending)" if scaffold_only else ""
    ax.set_title(f"A: 13 LD-block-design methods on MDP{title_suffix}", fontsize=10)

    for i, annot in enumerate(annotations):
        ax.text(min(widths[i], 1.0) + 0.02, i, annot,
                va="center", ha="left", fontsize=6.5, color="#333333")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=cat)
               for cat, color in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    return {
        "n_methods": n_methods,
        "scaffold_only": scaffold_only,
        "reference": "PLINK 1.9 --blocks (Gabriel)",
        "data_source": str(PLINK_BLOCKS_AGREEMENT.relative_to(REPO_ROOT)),
        "per_method": per_method,
    }


def _render_panel_b(ax):
    """Panel B: 9 haplotype GWAS methods on SoyMD."""
    per_method = {}
    n_methods = len(HAPLOTYPE_GWAS_METHODS)
    y_pos = np.arange(n_methods)
    widths = np.full(n_methods, 0.5, dtype=float)
    annotations = []
    bar_colors = []
    real_rows = 0

    for i, (label, family, source) in enumerate(HAPLOTYPE_GWAS_METHODS):
        bar_colors.append(FAMILY_COLORS[family])
        entry = {"family": family}
        if source is not None and source.exists():
            payload = _load_json_or_none(source)
            if payload is not None:
                checks = payload.get("checks", []) or []
                pval_check = None
                for c in checks:
                    if "global F-test p" in str(c.get("name", "")):
                        pval_check = c
                        break
                if pval_check is None:
                    pval_check = checks[0] if checks else None
                if pval_check is not None:
                    obs = pval_check.get("observed")
                    thr = pval_check.get("threshold")
                    passed = bool(pval_check.get("passed"))
                    if isinstance(obs, (int, float)) and math.isfinite(obs):
                        agreement = max(0.0, min(1.0, 1.0 - float(obs)))
                        widths[i] = agreement
                        status = "PASS" if passed else "FAIL"
                        annotations.append(f"{status}: rel|dp|={obs:.3g} (<= {thr})")
                    else:
                        annotations.append(_fmt_observed(obs))
                    entry.update({
                        "observed": obs,
                        "threshold": thr,
                        "passed": passed,
                        "check_name": pval_check.get("name"),
                        "source": str(source.relative_to(REPO_ROOT)),
                    })
                    real_rows += 1
                else:
                    annotations.append("scaffold pending")
                    entry["scaffold"] = True
            else:
                annotations.append("scaffold pending")
                entry["scaffold"] = True
        else:
            annotations.append("scaffold pending (see hapref harness)")
            entry["scaffold"] = True
        per_method[label] = entry

    ax.barh(y_pos, widths, color=bar_colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _, _ in HAPLOTYPE_GWAS_METHODS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("agreement on chr1 MDP window\n(1 = perfect; bars = 1 - rel.|dp|)", fontsize=8)
    ax.set_title(
        f"B: 9 haplotype GWAS methods on SoyMD\n({real_rows}/{n_methods} concrete; rest scaffold pending)",
        fontsize=10,
    )

    for i, annot in enumerate(annotations):
        ax.text(min(widths[i], 1.0) + 0.02, i, annot,
                va="center", ha="left", fontsize=6.5, color="#333333")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=family)
               for family, color in FAMILY_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    return {
        "n_methods": n_methods,
        "n_concrete_rows": real_rows,
        "scaffold_only": real_rows == 0,
        "per_method": per_method,
    }


def _render_panel_c(ax):
    """Panel C: per-haplotype beta / EM frequency at the real-data hit."""
    payload = _load_json_or_none(HAPREF_AGREEMENT)
    if payload is None or not payload.get("alignment"):
        ax.text(
            0.5, 0.5,
            "scaffold pending\n(BayesHap / PCHT / HHCT per-haplotype PIPs)\n"
            "see hapref harness when results land",
            ha="center", va="center", fontsize=10, color="#666666",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("C: Novel haplotype methods on real-data hit (scaffold)", fontsize=10)
        return {
            "scaffold_only": True,
            "data_source": str(HAPREF_AGREEMENT.relative_to(REPO_ROOT)),
        }

    rows = payload["alignment"].get("rows", []) or []
    labels = [r.get("tg_label", "?") for r in rows]
    r_beta = np.array([float(r.get("r_beta") or 0.0) for r in rows])
    tg_beta = np.array([float(r.get("tg_beta") or 0.0) for r in rows])

    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, r_beta, width, color="#4a7ab8", edgecolor="black",
           linewidth=0.4, label="haplo.stats beta")
    ax.bar(x + width / 2, tg_beta, width, color="#d97a2c", edgecolor="black",
           linewidth=0.4, label="TorchGWAS beta")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("haplotype effect beta", fontsize=8)
    extras = payload.get("extras", {}) or {}
    window_id = extras.get("window_id", "chr1 MDP window")
    snp_ids = extras.get("snp_ids", "")
    ax.set_title(f"C: Per-haplotype beta at {window_id}", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    max_rel_dbeta_check = None
    for c in payload.get("checks", []) or []:
        if "Delta beta" in str(c.get("name", "")):
            max_rel_dbeta_check = c
            break
    if max_rel_dbeta_check is not None:
        obs_str = _fmt_observed(max_rel_dbeta_check.get("observed"))
        thr_str = _fmt_observed(max_rel_dbeta_check.get("threshold"))
        ax.text(
            0.02, 0.98,
            f"max rel |dbeta| = {obs_str} (<= {thr_str})",
            ha="left", va="top", fontsize=7, color="#333333",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f3f3f3", edgecolor="none"),
        )

    return {
        "scaffold_only": False,
        "window_id": window_id,
        "snp_ids": snp_ids,
        "n_haplotypes": len(labels),
        "labels": labels,
        "r_beta": r_beta.tolist(),
        "tg_beta": tg_beta.tolist(),
        "max_rel_dbeta": (
            max_rel_dbeta_check.get("observed") if max_rel_dbeta_check else None
        ),
        "data_source": str(HAPREF_AGREEMENT.relative_to(REPO_ROOT)),
        "novel_overlay": "scaffold pending (BayesHap PIP / PCHT tree / HHCT cluster)",
    }


@register("F3")
def render_f3(output_dir):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 5))

    panel_a = _render_panel_a(ax_a)
    panel_b = _render_panel_b(ax_b)
    panel_c = _render_panel_c(ax_c)

    n_concrete_b = panel_b.get("n_concrete_rows", 0)
    n_methods_a = panel_a.get("n_methods", 0)
    n_methods_b = panel_b.get("n_methods", 0)
    fig.suptitle(
        f"F3 Haplotype layer  |  {n_methods_a} LD-block methods  |  "
        f"{n_methods_b} haplotype GWAS methods  |  "
        f"{n_concrete_b}/{n_methods_b} concrete rows (rest scaffold pending)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        "Genome Biology Methods Figure 3  |  scaffold-flagged rows render layout only; "
        "concrete numbers come from validation/external/hapref/results/agreement.json",
        ha="center", fontsize=7, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_path = output_dir / "F3.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "panels": {
            "A": panel_a,
            "B": panel_b,
            "C": panel_c,
        },
        "data_sources": [
            str(PLINK_BLOCKS_AGREEMENT.relative_to(REPO_ROOT)),
            str(HAPREF_AGREEMENT.relative_to(REPO_ROOT)),
            "validation/external/soymd/ (scaffold; SoyMD harness pending)",
        ],
        "n_ld_block_methods": n_methods_a,
        "n_haplotype_gwas_methods": n_methods_b,
        "scaffold_pending": {
            "panel_A": panel_a.get("scaffold_only", True),
            "panel_B_rows": [
                label for label, entry in panel_b.get("per_method", {}).items()
                if entry.get("scaffold")
            ],
            "panel_C_novel_overlay": panel_c.get("novel_overlay"),
        },
    }
    return out_path, manifest
