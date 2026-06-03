"""F4 - Multi-omics integration (TWAS / SMR / coloc / mediation).

Three panels covering the section 5 multi-omics-integration results of
the TorchGenomics Genome Biology Methods paper:

  Panel A - simulated ground-truth (B1 fixture). Truth-side summary of
            the staged ``validation/multiomics/sim`` fixture: TWAS-Z
            truth, SMR beta truth, coloc PP.H4 truth, and mediation
            a*b truth across 5 causal mediators + 2 negative-control
            loci + 43 null mediators. Recovery panels (TG vs truth)
            are annotated as PENDING: Tier 4 D2.4 stages the fixture,
            the TG downstream calls (``twas_sumstat`` / ``smr_test`` /
            ``coloc_pairwise`` / ``mediate_lmm``) run in the Tier 4 D2
            execution sub-task, not in this renderer.

  Panel B - GTEx liver eQTL x UKB LDL sumstats worked example (B2
            fixture). Scatter of z_eqtl vs z_gwas across the 48
            harmonised SORT1 variants, colored by orientation,
            annotated with the Pearson correlation. This is the
            headline biological-result panel.

  Panel C - plant multi-omics worked example (B3 fixture). 50 Arabi-
            dopsis ecotypes x 100 SNP dosages x 20 expression log-FPKM
            x FT10 flowering-time phenotype. Cross-correlation heatmap
            of expression columns vs FT10 (top-k shown), annotated
            with the three-way sample-ID intersection.

Tier 4 D2.4. Mirrors the structure of ``f2_equivalence_grid.py``.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive; reproducible regardless of DISPLAY
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

TRUTH_PATH = REPO_ROOT / "validation/multiomics/sim/fixtures/truth.json"
GTEX_UKB_PATH = REPO_ROOT / "validation/multiomics/gtex_ukb/fixtures/aligned.parquet"
PLANT_PATH = REPO_ROOT / "validation/multiomics/plant/fixtures/aligned.parquet"


# ---------------------------------------------------------------------------
# Panel A - simulated ground-truth (B1 fixture, truth-side only).
# ---------------------------------------------------------------------------

def _load_truth() -> dict[str, Any]:
    """Load the simulated multi-omics truth fixture (B1)."""
    if not TRUTH_PATH.exists():
        return {}
    return json.loads(TRUTH_PATH.read_text())


def _render_panel_a(ax: plt.Axes) -> dict[str, Any]:
    """Truth-side composite for Panel A.

    Renders a four-row strip plot of the four ground-truth quantities
    (TWAS z, SMR beta, coloc PP.H4, mediation a*b) across all m
    simulated mediators, with causal mediators highlighted in orange
    and negative-control mediators in dotted red. This keeps Panel A a
    single matplotlib Axes (per the ``fig, axes = plt.subplots(1, 3)``
    layout from the spec) while still surfacing every truth-vector the
    spec calls out for section 5 F4 panel A.

    A clear annotation declares that the TG-recovery comparison is
    pending Tier 4 D2 execution.
    """
    truth = _load_truth()
    if not truth:
        ax.text(
            0.5, 0.5,
            "validation/multiomics/sim/fixtures/truth.json\n"
            "not found - Tier 4 B1 fixture missing",
            ha="center", va="center", fontsize=8, color="#a94442",
            transform=ax.transAxes,
        )
        ax.set_title("A - simulated ground-truth (MISSING)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return {"status": "missing"}

    twas = np.asarray(truth.get("twas_z_truth", []), dtype=float)
    smr = np.asarray(truth.get("smr_beta_truth", []), dtype=float)
    coloc = np.asarray(truth.get("coloc_pp_h4_truth", []), dtype=float)
    med = np.asarray(truth.get("mediation_ab_truth", []), dtype=float)

    causal_idx = list(truth.get("causal_mediator_idx", []))
    neg_idx = list(truth.get("neg_control_mediator_idx", []))
    n_total = int(truth.get("m", len(twas)))
    n_causal = len(causal_idx)
    n_neg = len(neg_idx)

    # Stack the four truth-vectors into a 4-by-m strip plot. Each row
    # is one method's truth-side quantity; each column is a mediator.
    # The value is mapped to color intensity via a diverging colormap
    # so the planted-causal spikes pop out.
    rows: list[tuple[str, np.ndarray]] = []
    if twas.size:
        rows.append(("TWAS z (truth)", twas))
    if smr.size:
        rows.append(("SMR beta (truth)", smr))
    if coloc.size:
        rows.append(("coloc PP.H4 (truth)", coloc))
    if med.size:
        rows.append(("mediation a*b (truth)", med))

    if not rows:
        ax.text(0.5, 0.5, "no truth vectors in truth.json",
                ha="center", va="center", fontsize=9,
                transform=ax.transAxes)
        ax.set_title("A - simulated ground-truth", fontsize=10)
        return {"status": "empty"}

    # Per-row max-abs normalization so the four scales are comparable
    # in a single strip plot (units differ row-to-row; a single
    # colorbar would mislead). Raw values are exposed via the manifest.
    width = max(len(r[1]) for r in rows)
    img = np.full((len(rows), width), np.nan, dtype=float)
    for i, (_, vec) in enumerate(rows):
        max_abs = max(float(np.max(np.abs(vec))), 1e-12)
        img[i, : len(vec)] = vec / max_abs

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e8e8e8")
    masked = np.ma.masked_invalid(img)
    ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0,
              aspect="auto", interpolation="nearest")

    # Highlight causal + negative-control columns with vertical guides.
    for j in causal_idx:
        ax.axvline(j, color="#fa6900", linewidth=0.6, alpha=0.85)
    for j in neg_idx:
        ax.axvline(j, color="#a94442", linewidth=0.6, alpha=0.6,
                   linestyle=":")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xticks([0, width - 1])
    ax.set_xticklabels(["G000", f"G{width - 1:03d}"], fontsize=7)
    ax.set_xlabel("mediator (gene index)", fontsize=8)
    ax.set_title(
        f"A - simulated ground-truth (m={n_total} genes; "
        f"orange = {n_causal} causal . dotted red = "
        f"{n_neg} neg-ctrl)",
        fontsize=9,
    )

    # Pending-recovery annotation, per the spec - Tier 4 D2.4 renders
    # truth-side only; the head-to-head recovery panel lands in the
    # downstream execution sub-task.
    ax.text(
        0.5, -0.32,
        "TG recovery panel pending: run twas_sumstat / smr_test / "
        "coloc_pairwise / mediate_lmm against this staged fixture.",
        ha="center", va="center", fontsize=7, color="#555555",
        transform=ax.transAxes, wrap=True,
    )

    return {
        "n_total_mediators": int(n_total),
        "n_causal": int(n_causal),
        "n_neg_control": int(n_neg),
        "n_null": int(max(0, n_total - n_causal - n_neg)),
        "causal_mediator_ids": list(truth.get("causal_mediator_ids", [])),
        "neg_control_mediator_ids": list(
            truth.get("neg_control_mediator_ids", [])
        ),
        "twas_z_truth_summary": {
            "min": float(np.min(twas)) if twas.size else None,
            "max": float(np.max(twas)) if twas.size else None,
            "abs_max_idx": int(np.argmax(np.abs(twas))) if twas.size else None,
        },
        "smr_beta_truth_nonzero": int(np.count_nonzero(smr)) if smr.size else 0,
        "coloc_pp_h4_truth_nonzero": int(np.count_nonzero(coloc)) if coloc.size else 0,
        "mediation_ab_truth_nonzero": int(np.count_nonzero(med)) if med.size else 0,
        "recovery_floors": truth.get("recovery_floors", {}),
        "tg_recovery_status": "pending: Tier 4 D2 execution sub-task",
    }


# ---------------------------------------------------------------------------
# Panel B - GTEx liver eQTL x UKB LDL sumstats (SORT1).
# ---------------------------------------------------------------------------

def _render_panel_b(ax: plt.Axes) -> dict[str, Any]:
    """SORT1 worked-example scatter (B2 fixture)."""
    if not GTEX_UKB_PATH.exists():
        ax.text(0.5, 0.5,
                "validation/multiomics/gtex_ukb/fixtures/aligned.parquet\n"
                "not found - Tier 4 B2 fixture missing",
                ha="center", va="center", fontsize=8, color="#a94442",
                transform=ax.transAxes)
        ax.set_title("B - SORT1 worked example (MISSING)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return {"status": "missing"}

    df = pd.read_parquet(GTEX_UKB_PATH)
    n_variants = int(len(df))

    if not {"z_eqtl", "z_gwas"}.issubset(df.columns):
        ax.text(0.5, 0.5,
                "aligned.parquet missing z_eqtl / z_gwas\n"
                f"columns={list(df.columns)}",
                ha="center", va="center", fontsize=7, color="#a94442",
                transform=ax.transAxes)
        ax.set_title("B - SORT1 worked example (schema error)", fontsize=10)
        return {"status": "schema_error", "columns": list(df.columns)}

    z_eqtl = df["z_eqtl"].to_numpy(dtype=float)
    z_gwas = df["z_gwas"].to_numpy(dtype=float)

    # Pearson r on the harmonized (post sign-flip) z-scores. With
    # orientation honoured this matches the published-fixture
    # reference value of ~ -0.968 documented in
    # validation/multiomics/gtex_ukb/.
    if z_eqtl.size >= 2:
        pearson_r = float(np.corrcoef(z_eqtl, z_gwas)[0, 1])
    else:
        pearson_r = float("nan")

    orient = (df["orientation"].astype(str)
              if "orientation" in df.columns
              else pd.Series(["unknown"] * n_variants))
    palette = {
        "direct": "#2c7fb8",
        "flipped": "#d95f0e",
        "unknown": "#888888",
    }
    for label in pd.unique(orient):
        mask = (orient == label).to_numpy()
        ax.scatter(
            z_eqtl[mask], z_gwas[mask],
            s=22, alpha=0.85,
            color=palette.get(str(label), "#888888"),
            edgecolor="white", linewidth=0.4,
            label=f"{label} (n={int(mask.sum())})",
        )

    # OLS regression line for visual reference.
    if (z_eqtl.size >= 2
            and np.isfinite(z_eqtl).all()
            and np.isfinite(z_gwas).all()):
        slope, intercept = np.polyfit(z_eqtl, z_gwas, 1)
        xs = np.linspace(z_eqtl.min(), z_eqtl.max(), 64)
        ax.plot(xs, slope * xs + intercept, color="#444444",
                linewidth=1.0, linestyle="--",
                label=f"OLS slope={slope:.2f}")

    ax.axhline(0.0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.set_xlabel("z (GTEx liver eQTL, SORT1)", fontsize=8)
    ax.set_ylabel("z (UKB / GLGC LDL GWAS)", fontsize=8)
    ax.set_title(
        f"B - SORT1 worked example\n"
        f"Pearson r(z_eqtl, z_gwas) = {pearson_r:.3f} . "
        f"n={n_variants}",
        fontsize=9,
    )
    ax.legend(fontsize=6, loc="best", framealpha=0.9)

    return {
        "n_variants": n_variants,
        "sort1_pearson_z": pearson_r,
        "orientation_counts": {
            str(k): int(v)
            for k, v in orient.value_counts().to_dict().items()
        },
        "z_eqtl_range": [float(z_eqtl.min()), float(z_eqtl.max())]
        if z_eqtl.size else None,
        "z_gwas_range": [float(z_gwas.min()), float(z_gwas.max())]
        if z_gwas.size else None,
    }


# ---------------------------------------------------------------------------
# Panel C - Arabidopsis SNP / expression / FT10.
# ---------------------------------------------------------------------------

def _render_panel_c(ax: plt.Axes) -> dict[str, Any]:
    """Plant multi-omics worked-example heatmap (B3 fixture)."""
    if not PLANT_PATH.exists():
        ax.text(0.5, 0.5,
                "validation/multiomics/plant/fixtures/aligned.parquet\n"
                "not found - Tier 4 B3 fixture missing",
                ha="center", va="center", fontsize=8, color="#a94442",
                transform=ax.transAxes)
        ax.set_title("C - plant multi-omics (MISSING)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return {"status": "missing"}

    df = pd.read_parquet(PLANT_PATH)
    n_samples = int(len(df))
    expr_cols = [c for c in df.columns if c.startswith("expr_")]
    snp_cols = [c for c in df.columns if c.startswith("snp_")]
    has_ft10 = "FT10" in df.columns

    if not expr_cols or not has_ft10:
        ax.text(0.5, 0.5,
                "aligned.parquet missing expr_ / FT10 cols\n"
                f"n_expr={len(expr_cols)} "
                f"has_FT10={has_ft10}",
                ha="center", va="center", fontsize=7, color="#a94442",
                transform=ax.transAxes)
        ax.set_title("C - plant multi-omics (schema error)", fontsize=10)
        return {
            "status": "schema_error",
            "n_expr_cols": len(expr_cols),
            "has_FT10": has_ft10,
        }

    ft10 = df["FT10"].to_numpy(dtype=float)
    expr = df[expr_cols].to_numpy(dtype=float)

    # Per-gene Pearson correlation of expression vs FT10. Sort by |r|
    # so the most informative genes appear at the left of the heatmap.
    corrs = np.zeros(len(expr_cols), dtype=float)
    for j in range(expr.shape[1]):
        col = expr[:, j]
        if np.std(col) < 1e-12 or np.std(ft10) < 1e-12:
            corrs[j] = 0.0
        else:
            corrs[j] = float(np.corrcoef(col, ft10)[0, 1])
    order = np.argsort(-np.abs(corrs))
    sorted_corrs = corrs[order]
    sorted_labels = [expr_cols[i] for i in order]

    # Render as a horizontal heatmap bar (1 x n_expr) so the figure
    # reads as a single matplotlib Axes inside the 1x3 layout. The
    # diverging colormap makes the sign of the correlation clear.
    img = sorted_corrs.reshape(1, -1)
    cmap = plt.get_cmap("RdBu_r")
    vmax = max(float(np.max(np.abs(sorted_corrs))), 1e-6)
    ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax,
              aspect="auto", interpolation="nearest")
    ax.set_yticks([0])
    ax.set_yticklabels(["Pearson r(expr, FT10)"], fontsize=7)

    # X labels: show top-5 absolute correlations with their values.
    n_show = min(5, len(sorted_corrs))
    show_idx = list(range(n_show))
    ax.set_xticks(show_idx)
    ax.set_xticklabels(
        [
            f"{sorted_labels[i].replace('expr_', '')}\n"
            f"{sorted_corrs[i]:+.2f}"
            for i in show_idx
        ],
        fontsize=6, rotation=0,
    )
    ax.set_xlabel(
        f"expression cols sorted by |r| (top {n_show} of "
        f"{len(expr_cols)} shown)",
        fontsize=8,
    )
    ax.set_title(
        f"C - Arabidopsis multi-omics\n"
        f"n={n_samples} ecotypes . "
        f"{len(snp_cols)} SNPs . "
        f"{len(expr_cols)} genes . FT10",
        fontsize=9,
    )

    return {
        "n_samples": n_samples,
        "n_snps": len(snp_cols),
        "n_expr_cols": len(expr_cols),
        "top_gene_corr": {
            sorted_labels[i].replace("expr_", ""): float(sorted_corrs[i])
            for i in show_idx
        },
        "abs_corr_range": [
            float(np.min(np.abs(corrs))),
            float(np.max(np.abs(corrs))),
        ],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@register("F4")
def render_f4(output_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    a = _render_panel_a(axes[0])
    b = _render_panel_b(axes[1])
    c = _render_panel_c(axes[2])

    fig.suptitle(
        "F4 - Multi-omics integration "
        "(simulated ground-truth . GTEx-UKB SORT1/LDL . "
        "Arabidopsis 1001G x FT10)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "Genome Biology Methods Figure 4 . "
        "fixtures: validation/multiomics/[sim, gtex_ukb, plant]",
        ha="center", fontsize=7, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    out_path = output_dir / "F4.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    manifest: dict[str, Any] = {
        "panels": {
            "A": a,
            "B": b,
            "C": c,
        },
        "fixtures": {
            "A_truth_json": str(TRUTH_PATH.relative_to(REPO_ROOT))
            if TRUTH_PATH.exists() else None,
            "B_aligned_parquet": str(GTEX_UKB_PATH.relative_to(REPO_ROOT))
            if GTEX_UKB_PATH.exists() else None,
            "C_aligned_parquet": str(PLANT_PATH.relative_to(REPO_ROOT))
            if PLANT_PATH.exists() else None,
        },
    }
    return out_path, manifest
