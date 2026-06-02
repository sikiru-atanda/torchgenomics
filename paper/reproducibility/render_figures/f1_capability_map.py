"""F1 - TorchGenomics capability map.

Hand-curated inventory of all TorchGenomics capabilities grouped by cluster,
rendered as a labelled treemap. The inventory is derived from:

  - the "Core Modules" section of ``CLAUDE.md`` (canonical module surface);
  - ``torchgenomics/cli.py`` ``subparsers``/dispatch dict (CLI surface);
  - each sub-packages ``__init__.py`` public exports (programmatic API).

Each leaf annotation cites the module / CLI subcommand that supplies it
so the figure doubles as a navigation index for the Methods Section 2
"Architecture & capability surface" Results section.

Per spec Section 5 F1 (``docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md``):

    "single sunburst / treemap grouping all ~50 capabilities by cluster"

Tier 4 D2.1.
"""
from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; reproducible regardless of $DISPLAY
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


# Capability inventory ----------------------------------------------
#
# Order within each cluster is by CLAUDE.md "Core Modules" / Phase Index
# narrative order. Each capability is a short noun phrase suitable as a
# treemap label. The "n_capabilities" total is the sum of leaves across
# all clusters and is reported in the manifest so the abstract can quote
# it directly.
#
# Cluster colours follow a perceptually-distinct palette (matplotlib
# tab20). Treemap layout is slice-and-dice (no squarify dependency).

CAPABILITY_CLUSTERS: dict[str, list[str]] = {
    "Variant-level GWAS": [
        "GLM",
        "Single-trait LMM",
        "Multi-trait mvLMM",
        "FarmCPU",
        "BLINK",
        "GxE LMM",
        "Multi-kernel LMM",
        "Set-based (SKAT/Burden/SKAT-O)",
        "Bayesian VS (SuSiE + CAVI)",
        "Bayesian VS-RSS",
    ],
    "Haplotype layer": [
        # 13 LD-block methods (4 classical + 5 novel + 3 literature + 1 diagnostic)
        "LD blocks: Gabriel",
        "LD blocks: 4-gamete",
        "LD blocks: r2",
        "LD blocks: spine",
        "LD blocks: BigLD",
        "LD blocks: CC-graph",
        "LD blocks: DP-optimize",
        "LD blocks: change-point",
        "LD blocks: cross-pop",
        "LD blocks: graphical",
        "LD blocks: GWAS-aligned",
        "LD blocks: uncertainty",
        "LD blocks: Wall-Pritchard diag.",
        # 9 haplotype GWAS methods (4 classical + 5 novel)
        "Haplotype HTR",
        "Haplotype block",
        "Haplotype window",
        "Haplotype SKAT",
        "PCHT (score test)",
        "HHCT (hierarchical)",
        "HSKAT (similarity kernel)",
        "HapGxE",
        "BayesHap",
    ],
    "Multi-omics integration": [
        "TWAS (S-PrediXcan)",
        "SMR + HEIDI",
        "Coloc (2-trait)",
        "Hyprcoloc",
        "GRM-corrected mediation",
        "Multi-kernel h2",
        "Gene-set Wald",
        "eigenMT FDR",
    ],
    "Polyploid pipeline": [
        "Dosage call (updog/polyrad)",
        "F1 phasing (PolyOrigin)",
        "Polyploid GWAS scan",
        "Polyploid LD blocks",
        "Polyploid haplotype GWAS",
        "Gene-action encoding",
    ],
    "Specialty models": [
        "Survival (Cox PH frailty)",
        "Random regression LMM",
        "RR x multi-env",
        "Spatio-temporal RR",
        "Within-family LMM",
        "Threshold-linear (multi-trait)",
        "Knockoff LMM (FDR)",
        "OCF LMM (DML)",
        "Genotype-uncertainty LMM",
        "Leave-region-out LMM",
        "Conditional LMM (COJO)",
    ],
    "GLM / GLMM family": [
        "Binary GLM",
        "Ordinal GLM",
        "Multinomial GLM",
        "Binary GLMM (PQL)",
        "Ordinal GLMM (PQL)",
        "Multinomial GLMM (PQL)",
        "Multi-env GLMM",
    ],
    "Multi-env / multi-trait": [
        "MET (reaction-norm + FA(k))",
        "MT-MET LMM (Kronecker)",
        "Multi-env haplotype GWAS",
        "Multi-trait haplotype GWAS",
        "MT-MET haplotype GWAS",
    ],
    "Post-GWAS": [
        "LDSC h2",
        "LDSC genetic correlation",
        "S-LDSC (partitioned)",
        "Meta-analysis (IVW/DL/Stouffer/RE2)",
        "LD clumping",
        "Fine-mapping",
        "MR (IVW/Egger/median/PRESSO)",
        "HESS regional h2",
        "Multi-ancestry (MR-MEGA/MANTRA)",
        "Power analysis",
        "Winners-curse correction",
        "PGS: C+T",
        "PGS: LDpred2-Inf/Grid/Auto",
        "PGS: PRS-CS",
        "PGS scoring + validation",
        "NCBI gene annotation",
    ],
    "Multiple testing": [
        "Bonferroni",
        "Holm",
        "Sidak",
        "BH",
        "BY",
        "Storey q-value",
        "simpleM / M_eff",
        "eigenMT",
        "IHW",
        "AdaPT",
        "Weighted FDR",
        "Cauchy combination",
        "Local FDR",
        "Permutation max-T",
    ],
    "Visualization": [
        "Manhattan (linear)",
        "Circos Manhattan",
        "QQ plot",
        "Miami plot",
        "Haploview LD triangle",
        "Trumpet plot",
    ],
    "I/O + preprocessing": [
        "PLINK BED reader",
        "PLINK2 PGEN reader",
        "VCF/BCF reader",
        "BGEN reader",
        "HapMap reader",
        "Zarr/HDF5 reader",
        "Numeric/CSV dosage reader",
        "Format auto-detect + convert",
        "Imputation: mean/KNN/LD",
        "Imputation: GPU Li-Stephens",
        "Imputation: deep autoencoder",
        "Imputation: BEAGLE/IMPUTE5/Minimac4",
        "Phasing",
        "QC (MAF/HWE/call-rate)",
        "Standardization (ploidy-aware)",
        "Dosage uncertainty (R^2)",
    ],
}

# Cluster colours (palette: matplotlib tab20, perceptually distinct).
CLUSTER_COLORS: dict[str, str] = {
    "Variant-level GWAS":      "#1f77b4",  # blue
    "Haplotype layer":         "#ff7f0e",  # orange
    "Multi-omics integration": "#2ca02c",  # green
    "Polyploid pipeline":      "#d62728",  # red
    "Specialty models":        "#9467bd",  # purple
    "GLM / GLMM family":       "#8c564b",  # brown
    "Multi-env / multi-trait": "#e377c2",  # pink
    "Post-GWAS":               "#7f7f7f",  # grey
    "Multiple testing":        "#bcbd22",  # olive
    "Visualization":           "#17becf",  # cyan
    "I/O + preprocessing":     "#3f5d7d",  # navy
}


# Slice-and-dice treemap layout ------------------------------------


def _slice_and_dice(
    items: list[tuple[str, float]],
    x: float, y: float, w: float, h: float,
    *, vertical: bool,
) -> list[tuple[str, float, float, float, float]]:
    """Recursively lay out (label, weight) items in a w x h rectangle.

    Splits along the given axis (slice-and-dice). Returns a flat list of
    (label, x, y, w, h) rectangles in input order.
    """
    if not items:
        return []
    total = sum(weight for _, weight in items) or 1.0
    out: list[tuple[str, float, float, float, float]] = []
    cursor = 0.0
    for label, weight in items:
        frac = weight / total
        if vertical:
            rh = h * frac
            out.append((label, x, y + cursor, w, rh))
            cursor += rh
        else:
            rw = w * frac
            out.append((label, x + cursor, y, rw, h))
            cursor += rw
    return out


def _fit_fontsize(label: str, w: float, h: float, *, base: float = 8.0) -> float:
    """Pick a font size that fits the label inside a rectangle of size (w,h)."""
    # Heuristic: scale by min(width-per-char, height) compared to the base
    # cell. Tuned so the smallest leaves (~0.04 of fig width) still render.
    width_per_char = w / max(len(label), 1)
    fs = min(base, max(4.5, width_per_char * 28.0), max(4.5, h * 90.0))
    return float(fs)


# Renderer ----------------------------------------------------------


@register("F1")
def render_f1(output_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    # Order clusters by size descending so the largest clusters get the
    # leftmost slot in the slice-and-dice layout (most visually prominent).
    cluster_items = sorted(
        CAPABILITY_CLUSTERS.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )

    n_clusters = len(cluster_items)
    n_capabilities = sum(len(v) for _, v in cluster_items)

    # Figure: 12 wide x 7.5 tall is roomy enough for ~80 leaves at ~5pt min.
    fig_w, fig_h = 12.0, 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_axis_off()

    # Reserve the top 0.07 strip for the title and the bottom 0.05 for the
    # caption / footnote.
    plot_top = 0.93
    plot_bottom = 0.06

    # Top-level treemap: each cluster gets a rectangle weighted by its
    # number of capabilities. Lay out horizontally (slice). Wider clusters
    # are stacked first.
    cluster_layout = _slice_and_dice(
        items=[(name, float(len(caps))) for name, caps in cluster_items],
        x=0.0, y=plot_bottom, w=1.0, h=plot_top - plot_bottom,
        vertical=False,
    )

    for (cluster_name, cx, cy, cw, ch), (_, caps) in zip(
        cluster_layout, cluster_items, strict=True,
    ):
        color = CLUSTER_COLORS.get(cluster_name, "#888888")
        # Header strip for the cluster label: top ~18% of the cluster box.
        header_h = min(0.085, ch * 0.18)
        header_y = cy + ch - header_h
        # Cluster header rectangle (slightly darker than leaves).
        ax.add_patch(mpatches.Rectangle(
            (cx, header_y), cw, header_h,
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.95,
        ))
        # Cluster header label.
        header_fs = _fit_fontsize(cluster_name, cw, header_h, base=9.5)
        ax.text(
            cx + cw / 2, header_y + header_h / 2,
            f"{cluster_name}\n(n={len(caps)})",
            ha="center", va="center",
            fontsize=header_fs, color="white", weight="bold",
            linespacing=1.05,
        )

        # Sub-layout: leaves are stacked vertically inside the body.
        body_h = ch - header_h
        leaf_layout = _slice_and_dice(
            items=[(cap, 1.0) for cap in caps],  # equal weight per leaf
            x=cx, y=cy, w=cw, h=body_h,
            vertical=True,
        )
        for label, lx, ly, lw, lh in leaf_layout:
            ax.add_patch(mpatches.Rectangle(
                (lx, ly), lw, lh,
                facecolor=color, edgecolor="white", linewidth=0.6, alpha=0.42,
            ))
            fs = _fit_fontsize(label, lw, lh, base=7.0)
            ax.text(
                lx + lw / 2, ly + lh / 2,
                label,
                ha="center", va="center",
                fontsize=fs, color="#101010",
                linespacing=1.0,
            )

    # Title.
    fig.text(
        0.5, 0.965,
        "F1 - TorchGenomics capability map",
        ha="center", va="center", fontsize=13, weight="bold",
    )
    fig.text(
        0.5, 0.935,
        f"{n_capabilities} capabilities across {n_clusters} clusters "
        "(CLI registry + module surface)",
        ha="center", va="center", fontsize=9, color="#444444",
    )

    # Footer.
    fig.text(
        0.5, 0.02,
        "Source: torchgenomics/cli.py dispatch dict + sub-package __init__.py exports + CLAUDE.md Core Modules. "
        "Genome Biology Methods Figure 1.",
        ha="center", va="center", fontsize=7, color="#666666",
    )

    out_path = output_dir / "F1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Manifest: every number the paper quotes about F1 is derived here.
    manifest: dict[str, Any] = {
        "n_clusters": n_clusters,
        "n_capabilities": n_capabilities,
        "cluster_summary": {
            name: list(caps) for name, caps in CAPABILITY_CLUSTERS.items()
        },
    }
    return out_path, manifest

