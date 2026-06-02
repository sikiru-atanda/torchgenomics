"""Haploview-style LD triangle on the MDP maize fixture.

Uses ``torchgenomics.viz.haploview_plot`` directly to render a Haploview-
mirror triangular heatmap of pairwise r² (Barrett et al. 2005 layout)
with Gabriel-2002 haplotype blocks overlaid as black triangles.

Workflow
--------
1. Load the MDP maize genotype matrix (281 samples × 3093 SNPs).
2. Pick a tight high-LD window: 15 SNPs spanning ~8.7 kb at the maize
   ``tb1`` (teosinte branched 1) locus on chromosome 1 — a classic
   maize LD-block exemplar.
3. Compute the pairwise r² and |D'| matrices via
   ``torchgenomics.ld.compute_r2_matrix`` and ``compute_dprime_matrix``.
4. Run two block-detection methods for comparison:
   - Gabriel et al. (2002) — confidence-interval-based.
   - r² method — agglomerative on r² threshold.
5. Render the Haploview triangle with block overlays via
   ``torchgenomics.viz.haploview_plot``.

Outputs (next to this script in ``docs/haploview_demo/``):

- ``mdp_tb1_15snps_r2_haploview.png``      — r² triangle, 300 dpi
- ``mdp_tb1_15snps_dprime_haploview.png``  — |D'| triangle, 300 dpi
- ``mdp_tb1_15snps_blocks.tsv``            — Block table (TSV)
- ``run_log.txt`` — full text log of the run
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torchgenomics.ld import (
    compute_dprime_matrix,
    compute_r2_matrix,
    detect_blocks,
)
from torchgenomics.viz import haploview_plot

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = REPO_ROOT / "benchmark" / "data"
HERE = Path(__file__).resolve().parent


def main() -> None:
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log("=" * 72)
    log("Haploview-style LD triangle demo on the MDP maize panel")
    log("=" * 72)

    # ----- Load genotype + SNP map ----------------------------------------
    geno = pd.read_csv(BENCH / "mdp_numeric.txt", sep="\t")
    snpmap = pd.read_csv(BENCH / "mdp_SNP_information.txt", sep="\t")

    # Genotype matrix: first column is sample id ("taxa"); remaining are SNPs.
    sample_ids = geno["taxa"].astype(str).tolist()
    snp_cols = geno.columns[1:].tolist()
    G_full = torch.tensor(
        geno[snp_cols].to_numpy(np.float64),
        dtype=torch.float64,
    )
    log(f"\nLoaded MDP genotype matrix: n_samples = {G_full.shape[0]}, "
        f"n_snps = {G_full.shape[1]}")

    # Mean-impute the small fraction of NaNs that the MDP fixture carries
    # (Haploview itself would call them missing; for the r²/D' demo we use
    # the same allele-frequency-mean imputation TorchGenomics's scanners use).
    for j in range(G_full.shape[1]):
        col = G_full[:, j]
        nan_mask = torch.isnan(col)
        if nan_mask.any():
            mu = float(col[~nan_mask].mean().item())
            G_full[nan_mask, j] = mu

    # ----- Pick the tightest 15-SNP window (maize tb1 locus, chr 1) -------
    snpmap = snpmap.copy()
    snpmap["Chromosome"] = snpmap["Chromosome"].astype(str)
    snpmap["Position"] = snpmap["Position"].astype(int)
    chr1_snpmap = (
        snpmap[snpmap["Chromosome"] == "1"]
        .sort_values("Position")
        .reset_index(drop=True)
    )

    # Sliding 15-SNP minimum-span window across chromosome 1.
    pos_arr = chr1_snpmap["Position"].to_numpy()
    spans = pos_arr[14:] - pos_arr[:-14]
    best_start = int(np.argmin(spans))
    target_snps = chr1_snpmap.iloc[best_start:best_start + 15]

    snp_id_to_col = {s: i for i, s in enumerate(snp_cols)}
    col_idx = [snp_id_to_col[s] for s in target_snps["SNP"].tolist()]
    G_win = G_full[:, col_idx]
    snp_labels = target_snps["SNP"].tolist()
    positions = target_snps["Position"].tolist()
    chrs = ["1"] * len(positions)

    log(f"\nWindow selected (tightest 15-SNP cluster on chr 1, maize tb1 locus):")
    log(f"  chromosome:  1")
    log(f"  n_snps:      {len(positions)}")
    log(f"  span:        {positions[0]:,} – {positions[-1]:,} bp "
        f"({(positions[-1] - positions[0]) / 1e3:.1f} kb)")
    log(f"  first SNP:   {snp_labels[0]}")
    log(f"  last SNP:    {snp_labels[-1]}")

    # ----- Compute r² and |D'| matrices -----------------------------------
    log(f"\nComputing pairwise LD matrices...")
    r2 = compute_r2_matrix(G_win)
    dprime = compute_dprime_matrix(G_win)

    # Symmetric, diagonal = 1.0
    r2 = r2.detach().cpu().numpy()
    dprime = np.abs(dprime.detach().cpu().numpy())

    iu = np.triu_indices_from(r2, k=1)
    log(f"  r²:  mean upper-triangle = {r2[iu].mean():.4f}, "
        f"max = {r2[iu].max():.4f}")
    log(f"  D':  mean upper-triangle = {dprime[iu].mean():.4f}, "
        f"max = {dprime[iu].max():.4f}")

    # ----- Detect haplotype blocks (two methods for comparison) -----------
    log(f"\nRunning block detection (two methods)...")
    block_rows: list[dict] = []
    method_to_blocks: dict[str, list] = {}
    for method, kwargs in [
        ("gabriel", dict(max_kb=200.0, maf_min=0.05)),
        ("r2", dict(max_kb=200.0, maf_min=0.05, r2_threshold=0.5)),
    ]:
        blks = detect_blocks(
            G_win,
            variant_pos=positions,
            variant_chr=chrs,
            variant_ids=snp_labels,
            method=method,
            **kwargs,
        )
        method_to_blocks[method] = blks
        log(f"  {method:10s}: detected {len(blks)} block(s)")
        for k, b in enumerate(blks):
            idxs = sorted(b.variant_indices)
            start_snp = snp_labels[idxs[0]]
            end_snp = snp_labels[idxs[-1]]
            start_bp = positions[idxs[0]]
            end_bp = positions[idxs[-1]]
            log(f"    block #{k + 1}: SNPs {start_snp} → {end_snp} "
                f"(idx {idxs[0]}–{idxs[-1]}, n={len(idxs)}, "
                f"{start_bp:,}–{end_bp:,} bp)")
            block_rows.append({
                "block_id": k + 1,
                "method": method,
                "chr": "1",
                "first_idx": idxs[0],
                "last_idx": idxs[-1],
                "first_snp": start_snp,
                "last_snp": end_snp,
                "first_bp": start_bp,
                "last_bp": end_bp,
                "n_snps_in_block": len(idxs),
            })

    block_table = pd.DataFrame(block_rows)
    block_path = HERE / "mdp_tb1_15snps_blocks.tsv"
    block_table.to_csv(block_path, sep="\t", index=False)
    log(f"\n  block table → {block_path}")

    # Use Gabriel blocks for the overlay (Haploview's default). Fall back
    # to r² blocks when Gabriel returns empty.
    blocks = method_to_blocks["gabriel"] or method_to_blocks["r2"]
    overlay_method = "gabriel" if method_to_blocks["gabriel"] else "r2"
    log(f"  Triangle overlay uses: {overlay_method}")

    # ----- Render: r² triangle --------------------------------------------
    log(f"\nRendering Haploview-style LD triangles...")
    fig, ax = plt.subplots(figsize=(12, 6))
    haploview_plot(
        r2,
        blocks=blocks,
        metric="r2",
        snp_labels=snp_labels,
        title=f"MDP maize · chr 1 · tb1 locus (15 SNPs / 8.7 kb) · "
              f"r² (overlay: {overlay_method} blocks)",
        ax=ax,
    )
    fig.tight_layout()
    r2_path = HERE / "mdp_tb1_15snps_r2_haploview.png"
    fig.savefig(r2_path, dpi=300)
    plt.close(fig)
    log(f"  r²   → {r2_path}")

    # ----- Render: |D'| triangle ------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    haploview_plot(
        dprime,
        blocks=blocks,
        metric="dprime",
        snp_labels=snp_labels,
        title=f"MDP maize · chr 1 · tb1 locus (15 SNPs / 8.7 kb) · "
              f"|D'| (overlay: {overlay_method} blocks)",
        ax=ax,
    )
    fig.tight_layout()
    dprime_path = HERE / "mdp_tb1_15snps_dprime_haploview.png"
    fig.savefig(dprime_path, dpi=300)
    plt.close(fig)
    log(f"  |D'| → {dprime_path}")

    # ----- Persist log -----------------------------------------------------
    log_path = HERE / "run_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n")
    log(f"\nFull text log written to {log_path}")
    log("Done.")


if __name__ == "__main__":
    main()
