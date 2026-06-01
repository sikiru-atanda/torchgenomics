"""Run TorchGWAS KnockoffLMM across all replicate fixtures.

For each replicate r:
  1. Load (G_r, y_r) from data/replicates.npz
  2. Compute GRM, intercept covariate
  3. Run KnockoffLMM at target_fdr level
  4. Compute BLOCK-LEVEL empirical FDR + power vs known causal blocks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.linalg.kinship import grm_vanraden  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.knockoff_lmm import KnockoffLMM  # noqa: E402


def block_fdr_power(selected_blocks, block_indices, causal_idx_set):
    true_blocks = set(
        bi for bi, idx in enumerate(block_indices)
        if any(j in causal_idx_set for j in idx)
    )
    sel = set(selected_blocks)
    tp = len(sel & true_blocks)
    fp = len(sel - true_blocks)
    fdr = fp / max(1, len(sel))
    power = tp / max(1, len(true_blocks))
    return fdr, power, tp, fp, len(true_blocks), len(sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output-dir", default=str(HERE / "outputs"))
    ap.add_argument("--target-fdr", type=float, default=0.2)
    ap.add_argument("--ld-method", default="r2")
    ap.add_argument("--ld-window", type=int, default=20)
    ap.add_argument("--r2-threshold", type=float, default=0.3)
    ap.add_argument("--aggregation", default="max_stat", choices=["max_stat", "sum_sq"])
    ap.add_argument("--n-replicates", type=int, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth = json.loads((data_dir / "sim_truth.json").read_text())
    causal_idx_set = set(truth["causal_idx_zero_based"])
    n_replicates = args.n_replicates if args.n_replicates is not None else truth["n_replicates"]

    fixtures = np.load(data_dir / "replicates.npz")

    per_rep_rows = []
    print(f"[torchgwas] running {n_replicates} replicates at target_fdr={args.target_fdr}")

    for r in range(n_replicates):
        G_np = fixtures[f"G_rep_{r}"]
        y_np = fixtures[f"y_rep_{r}"]
        n, m = G_np.shape

        G = torch.tensor(G_np, dtype=torch.float64)
        Y = torch.tensor(y_np, dtype=torch.float64)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vm = VariantMeta(
            snp=[f"s{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(1, m + 1)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        t0 = time.time()
        res = KnockoffLMM(
            target_fdr=args.target_fdr,
            ld_method=args.ld_method,
            knockoff_method="equicorrelated",
            aggregation=args.aggregation,
            seed=42 + r,
            ld_window=args.ld_window,
            r2_threshold=args.r2_threshold,
        ).run(
            Y, X0, K, G, vm,
            variant_pos=list(range(1, m + 1)),
            variant_chr=["1"] * m,
        )
        elapsed = time.time() - t0

        fdr, power, tp, fp, n_true, n_sel = block_fdr_power(
            res.selected_blocks, res.block_indices, causal_idx_set,
        )
        per_rep_rows.append({
            "rep": r,
            "n_blocks": res.n_blocks,
            "n_selected": n_sel,
            "n_true_blocks": n_true,
            "tp": tp,
            "fp": fp,
            "fdr": fdr,
            "power": power,
            "threshold": float(res.threshold) if res.threshold != float("inf") else float("inf"),
            "elapsed_s": elapsed,
        })
        if (r + 1) % 10 == 0 or r == n_replicates - 1:
            print(f"  [{r+1:3d}/{n_replicates}] sel={n_sel:3d} tp={tp} fp={fp} fdr={fdr:.3f} pow={power:.3f} t={elapsed:.1f}s")

    tsv_path = out_dir / "torchgwas_per_replicate.tsv"
    with open(tsv_path, "w") as f:
        cols = ["rep", "n_blocks", "n_selected", "n_true_blocks", "tp", "fp",
                "fdr", "power", "threshold", "elapsed_s"]
        f.write("\t".join(cols) + "\n")
        for row in per_rep_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"[torchgwas] wrote {tsv_path}")

    fdrs = np.array([r["fdr"] for r in per_rep_rows])
    pows = np.array([r["power"] for r in per_rep_rows])
    summary = {
        "tool": "torchgwas.KnockoffLMM",
        "n_replicates": n_replicates,
        "target_fdr": args.target_fdr,
        "ld_method": args.ld_method,
        "aggregation": args.aggregation,
        "mean_fdr": float(fdrs.mean()),
        "sd_fdr": float(fdrs.std(ddof=1)) if len(fdrs) > 1 else 0.0,
        "sem_fdr": float(fdrs.std(ddof=1) / np.sqrt(len(fdrs))) if len(fdrs) > 1 else 0.0,
        "mean_power": float(pows.mean()),
        "sd_power": float(pows.std(ddof=1)) if len(pows) > 1 else 0.0,
        "sem_power": float(pows.std(ddof=1) / np.sqrt(len(pows))) if len(pows) > 1 else 0.0,
        "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in per_rep_rows])),
    }
    (out_dir / "torchgwas_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "[torchgwas] mean_fdr={:.4f} +/- {:.4f}   mean_power={:.4f} +/- {:.4f}".format(
            summary["mean_fdr"], summary["sem_fdr"],
            summary["mean_power"], summary["sem_power"],
        )
    )


if __name__ == "__main__":
    main()
