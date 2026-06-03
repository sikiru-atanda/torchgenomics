# -*- coding: utf-8 -*-
# OCFLMM TorchGenomics harness for the C6 fixture.
#
# Loads each replicate (Y, G, W) emitted by generate.R, fits OCFLMM with
# K-fold cross-fitting, and writes per-rep theta_hat / se / 95% CI / coverage
# into a TSV that compare.py compares against the R reference.
#
# Design notes:
#   * The fixture confounders W enter the GWAS path as fixed effect covariates
#     via X0. OCFLMM DML mechanic projects W out of both Y and g_test before
#     fitting, which is the orthogonal-score step in Chernozhukov 2018 eq 3.1.
#     TorchGenomics does this in score_chunk via project_genotype=True.
#   * Kinship K is identity (generator default).  This makes the LMM nuisance
#     fit degenerate to OLS with sig2_g ~ 0; the DML wrapper still applies a
#     full cross-fit on REML residuals so the DML mechanic is exercised.
#   * Inference target is the causal SNP only (g_idx = meta causal_snp);
#     bystander SNPs are present so we can confirm score_chunk handles an
#     (n, m) genotype matrix end-to-end.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.models.base import VariantMeta  # noqa: E402
from torchgenomics.models.ocf_lmm import OCFLMM     # noqa: E402

def load_replicate(rep_dir):
    Y = pd.read_csv(rep_dir / "Y.csv")["Y"].values.astype(np.float64)
    G = pd.read_csv(rep_dir / "G.csv").values.astype(np.float64)
    W = pd.read_csv(rep_dir / "W.csv").values.astype(np.float64)
    n = int(pd.read_csv(rep_dir / "K_id.csv")["n"].iloc[0])
    assert Y.shape[0] == n == G.shape[0] == W.shape[0]
    return Y, G, W, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir",  default="outputs")
    ap.add_argument("--k",        type=int, default=5)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "fixture_meta.json") as f:
        meta = json.load(f)

    reps        = int(meta["reps"])
    theta0      = float(meta["theta0"])
    n           = int(meta["n"])
    m_snps      = int(meta["m"])
    causal_idx0 = int(meta["causal_snp"]) - 1  # 1-based in fixture -> 0-based in numpy

    print(f"[torchgenomics] reps={reps} n={n} m={m_snps} causal=idx{causal_idx0} K={args.k} theta0={theta0:.4f}")

    rng = np.random.default_rng(args.seed + 1)
    rep_seeds = rng.integers(low=1, high=2**31 - 1, size=reps)

    rows = []
    for r in range(1, reps + 1):
        rep_dir = data_dir / f"rep_{r:03d}"
        Y_np, G_np, W_np, n_r = load_replicate(rep_dir)
        assert n_r == n

        Y_t  = torch.from_numpy(Y_np)
        # X0 = intercept + W (W is the confounder block, projected by OCFLMM).
        X0_t = torch.from_numpy(np.column_stack([np.ones(n), W_np]))
        # Identity kinship -- matches generator.
        K_t  = torch.eye(n, dtype=torch.float64)

        model = OCFLMM(
            n_folds = args.k,
            seed    = int(rep_seeds[r - 1]),
            variance_type   = "HC",
            project_genotype = True,
        )
        nf = model.fit_null(Y_t, X0_t, K=K_t)

        # Score causal SNP only (single-variant chunk).
        G_chunk = torch.from_numpy(G_np[:, causal_idx0 : causal_idx0 + 1])
        vmeta = VariantMeta(
            snp = [f"rs{causal_idx0:05d}"],
            chr = ["1"], pos = [causal_idx0 * 1000],
            a1  = ["A"], a2  = ["G"],
        )
        sr = model.score_chunk(G_chunk, nf, vmeta)

        beta = float(sr.beta.item())
        se   = float(sr.se.item())
        p    = float(sr.p.item())
        ci_lo = beta - 1.96 * se
        ci_hi = beta + 1.96 * se
        covered = bool(ci_lo <= theta0 <= ci_hi)
        z = beta / max(se, 1e-30)

        rows.append({
            "rep": r,
            "theta0": theta0,
            "theta_hat": beta,
            "se": se,
            "z": z,
            "p": p,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "covered": covered,
            "mean_h2": nf.mean_h2,
            "converged": nf.converged,
        })
        if r % 10 == 1:
            print(f"  rep {r:03d}/{reps}: theta_hat={beta:.4f} se={se:.4f} covered={covered}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "torchgenomics_results.tsv", sep="\t", index=False)

    emp_cov = float(df["covered"].mean())
    mean_bias = float(df["theta_hat"].mean() - theta0)
    summary = {
        "n_reps": int(len(df)),
        "theta0": theta0,
        "mean_theta_hat": float(df["theta_hat"].mean()),
        "mean_bias": mean_bias,
        "mean_se": float(df["se"].mean()),
        "sd_theta_hat": float(df["theta_hat"].std(ddof=1)),
        "empirical_coverage_95": emp_cov,
        "median_p": float(df["p"].median()),
        "k_folds": args.k,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
    }
    with open(out_dir / "torchgenomics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[torchgenomics] DONE. empirical_coverage={0:.3f} mean_theta_hat={1:.4f} mean_bias={2:.4f} mean_se={3:.4f}".format(emp_cov, float(df["theta_hat"].mean()), mean_bias, float(df["se"].mean())))


if __name__ == "__main__":
    main()
