"""Benchmark sparse-GRM path vs dense at n=10K.

At n=10K the dense GRM is 800 MB (n²×8B). The sparse path
(`torchgwas.linalg.sparse_grm`) keeps only off-diagonal entries above a
threshold (default 0.05 in correlation units), giving 10×-100× memory
reduction. This benchmark:

  1. Builds GRM in both modes (streaming dense vs streaming sparse)
  2. Records peak USS of each
  3. Runs lmm-scan with each (via CLI --approx-method sparse vs default)
  4. Compares β / SE / p across the two modes on the SAME data

The score test in the sparse path is the only test available there; the
dense path supports Wald. So we compare:
  - GRM construction wall + peak
  - lmm-scan wall + peak
  - β correlation between sparse-score and dense-wald (expected to differ
    in absolute β but agree on test statistic / p)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
from scipy.stats import pearsonr


def poll_memory(pid: int, samples: list, stop_event) -> None:
    proc = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            samples.append(proc.memory_full_info().uss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.05)


def run_with_peak(cmd: list, env=None) -> dict:
    samples = []
    stop_event = threading.Event()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
    t = threading.Thread(target=poll_memory, args=(proc.pid, samples, stop_event),
                         daemon=True)
    t.start()
    stdout, stderr = proc.communicate()
    stop_event.set()
    t.join(timeout=2.0)
    elapsed = time.time() - t0
    peak_mb = max(samples) / 1e6 if samples else 0.0
    return {
        "elapsed_s": elapsed,
        "peak_uss_mb": peak_mb,
        "n_samples": len(samples),
        "stdout": stdout, "stderr": stderr,
        "returncode": proc.returncode,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=Path,
                   default=Path("/tmp/streaming_bench/sparse_n10k"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("validation/streaming_memory/outputs/sparse_grm"))
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Sparse-GRM correlation threshold (default 0.05)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Split .pheno into pheno-only + covariate-only per the CLI's auto-dummies
    # bug we ran into earlier.
    pheno_df = pd.read_csv(args.prefix.with_suffix(".pheno"), sep="\t")
    pheno_path = args.out_dir / "pheno.tsv"
    covar_path = args.out_dir / "covar.tsv"
    pheno_df[["FID", "IID", "Y"]].to_csv(pheno_path, sep="\t", index=False)
    pheno_df[["IID", "PC1", "PC2"]].to_csv(covar_path, sep="\t", index=False)

    n = sum(1 for _ in open(args.prefix.with_suffix(".fam")))
    p_total = sum(1 for _ in open(args.prefix.with_suffix(".bim")))
    print(f"[fixture] n={n:,}  p={p_total:,}")
    print(f"[projection] dense GRM (n²×8): {n*n*8/1e6:.1f} MB")
    print(f"[projection] sparse GRM at threshold {args.threshold}: "
          f"~10-100× lower = {n*n*8/1e6/30:.1f} MB (rough estimate)")
    print()

    # --- Run 1: DENSE path (default) ---
    dense_out = args.out_dir / "dense"
    cmd_dense = [
        sys.executable, "-m", "torchgwas", "lmm-scan",
        "--genotype", f"{args.prefix}.bed",
        "--phenotype", str(pheno_path),
        "--covariate", str(covar_path),
        "--traits", "Y", "--test", "score",   # use score test for apples-to-apples vs sparse
        "--correction", "none",
        "--output", str(dense_out),
    ]
    print(f"[dense] running: {' '.join(cmd_dense[5:])}")
    r_dense = run_with_peak(cmd_dense)
    if r_dense["returncode"] != 0:
        print("DENSE STDOUT:", r_dense["stdout"][-1000:])
        print("DENSE STDERR:", r_dense["stderr"][-1000:])
        sys.exit("dense lmm-scan failed")
    print(f"[dense] elapsed={r_dense['elapsed_s']:.1f}s  peak USS={r_dense['peak_uss_mb']:.1f} MB")
    print()

    # --- Run 2: SPARSE path ---
    sparse_out = args.out_dir / "sparse"
    cmd_sparse = [
        sys.executable, "-m", "torchgwas", "lmm-scan",
        "--genotype", f"{args.prefix}.bed",
        "--phenotype", str(pheno_path),
        "--covariate", str(covar_path),
        "--traits", "Y", "--test", "score",
        "--correction", "none",
        "--approx-method", "sparse",
        "--sparse-threshold", str(args.threshold),
        "--output", str(sparse_out),
    ]
    print(f"[sparse] running: {' '.join(cmd_sparse[5:])}")
    r_sparse = run_with_peak(cmd_sparse)
    if r_sparse["returncode"] != 0:
        print("SPARSE STDOUT:", r_sparse["stdout"][-1000:])
        print("SPARSE STDERR:", r_sparse["stderr"][-1000:])
        sys.exit("sparse lmm-scan failed")
    print(f"[sparse] elapsed={r_sparse['elapsed_s']:.1f}s  peak USS={r_sparse['peak_uss_mb']:.1f} MB")
    print()

    # --- Comparison ---
    tg_dense = pd.read_csv(f"{dense_out}.assoc.tsv", sep="\t")
    tg_sparse = pd.read_csv(f"{sparse_out}.assoc.tsv", sep="\t")
    merged = tg_dense.merge(tg_sparse, on="SNP", suffixes=("_dense", "_sparse"))

    beta_r = float(pearsonr(merged["BETA_dense"], merged["BETA_sparse"])[0])
    se_r = float(pearsonr(merged["SE_dense"], merged["SE_sparse"])[0])
    p_dense = merged["P_dense"].clip(lower=1e-300)
    p_sparse = merged["P_sparse"].clip(lower=1e-300)
    import numpy as np
    log10p_dense = -np.log10(p_dense)
    log10p_sparse = -np.log10(p_sparse)
    log10p_r = float(pearsonr(log10p_dense, log10p_sparse)[0])
    log10p_max_abs = float((log10p_dense - log10p_sparse).abs().max())

    print(f"\n=== Sparse vs dense GRM (n={n:,}, p={p_total:,}, threshold={args.threshold}) ===")
    print(f"  Variants compared:       {len(merged):,}")
    print(f"  β Pearson:               {beta_r:.6f}")
    print(f"  SE Pearson:              {se_r:.6f}")
    print(f"  -log10 p Pearson:        {log10p_r:.6f}")
    print(f"  -log10 p max abs diff:   {log10p_max_abs:.6g}")
    print()
    print(f"  Wall-time dense:         {r_dense['elapsed_s']:.1f}s")
    print(f"  Wall-time sparse:        {r_sparse['elapsed_s']:.1f}s "
          f"({r_sparse['elapsed_s']/r_dense['elapsed_s']:.2f}×)")
    print(f"  Peak USS dense:          {r_dense['peak_uss_mb']:.1f} MB")
    print(f"  Peak USS sparse:         {r_sparse['peak_uss_mb']:.1f} MB "
          f"({r_sparse['peak_uss_mb']/r_dense['peak_uss_mb']:.2f}×)")
    print(f"  Memory savings:          {(1 - r_sparse['peak_uss_mb']/r_dense['peak_uss_mb'])*100:.1f}%")

    summary = {
        "n": n, "p": p_total, "threshold": args.threshold,
        "dense": {"elapsed_s": r_dense["elapsed_s"],
                  "peak_uss_mb": r_dense["peak_uss_mb"]},
        "sparse": {"elapsed_s": r_sparse["elapsed_s"],
                   "peak_uss_mb": r_sparse["peak_uss_mb"]},
        "parity": {
            "n_variants": len(merged),
            "beta_pearson": beta_r,
            "se_pearson": se_r,
            "log10p_pearson": log10p_r,
            "log10p_max_abs_diff": log10p_max_abs,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
