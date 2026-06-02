"""Track B: head-to-head β-parity vs regenie on the synthetic BED fixture.

Uses the largest synthetic from the p-sweep (default: sweep_p100000) and runs:
  1. TorchGenomics lmm-scan via CLI (streaming path)
  2. regenie Step 1 + Step 2

Compares per-variant β, SE, chi-sq, p across the join on SNP ID. Records
Pearson correlations and absolute max diffs. Per scientific-rigor mandate
the success criteria are observed-then-floored:
  - β Pearson ≥ 0.999 (typical Pillar B contract for quant traits)
  - SE Pearson ≥ 0.999
  - chi-sq Pearson ≥ 0.99
  - -log10 p max abs diff ≤ 0.1 (small near top hits)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REGENIE_BIN = ROOT / "validation/external/regenie/bin/regenie"


def run_torchgenomics_lmm_scan(prefix: Path, out_dir: Path) -> Path:
    """Run torchgenomics lmm-scan via the CLI on the BED fixture.

    Phenotype + covariates come from <prefix>.pheno. We split the combined
    .pheno into separate pheno + covar files (the format the CLI accepts).
    """
    out_path = out_dir / "torchgenomics"
    pheno_df = pd.read_csv(prefix.with_suffix(".pheno"), sep="\t")
    pheno_only = pheno_df[["FID", "IID", "Y"]]
    # CRITICAL: drop FID from covariates. The CLI auto-dummies non-numeric
    # covariate columns, and FID="IID_<i>" expands to n-1 dummy columns,
    # making the design matrix degenerate (residual variance → 0).
    covar_only = pheno_df[["IID", "PC1", "PC2"]]
    pheno_path = out_dir / "tg_pheno.tsv"
    covar_path = out_dir / "tg_covar.tsv"
    pheno_only.to_csv(pheno_path, sep="\t", index=False)
    covar_only.to_csv(covar_path, sep="\t", index=False)
    cmd = [
        sys.executable, "-m", "torchgenomics", "lmm-scan",
        "--genotype", f"{prefix}.bed",
        "--phenotype", str(pheno_path),
        "--covariate", str(covar_path),
        "--traits", "Y",
        "--test", "wald",
        "--correction", "none",
        "--output", str(out_path),
    ]
    print(f"[torchgenomics] running: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        sys.exit(f"torchgenomics lmm-scan failed (rc={r.returncode})")
    print(f"[torchgenomics] done in {dt:.2f}s")
    # CLI writes <out>.assoc.tsv (lmm-scan convention)
    for ext in (".assoc.tsv", ".tsv", ".csv"):
        cand = Path(f"{out_path}{ext}")
        if cand.exists():
            return cand
    cands = sorted(out_dir.glob("torchgenomics*"))
    print("CLI stdout:", r.stdout)
    sys.exit(f"torchgenomics output not found; candidates: {cands}")


def run_regenie(prefix: Path, out_dir: Path) -> Path:
    """Run regenie Step 1 + Step 2 quantitative GWAS."""
    if not REGENIE_BIN.exists():
        sys.exit(f"FATAL: regenie binary not found at {REGENIE_BIN}")
    out_dir = out_dir / "regenie"
    out_dir.mkdir(parents=True, exist_ok=True)

    # regenie wants pheno + covar as separate files; ours has both in .pheno
    pheno_df = pd.read_csv(prefix.with_suffix(".pheno"), sep="\t")
    pheno_only = pheno_df[["FID", "IID", "Y"]]
    covar_only = pheno_df[["FID", "IID", "PC1", "PC2"]]
    pheno_only.to_csv(out_dir / "pheno.tsv", sep="\t", index=False)
    covar_only.to_csv(out_dir / "covar.tsv", sep="\t", index=False)

    # Step 1
    step1_cmd = [
        str(REGENIE_BIN), "--step", "1",
        "--bed", str(prefix),
        "--phenoFile", str(out_dir / "pheno.tsv"),
        "--covarFile", str(out_dir / "covar.tsv"),
        "--bsize", "1000",
        "--lowmem",
        "--qt",
        "--threads", "4",
        "--out", str(out_dir / "step1"),
    ]
    print(f"[regenie] step 1: {' '.join(step1_cmd)}")
    t0 = time.time()
    r1 = subprocess.run(step1_cmd, capture_output=True, text=True)
    dt1 = time.time() - t0
    if r1.returncode != 0:
        print("STDOUT:\n", r1.stdout[-3000:])
        print("STDERR:\n", r1.stderr[-3000:])
        sys.exit(f"regenie step 1 failed (rc={r1.returncode})")
    print(f"[regenie] step 1 done in {dt1:.2f}s")

    # Step 2
    pred_list = out_dir / "step1_pred.list"
    if not pred_list.exists():
        sys.exit(f"regenie step 1 didn't write pred list at {pred_list}")
    step2_cmd = [
        str(REGENIE_BIN), "--step", "2",
        "--bed", str(prefix),
        "--phenoFile", str(out_dir / "pheno.tsv"),
        "--covarFile", str(out_dir / "covar.tsv"),
        "--bsize", "1000",
        "--qt",
        "--pred", str(pred_list),
        "--threads", "4",
        "--out", str(out_dir / "step2"),
    ]
    print(f"[regenie] step 2: {' '.join(step2_cmd)}")
    t0 = time.time()
    r2 = subprocess.run(step2_cmd, capture_output=True, text=True)
    dt2 = time.time() - t0
    if r2.returncode != 0:
        print("STDOUT:\n", r2.stdout[-3000:])
        print("STDERR:\n", r2.stderr[-3000:])
        sys.exit(f"regenie step 2 failed (rc={r2.returncode})")
    print(f"[regenie] step 2 done in {dt2:.2f}s")

    step2_out = out_dir / "step2_Y.regenie"
    if not step2_out.exists():
        cands = sorted(out_dir.glob("step2*"))
        sys.exit(f"regenie step 2 output not found; candidates: {cands}")
    return step2_out


def compare(tg_tsv: Path, regenie_tsv: Path, report_path: Path) -> int:
    tg = pd.read_csv(tg_tsv, sep="\t")
    re = pd.read_csv(regenie_tsv, sep=" ")  # regenie uses space-separated
    # Normalize columns:
    # regenie cols: CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P EXTRA
    # torchgenomics tsv cols vary by version; expect SNP, CHR, BP, BETA, SE, P, ...
    re = re.rename(columns={"ID": "SNP", "BETA": "BETA_re", "SE": "SE_re",
                            "CHISQ": "CHISQ_re", "LOG10P": "LOG10P_re"})
    # Detect torchgenomics column names
    tg_snp_col = "SNP" if "SNP" in tg.columns else (
        "snp" if "snp" in tg.columns else tg.columns[0]
    )
    tg_beta_col = "BETA" if "BETA" in tg.columns else "beta"
    tg_se_col   = "SE"   if "SE"   in tg.columns else "se"
    tg_p_col    = "P"    if "P"    in tg.columns else ("p_value" if "p_value" in tg.columns else "p")

    print(f"[compare] tg cols: {list(tg.columns)}")
    print(f"[compare] re cols: {list(re.columns)}")
    tg_renamed = tg.rename(columns={
        tg_snp_col: "SNP", tg_beta_col: "BETA_tg", tg_se_col: "SE_tg", tg_p_col: "P_tg"
    })
    merged = tg_renamed.merge(re[["SNP", "BETA_re", "SE_re", "CHISQ_re", "LOG10P_re"]],
                              on="SNP", how="inner")
    print(f"[compare] merged on SNP: {len(merged)} rows (tg={len(tg)}, re={len(re)})")
    if len(merged) < 100:
        sys.exit(f"FATAL: too few merged rows ({len(merged)}); SNP id mismatch?")

    # Derived TG metrics
    merged["CHISQ_tg"] = (merged["BETA_tg"] / merged["SE_tg"]) ** 2
    merged["LOG10P_tg"] = -np.log10(merged["P_tg"].clip(lower=1e-300))

    # Detect global effect-allele convention flip between regenie ALLELE1
    # convention and TG's BED A1/A2 convention. With consistent alleles
    # across all variants, the flip is a global sign change; we detect by
    # looking at the sign of corr(BETA_tg, BETA_re).
    beta_raw_r = float(pearsonr(merged["BETA_tg"], merged["BETA_re"])[0])
    sign_flip = beta_raw_r < 0
    if sign_flip:
        merged["BETA_tg_aligned"] = -merged["BETA_tg"]
    else:
        merged["BETA_tg_aligned"] = merged["BETA_tg"]

    # Stats — BETA correlated AFTER allele-convention alignment.
    beta_r = float(pearsonr(merged["BETA_tg_aligned"], merged["BETA_re"])[0])
    se_r   = float(pearsonr(merged["SE_tg"],   merged["SE_re"])[0])
    chi_r  = float(pearsonr(merged["CHISQ_tg"], merged["CHISQ_re"])[0])
    log10p_r = float(pearsonr(merged["LOG10P_tg"], merged["LOG10P_re"])[0])
    log10p_max_abs = float((merged["LOG10P_tg"] - merged["LOG10P_re"]).abs().max())
    beta_max_abs_rel = float(
        ((merged["BETA_tg_aligned"] - merged["BETA_re"]).abs() /
         merged[["BETA_tg_aligned","BETA_re"]].abs().max(axis=1).clip(lower=1e-6)).max()
    )

    thresholds = {
        "beta_pearson":   0.999,
        "se_pearson":     0.999,
        "chisq_pearson":  0.99,
        "log10p_pearson": 0.99,
        "log10p_max_abs": 0.10,
    }
    results = {
        "beta_pearson":   beta_r,
        "se_pearson":     se_r,
        "chisq_pearson":  chi_r,
        "log10p_pearson": log10p_r,
        "log10p_max_abs": log10p_max_abs,
        "beta_max_abs_rel": beta_max_abs_rel,
        "n_merged":       len(merged),
    }

    failures = []
    for k, thr in thresholds.items():
        v = results[k]
        if k.endswith("_pearson") and v < thr:
            failures.append(f"{k}: {v:.6g} < {thr}")
        elif k.endswith("_max_abs") and v > thr:
            failures.append(f"{k}: {v:.6g} > {thr}")

    print("\n=== Track B β-parity vs regenie ===")
    print(f"  N merged variants:       {len(merged):,}")
    print(f"  Effect-allele sign flip: {sign_flip} "
          f"(raw β Pearson = {beta_raw_r:+.6f})")
    print(f"  β Pearson (aligned):     {beta_r:.6f} (thr {thresholds['beta_pearson']:.3f})")
    print(f"  SE Pearson:              {se_r:.6f} (thr {thresholds['se_pearson']:.3f})")
    print(f"  χ² Pearson:              {chi_r:.6f} (thr {thresholds['chisq_pearson']:.3f})")
    print(f"  -log10 p Pearson:        {log10p_r:.6f} (thr {thresholds['log10p_pearson']:.3f})")
    print(f"  -log10 p max abs diff:   {log10p_max_abs:.6g} (thr {thresholds['log10p_max_abs']:.3f})")
    print(f"  β max abs relative diff: {beta_max_abs_rel:.4g}")
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
    else:
        print(f"\nPASS: all numeric thresholds met")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"Track B β-parity vs regenie\n"
        + "=" * 40 + "\n"
        + f"Fixture: {tg_tsv.parent}\n"
        + f"Results: " + repr(results) + "\n"
        + f"Failures: " + repr(failures) + "\n"
    )
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=Path,
                   default=Path("/tmp/streaming_bench/sweep_p10000"),
                   help="BED prefix; default uses the smallest p-sweep fixture")
    p.add_argument("--out-dir", type=Path,
                   default=Path("validation/streaming_memory/outputs/parity"))
    p.add_argument("--report", type=Path,
                   default=Path("validation/streaming_memory/outputs/parity_report.txt"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Sanity check the fixture exists
    for ext in (".bed", ".bim", ".fam", ".pheno"):
        if not args.prefix.with_suffix(ext).exists():
            sys.exit(f"FATAL: missing {args.prefix.with_suffix(ext)} — run p_sweep first")

    tg_tsv = run_torchgenomics_lmm_scan(args.prefix, args.out_dir)
    re_tsv = run_regenie(args.prefix, args.out_dir)
    return compare(tg_tsv, re_tsv, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
