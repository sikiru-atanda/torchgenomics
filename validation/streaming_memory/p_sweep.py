"""Run the streaming benchmark at multiple p values to expose linear-vs-flat
memory scaling. The streaming claim succeeds when peak scan-attributable
memory is flat (or near-flat) in p — i.e. depends on chunk_size only,
not the total variant count.

Each p:
  1. Build a synthetic BED at /tmp/streaming_bench/sweep_p{P}
  2. Run the streaming scan with peak RSS profiling
  3. Subtract the measured Python+torch baseline
  4. Record (p, bed_size_mb, peak_rss_mb, peak_uss_mb, scan_attributable_mb)

Output:
  outputs/p_sweep.tsv  — per-p row
  outputs/p_sweep.txt  — markdown table + scaling fit
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT_DIR = HERE
BUILD_SCRIPT = SCRIPT_DIR / "_build_synthetic_bed.py"
RUN_SCRIPT = SCRIPT_DIR / "run_lmm_scan.py"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--p-values", default="10000,25000,50000,100000,250000",
                   help="Comma-sep variant counts to sweep")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/streaming_bench"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("validation/streaming_memory/outputs"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    p_values = [int(x) for x in args.p_values.split(",")]

    # Baseline once
    print("=== measuring Python+torch baseline (no scan) ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_lmm_scan", RUN_SCRIPT)
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)
    baseline = run_mod.measure_baseline_rss()
    print(f"Baseline peak USS: {baseline['peak_uss_mb']:.1f} MB")
    print(f"Baseline peak RSS: {baseline['peak_rss_mb']:.1f} MB")
    print()

    rows = []
    for pv in p_values:
        prefix = args.work_dir / f"sweep_p{pv}"
        print(f"=== p = {pv:,} ===")

        # Build fixture (clean slate)
        for ext in (".bed", ".bim", ".fam", ".pheno"):
            (prefix.with_suffix(ext)).unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, str(BUILD_SCRIPT),
             "--n", str(args.n), "--p", str(pv),
             "--out", str(prefix), "--seed", "7"],
            check=True, capture_output=True,
        )
        bed_size = prefix.with_suffix(".bed").stat().st_size / 1e6

        # Run scan
        json_out = args.out_dir / f"run_p{pv}.json"
        t0 = time.time()
        rc = subprocess.run(
            [sys.executable, str(RUN_SCRIPT),
             "--prefix", str(prefix),
             "--chunk-size", str(args.chunk_size),
             "--work-dir", str(args.work_dir),
             "--output-json", str(json_out)],
            capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(f"FAIL at p={pv}:\n{rc.stdout}\n{rc.stderr}")
            return 1
        result = json.loads(json_out.read_text())
        scan_attrib = max(0.0, result["peak_uss_mb"] - baseline["peak_uss_mb"])
        rows.append({
            "p": pv,
            "bed_mb": bed_size,
            "materialized_mb": result["materialized_mb"],
            "peak_uss_mb": result["peak_uss_mb"],
            "peak_rss_mb": result["peak_rss_mb"],
            "scan_attributable_mb": scan_attrib,
            "elapsed_s": result["elapsed_s"],
            "samples": result["n_memory_samples"],
        })
        print(f"  peak USS = {result['peak_uss_mb']:.1f} MB"
              f"  attributable = {scan_attrib:.1f} MB"
              f"  elapsed = {result['elapsed_s']:.2f}s")
        print()

    # Emit TSV + markdown
    tsv_path = args.out_dir / "p_sweep.tsv"
    md_path = args.out_dir / "p_sweep.txt"
    with open(tsv_path, "w") as f:
        f.write("p\tbed_mb\tmaterialized_mb\tpeak_uss_mb\tpeak_rss_mb\t"
                "scan_attrib_mb\telapsed_s\n")
        for r in rows:
            f.write(f"{r['p']}\t{r['bed_mb']:.1f}\t{r['materialized_mb']:.1f}\t"
                    f"{r['peak_uss_mb']:.1f}\t{r['peak_rss_mb']:.1f}\t"
                    f"{r['scan_attributable_mb']:.1f}\t{r['elapsed_s']:.2f}\n")

    # Markdown summary + scaling diagnostic
    lines = []
    lines.append("# Streaming memory scaling vs p (variant count)")
    lines.append("")
    lines.append(f"Fixture: n={args.n:,}, chunk_size={args.chunk_size:,}")
    lines.append(f"Baseline (Python+torch imports only): {baseline['peak_uss_mb']:.1f} MB USS")
    lines.append("")
    lines.append("| p | BED MB | Materialized MB | Peak USS MB | Scan-attributable MB | Elapsed s |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['p']:,} | {r['bed_mb']:.1f} | {r['materialized_mb']:.1f} | "
            f"{r['peak_uss_mb']:.1f} | {r['scan_attributable_mb']:.1f} | "
            f"{r['elapsed_s']:.2f} |"
        )
    lines.append("")
    # Scaling fit: scan_attributable_mb = a + b * p
    if len(rows) >= 3:
        ps = [r["p"] for r in rows]
        ms = [r["scan_attributable_mb"] for r in rows]
        # Simple linear regression
        n = len(ps)
        sx = sum(ps); sy = sum(ms)
        sxx = sum(x*x for x in ps); sxy = sum(x*y for x, y in zip(ps, ms))
        slope = (n*sxy - sx*sy) / max(n*sxx - sx*sx, 1e-30)
        intercept = (sy - slope*sx) / n
        # Compare to materialized slope (which would be n*8/1e6 MB per variant)
        materialized_slope_mb_per_variant = args.n * 8 / 1e6
        lines.append(f"## Scaling diagnostic")
        lines.append("")
        lines.append(
            f"Linear fit: scan_attributable_mb ≈ {intercept:.1f} + "
            f"{slope*1e6:.4f} MB / 1M variants"
        )
        lines.append(
            f"Materialized slope (would-be): {materialized_slope_mb_per_variant*1e6:.1f} "
            f"MB / 1M variants (= n×8B)"
        )
        ratio = slope / max(materialized_slope_mb_per_variant, 1e-30)
        lines.append(
            f"Streaming slope as fraction of materialized: **{ratio:.4g}×** "
            f"({1/max(ratio,1e-30):.0f}× lower)"
        )
        lines.append("")
        lines.append("Streaming claim succeeds when slope ≪ materialized slope.")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\n=== Sweep complete ===")
    print(f"  TSV: {tsv_path}")
    print(f"  MD:  {md_path}")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
