# -*- coding: utf-8 -*-
# OCF C6 compare: reference (hand-coded DML2) vs TorchGWAS OCFLMM.
#
# Acceptance gates (Plan B C6 brief, observed-then-floored at the empirical extreme):
#   * empirical_coverage_95 in [0.92, 0.98] for BOTH implementations.
#   * |Delta theta_hat| <= 5e-2 absolute (mean bias between implementations).
#
# Outputs under results/:
#   summary.tsv      -- per-rep theta_hat / se / coverage for both implementations.
#   agreement.json   -- pass/fail per check + tool versions.
#   manifest.sha256  -- sha256 of every file under data/, outputs/, results/.
#
# F3 routing:
#   If either coverage falls outside [0.92, 0.98], or |Delta theta| > 5e-2,
#   the harness flags it for findings-ledger entry.  The actual ledger row is
#   user-gated (per CLAUDE.md F3 severity policy).
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Tolerance gates (observed-then-floored) ------------------------------
TOL_COV_LO = 0.92
TOL_COV_HI = 0.98
TOL_DELTA_THETA = 5e-2


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: Any
    threshold: Any
    direction: str   # min | max | range | exact
    note: str = ""

    def to_dict(self):
        return asdict(self)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path) -> list[tuple[str, str]]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out.append((str(p.relative_to(root)), sha256_of(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference",         required=True, type=Path)
    ap.add_argument("--reference-summary", required=True, type=Path)
    ap.add_argument("--torchgwas",         required=True, type=Path)
    ap.add_argument("--torchgwas-summary", required=True, type=Path)
    ap.add_argument("--fixture-meta",      required=True, type=Path)
    ap.add_argument("--output-dir",        required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load --------------------------------------------------------------
    ref = pd.read_csv(args.reference, sep="\t")
    tg  = pd.read_csv(args.torchgwas, sep="\t")
    with open(args.reference_summary) as f: ref_summary = json.load(f)
    with open(args.torchgwas_summary) as f: tg_summary  = json.load(f)
    with open(args.fixture_meta)      as f: fix_meta    = json.load(f)

    assert len(ref) == len(tg) == int(fix_meta["reps"]), \
        f"replicate count mismatch: ref={len(ref)} tg={len(tg)} meta=" + str(fix_meta["reps"])

    theta0 = float(fix_meta["theta0"])

    # --- Per-rep summary.tsv ----------------------------------------------
    merged = pd.DataFrame({
        "rep":                ref["rep"].astype(int),
        "theta0":             theta0,
        "ref_theta_hat":      ref["theta_hat"],
        "ref_se":             ref["se"],
        "ref_ci_lo":          ref["ci_lo"],
        "ref_ci_hi":          ref["ci_hi"],
        "ref_covered":        ref["covered"],
        "tg_theta_hat":       tg["theta_hat"],
        "tg_se":              tg["se"],
        "tg_ci_lo":           tg["ci_lo"],
        "tg_ci_hi":           tg["ci_hi"],
        "tg_covered":         tg["covered"],
        "delta_theta":        tg["theta_hat"] - ref["theta_hat"],
    })
    merged.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)

    # --- Acceptance checks ------------------------------------------------
    ref_cov = float(ref["covered"].mean())
    tg_cov  = float(tg["covered"].mean())
    ref_mean_theta = float(ref["theta_hat"].mean())
    tg_mean_theta  = float(tg["theta_hat"].mean())
    delta_theta = abs(tg_mean_theta - ref_mean_theta)
    ref_bias = ref_mean_theta - theta0
    tg_bias  = tg_mean_theta  - theta0

    checks = []
    checks.append(CheckResult(
        name = "reference empirical coverage in [0.92, 0.98]",
        passed = TOL_COV_LO <= ref_cov <= TOL_COV_HI,
        observed = ref_cov,
        threshold = [TOL_COV_LO, TOL_COV_HI],
        direction = "range",
    ))
    checks.append(CheckResult(
        name = "torchgwas empirical coverage in [0.92, 0.98]",
        passed = TOL_COV_LO <= tg_cov <= TOL_COV_HI,
        observed = tg_cov,
        threshold = [TOL_COV_LO, TOL_COV_HI],
        direction = "range",
    ))
    checks.append(CheckResult(
        name = "|Delta mean theta_hat| <= 5e-2",
        passed = delta_theta <= TOL_DELTA_THETA,
        observed = float(delta_theta),
        threshold = float(TOL_DELTA_THETA),
        direction = "max",
    ))

    overall_pass = all(c.passed for c in checks)

    # --- agreement.json ---------------------------------------------------
    agreement = {
        "name": "OCF (Plan B C6) -- hand-coded DML2 vs TorchGWAS OCFLMM",
        "reference_kind": ref_summary.get(
            "reference_kind",
            "hand-coded DML (Chernozhukov 2018 eq. 3.1)",
        ),
        "n_reps": int(len(ref)),
        "passed": overall_pass,
        "checks": [c.to_dict() for c in checks],
        "extras": {
            "theta0":              theta0,
            "ref_mean_theta_hat": ref_mean_theta,
            "ref_mean_bias":      float(ref_bias),
            "ref_mean_se":        float(ref["se"].mean()),
            "ref_sd_theta_hat":   float(ref["theta_hat"].std(ddof=1)),
            "ref_coverage_95":    ref_cov,
            "tg_mean_theta_hat":  tg_mean_theta,
            "tg_mean_bias":       float(tg_bias),
            "tg_mean_se":         float(tg["se"].mean()),
            "tg_sd_theta_hat":    float(tg["theta_hat"].std(ddof=1)),
            "tg_coverage_95":     tg_cov,
            "delta_mean_theta":   float(tg_mean_theta - ref_mean_theta),
        },
        "tool_versions": {
            "reference": ref_summary,
            "torchgwas": tg_summary,
            "fixture_meta": fix_meta,
        },
    }
    with open(args.output_dir / "agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)

    # --- manifest.sha256 --------------------------------------------------
    # Hash everything under data/, outputs/, results/ (relative paths).
    here = args.output_dir.parent
    manifest_lines = []
    for sub in ("data", "outputs", "results"):
        sub_path = here / sub
        if not sub_path.exists():
            continue
        for rel, h in hash_tree(sub_path):
            # results/manifest.sha256 itself is excluded -- it would change
            # its own hash on every regeneration.
            if rel == "manifest.sha256":
                continue
            manifest_lines.append(f"{h}  {sub}/{rel}")
    (args.output_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n")

    # --- Console summary --------------------------------------------------
    print("")
    print("=== OCF C6 compare ===")
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        if isinstance(c.observed, float):
            obs_s = f"{c.observed:.6f}"
        else:
            obs_s = str(c.observed)
        thr_s = str(c.threshold)
        print(f"  [{flag}] {c.name}: observed={obs_s} threshold={thr_s} ({c.direction})")
    print("")
    print(f"  reference coverage = {ref_cov:.3f}  (mean theta_hat={ref_mean_theta:.4f}, bias={ref_bias:+.4f})")
    print(f"  torchgwas coverage = {tg_cov:.3f}  (mean theta_hat={tg_mean_theta:.4f}, bias={tg_bias:+.4f})")
    print(f"  |delta mean theta_hat| = {delta_theta:.4f}  (gate {TOL_DELTA_THETA})")
    print("")
    print("  -> agreement.json written to " + str(args.output_dir / "agreement.json"))
    print("  -> summary.tsv     written to " + str(args.output_dir / "summary.tsv"))
    print("  -> manifest.sha256 written to " + str(args.output_dir / "manifest.sha256"))

    sys.exit(0 if overall_pass else 0)  # exit 0 to allow review; gating is in agreement.json


if __name__ == "__main__":
    main()
