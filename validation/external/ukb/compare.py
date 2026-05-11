"""NA3 UKB-scale comparison: TorchGWAS vs REGENIE + LDSC on chr22.

Three checks (per Task NA3 success criteria, SESSION_HANDOFF lines 100-102):

  1. β correlation (TorchGWAS vs REGENIE Step 2 quantitative LMM)
       target Pearson > 0.999
       (per the v0.3.0-v0.3.8 streaming-audit memory: when both tools
        consume the same UKB chr22 fileset with the same trait + same PC
        covariates, β disagreement should be at the BLAS-precision floor.)

  2. h² delta (TorchGWAS vs LDSC 1.0.1 --h2)
       target |Δ h²| < 5e-3
       (validated at fixture scale by the validation-branch
        validation/external/ldsc/compare.py: TG ↔ LDSC IRWLS port closed
        the intercept gap to ~3e-5; |Δ h²| ~1e-5. The 5e-3 gate leaves
        head-room for biobank-scale jackknife block-boundary drift.)

  3. Peak-RAM invariant
       Asserts the streaming claim O(n × chunk_size): the TG lmm-scan peak
       RSS must be lower than what a re-materialized n × m_chr22 float64
       matrix would require (`n × m_chr22 × 8 B`). Compared also to
       REGENIE Step 1 + Step 2 peak RSS for context.

Outputs:
  - Prints per-comparison report to stdout.
  - Optional --findings-row writes a markdown row to stdout (or to a file)
    formatted for direct append to docs/validation_findings.md.
  - Exit 0 on all-pass, 1 on any failure.

Inputs come from the harness directories:
  torchgwas_outputs/lmm_chr22.assoc.tsv        (TG lmm-scan output)
  torchgwas_outputs/lmm_chr22.time.log         (TG peak RSS via /usr/bin/time)
  torchgwas_outputs/h2_chr22.json              (TG h² via ldsc_h2 driver)
  torchgwas_outputs/h2_chr22.time.log
  reference_outputs/regenie_step2_<trait>.regenie  (REGENIE step 2 output)
  reference_outputs/regenie_step1.time.log
  reference_outputs/regenie_step2.time.log
  reference_outputs/ldsc_h2.log                (LDSC --h2 log)
  reference_outputs/ldsc_h2.time.log

This script intentionally has NO TorchGWAS imports — it operates purely on
the on-disk artifacts the run scripts produced. That keeps the harness
runnable even if the torchgwas package is in a partial install state, and
mirrors the LDSC Pillar B harness's pattern (compare.py as a thin parser +
checker over upstream output files).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Tolerance gates (per Task NA3 success criteria) ────────────────────────
TOL_BETA_PEARSON_R = 0.999
TOL_H2_ABSDIFF = 5e-3
# The streaming-RAM gate is multiplicative: TG peak must be <= ratio * the
# full-materialization size. With chunk_size=1024 and m_chr22 ~= 1.6e6, the
# ratio of (n*chunk*8B) / (n*m*8B) = chunk/m ~= 6.4e-4. We allow a 100x
# slack for K (the n*n kinship is the dominant always-resident cost) and
# Python interpreter overhead.
TOL_TG_PEAK_FRACTION_OF_FULL = 0.10


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    direction: str  # "min" or "max"
    note: str = ""

    def __str__(self) -> str:
        cmp = ">=" if self.direction == "min" else "<="
        flag = "PASS" if self.passed else "FAIL"
        return (
            f"  [{flag}] {self.name:50s} "
            f"observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"
        )


@dataclass
class ComparisonReport:
    name: str
    n_compared: int
    checks: list[CheckResult] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print(self) -> None:
        bar = "=" * 80
        print(bar)
        print(f"  {self.name}  (n_compared = {self.n_compared})")
        print(bar)
        for c in self.checks:
            print(c)
        if self.extras:
            print("  -- extras --")
            for k, v in self.extras.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6e}")
                else:
                    print(f"    {k}: {v}")
        print()


def _check_min(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs >= thr, observed=obs, threshold=thr, direction="min", note=note)


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr, direction="max", note=note)


# ── /usr/bin/time -v parser ────────────────────────────────────────────────
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_WALL_RE = re.compile(r"Elapsed \(wall clock\) time.*:\s*([\d:.]+)")


def parse_time_log(path: Path) -> dict[str, float]:
    """Extract peak RSS (kB) + wall time (s) from a `/usr/bin/time -v` log."""
    if not path.exists():
        return {"peak_rss_kb": float("nan"), "wall_seconds": float("nan")}
    text = path.read_text()
    rss_match = _RSS_RE.search(text)
    wall_match = _WALL_RE.search(text)
    rss_kb = int(rss_match.group(1)) if rss_match else 0
    wall_s = _wall_to_seconds(wall_match.group(1)) if wall_match else float("nan")
    return {"peak_rss_kb": float(rss_kb), "wall_seconds": wall_s}


def _wall_to_seconds(wall_str: str) -> float:
    parts = wall_str.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]


# ── LDSC log parser (lifted from validation-branch ldsc/compare.py) ───────
_VAL_SE_RE = re.compile(r":\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)")
_VAL_RE = re.compile(r":\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def _parse_val_se(line: str) -> tuple[float, float]:
    m = _VAL_SE_RE.search(line)
    if not m:
        raise ValueError(f"cannot parse value(SE) from: {line!r}")
    return float(m.group(1)), float(m.group(2))


def _parse_val(line: str) -> float:
    m = _VAL_RE.search(line)
    if not m:
        raise ValueError(f"cannot parse value from: {line!r}")
    return float(m.group(1))


@dataclass
class LDSCLog:
    h2: float
    h2_se: float
    intercept: float
    intercept_se: float
    mean_chi2: float
    lambda_gc: float


def parse_ldsc_log(path: Path) -> LDSCLog:
    text = path.read_text()
    h2_line = next(line for line in text.splitlines() if line.startswith("Total Observed scale h2:"))
    int_line = next(line for line in text.splitlines() if line.startswith("Intercept:"))
    chi2_line = next(line for line in text.splitlines() if line.startswith("Mean Chi^2:"))
    lgc_line = next(line for line in text.splitlines() if line.startswith("Lambda GC:"))
    h2, h2_se = _parse_val_se(h2_line)
    intercept, intercept_se = _parse_val_se(int_line)
    return LDSCLog(
        h2=h2, h2_se=h2_se,
        intercept=intercept, intercept_se=intercept_se,
        mean_chi2=_parse_val(chi2_line),
        lambda_gc=_parse_val(lgc_line),
    )


# ── REGENIE output parser ──────────────────────────────────────────────────
def load_regenie_step2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if "TEST" in df.columns:
        df = df[df["TEST"] == "ADD"].copy()
    return df


# ── TorchGWAS lmm-scan output parser ───────────────────────────────────────
def load_tg_lmm(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


# ── Comparison 1: β correlation TG vs REGENIE ──────────────────────────────
def compare_beta_correlation(tg_path: Path, regenie_path: Path) -> ComparisonReport:
    tg = load_tg_lmm(tg_path)
    rg = load_regenie_step2(regenie_path)

    # SNP id columns: TG writes `SNP`; REGENIE writes `ID`.
    tg_id_col = "SNP" if "SNP" in tg.columns else tg.columns[0]
    rg_id_col = "ID"
    tg = tg.rename(columns={tg_id_col: "_id"})
    rg = rg.rename(columns={rg_id_col: "_id"})
    tg["_id"] = tg["_id"].astype(str)
    rg["_id"] = rg["_id"].astype(str)

    # Allele check: TG default counts BIM A2; REGENIE counts ALLELE1 = A1.
    # For UKB-scale chr22 we cannot guarantee in this scaffold whether the
    # user's BED has A1/A2 in the order REGENIE expects. Best practice:
    # merge on SNP id, compare β magnitude correlation (sign + magnitude),
    # and warn if Pearson is < 0 (would indicate a global flip).
    merged = pd.merge(
        tg[["_id", "BETA", "SE"]].rename(columns={"BETA": "beta_tg", "SE": "se_tg"}),
        rg[["_id", "BETA", "SE"]].rename(columns={"BETA": "beta_rg", "SE": "se_rg"}),
        on="_id",
        how="inner",
    )

    if len(merged) == 0:
        raise SystemExit(
            "[compare] FAIL: zero SNP ID overlap between TG and REGENIE outputs. "
            "Check that lmm-scan and REGENIE Step 2 saw the same fileset."
        )

    a = merged["beta_tg"].to_numpy(dtype=np.float64)
    b = merged["beta_rg"].to_numpy(dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    r, _p = pearsonr(a, b)

    rep = ComparisonReport(
        name="β correlation TorchGWAS lmm-scan vs REGENIE Step 2 quant",
        n_compared=int(mask.sum()),
    )
    rep.checks.append(_check_min("Pearson r(β_TG, β_REGENIE)", float(r), TOL_BETA_PEARSON_R))
    rep.extras["β median |Δ|"] = float(np.median(np.abs(a - b)))
    rep.extras["β max |Δ|"] = float(np.max(np.abs(a - b)))
    rep.extras["mean |β_TG|"] = float(np.mean(np.abs(a)))
    rep.extras["mean |β_REGENIE|"] = float(np.mean(np.abs(b)))
    if r < 0:
        rep.extras["WARNING"] = "Pearson r is negative — likely allele convention flip; investigate."
    return rep


# ── Comparison 2: h² delta TG vs LDSC ──────────────────────────────────────
def compare_h2(tg_h2_json: Path, ldsc_log_path: Path) -> ComparisonReport:
    tg = json.loads(tg_h2_json.read_text())
    ldsc = parse_ldsc_log(ldsc_log_path)

    h2_diff = abs(tg["h2"] - ldsc.h2)
    int_diff = abs(tg["intercept"] - ldsc.intercept)
    chi2_diff = abs(tg["mean_chi2"] - ldsc.mean_chi2)

    rep = ComparisonReport(
        name="h² TorchGWAS ldsc_h2 vs LDSC --h2 (chr22)",
        n_compared=int(tg.get("n_snps_used", -1)),
    )
    rep.checks.append(_check_max("|Δ h²|", h2_diff, TOL_H2_ABSDIFF))
    # Intercept + mean χ² are diagnostic extras (not gated; LDSC IRWLS port
    # in TG already validated to ~3e-5 absolute on the fixture scale — see
    # validation/external/ldsc/compare.py on the validation branch).
    rep.extras["TG h² (SE)"] = f"{tg['h2']:.4f} ({tg['h2_se']:.4f})"
    rep.extras["LDSC h² (SE)"] = f"{ldsc.h2:.4f} ({ldsc.h2_se:.4f})"
    rep.extras["TG intercept (SE)"] = f"{tg['intercept']:.4f} ({tg['intercept_se']:.4f})"
    rep.extras["LDSC intercept (SE)"] = f"{ldsc.intercept:.4f} ({ldsc.intercept_se:.4f})"
    rep.extras["|Δ intercept|"] = int_diff
    rep.extras["TG mean χ²"] = tg["mean_chi2"]
    rep.extras["LDSC mean χ²"] = ldsc.mean_chi2
    rep.extras["|Δ mean χ²|"] = chi2_diff
    return rep


# ── Comparison 3: Peak-RAM invariant ──────────────────────────────────────
def compare_peak_memory(
    tg_lmm_time_log: Path,
    regenie_step1_time_log: Path,
    regenie_step2_time_log: Path,
    n_samples: int,
    m_variants: int,
) -> ComparisonReport:
    tg = parse_time_log(tg_lmm_time_log)
    rg1 = parse_time_log(regenie_step1_time_log)
    rg2 = parse_time_log(regenie_step2_time_log)

    # Projected full-materialization size (float64 dosage, n × m × 8 B):
    full_bytes = float(n_samples) * float(m_variants) * 8.0
    tg_peak_bytes = tg["peak_rss_kb"] * 1024.0
    tg_fraction_of_full = tg_peak_bytes / max(full_bytes, 1.0)

    rep = ComparisonReport(
        name="Peak RAM streaming invariant (TorchGWAS lmm-scan)",
        n_compared=int(m_variants),
    )
    rep.checks.append(
        _check_max(
            "TG peak / (n × m × 8 B)",
            tg_fraction_of_full,
            TOL_TG_PEAK_FRACTION_OF_FULL,
            note="streaming peak should be far below full-materialization",
        )
    )
    rep.extras["TG lmm-scan peak RSS (GB)"] = tg_peak_bytes / 1e9
    rep.extras["TG lmm-scan wall (s)"] = tg["wall_seconds"]
    rep.extras["REGENIE Step 1 peak RSS (GB)"] = rg1["peak_rss_kb"] * 1024.0 / 1e9
    rep.extras["REGENIE Step 1 wall (s)"] = rg1["wall_seconds"]
    rep.extras["REGENIE Step 2 peak RSS (GB)"] = rg2["peak_rss_kb"] * 1024.0 / 1e9
    rep.extras["REGENIE Step 2 wall (s)"] = rg2["wall_seconds"]
    rep.extras["projected full-materialization (TB)"] = full_bytes / 1e12
    rep.extras["n_samples"] = n_samples
    rep.extras["m_variants"] = m_variants
    return rep


# ── Findings-ledger row formatter (matches docs/validation_findings.md) ───
def render_findings_row(reports: list[ComparisonReport], date_iso: str) -> str:
    """Render a single markdown table row in the canonical findings schema.

    Schema (verified against the existing
    docs/validation_findings.md on the validation branch as of 2026-05-04):
      | Date | Pillar | Tier | Module / Function | Reference | Dataset |
      | Δ observed | Tolerance | F3 class | Resolution |
    """
    beta = next((r for r in reports if r.name.startswith("β correlation")), None)
    h2 = next((r for r in reports if r.name.startswith("h² ")), None)
    mem = next((r for r in reports if r.name.startswith("Peak RAM")), None)

    delta_parts = []
    tol_parts = []
    if beta:
        b_obs = beta.checks[0].observed
        delta_parts.append(f"β Pearson r = {b_obs:.6f}")
        tol_parts.append(f"r ≥ {beta.checks[0].threshold:.3f}")
    if h2:
        d = h2.checks[0].observed
        delta_parts.append(f"|Δ h²| = {d:.3e}")
        tol_parts.append(f"|Δ h²| ≤ {h2.checks[0].threshold:.0e}")
    if mem:
        f = mem.checks[0].observed
        peak_gb = mem.extras.get("TG lmm-scan peak RSS (GB)", 0.0)
        full_tb = mem.extras.get("projected full-materialization (TB)", 0.0)
        delta_parts.append(
            f"TG peak RSS = {peak_gb:.2f} GB ({f * 100:.3f}% of {full_tb:.1f} TB full materialization)"
        )
        tol_parts.append(f"TG peak / full ≤ {mem.checks[0].threshold:.0%}")

    delta = "; ".join(delta_parts) if delta_parts else "_no checks rendered_"
    tol = "; ".join(tol_parts) if tol_parts else "n/a"

    n_pass = sum(r.passed for r in reports)
    n_total = len(reports)
    if n_pass == n_total:
        f3_class = "post-V1 / empirical-validation"
        resolution = (
            "all 3 NA3 gates pass — biobank-scale streaming + numerical agreement "
            "empirically validated."
        )
    else:
        f3_class = "infra-blocker" if n_pass == 0 else "post-V1 / open issue"
        resolution = (
            f"{n_pass}/{n_total} NA3 gates pass; failures need triage. See "
            "validation/external/ukb/compare.py output for per-check detail."
        )

    return (
        f"| {date_iso} | NA3 | UKB | "
        f"`torchgwas.models.SingleTraitLMM` + `torchgwas.postgwas.ldsc_h2` "
        f"| REGENIE Step 1+2 + LDSC 1.0.1 `--h2` "
        f"| UKB chr22 (user-supplied; n × m_chr22) "
        f"| {delta} | {tol} | {f3_class} | {resolution} |"
    )


# ── Driver ─────────────────────────────────────────────────────────────────
def run_all(
    harness_dir: Path,
    n_samples: int,
    m_variants: int,
    trait_col: str,
    json_out: Path | None,
    findings_row: Path | None,
) -> int:
    tg_dir = harness_dir / "torchgwas_outputs"
    ref_dir = harness_dir / "reference_outputs"

    tg_lmm = tg_dir / "lmm_chr22.assoc.tsv"
    tg_lmm_time = tg_dir / "lmm_chr22.time.log"
    tg_h2 = tg_dir / "h2_chr22.json"
    rg_step2 = ref_dir / f"regenie_step2_{trait_col}.regenie"
    rg_step1_time = ref_dir / "regenie_step1.time.log"
    rg_step2_time = ref_dir / "regenie_step2.time.log"
    ldsc_log = ref_dir / "ldsc_h2.log"

    for f in [tg_lmm, tg_h2, rg_step2, ldsc_log]:
        if not f.exists():
            raise SystemExit(f"[compare] FAIL: missing required input {f}")

    reports = [
        compare_beta_correlation(tg_lmm, rg_step2),
        compare_h2(tg_h2, ldsc_log),
        compare_peak_memory(tg_lmm_time, rg_step1_time, rg_step2_time, n_samples, m_variants),
    ]

    print()
    for r in reports:
        r.print()

    n_pass = sum(r.passed for r in reports)
    n_fail = len(reports) - n_pass
    print(f"=== {n_pass}/{len(reports)} comparisons passed ===")

    if json_out is not None:
        json_out.write_text(json.dumps(
            [{
                "name": r.name,
                "n_compared": r.n_compared,
                "passed": r.passed,
                "checks": [asdict(c) for c in r.checks],
                "extras": r.extras,
            } for r in reports],
            indent=2, default=float,
        ))
        print(f"[compare] wrote {json_out}")

    # Findings-ledger row (always emit to stdout; optionally to file)
    today = os.environ.get("UKB_FINDINGS_DATE", "")
    if not today:
        from datetime import date
        today = date.today().isoformat()
    row = render_findings_row(reports, today)
    print()
    print("# Findings-ledger row (append to docs/validation_findings.md):")
    print(row)
    if findings_row is not None:
        findings_row.write_text(row + "\n")
        print(f"[compare] wrote {findings_row}")

    return 0 if n_fail == 0 else 1


def main() -> int:
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness-dir", default=str(HERE),
                    help="harness root (default: directory containing this script)")
    ap.add_argument("--n-samples", type=int, required=True,
                    help="post-QC sample size; should match what TG and REGENIE saw")
    ap.add_argument("--m-variants", type=int, required=True,
                    help="post-QC chr22 variant count; used for the streaming-RAM gate")
    ap.add_argument("--trait-col", default=os.environ.get("UKB_TRAIT_COL", "PHENO"),
                    help="trait column name (must match REGENIE Step 2 output suffix)")
    ap.add_argument("--json", type=str, default=None,
                    help="optional JSON report path")
    ap.add_argument("--findings-row", type=str, default=None,
                    help="optional file to write the findings-ledger row to")
    args = ap.parse_args()

    return run_all(
        harness_dir=Path(args.harness_dir),
        n_samples=args.n_samples,
        m_variants=args.m_variants,
        trait_col=args.trait_col,
        json_out=Path(args.json) if args.json else None,
        findings_row=Path(args.findings_row) if args.findings_row else None,
    )


if __name__ == "__main__":
    sys.exit(main())
