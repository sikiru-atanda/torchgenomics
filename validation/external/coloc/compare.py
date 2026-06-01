"""Compare R `coloc::coloc.abf` vs `torchgwas.postgwas.coloc_pairwise`.

Three scenarios (one per branch of the H0..H4 decomposition):
  1. shared    -- both traits share a causal variant -> H4 dominant
  2. distinct  -- each trait has its own causal variant -> H3 dominant
  3. null      -- neither trait carries any signal -> H0 dominant

Both tools read the *same* per-scenario sumstats TSV (simulated by
`fetch_data.sh` with seed = 42, M = 50 SNPs, planted causal at idx 25),
so disagreement reflects the implementation only.

Tolerance policy: observed-then-floored. Both implementations use the
closed-form Wakefield ABF (Wakefield 2009) with identical priors
(p1 = p2 = 1e-4, p12 = 1e-5, W = 0.15^2). Agreement should be near-FP.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

TOL_PP = 5e-3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.postgwas import (  # noqa: E402
    SumStats,
    coloc_pairwise,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    direction: str
    note: str = ""

    def __str__(self) -> str:
        cmp = ">=" if self.direction == "min" else "<="
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:42s} observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"


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
        bar = "=" * 78
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


def _check_max(name, obs, thr, note=""):
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr, direction="max", note=note)


def _check_eq(name, lhs, rhs, note=""):
    obs = 0.0 if lhs == rhs else 1.0
    return CheckResult(name=name, passed=lhs == rhs, observed=obs, threshold=0.0,
                        direction="max", note=note or f"{lhs!r} vs {rhs!r}")


def _phi(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _load_scenario(data_dir, file_name):
    df = pd.read_csv(data_dir / file_name, sep="\t")
    M = len(df)
    snp = df["SNP"].astype(str).tolist()

    b1 = torch.tensor(df["beta1"].to_numpy(np.float64), dtype=torch.float64)
    s1 = torch.tensor(df["se1"].to_numpy(np.float64), dtype=torch.float64)
    b2 = torch.tensor(df["beta2"].to_numpy(np.float64), dtype=torch.float64)
    s2 = torch.tensor(df["se2"].to_numpy(np.float64), dtype=torch.float64)

    z1 = (b1 / s1).numpy()
    z2 = (b2 / s2).numpy()
    p1 = 2.0 * (1.0 - _phi(np.abs(z1)))
    p2 = 2.0 * (1.0 - _phi(np.abs(z2)))

    common = dict(
        chr=df["chr"].astype(str).tolist(),
        pos=df["pos"].astype(int).tolist(),
        snp=snp,
        a1=df["a1"].astype(str).tolist(),
        a2=df["a2"].astype(str).tolist(),
        n=torch.tensor([10000] * M, dtype=torch.float64),
        af=None,
    )
    ss1 = SumStats(beta=b1, se=s1, p=torch.tensor(p1, dtype=torch.float64), **common)
    ss2 = SumStats(beta=b2, se=s2, p=torch.tensor(p2, dtype=torch.float64), **common)
    return ss1, ss2, snp


def _compare(scenario_name, ss1, ss2, snps, r_block):
    p1_pr  = r_block["priors"]["p1"]
    p2_pr  = r_block["priors"]["p2"]
    p12_pr = r_block["priors"]["p12"]
    pw_pr  = r_block["priors"]["prior_w"]

    tg = coloc_pairwise(
        ss1, ss2,
        prior_1=p1_pr, prior_2=p2_pr, prior_12=p12_pr, prior_w=pw_pr,
    )

    r_pp = [
        r_block["PP.H0.abf"], r_block["PP.H1.abf"], r_block["PP.H2.abf"],
        r_block["PP.H3.abf"], r_block["PP.H4.abf"],
    ]
    tg_pp = [tg.pp_h0, tg.pp_h1, tg.pp_h2, tg.pp_h3, tg.pp_h4]
    deltas = [abs(a - b) for a, b in zip(r_pp, tg_pp)]
    max_d  = max(deltas)

    tg_cand_id = snps[tg.candidate_snp] if 0 <= tg.candidate_snp < len(snps) else None
    r_cand_id  = r_block["candidate_snp_id"]

    rep = ComparisonReport(
        name=f"coloc.abf vs TG coloc_pairwise -- scenario={scenario_name!r}",
        n_compared=r_block["nsnps"],
    )
    rep.checks.append(_check_max("max |Delta PP.H0..H4|", max_d, TOL_PP))
    rep.checks.append(_check_max("|Delta PP.H0|", deltas[0], TOL_PP))
    rep.checks.append(_check_max("|Delta PP.H1|", deltas[1], TOL_PP))
    rep.checks.append(_check_max("|Delta PP.H2|", deltas[2], TOL_PP))
    rep.checks.append(_check_max("|Delta PP.H3|", deltas[3], TOL_PP))
    rep.checks.append(_check_max("|Delta PP.H4|", deltas[4], TOL_PP))
    rep.checks.append(_check_eq("candidate_snp id", tg_cand_id, r_cand_id))

    rep.extras["R PP.H0..H4"]  = [float(x) for x in r_pp]
    rep.extras["TG PP.H0..H4"] = [float(x) for x in tg_pp]
    rep.extras["R candidate_snp"]  = r_cand_id
    rep.extras["TG candidate_snp"] = tg_cand_id
    rep.extras["TG candidate_idx_0based"] = int(tg.candidate_snp)
    rep.extras["R candidate_idx_0based"]  = r_block["candidate_snp_index_zero_based"]
    rep.extras["expected_dominant"] = r_block.get("expected_pp")
    rep.extras["priors"] = dict(p1=p1_pr, p2=p2_pr, p12=p12_pr, prior_w=pw_pr)
    return rep


def run_all(data_dir, out_dir, results_dir=None):
    r_path = out_dir / "coloc_results.json"
    if not r_path.exists():
        print(f"[compare] ABORT: R reference output not found at {r_path}")
        print("[compare] Run install.sh + fetch_data.sh + run.sh first.")
        return 2
    r_results = json.loads(r_path.read_text())

    manifest = json.loads((data_dir / "scenarios.json").read_text())
    scenarios = manifest["scenarios"]

    reports = []
    for scenario_name, scenario_meta in scenarios.items():
        ss1, ss2, snps = _load_scenario(data_dir, scenario_meta["file"])
        rep = _compare(scenario_name, ss1, ss2, snps, r_results[scenario_name])
        reports.append(rep)

    print()
    for r in reports:
        r.print()

    r_all  = np.array([[r.extras["R PP.H0..H4"][k]  for k in range(5)] for r in reports]).reshape(-1)
    tg_all = np.array([[r.extras["TG PP.H0..H4"][k] for k in range(5)] for r in reports]).reshape(-1)
    if r_all.std() > 0 and tg_all.std() > 0:
        pearson = float(np.corrcoef(r_all, tg_all)[0, 1])
    else:
        pearson = float("nan")
    n_pass = sum(r.passed for r in reports)
    n_fail = len(reports) - n_pass
    print(f"=== {n_pass}/{len(reports)} scenarios passed; Pearson r(R PP, TG PP) = {pearson:.10f} ===")

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for rep, (scenario_name, _) in zip(reports, scenarios.items()):
            r_pp  = rep.extras["R PP.H0..H4"]
            tg_pp = rep.extras["TG PP.H0..H4"]
            for k in range(5):
                rows.append({
                    "scenario": scenario_name,
                    "hypothesis": f"PP.H{k}",
                    "r_coloc": r_pp[k],
                    "torchgwas": tg_pp[k],
                    "delta": abs(r_pp[k] - tg_pp[k]),
                })
        pd.DataFrame(rows).to_csv(results_dir / "summary.tsv", sep="\t", index=False)
        print(f"[compare] wrote {results_dir}/summary.tsv")

        agreement = {
            "name": "R coloc::coloc.abf vs torchgwas.postgwas.coloc_pairwise (3 scenarios)",
            "n_compared": sum(r.n_compared for r in reports),
            "passed": all(r.passed for r in reports),
            "reports": [
                {
                    "name": r.name,
                    "n_compared": r.n_compared,
                    "passed": r.passed,
                    "checks": [asdict(c) for c in r.checks],
                    "extras": r.extras,
                }
                for r in reports
            ],
            "cross_scenario_pearson_r_on_PP": pearson,
            "package_versions": r_results.get("package_versions", {}),
            "priors": r_results.get("defaults", {}),
            "tolerance_PP": TOL_PP,
        }
        (results_dir / "agreement.json").write_text(json.dumps(agreement, indent=2, default=float))
        print(f"[compare] wrote {results_dir}/agreement.json")

        import hashlib
        manifest_lines = []
        for rel in (
            "data/scenarios.json",
            "data/sumstats_shared.tsv",
            "data/sumstats_distinct.tsv",
            "data/sumstats_null.tsv",
            "outputs/coloc_results.json",
        ):
            p = HERE / rel
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                manifest_lines.append(f"{h}  {rel}")
        (results_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n")
        print(f"[compare] wrote {results_dir}/manifest.sha256")

    return 0 if n_fail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",    default=str(HERE / "data"))
    ap.add_argument("--out-dir",     default=str(HERE / "outputs"))
    ap.add_argument("--results-dir", default=str(HERE / "results"),
                    help="directory to write summary.tsv / agreement.json / manifest.sha256")
    args = ap.parse_args()
    return run_all(
        Path(args.data_dir),
        Path(args.out_dir),
        Path(args.results_dir) if args.results_dir else None,
    )


if __name__ == "__main__":
    sys.exit(main())
