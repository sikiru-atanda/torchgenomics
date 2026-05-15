"""Compare hyprcoloc R reference vs TorchGWAS on simulated 3-trait sumstats.

Comparison surface (Tier-1 hyprcoloc parity, Foley 2021):
  1. Best-cluster membership   - is the cluster R identifies the same set of
                                  trait indices that TG identifies?
  2. Regional posterior PP_S    - hyprcoloc::regional_prob vs TG
                                  HyprcolocResult.best_cluster_posterior.
  3. Per-SNP PP within cluster - hyprcoloc::posterior_explained_by_snp vs
                                  TG HyprcolocResult.candidate_snp_posterior.
  4. Candidate SNP identity     - hyprcoloc::candidate_snp vs TG
                                  HyprcolocResult.candidate_snp (index in
                                  the data).

Both tools see the *same* data/sumstats.tsv (simulated by simulate.R with
seed 42; 100 markers, 3 traits, one shared causal SNP at index 50 with
beta = 0.5 on all three traits). Disagreement reflects the implementation,
not the data prep.

Tolerance policy (per the harness spec): observed-then-floored.

Notes on expected divergences:

  * Cluster membership: hyprcoloc uses iterative branch-and-bound greedy
    clustering; TG uses exhaustive enumeration of non-singleton subsets. On
    the shared-causal-3-trait fixture both algorithms should identify
    {T1, T2, T3} exactly. We assert 100% agreement on this locus.

  * Regional PP: hyprcoloc and TG both use the Wakefield ABF + Foley 2021
    prior. With matched defaults (prior_1=1e-4, prior_2=1-0.02=0.98,
    prior_w=0.15**2) the regional PP should agree to ~1e-3. The brief
    asked for |Delta PP| <= 1e-3 which we enforce as the floor.

  * Candidate SNP: both tools compute the argmax over the same joint
    log-ABF, so the candidate SNP index must match exactly.
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

# Tolerance gates (observed-then-floored). Spec target: <= 1e-3.
TOL_REGIONAL_PP = 1e-3
TOL_CANDIDATE_PP = 1e-3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.postgwas import SumStats  # noqa: E402
from torchgwas.postgwas._hyprcoloc import hyprcoloc as tg_hyprcoloc  # noqa: E402



@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float | str
    threshold: float | str
    direction: str  # "min" | "max" | "exact"
    note: str = ""

    def __str__(self) -> str:
        if self.direction == "exact":
            cmp = "=="
        elif self.direction == "min":
            cmp = ">="
        else:
            cmp = "<="
        flag = "PASS" if self.passed else "FAIL"
        if isinstance(self.observed, float):
            obs_s = f"{self.observed:.6e}"
        else:
            obs_s = str(self.observed)
        if isinstance(self.threshold, float):
            thr_s = f"{self.threshold:.6e}"
        else:
            thr_s = str(self.threshold)
        return f"  [{flag}] {self.name:42s} observed={obs_s} {cmp} {thr_s}  {self.note}"


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


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs,
                       threshold=thr, direction="max", note=note)


def _check_exact(name: str, obs: Any, expected: Any, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs == expected, observed=str(obs),
                       threshold=str(expected), direction="exact", note=note)


def _load_inputs(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Load the simulated sumstats + truth + R reference output."""
    sumstats = pd.read_csv(data_dir / "sumstats.tsv", sep="\t")
    truth = json.loads((data_dir / "sim_truth.json").read_text())
    r_results = json.loads((out_dir / "hyprcoloc_results.json").read_text())

    trait_names = truth["trait_names"]
    snp_ids = sumstats.loc[sumstats["trait"] == trait_names[0], "SNP"].astype(str).tolist()
    m = truth["m"]
    K = truth["k"]
    assert len(snp_ids) == m

    # Build (K, m) beta and se arrays, preserving snp_ids order.
    beta = np.zeros((K, m), dtype=np.float64)
    se = np.zeros((K, m), dtype=np.float64)
    for k, name in enumerate(trait_names):
        sub = sumstats[sumstats["trait"] == name].set_index("SNP").loc[snp_ids]
        beta[k] = sub["beta"].to_numpy(dtype=np.float64)
        se[k] = sub["se"].to_numpy(dtype=np.float64)

    # SumStats wants ploidy-invariant columns; stub chr/pos/n.
    common = dict(
        chr=["1"] * m, pos=list(range(1, m + 1)),
        a1=["A"] * m, a2=["G"] * m,
        n=torch.tensor([10000] * m, dtype=torch.float64),
        af=None,
    )

    sumstats_list = []
    for k in range(K):
        z = beta[k] / np.clip(se[k], 1e-12, None)
        # Approximate two-sided p from |z| (irrelevant to hyprcoloc but
        # required by SumStats).
        from scipy.special import erfc
        p = erfc(np.abs(z) / np.sqrt(2.0))
        sumstats_list.append(SumStats(
            beta=torch.tensor(beta[k], dtype=torch.float64),
            se=torch.tensor(se[k], dtype=torch.float64),
            p=torch.tensor(p, dtype=torch.float64),
            snp=list(snp_ids),
            **common,
        ))

    return {
        "sumstats_list": sumstats_list,
        "snp_ids": snp_ids,
        "trait_names": trait_names,
        "m": m,
        "K": K,
        "truth": truth,
        "r_results": r_results,
    }


def compare_hyprcoloc(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """Multi-trait hyprcoloc parity (R vs TorchGWAS) on shared-causal sim."""
    inp = _load_inputs(data_dir, out_dir)
    r_results = inp["r_results"]

    # R reference: take the first (and expected only) cluster row.
    if not r_results["results"]:
        raise RuntimeError("R hyprcoloc returned no clusters; sim broken?")
    r_row = r_results["results"][0]
    r_cluster_zero_based = sorted(r_row["traits_zero_based"])
    r_regional_pp = float(r_row["regional_prob"])
    r_candidate_snp = str(r_row["candidate_snp"])
    r_candidate_pp = float(r_row["posterior_explained_by_snp"])

    # TorchGWAS run on the same sumstats.
    # Prior reparameterization: hyprcoloc::prior.c = 0.02 -> TG prior_2 = 0.98.
    tg = tg_hyprcoloc(
        inp["sumstats_list"],
        prior_1=1e-4,
        prior_2=0.98,
        prior_w=0.15 ** 2,
        trait_names=inp["trait_names"],
    )
    tg_cluster_zero_based = sorted(tg.best_cluster)
    tg_regional_pp = float(tg.best_cluster_posterior)
    tg_candidate_idx = int(tg.candidate_snp)
    tg_candidate_snp = inp["snp_ids"][tg_candidate_idx]
    tg_candidate_pp = float(tg.candidate_snp_posterior)

    # Cluster-membership agreement (set equality).
    cluster_match = (set(r_cluster_zero_based) == set(tg_cluster_zero_based))
    cluster_pct = 100.0 if cluster_match else 0.0

    # Posterior diffs.
    d_regional = abs(tg_regional_pp - r_regional_pp)
    d_candidate = abs(tg_candidate_pp - r_candidate_pp)

    rep = ComparisonReport(
        name="hyprcoloc (R hyprcoloc vs TorchGWAS hyprcoloc)",
        n_compared=inp["m"],
    )
    rep.checks.append(_check_exact(
        "cluster membership (zero-based)",
        tg_cluster_zero_based, r_cluster_zero_based,
    ))
    rep.checks.append(_check_max(
        "|Delta regional_pp|", d_regional, TOL_REGIONAL_PP,
    ))
    rep.checks.append(_check_max(
        "|Delta candidate_snp_pp|", d_candidate, TOL_CANDIDATE_PP,
    ))
    rep.checks.append(_check_exact(
        "candidate SNP id", tg_candidate_snp, r_candidate_snp,
    ))
    rep.extras["R cluster"]          = r_row["traits_str"]
    rep.extras["TG cluster"]         = ", ".join(
        inp["trait_names"][i] for i in tg_cluster_zero_based
    )
    rep.extras["R regional_pp"]      = r_regional_pp
    rep.extras["TG regional_pp"]     = tg_regional_pp
    rep.extras["R candidate SNP"]    = r_candidate_snp
    rep.extras["TG candidate SNP"]   = tg_candidate_snp
    rep.extras["R per-SNP PP"]       = r_candidate_pp
    rep.extras["TG per-SNP PP"]      = tg_candidate_pp
    rep.extras["truth causal SNP"]   = inp["truth"]["causal_snp"]
    rep.extras["truth cluster (zero-based)"] = inp["truth"]["expected_cluster_zero_based"]
    rep.extras["cluster-assignment agreement"] = f"{cluster_pct:.1f}%"
    return rep


def _write_results(rep: ComparisonReport, out_dir: Path,
                   r_results: dict[str, Any]) -> None:
    """Persist summary.tsv, agreement.json, manifest.sha256 to results/."""
    res_dir = HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # summary.tsv (one row per check)
    summary_rows = [{
        "metric": c.name,
        "observed": (f"{c.observed:.6e}" if isinstance(c.observed, float) else str(c.observed)),
        "threshold": (f"{c.threshold:.6e}" if isinstance(c.threshold, float) else str(c.threshold)),
        "direction": c.direction,
        "passed": int(c.passed),
    } for c in rep.checks]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(res_dir / "summary.tsv", sep="\t", index=False)

    # agreement.json (full structured report)
    agreement = {
        "name": rep.name,
        "n_compared": rep.n_compared,
        "passed": rep.passed,
        "checks": [asdict(c) for c in rep.checks],
        "extras": rep.extras,
        "tool_versions": r_results["package_versions"],
    }
    (res_dir / "agreement.json").write_text(
        json.dumps(agreement, indent=2, default=str)
    )

    # manifest.sha256 (record SHA256 of the data + outputs files)
    import hashlib

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    manifest_targets = [
        (HERE / "data" / "sumstats.tsv"),
        (HERE / "data" / "sim_truth.json"),
        (HERE / "outputs" / "hyprcoloc_results.json"),
        (res_dir / "summary.tsv"),
        (res_dir / "agreement.json"),
    ]
    manifest_lines = []
    for p in manifest_targets:
        if p.exists():
            manifest_lines.append(f"{_sha256(p)}  {p.relative_to(HERE)}")
    (res_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n")

    print(f"[compare] wrote {res_dir}/summary.tsv")
    print(f"[compare] wrote {res_dir}/agreement.json")
    print(f"[compare] wrote {res_dir}/manifest.sha256")


def run_all(data_dir: Path, out_dir: Path, json_out: Path | None = None) -> int:
    rep = compare_hyprcoloc(data_dir, out_dir)
    print()
    rep.print()

    n_pass = int(rep.passed)
    n_fail = 1 - n_pass
    print(f"=== {n_pass}/1 comparisons passed ===")

    # Always write results/ even on failure (the audit trail must be complete).
    r_results = json.loads((out_dir / "hyprcoloc_results.json").read_text())
    _write_results(rep, out_dir, r_results)

    if json_out is not None:
        json_out.write_text(json.dumps(
            {
                "name": rep.name,
                "n_compared": rep.n_compared,
                "passed": rep.passed,
                "checks": [asdict(c) for c in rep.checks],
                "extras": rep.extras,
            },
            indent=2, default=str,
        ))
        print(f"[compare] wrote {json_out}")

    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--json", default=None, help="optional JSON report path")
    args = ap.parse_args()
    return run_all(
        Path(args.data_dir),
        Path(args.out_dir),
        Path(args.json) if args.json else None,
    )


if __name__ == "__main__":
    sys.exit(main())
