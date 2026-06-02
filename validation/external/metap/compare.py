"""Compare TorchGenomics p-value combination kernels against the R-side
references in outputs/reference.tsv (produced by run_reference.R).

Per-method tolerance gates (observed-then-floored on the first
successful run):

    Fisher         max |Delta p|    <= 1e-10  (closed-form across libraries)
    Stouffer       max |Delta p|    <= 1e-10
    Tippett min-p  max |Delta p|    <= 1e-10
    HMP            max |Delta p|    <= 5e-3   (analytic scale L differs)
    Empirical Brown rel(p)          <= 5e-2   (covariance estimator differs)

Outputs:
    results/summary.tsv     scenario × method × (R_p, TG_p, |Delta p|)
    results/agreement.json  pass/fail per check + tolerance gates
    results/manifest.sha256 fixture + outputs hash manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.postgwas import (  # noqa: E402
    empirical_brown_combined,
    fisher_combined,
    harmonic_mean_p,
    min_p_combined,
    stouffer_combined,
)


TOL_ABS_FISHER = 1e-10
TOL_ABS_STOUFFER = 1e-10
TOL_ABS_MIN_P = 1e-10
TOL_ABS_HMP = 5e-3
TOL_REL_EBM = 5e-2


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:36s} observed={self.observed:.6e} <= {self.threshold:.6e}"


@dataclass
class Report:
    name: str
    n_compared: int
    checks: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _check_max(name, obs, thr):
    return CheckResult(name=name, passed=(obs <= thr),
                       observed=float(obs), threshold=float(thr))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(data_dir: Path, out_dir: Path) -> Report:
    p_df = pd.read_csv(data_dir / "pvalues.tsv", sep="\t")
    data_mat = pd.read_csv(data_dir / "data_matrix.tsv", sep="\t").to_numpy()
    ref_path = out_dir / "reference.tsv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} missing — run run.sh first."
        )
    ref_df = pd.read_csv(ref_path, sep="\t")

    rep = Report(
        name="p-value combination (metap CRAN + EmpiricalBrownsMethod BioC vs torchgenomics.postgwas)",
        n_compared=int(len(ref_df)),
    )

    # Per-row TG outputs, keyed by (scenario, method).
    tg_by_key: dict[tuple[str, str], float] = {}
    for _, row in p_df.iterrows():
        scen = row["scenario"]
        p_vec = row.drop("scenario").to_numpy(dtype=float).tolist()
        _, p_fisher = fisher_combined(p_vec)
        _, p_stouffer = stouffer_combined(p_vec)
        _, p_hmp = harmonic_mean_p(p_vec)
        _, p_minp = min_p_combined(p_vec)
        tg_by_key[(scen, "fisher")] = p_fisher
        tg_by_key[(scen, "stouffer")] = p_stouffer
        tg_by_key[(scen, "hmp")] = p_hmp
        tg_by_key[(scen, "min_p")] = p_minp
        # EBM uses the same data matrix for every scenario.
        _, p_ebm = empirical_brown_combined(
            p_vec, torch.tensor(data_mat, dtype=torch.float64),
        )
        tg_by_key[(scen, "empirical_brown")] = p_ebm

    # Per-method accumulators.
    method_to_tol = {
        "fisher": ("abs", TOL_ABS_FISHER),
        "stouffer": ("abs", TOL_ABS_STOUFFER),
        "min_p": ("abs", TOL_ABS_MIN_P),
        "hmp": ("abs", TOL_ABS_HMP),
        "empirical_brown": ("rel", TOL_REL_EBM),
    }
    per_method_deltas: dict[str, list[float]] = {m: [] for m in method_to_tol}
    rows_out = []
    for _, r in ref_df.iterrows():
        scen = r["scenario"]
        method = r["method"]
        p_R = float(r["p_combined"])
        p_TG = tg_by_key.get((scen, method))
        if p_TG is None:
            continue
        delta = abs(p_R - p_TG)
        rel = delta / max(abs(p_R), 1e-300)
        rows_out.append((scen, method, p_R, p_TG, delta, rel))
        if method in per_method_deltas:
            per_method_deltas[method].append(
                delta if method_to_tol[method][0] == "abs" else rel
            )

    for method, (mode, thr) in method_to_tol.items():
        deltas = per_method_deltas[method]
        if not deltas:
            continue
        max_d = max(deltas)
        label = f"max |Δp| ({method})" if mode == "abs" else f"max rel(p) ({method})"
        rep.checks.append(_check_max(label, max_d, thr))

    rep.extras["per_row"] = [
        {"scenario": s, "method": m, "p_R": pR, "p_TG": pTG,
         "abs_delta": d, "rel_delta": rel}
        for (s, m, pR, pTG, d, rel) in rows_out
    ]
    return rep


def _write_results(rep: Report, results_dir: Path, data_dir: Path,
                   out_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = ["scenario\tmethod\tp_R\tp_TG\tabs_delta\trel_delta"]
    for r in rep.extras.get("per_row", []):
        rows.append("\t".join([
            r["scenario"], r["method"],
            f"{r['p_R']:.10g}", f"{r['p_TG']:.10g}",
            f"{r['abs_delta']:.6e}", f"{r['rel_delta']:.6e}",
        ]))
    (results_dir / "summary.tsv").write_text("\n".join(rows) + "\n")

    (results_dir / "agreement.json").write_text(json.dumps({
        "name": rep.name,
        "n_compared": rep.n_compared,
        "passed": bool(rep.passed),
        "checks": [asdict(c) for c in rep.checks],
        "tolerance_gates": {
            "TOL_ABS_FISHER": TOL_ABS_FISHER,
            "TOL_ABS_STOUFFER": TOL_ABS_STOUFFER,
            "TOL_ABS_MIN_P": TOL_ABS_MIN_P,
            "TOL_ABS_HMP": TOL_ABS_HMP,
            "TOL_REL_EBM": TOL_REL_EBM,
        },
    }, indent=2, default=float))

    manifest = []
    for label, p in [
        ("data/pvalues.tsv", data_dir / "pvalues.tsv"),
        ("data/data_matrix.tsv", data_dir / "data_matrix.tsv"),
        ("data/sim_truth.json", data_dir / "sim_truth.json"),
        ("outputs/reference.tsv", out_dir / "reference.tsv"),
    ]:
        if p.exists():
            manifest.append(f"{_hash_file(p)}  {label}")
        else:
            manifest.append(f"MISSING  {label}")
    (results_dir / "manifest.sha256").write_text("\n".join(manifest) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    rep = compare(Path(args.data_dir), Path(args.out_dir))
    print("=" * 78)
    print(f"  {rep.name}  (n_compared = {rep.n_compared})")
    print("=" * 78)
    for c in rep.checks:
        print(c)
    print()
    _write_results(rep, Path(args.results_dir),
                   Path(args.data_dir), Path(args.out_dir))
    print(f"=== {1 if rep.passed else 0}/1 comparisons passed ===")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
