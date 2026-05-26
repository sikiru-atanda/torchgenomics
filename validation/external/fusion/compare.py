"""Compare FUSION measured-expression reference vs torchgwas.postgwas.
twas_observed_expression on the same fixture.

The reference (validation/external/fusion/run_reference.R) implements
the OLS Wald for the gene-expression coefficient in
``y ~ intercept + covariates + expression_g``. TorchGWAS reaches the
same Wald via ``torchgwas.models.glm.GLM`` driven by
``twas_observed_expression``. Both paths use F(1, n-p) p-values from
the residual variance estimator, so we expect agreement at FP precision.

Tolerance gates (observed-then-floored):
    TOL_REL_BETA          rel(beta)      <= 1e-8
    TOL_REL_SE            rel(se)        <= 1e-8
    TOL_ABS_DELTA_Z       |Δ z_twas|     <= 1e-6
    TOL_ABS_DELTA_NEGLOG10_P |Δ -log10 p| <= 1e-6

Outputs:
    results/summary.tsv     per-gene side-by-side
    results/agreement.json  pass/fail + tolerance gates + extras
    results/manifest.sha256 fixture + output SHA256 manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.postgwas import twas_observed_expression  # noqa: E402


TOL_REL_BETA = 1e-8
TOL_REL_SE = 1e-8
TOL_ABS_DELTA_Z = 1e-6
TOL_ABS_DELTA_NEGLOG10_P = 1e-6


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    note: str = ""

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (f"  [{flag}] {self.name:36s} observed={self.observed:.6e} "
                f"<= {self.threshold:.6e} {self.note}")


@dataclass
class ComparisonReport:
    name: str
    n_compared: int
    checks: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

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
    return CheckResult(name=name, passed=(obs <= thr),
                       observed=obs, threshold=thr, note=note)


def _safe_neglog10(p):
    p = float(p)
    if p <= 0.0 or math.isnan(p):
        return float("inf")
    return -math.log10(p)


def _read_expression(path: Path):
    df = pd.read_csv(path, sep="\t")
    sample_ids = df.iloc[:, 0].astype(str).tolist()
    gene_ids = list(df.columns[1:])
    matrix = torch.tensor(df.iloc[:, 1:].to_numpy(np.float64),
                          dtype=torch.float64)
    return matrix, sample_ids, gene_ids


def _read_phenotype(path: Path):
    df = pd.read_csv(path, sep="\t")
    if "IID" in df.columns:
        sample_ids = df["IID"].astype(str).tolist()
        y = torch.tensor(df["TRAIT"].to_numpy(np.float64),
                         dtype=torch.float64)
    else:
        sample_ids = df.iloc[:, 0].astype(str).tolist()
        y = torch.tensor(df.iloc[:, 1].to_numpy(np.float64),
                         dtype=torch.float64)
    return y, sample_ids


def _read_covariates(path: Path):
    df = pd.read_csv(path, sep="\t")
    if {"FID", "IID"} <= set(df.columns):
        sample_ids = df["IID"].astype(str).tolist()
        data = df.drop(columns=["FID", "IID"]).to_numpy(np.float64)
    else:
        sample_ids = df.iloc[:, 0].astype(str).tolist()
        data = df.iloc[:, 1:].to_numpy(np.float64)
    return torch.tensor(data, dtype=torch.float64), sample_ids


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_fusion(data_dir: Path, out_dir: Path) -> ComparisonReport:
    truth = json.loads((data_dir / "sim_truth.json").read_text())
    fusion_df = pd.read_csv(out_dir / "fusion_results.tsv", sep="\t")
    expr, expr_ids, gene_ids = _read_expression(data_dir / "expression.tsv")
    y, pheno_ids = _read_phenotype(data_dir / "phenotype.tsv")
    cov, cov_ids = _read_covariates(data_dir / "covariates.tsv")

    # Align: all three should already match on sample order, but be defensive.
    assert expr_ids == pheno_ids == cov_ids, (
        "Fixture sample order mismatch — regenerate with simulate_fixture.py."
    )

    tg_res = twas_observed_expression(expr, y, gene_ids, covariates=cov)
    tg_by_gene = {g.gene_id: g for g in tg_res.genes}
    fusion_by_gene = {row["gene_id"]: row for _, row in fusion_df.iterrows()}

    common = sorted(set(tg_by_gene) & set(fusion_by_gene))
    rep = ComparisonReport(
        name="FUSION measured-expression vs torchgwas.postgwas.twas_observed_expression",
        n_compared=len(common),
    )

    rel_betas = []
    rel_ses = []
    abs_dzs = []
    abs_dnegs = []
    for gene in common:
        tg = tg_by_gene[gene]
        ref = fusion_by_gene[gene]
        b_ref = float(ref["beta"])
        b_tg = float(tg.beta)
        s_ref = float(ref["se"])
        s_tg = float(tg.se)
        z_ref = float(ref["z_twas"])
        z_tg = float(tg.z_twas)
        p_ref = float(ref["p_twas"])
        p_tg = float(tg.p_twas)
        rel_betas.append(abs(b_ref - b_tg) / max(abs(b_ref), 1e-300))
        rel_ses.append(abs(s_ref - s_tg) / max(abs(s_ref), 1e-300))
        abs_dzs.append(abs(z_ref - z_tg))
        abs_dnegs.append(abs(_safe_neglog10(p_ref) - _safe_neglog10(p_tg)))

    rep.checks.append(_check_max(
        "max |Δβ|/|β|", max(rel_betas) if rel_betas else 0.0,
        TOL_REL_BETA,
    ))
    rep.checks.append(_check_max(
        "max |ΔSE|/|SE|", max(rel_ses) if rel_ses else 0.0,
        TOL_REL_SE,
    ))
    rep.checks.append(_check_max(
        "max |Δz_TWAS|", max(abs_dzs) if abs_dzs else 0.0,
        TOL_ABS_DELTA_Z,
    ))
    rep.checks.append(_check_max(
        "max |Δ-log10 p_TWAS|", max(abs_dnegs) if abs_dnegs else 0.0,
        TOL_ABS_DELTA_NEGLOG10_P,
    ))

    rep.extras["truth_causal_gene"] = truth["causal_gene_id"]
    rep.extras["truth_effect"] = truth["true_effect"]
    rep.extras["pearson_r_beta"] = float(np.corrcoef(
        [float(tg_by_gene[g].beta) for g in common],
        [float(fusion_by_gene[g]["beta"]) for g in common],
    )[0, 1])
    return rep


def _write_results(rep: ComparisonReport, results_dir: Path,
                   data_dir: Path, out_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = ["check\tobserved\tthreshold\tpassed"]
    for c in rep.checks:
        rows.append(f"{c.name}\t{c.observed:.6e}\t{c.threshold:.6e}\t{c.passed}")
    (results_dir / "summary.tsv").write_text("\n".join(rows) + "\n")

    (results_dir / "agreement.json").write_text(json.dumps({
        "name": rep.name,
        "n_compared": rep.n_compared,
        "passed": bool(rep.passed),
        "checks": [asdict(c) for c in rep.checks],
        "extras": rep.extras,
        "tolerance_gates": {
            "TOL_REL_BETA": TOL_REL_BETA,
            "TOL_REL_SE": TOL_REL_SE,
            "TOL_ABS_DELTA_Z": TOL_ABS_DELTA_Z,
            "TOL_ABS_DELTA_NEGLOG10_P": TOL_ABS_DELTA_NEGLOG10_P,
        },
    }, indent=2, default=float))

    manifest = []
    for label, p in [
        ("data/expression.tsv", data_dir / "expression.tsv"),
        ("data/phenotype.tsv", data_dir / "phenotype.tsv"),
        ("data/covariates.tsv", data_dir / "covariates.tsv"),
        ("data/genes.bed", data_dir / "genes.bed"),
        ("data/sim_truth.json", data_dir / "sim_truth.json"),
        ("outputs/fusion_results.tsv", out_dir / "fusion_results.tsv"),
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

    rep = compare_fusion(Path(args.data_dir), Path(args.out_dir))
    rep.print()
    _write_results(rep, Path(args.results_dir),
                   Path(args.data_dir), Path(args.out_dir))
    print(f"=== {1 if rep.passed else 0}/1 comparisons passed ===")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
