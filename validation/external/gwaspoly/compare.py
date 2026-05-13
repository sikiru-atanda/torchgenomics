"""Compare GWASpoly vs TorchGWAS on tetraploid potato across 5 gene-action models.

GWASpoly is the canonical reference for polyploid GWAS (Rosyara et al. 2016).
This harness re-runs the existing pre-Pillar-A comparison (mirroring
tests/test_golden_gwaspoly.py) end-to-end:

  1. additive       (TG additive   vs GWASpoly additive)
  2. 1-dom          (TG 1-dom      vs GWASpoly 1-dom-alt)
  3. 2-dom          (TG 2-dom      vs GWASpoly 2-dom-alt)
  4. 3-dom          (TG 3-dom      vs GWASpoly 2-dom-ref)
  5. diplo-additive (TG diplo-add  vs GWASpoly diplo-additive — different
                     encoding, asserts moderate corr only)

P3D approach: single REML fit on additive K (expanded to obs-level via
Z@K@Z'), per-model genotype recoding for the scan. Mirrors tests/test_golden_gwaspoly.py.

Tolerance gates anchor in docs/validation.md §16.
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

PLOIDY = 4

# ── Tolerance gates (mirror tests/test_golden_gwaspoly.py + §16) ─────────────
TOL_ADDITIVE_LOGP_CORR = 0.999
TOL_1DOM_LOGP_CORR = 0.999
TOL_2DOM_LOGP_CORR = 0.999
TOL_3DOM_LOGP_CORR = 0.999
# diplo-additive uses different encoding between GWASpoly and TG —
# GWASpoly: {0,1,2,3,4} → {0,1,1,1,2} (dichotomous diploidization)
# TG:       {0,1,2,3,4} → {0,1,2,1,0} (min(dose, ploidy-dose))
# Existing golden contract is 0.3 < r < 0.7
TOL_DIPLO_ADD_MIN = 0.3
TOL_DIPLO_ADD_MAX = 0.7

# Minimum number of common markers per comparison
TOL_MIN_MARKERS = 5000


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.config import STAT_DTYPE, NumericalConfig  # noqa: E402
from torchgwas.linalg.kinship_polyploid import grm_polyploid_gene_action  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.single_trait_lmm import SingleTraitLMM  # noqa: E402
from torchgwas.preprocess.polyploid import recode_gene_action  # noqa: E402


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    direction: str
    note: str = ""

    def __str__(self) -> str:
        cmp = ">=" if self.direction == "min" else "<=" if self.direction == "max" else "in range"
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:50s} observed={self.observed:.6e} {cmp} {self.threshold!r}  {self.note}"


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


def _check_in_range(name: str, obs: float, lo: float, hi: float, note: str = "") -> CheckResult:
    return CheckResult(
        name=name,
        passed=(obs >= lo) and (obs <= hi),
        observed=obs,
        threshold=(lo, hi),
        direction="range",
        note=note,
    )


def _logp_corr(p_torch: np.ndarray, snp_names: list, gwaspoly_dict: dict) -> tuple[float, int]:
    p1, p2 = [], []
    for i, snp in enumerate(snp_names):
        if snp in gwaspoly_dict and np.isfinite(p_torch[i]) and p_torch[i] > 0:
            gp = gwaspoly_dict[snp]
            if np.isfinite(gp) and gp > 0:
                p1.append(p_torch[i])
                p2.append(gp)
    if len(p1) < 10:
        return 0.0, 0
    logp1 = -np.log10(np.array(p1))
    logp2 = -np.log10(np.array(p2))
    return float(np.corrcoef(logp1, logp2)[0, 1]), len(p1)


# ── Data loader (mirrors tests/test_golden_gwaspoly.py) ──────────────────────


def _load_data(data_dir: Path) -> tuple:
    pheno = pd.read_csv(data_dir / "potato_pheno_with_genoid.csv")
    geno = pd.read_csv(data_dir / "potato_geno_aligned.csv", index_col=0)
    marker_map = pd.read_csv(data_dir / "potato_map.csv")

    geno.index = geno.index.astype(str)
    pheno = pheno.dropna(subset=["vine.maturity"]).copy()
    pheno["geno_id"] = pheno["geno_id"].astype(str)
    pheno = pheno[pheno["geno_id"].isin(geno.index)].reset_index(drop=True)

    unique_genos = sorted(pheno["geno_id"].unique())
    geno_sub = geno.loc[unique_genos]
    n_obs = len(pheno)
    n_geno = len(unique_genos)

    geno_id_to_idx = {gid: i for i, gid in enumerate(unique_genos)}
    Z = torch.zeros(n_obs, n_geno, dtype=STAT_DTYPE)
    for i, gid in enumerate(pheno["geno_id"]):
        Z[i, geno_id_to_idx[gid]] = 1.0

    env_dummies = pd.get_dummies(pheno["env"], drop_first=True, dtype=float)
    X0 = torch.tensor(
        np.column_stack([np.ones(n_obs), env_dummies.values]), dtype=STAT_DTYPE
    )

    G_geno = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G_geno.shape[1]):
        col = G_geno[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G_geno[mask, j] = col[~mask].mean()

    G_obs = Z @ G_geno
    Y = torch.tensor(pheno["vine.maturity"].values, dtype=STAT_DTYPE)

    snp_names = list(geno_sub.columns)
    chrs = marker_map["chrom"].astype(str).tolist()
    pos = marker_map["pos"].astype(int).tolist()
    vmeta = VariantMeta(
        snp=snp_names, chr=chrs, pos=pos,
        a1=["A"] * len(snp_names), a2=["G"] * len(snp_names),
    )
    return Y, X0, G_obs, G_geno, Z, vmeta, snp_names


def _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, model: str) -> np.ndarray:
    """P3D approach: single null fit with additive K, per-model scan recoding."""
    K_geno, _ = grm_polyploid_gene_action(G_geno, "additive", PLOIDY)
    K_obs = Z @ K_geno @ Z.T
    G_scan = recode_gene_action(G_obs, model, PLOIDY) if model != "additive" else G_obs
    config = NumericalConfig(reml_method="emma")
    lmm = SingleTraitLMM(config=config)
    nf = lmm.fit_null(Y, X0, K=K_obs)
    result = lmm.score_chunk(G_scan, nf, vmeta, test="wald")
    return result.p.detach().cpu().numpy()


def _load_ref(out_dir: Path, model_fname: str) -> dict:
    df = pd.read_csv(out_dir / f"gwaspoly_{model_fname}.csv")
    return dict(zip(df["marker"], df["pvalue"]))


# ── Comparisons ──────────────────────────────────────────────────────────────


def _compare_one(
    data_dir: Path, out_dir: Path,
    tg_model: str, ref_fname: str,
    floor: float, label: str,
    cached_data=None,
) -> ComparisonReport:
    Y, X0, G_obs, G_geno, Z, vmeta, snp_names = cached_data if cached_data else _load_data(data_dir)
    p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, tg_model)
    ref = _load_ref(out_dir, ref_fname)
    r, n = _logp_corr(p, snp_names, ref)

    rep = ComparisonReport(
        name=f"{label}: TG {tg_model} vs GWASpoly {ref_fname}",
        n_compared=n,
    )
    rep.checks.append(_check_min(f"Common markers (n)", n, TOL_MIN_MARKERS))
    rep.checks.append(_check_min(f"−log10(p) correlation", r, floor))
    return rep


def compare_additive(data_dir: Path, out_dir: Path, cached_data=None) -> ComparisonReport:
    return _compare_one(
        data_dir, out_dir, "additive", "additive",
        TOL_ADDITIVE_LOGP_CORR, "Additive", cached_data=cached_data,
    )


def compare_1dom(data_dir: Path, out_dir: Path, cached_data=None) -> ComparisonReport:
    return _compare_one(
        data_dir, out_dir, "1-dom", "1_dom_alt",
        TOL_1DOM_LOGP_CORR, "1-dom", cached_data=cached_data,
    )


def compare_2dom(data_dir: Path, out_dir: Path, cached_data=None) -> ComparisonReport:
    return _compare_one(
        data_dir, out_dir, "2-dom", "2_dom_alt",
        TOL_2DOM_LOGP_CORR, "2-dom", cached_data=cached_data,
    )


def compare_3dom(data_dir: Path, out_dir: Path, cached_data=None) -> ComparisonReport:
    return _compare_one(
        data_dir, out_dir, "3-dom", "2_dom_ref",
        TOL_3DOM_LOGP_CORR, "3-dom", cached_data=cached_data,
    )


def compare_diplo_additive(data_dir: Path, out_dir: Path, cached_data=None) -> ComparisonReport:
    """Diplo-additive: GWASpoly and TG use different encodings → moderate corr only."""
    Y, X0, G_obs, G_geno, Z, vmeta, snp_names = cached_data if cached_data else _load_data(data_dir)
    p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "diplo-additive")
    ref = _load_ref(out_dir, "diplo_additive")
    r, n = _logp_corr(p, snp_names, ref)
    rep = ComparisonReport(
        name="Diplo-additive: TG diplo-additive vs GWASpoly diplo_additive (encoding differs)",
        n_compared=n,
    )
    rep.checks.append(_check_min("Common markers (n)", n, TOL_MIN_MARKERS))
    rep.checks.append(_check_in_range(
        "−log10(p) correlation (encoding-mismatch)", r, TOL_DIPLO_ADD_MIN, TOL_DIPLO_ADD_MAX,
    ))
    rep.extras["note"] = "GWASpoly diplo {0,1,2,3,4}→{0,1,1,1,2}; TG min(d,k-d)→{0,1,2,1,0}"
    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def run_all(data_dir: Path, out_dir: Path, json_out: Path | None = None) -> int:
    # Load data once, share across comparisons
    cached = _load_data(data_dir)

    reports = [
        compare_additive(data_dir, out_dir, cached_data=cached),
        compare_1dom(data_dir, out_dir, cached_data=cached),
        compare_2dom(data_dir, out_dir, cached_data=cached),
        compare_3dom(data_dir, out_dir, cached_data=cached),
        compare_diplo_additive(data_dir, out_dir, cached_data=cached),
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
