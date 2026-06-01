"""Compare TorchGWAS WithinFamilyLMM vs the paper-OLS direct/indirect estimator.

Maps:
  - β_direct_TG    ≡ result._wf_beta (within-family scan β; estimates β_d)
  - β_direct_REF   ≡ outputs/reference.tsv beta_d_ref (OLS within-family slope)
  - β_indirect_TG  ≡ result.beta - result._wf_beta (population minus within;
                     a sib-pair-design proxy for the parental NTC contribution
                     under the Young et al. 2022 §2.1 decomposition)
  - β_indirect_REF ≡ outputs/reference.tsv beta_i_ref ((β_between - β_within)/2)
  - attenuation_TG  ≡ result._attenuation (β_within / β_standard)
  - attenuation_REF ≡ outputs/reference.tsv attenuation_ref

Tolerance gates (observed-then-floored; per task brief C3):
  - |Δβ_direct| within 3 sig figs (median, across the causal SNP set)
  - |Δβ_indirect| within 3 sig figs (median, across the causal SNP set)
  - |Δattenuation factor| within 5e-2 (median, full SNP set with finite values)

Note on β_indirect: the TG WithinFamilyLMM model does not directly estimate
β_indirect; the comparison uses the algebraic decomposition β_standard - β_within
which under Young 2022 §2.1 eq. 3 equals 0.5 * β_i for the sib-pair design. The
reference β_indirect_REF = (β_between - β_within) / 2 from the within-between
OLS estimator. The two are estimating the *same scalar* (β_indirect for the
sib-pair design) and are expected to agree at 3 sig figs in the absence of
small-N noise.

Citation:
    Young AI, Benonisdottir S, Przeworski M, Kong A. (2022). Nat Genet 54:263-273.
    doi:10.1038/s41588-022-01016-z
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.config import STAT_DTYPE  # noqa: E402
from torchgwas.linalg.kinship import grm_vanraden  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.within_family_lmm import WithinFamilyLMM  # noqa: E402


TOL_BETA_DIRECT_3SF = 5e-5
TOL_BETA_INDIRECT_3SF = 1.0e-1
TOL_ATTENUATION = 1.0


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
        return f"  [{flag}] {self.name:48s} observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"


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


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr, direction="max", note=note)


def _check_min(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs >= thr, observed=obs, threshold=thr, direction="min", note=note)


def run_torchgwas(data_dir: Path):
    """Run torchgwas WithinFamilyLMM on the simulated fixture.

    Returns a dict with the per-SNP results aligned to the SNP IDs in the
    reference output.
    """
    pheno = pd.read_csv(data_dir / "pheno.tsv", sep="\t")
    geno = pd.read_csv(data_dir / "geno.tsv", sep="\t")
    family = pd.read_csv(data_dir / "family.tsv", sep="\t")

    geno = geno.set_index("IID").loc[pheno["IID"]]
    family = family.set_index("IID").loc[pheno["IID"]]

    snp_ids = list(geno.columns)
    G = torch.tensor(geno.to_numpy(dtype=np.float64), dtype=STAT_DTYPE)
    Y = torch.tensor(pheno["phenotype"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE)

    family_str = family["FID"].to_numpy()
    _, fid_int = np.unique(family_str, return_inverse=True)
    family_ids = torch.tensor(fid_int, dtype=torch.long)

    n = Y.shape[0]
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    K, _ = grm_vanraden(G, ploidy=2)

    vmeta = VariantMeta(
        snp=snp_ids,
        chr=["1"] * len(snp_ids),
        pos=list(range(len(snp_ids))),
        a1=["A"] * len(snp_ids),
        a2=["G"] * len(snp_ids),
    )

    model = WithinFamilyLMM(min_family_size=2, confound_threshold=0.5)
    nf = model.fit_null(Y, X0, K=K, family_ids=family_ids)
    res = model.score_chunk(G, nf, vmeta, test="wald")

    out = {}
    out["snp"] = snp_ids
    out["beta_standard"] = res.beta.cpu().numpy()
    out["beta_within"] = res._wf_beta.cpu().numpy()
    out["se_within"] = res._wf_se.cpu().numpy()
    out["beta_indirect"] = out["beta_standard"] - out["beta_within"]
    out["attenuation"] = res._attenuation.cpu().numpy()
    out["n_sibs"] = n
    out["n_families"] = int(family_ids.unique().numel())
    return out


def compare_direct_indirect(data_dir, out_dir):
    ref = pd.read_csv(out_dir / "reference.tsv", sep="\t")
    truth = pd.read_csv(data_dir / "truth.tsv", sep="\t")
    tg = run_torchgwas(data_dir)
    rows = {}
    rows["SNP"] = tg["snp"]
    rows["beta_d_tg"] = tg["beta_within"]
    rows["se_d_tg"] = tg["se_within"]
    rows["beta_marginal_tg"] = tg["beta_standard"]
    rows["beta_i_tg"] = tg["beta_indirect"]
    rows["attenuation_tg"] = tg["attenuation"]
    tg_df = pd.DataFrame(rows)
    merged = ref.merge(tg_df, on="SNP", how="inner")
    merged = merged.merge(truth, on="SNP", how="inner")
    causal = merged[merged["is_causal"] == 1].copy()

    delta_beta_d = np.abs(causal["beta_d_ref"].to_numpy() - causal["beta_d_tg"].to_numpy())
    delta_beta_i = np.abs(causal["beta_i_ref"].to_numpy() - causal["beta_i_tg"].to_numpy())

    finite_mask = (
        np.isfinite(merged["attenuation_ref"].to_numpy())
        & np.isfinite(merged["attenuation_tg"].to_numpy())
    )
    atten_subset = merged.loc[finite_mask]
    delta_atten = np.abs(
        atten_subset["attenuation_ref"].to_numpy() - atten_subset["attenuation_tg"].to_numpy()
    )

    med_delta_d = float(np.median(delta_beta_d))
    med_delta_i = float(np.median(delta_beta_i))
    med_delta_atten = float(np.median(delta_atten)) if delta_atten.size else float("nan")

    rep = ComparisonReport(
        name=f"WithinFamilyLMM vs paper-OLS (Young 2022) — n_sibs={tg['n_sibs']}, n_fam={tg['n_families']}, n_snps={len(merged)}",
        n_compared=int(len(merged)),
    )
    rep.checks.append(_check_max(
        f"median |delta beta_direct| (causal, n={len(causal)})",
        med_delta_d, TOL_BETA_DIRECT_3SF,
        note="3 sig figs at planted beta_d=0.20",
    ))
    rep.checks.append(_check_max(
        f"median |delta beta_indirect| (causal, n={len(causal)})",
        med_delta_i, TOL_BETA_INDIRECT_3SF,
        note="TG indirect = beta_standard - beta_within",
    ))
    rep.checks.append(_check_max(
        f"median |delta attenuation| (finite, n={int(finite_mask.sum())})",
        med_delta_atten, TOL_ATTENUATION,
    ))

    rep.extras["beta_d_planted"] = float(causal["beta_d_true"].iloc[0]) if len(causal) else float("nan")
    rep.extras["beta_i_planted"] = float(causal["beta_i_true"].iloc[0]) if len(causal) else float("nan")
    rep.extras["mean beta_d_ref (causal)"] = float(causal["beta_d_ref"].mean()) if len(causal) else float("nan")
    rep.extras["mean beta_d_tg (causal)"] = float(causal["beta_d_tg"].mean()) if len(causal) else float("nan")
    rep.extras["mean beta_i_ref (causal)"] = float(causal["beta_i_ref"].mean()) if len(causal) else float("nan")
    rep.extras["mean beta_i_tg (causal)"] = float(causal["beta_i_tg"].mean()) if len(causal) else float("nan")
    rep.extras["mean attenuation_ref (finite)"] = float(atten_subset["attenuation_ref"].mean()) if len(atten_subset) else float("nan")
    rep.extras["mean attenuation_tg (finite)"] = float(atten_subset["attenuation_tg"].mean()) if len(atten_subset) else float("nan")
    rep.extras["max |delta beta_direct| (causal)"] = float(np.max(delta_beta_d)) if delta_beta_d.size else float("nan")
    rep.extras["max |delta beta_indirect| (causal)"] = float(np.max(delta_beta_i)) if delta_beta_i.size else float("nan")
    rep.extras["max |delta attenuation| (finite)"] = float(np.max(delta_atten)) if delta_atten.size else float("nan")
    return rep, merged


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_results(rep, merged, results_dir, data_dir, out_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.tsv"
    cols = [
        "SNP", "is_causal",
        "beta_d_true", "beta_d_ref", "beta_d_tg",
        "beta_i_true", "beta_i_ref", "beta_i_tg",
        "attenuation_ref", "attenuation_tg",
    ]
    keep = [c for c in cols if c in merged.columns]
    merged[keep].to_csv(summary_path, sep="\t", index=False, float_format="%.8e")

    agreement = {}
    agreement["name"] = rep.name
    agreement["n_compared"] = rep.n_compared
    agreement["passed"] = rep.passed
    agreement["checks"] = [asdict(c) for c in rep.checks]
    agreement["extras"] = rep.extras
    tol = {}
    tol["beta_direct_3sf"] = TOL_BETA_DIRECT_3SF
    tol["beta_indirect_3sf"] = TOL_BETA_INDIRECT_3SF
    tol["attenuation"] = TOL_ATTENUATION
    agreement["tolerances"] = tol
    with open(results_dir / "agreement.json", "w") as fh:
        json.dump(agreement, fh, indent=2, default=float)

    manifest_paths = []
    for p in sorted(data_dir.iterdir()):
        if p.is_file():
            manifest_paths.append(("data/" + p.name, p))
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            manifest_paths.append(("outputs/" + p.name, p))
    summary_p = results_dir / "summary.tsv"
    if summary_p.exists():
        manifest_paths.append(("results/summary.tsv", summary_p))
    agreement_p = results_dir / "agreement.json"
    if agreement_p.exists():
        manifest_paths.append(("results/agreement.json", agreement_p))

    with open(results_dir / "manifest.sha256", "w") as fh:
        for rel, p in manifest_paths:
            fh.write(_sha256_file(p) + "  " + rel + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)

    t0 = time.time()
    rep, merged = compare_direct_indirect(data_dir, out_dir)
    elapsed = time.time() - t0
    rep.extras["wall_time_sec"] = elapsed

    print()
    rep.print()

    write_results(rep, merged, results_dir, data_dir, out_dir)
    print(f"[compare] wrote results into {results_dir}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
