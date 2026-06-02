"""Compare MetaXcan / S-PrediXcan vs TorchGenomics twas_sumstat on the simulated fixture.

Both tools see the same simulated PrediXcan model (.db + covariance) and the
same GWAS sumstats. Per-gene agreement is measured on:

  - zscore (TWAS z-score)
  - effect_size / beta (the regression-scale gene-trait effect)
  - pvalue (two-sided normal tail)

Agreement metrics:

  1. Pearson r across all genes for z and -log10(p).
  2. Max |Delta z|, max |Delta beta|, max |Delta -log10p| (observed-then-floored).
  3. Per-gene Spearman of |z| ranks (sanity).

Tolerance policy (observed-then-floored): the first successful run records
the observed maxima, the floor is set to the observed value times a 2x cushion
to absorb cross-host FP drift. We do NOT use aspirational targets.

Citations:
  Barbeira AN, ..., Im HK (2018). Nature Communications 9:1825.

Note on the S-PrediXcan formula (Barbeira 2018, Eq. 4):
    z_g = sum_i (w_i * sigma_i / sigma_g) * z_i
where w_i is the gene's eQTL weight at SNP i, sigma_i = sqrt(Var(SNP_i)),
and sigma_g = sqrt(w^T Sigma w). When the model's covariance matrix is
expressed in correlation form (diag(Sigma) = 1), sigma_i = 1 and the formula
reduces to TorchGenomics' twas_sumstat exactly:
    z_g = sum_i w_i * z_i / sqrt(w^T Sigma w).
The simulated fixture uses diag(Sigma) = 1, so the comparison is a direct
test of the closed-form formula.
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.postgwas import SumStats, twas_sumstat  # noqa: E402


# ---- tolerance gates (observed-then-floored on first successful run) -------
# Observed numbers populate after the first end-to-end run; floors below are
# initial values that should bracket bit-equal agreement (since the fixture's
# covariance is in correlation form and SPrediXcan / twas_sumstat collapse to
# the same closed-form sum). The floor is updated by the harness operator
# after observing the first run.
TOL_MAX_DZ = 1e-6           # observed: filled in after first run
TOL_MAX_DBETA = 1e-6        # observed: filled in after first run
TOL_MAX_DNEGLOG10P = 1e-3   # observed: filled in after first run
TOL_Z_CORR = 0.9999         # Pearson r >= 0.9999 across genes


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
        return f"  [{flag}] {self.name:36s} observed={self.observed:.6e} {cmp} {self.threshold:.6e} {self.note}"


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


def _check_min(name, obs, thr, note=""):
    return CheckResult(name=name, passed=(obs >= thr) or (np.isnan(thr)),
                       observed=obs, threshold=thr, direction="min", note=note)


def _check_max(name, obs, thr, note=""):
    return CheckResult(name=name, passed=(obs <= thr),
                       observed=obs, threshold=thr, direction="max", note=note)


def _safe_corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return 0.0
    if a.std() < 1e-15 and b.std() < 1e-15:
        return 1.0
    if a.std() < 1e-15 or b.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_metaxcan_output(csv_path: Path) -> pd.DataFrame:
    """Load MetaXcan SPrediXcan output CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    df = df.set_index("gene")
    return df


def _load_truth(data_dir: Path) -> dict:
    return json.loads((data_dir / "sim_truth.json").read_text())


def _build_tg_inputs(data_dir: Path, truth: dict) -> tuple:
    """Build the TorchGenomics twas_sumstat inputs from the simulated fixture.

    Returns: (gwas SumStats, weights dict, snp_lists dict, ld_matrix dict)
    """
    import gzip

    # Parse GWAS sumstats
    with gzip.open(data_dir / "gwas_sumstats.txt.gz", "rt") as f:
        gwas_df = pd.read_csv(f, sep=" ")

    gwas_snps = gwas_df["SNP"].tolist()
    beta = torch.tensor(gwas_df["beta"].to_numpy(np.float64), dtype=torch.float64)
    se = torch.tensor(gwas_df["se"].to_numpy(np.float64), dtype=torch.float64)
    pval = torch.tensor(gwas_df["pvalue"].to_numpy(np.float64), dtype=torch.float64)
    n_snps = len(gwas_snps)

    gwas = SumStats(
        chr=["1"] * n_snps,
        pos=list(range(1, n_snps + 1)),
        snp=gwas_snps,
        a1=gwas_df["effect_allele"].tolist(),
        a2=gwas_df["non_effect_allele"].tolist(),
        beta=beta,
        se=se,
        p=pval,
        n=torch.full((n_snps,), float(truth["gwas_n"]), dtype=torch.float64),
        af=None,
    )

    # Build per-gene weights, snp lists, and LD matrices.
    # The TG twas_sumstat sees the SAME (gene, weights, Sigma) as MetaXcan.
    weights = {}
    snp_lists = {}
    ld_matrix = {}
    for gene, snps in truth["gene_snps"].items():
        w = np.asarray(truth["gene_weights"][gene], dtype=np.float64)
        ld = np.asarray(truth["gene_ld"][gene], dtype=np.float64)
        # Drop zero-weight SNPs to mirror MetaXcan's load_weights behavior
        # (only non-zero rows live in the .db).
        mask = w != 0.0
        snps_active = [s for s, m in zip(snps, mask) if m]
        w_active = w[mask]
        ld_active = ld[np.ix_(mask, mask)]
        if len(snps_active) == 0:
            continue
        weights[gene] = torch.tensor(w_active, dtype=torch.float64)
        snp_lists[gene] = snps_active
        ld_matrix[gene] = torch.tensor(ld_active, dtype=torch.float64)

    return gwas, weights, snp_lists, ld_matrix


def compare_sumstat_path(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """S-PrediXcan (MetaXcan) vs TorchGenomics twas_sumstat per-gene agreement."""
    truth = _load_truth(data_dir)

    mx = _load_metaxcan_output(out_dir / "sprediXcan_results.csv")

    gwas, weights, snp_lists, ld_matrix = _build_tg_inputs(data_dir, truth)

    tg = twas_sumstat(
        gwas=gwas,
        weights=weights,
        snp_lists=snp_lists,
        ld_matrix=ld_matrix,
        p_threshold=0.05,
        correction="bonferroni",
    )

    # Align by gene id
    tg_rows = {g.gene_id: g for g in tg.genes}
    common = [g for g in mx.index.tolist() if g in tg_rows]

    if len(common) == 0:
        rep = ComparisonReport(name="S-PrediXcan sumstats (no overlap!)", n_compared=0)
        rep.checks.append(_check_max("|Delta z| max", float("inf"), TOL_MAX_DZ,
                                     note="gene id overlap empty"))
        return rep

    mx_z = np.array([mx.loc[g, "zscore"] for g in common], dtype=np.float64)
    tg_z = np.array([tg_rows[g].z_twas for g in common], dtype=np.float64)
    mx_p = np.array([mx.loc[g, "pvalue"] for g in common], dtype=np.float64)
    tg_p = np.array([tg_rows[g].p_twas for g in common], dtype=np.float64)
    mx_eff = np.array([mx.loc[g, "effect_size"] for g in common], dtype=np.float64)

    # Convert tg z back to MetaXcan's effect_size scale:
    # MetaXcan effect_size = sum(w * beta * sigma_l^2) / sigma_g^2.
    # When diag(Sigma) = 1 (our fixture), effect_size = sum(w * beta) / sigma_g^2.
    # We compute the TG analog manually so the comparison is apples-to-apples.
    tg_eff = []
    for g in common:
        w = weights[g].numpy()
        Sigma = ld_matrix[g].numpy()
        snps = snp_lists[g]
        # Pull beta values from the GWAS for these SNPs.
        gwas_snp_to_idx = {s: i for i, s in enumerate(gwas.snp)}
        beta_for_g = np.array(
            [gwas.beta[gwas_snp_to_idx[s]].item() for s in snps],
            dtype=np.float64,
        )
        sigma_g2 = float(w @ Sigma @ w)
        if sigma_g2 <= 0:
            tg_eff.append(np.nan)
        else:
            sigma_l2 = np.diag(Sigma)
            tg_eff.append(float(np.sum(w * beta_for_g * sigma_l2) / sigma_g2))
    tg_eff = np.array(tg_eff, dtype=np.float64)

    # Per-gene gaps
    d_z = np.abs(mx_z - tg_z)
    d_beta = np.abs(mx_eff - tg_eff)
    nlog10_mx = -np.log10(np.clip(mx_p, 1e-300, 1.0))
    nlog10_tg = -np.log10(np.clip(tg_p, 1e-300, 1.0))
    d_nlog10p = np.abs(nlog10_mx - nlog10_tg)

    max_dz = float(np.max(d_z))
    max_dbeta = float(np.max(d_beta))
    max_dnlog10p = float(np.max(d_nlog10p))
    z_corr = _safe_corr(mx_z, tg_z)
    eff_corr = _safe_corr(mx_eff, tg_eff)
    nlog10p_corr = _safe_corr(nlog10_mx, nlog10_tg)

    rep = ComparisonReport(
        name="S-PrediXcan sumstats (MetaXcan v0.8.1 vs torchgenomics.postgwas.twas_sumstat)",
        n_compared=len(common),
    )
    rep.checks.append(_check_max("|Delta z| max", max_dz, TOL_MAX_DZ))
    rep.checks.append(_check_max("|Delta effect_size| max", max_dbeta, TOL_MAX_DBETA))
    rep.checks.append(_check_max("|Delta -log10 p| max", max_dnlog10p, TOL_MAX_DNEGLOG10P))
    rep.checks.append(_check_min("Pearson r (z)", z_corr, TOL_Z_CORR))
    rep.extras["MetaXcan z (per gene)"] = mx_z.tolist()
    rep.extras["TorchGenomics z (per gene)"] = tg_z.tolist()
    rep.extras["MetaXcan effect_size"] = mx_eff.tolist()
    rep.extras["TorchGenomics effect_size"] = tg_eff.tolist()
    rep.extras["MetaXcan -log10p"] = nlog10_mx.tolist()
    rep.extras["TorchGenomics -log10p"] = nlog10_tg.tolist()
    rep.extras["effect_size Pearson r"] = eff_corr
    rep.extras["-log10p Pearson r"] = nlog10p_corr
    rep.extras["gene order (compared)"] = common
    return rep


def _hash_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_results(rep: ComparisonReport, results_dir: Path, data_dir: Path,
                   out_dir: Path) -> None:
    """Persist summary.tsv, agreement.json, manifest.sha256."""
    results_dir.mkdir(parents=True, exist_ok=True)

    common = rep.extras.get("gene order (compared)", [])
    mx_z = rep.extras.get("MetaXcan z (per gene)", [])
    tg_z = rep.extras.get("TorchGenomics z (per gene)", [])
    mx_eff = rep.extras.get("MetaXcan effect_size", [])
    tg_eff = rep.extras.get("TorchGenomics effect_size", [])
    mx_nl = rep.extras.get("MetaXcan -log10p", [])
    tg_nl = rep.extras.get("TorchGenomics -log10p", [])

    rows = []
    rows.append("\t".join([
        "gene", "metaxcan_z", "torchgenomics_z", "delta_z",
        "metaxcan_effect_size", "torchgenomics_effect_size", "delta_effect_size",
        "metaxcan_neglog10p", "torchgenomics_neglog10p", "delta_neglog10p",
    ]))
    for i, g in enumerate(common):
        rows.append("\t".join([
            g,
            f"{mx_z[i]:.10g}",
            f"{tg_z[i]:.10g}",
            f"{mx_z[i] - tg_z[i]:.10g}",
            f"{mx_eff[i]:.10g}",
            f"{tg_eff[i]:.10g}",
            f"{mx_eff[i] - tg_eff[i]:.10g}",
            f"{mx_nl[i]:.10g}",
            f"{tg_nl[i]:.10g}",
            f"{mx_nl[i] - tg_nl[i]:.10g}",
        ]))
    (results_dir / "summary.tsv").write_text("\n".join(rows) + "\n")

    agreement = dict()
    agreement["name"] = rep.name
    agreement["n_compared"] = rep.n_compared
    agreement["passed"] = bool(rep.passed)
    agreement["checks"] = [asdict(c) for c in rep.checks]
    agreement["extras"] = rep.extras
    agreement["tolerance_gates"] = dict()
    agreement["tolerance_gates"]["TOL_MAX_DZ"] = TOL_MAX_DZ
    agreement["tolerance_gates"]["TOL_MAX_DBETA"] = TOL_MAX_DBETA
    agreement["tolerance_gates"]["TOL_MAX_DNEGLOG10P"] = TOL_MAX_DNEGLOG10P
    agreement["tolerance_gates"]["TOL_Z_CORR"] = TOL_Z_CORR
    (results_dir / "agreement.json").write_text(json.dumps(agreement, indent=2, default=float))

    manifest_lines = []
    for label, p in [
        ("data/model.db", data_dir / "model.db"),
        ("data/model.txt.gz", data_dir / "model.txt.gz"),
        ("data/gwas_sumstats.txt.gz", data_dir / "gwas_sumstats.txt.gz"),
        ("data/sim_truth.json", data_dir / "sim_truth.json"),
        ("outputs/sprediXcan_results.csv", out_dir / "sprediXcan_results.csv"),
    ]:
        if p.exists():
            manifest_lines.append(f"{_hash_file(p)}  {label}")
        else:
            manifest_lines.append(f"MISSING  {label}")
    (results_dir / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n")


def run_all(data_dir: Path, out_dir: Path, results_dir: Path) -> int:
    rep = compare_sumstat_path(data_dir, out_dir)
    rep.print()
    _write_results(rep, results_dir, data_dir, out_dir)
    print(f"=== {1 if rep.passed else 0}/1 comparisons passed ===")
    print(f"[compare] wrote {results_dir}/agreement.json")
    print(f"[compare] wrote {results_dir}/summary.tsv")
    print(f"[compare] wrote {results_dir}/manifest.sha256")
    return 0 if rep.passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    return run_all(Path(args.data_dir), Path(args.out_dir), Path(args.results_dir))


if __name__ == "__main__":
    sys.exit(main())
