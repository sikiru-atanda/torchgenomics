#!/usr/bin/env python3
"""Pillar D — diff fresh upstream tool output vs committed reference fixture.

This is fundamentally different from the Pillar B compare scripts:
  - Pillar B (validation/external/<tool>/compare.py) diffs <tool> ↔ TorchGWAS.
  - Pillar D (this script)                            diffs <tool> ↔ committed fixture.

The "committed fixture" lives at:
  - GEMMA:    ${ROOT}/gemma_demo/output/                 (5 reference outputs)
  - GAPIT:    ${ROOT}/benchmark/gapit_results/           (4 GWAS CSVs)
  - GWASpoly: ${ROOT}/benchmark/gwaspoly_results/        (8 model CSVs)

The "fresh" output is whatever the matching Pillar B harness just produced
under validation/external/<tool>/outputs/ (run via rerun_goldens.py).

Both should be deterministic with the same input + same upstream version,
so the tolerance bar is *tight*:

  - β:           |Δ| < 1e-6 absolute
  - SE:          |Δ| < 1e-6 absolute
  - p-value:     |Δ| < 1e-8 absolute (or |Δ -log10p| < 1e-4)
  - log-likelihood: |Δ| < 1e-4 absolute
  - Vg / Ve:     |Δ| relative < 1e-6

Drift beyond these thresholds is a `fixture-drift` finding (the upstream
tool's behavior or the input data has changed since the fixture was
captured) — distinct from a `torchgwas-divergence` finding logged in
Pillars A/B/C.

CLI:
    python3 compare_fixtures.py --tool {gemma,gapit,gwaspoly}
    python3 compare_fixtures.py --tool all  (default)
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

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                  # /…/.claude/worktrees/<wtree>
PROJECT_ROOT = REPO.parent              # /…/Documents/GWAS_Expert (canonical)

# Search order for committed fixtures: worktree first, then project root.
_FIXTURE_SEARCH = [
    REPO / "gemma_demo" / "output",
    PROJECT_ROOT / "gemma_demo" / "output",
    Path.home() / "Documents" / "GWAS_Expert" / "gemma_demo" / "output",
]
_GAPIT_SEARCH = [
    REPO / "benchmark" / "gapit_results",
    PROJECT_ROOT / "benchmark" / "gapit_results",
    Path.home() / "Documents" / "GWAS_Expert" / "benchmark" / "gapit_results",
]
_GWASPOLY_SEARCH = [
    REPO / "benchmark" / "gwaspoly_results",
    PROJECT_ROOT / "benchmark" / "gwaspoly_results",
    Path.home() / "Documents" / "GWAS_Expert" / "benchmark" / "gwaspoly_results",
]


def _first_existing(paths: list[Path], probe: str) -> Path | None:
    for p in paths:
        if (p / probe).exists():
            return p
    return None


# ── Tolerance gates (Pillar D — same input + same version → near-bit-equal) ──

# Numeric outputs of deterministic tool kernels should match to within
# float64 round-off propagated through their I/O. We set tight defaults
# and surface observed values regardless.
TOL_BETA_ABS = 1e-6
TOL_SE_ABS = 1e-6
TOL_PVAL_ABS = 1e-8           # raw p in [0, 1]; tightest gate
TOL_NEGLOG10P_ABS = 1e-4      # relaxed for very small p (log-domain)
TOL_LOGL_ABS = 1e-4
TOL_VARCOMP_REL = 1e-6        # relative; vg, ve, Vg, Ve
TOL_GRM_ABS = 1e-10           # GEMMA writes 7-digit floats; this is generous
TOL_AF_ABS = 1e-6             # allele freq

# Drift threshold: any observed |Δ| greater than the gate is logged as
# fixture-drift. The script exits 0 either way (drift is informational,
# not a hard failure — the user decides whether to regenerate the fixture).


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    n_compared: int
    metric: str            # "max |Δ|", "max relative |Δ|", "max -log10p |Δ|"
    observed: float
    threshold: float
    passed: bool
    note: str = ""

    def __str__(self) -> str:
        flag = "OK" if self.passed else "DRIFT"
        return (
            f"  [{flag}] {self.name:55s}  {self.metric:20s}  "
            f"observed={self.observed:.3e}  thr={self.threshold:.1e}  "
            f"n={self.n_compared}  {self.note}"
        )


@dataclass
class ToolReport:
    tool: str
    fresh_dir: Path
    fixture_dir: Path
    checks: list[CheckResult] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_drifted(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def passed(self) -> bool:
        return self.n_drifted == 0

    def print(self) -> None:
        bar = "=" * 90
        print(bar)
        print(f"  {self.tool.upper()}  fixture-drift audit")
        print(f"    fresh   : {self.fresh_dir}")
        print(f"    fixture : {self.fixture_dir}")
        print(bar)
        for c in self.checks:
            print(c)
        if self.extras:
            print("  -- extras --")
            for k, v in self.extras.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3e}")
                else:
                    print(f"    {k}: {v}")
        if self.n_drifted == 0:
            print(f"  → fixture is AUTHORITATIVE ({len(self.checks)} checks within tolerance)")
        else:
            print(
                f"  → fixture-drift on {self.n_drifted}/{len(self.checks)} checks "
                f"(consider regenerating {self.tool} reference outputs)"
            )
        print()


# ── Common helpers ───────────────────────────────────────────────────────────


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| with NaN handling (NaN-vs-NaN counts as 0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask])))


def _max_rel(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask]) / (np.abs(b[mask]) + eps)))


def _max_neglog10_diff(p_a: np.ndarray, p_b: np.ndarray) -> float:
    p_a = np.clip(np.asarray(p_a, dtype=np.float64), 1e-300, 1.0)
    p_b = np.clip(np.asarray(p_b, dtype=np.float64), 1e-300, 1.0)
    mask = np.isfinite(p_a) & np.isfinite(p_b)
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(-np.log10(p_a[mask]) - (-np.log10(p_b[mask])))))


def _check(
    name: str, metric: str, observed: float, threshold: float, n: int, note: str = ""
) -> CheckResult:
    if not np.isfinite(observed):
        passed = False
        note = (note + " (non-finite observed)").strip()
    else:
        passed = observed <= threshold
    return CheckResult(
        name=name, n_compared=n, metric=metric,
        observed=observed, threshold=threshold, passed=passed, note=note,
    )


# ── GEMMA ────────────────────────────────────────────────────────────────────


def _parse_gemma_log_scalars(path: Path) -> dict[str, float]:
    """Pull vg / ve / pve / log-likelihoods from a GEMMA .log.txt."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        low = line.lower()
        if "vg estimate in the null model" in low:
            out["vg"] = float(line.split("=")[-1].strip())
        elif "ve estimate in the null model" in low:
            out["ve"] = float(line.split("=")[-1].strip())
        elif "pve estimate in the null model" in low:
            out["pve"] = float(line.split("=")[-1].strip())
        elif "remle log-likelihood in the null model" in low:
            out["ll_reml"] = float(line.split("=")[-1].strip())
        elif "mle log-likelihood in the null model" in low:
            out["ll_ml"] = float(line.split("=")[-1].strip())
    return out


def _parse_gemma_log_matrix(path: Path) -> dict[str, np.ndarray]:
    """Pull Vg / Ve symmetric matrices from a GEMMA mvLMM .log.txt.

    GEMMA writes the lower-triangle of a 2×2 (or larger) symmetric matrix
    over two lines:
        ## REMLE estimate for Vg in the null model:
        a11
        a21    a22
    or — for a single-trait run — just one scalar on one line.
    """
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        return out
    lines = path.read_text().splitlines()

    def _try_2x2(i: int) -> np.ndarray | None:
        if i + 2 >= len(lines):
            return None
        try:
            r1 = [float(x) for x in lines[i + 1].split()]
            r2 = [float(x) for x in lines[i + 2].split()]
        except ValueError:
            return None
        # Pattern: 1 number on row 1, 2 numbers on row 2 → 2×2 lower-tri.
        if len(r1) == 1 and len(r2) == 2:
            return np.array([[r1[0], r2[0]], [r2[0], r2[1]]])
        # Fallback: 2 + 2 → upper/full.
        if len(r1) >= 2 and len(r2) >= 2:
            return np.array([[r1[0], r1[1]], [r2[0], r2[1]]])
        # 1×1 (single trait)
        if len(r1) == 1:
            return np.array([[r1[0]]])
        return None

    for i, line in enumerate(lines):
        low = line.lower()
        if "remle estimate for vg in the null model" in low:
            mat = _try_2x2(i)
            if mat is not None:
                out["Vg"] = mat
        elif "remle estimate for ve in the null model" in low:
            mat = _try_2x2(i)
            if mat is not None:
                out["Ve"] = mat
    return out


def compare_gemma(fresh_dir: Path, fixture_dir: Path) -> ToolReport:
    """Diff fresh GEMMA outputs vs committed gemma_demo/output/ fixture."""
    rep = ToolReport(tool="gemma", fresh_dir=fresh_dir, fixture_dir=fixture_dir)

    # 1. Centered kinship cXX matrix --------------------------------------
    fresh_K = fresh_dir / "mdp_kinship.cXX.txt"
    fix_K = fixture_dir / "mdp_kinship.cXX.txt"
    if fresh_K.exists() and fix_K.exists():
        K_fresh = pd.read_csv(fresh_K, sep="\t", header=None).values
        K_fix = pd.read_csv(fix_K, sep="\t", header=None).values
        if K_fresh.shape != K_fix.shape:
            rep.checks.append(_check(
                "GRM cXX shape", "shape mismatch", float("nan"), 0.0,
                K_fresh.size, note=f"fresh {K_fresh.shape} vs fixture {K_fix.shape}",
            ))
        else:
            rep.checks.append(_check(
                "GRM cXX max |Δ|", "max |Δ|",
                _max_abs(K_fresh, K_fix), TOL_GRM_ABS, K_fresh.size,
            ))
    else:
        rep.extras["grm_skipped"] = f"fresh={fresh_K.exists()} fix={fix_K.exists()}"

    # 2. Single-trait LMM scalars (vg, ve, pve, log-likelihood) -----------
    fresh_log = fresh_dir / "mdp_lmm_all.log.txt"
    fix_log = fixture_dir / "mdp_lmm_all.log.txt"
    fresh_scalars = _parse_gemma_log_scalars(fresh_log)
    fix_scalars = _parse_gemma_log_scalars(fix_log)
    common_keys = set(fresh_scalars) & set(fix_scalars)
    for k in sorted(common_keys):
        a, b = fresh_scalars[k], fix_scalars[k]
        if k in {"ll_reml", "ll_ml"}:
            rep.checks.append(_check(
                f"LMM null {k}", "max |Δ|",
                abs(a - b), TOL_LOGL_ABS, 1,
                note=f"fresh={a:.6g} fixture={b:.6g}",
            ))
        else:
            rel = abs(a - b) / max(abs(b), 1e-30)
            rep.checks.append(_check(
                f"LMM null {k}", "rel |Δ|",
                rel, TOL_VARCOMP_REL, 1,
                note=f"fresh={a:.6g} fixture={b:.6g}",
            ))

    # 3. Single-trait LMM .assoc.txt (β, SE, p_wald, p_lrt, p_score, af) --
    fresh_assoc = fresh_dir / "mdp_lmm_all.assoc.txt"
    fix_assoc = fixture_dir / "mdp_lmm_all.assoc.txt"
    if fresh_assoc.exists() and fix_assoc.exists():
        df_fresh = pd.read_csv(fresh_assoc, sep="\t")
        df_fix = pd.read_csv(fix_assoc, sep="\t")
        merged = pd.merge(
            df_fresh, df_fix, on="rs", how="inner", suffixes=("_fresh", "_fix"),
        )
        n = len(merged)
        if n > 0:
            for col, tol in [
                ("af", TOL_AF_ABS),
                ("beta", TOL_BETA_ABS),
                ("se", TOL_SE_ABS),
                ("logl_H1", TOL_LOGL_ABS),
            ]:
                a = merged[f"{col}_fresh"].values
                b = merged[f"{col}_fix"].values
                rep.checks.append(_check(
                    f"LMM single {col}", "max |Δ|",
                    _max_abs(a, b), tol, n,
                ))
            for col in ("p_wald", "p_lrt", "p_score"):
                a = merged[f"{col}_fresh"].values
                b = merged[f"{col}_fix"].values
                rep.checks.append(_check(
                    f"LMM single {col}", "max |Δ|",
                    _max_abs(a, b), TOL_PVAL_ABS, n,
                ))
                rep.checks.append(_check(
                    f"LMM single -log10 {col}", "max |Δ|",
                    _max_neglog10_diff(a, b), TOL_NEGLOG10P_ABS, n,
                ))
            rep.extras["lmm_n_common_rs"] = int(n)
            rep.extras["lmm_n_fresh_only"] = int(len(df_fresh) - n)
            rep.extras["lmm_n_fixture_only"] = int(len(df_fix) - n)
        else:
            rep.checks.append(_check(
                "LMM single assoc rs overlap", "rs intersection",
                0.0, 1.0, 0, note="no common SNP IDs",
            ))
    else:
        rep.extras["lmm_assoc_skipped"] = (
            f"fresh={fresh_assoc.exists()} fix={fix_assoc.exists()}"
        )

    # 4. Multi-trait mvLMM Vg / Ve matrices -------------------------------
    fresh_mv_log = fresh_dir / "mdp_mvlmm_wald.log.txt"
    fix_mv_log = fixture_dir / "mdp_mvlmm_wald.log.txt"
    Vg_fresh = _parse_gemma_log_matrix(fresh_mv_log)
    Vg_fix = _parse_gemma_log_matrix(fix_mv_log)
    for k in ("Vg", "Ve"):
        if k in Vg_fresh and k in Vg_fix:
            A = Vg_fresh[k]
            B = Vg_fix[k]
            if A.shape == B.shape:
                rep.checks.append(_check(
                    f"mvLMM {k} max rel |Δ|", "rel |Δ|",
                    _max_rel(A, B), TOL_VARCOMP_REL, A.size,
                ))

    # 5. Multi-trait mvLMM assoc (p_wald per SNP) -------------------------
    fresh_mv = fresh_dir / "mdp_mvlmm_wald.assoc.txt"
    fix_mv = fixture_dir / "mdp_mvlmm_wald.assoc.txt"
    if fresh_mv.exists() and fix_mv.exists():
        df_fresh = pd.read_csv(fresh_mv, sep="\t")
        df_fix = pd.read_csv(fix_mv, sep="\t")
        merged = pd.merge(df_fresh, df_fix, on="rs", how="inner",
                          suffixes=("_fresh", "_fix"))
        n = len(merged)
        if n > 0 and "p_wald_fresh" in merged.columns:
            a = merged["p_wald_fresh"].values
            b = merged["p_wald_fix"].values
            rep.checks.append(_check(
                "mvLMM p_wald", "max |Δ|",
                _max_abs(a, b), TOL_PVAL_ABS, n,
            ))
            rep.checks.append(_check(
                "mvLMM -log10 p_wald", "max |Δ|",
                _max_neglog10_diff(a, b), TOL_NEGLOG10P_ABS, n,
            ))
            rep.extras["mvlmm_n_common_rs"] = int(n)

    return rep


# ── GAPIT ────────────────────────────────────────────────────────────────────


def compare_gapit(fresh_dir: Path, fixture_dir: Path) -> ToolReport:
    """Diff fresh GAPIT outputs vs committed benchmark/gapit_results/ fixture.

    Per-method GAPIT writes a CSV with columns: SNP, Chromosome, Position,
    P.value, maf, nobs, [Rsquare.of.Model.with(out)_SNP, effect].
    GLM/MLM are deterministic; FarmCPU/BLINK are stochastic (pseudo-QTN
    selection trajectory) — for those we additionally compare top-K
    overlap as a sanity check that the *signal* is preserved.
    """
    rep = ToolReport(tool="gapit", fresh_dir=fresh_dir, fixture_dir=fixture_dir)

    # FarmCPU / BLINK use pseudo-QTN bootstrapping that is RNG-seeded by
    # GAPIT internally. Different GAPIT versions may seed differently
    # → loose tolerance (top-K overlap), surface raw |Δ| as informational.
    GLM_LIKE = {"GLM": True, "MLM": True}
    MULTI_LOCUS = {"FarmCPU": True, "BLINK": True}

    for model in ("GLM", "MLM", "FarmCPU", "BLINK"):
        fresh_csv = fresh_dir / f"{model}_GWAS.csv"
        fix_csv = fixture_dir / f"{model}_GWAS.csv"
        if not (fresh_csv.exists() and fix_csv.exists()):
            rep.extras[f"{model}_skipped"] = (
                f"fresh={fresh_csv.exists()} fix={fix_csv.exists()}"
            )
            continue
        df_fresh = pd.read_csv(fresh_csv)
        df_fix = pd.read_csv(fix_csv)
        # GAPIT writes "Position " (trailing space) for some methods; normalize.
        df_fresh.columns = [c.strip() for c in df_fresh.columns]
        df_fix.columns = [c.strip() for c in df_fix.columns]

        merged = pd.merge(
            df_fresh, df_fix, on="SNP", how="inner", suffixes=("_fresh", "_fix"),
        )
        n = len(merged)
        if n == 0:
            rep.checks.append(_check(
                f"{model} SNP overlap", "rs intersection",
                0.0, 1.0, 0, note="no common SNP IDs",
            ))
            continue

        if model in GLM_LIKE:
            for col, tol in [
                ("P.value", TOL_PVAL_ABS),
                ("maf", TOL_AF_ABS),
                ("effect", TOL_BETA_ABS),
            ]:
                if f"{col}_fresh" not in merged.columns:
                    continue
                a = merged[f"{col}_fresh"].values
                b = merged[f"{col}_fix"].values
                rep.checks.append(_check(
                    f"{model} {col}", "max |Δ|",
                    _max_abs(a, b), tol, n,
                ))
            if "P.value_fresh" in merged.columns:
                rep.checks.append(_check(
                    f"{model} -log10 P", "max |Δ|",
                    _max_neglog10_diff(
                        merged["P.value_fresh"].values,
                        merged["P.value_fix"].values,
                    ),
                    TOL_NEGLOG10P_ABS, n,
                ))
        elif model in MULTI_LOCUS:
            # Stochastic: assert top-10 overlap (informational on raw |Δ|).
            df_fresh_sorted = df_fresh.sort_values("P.value").head(10)
            df_fix_sorted = df_fix.sort_values("P.value").head(10)
            top10_fresh = set(df_fresh_sorted["SNP"].tolist())
            top10_fix = set(df_fix_sorted["SNP"].tolist())
            overlap = len(top10_fresh & top10_fix) / 10.0
            rep.checks.append(_check(
                f"{model} top-10 SNP overlap", "fraction",
                1.0 - overlap, 0.3, n,
                note=f"overlap={overlap:.2f} (drift threshold = ≥ 0.7)",
            ))
            # Surface raw |Δ| on the union (informational; does NOT gate)
            if "P.value_fresh" in merged.columns:
                raw = _max_neglog10_diff(
                    merged["P.value_fresh"].values,
                    merged["P.value_fix"].values,
                )
                rep.extras[f"{model}_max_neglog10p_diff"] = raw

    return rep


# ── GWASpoly ─────────────────────────────────────────────────────────────────


def compare_gwaspoly(fresh_dir: Path, fixture_dir: Path) -> ToolReport:
    """Diff fresh GWASpoly outputs vs committed benchmark/gwaspoly_results/.

    GWASpoly runs P3D on a single REML fit, then per-marker score tests
    across 5 gene-action models. With the same upstream version + same
    input the math is fully deterministic. Each model writes:
        marker, chrom, pos, score, pvalue
    """
    rep = ToolReport(tool="gwaspoly", fresh_dir=fresh_dir, fixture_dir=fixture_dir)

    models = [
        "additive",
        "1_dom_alt", "1_dom_ref",
        "2_dom_alt", "2_dom_ref",
        "diplo_additive", "diplo_general",
        "general",
    ]

    for model in models:
        fname = f"gwaspoly_{model}.csv"
        fresh_csv = fresh_dir / fname
        fix_csv = fixture_dir / fname
        if not (fresh_csv.exists() and fix_csv.exists()):
            rep.extras[f"{model}_skipped"] = (
                f"fresh={fresh_csv.exists()} fix={fix_csv.exists()}"
            )
            continue
        df_fresh = pd.read_csv(fresh_csv)
        df_fix = pd.read_csv(fix_csv)
        merged = pd.merge(
            df_fresh, df_fix, on="marker", how="inner", suffixes=("_fresh", "_fix"),
        )
        n = len(merged)
        if n == 0:
            rep.checks.append(_check(
                f"{model} marker overlap", "intersection",
                0.0, 1.0, 0, note="no common marker IDs",
            ))
            continue

        # `score` is a log-domain transformed pvalue; `pvalue` is raw.
        for col, tol in [
            ("pvalue", TOL_PVAL_ABS),
            ("score", TOL_NEGLOG10P_ABS),
        ]:
            if f"{col}_fresh" not in merged.columns:
                continue
            a = merged[f"{col}_fresh"].values
            b = merged[f"{col}_fix"].values
            rep.checks.append(_check(
                f"{model} {col}", "max |Δ|",
                _max_abs(a, b), tol, n,
            ))
        if "pvalue_fresh" in merged.columns:
            rep.checks.append(_check(
                f"{model} -log10 pvalue", "max |Δ|",
                _max_neglog10_diff(
                    merged["pvalue_fresh"].values,
                    merged["pvalue_fix"].values,
                ),
                TOL_NEGLOG10P_ABS, n,
            ))
        rep.extras[f"{model}_n_common"] = int(n)

    # GWASpoly kinship CSV (if both present)
    fresh_K = fresh_dir / "gwaspoly_kinship.csv"
    fix_K = fixture_dir / "gwaspoly_kinship.csv"
    if fresh_K.exists() and fix_K.exists():
        df_fresh = pd.read_csv(fresh_K, index_col=0)
        df_fix = pd.read_csv(fix_K, index_col=0)
        # Reindex on the intersection of indices
        common_idx = df_fresh.index.intersection(df_fix.index)
        common_cols = df_fresh.columns.intersection(df_fix.columns)
        if len(common_idx) > 0 and len(common_cols) > 0:
            A = df_fresh.loc[common_idx, common_cols].values
            B = df_fix.loc[common_idx, common_cols].values
            rep.checks.append(_check(
                "GRM kinship max |Δ|", "max |Δ|",
                _max_abs(A, B), 1e-6, A.size,
                note=f"{len(common_idx)}×{len(common_cols)} common entries",
            ))

    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def _resolve_dirs(tool: str) -> tuple[Path, Path]:
    """Return (fresh_dir, fixture_dir) for one tool."""
    fresh_dir = REPO / "validation" / "external" / tool / "outputs"
    if tool == "gemma":
        fix_root = _first_existing(_FIXTURE_SEARCH, "mdp_lmm_all.assoc.txt")
    elif tool == "gapit":
        fix_root = _first_existing(_GAPIT_SEARCH, "GLM_GWAS.csv")
    elif tool == "gwaspoly":
        fix_root = _first_existing(_GWASPOLY_SEARCH, "gwaspoly_additive.csv")
    else:
        raise ValueError(f"unknown tool: {tool}")
    if fix_root is None:
        raise FileNotFoundError(
            f"committed fixture for {tool} not found in any of:\n  "
            + "\n  ".join(str(p) for p in
                          (_FIXTURE_SEARCH if tool == "gemma"
                           else _GAPIT_SEARCH if tool == "gapit"
                           else _GWASPOLY_SEARCH))
        )
    return fresh_dir, fix_root


_DISPATCH = {
    "gemma": compare_gemma,
    "gapit": compare_gapit,
    "gwaspoly": compare_gwaspoly,
}


def run_one(tool: str, json_out: Path | None = None) -> ToolReport:
    fresh, fix = _resolve_dirs(tool)
    if not fresh.exists():
        raise FileNotFoundError(
            f"fresh outputs not found at {fresh}; "
            f"run validation/external/{tool}/run_{tool}.sh first "
            f"(or use rerun_goldens.py)."
        )
    rep = _DISPATCH[tool](fresh, fix)
    rep.print()
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "tool": rep.tool,
            "fresh_dir": str(rep.fresh_dir),
            "fixture_dir": str(rep.fixture_dir),
            "n_drifted": rep.n_drifted,
            "n_checks": len(rep.checks),
            "checks": [asdict(c) for c in rep.checks],
            "extras": rep.extras,
        }, indent=2, default=float))
        print(f"  → wrote JSON report: {json_out}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tool", choices=["gemma", "gapit", "gwaspoly", "all"], default="all",
        help="which fixture to audit (default: all three)",
    )
    ap.add_argument(
        "--json-dir", default=str(HERE / "outputs"),
        help="directory to write per-tool JSON reports (default: outputs/)",
    )
    args = ap.parse_args(argv)

    tools = ["gemma", "gapit", "gwaspoly"] if args.tool == "all" else [args.tool]
    json_dir = Path(args.json_dir)

    reports: list[ToolReport] = []
    for tool in tools:
        try:
            rep = run_one(tool, json_out=json_dir / f"{tool}_drift.json")
        except FileNotFoundError as e:
            print(f"[{tool}] SKIP (infra-blocker): {e}", file=sys.stderr)
            # Skip cleanly; do NOT count as drift.
            continue
        reports.append(rep)

    print("=" * 90)
    print("  Pillar D summary")
    print("=" * 90)
    n_drifted_tools = sum(1 for r in reports if r.n_drifted > 0)
    for r in reports:
        verdict = "AUTHORITATIVE" if r.passed else f"DRIFT ({r.n_drifted} checks)"
        print(f"  {r.tool:10s}  {verdict}")
    if not reports:
        print("  (no tools produced fresh output; nothing to audit)")
        # Returning 0 — the orchestrator will surface the infra-blocker upstream.
        return 0
    print(
        f"\n  → {len(reports) - n_drifted_tools}/{len(reports)} fixtures authoritative; "
        f"{n_drifted_tools} drifted"
    )

    # Pillar D exits 0 even on drift: drift is informational
    # (regenerate-or-not is a user decision per spec §7).
    return 0


if __name__ == "__main__":
    sys.exit(main())
