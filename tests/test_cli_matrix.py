"""End-to-end CLI smoke matrix (Pillar C / spec §6).

This is the parametric harness that runs every public ``torchgenomics`` CLI
subcommand against every supported committed fixture format and asserts
the documented exit-0 + output-files-exist + p-values-in-[0,1] +
row-count contract.

The full cell map lives in :mod:`tests.cli_matrix_spec`. This module is
the executor: it builds per-cell scratch inputs (synthetic phenotypes,
sumstats, mediation tensors), spawns the CLI subprocess, and applies
the per-kind assertions.

Skipped by default; opt in with ``pytest -m cli_matrix``.

GPU cells (``test_cli_gpu_cell``) are gated on
``torch.cuda.is_available()``. They use the same CELL_SPEC entries but
override ``--device cuda``.

Failure mode policy (per spec §9):

- Exit non-zero or expected output missing: real F3 finding. Test fails;
  fix-now if V1-platform.
- Subcommand requires unsupported feature (Julia, Docker, NCBI network,
  external binary): ``pytest.skip`` with documented reason.
- Numerical degeneracy on n=10 / m=20 (e.g., LDSC): documented in
  CELL_SPEC's ``skip_reason``; test skips with that message.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.cli_matrix_spec import (
    CELL_SPEC,
    FIXTURES,
    FORMAT_GENOTYPE,
    N_VARIANTS,
    Ctx,
    expand_cells,
    expand_gpu_cells,
)

pytestmark = pytest.mark.cli_matrix

# Repo root (this file's parent's parent). The CLI subprocess uses
# PYTHONPATH=<repo_root> so `python -m torchgenomics.cli` resolves the
# in-tree package without a pip install.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-CLI subprocess timeout (seconds). Tiny fixture cells finish in
# under 5s on CPU; the cap is generous to absorb variance on shared
# CI nodes without masking real hangs.
CLI_TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Per-cell scratch input fabrication
# ---------------------------------------------------------------------------


def _write_pheno_clean(tmp: Path) -> None:
    """Phenotype with no NA cells (mvlmm / threshold need this)."""
    df = pd.read_csv(FIXTURES / "tiny_pheno.txt", sep="\t")
    # Filter out rows with NA in either Y1 or Y2; keep only fixture sample IDs.
    df = df[df["IID"].str.startswith("IND")].copy()
    df["Y2"] = df["Y2"].fillna(0.5)
    df.to_csv(tmp / "pheno_clean.txt", sep="\t", index=False, na_rep="NA")


def _write_pheno_ord(tmp: Path) -> None:
    """3-category ordinal Y1, continuous Y2, no NA."""
    df = pd.read_csv(FIXTURES / "tiny_pheno.txt", sep="\t")
    df = df[df["IID"].str.startswith("IND")].copy()
    df["Y2"] = df["Y2"].fillna(0.5)
    rng = np.random.default_rng(0)
    df["Y1"] = rng.integers(0, 3, size=len(df)).astype(int)
    df.to_csv(tmp / "pheno_ord.txt", sep="\t", index=False)


def _write_pheno_bin(tmp: Path) -> None:
    """Binary phenotype (Y1 ∈ {0,1})."""
    rng = np.random.default_rng(1)
    df = pd.read_csv(FIXTURES / "tiny_pheno.txt", sep="\t")
    df = df[df["IID"].str.startswith("IND")].copy()
    df["Y1"] = rng.integers(0, 2, size=len(df)).astype(int)
    df["Y2"] = rng.integers(0, 2, size=len(df)).astype(int)
    df.to_csv(tmp / "pheno_bin.txt", sep="\t", index=False)


def _write_pheno_me_bin(tmp: Path) -> None:
    """Per-environment binary phenotypes for me-glmm-scan."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"IID": [f"IND{i:03d}" for i in range(10)]})
    df["E1"] = rng.integers(0, 2, size=10).astype(int)
    df["E2"] = rng.integers(0, 2, size=10).astype(int)
    df.to_csv(tmp / "pheno_me_bin.txt", sep="\t", index=False)


def _write_pheno_fam(tmp: Path) -> None:
    """Family-stratified phenotype (FID col)."""
    df = pd.read_csv(FIXTURES / "tiny_pheno.txt", sep="\t")
    df = df[df["IID"].str.startswith("IND")].copy()
    # 4 samples in F0, 3 in F1, 3 in F2 → all families have ≥2 members.
    df["FID"] = ["F0"] * 4 + ["F1"] * 3 + ["F2"] * 3
    df["Y2"] = df["Y2"].fillna(0.0)
    df.to_csv(tmp / "pheno_fam.txt", sep="\t", index=False)


def _write_pheno_met(tmp: Path) -> None:
    """Multi-env wide phenotype (E1/E2/E3)."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"IID": [f"IND{i:03d}" for i in range(10)]})
    for env in ("E1", "E2", "E3"):
        df[env] = rng.normal(0, 1, size=10)
    df.to_csv(tmp / "pheno_met.txt", sep="\t", index=False)


def _write_pheno_mtmet(tmp: Path) -> None:
    """2-trait × 2-env MT-MET phenotype (long format)."""
    rng = np.random.default_rng(4)
    rows = []
    for iid in [f"IND{i:03d}" for i in range(10)]:
        rows.append({
            "IID": iid,
            "Y1_E1": float(rng.normal(0, 1)),
            "Y1_E2": float(rng.normal(0, 1)),
            "Y2_E1": float(rng.normal(0, 1)),
            "Y2_E2": float(rng.normal(0, 1)),
            "E1": float(rng.normal(0, 1)),
            "E2": float(rng.normal(0, 1)),
        })
    df = pd.DataFrame(rows)
    # Column-name convention required by mtmet-scan: traits Y1,Y2 and
    # env-cols E1,E2; the scanner finds Y1_E1/Y1_E2/Y2_E1/Y2_E2 by
    # cross-product and uses E1/E2 as raw env covariates.
    df.to_csv(tmp / "pheno_mtmet.txt", sep="\t", index=False)


def _write_pheno_long(tmp: Path) -> None:
    """Long-format longitudinal phenotype (T_i=4 per individual)."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(10):
        for t in (1, 2, 3, 4):
            rows.append({
                "IID": f"IND{i:03d}",
                "TIME": t,
                "Y": float(rng.normal(0, 1) + 0.1 * t),
            })
    pd.DataFrame(rows).to_csv(tmp / "pheno_long.txt", sep="\t", index=False)


def _write_pheno_rr_met(tmp: Path) -> None:
    """Long-format MET phenotype (T_i=4 per individual per env)."""
    rng = np.random.default_rng(6)
    rows = []
    for i in range(10):
        for env in ("E1", "E2"):
            for t in (1, 2, 3, 4):
                rows.append({
                    "IID": f"IND{i:03d}",
                    "ENV": env,
                    "TIME": t,
                    "Y": float(rng.normal(0, 1) + 0.1 * t),
                })
    pd.DataFrame(rows).to_csv(tmp / "pheno_rr_met.txt", sep="\t", index=False)


def _write_pheno_surv(tmp: Path) -> None:
    """(time, event) phenotype for survival-scan."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(10):
        rows.append({
            "IID": f"IND{i:03d}",
            "TIME": float(rng.uniform(1, 10)),
            "EVENT": int(rng.integers(0, 2)),
        })
    pd.DataFrame(rows).to_csv(tmp / "pheno_surv.txt", sep="\t", index=False)


def _write_env_tsv(tmp: Path) -> None:
    """Continuous env covariate (SAMPLE, ENV) for gxe-scan."""
    rng = np.random.default_rng(8)
    rows = [{"SAMPLE": f"IND{i:03d}", "ENV": float(rng.normal(0, 1))} for i in range(10)]
    pd.DataFrame(rows).to_csv(tmp / "env.tsv", sep="\t", index=False)


def _write_regions_bed(tmp: Path) -> None:
    """3-region BED file for set-scan."""
    with open(tmp / "regions.bed", "w") as fh:
        fh.write("1\t0\t10000\tregion1\n")
        fh.write("2\t0\t10000\tregion2\n")
        fh.write("3\t0\t10000\tregion3\n")


def _write_sumstats_lc(tmp: Path) -> None:
    """Lowercase-column sumstats by running glm-scan + relabeling.

    `torchgenomics.postgwas.load_sumstats` (and its consumers: meta, clump,
    ldsc, ldsc-rg) expect lowercase column names ``chr/pos/snp/a1/a2/
    beta/se/p/n``. The scanner emits uppercase. We run glm-scan once
    and lowercase the columns + add an ``n`` column so this fixture is
    valid for every consumer.
    """
    if (tmp / "sumstats_lc.tsv").exists():
        return
    base = tmp / "_glm_for_sumstats"
    cmd = [
        sys.executable, "-m", "torchgenomics.cli", "glm-scan",
        "--genotype", str(FIXTURES / "tiny.bed"),
        "--phenotype", str(FIXTURES / "tiny_pheno.txt"),
        "--traits", "Y1",
        "--device", "cpu",
        "--output", str(base),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(tmp), check=True, env=env, timeout=CLI_TIMEOUT_S,
                   capture_output=True)
    df = pd.read_csv(f"{base}.assoc.tsv", sep="\t")
    df.columns = [c.lower() for c in df.columns]
    df["n"] = 10
    df.to_csv(tmp / "sumstats_lc.tsv", sep="\t", index=False)


def _write_pgs_weights(tmp: Path) -> None:
    """3-SNP pgs weights TSV for pgs-score (snp/chr/pos/a1/a2/weight)."""
    df = pd.DataFrame({
        "snp": ["rs0000", "rs0001", "rs0002"],
        "chr": ["1", "2", "3"],
        "pos": [1000, 2000, 3000],
        "a1": ["A", "C", "A"],
        "a2": ["G", "T", "T"],
        "weight": [0.05, -0.03, 0.10],
    })
    df.to_csv(tmp / "pgs_weights.tsv", sep="\t", index=False)


def _write_ld_ref(tmp: Path) -> None:
    """LD reference panel built from the tiny BED fixture (full mode)."""
    if (tmp / "ld_ref.pt").exists():
        return
    from torchgenomics.io.plink import PlinkBedReader
    from torchgenomics.pgs import build_ld_reference, save_ld_reference

    reader = PlinkBedReader(str(FIXTURES / "tiny"))
    chunks = []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks.append((G_chunk, vmeta))
    G = torch.cat([c[0] for c in chunks], dim=1).double()
    G = torch.nan_to_num(G, nan=0.0)
    vm = chunks[0][1]
    snp_meta = {
        "snp": list(vm.snp),
        "chr": list(vm.chr),
        "pos": list(vm.pos),
        "a1": list(vm.a1),
        "a2": list(vm.a2),
    }
    ld = build_ld_reference(G, snp_meta, mode="full")
    save_ld_reference(ld, str(tmp / "ld_ref.pt"))


def _write_mediate_tensors(tmp: Path) -> None:
    """4-tensor scratch (y, snp, m, K) for mediate."""
    rng = np.random.default_rng(9)
    n = 30  # n=10 is too small for the LMM null fit on a single snp
    np.save(tmp / "y.npy", rng.normal(0, 1, n).astype(np.float64))
    np.save(tmp / "snp.npy", rng.binomial(2, 0.3, n).astype(np.float64))
    np.save(tmp / "m.npy", rng.normal(0, 1, n).astype(np.float64))
    K = rng.normal(0, 1, (n, n))
    K = K @ K.T / n + np.eye(n) * 1e-3
    np.save(tmp / "K.npy", K.astype(np.float64))


def _write_mediate_scan_tensors(tmp: Path) -> None:
    """5-tensor scratch (Y, G, M, K) for mediate-scan."""
    rng = np.random.default_rng(10)
    n, s, f = 30, 5, 3
    np.save(tmp / "Y.npy", rng.normal(0, 1, n).astype(np.float64))
    np.save(tmp / "G.npy", rng.binomial(2, 0.3, (n, s)).astype(np.float64))
    np.save(tmp / "M.npy", rng.normal(0, 1, (n, f)).astype(np.float64))
    K = rng.normal(0, 1, (n, n))
    K = K @ K.T / n + np.eye(n) * 1e-3
    np.save(tmp / "K.npy", K.astype(np.float64))


# Per-subcommand scratch fabricators. Keys mirror CELL_SPEC keys; missing
# keys mean no per-cell scratch input is needed.
_PER_CELL_SCRATCH: dict[str, list] = {
    "mvlmm-scan":     [_write_pheno_clean],
    "threshold-scan": [_write_pheno_ord],
    "glmm-scan":      [_write_pheno_bin],
    "me-glmm-scan":   [_write_pheno_me_bin],
    "family-scan":    [_write_pheno_fam],
    "met-scan":       [_write_pheno_met],
    "mtmet-scan":     [_write_pheno_mtmet],
    "rr-scan":        [_write_pheno_long],
    "rr-met-scan":    [_write_pheno_rr_met],
    "survival-scan":  [_write_pheno_surv],
    "gxe-scan":       [_write_env_tsv],
    "set-scan":       [_write_regions_bed],
    "ldsc":           [_write_sumstats_lc],
    "ldsc-rg":        [_write_sumstats_lc],
    "meta":           [_write_sumstats_lc],
    "clump":          [_write_sumstats_lc],
    "annotate":       [_write_sumstats_lc],
    "pgs-fit":        [_write_sumstats_lc, _write_ld_ref],
    "pgs-score":      [_write_pgs_weights],
    "mediate":        [_write_mediate_tensors],
    "mediate-scan":   [_write_mediate_scan_tensors],
}


def _stage_scratch(subcmd: str, tmp: Path) -> None:
    """Run scratch fabricators for a given subcommand cell."""
    for fabricator in _PER_CELL_SCRATCH.get(subcmd, []):
        fabricator(tmp)


# ---------------------------------------------------------------------------
# Per-kind assertions
# ---------------------------------------------------------------------------


def _assert_p_columns_in_unit_interval(df: pd.DataFrame, *, allow_nan: bool = True) -> None:
    """Every column whose name starts with 'P' (case-insensitive) must
    contain values in [0, 1].

    NaN tolerance is on by default at the smoke tier: monomorphic SNPs
    (whose AF is NaN after QC) propagate NaN into every test stat, and
    correction families (BH / IHW / AdaPT) propagate that NaN to P_ADJ.
    The contract that matters at smoke is "every finite p in [0, 1]" —
    the strict-finite gate belongs to Tier 1 / Tier 2, not Pillar C.

    Pass ``allow_nan=False`` if a per-cell invariant says no monomorphic
    SNPs and every p must be finite. The default ``True`` is appropriate
    for the tiny fixture which has rs0002 monomorphic by construction.
    """
    p_cols = [c for c in df.columns if c.upper() == "P" or c.upper().startswith("P_")]
    assert p_cols, f"No P / P_* columns in {df.columns.tolist()}"
    for col in p_cols:
        s = df[col]
        if s.isna().any():
            assert allow_nan, (
                f"NaN p-values in column {col!r} but allow_nan=False; "
                f"head:\n{df.head()}"
            )
            s = s.dropna()
        assert s.between(0.0, 1.0).all(), (
            f"P column {col!r} has values outside [0, 1]:\n{s[~s.between(0.0, 1.0)]}"
        )


def _assert_scan_output(out: Path, expected_n: int, *, allow_nan_p: bool = True) -> None:
    """Assertions for ``{prefix}.assoc.tsv`` (default scan).

    ``allow_nan_p`` defaults to True at the smoke tier — see the docstring
    on :func:`_assert_p_columns_in_unit_interval` for the rationale. Cells
    whose contract is "no monomorphic SNPs allowed" can override per-cell
    via CELL_SPEC's ``allow_nan_p: False``.
    """
    assert out.exists() and out.stat().st_size > 0, f"Missing/empty {out}"
    df = pd.read_csv(out, sep="\t")
    # Allow ≤ expected (some scans drop monomorphic), require ≥ 1.
    assert 1 <= len(df) <= expected_n, (
        f"{out}: got {len(df)} rows, expected up to {expected_n}"
    )
    _assert_p_columns_in_unit_interval(df, allow_nan=allow_nan_p)


def _assert_set_scan_output(out: Path) -> None:
    """Assertions for ``{prefix}_set_based.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) >= 1
    _assert_p_columns_in_unit_interval(df, allow_nan=True)


def _assert_bayes_output(out: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}_bayesian_vs.tsv``.

    The Bayesian VS output is a per-SNP table of (PIP, posterior mean,
    posterior SD); it does not contain a frequentist p-value column.
    """
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) == expected_n
    pip_cols = [c for c in df.columns if "pip" in c.lower()]
    assert pip_cols, f"No PIP column in {df.columns.tolist()}"
    for c in pip_cols:
        assert df[c].between(0.0, 1.0).all(), f"PIP column {c!r} out of [0,1]"


def _assert_met_scan_output(out: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}_met_scan.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert 1 <= len(df) <= expected_n
    _assert_p_columns_in_unit_interval(df, allow_nan=True)


def _assert_knockoff_output(prefix: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}.knockoff.tsv`` + ``.knockoff_blocks.tsv``."""
    knock_path = prefix.with_suffix(".knockoff.tsv")
    blocks_path = prefix.with_suffix(".knockoff_blocks.tsv")
    assert knock_path.exists() and knock_path.stat().st_size > 0
    assert blocks_path.exists() and blocks_path.stat().st_size > 0
    df = pd.read_csv(knock_path, sep="\t")
    assert 1 <= len(df) <= expected_n


def _assert_rr_output(out: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}.rr.tsv`` (random-regression scan)."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert 1 <= len(df) <= expected_n
    _assert_p_columns_in_unit_interval(df, allow_nan=True)


def _assert_rr_met_output(out: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}.rr_met.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert 1 <= len(df) <= expected_n
    _assert_p_columns_in_unit_interval(df, allow_nan=True)


def _assert_ld_blocks_output(prefix_str: str, tmp: Path) -> None:
    """Assertions for ld-blocks output triple."""
    bed = tmp / f"{prefix_str}.bed"
    det = tmp / f"{prefix_str}.blocks.det"
    summary = tmp / f"{prefix_str}.summary.txt"
    for f in (bed, det, summary):
        assert f.exists(), f"Missing ld-blocks output {f}"
    # det may be empty (0 blocks detected); summary should have content.
    assert summary.stat().st_size > 0


def _assert_ldsc_output(out: Path) -> None:
    """Assertions for ``{prefix}.ldsc.txt`` (h² report)."""
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text()
    assert "h2" in text.lower()


def _assert_ldsc_rg_output(out: Path) -> None:
    """Assertions for ``{prefix}.ldsc_rg.txt``."""
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text()
    assert "rg" in text.lower()


def _assert_meta_output(out: Path, expected_n: int) -> None:
    """Assertions for ``{prefix}.meta.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) == expected_n
    assert "p_meta" in df.columns
    s = df["p_meta"].dropna()
    assert s.between(0.0, 1.0).all()


def _assert_clump_output(out: Path) -> None:
    """Assertions for ``{prefix}.clumps.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) >= 0  # may be 0 if no SNPs pass p-threshold
    if len(df) > 0:
        assert "p" in df.columns
        assert df["p"].between(0.0, 1.0).all()


def _assert_pgs_fit_output(out: Path) -> None:
    """Assertions for pgs-fit output TSV (snp/chr/pos/a1/a2/weight)."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) >= 1
    assert "weight" in df.columns


def _assert_pgs_score_output(out: Path) -> None:
    """Assertions for pgs-score output TSV (FID / IID / PGS).

    The CLI emits the canonical column ``PGS`` (a polygenic score is by
    definition one column per individual); accept ``PGS`` or any
    case-insensitive variant containing ``score``.
    """
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) >= 1
    score_cols = [
        c for c in df.columns
        if "score" in c.lower() or c.upper() in {"PGS", "SCORE"}
    ]
    assert score_cols, f"No PGS / score column in {df.columns.tolist()}"


def _assert_annotate_output(out: Path) -> None:
    """Assertions for annotate ``{prefix}.genes.tsv``."""
    assert out.exists() and out.stat().st_size > 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) >= 1


def _assert_convert_output(out: Path) -> None:
    """Assertions for convert output (BED/zarr/VCF)."""
    if out.suffix == ".zarr":
        assert out.is_dir(), f"zarr store missing: {out}"
        # Standard zarr v2 / v3 layout: at least one .zarray/.zattrs/zarr.json
        sentinels = list(out.rglob(".zarray")) + list(out.rglob("zarr.json"))
        assert sentinels, f"zarr {out} has no .zarray / zarr.json sentinel"
    elif out.suffix == ".bed":
        assert out.exists() and out.stat().st_size > 0
        with open(out, "rb") as fh:
            magic = fh.read(3)
        assert magic == bytes([0x6C, 0x1B, 0x01]), (
            f"BED {out} bad magic: {magic!r}"
        )
    else:
        assert out.exists() and out.stat().st_size > 0


def _assert_impute_output(out: Path) -> None:
    """Assertions for impute output ``.pt`` blob."""
    assert out.exists() and out.stat().st_size > 0
    blob = torch.load(out, weights_only=False)
    assert isinstance(blob, dict) and "dosage" in blob
    G = blob["dosage"]
    assert G.numel() > 0
    assert not torch.isnan(G).any(), f"impute output has NaN: {out}"


def _assert_validate_output(out: Path) -> None:
    """Assertions for validate JSON report."""
    assert out.exists() and out.stat().st_size > 0
    blob = json.loads(out.read_text())
    assert "n_samples_genotype" in blob
    assert "n_variants_total" in blob


def _assert_mediate_output(out: Path) -> None:
    """Assertions for mediate JSON output."""
    assert out.exists() and out.stat().st_size > 0
    blob = json.loads(out.read_text())
    assert "indirect" in blob or "ACME" in blob or "estimate" in blob, blob


def _assert_mediate_scan_output(prefix: Path) -> None:
    """Assertions for mediate-scan ``{prefix}.tsv`` + ``.top.tsv``."""
    flat = Path(f"{prefix}.tsv")
    top = Path(f"{prefix}.top.tsv")
    assert flat.exists() and flat.stat().st_size > 0
    assert top.exists()  # may be empty if no significant pairs


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def _run_cli(subcmd: str, args: list[str], tmp: Path) -> subprocess.CompletedProcess:
    """Run ``python -m torchgenomics.cli <subcmd> <args>`` from ``tmp``.

    Sets PYTHONPATH so the in-tree package resolves without a pip install.
    Captures stdout + stderr for diagnostic output on failure.
    """
    cmd = [sys.executable, "-m", "torchgenomics.cli", subcmd, *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        cwd=str(tmp),
        env=env,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_S,
    )


# ---------------------------------------------------------------------------
# Per-cell test
# ---------------------------------------------------------------------------


def _resolve_skip(subcmd: str, fmt: str) -> str | None:
    """Return a skip reason for the given (subcommand, format) cell, or None."""
    spec = CELL_SPEC[subcmd]
    sk = spec.get("skip_reason")
    if sk is None:
        return None
    if isinstance(sk, str):
        return sk
    if isinstance(sk, dict):
        return sk.get(fmt)
    return None


def _validate_kind_outputs(
    spec: dict,
    ctx: Ctx,
    fmt: str,
) -> None:
    """Apply the kind-specific output assertions for one cell."""
    kind = spec.get("kind", "scan")
    expected_n = spec.get("expected_n_variants", N_VARIANTS)
    # Smoke tier: NaN p-values from monomorphic SNPs are tolerated by
    # default. Cells that contract for "every variant has finite p" may
    # override via CELL_SPEC entry: ``allow_nan_p: False``.
    allow_nan_p = spec.get("allow_nan_p", True)
    expected_outputs = spec["expected_outputs"](ctx, fmt)

    # The harness changed cwd to ctx.tmp before running; expected_outputs
    # are relative paths.
    out_paths = [ctx.tmp / o if not Path(o).is_absolute() else Path(o)
                 for o in expected_outputs]

    # First gate: every declared output must exist (size check varies).
    for o in out_paths:
        assert o.exists(), f"Expected output not produced: {o}"

    # Kind-specific asserts.
    if kind == "scan":
        _assert_scan_output(out_paths[0], expected_n, allow_nan_p=allow_nan_p)
    elif kind == "mvlmm_scan":
        # mvlmm now writes per-trait BETA_d0/BETA_d1 + scalar P (joint).
        _assert_scan_output(out_paths[0], expected_n, allow_nan_p=True)
    elif kind == "gxe_scan":
        # gxe writes BETA_MAIN/BETA_INTERACT/STAT_JOINT + P_MAIN/P_INTERACT/P.
        _assert_scan_output(out_paths[0], expected_n, allow_nan_p=True)
    elif kind == "set_scan":
        _assert_set_scan_output(out_paths[0])
    elif kind == "bayes":
        _assert_bayes_output(out_paths[0], expected_n)
    elif kind == "met_scan":
        _assert_met_scan_output(out_paths[0], expected_n)
    elif kind == "knockoff":
        # First expected_outputs entry is the knockoff TSV; second the blocks TSV.
        prefix = ctx.tmp / ctx.output_prefix
        _assert_knockoff_output(prefix, expected_n)
    elif kind == "rr":
        _assert_rr_output(out_paths[0], expected_n)
    elif kind == "rr_met":
        _assert_rr_met_output(out_paths[0], expected_n)
    elif kind == "ld_blocks":
        _assert_ld_blocks_output(ctx.output_prefix, ctx.tmp)
    elif kind == "ldsc":
        _assert_ldsc_output(out_paths[0])
    elif kind == "ldsc_rg":
        _assert_ldsc_rg_output(out_paths[0])
    elif kind == "meta":
        _assert_meta_output(out_paths[0], expected_n)
    elif kind == "clump":
        _assert_clump_output(out_paths[0])
    elif kind == "pgs_fit":
        _assert_pgs_fit_output(out_paths[0])
    elif kind == "pgs_score":
        _assert_pgs_score_output(out_paths[0])
    elif kind == "annotate":
        _assert_annotate_output(out_paths[0])
    elif kind == "convert":
        _assert_convert_output(out_paths[0])
    elif kind == "impute":
        _assert_impute_output(out_paths[0])
    elif kind == "validate":
        _assert_validate_output(out_paths[0])
    elif kind == "mediate":
        _assert_mediate_output(out_paths[0])
    elif kind == "mediate_scan":
        _assert_mediate_scan_output(ctx.tmp / ctx.output_prefix)
    else:
        raise AssertionError(f"Unknown CELL_SPEC kind {kind!r} for cell")


@pytest.mark.parametrize("subcmd, fmt", expand_cells())
def test_cli_cpu_cell(subcmd: str, fmt: str, tmp_path: Path) -> None:
    """One CPU smoke cell: ``(subcommand, format)``.

    Runs the CLI subprocess and applies the per-kind assertions. Cells
    with a documented ``skip_reason`` skip cleanly with that message.
    """
    skip_reason = _resolve_skip(subcmd, fmt)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    if fmt != "n/a" and FORMAT_GENOTYPE.get(fmt) is None:
        pytest.skip(f"no committed fixture for format {fmt!r}")

    spec = CELL_SPEC[subcmd]
    ctx = Ctx(fixtures=FIXTURES, tmp=tmp_path)

    _stage_scratch(subcmd, tmp_path)
    args = spec["build_args"](ctx, fmt)

    proc = _run_cli(subcmd, args, tmp_path)
    if proc.returncode != 0:
        pytest.fail(
            f"CLI cell ({subcmd}, {fmt}) exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- args   ---\n{args}\n"
        )

    _validate_kind_outputs(spec, ctx, fmt)


# ---------------------------------------------------------------------------
# GPU subset
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("subcmd, fmt", expand_gpu_cells())
def test_cli_gpu_cell(subcmd: str, fmt: str, tmp_path: Path) -> None:
    """One GPU smoke cell: same CELL_SPEC, ``--device cuda`` instead of cpu.

    Skipped when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    skip_reason = _resolve_skip(subcmd, fmt)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    spec = CELL_SPEC[subcmd]
    ctx = Ctx(fixtures=FIXTURES, tmp=tmp_path)

    _stage_scratch(subcmd, tmp_path)
    args = spec["build_args"](ctx, fmt)

    # Swap "--device cpu" → "--device cuda".
    new_args: list[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            new_args.append("cuda")
            continue
        if a == "--device":
            new_args.append(a)
            skip_next = True
        else:
            new_args.append(a)
    if "--device" not in new_args:
        new_args += ["--device", "cuda"]
    args = new_args

    proc = _run_cli(subcmd, args, tmp_path)
    if proc.returncode != 0:
        pytest.fail(
            f"GPU cell ({subcmd}, {fmt}) exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- args   ---\n{args}\n"
        )

    _validate_kind_outputs(spec, ctx, fmt)


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------


def test_cli_matrix_covers_every_subcommand() -> None:
    """The matrix must enumerate every subcommand exposed by ``torchgenomics.cli``.

    Guards against the case where a new subparser lands in ``torchgenomics/cli.py``
    but no one updated CELL_SPEC. Failure here means an entry must be added
    (with ``skip_reason`` if the subcommand can't run on the tiny fixture).
    """
    import argparse
    import torchgenomics.cli as cli_mod

    # Build the parser the same way `cli_mod.main` does, then enumerate the
    # subparsers it registers. We don't dispatch.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    for fn_name in dir(cli_mod):
        fn = getattr(cli_mod, fn_name)
        if fn_name.startswith("_add_") and fn_name.endswith("_parser") and callable(fn):
            try:
                fn(sub)
            except Exception:
                # Some _add_*_parser helpers may have signature variations.
                continue

    # Fallback: parse the help output of the in-tree CLI to be authoritative.
    proc = subprocess.run(
        [sys.executable, "-m", "torchgenomics.cli", "--help"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    # Extract the {x,y,z,...} positional list from the help banner.
    import re
    match = re.search(r"\{([^}]+)\}", proc.stdout)
    assert match, f"Could not find subcommand list in --help:\n{proc.stdout}"
    declared = [s.strip() for s in match.group(1).split(",")]

    missing = [s for s in declared if s not in CELL_SPEC]
    assert not missing, (
        f"CELL_SPEC missing entries for these CLI subcommands: {missing}\n"
        "Add an entry (with skip_reason if the cell can't smoke-test)."
    )
    extra = [s for s in CELL_SPEC if s not in declared]
    assert not extra, (
        f"CELL_SPEC has stale entries for non-existent subcommands: {extra}"
    )
