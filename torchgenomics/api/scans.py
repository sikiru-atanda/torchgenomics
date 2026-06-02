"""GWAS-scan tier-1 API: :func:`lmm_scan`, :func:`glm_scan`.

These are the headline functions. They orchestrate format detection,
sample alignment, GRM (auto-streaming if not provided), null-model fit,
chunked variant scanning, multiple-testing correction, and result write —
end-to-end from file paths to a :class:`~torchgenomics.api._results.ScanRun`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import (
    ProgressCallback,
    emit_progress,
    resolve_output_dir,
    select_device,
    timed,
    top_hits,
)
from ._results import ScanRun

logger = logging.getLogger("torchgenomics.api")


def _load_genotype_matrix(genotype_path: str):
    """Re-export of cli._load_genotype_matrix (kept for backward import compat)."""
    from ..cli import _load_genotype_matrix as _impl

    return _impl(genotype_path)


def _genomic_inflation(p: pd.Series | list[float] | None) -> float | None:
    """λ_GC = median(chi²) / 0.4549. Returns None if input invalid/empty."""
    if p is None:
        return None
    try:
        from scipy.stats import chi2  # type: ignore[import-untyped]
        import numpy as np

        arr = pd.Series(p).dropna().astype(float).values
        if len(arr) == 0:
            return None
        chi2_obs = chi2.isf(arr, df=1)
        return float(np.nanmedian(chi2_obs) / 0.4549)
    except Exception:  # noqa: BLE001
        return None


def _resolve_single_trait(phenotype: str | Path, trait: str | None) -> str | None:
    """Resolve which single trait to scan against.

    - If ``trait`` is explicitly provided, return it unchanged.
    - Otherwise peek at the phenotype file header. If exactly one trait
      column exists, return None (CLI default behaviour — uses the only
      one available). If multiple traits exist, pick the first and return
      its name (matches what a novice user would expect on a single-trait
      function).
    """
    if trait is not None:
        return trait
    try:
        # Phenotype files start with FID/IID/sample_id columns; everything
        # after is trait columns. Peek the header.
        import pandas as pd

        header = pd.read_csv(phenotype, sep=None, engine="python", nrows=0).columns.tolist()
    except Exception:  # noqa: BLE001
        return None
    id_cols = {"FID", "IID", "fid", "iid", "sample_id", "Sample_ID", "ID", "id"}
    trait_cols = [c for c in header if c not in id_cols]
    if len(trait_cols) <= 1:
        return None  # No selection needed
    return trait_cols[0]


def _build_scan_namespace(**kwargs) -> argparse.Namespace:
    """Build a synthetic argparse.Namespace for the existing CLI orchestration.

    The CLI handler chain (`_cmd_lmm_scan_single`, `_align_samples`,
    `_apply_correction_and_save`) was written before the api layer existed
    and reads ``args.foo`` directly. Until those helpers are refactored
    to accept primitives (Task 25), the api layer passes a Namespace.
    """
    return argparse.Namespace(**kwargs)


@tool(
    name="tg_lmm_scan",
    title="Single-trait LMM GWAS scan",
    description=(
        "Run a single-trait linear mixed-model GWAS. Streams genotype chunks, "
        "auto-computes a VanRaden GRM (unless one is provided), fits the null "
        "with REML, runs the variant-wise score / Wald / LRT test, applies "
        "multiple-testing correction, and writes a TSV + Parquet summary. "
        "Returns ScanRun with .top_hits DataFrame, .lambda_gc, .manhattan() / .qq() plotting."
    ),
    long_running=True,
    category="scan",
    tags=["gwas", "lmm", "association"],
)
def lmm_scan(
    genotype: str | Path,
    phenotype: str | Path,
    *,
    trait: str | None = None,
    covariate: str | Path | None = None,
    output: str | Path | None = None,
    test: Literal["wald", "score", "lrt"] = "wald",
    correction: Literal[
        "bonferroni", "bh", "by", "holm", "storey", "weighted-bh", "lfdr",
        "hierarchical", "ihw", "adapt", "none",
    ] = "bh",
    chunk_size: int = 10_000,
    maf_min: float = 0.01,
    miss_max: float = 0.1,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    grm: str | Path | None = None,
    grm_method: Literal["vanraden", "zhang"] = "vanraden",
    n_pcs: int = 0,
    p3d: bool = True,
    significance_threshold: float = 5e-8,
    top_k: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> ScanRun:
    """Run a single-trait LMM GWAS — one call, end-to-end.

    Examples
    --------
    Novice one-liner::

        >>> import torchgenomics as tg
        >>> r = tg.lmm_scan(genotype="data.bed", phenotype="pheno.tsv")
        >>> print(r.summary())
        >>> r.manhattan()

    Parameters
    ----------
    genotype, phenotype
        Paths to the input files. Genotype format auto-detected.
    covariate
        Optional covariate TSV.
    output
        Output directory (auto-created if missing; auto-named under
        ``./torchgenomics_results/<id>/`` when ``None``).
    test
        Statistical test: ``"wald"`` (default), ``"score"`` (fastest), or ``"lrt"``.
    correction
        Multiple-testing correction (default ``"bh"`` = Benjamini-Hochberg).
    chunk_size
        Variants per streaming chunk. Tune for memory/speed; biobank default 10k.
    maf_min, miss_max
        Per-variant QC thresholds applied during the scan.
    device
        ``"cuda"`` / ``"cpu"`` / ``"auto"`` (CUDA if available).
    grm
        Optional pre-computed GRM file. If omitted, computes VanRaden GRM
        streaming (no full G in memory).
    grm_method
        ``"vanraden"`` (streaming, default) or ``"zhang"`` (requires full G).
    n_pcs
        Number of principal components extracted from the GRM and added to
        covariates (default 0: none).
    p3d
        Plug-in covariance ("population-parameters-previously-determined"):
        re-use null variance components for every variant. Set False for
        exact REML per variant (much slower).
    significance_threshold
        Cutoff for ``n_significant``. Default 5e-8 (genome-wide).
    top_k
        Number of top hits returned inline in :attr:`ScanRun.top_hits`.
    progress_callback
        Optional ``(fraction, message) -> None`` callback for long-running ops.

    Returns
    -------
    ScanRun
        Result object with summary stats, top hits DataFrame, and paths to
        the full TSV + Parquet on disk.
    """
    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_lmm_scan")
    output_prefix = output_dir / "results"
    device_resolved = select_device(device)
    selected_trait = _resolve_single_trait(phenotype, trait)

    emit_progress(progress_callback, 0.0, "starting LMM scan")

    with timed() as elapsed:
        # Build the synthetic Namespace matching the CLI's expected shape
        ns = _build_scan_namespace(
            genotype=str(genotype),
            phenotype=str(phenotype),
            covariate=str(covariate) if covariate else None,
            output=str(output_prefix),
            test=test,
            correction=correction,
            chunk_size=chunk_size,
            maf_min=maf_min,
            miss_max=miss_max,
            device=device_resolved,
            grm=str(grm) if grm else None,
            grm_method=grm_method,
            n_pcs=n_pcs,
            p3d=p3d,
            traits=selected_trait,
            loco=False,
            approx_method=None,
            approx_components=100,
            approx_landmarks=500,
            sparse_threshold=0.05,
            id_column=None,
            max_iter=None,
            no_save_parquet=False,
        )

        # Reuse the existing CLI dispatcher to do the heavy lifting.
        # _cmd_lmm_scan_single writes ``<output>.tsv`` (and parquet) on disk.
        from ..cli import _cmd_lmm_scan_single

        emit_progress(progress_callback, 0.1, "aligning samples + computing GRM")
        rc = _cmd_lmm_scan_single(ns)
        if rc != 0:
            raise RuntimeError(f"lmm_scan returned non-zero status {rc}")
        emit_progress(progress_callback, 0.9, "reading results")

        # Read the produced TSV back to summarize
        # CLI writes <prefix>.assoc.tsv (and .assoc.parquet); fall back to
        # plain .tsv for compatibility with older paths.
        tsv_path = Path(str(output_prefix) + ".assoc.tsv")
        if not tsv_path.exists():
            tsv_path = Path(str(output_prefix) + ".tsv")
        parquet_path = Path(str(output_prefix) + ".assoc.parquet")
        if not parquet_path.exists():
            parquet_path = Path(str(output_prefix) + ".parquet")
        output_files: dict[str, Path] = {}
        if tsv_path.exists():
            output_files["tsv"] = tsv_path
        if parquet_path.exists():
            output_files["parquet"] = parquet_path

        # Prefer Parquet (faster, typed) when available
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif tsv_path.exists():
            df = pd.read_csv(tsv_path, sep="\t")
        else:
            raise RuntimeError(
                f"lmm_scan completed but no results file found at {output_prefix}.tsv"
            )

        p_col = "P" if "P" in df.columns else "P_JOINT" if "P_JOINT" in df.columns else None
        n_sig = int((df[p_col] < significance_threshold).sum()) if p_col else 0
        lambda_gc = _genomic_inflation(df[p_col]) if p_col else None
        top = top_hits(df, k=top_k, p_column=p_col or "P")
        n_samples = int(df.attrs.get("n_samples", 0)) if hasattr(df, "attrs") else 0

        emit_progress(progress_callback, 1.0, "done")

        return ScanRun(
            runtime_s=elapsed(),
            output_files=output_files,
            model="SingleTraitLMM",
            test=test,
            correction=correction,
            n_variants=int(len(df)),
            n_significant=n_sig,
            significance_threshold=significance_threshold,
            lambda_gc=lambda_gc,
            n_samples=n_samples,
            top_hits=top,
        )


@tool(
    name="tg_glm_scan",
    title="GLM GWAS scan (Gaussian / binary / ordinal / multinomial)",
    description=(
        "Run a generalized-linear-model GWAS without random effects. Supports "
        "Gaussian (default), binary (logistic + optional Firth), ordinal "
        "(cumulative logit), and multinomial. Streams variant chunks, applies "
        "multiple-testing correction, writes TSV + Parquet. Returns ScanRun "
        "with the standard summary, top hits, and plotting helpers."
    ),
    long_running=True,
    category="scan",
    tags=["gwas", "glm", "binary", "ordinal", "multinomial", "logistic"],
)
def glm_scan(
    genotype: str | Path,
    phenotype: str | Path,
    *,
    trait: str | None = None,
    covariate: str | Path | None = None,
    output: str | Path | None = None,
    family: Literal["gaussian", "binary", "ordinal", "multinomial"] = "gaussian",
    n_categories: int | None = None,
    firth: bool = False,
    test: Literal["wald", "score", "lrt"] = "wald",
    correction: Literal[
        "bonferroni", "bh", "by", "holm", "storey", "weighted-bh", "lfdr",
        "hierarchical", "ihw", "adapt", "none",
    ] = "bh",
    chunk_size: int = 10_000,
    maf_min: float = 0.01,
    miss_max: float = 0.1,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    significance_threshold: float = 5e-8,
    top_k: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> ScanRun:
    """Run a GLM-family GWAS — Gaussian / binary / ordinal / multinomial."""
    if family in ("ordinal", "multinomial") and n_categories is None:
        raise ValueError(f"n_categories is required for family={family!r}")

    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_glm_scan")
    output_prefix = output_dir / "results"
    device_resolved = select_device(device)
    selected_trait = _resolve_single_trait(phenotype, trait)

    emit_progress(progress_callback, 0.0, f"starting GLM ({family}) scan")

    with timed() as elapsed:
        ns = _build_scan_namespace(
            genotype=str(genotype),
            phenotype=str(phenotype),
            covariate=str(covariate) if covariate else None,
            output=str(output_prefix),
            family=family,
            n_categories=n_categories,
            firth=firth,
            test=test,
            correction=correction,
            chunk_size=chunk_size,
            maf_min=maf_min,
            miss_max=miss_max,
            device=device_resolved,
            traits=selected_trait,
            id_column=None,
            no_save_parquet=False,
        )

        from ..cli import _cmd_glm_scan

        rc = _cmd_glm_scan(ns)
        if rc != 0:
            raise RuntimeError(f"glm_scan returned non-zero status {rc}")
        emit_progress(progress_callback, 0.9, "reading results")

        # CLI writes <prefix>.assoc.tsv (and .assoc.parquet); fall back to
        # plain .tsv for compatibility with older paths.
        tsv_path = Path(str(output_prefix) + ".assoc.tsv")
        if not tsv_path.exists():
            tsv_path = Path(str(output_prefix) + ".tsv")
        parquet_path = Path(str(output_prefix) + ".assoc.parquet")
        if not parquet_path.exists():
            parquet_path = Path(str(output_prefix) + ".parquet")
        output_files: dict[str, Path] = {}
        if tsv_path.exists():
            output_files["tsv"] = tsv_path
        if parquet_path.exists():
            output_files["parquet"] = parquet_path

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif tsv_path.exists():
            df = pd.read_csv(tsv_path, sep="\t")
        else:
            raise RuntimeError(
                f"glm_scan completed but no results file found at {output_prefix}.tsv"
            )

        p_col = "P" if "P" in df.columns else None
        n_sig = int((df[p_col] < significance_threshold).sum()) if p_col else 0
        lambda_gc = _genomic_inflation(df[p_col]) if p_col else None
        top = top_hits(df, k=top_k, p_column=p_col or "P")

        emit_progress(progress_callback, 1.0, "done")

        return ScanRun(
            runtime_s=elapsed(),
            output_files=output_files,
            model=f"GLM-{family}" + ("-firth" if firth else ""),
            test=test,
            correction=correction,
            n_variants=int(len(df)),
            n_significant=n_sig,
            significance_threshold=significance_threshold,
            lambda_gc=lambda_gc,
            top_hits=top,
        )
