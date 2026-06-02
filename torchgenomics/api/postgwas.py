"""Post-GWAS tier-1 API: :func:`clump`, :func:`meta`."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, resolve_output_dir, timed
from ._results import ClumpRun, MetaRun


@tool(
    name="tg_clump",
    title="LD clumping for independent loci",
    description=(
        "LD-clump GWAS summary statistics to identify independent index variants. "
        "Operates per-chromosome (an index SNP on chr 1 cannot clump a SNP on chr 2); "
        "streams the genotype matrix to bound memory. Returns ClumpRun with a "
        "DataFrame of clumps (each row = one index variant with its tag SNPs)."
    ),
    long_running=False,
    category="postgwas",
    tags=["clumping", "independence", "ld", "postgwas"],
)
def clump(
    sumstats: str | Path,
    genotype: str | Path,
    *,
    output: str | Path | None = None,
    p_threshold: float = 5e-8,
    r2_threshold: float = 0.1,
    bp_window_kb: int = 250,
    progress_callback: ProgressCallback | None = None,
) -> ClumpRun:
    """LD-clump GWAS sumstats against an LD-reference genotype panel."""
    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_clump")
    output_prefix = output_dir / "clumps"

    emit_progress(progress_callback, 0.0, "clumping")

    with timed() as elapsed:
        ns = argparse.Namespace(
            sumstats=str(sumstats),
            genotype=str(genotype),
            output=str(output_prefix),
            p_threshold=p_threshold,
            r2_threshold=r2_threshold,
            bp_window_kb=bp_window_kb,
        )

        from ..cli import _cmd_clump

        rc = _cmd_clump(ns)
        if rc not in (None, 0):
            raise RuntimeError(f"clump returned non-zero status {rc}")

        out_tsv = Path(f"{output_prefix}.tsv")
        output_files: dict[str, Path] = {}
        if out_tsv.exists():
            output_files["tsv"] = out_tsv
            df = pd.read_csv(out_tsv, sep="\t")
        else:
            df = pd.DataFrame()

        # Count input variants for context
        n_input = 0
        try:
            n_input = int(sum(1 for _ in open(sumstats)) - 1)  # minus header
        except Exception:  # noqa: BLE001
            pass

        return ClumpRun(
            runtime_s=elapsed(),
            output_files=output_files,
            n_input_variants=n_input,
            n_clumps=int(len(df)),
            n_index_variants=int(len(df)),
            p_threshold=p_threshold,
            r2_threshold=r2_threshold,
            clumps=df,
        )


@tool(
    name="tg_meta",
    title="Meta-analysis across GWAS sumstats",
    description=(
        "Combine GWAS summary statistics across studies (fixed effect / random "
        "effect / Han-Eskin / Stouffer sample-size). Inputs are aligned by SNP; "
        "writes a single TSV with per-variant meta beta/se/p plus heterogeneity. "
        "Returns MetaRun with the results DataFrame inline and a summary I²."
    ),
    long_running=False,
    category="postgwas",
    tags=["meta-analysis", "ivw", "stouffer", "random-effect", "han-eskin"],
)
def meta(
    inputs: list[str | Path],
    *,
    output: str | Path | None = None,
    method: Literal["fixed", "random", "han_eskin", "stouffer"] = "fixed",
    significance_threshold: float = 5e-8,
) -> MetaRun:
    """Meta-analyze GWAS summary statistics across studies.

    Parameters
    ----------
    inputs
        List of paths to sumstats files (at least 2).
    method
        ``"fixed"`` (IVW), ``"random"`` (DerSimonian-Laird), ``"han_eskin"``
        (RE2 with heterogeneity-aware test), or ``"stouffer"`` (sample-size
        weighted z-combination).
    """
    if len(inputs) < 2:
        raise ValueError(f"meta requires at least 2 input sumstats; got {len(inputs)}")

    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_meta")
    output_prefix = output_dir / "meta"

    with timed() as elapsed:
        ns = argparse.Namespace(
            input=[str(p) for p in inputs],
            output=str(output_prefix),
            method=method,
        )

        from ..cli import _cmd_meta

        rc = _cmd_meta(ns)
        if rc not in (None, 0):
            raise RuntimeError(f"meta returned non-zero status {rc}")

        meta_tsv = Path(f"{output_prefix}.meta.tsv")
        output_files: dict[str, Path] = {}
        if meta_tsv.exists():
            output_files["tsv"] = meta_tsv
            df = pd.read_csv(meta_tsv, sep="\t")
        else:
            df = pd.DataFrame()

        n_sig = int((df["p_meta"] < significance_threshold).sum()) if "p_meta" in df.columns else 0
        het_i2 = float(df["i2"].median()) if "i2" in df.columns and not df.empty else None

        return MetaRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=method,
            n_studies=len(inputs),
            n_variants=int(len(df)),
            n_significant=n_sig,
            results=df,
            heterogeneity_i2_median=het_i2,
        )
