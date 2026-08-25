"""Post-GWAS tier-1 API: :func:`clump`, :func:`meta`, :func:`mr`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, resolve_output_dir, timed
from ._results import ClumpRun, MetaRun, MRRun


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
    r2: float = 0.1,
    window_kb: float = 250.0,
    progress_callback: ProgressCallback | None = None,
) -> ClumpRun:
    """LD-clump GWAS sumstats against an LD-reference genotype panel.

    Parameters
    ----------
    sumstats, genotype
        Paths to the GWAS summary statistics and the LD-reference genotype panel.
    output
        Output directory or file prefix. If None, auto-creates a unique run dir.
    p_threshold
        P-value cutoff for index variants (default ``5e-8``).
    r2
        r² ceiling for clumping (default ``0.1``).
    window_kb
        Clumping window in kilobases (default ``250``).
    """
    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_clump")
    output_prefix = output_dir / "clumps"

    emit_progress(progress_callback, 0.0, "clumping")

    with timed() as elapsed:
        from ..cli import _run_clump

        info = _run_clump(
            sumstats_path=str(sumstats),
            genotype_path=str(genotype),
            output_prefix=str(output_prefix),
            r2=r2,
            p_threshold=p_threshold,
            window_kb=window_kb,
        )

        out_tsv = Path(info["output_path"])
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
            n_clumps=int(info["n_clumps"]),
            n_index_variants=int(info["n_clumps"]),
            p_threshold=p_threshold,
            r2_threshold=r2,
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
        from ..cli import _run_meta

        info = _run_meta(
            inputs=[str(p) for p in inputs],
            output_prefix=str(output_prefix),
            method=method,
        )

        meta_tsv = Path(info["output_path"])
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


_MR_METHODS = ("ivw", "egger", "weighted_median", "presso", "all")

#: `postgwas.mr_presso`'s `MRResult.method` field is literally `"mr_presso"`
#: (see `torchgenomics/postgwas/_mr.py`); normalize it to the same short
#: label used everywhere else in this API (`method="presso"` request, CLI
#: flag, R wrapper) so the results table is self-consistent.
_MR_RESULT_LABEL = {"mr_presso": "presso"}


@tool(
    name="tg_mr",
    title="Two-sample Mendelian randomization",
    description=(
        "Two-sample Mendelian randomization from exposure + outcome GWAS "
        "summary statistics (IVW / Egger / weighted median / MR-PRESSO). "
        "Instruments are the SNPs shared between the two sumstats files. "
        "Returns MRRun with a per-method results table."
    ),
    long_running=False,
    category="postgwas",
    tags=["mendelian-randomization", "mr", "causal-inference", "postgwas"],
)
def mr(
    exposure: str | Path,
    outcome: str | Path,
    *,
    method: Literal["ivw", "egger", "weighted_median", "presso", "all"] = "ivw",
    output: str | Path | None = None,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
    sep: str = "\t",
) -> MRRun:
    """Two-sample Mendelian randomization from exposure + outcome sumstats.

    Loads both sumstats files, runs the requested estimator(s), and returns a
    :class:`~torchgenomics.api.MRRun` with a per-method results table.

    Parameters
    ----------
    exposure, outcome
        Paths to instrument sumstats TSVs (columns chr/pos/snp/a1/a2/beta/se/p,
        harmonized to the same effect allele) — the shared instruments define
        the MR panel.
    method
        One of ``"ivw"``, ``"egger"``, ``"weighted_median"``, ``"presso"``, or
        ``"all"`` (runs the full sensitivity panel, one row per method).
    output
        Optional path to write the results TSV.
    n_boot, n_perm, seed
        Bootstrap / permutation / seed controls for weighted-median and
        MR-PRESSO.
    sep
        Field delimiter for the sumstats files (default tab).

    Returns
    -------
    MRRun
    """
    from ..postgwas import (
        load_sumstats,
        mr_all,
        mr_egger,
        mr_ivw,
        mr_presso,
        mr_weighted_median,
    )

    m = str(method).lower()
    if m not in _MR_METHODS:
        raise ValueError(
            f"Unknown MR method '{method}'. Valid: {', '.join(_MR_METHODS)}."
        )

    exp = load_sumstats(str(exposure), sep=sep)
    out = load_sumstats(str(outcome), sep=sep)

    with timed() as elapsed:
        if m == "all":
            results = mr_all(exp, out, n_boot=n_boot, n_perm=n_perm, seed=seed)
        elif m == "ivw":
            results = [mr_ivw(exp, out)]
        elif m == "egger":
            results = [mr_egger(exp, out)]
        elif m == "weighted_median":
            results = [mr_weighted_median(exp, out, n_boot=n_boot, seed=seed)]
        else:  # presso
            results = [mr_presso(exp, out, n_perm=n_perm, seed=seed)]

        rows = [
            {
                "method": _MR_RESULT_LABEL.get(r.method, r.method),
                "beta": r.beta_hat,
                "se": r.se,
                "pval": r.p_value,
                "n_instruments": r.n_instruments,
                "egger_intercept": r.intercept,
                "egger_intercept_p": r.intercept_p,
                "n_outliers": r.n_outliers,
                "beta_corrected": r.beta_corrected,
                "pval_corrected": r.p_corrected,
            }
            for r in results
        ]
        df = pd.DataFrame(rows)

        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))

        primary = results[0]
        return MRRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=m,
            n_instruments=int(primary.n_instruments),
            primary_beta=float(primary.beta_hat),
            primary_p=float(primary.p_value),
            results=df,
        )
