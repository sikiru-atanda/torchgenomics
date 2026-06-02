"""Polygenic Score tier-1 API: :func:`pgs_fit`, :func:`pgs_score`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, resolve_output_dir, select_device, timed
from ._results import PgsFitRun, PgsScoreRun


@tool(
    name="tg_pgs_fit",
    title="Fit polygenic-score weights",
    description=(
        "Fit PGS weights from GWAS summary statistics + LD reference panel "
        "using one of five methods: C+T (clumping + thresholding), LDpred2-Inf "
        "(infinitesimal), LDpred2-Grid (grid over p × h²), LDpred2-Auto (MCMC), "
        "or PRS-CS (continuous-shrinkage prior). Returns PgsFitRun with the "
        "saved weights path and per-method diagnostics (h², convergence, ESS)."
    ),
    long_running=True,
    category="pgs",
    tags=["pgs", "polygenic-score", "ldpred2", "prs-cs", "ct"],
)
def pgs_fit(
    sumstats: str | Path,
    ld_ref: str | Path,
    output: str | Path,
    *,
    method: Literal["ct", "ldpred2-inf", "ldpred2-grid", "ldpred2-auto", "prscs"] = "ldpred2-auto",
    h2: float | None = None,
    p_causal: float | None = None,
    n_iter: int = 1000,
    n_burnin: int = 500,
    n_chains: int = 3,
    clump_p: float = 1.0,
    clump_r2: float = 0.1,
    clump_kb: int = 250,
    grid_p: str = "0.001,0.01,0.1",
    grid_h2: str = "0.3,0.5,0.7",
    grid_sparse: bool = False,
    phi: float | None = None,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    seed: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PgsFitRun:
    """Fit PGS weights with the chosen method.

    Parameters
    ----------
    sumstats, ld_ref
        Paths to the GWAS sumstats and LD reference panel.
    output
        Path to write the fitted weights (``.tsv`` written; result metadata
        in a sibling ``.json``).
    method
        Default ``"ldpred2-auto"`` (no h² / p tuning needed). For best speed,
        try ``"ct"``.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_resolved = select_device(device)

    emit_progress(progress_callback, 0.0, f"fitting {method}")

    with timed() as elapsed:
        from ..cli import _run_pgs_fit

        info = _run_pgs_fit(
            sumstats_path=str(sumstats),
            ld_ref_path=str(ld_ref),
            output=str(output_path),
            method=method,
            device=device_resolved,
            seed=seed,
            h2=h2,
            p_causal=p_causal,
            n_iter=n_iter,
            n_burnin=n_burnin,
            n_chains=n_chains,
            clump_p=clump_p,
            clump_r2=clump_r2,
            clump_kb=clump_kb,
            grid_p=grid_p,
            grid_h2=grid_h2,
            grid_sparse=grid_sparse,
            phi=phi,
        )

        # Read back the result (CLI handler writes <output>.tsv + meta)
        output_files: dict[str, Path] = {}
        tsv_path = output_path if output_path.suffix == ".tsv" else Path(f"{output_path}.tsv")
        if tsv_path.exists():
            output_files["weights"] = tsv_path

        # Try to load PGSResult metadata for richer reporting
        n_input = int(info["m"])
        n_used = int(info["m"])
        n_with_weights = int(info["m"])
        h2_est = info["h2"]
        converged = bool(info["converged"])
        diagnostics: dict = {}
        try:
            from ..pgs import PGSResult

            result = PGSResult.load(str(tsv_path))
            if result.diagnostics is not None:
                diagnostics = {k: float(v) if hasattr(v, "__float__") else v for k, v in result.diagnostics.items()}
        except Exception:  # noqa: BLE001
            pass

        emit_progress(progress_callback, 1.0, "done")

        return PgsFitRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=method,
            n_variants_input=n_input,
            n_variants_used=n_used,
            n_variants_with_weights=n_with_weights,
            h2=h2_est,
            converged=converged,
            diagnostics=diagnostics,
        )


@tool(
    name="tg_pgs_score",
    title="Apply PGS weights to target genotypes",
    description=(
        "Compute per-individual polygenic scores by applying fitted weights "
        "to target genotypes. Streams variant chunks; handles missing data "
        "via mean imputation; optionally standardizes scores. Returns "
        "PgsScoreRun with mean/SD/min/max of the score distribution and "
        "the per-individual TSV path."
    ),
    long_running=True,
    category="pgs",
    tags=["pgs", "scoring", "polygenic-score"],
)
def pgs_score(
    genotype: str | Path,
    weights: str | Path,
    output: str | Path,
    *,
    standardize: bool = False,
    handle_missing: Literal["mean", "zero", "drop"] = "mean",
    chunk_size: int = 10_000,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    progress_callback: ProgressCallback | None = None,
) -> PgsScoreRun:
    """Compute per-individual polygenic scores."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device_resolved = select_device(device)

    emit_progress(progress_callback, 0.0, "scoring individuals")

    with timed() as elapsed:
        from ..cli import _run_pgs_score

        info = _run_pgs_score(
            genotype_path=str(genotype),
            weights_path=str(weights),
            output=str(output_path),
            standardize=standardize,
            handle_missing=handle_missing,
            chunk_size=chunk_size,
            device=device_resolved,
        )

        output_files: dict[str, Path] = {}
        if output_path.exists():
            output_files["scores"] = output_path

        # Summarize the score distribution
        n_samples = 0
        score_mean = 0.0
        score_sd = 0.0
        score_min = 0.0
        score_max = 0.0
        try:
            import pandas as pd

            df = pd.read_csv(output_path, sep="\t")
            score_col = next((c for c in df.columns if c.lower() in ("score", "prs", "pgs")), None)
            if score_col is not None:
                s = df[score_col].astype(float)
                n_samples = int(len(s))
                score_mean = float(s.mean())
                score_sd = float(s.std())
                score_min = float(s.min())
                score_max = float(s.max())
        except Exception:  # noqa: BLE001
            pass

        # Read number of weighted variants
        n_used = 0
        try:
            from ..pgs import PGSResult
            res = PGSResult.load(str(weights))
            n_used = int(res.m)
        except Exception:  # noqa: BLE001
            pass

        emit_progress(progress_callback, 1.0, "done")

        return PgsScoreRun(
            runtime_s=elapsed(),
            output_files=output_files,
            n_samples=n_samples,
            n_variants_used=n_used,
            score_mean=score_mean,
            score_sd=score_sd,
            score_min=score_min,
            score_max=score_max,
            standardized=standardize,
        )
