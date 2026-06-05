"""Data-management tier-1 API: :func:`validate`, :func:`convert`, :func:`impute`."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, timed
from ._results import ConvertRun, ImputeRun, ValidateRun


@tool(
    name="tg_validate",
    title="Validate a TorchGenomics dataset",
    description=(
        "Run pre-flight validation of a genotype + phenotype dataset before any GWAS scan. "
        "Detects format, checks sample alignment across files, flags duplicate IDs, "
        "computes missingness, and reports warnings and errors. Returns a structured "
        "ValidateRun with `.ok`, per-file counts, and optional JSON report file."
    ),
    long_running=False,
    category="data",
    tags=["validation", "preflight", "qc"],
)
def validate(
    genotype: str | Path,
    phenotype: str | Path,
    *,
    covariate: str | Path | None = None,
    ploidy: int = 2,
    output: str | Path | None = None,
) -> ValidateRun:
    """Pre-flight validation of a TorchGenomics dataset.

    Parameters
    ----------
    genotype, phenotype
        Paths to the input files (any supported format; auto-detected).
    covariate
        Optional covariate TSV.
    ploidy
        Expected ploidy (used for allele-coding checks). Default 2.
    output
        Optional path to write a JSON report. If omitted, no file is written
        but the return value carries all fields.

    Returns
    -------
    ValidateRun
        :attr:`ValidateRun.ok` is True iff no errors were found. Inspect
        ``.warnings`` and ``.errors`` for details.
    """
    from ..io.validate import run_preflight

    output_files: dict[str, Path] = {}
    with timed() as elapsed:
        report = run_preflight(
            genotype_path=str(genotype),
            phenotype_path=str(phenotype),
            covariate_path=str(covariate) if covariate else None,
            ploidy=ploidy,
        )
        if output is not None:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as fh:
                json.dump(asdict(report), fh, indent=2)
            output_files["json"] = output_path

    return ValidateRun(
        runtime_s=elapsed(),
        output_files=output_files,
        ok=len(report.errors) == 0,
        format_name=report.format_name,
        n_samples_genotype=report.n_samples_genotype,
        n_samples_phenotype=report.n_samples_phenotype,
        n_samples_aligned=report.n_samples_aligned,
        n_variants_total=report.n_variants_total,
        n_traits=report.n_traits,
        ploidy=report.ploidy,
        genotype_missingness_pct=report.genotype_missingness_pct,
        phenotype_missingness_pct=report.phenotype_missingness_pct,
        warnings=list(report.warnings),
        errors=list(report.errors),
    )


@tool(
    name="tg_convert",
    title="Convert genotype file format",
    description=(
        "Convert a genotype file between supported formats (PLINK BED, Zarr, VCF). "
        "Inputs are auto-detected by extension/magic. Returns a ConvertRun with "
        "input/output formats and sample/variant counts."
    ),
    long_running=False,
    category="data",
    tags=["format", "conversion", "bed", "vcf", "zarr"],
)
def convert(
    input: str | Path,
    output: str | Path,
    *,
    output_format: Literal["bed", "zarr", "vcf"],
    sample: str | Path | None = None,
    map: str | Path | None = None,
) -> ConvertRun:
    """Convert a genotype file to ``output_format``.

    Parameters
    ----------
    input
        Source file (any supported format; auto-detected).
    output
        Destination path.
    output_format
        Target format: ``"bed"``, ``"zarr"``, or ``"vcf"``.
    sample
        Optional sample metadata file (e.g. for VCF→BED workflows).
    map
        Optional map file with variant positions.
    """
    from ..io.convert import convert as _convert
    from ..io.detect import detect_format

    with timed() as elapsed:
        input_fmt = detect_format(str(input))
        _convert(
            input_path=str(input),
            output_path=str(output),
            output_format=output_format,
            sample_path=str(sample) if sample else None,
            map_path=str(map) if map else None,
        )
        # Best-effort: re-open the produced file to report sizes
        n_samples = 0
        n_variants = 0
        try:
            from ..io.validate import _open_reader

            reader = _open_reader(str(output), output_format)
            n_samples = reader.n_samples
            n_variants = reader.n_variants
        except Exception:  # noqa: BLE001
            pass

    return ConvertRun(
        runtime_s=elapsed(),
        output_files={output_format: Path(output).resolve()},
        input_format=input_fmt,
        output_format=output_format,
        n_samples=n_samples,
        n_variants=n_variants,
    )


@tool(
    name="tg_impute",
    title="Impute missing genotypes",
    description=(
        "Impute missing genotypes using one of: mean, mode, knn, ld (built-in "
        "streaming methods); li-stephens or deep-learning (GPU-accelerated); "
        "or external tools (beagle, impute5, minimac4). Streaming methods bound "
        "peak memory by chunk_size × n × 8B. Returns ImputeRun with the saved "
        "output path and imputation quality (when computable)."
    ),
    long_running=True,
    category="data",
    tags=["imputation", "missing", "beagle", "impute5", "minimac4"],
)
def impute(
    genotype: str | Path,
    output: str | Path,
    *,
    method: Literal["mean", "mode", "knn", "ld", "li-stephens", "deep-learning"] = "mean",
    ploidy: int = 2,
    chunk_size: int = 1024,
    window_size: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> ImputeRun:
    """Impute missing genotypes from ``genotype`` and write the result to ``output``.

    Six built-in methods (no external tooling needed):

      - ``mean`` / ``mode``: two-pass streaming, per-column statistics.
      - ``knn`` : streaming GRM then per-chunk KNN fill (K held at n×n×8B).
      - ``ld``: sliding-window flanking-SNP regression (window_size × n × 8B).
      - ``li-stephens``: GPU-accelerated HMM. Requires full G in memory.
      - ``deep-learning``: GPU autoencoder. Requires full G in memory.

    External tools (BEAGLE / IMPUTE5 / Minimac4 / STITCH) are wired in the CLI
    but not in the tier-1 facade — use the CLI directly for those.

    Notes
    -----
    Output extension determines sink:
      - ``.zarr`` → streaming chunked write (peak memory bounded by chunk_size)
      - any other → torch.save of the materialized tensor (full G in RAM)
    """
    import torch

    emit_progress(progress_callback, 0.0, f"imputing via {method}")

    with timed() as elapsed:
        if method in ("li-stephens", "deep-learning"):
            from ..preprocess.impute_gpu import impute_deep_learning, impute_li_stephens
            from .scans import _load_genotype_matrix  # re-use loader

            G = _load_genotype_matrix(str(genotype))
            emit_progress(progress_callback, 0.3, f"loaded {G.shape[0]}×{G.shape[1]}")

            if method == "li-stephens":
                G_imp, r2 = impute_li_stephens(G, ploidy=ploidy)
            else:
                G_imp, r2 = impute_deep_learning(G, ploidy=ploidy)

            torch.save({"dosage": G_imp, "dosage_r2": r2}, str(output))
            emit_progress(progress_callback, 1.0, "saved")

            return ImputeRun(
                runtime_s=elapsed(),
                output_files={"imputed": Path(output).resolve()},
                method=method,
                n_samples=int(G_imp.shape[0]),
                n_variants=int(G_imp.shape[1]),
                n_imputed=int((G != G).sum().item()),  # NaNs in input
                imputation_quality=float(r2.mean().item()) if r2 is not None else None,
            )

        # Built-in streaming methods
        from ..io.detect import detect_format
        from ..io.validate import _open_reader
        from ..preprocess.impute import (
            compute_column_means_streaming,
            compute_column_modes_streaming,
            impute_chunk_with_knn,
            impute_chunk_with_ld_window,
            impute_chunk_with_means,
            impute_chunk_with_modes,
        )

        fmt = detect_format(str(genotype))
        reader = _open_reader(str(genotype), fmt)
        n_samples = reader.n_samples
        n_variants = reader.n_variants
        emit_progress(progress_callback, 0.05, f"streaming {n_samples}×{n_variants}")

        # Output sink (Zarr → streaming, else materialized)
        from ..cli import _open_impute_output_sink  # re-use existing sink factory

        sink, is_zarr = _open_impute_output_sink(str(output), n_samples, n_variants)
        col_offset = 0
        n_imputed = 0

        if method == "mean":
            col_means = compute_column_means_streaming(reader.iter_chunks(chunk_size))
            emit_progress(progress_callback, 0.5, "imputing")
            for G_chunk, _ in reader.iter_chunks(chunk_size):
                n_imputed += int((G_chunk != G_chunk).sum().item())
                G_imp = impute_chunk_with_means(G_chunk, col_means, col_offset)
                sink.write_chunk(G_imp, col_offset)
                col_offset += G_chunk.shape[1]

        elif method == "mode":
            col_modes = compute_column_modes_streaming(
                reader.iter_chunks(chunk_size), max_dosage=ploidy
            )
            emit_progress(progress_callback, 0.5, "imputing")
            for G_chunk, _ in reader.iter_chunks(chunk_size):
                n_imputed += int((G_chunk != G_chunk).sum().item())
                G_imp = impute_chunk_with_modes(G_chunk, col_modes, col_offset)
                sink.write_chunk(G_imp, col_offset)
                col_offset += G_chunk.shape[1]

        elif method == "knn":
            from ..cli import _impute_chunk_iter
            from ..linalg.kinship import grm_vanraden_streaming

            K, _ = grm_vanraden_streaming(
                _impute_chunk_iter(reader.iter_chunks(chunk_size)),
                n_samples=n_samples,
                ploidy=ploidy,
            )
            emit_progress(progress_callback, 0.5, "imputing")
            for G_chunk, _ in reader.iter_chunks(chunk_size):
                n_imputed += int((G_chunk != G_chunk).sum().item())
                G_imp = impute_chunk_with_knn(G_chunk, K)
                sink.write_chunk(G_imp, col_offset)
                col_offset += G_chunk.shape[1]

        else:  # ld
            chunks = list(reader.iter_chunks(chunk_size))
            G_all = torch.cat([c[0] for c in chunks], dim=1)
            n_imputed = int((G_all != G_all).sum().item())
            G_imp_all = impute_chunk_with_ld_window(G_all, window_size=window_size)
            sink.write_chunk(G_imp_all, 0)

        # Finalize sink
        if is_zarr:
            sink.close()
        else:
            G_imp = sink.materialize()
            torch.save({"dosage": G_imp}, str(output))

        emit_progress(progress_callback, 1.0, "done")

        return ImputeRun(
            runtime_s=elapsed(),
            output_files={"imputed": Path(output).resolve()},
            method=method,
            n_samples=n_samples,
            n_variants=n_variants,
            n_imputed=n_imputed,
        )
