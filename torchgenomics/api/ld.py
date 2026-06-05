"""LD tier-1 API: :func:`ld_blocks`."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ._decorator import tool
from ._helpers import ProgressCallback, emit_progress, resolve_output_dir, timed
from ._results import LDBlocksRun

_LDBlockMethod = Literal[
    "gabriel", "four_gamete", "spine", "r2", "gwas_aligned", "uncertainty",
    "cross_pop", "graphical", "changepoint", "big_ld", "cc_graph",
    "dp_optimize", "wall_pritchard",
]


@tool(
    name="tg_ld_blocks",
    title="Detect haplotype blocks",
    description=(
        "Detect LD haplotype blocks via one of 13 methods (gabriel, four_gamete, "
        "spine, r2, gwas_aligned, uncertainty, cross_pop, graphical, changepoint, "
        "big_ld, cc_graph, dp_optimize, wall_pritchard). Streams per-chromosome; "
        "free the slice before reading the next chromosome. Writes BED + .blocks.det + "
        "summary files; returns LDBlocksRun with .blocks DataFrame inline."
    ),
    long_running=True,
    category="ld",
    tags=["ld", "haplotype", "blocks", "gabriel", "plink"],
)
def ld_blocks(
    genotype: str | Path,
    *,
    output: str | Path | None = None,
    method: _LDBlockMethod = "gabriel",
    max_kb: int = 200,
    ci_low: float = 0.7,
    ci_high: float = 0.98,
    freq_threshold: float = 0.01,
    dprime_threshold: float = 0.7,
    r2_threshold: float = 0.5,
    condition_penalty: float = 0.0,
    max_block_snps: int = 1_000,
    l1_penalty: float = 0.1,
    window_size: int = 100,
    cp_penalty: float = 0.0,
    ld_window: int = 100,
    include_singletons: bool = False,
    objective: str = "ldscore",
    phased_vcf: str | Path | None = None,
    device: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> LDBlocksRun:
    """Detect haplotype blocks using one of 13 methods.

    Smart-defaults map to the existing CLI's per-method kwargs. Pass only
    the method-specific knobs you care about; the rest stay at proven defaults.

    Returns
    -------
    LDBlocksRun
        With ``.blocks`` (DataFrame of chr/start_bp/end_bp/num_snps/snp_ids),
        and paths to the produced BED + .blocks.det + summary files.
    """
    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_ld_blocks")
    output_prefix = output_dir / "blocks"

    emit_progress(progress_callback, 0.0, f"detecting LD blocks via {method}")

    with timed() as elapsed:
        from ..cli import _run_ld_blocks

        info = _run_ld_blocks(
            genotype_path=str(genotype),
            output_prefix=str(output_prefix),
            method=method,
            max_kb=max_kb,
            device=device,
            phased_vcf=str(phased_vcf) if phased_vcf else None,
            ci_low=ci_low,
            ci_high=ci_high,
            freq_threshold=freq_threshold,
            dprime_threshold=dprime_threshold,
            r2_threshold=r2_threshold,
            condition_penalty=condition_penalty,
            max_block_snps=max_block_snps,
            l1_penalty=l1_penalty,
            window_size=window_size,
            cp_penalty=cp_penalty,
            ld_window=ld_window,
            include_singletons=include_singletons,
            objective=objective,
        )

        output_files: dict[str, Path] = {k: Path(v) for k, v in info["output_files"].items()}
        det_path = output_files.get("det")
        bed_path = output_files.get("bed")
        summary_path = output_files.get("summary")

        # Parse the .blocks.det (PLINK-format text table)
        if det_path.exists():
            blocks_df = pd.read_csv(det_path, sep=r"\s+")
        else:
            blocks_df = pd.DataFrame()

        n_blocks = int(len(blocks_df))
        n_variants_in_blocks = int(blocks_df["NSNPS"].sum()) if "NSNPS" in blocks_df.columns else 0
        median_size_bp = float(blocks_df["KB"].median() * 1000) if "KB" in blocks_df.columns and not blocks_df.empty else 0.0
        median_n_snps = float(blocks_df["NSNPS"].median()) if "NSNPS" in blocks_df.columns and not blocks_df.empty else 0.0

        emit_progress(progress_callback, 1.0, "done")

        return LDBlocksRun(
            runtime_s=elapsed(),
            output_files=output_files,
            method=method,
            n_blocks=n_blocks,
            n_variants_in_blocks=n_variants_in_blocks,
            median_block_size_bp=median_size_bp,
            median_block_n_snps=median_n_snps,
            blocks=blocks_df,
        )
