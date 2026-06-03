"""Typed result objects returned by the high-level :mod:`torchgenomics.api`.

Each tier-1 function returns one of these classes. They have a uniform
shape so that:

  - Novice users get a one-liner: ``result.summary()`` prints a pretty report.
  - Notebook users get inline DataFrames / matplotlib plots.
  - MCP / LLM-tool callers get ``result.to_dict()`` — JSON-safe (no torch
    Tensors, no numpy arrays, no Path objects), small payload (DataFrames
    serialized as ``list[dict]``, truncated to top-K rows).

All result classes are :class:`dataclasses.dataclass` (not Pydantic) so the
api module stays dependency-light. The MCP server (:mod:`torchgenomics.mcp`)
converts these to JSON via :meth:`to_dict` at the tool boundary.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd


def _serialize_value(v: Any) -> Any:
    """Convert a Python value to a JSON-safe form."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, pd.DataFrame):
        return v.to_dict(orient="records")
    if isinstance(v, pd.Series):
        return v.to_list()
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if hasattr(v, "tolist"):  # numpy / torch tensors
        try:
            return v.tolist()
        except Exception:
            return repr(v)
    return v


@dataclass
class _BaseRun:
    """Shared base for all api result objects."""

    runtime_s: float = 0.0
    output_files: dict[str, Path] = field(default_factory=dict)
    log_excerpt: list[str] = field(default_factory=list)

    #: Human-readable name shown in :meth:`summary` (overridden per subclass).
    _kind: ClassVar[str] = "run"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this result.

        DataFrames are serialized as ``list[dict]``, Paths as strings, and
        torch / numpy arrays via ``.tolist()``. Suitable for direct return
        from an MCP tool or for ``json.dumps()`` without further conversion.
        """
        from dataclasses import asdict

        return _serialize_value(asdict(self))

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:  # overridden per subclass
        return f"{self._kind} completed in {self.runtime_s:.1f}s"

    def __str__(self) -> str:
        return self.summary()


# --- Data tier --------------------------------------------------------------


@dataclass
class ValidateRun(_BaseRun):
    """Result of :func:`torchgenomics.api.validate`."""

    ok: bool = False
    format_name: str = ""
    n_samples_genotype: int = 0
    n_samples_phenotype: int = 0
    n_samples_aligned: int = 0
    n_variants_total: int = 0
    n_traits: int = 0
    ploidy: int = 2
    genotype_missingness_pct: float = 0.0
    phenotype_missingness_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    _kind: ClassVar[str] = "validate"

    def summary(self) -> str:
        lines = [
            f"Format: {self.format_name}",
            f"Samples: {self.n_samples_genotype} genotype, {self.n_samples_phenotype} phenotype, {self.n_samples_aligned} aligned",
            f"Variants: {self.n_variants_total}",
            f"Traits: {self.n_traits}",
            f"Missingness: {self.genotype_missingness_pct:.1f}% genotype, {self.phenotype_missingness_pct:.1f}% phenotype",
        ]
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        lines.append(f"OK: {self.ok}  Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class ConvertRun(_BaseRun):
    """Result of :func:`torchgenomics.api.convert`."""

    input_format: str = ""
    output_format: str = ""
    n_samples: int = 0
    n_variants: int = 0

    _kind: ClassVar[str] = "convert"

    def summary(self) -> str:
        return (
            f"Converted {self.n_samples} samples × {self.n_variants} variants "
            f"from {self.input_format} → {self.output_format} in {self.runtime_s:.1f}s"
        )


@dataclass
class ImputeRun(_BaseRun):
    """Result of :func:`torchgenomics.api.impute`."""

    method: str = ""
    n_samples: int = 0
    n_variants: int = 0
    n_imputed: int = 0
    imputation_quality: float | None = None

    _kind: ClassVar[str] = "impute"

    def summary(self) -> str:
        quality = f", quality={self.imputation_quality:.3f}" if self.imputation_quality is not None else ""
        return (
            f"Imputed {self.n_imputed} missing genotypes via {self.method} "
            f"({self.n_samples} samples × {self.n_variants} variants{quality}) "
            f"in {self.runtime_s:.1f}s"
        )


# --- Scan tier --------------------------------------------------------------


@dataclass
class ScanRun(_BaseRun):
    """Result of :func:`torchgenomics.api.lmm_scan` / :func:`glm_scan`.

    Contains the headline summary (n_variants, n_significant, λ_GC), the
    top-K hits inline as a DataFrame, and paths to the full result tables.
    Plotting helpers (:meth:`manhattan`, :meth:`qq`) read from disk so the
    in-memory payload stays small.
    """

    model: str = ""  # "SingleTraitLMM", "GLM-binary", etc.
    test: str = ""  # "wald", "score", "lrt"
    correction: str | None = None
    n_variants: int = 0
    n_significant: int = 0
    significance_threshold: float = 5e-8
    lambda_gc: float | None = None
    sigma2_g: float | None = None
    sigma2_e: float | None = None
    h2: float | None = None
    n_samples: int = 0
    #: Top-K rows by p-value (smallest first).
    top_hits: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "scan"

    def summary(self) -> str:
        lines = [
            f"Model: {self.model}  Test: {self.test}  Correction: {self.correction or 'none'}",
            f"Samples: {self.n_samples}  Variants tested: {self.n_variants}",
            f"Significant @ α={self.significance_threshold:.2e}: {self.n_significant}",
        ]
        if self.lambda_gc is not None:
            lines.append(f"λ_GC: {self.lambda_gc:.4f}")
        if self.h2 is not None:
            lines.append(f"Variance components: σ²_g={self.sigma2_g:.4f}, σ²_e={self.sigma2_e:.4f}, h²={self.h2:.4f}")
        if not self.top_hits.empty:
            lines.append(f"Top hits ({len(self.top_hits)} shown):")
            lines.append(self.top_hits.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)

    def manhattan(self, **kwargs):
        """Return a matplotlib Figure with a Manhattan plot of this scan.

        Reads full results from :attr:`output_files` ``["tsv"]`` if present;
        otherwise plots only the :attr:`top_hits` (limited view).
        """
        try:
            from ..viz import manhattan_plot
        except ImportError as e:
            raise ImportError("Install matplotlib to use plotting helpers.") from e

        df = self._full_results_or_top_hits()
        if df is None or df.empty:
            warnings.warn("No results to plot.", stacklevel=2)
            return None
        return manhattan_plot(
            chrom=df["CHR"].astype(str).tolist(),
            pos=df["POS"].astype(int).tolist(),
            p=df["P"].astype(float).tolist(),
            **kwargs,
        )

    def qq(self, **kwargs):
        """Return a matplotlib Figure with a Q-Q plot of this scan."""
        try:
            from ..viz import qq_plot
        except ImportError as e:
            raise ImportError("Install matplotlib to use plotting helpers.") from e

        df = self._full_results_or_top_hits()
        if df is None or df.empty:
            warnings.warn("No results to plot.", stacklevel=2)
            return None
        return qq_plot(p=df["P"].astype(float).tolist(), **kwargs)

    def _full_results_or_top_hits(self) -> pd.DataFrame | None:
        path = self.output_files.get("tsv") or self.output_files.get("parquet")
        if path is not None:
            p = Path(path)
            if p.exists():
                if p.suffix == ".parquet":
                    return pd.read_parquet(p)
                return pd.read_csv(p, sep="\t")
        return self.top_hits if not self.top_hits.empty else None


# --- LD / post-GWAS tier ----------------------------------------------------


@dataclass
class LDBlocksRun(_BaseRun):
    """Result of :func:`torchgenomics.api.ld_blocks`."""

    method: str = ""
    n_blocks: int = 0
    n_variants_in_blocks: int = 0
    median_block_size_bp: float = 0.0
    median_block_n_snps: float = 0.0
    blocks: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "ld_blocks"

    def summary(self) -> str:
        return (
            f"Detected {self.n_blocks} LD blocks via {self.method} "
            f"({self.n_variants_in_blocks} variants; median size "
            f"{self.median_block_size_bp/1000:.1f} kb / {self.median_block_n_snps:.0f} SNPs) "
            f"in {self.runtime_s:.1f}s"
        )


@dataclass
class ClumpRun(_BaseRun):
    """Result of :func:`torchgenomics.api.clump`."""

    n_input_variants: int = 0
    n_clumps: int = 0
    n_index_variants: int = 0
    p_threshold: float = 5e-8
    r2_threshold: float = 0.1
    clumps: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "clump"

    def summary(self) -> str:
        return (
            f"Clumped {self.n_input_variants} variants → {self.n_clumps} independent loci "
            f"(p<{self.p_threshold:.2e}, r²<{self.r2_threshold}) "
            f"in {self.runtime_s:.1f}s"
        )


@dataclass
class MetaRun(_BaseRun):
    """Result of :func:`torchgenomics.api.meta`."""

    method: str = ""  # "ivw" | "dl" | "stouffer" | "re2"
    n_studies: int = 0
    n_variants: int = 0
    n_significant: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    heterogeneity_i2_median: float | None = None

    _kind: ClassVar[str] = "meta"

    def summary(self) -> str:
        het = f", median I²={self.heterogeneity_i2_median:.2f}" if self.heterogeneity_i2_median is not None else ""
        return (
            f"Meta-analysis ({self.method}) across {self.n_studies} studies, "
            f"{self.n_variants} variants, {self.n_significant} significant{het} "
            f"in {self.runtime_s:.1f}s"
        )


# --- PGS tier ---------------------------------------------------------------


@dataclass
class PgsFitRun(_BaseRun):
    """Result of :func:`torchgenomics.api.pgs_fit`."""

    method: str = ""  # "ct" | "ldpred2-inf" | "ldpred2-grid" | "ldpred2-auto" | "prscs"
    n_variants_input: int = 0
    n_variants_used: int = 0
    n_variants_with_weights: int = 0
    h2: float | None = None
    converged: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)

    _kind: ClassVar[str] = "pgs_fit"

    def summary(self) -> str:
        h2 = f", h²={self.h2:.4f}" if self.h2 is not None else ""
        conv = "" if self.converged else " (NOT converged)"
        return (
            f"PGS fit ({self.method}): {self.n_variants_with_weights} weighted variants "
            f"from {self.n_variants_input} input{h2}{conv} in {self.runtime_s:.1f}s"
        )


@dataclass
class PgsScoreRun(_BaseRun):
    """Result of :func:`torchgenomics.api.pgs_score`."""

    n_samples: int = 0
    n_variants_used: int = 0
    score_mean: float = 0.0
    score_sd: float = 0.0
    score_min: float = 0.0
    score_max: float = 0.0
    standardized: bool = False

    _kind: ClassVar[str] = "pgs_score"

    def summary(self) -> str:
        std = " (standardized)" if self.standardized else ""
        return (
            f"PGS scored: {self.n_samples} individuals, {self.n_variants_used} variants{std}. "
            f"Mean={self.score_mean:.4f}, SD={self.score_sd:.4f}, "
            f"range=[{self.score_min:.4f}, {self.score_max:.4f}]. "
            f"Runtime: {self.runtime_s:.1f}s"
        )


# --- Annotation + utility ---------------------------------------------------


@dataclass
class AnnotateRun(_BaseRun):
    """Result of :func:`torchgenomics.api.annotate_hits`."""

    n_input_hits: int = 0
    n_genes: int = 0
    crop: str | None = None
    assembly: str | None = None
    window_bp: int = 50_000
    hits: pd.DataFrame = field(default_factory=pd.DataFrame)
    genes: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "annotate"

    def summary(self) -> str:
        target = self.crop or self.assembly or "(unknown)"
        return (
            f"Annotated {self.n_input_hits} hits → {self.n_genes} unique genes "
            f"({target}, ±{self.window_bp/1000:.0f} kb window) in {self.runtime_s:.1f}s"
        )


@dataclass
class PlotResult(_BaseRun):
    """Result of plot-producing api functions (:func:`manhattan`, :func:`qq`).

    Contains the figure (when not headless) and, for MCP callers, a base64
    PNG that travels well through JSON.
    """

    figure: Any = None  # matplotlib.figure.Figure; None when headless / not returned
    png_path: Path | None = None
    png_base64: str | None = None
    width_px: int = 800
    height_px: int = 600

    _kind: ClassVar[str] = "plot"

    def summary(self) -> str:
        return (
            f"Plot generated ({self.width_px}×{self.height_px}px"
            + (f", saved to {self.png_path}" if self.png_path else "")
            + (f", base64 ({len(self.png_base64)} chars)" if self.png_base64 else "")
            + f"). Runtime: {self.runtime_s:.1f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        # Drop the (unserializable) matplotlib Figure from MCP payloads.
        d = super().to_dict()
        d.pop("figure", None)
        return d
