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

    Also exported under the friendly-API alias :data:`GwasResult` (see
    bottom of this module) — the object returned by ``tg.gwas(...)``.
    Beyond the raw fields, three convenience surfaces are provided for the
    novice / notebook audience:

    - :attr:`hits` — a readable alias for :attr:`top_hits`.
    - :attr:`diagnostics` — a small JSON-safe dict for programmatic checks
      (e.g. ``if result.diagnostics["lambda_gc"] > 1.1: ...``).
    - :meth:`summary` — a plain-language, novice-oriented report (not just
      a field dump): explains what λ_GC means, calls out inflation, and
      suggests next steps.
    - :meth:`report` — writes :meth:`summary` plus best-effort Manhattan /
      Q-Q plots and the full results table to a directory, for sharing.
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
    #: Trait kind as classified by the friendly-API input layer, e.g.
    #: ``"continuous"``, ``"binary"``, ``"ordinal"``, ``"count"``. Empty
    #: string when unknown/unset (e.g. constructed directly by advanced
    #: callers who bypass ``tg.gwas``).
    trait_type: str = ""
    #: Name of the scanned phenotype column (e.g. ``"Y1"``, ``"EarHT"``), as
    #: selected by the friendly-API input layer. Empty string when
    #: unknown/unset (e.g. a low-level scan constructed directly by an
    #: advanced caller). Used only to label :meth:`summary`'s header.
    trait_name: str = ""
    #: Human-readable warnings accumulated during the run (e.g. sample
    #: mismatches, low MAF filtering out most variants). Populated by the
    #: input-validation layer; empty by default. Surfaced in
    #: :meth:`summary` and :attr:`diagnostics`.
    warnings: list[str] = field(default_factory=list)

    _kind: ClassVar[str] = "scan"

    @property
    def hits(self) -> pd.DataFrame:
        """Alias for :attr:`top_hits` — the top-K variants by p-value.

        Provided so novice callers can write ``result.hits`` without
        needing to know the underlying field name.
        """
        return self.top_hits

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Small JSON-safe dict of scan diagnostics for programmatic checks.

        Keys: ``lambda_gc``, ``n_significant``, ``n_variants``,
        ``n_samples``, ``model``, ``trait_type``, ``warnings``. Intended
        for quick conditionals (e.g. inflation gating) without parsing
        :meth:`summary`'s prose, and for MCP / LLM callers that want a
        compact status object rather than the full :meth:`to_dict`.
        """
        return {
            "lambda_gc": self.lambda_gc,
            "n_significant": self.n_significant,
            "n_variants": self.n_variants,
            "n_samples": self.n_samples,
            "model": self.model,
            "trait_type": self.trait_type,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Plain-language report of the scan, for novice / notebook use.

        Unlike a raw field dump, this explains what the numbers mean:
        λ_GC gets an inline calibration verdict (``≤1.05`` well-calibrated,
        ``>1.10`` inflated, in between borderline), the top hit is named
        rather than just counted, any accumulated :attr:`warnings` are
        surfaced, and a final ``Next:`` line suggests follow-up actions
        (inspect hits, re-run with more PCs, save a report).
        """
        name = self.trait_name or "trait"
        trait_label = f"{name} ({self.trait_type})" if self.trait_type else name
        lines = [
            f"GWAS scan summary — {trait_label}",
            f"Model: {self.model or '(unspecified)'}  Test: {self.test or '(unspecified)'}  "
            f"Correction: {self.correction or 'none'}",
            f"Samples: {self.n_samples}   Variants tested: {self.n_variants}",
        ]

        if self.lambda_gc is not None:
            if self.lambda_gc <= 1.05:
                calibration = "✓ well-calibrated"
            elif self.lambda_gc > 1.10:
                calibration = "⚠ inflated — consider more PCs"
            else:
                calibration = "borderline — monitor for inflation"
            lines.append(f"λ_GC: {self.lambda_gc:.2f} ({calibration})")
        else:
            lines.append("λ_GC: not computed")

        if self.h2 is not None:
            lines.append(
                f"Variance components: σ²_g={self.sigma2_g:.4f}, "
                f"σ²_e={self.sigma2_e:.4f}, h²={self.h2:.4f}"
            )

        lines.append(f"Significant @ α={self.significance_threshold:.2e}: {self.n_significant}")
        if not self.top_hits.empty:
            top_row = self.top_hits.iloc[0]
            locus_cols = [c for c in ("SNP", "CHR", "BP", "POS") if c in self.top_hits.columns]
            locus = " / ".join(str(top_row[c]) for c in locus_cols) if locus_cols else "(unnamed)"
            p_col = "P" if "P" in self.top_hits.columns else None
            p_str = f", p={top_row[p_col]:.2e}" if p_col is not None else ""
            lines.append(f"Top locus: {locus}{p_str}")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)

        lines.append(f"Runtime: {self.runtime_s:.1f}s")

        next_steps = []
        if self.n_significant > 0:
            next_steps.append("inspect .hits and .manhattan() for the top loci")
        else:
            next_steps.append(
                "no genome-wide-significant hits — check power or relax significance_threshold"
            )
        if self.lambda_gc is not None and self.lambda_gc > 1.10:
            next_steps.append("re-run with more PCs (n_pcs) to reduce inflation")
        next_steps.append("call .report(dir) to save the summary and plots")
        lines.append("Next: " + "; ".join(next_steps))

        return "\n".join(lines)

    def report(self, dir: str | Path) -> Path:
        """Write a self-contained report folder for this scan.

        Always writes ``summary.txt`` (the :meth:`summary` text). Also
        attempts, best-effort, to write ``manhattan.png`` / ``qq.png`` via
        :meth:`manhattan` / :meth:`qq`, and to copy the full results table
        to ``results.tsv`` if one is referenced in :attr:`output_files`.
        Each of these extras is wrapped in its own ``try/except`` so a
        headless environment (no display backend) or a :attr:`top_hits`
        schema that doesn't match the ``CHR``/``POS``/``P`` columns
        expected by the plotting helpers cannot make the report fail —
        only ``summary.txt`` is guaranteed.

        Parameters
        ----------
        dir : str | Path
            Directory to create (including parents) and populate.

        Returns
        -------
        Path
            The report directory (``dir``, resolved to a :class:`Path`).
        """
        out_dir = Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "summary.txt").write_text(self.summary())

        try:
            fig = self.manhattan()
            if fig is not None:
                fig.savefig(out_dir / "manhattan.png", dpi=150, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        try:
            fig = self.qq()
            if fig is not None:
                fig.savefig(out_dir / "qq.png", dpi=150, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        tsv_path = self.output_files.get("tsv")
        if tsv_path is not None:
            try:
                import shutil

                shutil.copy(str(tsv_path), out_dir / "results.tsv")
            except Exception:  # noqa: BLE001
                pass

        return out_dir

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


@dataclass
class MRRun(_BaseRun):
    """Result of :func:`torchgenomics.api.mr` — a Mendelian-randomization panel."""

    method: str = ""  # "ivw" | "egger" | "weighted_median" | "presso" | "all"
    n_instruments: int = 0
    primary_beta: float | None = None
    primary_p: float | None = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "mr"

    def summary(self) -> str:
        lines = [
            f"MR ({self.method}) — {self.n_instruments} instruments",
        ]
        if self.primary_beta is not None:
            lines.append(f"Causal estimate: beta={self.primary_beta:.4f} "
                         f"p={self.primary_p:.3e}")
        if not self.results.empty:
            lines.append(self.results.to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class ColocRun(_BaseRun):
    """Result of :func:`torchgenomics.api.coloc` — colocalization posteriors."""

    method: str = ""  # "pairwise" | "hyprcoloc"
    pp: dict = field(default_factory=dict)  # posterior probabilities
    candidate_snp: int | None = None
    headline: float | None = None  # PP.H4 (pairwise) or PP(all colocalize)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "coloc"

    def summary(self) -> str:
        if self.method == "pairwise":
            verdict = ("strong" if (self.headline or 0) >= 0.8 else
                       "moderate" if (self.headline or 0) >= 0.5 else "weak")
            lines = [
                f"Colocalization (pairwise) — PP.H4={self.headline:.3f} "
                f"({verdict} evidence of a shared causal variant)",
                "  " + "  ".join(f"PP.{k.upper()}={v:.3f}" for k, v in self.pp.items()),
            ]
        else:
            lines = [
                f"Colocalization (hyprcoloc) — PP(all colocalize)={self.headline:.3f}",
            ]
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class SMRRun(_BaseRun):
    """Result of :func:`torchgenomics.api.smr` — SMR + HEIDI across genes."""

    n_genes_tested: int = 0
    n_significant_smr: int = 0
    n_pass_heidi: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "smr"

    def summary(self) -> str:
        lines = [
            f"SMR — {self.n_genes_tested} genes tested, "
            f"{self.n_significant_smr} SMR-significant, {self.n_pass_heidi} pass HEIDI",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class MRMegaRun(_BaseRun):
    """Result of :func:`torchgenomics.api.mr_mega` — multi-ancestry MR meta-analysis."""

    method: str = ""
    n_axes: int = 0
    n_populations: int = 0
    n_snps: int = 0
    min_p_meta: float | None = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "mr_mega"

    def summary(self) -> str:
        mp = f"min p_meta={self.min_p_meta:.3e}" if self.min_p_meta is not None else ""
        lines = [
            f"MR-MEGA — {self.n_populations} populations, {self.n_axes} axes, "
            f"{self.n_snps} SNPs {mp}".rstrip(),
        ]
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class PowerRun(_BaseRun):
    """Result of :func:`torchgenomics.api.power` — per-variant GWAS detection power."""

    alpha: float = 5e-8
    n: float = 0.0
    target_power: float = 0.8
    n_variants: int = 0
    n_powered: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    curve: pd.DataFrame | None = None

    _kind: ClassVar[str] = "power"

    def summary(self) -> str:
        lines = [
            f"Power — {self.n_variants} variants, {self.n_powered} at power>={self.target_power} "
            f"(alpha={self.alpha:.1e}, N={self.n:g})",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class WinnersCurseRun(_BaseRun):
    """Result of :func:`torchgenomics.api.winners_curse` — effect-size de-biasing."""

    method: str = ""
    n_corrected: int = 0
    n_variants: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "winners_curse"

    def summary(self) -> str:
        lines = [
            f"Winner's-curse ({self.method}) — {self.n_corrected} of {self.n_variants} "
            f"variants corrected",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class EnrichmentRun(_BaseRun):
    """Result of :func:`torchgenomics.api.gene_set_enrichment` — MAGMA-style test."""

    n_genes_total: int = 0
    n_gene_sets: int = 0
    n_significant: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    genes: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "enrichment"

    def summary(self) -> str:
        lines = [
            f"Gene-set enrichment — {self.n_gene_sets} sets over {self.n_genes_total} genes, "
            f"{self.n_significant} significant (p<0.05)",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class HessRun(_BaseRun):
    """Result of :func:`torchgenomics.api.hess` — local heritability / genetic correlation."""

    mode: str = "h2"  # "h2" | "rg"
    h2_total: float = 0.0
    h2_total_se: float = 0.0
    n_regions: int = 0
    n_snps_total: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "hess"

    def summary(self) -> str:
        label = "local h²" if self.mode == "h2" else "local rg"
        lines = [
            f"HESS {label} — {self.n_regions} regions, {self.n_snps_total} SNPs, "
            f"total={self.h2_total:.4f} (se {self.h2_total_se:.4f})",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


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


# --- LGEBV (Local Genomic Estimated Breeding Values) ------------------------


@dataclass
class LGEBVResult(_BaseRun):
    """Result of :func:`torchgenomics.api.lgebv`.

    Holds per-haplo-block summed marker effects (the "local" GEBV) and
    a sign-based favorable/unfavorable classification.

    Attributes
    ----------
    block_id, chrom, start, end, n_variants
        Per-block bookkeeping (lists of length ``n_blocks``).
    lgebv : torch.Tensor
        ``(n_blocks,)`` sum of per-marker effects within each block.
        This is the breeding-value contribution from that block in the
        (centered, ploidy-aware standardized) space.
    block_variance : torch.Tensor
        ``(n_blocks,)`` ``Var(Z[:, block] @ u_hat[block])`` across
        genotypes, i.e. the empirical variance explained by the block.
    favorable : list[bool]
        Sign-based classification (see ``favorable_direction`` in the
        function signature).
    marker_effects : torch.Tensor | None
        ``(m,)`` per-marker BLUP effect ``u_hat``; populated only when
        ``return_marker_effects=True``.
    h2_used : float
        Heritability value used for the rrBLUP shrinkage. If the caller
        passed ``h2=None`` this is the REML-estimated value.
    method : str
        ``"rrblup"`` or ``"gblup"``.
    """

    block_id: list[str] = field(default_factory=list)
    chrom: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    lgebv: Any = None              # torch.Tensor (n_blocks,)
    block_variance: Any = None     # torch.Tensor (n_blocks,)
    favorable: list[bool] = field(default_factory=list)
    marker_effects: Any = None     # torch.Tensor (m,) | None
    h2_used: float = 0.0
    method: str = "rrblup"

    _kind: ClassVar[str] = "lgebv"

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a per-block table.

        Columns: ``block_id``, ``chrom``, ``start``, ``end``, ``n_variants``,
        ``lgebv``, ``block_variance``, ``favorable``.
        """
        lgebv_list = self.lgebv.detach().cpu().tolist() if self.lgebv is not None else []
        var_list = (
            self.block_variance.detach().cpu().tolist()
            if self.block_variance is not None
            else []
        )
        return pd.DataFrame(
            {
                "block_id": list(self.block_id),
                "chrom": list(self.chrom),
                "start": list(self.start),
                "end": list(self.end),
                "n_variants": list(self.n_variants),
                "lgebv": lgebv_list,
                "block_variance": var_list,
                "favorable": list(self.favorable),
            }
        )

    def top_blocks(
        self,
        k: int = 10,
        by: str = "variance",
    ) -> pd.DataFrame:
        """Return the top-``k`` blocks sorted by ``block_variance`` or ``|lgebv|``.

        Parameters
        ----------
        k : int
            Number of blocks to return.
        by : {"variance", "abs_lgebv"}
            Sort key.
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        if by == "variance":
            df = df.sort_values("block_variance", ascending=False)
        elif by == "abs_lgebv":
            df = df.reindex(df["lgebv"].abs().sort_values(ascending=False).index)
        else:
            raise ValueError(f"Unknown sort key '{by}'. Use 'variance' or 'abs_lgebv'.")
        return df.head(k).reset_index(drop=True)

    def summary(self) -> str:
        n_blocks = len(self.block_id)
        n_fav = sum(1 for b in self.favorable if b)
        lines = [
            f"LGEBV ({self.method}): {n_blocks} haplo-blocks scored",
            f"h² used: {self.h2_used:.4f}",
            f"Favorable blocks: {n_fav}/{n_blocks}",
        ]
        if n_blocks > 0 and self.lgebv is not None:
            l = self.lgebv.detach().cpu()
            v = (
                self.block_variance.detach().cpu()
                if self.block_variance is not None
                else None
            )
            lines.append(
                f"LGEBV range: [{float(l.min()):.4f}, {float(l.max()):.4f}], "
                f"mean={float(l.mean()):.4f}"
            )
            if v is not None and len(v) > 0:
                lines.append(
                    f"Block variance: max={float(v.max()):.4f}, "
                    f"median={float(v.median()):.4f}"
                )
            head_df = self.top_blocks(k=min(5, n_blocks), by="variance")
            lines.append("Top blocks by variance:")
            lines.append(head_df.to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class CliRun(_BaseRun):
    """Result of running a tier-2 CLI subcommand via the api CLI-runner bridge.

    These commands (e.g. ``bayes-scan``, ``ldsc``, ``mediate``) are exposed for
    Python/R parity by invoking the real CLI subcommand programmatically. The
    command does its own file I/O; this object reports what ran, its exit code,
    and any output files it wrote — not a rich per-command typed result.
    """

    command: str = ""
    exit_code: int = 0
    args: dict = field(default_factory=dict)

    _kind: ClassVar[str] = "cli"

    def summary(self) -> str:
        lines = [f"CLI '{self.command}' — exit {self.exit_code}"]
        for label, path in self.output_files.items():
            lines.append(f"  {label}: {path}")
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


# --- Friendly-API alias ------------------------------------------------------

#: Result object returned by the friendly-API entry point ``tg.gwas(...)``.
#: Identical to :class:`ScanRun` (same fields, same ``.hits`` /
#: ``.diagnostics`` / ``.summary`` / ``.report`` surface) — the alias just
#: gives the novice-facing entry point a name that doesn't presuppose the
#: underlying scan model.
GwasResult = ScanRun
