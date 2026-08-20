"""Input loading, sample alignment, and trait-type detection for tg.gwas().

This module is the friendly front door for the ``tg.gwas(phenotype, genotype,
...)`` orchestrator (Task 5) and its shared dispatch core (Task 4). It accepts
whatever a notebook user is likely to already have in hand — a file path, a
pandas Series/DataFrame, or a numpy array — resolves it to a single aligned
trait plus a genotype source, and raises actionable ``ValueError`` messages
(never a raw ``KeyError``/torch traceback) whenever inputs don't line up.

Two genotype input shapes are supported:

  - **Path** (``str``/``pathlib.Path``): stored as-is. This module does *not*
    open the file — the heavier format-specific readers (PLINK/VCF/BGEN/...)
    and the real sample-id alignment against the phenotype are deferred to
    the scan/dispatch layer (Task 4), which already owns that machinery via
    :mod:`torchgenomics.io.detect` and :class:`torchgenomics.io.aligned.
    SampleAlignedReader`. ``n_samples`` in this case is simply the aligned
    phenotype length; the true genotype sample count is only known once the
    file is opened downstream.
  - **In-memory array** (``numpy.ndarray`` or ``pandas.DataFrame``): wrapped
    in :class:`ArrayReader`, a minimal adapter that satisfies the
    :class:`torchgenomics.io.base.GenotypeReader` protocol so it can be
    streamed through :class:`~torchgenomics.scan.UnifiedScanner` exactly like
    a file-backed reader. Sample ids are taken from the DataFrame's index
    (real id-based alignment against the phenotype) or, for a bare ndarray
    (which carries no id information), positionally from the phenotype's own
    index (the caller asserts the two are already row-aligned).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..models.base import VariantMeta

# "Trivially tiny" cutoff for the shared-sample-id intersection: below this
# a GWAS is not meaningful (no residual degrees of freedom to speak of), so
# we fail fast with an actionable message rather than let a downstream
# linear-algebra routine raise an opaque singular-matrix error.
_MIN_SHARED_SAMPLES = 3

# Trait-type detection thresholds. See `detect_trait_type` docstring for the
# rationale; these numbers are a practical heuristic (not a literature-cited
# statistical threshold) and intentionally documented as such.
_MAX_CATEGORICAL_LEVELS = 10

# The only trait types the dispatch layer (Task 4) knows how to route.
_VALID_TRAIT_TYPES = frozenset({"continuous", "binary", "categorical"})


@dataclass
class GwasInputs:
    """Resolved, sample-aligned inputs ready for :func:`_dispatch.run_model`.

    Produced by :func:`load_inputs`. This is a plain data bundle — it does
    no further validation of its own; all checks happen while it is built.

    Attributes
    ----------
    genotype_path_or_reader : str | ArrayReader | object
        Either a filesystem path (``str``) to a genotype file in any
        auto-detectable format (opened downstream by Task 4 via the
        existing format-specific readers), or an in-memory reader — an
        :class:`ArrayReader` wrapping a user-supplied ndarray/DataFrame, or
        (if the caller passed one directly) a
        :class:`~torchgenomics.io.aligned.SampleAlignedReader` wrapping an
        arbitrary :class:`~torchgenomics.io.base.GenotypeReader`.
    phenotype : pandas.Series
        The single selected trait, aligned and indexed by sample id.
    covariates : pandas.DataFrame | None
        Optional covariates aligned to the same sample order/index as
        ``phenotype``. ``None`` if no covariates were supplied.
    trait_type : str
        One of ``"continuous"``, ``"binary"``, ``"categorical"`` — either
        user-supplied or inferred by :func:`detect_trait_type`.
    n_samples : int
        Number of aligned samples (``== len(phenotype)``).
    trait_name : str
        Name of the selected trait column (``phenotype.name``, coerced to
        ``str``; falls back to ``"trait"`` if unnamed).
    phenotypes : pandas.DataFrame | None
        Multi-trait mode only (``traits=`` was passed to :func:`load_inputs`):
        all requested trait columns, sample-aligned, in the order given by
        ``traits``. ``None`` in single-trait mode.
    trait_names : list[str] | None
        Multi-trait mode only: the requested trait column names, in order
        (``== list(phenotypes.columns)``). ``None`` in single-trait mode.
    """

    genotype_path_or_reader: object
    phenotype: pd.Series
    covariates: pd.DataFrame | None
    trait_type: str
    n_samples: int
    trait_name: str
    phenotypes: pd.DataFrame | None = None
    trait_names: list[str] | None = None


class ArrayReader:
    """Minimal :class:`~torchgenomics.io.base.GenotypeReader` over an in-memory array.

    Wraps a dense ``(n_samples, n_variants)`` genotype matrix so it can be
    consumed by the same ``iter_chunks``-based scan machinery as file-backed
    readers, without ever writing it to disk. Used by :func:`load_inputs`
    when the caller passes ``genotype=`` as a numpy array or pandas
    DataFrame instead of a path.

    Parameters
    ----------
    G : numpy.ndarray, shape (n_samples, n_variants)
        Dense dosage matrix. Copied to ``float64`` on construction.
    sample_ids : list[str]
        Row labels, in the same order as ``G``'s rows.
    variant_meta : VariantMeta | None, default None
        Per-variant metadata. If omitted, synthesized as
        ``snp=["snp0", "snp1", ...]``, ``chr=["0", "0", ...]``,
        ``pos=[0, 1, 2, ...]``, ``a1=["A"]*m``, ``a2=["B"]*m`` — placeholder
        labels good enough for scanning and reporting row/column identity,
        but not real genomic coordinates.

    Raises
    ------
    ValueError
        If ``G`` is not 2-D, or if ``len(sample_ids) != G.shape[0]``.
    """

    def __init__(
        self,
        G: np.ndarray,
        sample_ids: list[str],
        variant_meta: VariantMeta | None = None,
    ) -> None:
        G = np.asarray(G, dtype=np.float64)
        if G.ndim != 2:
            raise ValueError(
                f"genotype array must be 2-D (n_samples, n_variants); got shape {G.shape}."
            )
        if len(sample_ids) != G.shape[0]:
            raise ValueError(
                f"genotype has {G.shape[0]} rows but {len(sample_ids)} sample ids were "
                "supplied — these must match 1:1."
            )
        self.G = G
        self._sample_ids = [str(s) for s in sample_ids]
        m = G.shape[1]
        self._variant_meta = variant_meta or VariantMeta(
            snp=[f"snp{i}" for i in range(m)],
            chr=["0"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["B"] * m,
        )

    @property
    def n_samples(self) -> int:
        """Number of samples (rows of ``G``)."""
        return self.G.shape[0]

    @property
    def n_variants(self) -> int:
        """Number of variants (columns of ``G``)."""
        return self.G.shape[1]

    @property
    def sample_ids(self) -> list[str]:
        """Ordered list of sample IDs (for alignment with phenotype)."""
        return self._sample_ids

    @property
    def variant_meta(self) -> VariantMeta:
        """Full :class:`VariantMeta` for all variants in ``G``."""
        return self._variant_meta

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[tuple[Tensor, VariantMeta]]:
        """Yield ``(G_chunk, variant_meta)`` as column slices of ``G``.

        Parameters
        ----------
        chunk_size : int, default 1024
            Number of variants (columns) per yielded chunk.

        Yields
        ------
        tuple[torch.Tensor, VariantMeta]
            ``G_chunk`` has shape ``(n_samples, m)`` and dtype ``float64``
            (FP64 is mandatory for statistical inference; see CLAUDE.md),
            matching every other reader's ``iter_chunks`` in this codebase;
            ``variant_meta`` carries the matching column slice of the
            reader's ``VariantMeta``.
        """
        m = self.G.shape[1]
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            G_chunk = torch.as_tensor(self.G[:, start:end], dtype=torch.float64)
            vmeta_chunk = VariantMeta(
                snp=self._variant_meta.snp[start:end],
                chr=self._variant_meta.chr[start:end],
                pos=self._variant_meta.pos[start:end],
                a1=self._variant_meta.a1[start:end],
                a2=self._variant_meta.a2[start:end],
            )
            yield G_chunk, vmeta_chunk


def detect_trait_type(y: pd.Series) -> str:
    """Infer whether a phenotype vector is continuous, binary, or categorical.

    Heuristic (deliberately simple — this is a UX convenience, not a
    statistical test):

      - **binary**: exactly 2 distinct non-missing values (e.g. case/control
        coded 0/1, or any two-level factor).
      - **categorical**: all non-missing values are integer-valued (no
        fractional part) *and* there are more than 2 but at most
        :data:`_MAX_CATEGORICAL_LEVELS` (10) distinct values — enough to
        rule out a mis-coded continuous trait with few observed levels.
      - **continuous**: everything else (non-integer values, or an integer
        trait with more than 10 distinct levels — e.g. a count trait or a
        Likert-like scale wide enough to model as continuous).

    Parameters
    ----------
    y : pandas.Series
        Phenotype values for one trait. Missing values (``NaN``) are
        excluded before counting distinct levels.

    Returns
    -------
    str
        One of ``"binary"``, ``"categorical"``, ``"continuous"``.

    Raises
    ------
    ValueError
        If fewer than 2 distinct non-missing values are present (a constant
        trait carries no information for a GWAS).

    Examples
    --------
    >>> detect_trait_type(pd.Series([0, 1, 1, 0, 1]))
    'binary'
    >>> detect_trait_type(pd.Series([1.2, 3.4, 0.1, 9.9, 2.2]))
    'continuous'
    >>> detect_trait_type(pd.Series([0, 1, 2, 1, 2, 0, 1]))
    'categorical'
    """
    vals = pd.Series(y).dropna()
    distinct = vals.unique()
    n_distinct = len(distinct)

    if n_distinct < 2:
        raise ValueError(
            f"Trait '{y.name if y.name is not None else ''}' has {n_distinct} distinct "
            "non-missing value(s) — a constant trait carries no information for a GWAS. "
            "Check that the correct trait column was selected."
        )

    if n_distinct == 2:
        return "binary"

    numeric_vals = pd.to_numeric(pd.Series(distinct), errors="coerce")
    is_integer_like = numeric_vals.notna().all() and np.allclose(
        numeric_vals.to_numpy(dtype=float), np.round(numeric_vals.to_numpy(dtype=float))
    )
    if is_integer_like and n_distinct <= _MAX_CATEGORICAL_LEVELS:
        return "categorical"

    return "continuous"


def _load_phenotype_table(path: Any) -> pd.DataFrame:
    """Load a phenotype/covariate table from disk and index it by sample id.

    Thin wrapper around the existing tabular loader + sample-id-column
    detection in :mod:`torchgenomics.io.phenotype`
    (:func:`_load_tabular` / :func:`_detect_id_column`). Factored out so this
    load-then-index sequence lives in exactly one place; shared by
    :func:`_resolve_phenotype` (single-trait) and :func:`_resolve_phenotypes`
    (multi-trait).
    """
    from ..io.phenotype import _detect_id_column, _load_tabular

    df = _load_tabular(path)
    id_col = _detect_id_column(df)
    return df.set_index(id_col)


def _resolve_phenotype(
    phenotype: Any, trait: str | None
) -> pd.Series:
    """Resolve the ``phenotype`` argument of :func:`load_inputs` to a Series.

    Accepts a file path (delegates to the existing tabular loader + sample-id
    detection in :mod:`torchgenomics.io.phenotype`), a ``pandas.DataFrame``
    (picks the ``trait`` column, or the first column if unspecified), a
    ``pandas.Series`` (used directly), or a bare 1-D ``numpy.ndarray`` (no
    sample ids of its own — positional alignment happens downstream).
    """
    if isinstance(phenotype, (str, Path)):
        from ..io.phenotype import _is_numeric_column

        df = _load_phenotype_table(phenotype)

        if trait is not None:
            if trait not in df.columns:
                raise ValueError(
                    f"trait='{trait}' not found in phenotype file '{phenotype}'. "
                    f"Available columns: {list(df.columns)}."
                )
            col = trait
        else:
            col = next((c for c in df.columns if _is_numeric_column(df[c])), None)
            if col is None:
                raise ValueError(
                    f"Could not find a numeric trait column in phenotype file "
                    f"'{phenotype}'. Available columns: {list(df.columns)}. "
                    "Pass trait=<column name> to select one explicitly."
                )
        series = df[col].copy()
        series.name = col
        return series

    if isinstance(phenotype, pd.DataFrame):
        col = trait if trait is not None else phenotype.columns[0]
        if col not in phenotype.columns:
            raise ValueError(
                f"trait='{col}' not found in phenotype columns: {list(phenotype.columns)}."
            )
        series = phenotype[col].copy()
        series.name = col
        return series

    if isinstance(phenotype, pd.Series):
        series = phenotype.copy()
        if trait is not None and series.name is not None and str(series.name) != str(trait):
            raise ValueError(
                f"trait='{trait}' was requested but the supplied phenotype Series is "
                f"named '{series.name}'. Pass the correctly named Series, rename it, "
                "or omit trait= to use it as-is."
            )
        if series.name is None:
            series.name = trait if trait is not None else "trait"
        return series

    if isinstance(phenotype, np.ndarray):
        if phenotype.ndim != 1:
            raise ValueError(
                f"phenotype array must be 1-D (one value per sample); got shape "
                f"{phenotype.shape}. For multiple traits, call load_inputs once per trait."
            )
        return pd.Series(phenotype, name=trait if trait is not None else "trait")

    raise ValueError(
        f"Unsupported phenotype input type: {type(phenotype).__name__}. Pass a file "
        "path, a pandas Series/DataFrame, or a 1-D numpy array."
    )


def _resolve_phenotypes(phenotype: Any, traits: list[str]) -> pd.DataFrame:
    """Resolve a multi-trait ``phenotype`` to a DataFrame of the named columns.

    Multi-trait counterpart to :func:`_resolve_phenotype`, used by
    :func:`load_inputs` when ``traits=[...]`` is passed. ``phenotype`` must
    be a :class:`pandas.DataFrame` (indexed by sample id, or with an
    auto-detectable id column) or a path to a table containing every column
    in ``traits`` — loaded via the same :func:`_load_phenotype_table` helper
    :func:`_resolve_phenotype` uses for its path branch, so the tabular-load
    + sample-id-index logic is not duplicated. A bare ``pandas.Series`` (a
    single trait) is rejected with a friendly error pointing at multi-trait
    mode's DataFrame requirement. Columns are returned in the order given in
    ``traits`` and coerced to numeric (invalid entries become ``NaN``, same
    as :func:`_resolve_phenotype`'s numeric-column handling).

    Raises
    ------
    ValueError
        If ``phenotype`` is a bare Series, an unsupported type, or is
        missing one or more of the requested ``traits`` columns.
    """
    if isinstance(phenotype, pd.Series):
        raise ValueError(
            "multi-trait mode (traits=[...]) needs a phenotype DataFrame or a "
            "path/file with the named columns, not a single Series. Provide a "
            f"DataFrame with columns {traits}."
        )
    if isinstance(phenotype, (str, Path)):
        df = _load_phenotype_table(phenotype)
    elif isinstance(phenotype, pd.DataFrame):
        df = phenotype.copy()
    else:
        raise ValueError(
            f"Unsupported phenotype type for multi-trait mode: {type(phenotype).__name__}. "
            "Pass a DataFrame indexed by sample id or a path to a phenotype table."
        )

    missing = [t for t in traits if t not in df.columns]
    if missing:
        raise ValueError(
            f"traits {missing} not found in the phenotype columns "
            f"({list(df.columns)[:10]}...). Check the column names."
        )
    return df[list(traits)].apply(pd.to_numeric, errors="coerce")


def _resolve_covariates(covariates: Any, fallback_index: pd.Index) -> pd.DataFrame:
    """Resolve the ``covariates`` argument of :func:`load_inputs` to a DataFrame.

    Accepts a file path (delegates to the tabular loader + sample-id
    detection), a ``pandas.DataFrame`` (used directly), or a bare 2-D
    ``numpy.ndarray`` (no ids of its own — indexed positionally by
    ``fallback_index``, i.e. the phenotype's index).
    """
    if isinstance(covariates, (str, Path)):
        from ..io.phenotype import _detect_id_column, _load_tabular

        df = _load_tabular(covariates)
        id_col = _detect_id_column(df)
        return df.set_index(id_col)

    if isinstance(covariates, pd.DataFrame):
        return covariates.copy()

    if isinstance(covariates, np.ndarray):
        if covariates.ndim != 2:
            raise ValueError(
                f"covariates array must be 2-D (n_samples, n_covariates); got shape "
                f"{covariates.shape}."
            )
        if covariates.shape[0] != len(fallback_index):
            raise ValueError(
                f"covariates has {covariates.shape[0]} rows but phenotype has "
                f"{len(fallback_index)} samples — a bare covariates array (no sample "
                "ids) must align 1:1 with the phenotype. Pass a DataFrame with a "
                "sample-id index instead."
            )
        return pd.DataFrame(covariates, index=fallback_index)

    raise ValueError(
        f"Unsupported covariates input type: {type(covariates).__name__}. Pass a file "
        "path, a pandas DataFrame, or a 2-D numpy array."
    )


def _make_array_reader(genotype: Any, phenotype_index: pd.Index) -> ArrayReader:
    """Wrap an in-memory ndarray/DataFrame genotype matrix as an :class:`ArrayReader`.

    For a ``pandas.DataFrame``, its own row index becomes the reader's
    sample ids (real id-based alignment happens against the phenotype in
    :func:`_align_by_ids`). For a bare ``numpy.ndarray`` — which carries no
    id information of its own — the phenotype's own index is reused
    positionally, so both must already be in the same row order; a shape
    mismatch is reported with an actionable hint rather than a raw
    ``IndexError``.
    """
    if isinstance(genotype, pd.DataFrame):
        return ArrayReader(genotype.to_numpy(dtype=np.float64), list(genotype.index))

    G = np.asarray(genotype, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(
            f"genotype array must be 2-D (n_samples, n_variants); got shape {G.shape}."
        )
    if G.shape[0] != len(phenotype_index):
        raise ValueError(
            f"genotype has {G.shape[0]} rows but phenotype has {len(phenotype_index)} "
            "samples. A bare genotype ndarray (no sample ids) must align 1:1, in row "
            "order, with the phenotype. Pass a pandas DataFrame with a sample-id index, "
            "or a genotype file path, if the row order might differ."
        )
    return ArrayReader(G, [str(i) for i in phenotype_index])


def _align_by_ids(
    phenotype: pd.Series,
    covariates: pd.DataFrame | None,
    reader: Any,
) -> tuple[pd.Series, pd.DataFrame | None, object]:
    """Intersect phenotype and genotype-reader sample ids; reindex both.

    Sample ids are matched as strings and the intersection is sorted
    lexicographically for a deterministic, reproducible alignment order
    (matching the convention used by
    :func:`torchgenomics.io.phenotype.load_phenotype`).

    Raises
    ------
    ValueError
        If the shared-id intersection has fewer than :data:`_MIN_SHARED_SAMPLES`
        samples (this includes the empty-intersection case). The message
        names both input sample counts and includes the word "shared" so it
        reads as an actionable diagnostic rather than a raw traceback. Also
        raised (before any ``.loc[]`` lookup that could otherwise surface a
        raw ``KeyError``) if ``covariates`` is supplied but does not cover
        every id in the phenotype/genotype intersection — e.g. PCs computed
        for only a genotyped subset of samples.
    """
    pheno_ids = [str(i) for i in phenotype.index]
    geno_ids = [str(s) for s in reader.sample_ids]
    shared = sorted(set(pheno_ids) & set(geno_ids))

    if len(shared) < _MIN_SHARED_SAMPLES:
        raise ValueError(
            f"Only {len(shared)} shared sample IDs between phenotype "
            f"({len(pheno_ids)} samples) and genotype ({len(geno_ids)} samples) — "
            "too few to run a GWAS. Check that sample IDs match between the two "
            "inputs (same casing, no extra whitespace, matching ID column)."
        )

    if covariates is not None:
        covar_ids = {str(i) for i in covariates.index}
        missing_from_covariates = [sid for sid in shared if sid not in covar_ids]
        if missing_from_covariates:
            raise ValueError(
                f"covariates cover {len(covar_ids)} sample ids, but "
                f"{len(missing_from_covariates)} of the {len(shared)} samples shared "
                "between phenotype and genotype are missing from covariates "
                f"(e.g. {missing_from_covariates[:5]}). Supply covariates for every "
                "shared sample (e.g. PCs computed on the full genotyped set), or omit "
                "the covariates= argument to run without them."
            )

    pheno_reindexed = phenotype.copy()
    pheno_reindexed.index = pheno_ids
    aligned_phenotype = pheno_reindexed.loc[shared]

    aligned_covariates = None
    if covariates is not None:
        covar_reindexed = covariates.copy()
        covar_reindexed.index = [str(i) for i in covariates.index]
        aligned_covariates = covar_reindexed.loc[shared]

    id_to_row = {sid: i for i, sid in enumerate(geno_ids)}
    keep_indices = [id_to_row[sid] for sid in shared]

    if isinstance(reader, ArrayReader):
        aligned_reader: object = ArrayReader(
            reader.G[keep_indices, :], shared, reader.variant_meta
        )
    else:
        from ..io.aligned import SampleAlignedReader

        aligned_reader = SampleAlignedReader(reader, keep_indices, shared)

    return aligned_phenotype, aligned_covariates, aligned_reader


def _align_by_ids_multi(
    pheno_df: pd.DataFrame,
    covariates: pd.DataFrame | None,
    reader: Any,
) -> tuple[pd.DataFrame, pd.DataFrame | None, object]:
    """Multi-trait counterpart to :func:`_align_by_ids`, operating on a DataFrame.

    Mirrors :func:`_align_by_ids` exactly (string-id intersection,
    lexicographic sort, ``.loc[shared]`` reindex, same friendly errors) but
    aligns every trait column of ``pheno_df`` at once instead of a single
    Series.

    Raises
    ------
    ValueError
        Same conditions as :func:`_align_by_ids`: too few (including zero)
        shared sample IDs, or covariates that don't cover the full shared-id
        set.
    """
    pheno_ids = [str(i) for i in pheno_df.index]
    geno_ids = [str(s) for s in reader.sample_ids]
    shared = sorted(set(pheno_ids) & set(geno_ids))

    if len(shared) < _MIN_SHARED_SAMPLES:
        raise ValueError(
            f"Only {len(shared)} shared sample IDs between phenotype "
            f"({len(pheno_ids)} samples) and genotype ({len(geno_ids)} samples) — "
            "too few to run a GWAS. Check that sample IDs match between the two "
            "inputs (same casing, no extra whitespace, matching ID column)."
        )

    if covariates is not None:
        covar_ids = {str(i) for i in covariates.index}
        missing_from_covariates = [sid for sid in shared if sid not in covar_ids]
        if missing_from_covariates:
            raise ValueError(
                f"covariates cover {len(covar_ids)} sample ids, but "
                f"{len(missing_from_covariates)} of the {len(shared)} samples shared "
                "between phenotype and genotype are missing from covariates "
                f"(e.g. {missing_from_covariates[:5]}). Supply covariates for every "
                "shared sample (e.g. PCs computed on the full genotyped set), or omit "
                "the covariates= argument to run without them."
            )

    pheno_reindexed = pheno_df.copy()
    pheno_reindexed.index = pheno_ids
    aligned_pheno_df = pheno_reindexed.loc[shared]

    aligned_covariates = None
    if covariates is not None:
        covar_reindexed = covariates.copy()
        covar_reindexed.index = [str(i) for i in covariates.index]
        aligned_covariates = covar_reindexed.loc[shared]

    id_to_row = {sid: i for i, sid in enumerate(geno_ids)}
    keep_indices = [id_to_row[sid] for sid in shared]

    if isinstance(reader, ArrayReader):
        aligned_reader: object = ArrayReader(
            reader.G[keep_indices, :], shared, reader.variant_meta
        )
    else:
        from ..io.aligned import SampleAlignedReader

        aligned_reader = SampleAlignedReader(reader, keep_indices, shared)

    return aligned_pheno_df, aligned_covariates, aligned_reader


def _load_inputs_multitrait(
    phenotype: Any,
    genotype: Any,
    covariates: Any,
    traits: list[str],
) -> GwasInputs:
    """Multi-trait implementation of :func:`load_inputs` (``traits=[...]``).

    Mirrors the single-trait branch of :func:`load_inputs` structurally
    (same path-vs-in-memory genotype predicate, same
    :func:`_make_array_reader` / duck-typed-reader handling) but resolves and
    aligns *all* named trait columns at once via :func:`_resolve_phenotypes`
    / :func:`_align_by_ids_multi`. ``trait_type`` is fixed to
    ``"continuous"``; ``phenotype``/``trait_name`` on the returned
    :class:`GwasInputs` are the *first* named trait, so single-trait
    consumers of those two fields keep working unchanged.

    Raises
    ------
    ValueError
        If ``traits`` is empty, or via the same failure modes as
        :func:`_resolve_phenotypes` / :func:`_align_by_ids_multi`.
    """
    if len(traits) < 1:
        raise ValueError("traits=[] is empty; pass the trait column names.")

    pheno_df = _resolve_phenotypes(phenotype, traits)
    covar_df = _resolve_covariates(covariates, pheno_df.index) if covariates is not None else None

    if isinstance(genotype, (str, Path)):
        # Path genotype: defer opening + real alignment to the scan/dispatch
        # layer, exactly like the single-trait branch above.
        genotype_path_or_reader: object = str(genotype)
        aligned_df = pheno_df
        aligned_covariates = covar_df
        n_samples = len(aligned_df)
    else:
        if isinstance(genotype, (np.ndarray, pd.DataFrame)):
            reader: object = _make_array_reader(genotype, pheno_df.index)
        else:
            # Duck-typed GenotypeReader passed directly.
            reader = genotype
        aligned_df, aligned_covariates, aligned_reader = _align_by_ids_multi(
            pheno_df, covar_df, reader
        )
        genotype_path_or_reader = aligned_reader
        n_samples = aligned_reader.n_samples

    first = aligned_df.iloc[:, 0]
    return GwasInputs(
        genotype_path_or_reader=genotype_path_or_reader,
        phenotype=first,
        covariates=aligned_covariates,
        trait_type="continuous",
        n_samples=n_samples,
        trait_name=str(aligned_df.columns[0]),
        phenotypes=aligned_df,
        trait_names=list(aligned_df.columns),
    )


def load_inputs(
    phenotype: Any,
    genotype: Any,
    covariates: Any = None,
    trait: str | None = None,
    trait_type: str | None = None,
    traits: list[str] | None = None,
) -> GwasInputs:
    """Resolve and sample-align phenotype/genotype/covariate inputs for ``tg.gwas``.

    This is the friendly front door consumed by the shared dispatch core
    (:func:`torchgenomics.api._dispatch.run_model`, Task 4): it accepts
    whatever a notebook user already has on hand, picks a single trait,
    aligns samples across inputs, infers (or validates) the trait type, and
    raises actionable ``ValueError`` messages on any mismatch — never a raw
    ``KeyError``, shape-mismatch, or torch traceback.

    Parameters
    ----------
    phenotype : str | pathlib.Path | pandas.Series | pandas.DataFrame | numpy.ndarray
        Phenotype source. A path is read via the existing tabular loader
        (:func:`torchgenomics.io.phenotype._load_tabular`) with sample-id
        column auto-detection; a DataFrame/Series lets ``trait`` select the
        column; a bare 1-D ndarray has no sample ids of its own.
    genotype : str | pathlib.Path | numpy.ndarray | pandas.DataFrame | GenotypeReader
        Genotype source. A path is stored as-is (opened downstream by Task 4
        — this function does not touch the file, so no real sample-id
        alignment happens against a path genotype here; ``n_samples`` falls
        back to the phenotype length). An ndarray/DataFrame is wrapped in
        :class:`ArrayReader` and sample-aligned against the phenotype by id
        intersection. An already-constructed reader object (duck-typed to
        the :class:`~torchgenomics.io.base.GenotypeReader` protocol) is
        accepted directly and aligned the same way via
        :class:`~torchgenomics.io.aligned.SampleAlignedReader`.
    covariates : str | pathlib.Path | pandas.DataFrame | numpy.ndarray | None, default None
        Optional covariates, resolved the same way as ``genotype``'s
        in-memory forms and aligned to the same sample order as the
        phenotype.
    trait : str | None, default None
        Column name to select when ``phenotype`` is a DataFrame or a path;
        ignored (or checked for consistency) when ``phenotype`` is already a
        named Series.
    trait_type : str | None, default None
        Force the trait type instead of inferring it via
        :func:`detect_trait_type`. One of ``"continuous"``, ``"binary"``,
        ``"categorical"``; anything else raises a friendly ``ValueError``.
    traits : list[str] | None, default None
        Multi-trait mode switch. When given, ``phenotype`` must be a
        ``pandas.DataFrame`` (or a path to a table) containing every named
        column; the returned :class:`GwasInputs` carries all of them as
        ``phenotypes``/``trait_names``, sample-aligned together, while
        ``phenotype``/``trait_name`` are still populated with just the
        *first* named trait so single-trait consumers keep working
        unchanged. ``trait_type`` is fixed to ``"continuous"`` in this mode
        (multi-trait models are Gaussian-only) and ``trait``/``trait_type``
        arguments are ignored. ``None`` (the default) keeps the existing
        single-trait behavior entirely untouched.

    Returns
    -------
    GwasInputs
        Sample-aligned bundle ready for the dispatch layer.

    Raises
    ------
    ValueError
        On an unsupported input type, a missing/ambiguous trait column, a
        genotype/covariate shape mismatch, an invalid ``trait_type``, or too
        few (including zero) shared sample IDs between phenotype and
        genotype (or between the shared set and supplied covariates).
    """
    if trait_type is not None and trait_type not in _VALID_TRAIT_TYPES:
        raise ValueError(
            f"trait_type={trait_type!r} is not a recognized trait type. Valid options "
            f"are: {sorted(_VALID_TRAIT_TYPES)}. Omit trait_type to infer it "
            "automatically instead."
        )

    if traits is not None:
        return _load_inputs_multitrait(phenotype, genotype, covariates, traits)

    pheno_series = _resolve_phenotype(phenotype, trait)
    covar_df = (
        _resolve_covariates(covariates, pheno_series.index) if covariates is not None else None
    )

    if isinstance(genotype, (str, Path)):
        # Path genotype: defer opening + real alignment to the scan/dispatch
        # layer (Task 4), which already owns format detection and
        # SampleAlignedReader. n_samples is the aligned-phenotype length
        # here; the true genotype sample count is only known once opened.
        genotype_path_or_reader: object = str(genotype)
        aligned_phenotype = pheno_series
        aligned_covariates = covar_df
        n_samples = len(aligned_phenotype)
    else:
        if isinstance(genotype, (np.ndarray, pd.DataFrame)):
            reader: object = _make_array_reader(genotype, pheno_series.index)
        else:
            # Duck-typed GenotypeReader passed directly.
            reader = genotype
        aligned_phenotype, aligned_covariates, aligned_reader = _align_by_ids(
            pheno_series, covar_df, reader
        )
        genotype_path_or_reader = aligned_reader
        n_samples = aligned_reader.n_samples

    resolved_trait_type = trait_type if trait_type is not None else detect_trait_type(
        aligned_phenotype
    )
    trait_name = str(aligned_phenotype.name) if aligned_phenotype.name is not None else "trait"

    return GwasInputs(
        genotype_path_or_reader=genotype_path_or_reader,
        phenotype=aligned_phenotype,
        covariates=aligned_covariates,
        trait_type=resolved_trait_type,
        n_samples=n_samples,
        trait_name=trait_name,
    )
