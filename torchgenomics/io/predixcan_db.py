"""Reader for PrediXcan / FUSION pre-trained eQTL model files (SQLite ``.db``).

PrediXcan, S-PrediXcan (MetaXcan), and FUSION distribute their per-tissue
elastic-net prediction models as SQLite databases with two tables:

- ``weights``: ``rsid``, ``gene``, ``weight``, ``ref_allele``, ``eff_allele``
- ``extra``:   ``gene``, ``genename``, ``"n.snps.in.model"``,
               ``"pred.perf.R2"``, ``"pred.perf.pval"``, ``"pred.perf.qval"``

This module loads such a ``.db`` into the dict-of-tensors shape that
:func:`torchgenomics.postgwas.twas_sumstat` and
:func:`torchgenomics.postgwas.twas_individual` already expect. No external
dependencies beyond the stdlib ``sqlite3`` module.

See `<https://predictdb.org>`_ for canonical GTEx / PsychENCODE /
UTMOST model packs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredixcanModel:
    """Container for the dict-of-tensors view of a PrediXcan/FUSION ``.db``.

    Attributes
    ----------
    weights : dict[str, Tensor]
        ``gene_id -> (k,)`` weight vector. ``k`` is the per-gene
        non-zero-weight count.
    snp_lists : dict[str, list[str]]
        ``gene_id -> list of rsids`` in the same order as ``weights``.
    eff_alleles : dict[str, list[str]]
        ``gene_id -> list of effect alleles`` (same order as ``snp_lists``).
    ref_alleles : dict[str, list[str]]
        ``gene_id -> list of non-effect alleles`` (same order).
    r2_models : dict[str, float]
        ``gene_id -> pred_perf_R2``. Empty if the ``extra`` table is
        missing the column.
    gene_names : dict[str, str]
        ``gene_id -> genename`` (HGNC symbol). Empty if missing.
    n_genes : int
        Number of distinct genes carrying at least one non-zero weight.
    n_weights : int
        Total non-zero weights across all genes.
    """

    weights: dict[str, Tensor]
    snp_lists: dict[str, list[str]]
    eff_alleles: dict[str, list[str]]
    ref_alleles: dict[str, list[str]]
    r2_models: dict[str, float]
    gene_names: dict[str, str]
    n_genes: int
    n_weights: int


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_predixcan_db(
    db_path: str | Path,
    *,
    gene_filter: list[str] | None = None,
    weight_threshold: float = 0.0,
) -> PredixcanModel:
    """Load a PrediXcan / FUSION ``.db`` into the dict-of-tensors shape
    consumed by :func:`twas_sumstat` and :func:`twas_individual`.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite ``.db`` file.
    gene_filter : list[str], optional
        If supplied, only load these gene IDs. The list is matched
        against the ``gene`` column in ``weights`` (typically Ensembl IDs).
        Default: load all genes.
    weight_threshold : float
        Drop SNP-weight rows whose absolute weight is below this
        threshold. Default 0.0 (keep all rows the file carries).

    Returns
    -------
    PredixcanModel
        Dataclass with the four per-gene dicts (weights / snp_lists /
        eff_alleles / ref_alleles), plus the optional r2_models +
        gene_names lookups.

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist.
    sqlite3.DatabaseError
        If the file is not a valid SQLite database.
    ValueError
        If neither the ``weights`` nor a recognisable alternative table
        is present.

    Examples
    --------
    >>> model = read_predixcan_db("gtex_v8_Whole_Blood.db")
    >>> from torchgenomics.postgwas import twas_sumstat
    >>> result = twas_sumstat(
    ...     gwas=gwas_sumstats,
    ...     weights=model.weights,
    ...     snp_lists=model.snp_lists,
    ...     r2_models=model.r2_models,
    ... )
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"PrediXcan .db not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Verify schema.
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        tables = {row[0] for row in cur.fetchall()}
        if "weights" not in tables:
            raise ValueError(
                f"{db_path}: expected a 'weights' table; found {sorted(tables)}."
            )
        has_extra = "extra" in tables

        # Pull weights.
        sql = (
            "SELECT gene, rsid, weight, ref_allele, eff_allele "
            "FROM weights"
        )
        params: tuple = ()
        if gene_filter is not None:
            placeholders = ",".join("?" * len(gene_filter))
            sql += f" WHERE gene IN ({placeholders})"
            params = tuple(gene_filter)
        cur.execute(sql, params)
        rows = cur.fetchall()

        # Group rows by gene, preserving the per-gene SNP order from the file.
        per_gene: dict[str, list[tuple[str, float, str, str]]] = {}
        for gene, rsid, weight, ref, eff in rows:
            if abs(weight) < weight_threshold:
                continue
            per_gene.setdefault(gene, []).append((rsid, float(weight), ref, eff))

        weights_d: dict[str, Tensor] = {}
        snp_lists_d: dict[str, list[str]] = {}
        eff_alleles_d: dict[str, list[str]] = {}
        ref_alleles_d: dict[str, list[str]] = {}
        n_weights = 0
        for gene, entries in per_gene.items():
            if not entries:
                continue
            snp_lists_d[gene] = [e[0] for e in entries]
            weights_d[gene] = torch.tensor(
                [e[1] for e in entries], dtype=torch.float64,
            )
            ref_alleles_d[gene] = [e[2] for e in entries]
            eff_alleles_d[gene] = [e[3] for e in entries]
            n_weights += len(entries)

        # Pull extra table for r2 and gene_name if present.
        r2_d: dict[str, float] = {}
        names_d: dict[str, str] = {}
        if has_extra:
            cur.execute("PRAGMA table_info(extra);")
            extra_cols = {row[1] for row in cur.fetchall()}
            select_cols = ["gene"]
            if "genename" in extra_cols:
                select_cols.append("genename")
            r2_col = next(
                (c for c in extra_cols
                 if c in ("pred.perf.R2", "pred_perf_R2", "cv_R2", "R2")),
                None,
            )
            if r2_col is not None:
                select_cols.append(f'"{r2_col}"')
            cur.execute(f"SELECT {', '.join(select_cols)} FROM extra")
            for row in cur.fetchall():
                gene = row[0]
                if gene not in weights_d:
                    continue
                idx = 1
                if "genename" in select_cols[1] if len(select_cols) > 1 else False:
                    pass
                # robust per-column unpacking by name
                if "genename" in extra_cols:
                    names_d[gene] = str(row[idx]) if row[idx] is not None else ""
                    idx += 1
                if r2_col is not None:
                    val = row[idx]
                    if val is not None:
                        try:
                            r2_d[gene] = float(val)
                        except (TypeError, ValueError):
                            pass
    finally:
        conn.close()

    return PredixcanModel(
        weights=weights_d,
        snp_lists=snp_lists_d,
        eff_alleles=eff_alleles_d,
        ref_alleles=ref_alleles_d,
        r2_models=r2_d,
        gene_names=names_d,
        n_genes=len(weights_d),
        n_weights=n_weights,
    )


def list_genes_in_db(db_path: str | Path) -> list[str]:
    """Return all gene IDs present in the ``weights`` table of a
    PrediXcan ``.db`` file, in alphabetical order.

    Convenience helper for users who want to inspect a model before
    loading the full per-gene weight tensors.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"PrediXcan .db not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT gene FROM weights ORDER BY gene;")
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
