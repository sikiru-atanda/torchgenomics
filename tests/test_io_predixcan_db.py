"""Tests for the PrediXcan / FUSION .db reader.

Builds tiny SQLite fixtures in tmp_path matching the PrediXcan schema
(``weights`` + ``extra`` tables) and verifies that ``read_predixcan_db``
returns the dict-of-tensors shape consumed by ``twas_sumstat``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import torch

from torchgenomics.io import (
    PredixcanModel,
    list_genes_in_db,
    read_predixcan_db,
)


def _build_db(
    db_path: Path,
    rows: list[tuple[str, str, float, str, str]],
    extras: list[tuple] | None = None,
    extra_schema: str | None = None,
) -> None:
    """Write a PrediXcan-style .db with `weights` + optional `extra`."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE weights (rsid TEXT, gene TEXT, weight REAL, "
        "ref_allele TEXT, eff_allele TEXT)"
    )
    cur.executemany(
        "INSERT INTO weights (rsid, gene, weight, ref_allele, eff_allele) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    if extras is not None:
        schema = extra_schema or (
            'CREATE TABLE extra (gene TEXT, genename TEXT, '
            '"n.snps.in.model" INTEGER, "pred.perf.R2" REAL, '
            '"pred.perf.pval" REAL, "pred.perf.qval" REAL)'
        )
        cur.execute(schema)
        cols = schema.split("(", 1)[1].rstrip(")").count(",") + 1
        ph = ", ".join(["?"] * cols)
        cur.executemany(f"INSERT INTO extra VALUES ({ph})", extras)
    conn.commit()
    conn.close()


def test_basic_load(tmp_path):
    db = tmp_path / "model.db"
    rows = [
        ("rs1", "ENSG001", 0.12, "A", "G"),
        ("rs2", "ENSG001", -0.05, "C", "T"),
        ("rs10", "ENSG002", 0.30, "G", "A"),
        ("rs11", "ENSG002", 0.10, "T", "C"),
    ]
    _build_db(db, rows, extras=[
        ("ENSG001", "GENE_A", 2, 0.18, 1e-4, 1e-3),
        ("ENSG002", "GENE_B", 2, 0.42, 1e-7, 1e-6),
    ])
    model = read_predixcan_db(db)
    assert isinstance(model, PredixcanModel)
    assert model.n_genes == 2
    assert model.n_weights == 4
    assert set(model.weights) == {"ENSG001", "ENSG002"}
    assert torch.allclose(
        model.weights["ENSG001"],
        torch.tensor([0.12, -0.05], dtype=torch.float64),
    )
    assert model.snp_lists["ENSG002"] == ["rs10", "rs11"]
    assert model.eff_alleles["ENSG001"] == ["G", "T"]
    assert model.ref_alleles["ENSG001"] == ["A", "C"]
    assert model.r2_models == {"ENSG001": 0.18, "ENSG002": 0.42}
    assert model.gene_names == {"ENSG001": "GENE_A", "ENSG002": "GENE_B"}


def test_weight_threshold_drops_small_weights(tmp_path):
    db = tmp_path / "model.db"
    rows = [
        ("rs1", "ENSG001", 0.12, "A", "G"),
        ("rs2", "ENSG001", 1e-15, "C", "T"),  # negligible
        ("rs3", "ENSG001", -0.05, "A", "G"),
    ]
    _build_db(db, rows)
    model = read_predixcan_db(db, weight_threshold=1e-10)
    assert len(model.weights["ENSG001"]) == 2
    assert model.snp_lists["ENSG001"] == ["rs1", "rs3"]


def test_gene_filter(tmp_path):
    db = tmp_path / "model.db"
    rows = [
        ("rs1", "ENSG001", 0.12, "A", "G"),
        ("rs10", "ENSG002", 0.30, "G", "A"),
        ("rs20", "ENSG003", 0.10, "T", "C"),
    ]
    _build_db(db, rows)
    model = read_predixcan_db(db, gene_filter=["ENSG001", "ENSG003"])
    assert set(model.weights) == {"ENSG001", "ENSG003"}
    assert model.n_genes == 2


def test_missing_extra_table_is_ok(tmp_path):
    db = tmp_path / "model.db"
    rows = [("rs1", "ENSG001", 0.5, "A", "G")]
    _build_db(db, rows, extras=None)
    model = read_predixcan_db(db)
    assert model.weights["ENSG001"].numel() == 1
    assert model.r2_models == {}
    assert model.gene_names == {}


def test_extra_with_alternate_r2_column_name(tmp_path):
    """Some FUSION exports use ``pred_perf_R2`` (underscores) instead of
    the PrediXcan-style ``pred.perf.R2``. Both should resolve."""
    db = tmp_path / "model.db"
    rows = [("rs1", "ENSG001", 0.5, "A", "G")]
    _build_db(
        db, rows,
        extras=[("ENSG001", "GENE_A", 1, 0.55)],
        extra_schema=(
            'CREATE TABLE extra (gene TEXT, genename TEXT, '
            '"n.snps.in.model" INTEGER, pred_perf_R2 REAL)'
        ),
    )
    model = read_predixcan_db(db)
    assert model.r2_models == {"ENSG001": 0.55}


def test_list_genes_in_db(tmp_path):
    db = tmp_path / "model.db"
    rows = [
        ("rs1", "ENSG_C", 0.1, "A", "G"),
        ("rs2", "ENSG_A", 0.2, "A", "G"),
        ("rs3", "ENSG_B", 0.3, "A", "G"),
    ]
    _build_db(db, rows)
    genes = list_genes_in_db(db)
    assert genes == ["ENSG_A", "ENSG_B", "ENSG_C"]  # sorted


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_predixcan_db(tmp_path / "does_not_exist.db")


def test_missing_weights_table_raises(tmp_path):
    db = tmp_path / "bad.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE other_table (x INT)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="weights"):
        read_predixcan_db(db)


def test_integration_with_twas_sumstat(tmp_path):
    """Verify the .db reader's output plugs into twas_sumstat unchanged."""
    db = tmp_path / "model.db"
    rows = [
        ("rs1", "ENSG001", 0.4, "A", "G"),
        ("rs2", "ENSG001", -0.2, "C", "T"),
        ("rs10", "ENSG002", 0.6, "G", "A"),
    ]
    _build_db(db, rows, extras=[
        ("ENSG001", "GENE_A", 2, 0.20, 1e-3, 1e-2),
        ("ENSG002", "GENE_B", 1, 0.35, 1e-6, 1e-5),
    ])
    model = read_predixcan_db(db)

    from torchgenomics.postgwas import SumStats, twas_sumstat
    snps = ["rs1", "rs2", "rs10", "rs99"]
    m = len(snps)
    gwas = SumStats(
        chr=["1"] * m, pos=list(range(100, 100 + m)),
        snp=snps,
        a1=["A", "C", "G", "X"], a2=["G", "T", "A", "X"],
        beta=torch.tensor([0.1, -0.05, 0.2, 0.0], dtype=torch.float64),
        se=torch.tensor([0.02] * m, dtype=torch.float64),
        p=torch.tensor([1e-3, 1e-2, 1e-7, 1.0], dtype=torch.float64),
        n=torch.full((m,), 1000.0, dtype=torch.float64),
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = twas_sumstat(
            gwas, model.weights, model.snp_lists,
            r2_models=model.r2_models,
        )
    assert res.n_genes_tested == 2
    # r2_model should propagate from the .db extras table.
    r2s = {g.gene_id: g.r2_model for g in res.genes}
    assert r2s == {"ENSG001": 0.20, "ENSG002": 0.35}
