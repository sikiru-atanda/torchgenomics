"""Phase 56: polyploid phasing via PolyOrigin (Tier 1, always-on)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import torchgwas.preprocess.phase_polyorigin as mod
from torchgwas.preprocess.phase_polyorigin import (
    PhasingResult,
    _build_polyorigin_genofile,
    _build_polyorigin_pedfile,
    _load_map_tsv,
    _validate_inputs,
)


def test_module_imports_without_juliacall():
    # juliacall must NOT be imported at module load — only inside
    # get_runtime(). Verify the import surface is inert.
    assert "juliacall" not in sys.modules, (
        "phase_polyorigin.py must not import juliacall at module scope."
    )
    assert hasattr(mod, "_VALID_PLOIDIES")
    assert mod._VALID_PLOIDIES == frozenset({2, 4, 6})


def test_phasing_result_dataclass_fields():
    r = PhasingResult(
        haplotypes=torch.zeros(3, 4, 2, dtype=torch.int8),
        origin_probs=torch.zeros(3, 4, 2, 4, dtype=torch.float64),
        parent_phased=torch.zeros(2, 4, 2, dtype=torch.int8),
        offspring_ids=["o1", "o2", "o3"],
        parent_ids=["p1", "p2"],
        variant_ids=["v1", "v2"],
        chrom=["1", "1"],
        pos_bp=torch.tensor([100, 200], dtype=torch.int64),
        pos_cm=torch.tensor([0.0001, 0.0002], dtype=torch.float64),
        per_individual_ploidy={"p1": 4, "p2": 4, "o1": 4, "o2": 4, "o3": 4},
        map_refined=False,
        valent_diag=pd.DataFrame({"marker": ["v1"], "valent": ["4x"]}),
        postdose_probs=torch.zeros(3, 2, 5, dtype=torch.float64),
        tool="polyorigin",
        tool_version="1.0.3",
        input_hash="abc",
        cmd="polyOrigin(...)",
        workdir=None,
    )
    assert r.haplotypes.shape == (3, 4, 2)
    assert r.origin_probs.shape == (3, 4, 2, 4)
    assert r.tool == "polyorigin"
    assert r.map_refined is False


# ---------------------------------------------------------------------------
# _build_polyorigin_pedfile tests (Task 3)
# ---------------------------------------------------------------------------

def _write_ped(tmp: Path, rows: list[str]) -> Path:
    p = tmp / "ped.tsv"
    p.write_text("offspring\tparent1\tparent2\n" + "\n".join(rows) + "\n")
    return p


def test_pedfile_single_biparental(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o2\tp1\tp2", "o3\tp1\tp2"])
    out = _build_polyorigin_pedfile(
        str(ped_in), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "o1", "o2", "o3"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert list(df.columns) == ["individual", "population", "motherid", "fatherid", "ploidy"]
    founders = df[df["population"] == 0]
    assert set(founders["individual"]) == {"p1", "p2"}
    assert (founders["motherid"].astype(str) == "0").all()
    assert (founders["fatherid"].astype(str) == "0").all()
    offspring = df[df["population"] != 0]
    assert set(offspring["individual"]) == {"o1", "o2", "o3"}
    assert (offspring["population"] == 1).all()
    assert (offspring["ploidy"] == 4).all()


def test_pedfile_multi_family_connected(tmp_path):
    ped_in = _write_ped(tmp_path, [
        "o1\tp1\tp2", "o2\tp1\tp2",
        "o3\tp1\tp3", "o4\tp1\tp3",
    ])
    out = _build_polyorigin_pedfile(
        str(ped_in), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "p3", "o1", "o2", "o3", "o4"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    founders = df[df["population"] == 0]
    assert set(founders["individual"]) == {"p1", "p2", "p3"}
    offspring = df[df["population"] != 0]
    assert offspring["population"].nunique() == 2
    assert (offspring.groupby("population").size() == 2).all()


def test_pedfile_multi_generation_rejected(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o2\to1\tp2"])
    with pytest.raises(ValueError, match="multi-generation|F1.*only"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1", "o2"},
            workdir=str(tmp_path),
        )


def test_pedfile_duplicate_offspring_rejected(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o1\tp1\tp2"])
    with pytest.raises(ValueError, match="duplicate"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1"},
            workdir=str(tmp_path),
        )


def test_pedfile_missing_parent_rejected(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp3"])
    with pytest.raises(ValueError, match="p3"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1"},
            workdir=str(tmp_path),
        )


def test_pedfile_per_row_ploidy_column(tmp_path):
    p = tmp_path / "ped.tsv"
    p.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t6\n"
        "o2\tp1\tp2\t6\n"
    )
    out = _build_polyorigin_pedfile(
        str(p), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "o1", "o2"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert df.loc[df["individual"] == "o1", "ploidy"].iloc[0] == 6
    assert df.loc[df["individual"] == "p1", "ploidy"].iloc[0] == 4


# ---------------------------------------------------------------------------
# _load_map_tsv tests (Task 4)
# ---------------------------------------------------------------------------

def _write_map(tmp_path: Path, rows: list[str], header="marker\tchrom\tpos_bp") -> Path:
    p = tmp_path / "map.tsv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    return p


def test_load_map_with_cm_column_used_verbatim(tmp_path):
    m = _write_map(
        tmp_path,
        ["v1\t1\t1000000\t0.1", "v2\t1\t2000000\t0.2"],
        header="marker\tchrom\tpos_bp\tcm",
    )
    df = _load_map_tsv(str(m), recomrate=1.0)
    assert list(df.columns) == ["marker", "chrom", "pos_bp", "cm"]
    assert df["cm"].tolist() == [0.1, 0.2]


def test_load_map_without_cm_synthesized_with_warning(tmp_path, caplog):
    m = _write_map(tmp_path, ["v1\t1\t1000000", "v2\t1\t2000000"])
    with caplog.at_level("WARNING"):
        df = _load_map_tsv(str(m), recomrate=1.0)
    assert df["cm"].tolist() == [1.0, 2.0]
    assert any("synthesiz" in rec.message.lower() for rec in caplog.records)


def test_load_map_non_monotonic_bp_rejected(tmp_path):
    m = _write_map(tmp_path, ["v1\t1\t2000000", "v2\t1\t1000000"])
    with pytest.raises(ValueError, match="monotonic|v2"):
        _load_map_tsv(str(m), recomrate=1.0)


def test_load_map_missing_required_col_rejected(tmp_path):
    p = tmp_path / "map.tsv"
    p.write_text("marker\tpos_bp\nv1\t1000\n")  # no chrom
    with pytest.raises(ValueError, match="chrom"):
        _load_map_tsv(str(p), recomrate=1.0)


# ---------------------------------------------------------------------------
# _build_polyorigin_genofile tests (Task 5)
# ---------------------------------------------------------------------------

def _minimal_map_df():
    return pd.DataFrame({
        "marker": ["v1", "v2"],
        "chrom": ["1", "1"],
        "pos_bp": [1000, 2000],
        "cm": [0.001, 0.002],
    })


def test_genofile_probability_encoded_cells(tmp_path):
    probs = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0  # dosage 2 for everything
    out = _build_polyorigin_genofile(
        probs=probs,
        sample_ids=["p1", "o1", "o2"],
        variant_ids=["v1", "v2"],
        map_df=_minimal_map_df(),
        parent_phased_df=None,
        parent_ids={"p1"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert list(df.columns[:3]) == ["marker", "chromosome", "pos"]
    assert df["marker"].tolist() == ["v1", "v2"]
    # Parents alphabetical, then offspring alphabetical
    assert list(df.columns[3:]) == ["p1", "o1", "o2"]
    assert df.loc[0, "p1"] == "0.0000|0.0000|1.0000|0.0000|0.0000"


def test_genofile_parent_phased_escape_hatch(tmp_path):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    parent_phased = pd.DataFrame(
        {"v1": ["1|0|1|0"], "v2": ["0|1|0|1"]}, index=["p1"]
    )
    parent_phased.index.name = "individual"
    out = _build_polyorigin_genofile(
        probs=probs,
        sample_ids=["p1", "o1"],
        variant_ids=["v1", "v2"],
        map_df=_minimal_map_df(),
        parent_phased_df=parent_phased,
        parent_ids={"p1"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert df.loc[0, "p1"] == "1|0|1|0"
    assert df.loc[0, "o1"].count("|") == 4  # 5 probability slots


def test_genofile_parent_in_both_sources_prefers_phased(tmp_path, caplog):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    parent_phased = pd.DataFrame(
        {"v1": ["1|0|1|0"], "v2": ["0|1|0|1"]}, index=["p1"]
    )
    parent_phased.index.name = "individual"
    with caplog.at_level("WARNING"):
        out = _build_polyorigin_genofile(
            probs=probs,
            sample_ids=["p1", "o1"],
            variant_ids=["v1", "v2"],
            map_df=_minimal_map_df(),
            parent_phased_df=parent_phased,
            parent_ids={"p1"},
            workdir=str(tmp_path),
        )
    df = pd.read_csv(out)
    assert df.loc[0, "p1"] == "1|0|1|0"
    assert any("pre-phased" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# _validate_inputs tests (Task 6)
# ---------------------------------------------------------------------------

def test_validate_rejects_bad_ploidy():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match=r"ploidy.*2.*4.*6"):
        _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=3)


def test_validate_rejects_probs_shape_mismatch():
    probs = torch.zeros(2, 2, 4, dtype=torch.float64)
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="probs"):
        _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)


def test_validate_rejects_length_mismatch_sample_ids():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="sample_ids"):
        _validate_inputs(probs, ["s1"], ["v1", "v2"], ploidy=4)


def test_validate_renormalizes_slightly_off_rows():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 0.9995
    out = _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)
    assert torch.allclose(
        out.sum(dim=-1),
        torch.ones_like(out.sum(dim=-1)),
        atol=1e-6,
    )


def test_validate_warns_on_far_off_rows(caplog):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 0.9
    with caplog.at_level("WARNING"):
        _ = _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)
    assert any("renormaliz" in rec.message.lower() for rec in caplog.records)
