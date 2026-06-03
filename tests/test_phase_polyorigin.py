"""Phase 56: polyploid phasing via PolyOrigin (Tier 1, always-on)."""
from __future__ import annotations

import json
import platform
import shutil as _shutil
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import torchgenomics.preprocess.phase_polyorigin as mod
from torchgenomics.preprocess._polyorigin_runtime import _find_existing_julia, _probe_version
from torchgenomics.preprocess.phase_polyorigin import (
    PhasingResult,
    _build_polyorigin_genofile,
    _build_polyorigin_pedfile,
    _decode_haplotypes_per_copy,
    _enumerate_state_table,
    _load_map_tsv,
    _parse_genoprob,
    _parse_maprefined,
    _parse_parentphased,
    _parse_polyancestry,
    _parse_postdose,
    _validate_inputs,
    _validate_state_table,
)

# Task 10: run_polyorigin (imported later in test functions to allow ImportError detection)

FIXTURES = Path(__file__).parent / "fixtures" / "phase_polyorigin"


def test_module_imports_without_juliacall():
    # juliacall must NOT be imported at module load — only inside
    # get_runtime(). Verify the import surface is inert.
    assert "juliacall" not in sys.modules, (
        "phase_polyorigin.py must not import juliacall at module scope."
    )
    assert hasattr(mod, "_VALID_PLOIDIES")
    assert mod._VALID_PLOIDIES == frozenset({2, 4, 6})


def test_phasing_result_dataclass_fields():
    # haplotypes: (n_off, m) int64 — argmax joint-origin-combo index
    # origin_probs: (n_off, m, n_states) float64
    # parent_phased: (n_parents, m, max_ploidy) int8
    r = PhasingResult(
        haplotypes=torch.zeros(3, 2, dtype=torch.int64),
        origin_probs=torch.zeros(3, 2, 10, dtype=torch.float64),
        parent_phased=torch.zeros(2, 2, 4, dtype=torch.int8),
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
        state_table=torch.zeros(36, 4, dtype=torch.int8),
        haplotypes_per_copy=torch.zeros(3, 4, 2, dtype=torch.int8),
    )
    assert r.haplotypes.shape == (3, 2)
    assert r.origin_probs.shape == (3, 2, 10)
    assert r.parent_phased.shape == (2, 2, 4)
    assert r.tool == "polyorigin"
    assert r.map_refined is False
    assert r.state_table.shape == (36, 4)
    assert r.haplotypes_per_copy.shape == (3, 4, 2)


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


# ---------------------------------------------------------------------------
# Task 7: Julia discovery + version probe
# ---------------------------------------------------------------------------

def _make_fake_julia(parent_dir: Path, version: str) -> Path:
    """Cross-platform stub 'julia' responding to --version."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        p = parent_dir / "julia.bat"
        p.write_text(f"@echo off\necho julia version {version}\n")
    else:
        p = parent_dir / "julia"
        p.write_text(f'#!/bin/sh\necho "julia version {version}"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_find_existing_julia_via_explicit_path(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    found = _find_existing_julia(override=str(fj))
    assert found == str(fj) or Path(found).resolve() == Path(fj).resolve()


def test_find_existing_julia_via_env_var(tmp_path, monkeypatch):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    monkeypatch.setenv("TORCHGENOMICS_JULIA", str(fj))
    found = _find_existing_julia(override=None)
    assert Path(found).resolve() == Path(fj).resolve()


def test_find_existing_julia_nothing_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    # Neutralize the common-paths search and juliapkg discovery
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    monkeypatch.setattr(rt, "_common_julia_paths", lambda: [])
    # Patch out juliapkg so its managed install is not found
    import sys
    import types
    fake_juliapkg = types.ModuleType("juliapkg")
    fake_juliapkg.executable = lambda: None
    monkeypatch.setitem(sys.modules, "juliapkg", fake_juliapkg)
    found = _find_existing_julia(override=None)
    assert found is None


def test_probe_version_ok(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    ok, ver = _probe_version(str(fj))
    assert ok is True
    assert ver == "1.10.2"


def test_probe_version_too_low(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.8.5")
    ok, ver = _probe_version(str(fj))
    assert ok is False
    assert ver == "1.8.5"


def test_override_to_nonexistent_file_raises(tmp_path):
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="not.*exist|not a file"):
        _find_existing_julia(override=str(bogus))


# ---------------------------------------------------------------------------
# Task 8: get_runtime — consent gate, version guard, juliacall bootstrap
# ---------------------------------------------------------------------------

import types  # noqa: E402 — appended block; already imported at top via sys


def test_get_runtime_raises_when_not_found_and_auto_install_false(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt

    monkeypatch.delenv("TORCHGENOMICS_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(rt, "_common_julia_paths", lambda: [])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    # Also patch juliapkg so its managed install is not found
    fake_juliapkg = types.ModuleType("juliapkg")
    fake_juliapkg.executable = lambda: None
    monkeypatch.setitem(sys.modules, "juliapkg", fake_juliapkg)

    rt._jl = None
    rt._polyorigin = None
    rt._version = None
    with pytest.raises(RuntimeError, match="Julia not detected|auto_install_julia"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_raises_when_found_julia_too_old(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt

    fj = _make_fake_julia(tmp_path, "1.8.5")
    monkeypatch.setenv("TORCHGENOMICS_JULIA", str(fj))
    rt._jl = None
    rt._polyorigin = None
    rt._version = None
    with pytest.raises(RuntimeError, match=r"1\.8\.5|>= 1\.10"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_raises_without_juliacall_installed(tmp_path, monkeypatch):
    """Stub a valid Julia, then force juliacall import to fail."""
    import builtins

    from torchgenomics.preprocess import _polyorigin_runtime as rt

    fj = _make_fake_julia(tmp_path, "1.10.2")
    monkeypatch.setenv("TORCHGENOMICS_JULIA", str(fj))
    rt._jl = None
    rt._polyorigin = None
    rt._version = None

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "juliacall":
            raise ImportError("stubbed absence of juliacall")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="polyploid-phase|pip install"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_cached_after_first_success():
    from torchgenomics.preprocess import _polyorigin_runtime as rt

    sentinel_jl = object()
    sentinel_po = object()
    rt._jl = sentinel_jl
    rt._polyorigin = sentinel_po
    rt._version = "9.9.9"

    jl, po, ver = rt.get_runtime(julia_path=None, auto_install_julia=False)
    assert jl is sentinel_jl
    assert po is sentinel_po
    assert ver == "9.9.9"

    # Reset for subsequent tests
    rt._jl = None
    rt._polyorigin = None
    rt._version = None


# ---------------------------------------------------------------------------
# Parser tests (Task 9)
# ---------------------------------------------------------------------------

def test_parse_genoprob_shape():
    # Fixture: 1 offspring (o1), 2 markers.
    # Raw CSV has state index 1 (1-based) → 0-based index 0 → n_states=1.
    # PolyOrigin uses 1-based state indices; we convert to 0-based on parse.
    origin_probs, offspring_ids = _parse_genoprob(
        str(FIXTURES / "out_genoprob.csv"),
        expected_offspring=["o1"],
        ploidy=4,
    )
    assert origin_probs.shape == (1, 2, 1)  # (n_off, m, n_states=1)
    assert torch.allclose(
        origin_probs.sum(dim=-1),
        torch.ones((1, 2), dtype=torch.float64),
        atol=1e-3,
    )
    assert offspring_ids == ["o1"]


def test_parse_parentphased_values():
    # Fixture: 2 parents, 2 markers, ploidy=4 → shape (2, 2, 4)
    parent_phased, parent_ids = _parse_parentphased(
        str(FIXTURES / "out_parentphased.csv"),
        expected_parents=["p1", "p2"],
        max_ploidy=4,
    )
    assert parent_phased.shape == (2, 2, 4)  # (n_parents, m, max_ploidy)
    # PolyOrigin uses allele coding 1=ref, 2=alt.
    unique = set(parent_phased.unique().tolist())
    assert unique <= {1, 2, -1}


def test_parse_postdose_shape():
    pd_t, offspring_ids = _parse_postdose(
        str(FIXTURES / "out_postdoseprob.csv"),
        expected_offspring=["o1"],
        max_ploidy=4,
    )
    assert pd_t.shape == (1, 2, 5)
    assert torch.allclose(
        pd_t.sum(dim=-1),
        torch.ones((1, 2), dtype=torch.float64),
        atol=1e-3,
    )


def test_parse_maprefined_preserves_input_order():
    chrom, variant_ids, pos_cm = _parse_maprefined(
        str(FIXTURES / "out_maprefined.csv"),
        expected_markers=["v1", "v2"],
    )
    assert variant_ids == ["v1", "v2"]
    assert chrom == ["1", "1"]
    assert pos_cm.shape == (2,)


def test_parse_polyancestry_returns_dataframe():
    df = _parse_polyancestry(str(FIXTURES / "out_polyancestry.csv"))
    assert "marker" in df.columns
    assert "valent" in df.columns


# ---------------------------------------------------------------------------
# Task 10: run_polyorigin orchestration tests
# ---------------------------------------------------------------------------

def _stub_runtime(workdir_spy: dict):
    """Return a fake PolyOrigin module that copies the canned fixtures
    into the workdir so the parser path can run end-to-end.
    """
    class FakePO:
        @staticmethod
        def polyOrigin(genofile, pedfile, **kwargs):
            workdir_spy["genofile"] = genofile
            workdir_spy["pedfile"] = pedfile
            wd = Path(kwargs["workdir"])
            outstem = kwargs.get("outstem", "out")
            for name in ("genoprob", "postdoseprob", "parentphased",
                         "maprefined", "polyancestry"):
                _shutil.copy(
                    FIXTURES / f"out_{name}.csv",
                    wd / f"{outstem}_{name}.csv",
                )
            return None

    class FakeMain:  # juliacall Main proxy
        PolyOrigin = FakePO

    return FakeMain, FakePO, "1.0.3-fake"


def test_run_polyorigin_happy_path(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    # Stub the runtime — no Julia touched
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    # Minimal inputs: 1 parent, 1 offspring (edge case — but shape is what we're testing)
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp1\n")  # selfing
    # NOTE: the parentphased fixture ships p1 + p2; for this test we align
    # on a 2-parent setup — regenerate ped/map to match the fixture.
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0

    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")

    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped),
        map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"],
        variant_ids=["v1", "v2"],
        auto_install_julia=False,  # stubbed away
    )

    assert isinstance(result, PhasingResult)
    assert result.tool == "polyorigin"
    assert result.per_individual_ploidy["o1"] == 4
    assert Path(f"{out_prefix}.haplotypes.pt").is_file()
    assert Path(f"{out_prefix}.origin_probs.pt").is_file()
    assert Path(f"{out_prefix}.parent_phased.pt").is_file()
    assert Path(f"{out_prefix}.postdose_probs.pt").is_file()
    assert Path(f"{out_prefix}.meta.json").is_file()
    meta = json.loads(Path(f"{out_prefix}.meta.json").read_text())
    assert meta["tool_version"] == "1.0.3-fake"
    assert "input_hash" in meta


def test_run_polyorigin_partial_failure_no_persistent_output(tmp_path, monkeypatch):
    """If the runtime 'succeeds' but emits fewer CSVs than expected, we
    must raise AND leave no <output>.* artifacts."""
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    def _bad_runtime():
        class FakePO:
            @staticmethod
            def polyOrigin(genofile, pedfile, **kwargs):
                wd = Path(kwargs["workdir"])
                # Only write 2 of the 5 expected CSVs
                _shutil.copy(FIXTURES / "out_genoprob.csv", wd / "out_genoprob.csv")
                _shutil.copy(FIXTURES / "out_parentphased.csv", wd / "out_parentphased.csv")
                return None

        class FakeMain:
            PolyOrigin = FakePO
        return FakeMain, FakePO, "1.0.3-fake"

    monkeypatch.setattr(rt, "get_runtime", lambda **_: _bad_runtime())

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="missing|parse"):
        run_polyorigin(
            probs=probs3,
            pedigree_tsv=str(ped),
            map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1"],
            variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )

    # No persistent artifacts
    assert not Path(f"{out_prefix}.haplotypes.pt").exists()
    assert not Path(f"{out_prefix}.meta.json").exists()


def test_run_polyorigin_julia_error_bubbles(tmp_path, monkeypatch):
    """juliacall.JuliaError raised by polyOrigin() → RuntimeError with __cause__."""
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    class FakeJuliaError(Exception):
        pass

    class FakePO:
        @staticmethod
        def polyOrigin(*a, **kw):
            raise FakeJuliaError("convergence failed")

    class FakeMain:
        PolyOrigin = FakePO

    monkeypatch.setattr(rt, "get_runtime", lambda **_: (FakeMain, FakePO, "1.0.3-fake"))
    # Also patch juliacall.JuliaError to our fake class so our except clause matches
    monkeypatch.setattr(
        "torchgenomics.preprocess.phase_polyorigin._JULIA_ERROR",
        FakeJuliaError,
    )

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="PolyOrigin failed") as ei:
        run_polyorigin(
            probs=probs3,
            pedigree_tsv=str(ped),
            map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1"],
            variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )
    assert isinstance(ei.value.__cause__, FakeJuliaError)


# ---------------------------------------------------------------------------
# Task 11: _persist_result atomicity — rollback on partial write
# ---------------------------------------------------------------------------

def test_persist_rolls_back_on_partial_write(tmp_path, monkeypatch):
    from torchgenomics.preprocess.phase_polyorigin import _persist_result

    prefix = tmp_path / "out" / "phased"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Build a minimal PhasingResult (new shape conventions)
    r = PhasingResult(
        haplotypes=torch.zeros(1, 2, dtype=torch.int64),
        origin_probs=torch.zeros(1, 2, 10, dtype=torch.float64),
        parent_phased=torch.zeros(2, 2, 4, dtype=torch.int8),
        offspring_ids=["o1"],
        parent_ids=["p1", "p2"],
        variant_ids=["v1", "v2"],
        chrom=["1", "1"],
        pos_bp=torch.tensor([100, 200], dtype=torch.int64),
        pos_cm=torch.tensor([0.0001, 0.0002], dtype=torch.float64),
        per_individual_ploidy={"p1": 4, "p2": 4, "o1": 4},
        map_refined=False,
        valent_diag=pd.DataFrame({"marker": ["v1"], "valent": ["bivalent"]}),
        postdose_probs=torch.zeros(1, 2, 5, dtype=torch.float64),
        tool="polyorigin",
        tool_version="1.0.3-fake",
        input_hash="abc",
        cmd="...",
        workdir=None,
        state_table=torch.zeros(36, 4, dtype=torch.int8),
        haplotypes_per_copy=torch.zeros(1, 4, 2, dtype=torch.int8),
    )

    # Force meta.json write to fail
    real_write_text = Path.write_text
    call_count = {"n": 0}

    def fake_write_text(self, content, *a, **kw):
        call_count["n"] += 1
        if self.name.startswith("phased.meta.json"):
            raise OSError("simulated disk full")
        return real_write_text(self, content, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    with pytest.raises(OSError):
        _persist_result(r, str(prefix))

    # No persistent artifacts survived
    leftovers = list(prefix.parent.glob("phased.*"))
    assert leftovers == [], f"Unexpected leftover artifacts: {leftovers}"


# ---------------------------------------------------------------------------
# Task 12: CLI phase-poly subcommand
# ---------------------------------------------------------------------------

import subprocess as _sp  # noqa: E402 — appended block


def test_cli_phase_poly_help():
    res = _sp.run(
        [sys.executable, "-m", "torchgenomics", "phase-poly", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    # Core flags present
    for flag in ["--probs", "--pedigree", "--map", "--output", "--ploidy",
                 "--parent-phased", "--auto-install", "--julia-path",
                 "--keep-workdir"]:
        assert flag in res.stdout, (
            f"Flag {flag!r} missing from phase-poly --help output:\n{res.stdout}"
        )


# ---------------------------------------------------------------------------
# Task 13: Tier 1 top-up tests
# ---------------------------------------------------------------------------

def test_meta_json_hash_matches_inputs(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )

    meta = json.loads(Path(f"{out_prefix}.meta.json").read_text())
    assert meta["input_hash"] == result.input_hash
    # Julia cmd captured for reproducibility
    assert "polyOrigin" in meta["cmd"]


def test_mixed_ploidy_per_row_honored(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t4\n"
    )
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )
    assert result.per_individual_ploidy["o1"] == 4
    assert result.per_individual_ploidy["p1"] == 4


def test_env_override_wins_over_path(tmp_path, monkeypatch):
    from torchgenomics.preprocess._polyorigin_runtime import _find_existing_julia

    path_bin = tmp_path / "path_bin"
    env_bin = tmp_path / "env_bin"
    _make_fake_julia(path_bin, "1.10.0")
    env_julia = _make_fake_julia(env_bin, "1.10.2")
    monkeypatch.setenv("PATH", str(path_bin))
    monkeypatch.setenv("TORCHGENOMICS_JULIA", str(env_julia))
    found = _find_existing_julia(override=None)
    assert Path(found).resolve() == Path(env_julia).resolve()


# ---------------------------------------------------------------------------
# _enumerate_state_table tests (Task 2)
# ---------------------------------------------------------------------------

def test_enumerate_state_table_count():
    # PolyOrigin allows double reduction (multisets with replacement):
    # ploidy=2 → C_wr(2,1)² = 2² = 4 states
    st2 = _enumerate_state_table(2)
    assert st2.shape == (4, 2)
    assert st2.dtype == torch.int8
    # ploidy=4 → C_wr(4,2)² = 10² = 100 states
    st4 = _enumerate_state_table(4)
    assert st4.shape == (100, 4)
    assert st4.dtype == torch.int8
    # ploidy=6 → C_wr(6,3)² = 56² = 3136 states
    st6 = _enumerate_state_table(6)
    assert st6.shape == (3136, 6)
    assert st6.dtype == torch.int8


def test_enumerate_state_table_first_state_canonical():
    # For ploidy=4, state 0 = (parent1 gamete (1,1) in 1-based, parent2 gamete (5,5))
    # PolyOrigin double-reduction: first gamete is the repeated-copy (1,1).
    # Encoding: v = parent_id * ploidy + copy_in_parent (0-based)
    # copy 1 (1-based) → parent1, copy 0 → v = 0*4+0 = 0
    # copy 1 (1-based) → parent1, copy 0 → v = 0*4+0 = 0  (double reduction)
    # copy 5 (1-based) → parent2, copy 0 → v = 1*4+0 = 4
    # copy 5 (1-based) → parent2, copy 0 → v = 1*4+0 = 4  (double reduction)
    # So state 0 = [0, 0, 4, 4]
    st4 = _enumerate_state_table(4)
    assert st4[0].tolist() == [0, 0, 4, 4]


def test_enumerate_state_table_lex_ordering():
    # For ploidy=4, parent1 gametes are 2-multisets of {1,2,3,4} (1-based) in lex
    # order: (1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4).
    # Parent2 gametes are 2-multisets of {5,6,7,8} in lex order:
    # (5,5), (5,6), (5,7), (5,8), (6,6), (6,7), (6,8), (7,7), (7,8), (8,8).
    # State s = p1_gamete_idx * 10 + p2_gamete_idx.
    st4 = _enumerate_state_table(4)
    # State 1 = (p1 gamete 0 = (1,1), p2 gamete 1 = (5,6))
    #         → copies: 1→0, 1→0, 5→4, 6→5  → [0, 0, 4, 5]
    assert st4[1].tolist() == [0, 0, 4, 5]
    # State 10 = (p1 gamete 1 = (1,2), p2 gamete 0 = (5,5))
    #         → copies: 1→0, 2→1, 5→4, 5→4  → [0, 1, 4, 4]
    assert st4[10].tolist() == [0, 1, 4, 4]
    # State 99 = (p1 gamete 9 = (4,4), p2 gamete 9 = (8,8))
    #         → copies: 4→3, 4→3, 8→7, 8→7  → [3, 3, 7, 7]
    assert st4[99].tolist() == [3, 3, 7, 7]


# ---------------------------------------------------------------------------
# _decode_haplotypes_per_copy tests (Task 3)
# ---------------------------------------------------------------------------

def test_decode_known_inputs():
    # Setup: ploidy=4, 2 parents, 1 offspring, 1 marker.
    # PolyOrigin allele coding: 1=ref, 2=alt.
    # parent_phased shape (2 parents, 1 marker, 4 copies):
    #   parent1 alleles (1-based) = [2, 1, 2, 1]  → 0-based output: [1, 0, 1, 0]
    #   parent2 alleles (1-based) = [1, 2, 1, 2]  → 0-based output: [0, 1, 0, 1]
    parent_phased = torch.tensor([
        [[2, 1, 2, 1]],  # parent1 at marker 0 (PolyOrigin 1/2 coding)
        [[1, 2, 1, 2]],  # parent2 at marker 0
    ], dtype=torch.int8)

    # haplotypes shape (1 offspring, 1 marker): offspring inherits state 0
    # State 0 in corrected table = (1,1)+(5,5) double-reduction → [0, 0, 4, 4]
    # Use state 1 instead: (1,1)+(5,6) → [0, 0, 4, 5]
    # parent1 copy 0 = 2, copy 0 = 2; parent2 copy 0 = 1, copy 1 = 2
    # output (allele-1): [1, 1, 0, 1]
    haplotypes = torch.tensor([[1]], dtype=torch.int64)

    state_table = _enumerate_state_table(4)

    out = _decode_haplotypes_per_copy(
        haplotypes=haplotypes,
        parent_phased=parent_phased,
        state_table=state_table,
        ploidy=4,
    )
    # Expected shape (1 offspring, 4 copies, 1 marker)
    assert out.shape == (1, 4, 1)
    assert out.dtype == torch.int8
    # State 1 = [0, 0, 4, 5]:
    # Copy 0: parent_id=0, copy_in_parent=0 → allele=2 → 0-based=1
    # Copy 1: parent_id=0, copy_in_parent=0 → allele=2 → 0-based=1
    # Copy 2: parent_id=1, copy_in_parent=0 → allele=1 → 0-based=0
    # Copy 3: parent_id=1, copy_in_parent=1 → allele=2 → 0-based=1
    assert out[0, :, 0].tolist() == [1, 1, 0, 1]


# ---------------------------------------------------------------------------
# _validate_state_table tests (Task 4)
# ---------------------------------------------------------------------------

def _self_consistent_validate_inputs():
    """Build (state_table, origin_probs, parent_phased, postdose_probs) that
    are self-consistent under ploidy=4.

    Uses PolyOrigin's real allele coding (1=ref, 2=alt).
    State 0 = (1,1)+(5,5) double reduction:
      copies 1,1 from parent1 → both copy 0 → alleles [2, 2] (alt, alt) → dose 2
      copies 5,5 from parent2 → both copy 0 → alleles [1, 1] (ref, ref) → dose 0
      Total dose at state 0 = 2 (sum of (allele-1) over 4 copies = 1+1+0+0 = 2).
    """
    ploidy = 4
    state_table = _enumerate_state_table(ploidy)

    # PolyOrigin 1/2 allele coding: parent1 copy 0 = alt(2), copy 1 = ref(1), etc.
    # parent2 copy 0 = ref(1), copy 1 = alt(2), etc.
    parent_phased = torch.tensor([
        [[2, 1, 2, 1]],  # parent1 at marker 0: copy0=2(alt), copy1=1(ref), copy2=2, copy3=1
        [[1, 2, 1, 2]],  # parent2 at marker 0: copy0=1(ref), copy1=2(alt), copy2=1, copy3=2
    ], dtype=torch.int8)  # (2, 1, 4)

    # State 0 = [0, 0, 4, 4]:
    # copy0 = parent_id=0, copy_in_parent=0 → allele=2 → (allele-1)=1
    # copy0 = parent_id=0, copy_in_parent=0 → allele=2 → (allele-1)=1  [double reduction]
    # copy4 = parent_id=1, copy_in_parent=0 → allele=1 → (allele-1)=0
    # copy4 = parent_id=1, copy_in_parent=0 → allele=1 → (allele-1)=0  [double reduction]
    # dose = 1+1+0+0 = 2
    n_states = state_table.shape[0]
    origin_probs = torch.zeros(1, 1, n_states, dtype=torch.float64)
    origin_probs[0, 0, 0] = 1.0

    # Consistent postdose_probs: dose 2 with probability 1
    postdose_probs = torch.zeros(1, 1, ploidy + 1, dtype=torch.float64)
    postdose_probs[0, 0, 2] = 1.0

    return state_table, origin_probs, parent_phased, postdose_probs, ploidy


def test_validate_passes_on_self_consistent_input():
    state_table, origin_probs, parent_phased, postdose_probs, ploidy = \
        _self_consistent_validate_inputs()
    # Should return None (pass silently)
    _validate_state_table(
        state_table, origin_probs, parent_phased, postdose_probs,
        ploidy=ploidy,
    )


def test_validate_fails_on_reorder():
    # Use state_table with rows 0 and 1 swapped.
    # State 0 = [0,0,4,4] → dose = 1+1+0+0 = 2 (with parent_phased above)
    # State 1 = [0,0,4,5]:
    #   copies: parent1[0]=2, parent1[0]=2, parent2[0]=1, parent2[1]=2
    #   dose = 1+1+0+1 = 3
    # Swapping changes dose at position 0 (which has full prob on state 0)
    # from 2 to 3, breaking the round-trip.
    state_table, origin_probs, parent_phased, postdose_probs, ploidy = \
        _self_consistent_validate_inputs()

    swapped = state_table.clone()
    swapped[[0, 1]] = swapped[[1, 0]]

    with pytest.raises(RuntimeError, match="State-table round-trip mismatch"):
        _validate_state_table(
            swapped, origin_probs, parent_phased, postdose_probs,
            ploidy=ploidy,
        )


# ---------------------------------------------------------------------------
# Task 5: run_polyorigin orchestration — state_table + decoder wired in
# ---------------------------------------------------------------------------

def test_run_polyorigin_haplotypes_per_copy_field_populated(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin
    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )
    # 1 offspring, 4 copies, 2 markers
    assert result.haplotypes_per_copy.shape == (1, 4, 2)
    assert result.haplotypes_per_copy.dtype == torch.int8
    assert set(result.haplotypes_per_copy.unique().tolist()) <= {0, 1}


def test_run_polyorigin_state_table_field_populated(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin
    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )
    # ploidy=4 → 100 states (CwR), 4 copies per state
    assert result.state_table.shape == (100, 4)
    assert result.state_table.dtype == torch.int8
    # Values in [0, 2*ploidy) = [0, 8)
    vmin = int(result.state_table.min())
    vmax = int(result.state_table.max())
    assert vmin >= 0
    assert vmax < 8


def test_run_polyorigin_mixed_ploidy_rejected(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    # probs shape must include 2 offspring + 2 parents = 4 samples
    probs4 = torch.zeros(4, 2, 5, dtype=torch.float64)
    probs4[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t4\n"
        "o2\tp1\tp2\t2\n"
    )
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin
    with pytest.raises(ValueError, match="uniform ploidy"):
        run_polyorigin(
            probs=probs4,
            pedigree_tsv=str(ped), map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1", "o2"], variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )

    # Mixed-ploidy rejection happens before persistence — no artifacts
    assert not (out_prefix.parent / "phased.haplotypes_per_copy.pt").exists()
    assert not (out_prefix.parent / "phased.meta.json").exists()


def test_persist_per_copy_artifacts(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )

    st_path = Path(f"{out_prefix}.state_table.pt")
    hpc_path = Path(f"{out_prefix}.haplotypes_per_copy.pt")
    assert st_path.is_file()
    assert hpc_path.is_file()

    st_loaded = torch.load(st_path)
    hpc_loaded = torch.load(hpc_path)
    assert torch.equal(st_loaded, result.state_table)
    assert torch.equal(hpc_loaded, result.haplotypes_per_copy)
