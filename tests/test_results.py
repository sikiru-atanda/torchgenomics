"""Phase D.2 — serialisation tests for ScanResult.

Every scan subcommand in the CLI used to roll its own pd.DataFrame
construction; the canonical ``to_dict`` / ``to_dataframe`` / ``to_tsv`` /
``to_parquet`` methods replace that. These tests gate:

* field round-trip (no silent data loss for any dataclass field)
* multi-trait expansion into ``beta_1`` / ``se_1`` / ... columns
* TSV + Parquet readback parity
* clear error when pyarrow is missing
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch

from torchgwas.config import STAT_DTYPE
from torchgwas.models.base import ScanResult

pyarrow_available = importlib.util.find_spec("pyarrow") is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_trait_result(m: int = 6) -> ScanResult:
    return ScanResult(
        chr=[str(i % 3 + 1) for i in range(m)],
        pos=list(range(100, 100 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.linspace(0.05, 0.45, m, dtype=STAT_DTYPE),
        beta=torch.linspace(-0.2, 0.2, m, dtype=STAT_DTYPE),
        se=torch.full((m,), 0.1, dtype=STAT_DTYPE),
        stat=torch.linspace(0.0, 10.0, m, dtype=STAT_DTYPE),
        p=torch.linspace(0.001, 0.999, m, dtype=STAT_DTYPE),
        test="wald",
        n_obs=torch.full((m,), 200, dtype=torch.int64),
    )


def _multi_trait_result(m: int = 4, d: int = 3) -> ScanResult:
    return ScanResult(
        chr=["1"] * m,
        pos=list(range(m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.3, dtype=STAT_DTYPE),
        beta=torch.arange(m * d, dtype=STAT_DTYPE).reshape(m, d),
        se=torch.full((m, d), 0.05, dtype=STAT_DTYPE),
        stat=torch.full((m,), 5.0, dtype=STAT_DTYPE),
        p=torch.full((m,), 1e-4, dtype=STAT_DTYPE),
        test="lrt",
    )


# ---------------------------------------------------------------------------
# Dataclass construction + __len__
# ---------------------------------------------------------------------------


def test_scanresult_len_matches_snp_count():
    r = _single_trait_result(m=7)
    assert len(r) == 7


def test_scanresult_default_inference_type_is_marginal():
    r = _single_trait_result(m=2)
    assert r.inference_type == "marginal"


def test_scanresult_accepts_post_selection_inference_type():
    """FarmCPU / BLINK rely on a non-default inference_type tag."""
    r = ScanResult(
        chr=["1"], pos=[1], snp=["rs1"], a1=["A"], a2=["G"],
        af=torch.tensor([0.1], dtype=STAT_DTYPE),
        beta=torch.tensor([0.5], dtype=STAT_DTYPE),
        se=torch.tensor([0.1], dtype=STAT_DTYPE),
        stat=torch.tensor([5.0], dtype=STAT_DTYPE),
        p=torch.tensor([1e-4], dtype=STAT_DTYPE),
        test="wald",
        inference_type="post_selection",
    )
    assert r.inference_type == "post_selection"


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_includes_every_field_for_single_trait():
    r = _single_trait_result(m=3)
    d = r.to_dict()
    expected_keys = {
        "snp", "chr", "pos", "a1", "a2", "af", "beta", "se",
        "stat", "p", "test", "inference_type", "n_obs",
    }
    assert expected_keys.issubset(d.keys())
    assert d["snp"] == ["rs0", "rs1", "rs2"]
    assert d["test"] == "wald"
    assert d["inference_type"] == "marginal"
    # Tensors become nested lists
    assert len(d["beta"]) == 3
    assert len(d["n_obs"]) == 3


def test_to_dict_omits_n_obs_when_absent():
    """n_obs is Optional — to_dict must not write a null entry."""
    r = _single_trait_result(m=2)
    r.n_obs = None
    assert "n_obs" not in r.to_dict()


def test_to_dict_preserves_multi_trait_beta_as_nested_lists():
    r = _multi_trait_result(m=3, d=2)
    d = r.to_dict()
    assert len(d["beta"]) == 3  # one per SNP
    assert all(len(row) == 2 for row in d["beta"])  # one sub-list per trait


# ---------------------------------------------------------------------------
# to_dataframe — single-trait
# ---------------------------------------------------------------------------


def test_dataframe_single_trait_schema():
    r = _single_trait_result(m=5)
    df = r.to_dataframe()
    expected = [
        "snp", "chr", "pos", "a1", "a2", "af", "beta", "se",
        "stat", "p", "n_obs", "test", "inference_type",
    ]
    assert list(df.columns) == expected
    assert len(df) == 5
    assert df["test"].unique().tolist() == ["wald"]


def test_dataframe_values_round_trip_from_tensor():
    r = _single_trait_result(m=4)
    df = r.to_dataframe()
    np.testing.assert_allclose(df["beta"].to_numpy(), r.beta.numpy())
    np.testing.assert_allclose(df["p"].to_numpy(), r.p.numpy())
    assert df["snp"].tolist() == r.snp


def test_dataframe_omits_n_obs_column_when_none():
    r = _single_trait_result(m=3)
    r.n_obs = None
    df = r.to_dataframe()
    assert "n_obs" not in df.columns


# ---------------------------------------------------------------------------
# to_dataframe — multi-trait expansion
# ---------------------------------------------------------------------------


def test_dataframe_multi_trait_expands_beta_and_se_columns():
    r = _multi_trait_result(m=3, d=3)
    df = r.to_dataframe()
    assert "beta" not in df.columns
    assert {"beta_1", "beta_2", "beta_3"}.issubset(df.columns)
    assert {"se_1", "se_2", "se_3"}.issubset(df.columns)


def test_dataframe_multi_trait_values_match_tensor_columns():
    r = _multi_trait_result(m=3, d=2)
    df = r.to_dataframe()
    np.testing.assert_allclose(df["beta_1"].to_numpy(), r.beta[:, 0].numpy())
    np.testing.assert_allclose(df["beta_2"].to_numpy(), r.beta[:, 1].numpy())
    np.testing.assert_allclose(df["se_1"].to_numpy(), r.se[:, 0].numpy())


# ---------------------------------------------------------------------------
# to_tsv round-trip
# ---------------------------------------------------------------------------


def test_tsv_round_trip_preserves_schema_and_values(tmp_path):
    r = _single_trait_result(m=4)
    path = tmp_path / "out.tsv"
    r.to_tsv(str(path))
    round_trip = pd.read_csv(path, sep="\t")
    assert list(round_trip.columns) == list(r.to_dataframe().columns)
    assert round_trip["snp"].tolist() == r.snp
    np.testing.assert_allclose(round_trip["p"].to_numpy(), r.p.numpy())


def test_tsv_round_trip_multi_trait(tmp_path):
    r = _multi_trait_result(m=3, d=3)
    path = tmp_path / "mt.tsv"
    r.to_tsv(str(path))
    round_trip = pd.read_csv(path, sep="\t")
    assert {"beta_1", "beta_2", "beta_3"}.issubset(round_trip.columns)
    np.testing.assert_allclose(round_trip["beta_2"].to_numpy(), r.beta[:, 1].numpy())


# ---------------------------------------------------------------------------
# to_parquet round-trip (gated on pyarrow availability)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not pyarrow_available, reason="pyarrow not installed")
def test_parquet_round_trip_preserves_values(tmp_path):
    r = _single_trait_result(m=5)
    path = tmp_path / "out.parquet"
    r.to_parquet(str(path))
    df = pd.read_parquet(path)
    assert df["snp"].tolist() == r.snp
    np.testing.assert_allclose(df["beta"].to_numpy(), r.beta.numpy())


@pytest.mark.skipif(pyarrow_available, reason="pyarrow is installed")
def test_parquet_without_pyarrow_raises_clear_import_error(tmp_path):
    r = _single_trait_result(m=2)
    with pytest.raises(ImportError, match="pyarrow"):
        r.to_parquet(str(tmp_path / "out.parquet"))


# ---------------------------------------------------------------------------
# Device safety — results constructed on CUDA still serialise
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_to_dataframe_from_cuda_tensors():
    """Tensors living on CUDA must be moved to CPU before numpy conversion.
    Regression guard against accidental ``.numpy()`` on a CUDA tensor."""
    r = _single_trait_result(m=3)
    r.af = r.af.cuda()
    r.beta = r.beta.cuda()
    r.se = r.se.cuda()
    r.stat = r.stat.cuda()
    r.p = r.p.cuda()
    r.n_obs = r.n_obs.cuda() if r.n_obs is not None else None
    df = r.to_dataframe()
    assert len(df) == 3
