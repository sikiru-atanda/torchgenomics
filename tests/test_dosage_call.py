"""Phase 55: Polyploid allele dosage assignment (Tier 1, always-on)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from torchgwas.preprocess import dosage_call as dc_module
from torchgwas.preprocess.dosage_call import DosageCallResult, _VALID_MODELS


def test_dosage_call_result_dataclass_fields():
    r = DosageCallResult(
        probs=torch.zeros(2, 3, 5, dtype=torch.float64),
        sample_ids=["s1", "s2"],
        variant_ids=["v1", "v2", "v3"],
        ploidy=4,
        tool="updog",
        tool_version="2.0.2",
        model="norm",
        mean_dosage_var=torch.zeros(3, dtype=torch.float64),
        allele_freq=torch.zeros(3, dtype=torch.float64),
        n_missing=0,
        input_hash="abc",
        cmd="Rscript driver.R ...",
    )
    assert r.probs.shape == (2, 3, 5)
    assert r.ploidy == 4
    assert r.tool == "updog"


def test_valid_models_contains_updog_flexdog_set():
    assert {"norm", "hw", "bb", "s1", "f1", "s1pp", "f1pp", "flex", "uniform"} <= _VALID_MODELS


def _reset_cache():
    dc_module._UPDOG_CHECKED = None


def test_check_environment_missing_rscript_raises(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript not found"):
        dc_module._check_environment(rscript=None)


def test_check_environment_missing_updog_raises(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which",
                        lambda x: "/usr/bin/Rscript" if x == "Rscript" else None)

    class FakeCompleted:
        returncode = 1
        stderr = "there is no package called 'updog'"
        stdout = ""
    monkeypatch.setattr(dc_module.subprocess, "run",
                        lambda *a, **kw: FakeCompleted())

    with pytest.raises(RuntimeError, match="updog R package not installed"):
        dc_module._check_environment(rscript=None)


def test_check_environment_is_cached(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    calls = {"n": 0}

    def fake_run(*a, **kw):
        calls["n"] += 1
        class OK:
            returncode = 0
            stderr = ""
            stdout = "2.0.2"
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    dc_module._check_environment(rscript=None)
    dc_module._check_environment(rscript=None)
    assert calls["n"] == 1  # second call hit the cache


def test_check_environment_rscript_override(monkeypatch):
    _reset_cache()
    # shutil.which returns None — override should be used directly
    monkeypatch.setattr(dc_module.shutil, "which", lambda _: None)

    class OK:
        returncode = 0
        stderr = ""
        stdout = "2.0.2"
    monkeypatch.setattr(dc_module.subprocess, "run", lambda *a, **kw: OK())

    rscript_path = dc_module._check_environment(rscript="/my/custom/Rscript")
    assert rscript_path == "/my/custom/Rscript"


def test_validate_kwargs_rejects_ploidy_out_of_range():
    with pytest.raises(ValueError, match="ploidy"):
        dc_module._validate_kwargs(ploidy=1, model="norm")
    with pytest.raises(ValueError, match="ploidy"):
        dc_module._validate_kwargs(ploidy=9, model="norm")


def test_validate_kwargs_rejects_bad_model_name():
    with pytest.raises(ValueError, match="model"):
        dc_module._validate_kwargs(ploidy=4, model="not-a-model")


def test_validate_kwargs_accepts_valid_combinations():
    for m in ("norm", "hw", "bb", "s1", "f1", "flex", "uniform"):
        dc_module._validate_kwargs(ploidy=4, model=m)  # no raise
    for p in (2, 3, 4, 6, 8):
        dc_module._validate_kwargs(ploidy=p, model="norm")  # no raise


# --- Module-level VCF constants (for _extract_ad_from_vcf tests) ---

_TOY_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
1\t100\trs001\tA\tT\t.\tPASS\t.\tGT:AD\t0/0/0/0:20,0\t0/0/0/1:15,5\t0/0/1/1:10,10
1\t200\trs002\tC\tG\t.\tPASS\t.\tGT:AD\t0/0/1/1:12,8\t0/1/1/1:6,14\t1/1/1/1:0,22
1\t300\trs003\tG\tA\t.\tPASS\t.\tGT:AD\t0/0/0/0:25,1\t.:.,.\t0/0/0/1:18,4
1\t400\trs004\tT\tC\t.\tPASS\t.\tGT:AD\t0/1/1/1:4,16\t0/0/1/1:9,11\t0/0/0/1:13,3
"""

_TOY_VCF_NO_AD = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tA\tT\t.\tPASS\t.\tGT\t0/0/0/0\t0/0/1/1
"""

_TOY_VCF_MULTIALLELIC = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tA\tT,G\t.\tPASS\t.\tGT:AD\t0/0/1/2:5,5,5\t0/0/0/0:20,0,0
"""


@pytest.fixture
def toy_vcf(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "toy.vcf"
    path.write_text(_TOY_VCF)
    return str(path)


@pytest.fixture
def toy_vcf_no_ad(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "no_ad.vcf"
    path.write_text(_TOY_VCF_NO_AD)
    return str(path)


@pytest.fixture
def toy_vcf_multiallelic(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "multi.vcf"
    path.write_text(_TOY_VCF_MULTIALLELIC)
    return str(path)


def test_extract_ad_from_vcf_returns_shapes(toy_vcf):
    sample_ids, variant_ids, refmat, sizemat = dc_module._extract_ad_from_vcf(toy_vcf)
    assert sample_ids == ["S1", "S2", "S3"]
    assert variant_ids == ["rs001", "rs002", "rs003", "rs004"]
    assert refmat.shape == (4, 3)   # (m, n)
    assert sizemat.shape == (4, 3)
    # rs001 sample S1: AD=20,0 → ref=20, size=20
    assert refmat[0, 0] == 20
    assert sizemat[0, 0] == 20
    # rs002 sample S3: AD=0,22 → ref=0, size=22
    assert refmat[1, 2] == 0
    assert sizemat[1, 2] == 22


def test_extract_ad_from_vcf_zeroes_missing_samples(toy_vcf):
    _, _, refmat, sizemat = dc_module._extract_ad_from_vcf(toy_vcf)
    # rs003 sample S2 has AD=.,. → both ref and size set to 0
    assert refmat[2, 1] == 0
    assert sizemat[2, 1] == 0


def test_extract_ad_from_vcf_missing_ad_header_raises(toy_vcf_no_ad):
    with pytest.raises(ValueError, match="AD"):
        dc_module._extract_ad_from_vcf(toy_vcf_no_ad)


def test_extract_ad_from_vcf_rejects_multiallelic(toy_vcf_multiallelic):
    with pytest.raises(ValueError, match="multi"):
        dc_module._extract_ad_from_vcf(toy_vcf_multiallelic)


def test_write_input_tsvs_round_trip(tmp_path):
    sample_ids = ["S1", "S2"]
    variant_ids = ["v1", "v2", "v3"]
    refmat = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    sizemat = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    ref_tsv, size_tsv = dc_module._write_input_tsvs(
        tmp_path, sample_ids, variant_ids, refmat, sizemat
    )
    df_ref = pd.read_csv(ref_tsv, sep="\t", index_col=0)
    df_size = pd.read_csv(size_tsv, sep="\t", index_col=0)
    assert list(df_ref.index) == variant_ids
    assert list(df_ref.columns) == sample_ids
    np.testing.assert_array_equal(df_ref.to_numpy(), refmat)
    np.testing.assert_array_equal(df_size.to_numpy(), sizemat)


def test_r_driver_constant_calls_multidog_and_format_multidog():
    src = dc_module._UPDOG_DRIVER_R
    assert "multidog(" in src
    assert "format_multidog(" in src
    assert 'library(updog)' in src


def test_run_r_subprocess_passes_exact_args(tmp_path, monkeypatch):
    captured = {}

    class OK:
        returncode = 0
        stderr = ""
        stdout = "updog multidog complete"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    ref_tsv = tmp_path / "ref.tsv"
    size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    dc_module._run_r_subprocess(
        rscript="/usr/bin/Rscript",
        tmpdir=tmp_path,
        ref_tsv=ref_tsv,
        size_tsv=size_tsv,
        ploidy=4,
        model="norm",
        bias=True,
        od=True,
        seq_error=None,
        n_cores=2,
    )
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/Rscript"
    # driver script path comes next
    assert Path(cmd[1]).name == "driver.R"
    assert cmd[2:] == [str(ref_tsv), str(size_tsv), "4", "norm",
                       "TRUE", "TRUE", "NULL", "2", str(tmp_path)]


def test_run_r_subprocess_nonzero_exit_raises(tmp_path, monkeypatch):
    class Fail:
        returncode = 42
        stderr = "something went wrong"
        stdout = ""

    monkeypatch.setattr(dc_module.subprocess, "run", lambda *a, **kw: Fail())
    ref_tsv = tmp_path / "ref.tsv"; size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    with pytest.raises(RuntimeError, match="exit 42"):
        dc_module._run_r_subprocess(
            rscript="/usr/bin/Rscript", tmpdir=tmp_path,
            ref_tsv=ref_tsv, size_tsv=size_tsv,
            ploidy=4, model="norm", bias=True, od=True,
            seq_error=None, n_cores=1,
        )


def test_run_r_subprocess_passes_seq_error_as_float_string(tmp_path, monkeypatch):
    captured = {}

    class OK:
        returncode = 0; stderr = ""; stdout = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd; return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)
    ref_tsv = tmp_path / "ref.tsv"; size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    dc_module._run_r_subprocess(
        rscript="Rscript", tmpdir=tmp_path,
        ref_tsv=ref_tsv, size_tsv=size_tsv,
        ploidy=4, model="norm", bias=False, od=False,
        seq_error=0.005, n_cores=1,
    )
    cmd = captured["cmd"]
    # boolean flags become FALSE; seq_error becomes "0.005"
    assert cmd[6] == "FALSE"  # bias
    assert cmd[7] == "FALSE"  # od
    assert cmd[8] == "0.005"


# --- Parse output tests ---

def _write_canned_pr_tsvs(tmpdir, sample_ids, variant_ids, ploidy, probs):
    """Write pr_0..pr_k TSVs matching the format R's format_multidog emits.
    probs: (n, m, k+1) array; we write m-row, n-col per file, with a blank
    first column header and variant_ids as rownames.
    """
    for d in range(ploidy + 1):
        df = pd.DataFrame(
            probs[:, :, d].T,   # (m, n)
            index=variant_ids,
            columns=sample_ids,
        )
        df.to_csv(tmpdir / f"pr_{d}.tsv", sep="\t", index=True, index_label="")
    # snp_diag stub
    pd.DataFrame({"snp": variant_ids, "bias": [1.0]*len(variant_ids),
                  "seq": [0.005]*len(variant_ids),
                  "od": [0.01]*len(variant_ids)}).to_csv(
        tmpdir / "snp_diag.tsv", sep="\t", index=False)


def test_parse_output_stacks_in_correct_order(tmp_path):
    sample_ids = ["S1", "S2"]
    variant_ids = ["v1", "v2", "v3"]
    ploidy = 4
    rng = np.random.default_rng(0)
    raw = rng.random((2, 3, 5))
    probs_true = raw / raw.sum(axis=-1, keepdims=True)
    _write_canned_pr_tsvs(tmp_path, sample_ids, variant_ids, ploidy, probs_true)

    probs, snp_diag = dc_module._parse_output(
        tmp_path, sample_ids, variant_ids, ploidy
    )
    assert probs.shape == (2, 3, 5)
    assert probs.dtype == torch.float64
    np.testing.assert_allclose(probs.numpy(), probs_true, atol=1e-6)
    assert list(snp_diag["snp"]) == variant_ids


def test_parse_output_missing_pr_file_raises(tmp_path):
    sample_ids = ["S1"]; variant_ids = ["v1"]; ploidy = 4
    probs = np.ones((1, 1, 5)) / 5
    _write_canned_pr_tsvs(tmp_path, sample_ids, variant_ids, ploidy, probs)
    # remove pr_3.tsv
    (tmp_path / "pr_3.tsv").unlink()
    with pytest.raises(RuntimeError, match="pr_3.tsv"):
        dc_module._parse_output(tmp_path, sample_ids, variant_ids, ploidy)


# --- Normalize probs tests ---

def test_normalize_probs_accepts_within_tolerance():
    probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float64)  # sum=1
    out, n_missing = dc_module._normalize_probs(probs, ploidy=2)
    assert n_missing == 0
    torch.testing.assert_close(out, probs)


def test_normalize_probs_zero_row_becomes_uniform_missing():
    probs = torch.zeros(1, 2, 3, dtype=torch.float64)
    probs[0, 0] = torch.tensor([0.5, 0.3, 0.2])
    # probs[0, 1] is all zero → missing
    out, n_missing = dc_module._normalize_probs(probs, ploidy=2)
    assert n_missing == 1
    torch.testing.assert_close(out[0, 1], torch.tensor([1/3, 1/3, 1/3],
                                                         dtype=torch.float64))
    torch.testing.assert_close(out[0, 0], probs[0, 0])


def test_normalize_probs_negative_is_clamped_and_renormalized(caplog):
    probs = torch.tensor([[[-0.1, 0.6, 0.5]]], dtype=torch.float64)
    with caplog.at_level("WARNING"):
        out, _ = dc_module._normalize_probs(probs, ploidy=2)
    # clamped to [0, 0.6, 0.5], renormalized
    expected = torch.tensor([[[0.0, 0.6/1.1, 0.5/1.1]]], dtype=torch.float64)
    torch.testing.assert_close(out, expected)
    assert any("negative" in rec.message.lower() for rec in caplog.records)


def test_normalize_probs_rejects_wrong_last_dim():
    probs = torch.ones(1, 1, 4, dtype=torch.float64) / 4
    with pytest.raises(RuntimeError, match="shape"):
        dc_module._normalize_probs(probs, ploidy=4)  # expects k+1 = 5


def test_build_result_computes_quality_metrics():
    n, m, ploidy = 4, 3, 4
    probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
    # marker 0: everyone dosage 0 (AF = 0)
    probs[:, 0, 0] = 1.0
    # marker 1: everyone dosage 4 (AF = 1)
    probs[:, 1, 4] = 1.0
    # marker 2: uniform over dosage classes (AF = 0.5)
    probs[:, 2, :] = 1.0 / (ploidy + 1)

    r = dc_module._build_result(
        probs=probs,
        sample_ids=[f"S{i}" for i in range(n)],
        variant_ids=[f"v{j}" for j in range(m)],
        ploidy=ploidy,
        tool_version="2.0.2",
        model="norm",
        n_missing=0,
        input_hash="deadbeef",
        cmd="Rscript driver.R ...",
    )
    assert r.probs is probs
    assert r.tool == "updog"
    assert r.ploidy == 4
    assert r.mean_dosage_var.shape == (m,)
    assert r.allele_freq.shape == (m,)
    # marker 0: AF ≈ 0
    assert r.allele_freq[0].item() == pytest.approx(0.0)
    # marker 1: AF ≈ 1
    assert r.allele_freq[1].item() == pytest.approx(1.0)
    # marker 2: AF ≈ 0.5
    assert r.allele_freq[2].item() == pytest.approx(0.5)


def test_persist_artifacts_writes_three_files(tmp_path):
    n, m, ploidy = 2, 2, 4
    probs = torch.ones(n, m, ploidy + 1, dtype=torch.float64) / (ploidy + 1)
    r = dc_module._build_result(
        probs=probs,
        sample_ids=["S1", "S2"],
        variant_ids=["v1", "v2"],
        ploidy=ploidy, tool_version="2.0.2", model="norm",
        n_missing=0, input_hash="abc", cmd="Rscript ...",
    )
    snp_diag = pd.DataFrame({"snp": ["v1", "v2"], "bias": [1.0, 1.0]})

    prefix = tmp_path / "out"
    dc_module._persist_artifacts(r, snp_diag, prefix=str(prefix))

    loaded = torch.load(str(prefix) + ".probs.pt")
    assert loaded.shape == probs.shape
    meta = json.loads((Path(str(prefix) + ".meta.json")).read_text())
    assert meta["tool"] == "updog"
    assert meta["ploidy"] == 4
    assert meta["sample_ids"] == ["S1", "S2"]
    assert meta["input_hash"] == "abc"
    diag_df = pd.read_csv(str(prefix) + ".snp_diag.tsv", sep="\t")
    assert list(diag_df["snp"]) == ["v1", "v2"]
