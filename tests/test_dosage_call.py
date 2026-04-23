"""Phase 55: Polyploid allele dosage assignment (Tier 1, always-on)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from torchgwas.preprocess import dosage_call as dc_module
from torchgwas.preprocess.dosage_call import _VALID_MODELS, DosageCallResult


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

_TOY_VCF_SYMBOLIC_ALT = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
##ALT=<ID=DEL,Description="Deletion">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tA\t<DEL>\t.\tPASS\t.\tGT:AD\t0/0/0/0:20,0\t0/0/0/1:15,5
"""

_TOY_VCF_INDEL = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tAC\tA\t.\tPASS\t.\tGT:AD\t0/0/0/0:20,0\t0/0/0/1:15,5
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


@pytest.fixture
def toy_vcf_symbolic_alt(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "symbolic.vcf"
    path.write_text(_TOY_VCF_SYMBOLIC_ALT)
    return str(path)


@pytest.fixture
def toy_vcf_indel(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "indel.vcf"
    path.write_text(_TOY_VCF_INDEL)
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


def test_extract_ad_from_vcf_rejects_symbolic_alt(toy_vcf_symbolic_alt):
    # `<DEL>` passes the multi-allelic (`len(ALT) != 1`) check but has no
    # AD semantics for updog. Expect a clearer error than "multi-allelic".
    with pytest.raises(ValueError, match="[Ss]ymbolic"):
        dc_module._extract_ad_from_vcf(toy_vcf_symbolic_alt)


def test_extract_ad_from_vcf_rejects_indel(toy_vcf_indel):
    # REF=AC, ALT=A is a deletion — biallelic by count but not a SNP.
    # updog's dosage model applies only to single-nt REF/ALT pairs.
    with pytest.raises(ValueError, match="[Nn]on-SNP"):
        dc_module._extract_ad_from_vcf(toy_vcf_indel)


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

    loaded = torch.load(str(prefix) + ".probs.pt", weights_only=True)
    assert loaded.shape == probs.shape
    meta = json.loads((Path(str(prefix) + ".meta.json")).read_text())
    assert meta["tool"] == "updog"
    assert meta["ploidy"] == 4
    assert meta["sample_ids"] == ["S1", "S2"]
    assert meta["input_hash"] == "abc"
    diag_df = pd.read_csv(str(prefix) + ".snp_diag.tsv", sep="\t")
    assert list(diag_df["snp"]) == ["v1", "v2"]


# --- Integration tests for run_updog ---

def _stub_updog_subprocess(tmpdir, sample_ids, variant_ids, ploidy,
                            seed=0, n_missing_slots=0):
    """Return a subprocess.run replacement that writes canned pr_*.tsv +
    snp_diag.tsv into whatever --out-dir was passed. Inspects argv to
    locate the output directory.
    """
    def fake_run(cmd, **kwargs):
        # find out_dir = cmd[-1] per our contract
        out_dir = Path(cmd[-1])
        rng = np.random.default_rng(seed)
        raw = rng.random((len(sample_ids), len(variant_ids), ploidy + 1))
        probs = raw / raw.sum(axis=-1, keepdims=True)
        if n_missing_slots:
            probs[0, 0, :] = 0.0  # flag one slot missing
        _write_canned_pr_tsvs(out_dir, sample_ids, variant_ids, ploidy, probs)

        class OK:
            returncode = 0
            stderr = ""
            stdout = "updog multidog complete"
        return OK()
    return fake_run


def test_run_updog_end_to_end_stubbed(tmp_path, monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    # Monkeypatch the VCF extractor so we don't need cyvcf2
    sample_ids = ["S1", "S2", "S3"]
    variant_ids = ["rs001", "rs002", "rs003", "rs004"]
    ploidy = 4
    refmat = np.zeros((len(variant_ids), len(sample_ids)), dtype=np.int64)
    sizemat = np.ones((len(variant_ids), len(sample_ids)), dtype=np.int64) * 20
    monkeypatch.setattr(
        dc_module, "_extract_ad_from_vcf",
        lambda _vcf: (sample_ids, variant_ids, refmat, sizemat),
    )

    # Dummy VCF so hashlib.sha256(Path(input_vcf).read_bytes()) succeeds
    fake_vcf = tmp_path / "fake.vcf"
    fake_vcf.write_text("#placeholder\n")

    # Two subprocess calls happen: (1) updog-version probe, (2) real driver.
    driver_run = _stub_updog_subprocess(tmp_path, sample_ids, variant_ids, ploidy)

    def fake_run(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1] == "-e":
            class Probe:
                returncode = 0; stderr = ""; stdout = "2.0.2"
            return Probe()
        return driver_run(cmd, **kwargs)

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    prefix = tmp_path / "out"
    result = dc_module.run_updog(
        input_vcf=str(fake_vcf),
        output_path=str(prefix),
        ploidy=ploidy,
        model="norm",
    )
    assert result.tool == "updog"
    assert result.ploidy == ploidy
    assert result.sample_ids == sample_ids
    assert result.variant_ids == variant_ids
    assert result.probs.shape == (3, 4, 5)
    assert (tmp_path / "out.probs.pt").is_file()
    assert (tmp_path / "out.meta.json").is_file()
    assert (tmp_path / "out.snp_diag.tsv").is_file()


def test_run_updog_partial_output_raises_and_leaves_no_artifacts(
    tmp_path, monkeypatch
):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    sample_ids = ["S1", "S2", "S3"]
    variant_ids = ["rs001", "rs002", "rs003", "rs004"]
    refmat = np.zeros((len(variant_ids), len(sample_ids)), dtype=np.int64)
    sizemat = np.ones((len(variant_ids), len(sample_ids)), dtype=np.int64) * 20
    monkeypatch.setattr(
        dc_module, "_extract_ad_from_vcf",
        lambda _vcf: (sample_ids, variant_ids, refmat, sizemat),
    )

    fake_vcf = tmp_path / "fake.vcf"
    fake_vcf.write_text("#placeholder\n")

    def broken_driver(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1] == "-e":
            class Probe:
                returncode = 0; stderr = ""; stdout = "2.0.2"
            return Probe()
        # Only write 2 of 5 pr files, no snp_diag — triggers _parse_output error
        out_dir = Path(cmd[-1])
        (out_dir / "pr_0.tsv").write_text("\tS1\nrs001\t0.2\n")
        (out_dir / "pr_1.tsv").write_text("\tS1\nrs001\t0.2\n")
        class OK:
            returncode = 0; stderr = ""; stdout = ""
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", broken_driver)

    prefix = tmp_path / "out"
    with pytest.raises(RuntimeError):
        dc_module.run_updog(input_vcf=str(fake_vcf), output_path=str(prefix),
                             ploidy=4, model="norm")

    # No user-visible output artifacts written
    assert not (tmp_path / "out.probs.pt").exists()
    assert not (tmp_path / "out.meta.json").exists()
    assert not (tmp_path / "out.snp_diag.tsv").exists()


def test_cli_dosage_call_dispatches_to_run_updog(tmp_path, monkeypatch):
    from torchgwas.cli import main as cli_main

    fake_vcf = tmp_path / "fake.vcf"
    fake_vcf.write_text("#placeholder\n")

    captured = {}

    def fake_run_updog(**kwargs):
        captured.update(kwargs)
        probs = torch.ones(3, 4, 5, dtype=torch.float64) / 5
        return dc_module.DosageCallResult(
            probs=probs, sample_ids=["S1","S2","S3"],
            variant_ids=["v1","v2","v3","v4"], ploidy=4, tool="updog",
            tool_version="2.0.2", model="norm",
            mean_dosage_var=torch.zeros(4, dtype=torch.float64),
            allele_freq=torch.zeros(4, dtype=torch.float64),
            n_missing=0, input_hash="x", cmd="",
        )

    monkeypatch.setattr("torchgwas.cli.run_updog", fake_run_updog,
                        raising=False)
    monkeypatch.setattr("torchgwas.preprocess.dosage_call.run_updog",
                        fake_run_updog)

    out_prefix = tmp_path / "out"
    rc = cli_main([
        "dosage-call",
        "--vcf", str(fake_vcf),
        "--output", str(out_prefix),
        "--ploidy", "4",
        "--model", "norm",
        "--n-cores", "2",
    ])
    assert rc == 0
    assert captured["input_vcf"] == str(fake_vcf)
    assert captured["output_path"] == str(out_prefix)
    assert captured["ploidy"] == 4
    assert captured["model"] == "norm"
    assert captured["n_cores"] == 2
    # defaults
    assert captured["bias"] is True
    assert captured["od"] is True
    assert captured["seq_error"] is None


def test_cli_dosage_call_no_bias_no_od(tmp_path, monkeypatch):
    from torchgwas.cli import main as cli_main

    fake_vcf = tmp_path / "fake.vcf"
    fake_vcf.write_text("#placeholder\n")

    captured = {}
    def fake_run_updog(**kwargs):
        captured.update(kwargs)
        probs = torch.ones(1, 1, 5, dtype=torch.float64) / 5
        return dc_module.DosageCallResult(
            probs=probs, sample_ids=["S1"], variant_ids=["v1"],
            ploidy=4, tool="updog", tool_version="2.0.2", model="norm",
            mean_dosage_var=torch.zeros(1, dtype=torch.float64),
            allele_freq=torch.zeros(1, dtype=torch.float64),
            n_missing=0, input_hash="x", cmd="",
        )
    monkeypatch.setattr("torchgwas.preprocess.dosage_call.run_updog",
                        fake_run_updog)

    cli_main([
        "dosage-call", "--vcf", str(fake_vcf), "--output", str(tmp_path / "out"),
        "--ploidy", "4", "--no-bias", "--no-od", "--seq-error", "0.005",
    ])
    assert captured["bias"] is False
    assert captured["od"] is False
    assert captured["seq_error"] == 0.005


def test_reexports_from_torchgwas_preprocess():
    from torchgwas.preprocess import DosageCallResult, run_updog
    assert callable(run_updog)
    assert DosageCallResult is dc_module.DosageCallResult


def test_preprocess_reexport_run_updog_points_to_dosage_call():
    # Disambiguate from the legacy polyrad_wrapper re-export that was
    # removed in this commit: `torchgwas.preprocess.run_updog` is now the
    # Phase 55 dosage_call implementation, not the older flexdog-loop one.
    from torchgwas.preprocess import run_updog as reexported
    from torchgwas.preprocess.dosage_call import run_updog as canonical
    assert reexported is canonical


def test_keep_tmpdir_true_actually_retains_directory(tmp_path, monkeypatch):
    """Regression for the original TemporaryDirectory finalizer bug:
    `keep_tmpdir=True` silently lost the directory on function return.
    Switched to mkdtemp+rmtree so the flag is honored.
    """
    import shutil

    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    sample_ids = ["S1"]
    variant_ids = ["v1", "v2"]
    ploidy = 4
    refmat = np.zeros((len(variant_ids), len(sample_ids)), dtype=np.int64)
    sizemat = np.ones((len(variant_ids), len(sample_ids)), dtype=np.int64) * 20
    monkeypatch.setattr(
        dc_module, "_extract_ad_from_vcf",
        lambda _vcf: (sample_ids, variant_ids, refmat, sizemat),
    )

    fake_vcf = tmp_path / "fake.vcf"
    fake_vcf.write_text("#placeholder\n")

    captured_tmpdir: dict[str, Path] = {}
    driver_run = _stub_updog_subprocess(tmp_path, sample_ids, variant_ids, ploidy)

    def fake_run(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1] == "-e":
            class Probe:
                returncode = 0; stderr = ""; stdout = "2.0.2"
            return Probe()
        captured_tmpdir["path"] = Path(cmd[-1])
        return driver_run(cmd, **kwargs)

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    dc_module.run_updog(
        input_vcf=str(fake_vcf),
        output_path=str(tmp_path / "out"),
        ploidy=ploidy, model="norm",
        keep_tmpdir=True,
    )

    # After run_updog returns, the tempdir must still exist on disk.
    td = captured_tmpdir["path"]
    try:
        assert td.is_dir(), f"keep_tmpdir=True did not retain {td}"
        assert (td / "ref.tsv").is_file()
        assert (td / "driver.R").is_file()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_persist_artifacts_rollback_on_mid_write_failure(tmp_path, monkeypatch):
    """Regression for atomicity: if the second file write fails, the
    final `.probs.pt` must not exist on disk (no dangling first file).
    """
    n, m, ploidy = 2, 2, 4
    probs = torch.ones(n, m, ploidy + 1, dtype=torch.float64) / (ploidy + 1)
    r = dc_module._build_result(
        probs=probs,
        sample_ids=["S1", "S2"], variant_ids=["v1", "v2"],
        ploidy=ploidy, tool_version="2.0.2", model="norm",
        n_missing=0, input_hash="abc", cmd="Rscript ...",
    )
    snp_diag = pd.DataFrame({"snp": ["v1", "v2"], "bias": [1.0, 1.0]})

    # Force the meta.json write to fail.
    real_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):
        if self.name.endswith("meta.json.tmp"):
            raise OSError("simulated mid-write failure")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    prefix = tmp_path / "out"
    with pytest.raises(OSError, match="simulated"):
        dc_module._persist_artifacts(r, snp_diag, prefix=str(prefix))

    # None of the final user-visible paths should exist.
    assert not Path(str(prefix) + ".probs.pt").exists()
    assert not Path(str(prefix) + ".meta.json").exists()
    assert not Path(str(prefix) + ".snp_diag.tsv").exists()
    # And the .tmp siblings should have been cleaned up.
    assert not Path(str(prefix) + ".probs.pt.tmp").exists()
    assert not Path(str(prefix) + ".meta.json.tmp").exists()
    assert not Path(str(prefix) + ".snp_diag.tsv.tmp").exists()
