"""Phase 55: Polyploid allele dosage assignment (Tier 1, always-on)."""
from __future__ import annotations

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
