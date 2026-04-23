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
