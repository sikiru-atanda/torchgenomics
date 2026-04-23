"""Phase 55: Polyploid allele dosage assignment (Tier 1, always-on)."""
from __future__ import annotations

import torch

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
