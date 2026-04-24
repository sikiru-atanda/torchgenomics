"""Phase 56: polyploid phasing via PolyOrigin (Tier 1, always-on)."""
from __future__ import annotations

import sys

import pandas as pd
import torch

import torchgwas.preprocess.phase_polyorigin as mod
from torchgwas.preprocess.phase_polyorigin import PhasingResult


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
