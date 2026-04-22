"""Golden tests: TorchGWAS vs GAPIT3 reference outputs.

GAPIT3 (an R package) generates reference outputs for FarmCPU and BLINK on the
MDP maize dataset. TorchGWAS's FarmCPU / BLINK models are compared against these
references with Section-16 tolerances.

Reference outputs are expected at ``gapit_demo/output/``:
  - ``mdp_farmcpu.GWAS.Results.csv`` — FarmCPU per-SNP p-values, pseudo-QTN list
  - ``mdp_blink.GWAS.Results.csv``   — BLINK per-SNP p-values

They are not committed to this repository: GAPIT3 is an R package and regeneration
requires a working R environment with GAPIT3 installed. To generate them on a
machine that has GAPIT3::

    Rscript scripts/generate_golden_data.R --gapit

See ``scripts/generate_golden_data.py`` for orchestration and
``docs/validation_protocol.md`` for the canonical regeneration procedure.

Until the reference outputs are present, these tests skip with a pointer to
the regeneration command.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.golden

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAPIT_OUT = os.path.join(REPO_ROOT, "gapit_demo", "output")

_MISSING_MSG = (
    "GAPIT3 reference outputs not found at {path}. Run "
    "`Rscript scripts/generate_golden_data.R --gapit` on a machine with "
    "GAPIT3 installed (see docs/validation_protocol.md)."
)


def _have(fname: str) -> bool:
    return os.path.isfile(os.path.join(GAPIT_OUT, fname))


def _pvalue_correlation(p_ref: np.ndarray, p_tg: np.ndarray) -> float:
    logp_ref = np.log10(np.clip(p_ref, 1e-300, 1))
    logp_tg = np.log10(np.clip(p_tg, 1e-300, 1))
    return float(np.corrcoef(logp_ref, logp_tg)[0, 1])


class TestGAPITFarmCPU:
    """GAPIT3 FarmCPU reference comparison (MDP EarHT)."""

    def test_pvalues(self):
        """FarmCPU per-SNP -log10(p) correlation against GAPIT3 > 0.99."""
        fname = "mdp_farmcpu.GWAS.Results.csv"
        if not _have(fname):
            pytest.skip(_MISSING_MSG.format(path=GAPIT_OUT))
        # When fixtures land, the body below activates automatically:
        # ref = pd.read_csv(os.path.join(GAPIT_OUT, fname))
        # p_ref, p_tg = _aligned_pvalues(ref, torchgwas_farmcpu_result)
        # assert _pvalue_correlation(p_ref, p_tg) > 0.99
        pytest.fail(
            "FarmCPU comparison body not implemented — fixture is present but "
            "test logic has not been wired yet. Remove this fail() after "
            "implementing FarmCPU comparison against "
            f"{os.path.join(GAPIT_OUT, fname)}."
        )

    def test_pseudo_qtns(self):
        """Pseudo-QTN set convergence: TorchGWAS ∩ GAPIT3 / GAPIT3 > 0.5."""
        fname = "mdp_farmcpu.GWAS.Results.csv"
        if not _have(fname):
            pytest.skip(_MISSING_MSG.format(path=GAPIT_OUT))
        pytest.fail(
            "FarmCPU pseudo-QTN comparison body not implemented. Remove this "
            "fail() after wiring pseudo-QTN set intersection against "
            f"{os.path.join(GAPIT_OUT, fname)}."
        )


class TestGAPITBLINK:
    """GAPIT3 BLINK reference comparison (MDP EarHT)."""

    def test_pvalues(self):
        """BLINK per-SNP -log10(p) correlation against GAPIT3 > 0.99."""
        fname = "mdp_blink.GWAS.Results.csv"
        if not _have(fname):
            pytest.skip(_MISSING_MSG.format(path=GAPIT_OUT))
        pytest.fail(
            "BLINK comparison body not implemented — remove this fail() after "
            "wiring BLINK comparison against "
            f"{os.path.join(GAPIT_OUT, fname)}."
        )
