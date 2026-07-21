"""Tests for best gene-action model selection (charter Section 9g)."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.models.base import ScanResult
from torchgenomics.stats.best_model import select_best_model


def _make_scan_result(p_values: list[float]) -> ScanResult:
    m = len(p_values)
    return ScanResult(
        chr=["1"] * m,
        pos=list(range(m)),
        snp=[f"snp_{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.5),
        beta=torch.zeros(m),
        se=torch.ones(m),
        stat=torch.ones(m),
        p=torch.tensor(p_values, dtype=torch.float64),
        test="wald",
    )


class TestSelectBestModel:
    def test_picks_lowest_p_no_bic(self):
        """Without BIC, the model with the lowest p-value wins."""
        results = {
            "additive": _make_scan_result([0.05, 0.01, 0.10]),
            "1-dom": _make_scan_result([0.01, 0.05, 0.10]),
            "general": _make_scan_result([0.10, 0.10, 0.001]),
        }
        bm = select_best_model(results, n_samples=100, bic_penalty=False)
        assert bm.best_model[0] == "1-dom"  # 0.01 < 0.05
        assert bm.best_model[1] == "additive"  # 0.01 < 0.05
        assert bm.best_model[2] == "general"  # 0.001 < 0.10

    def test_bic_penalizes_multicolumn(self):
        """BIC penalty should favor single-param models when p-values are similar."""
        # general model (full-rank genotypic: k=4 params for ploidy=4) gets
        # penalized: penalty = 0.5 * (4-1) * log10(100) = 0.5 * 3 * 2 = 3.0
        # So general needs -log10(p) > additive -log10(p) + 3.0 to win.
        results = {
            "additive": _make_scan_result([0.001]),  # -log10p = 3.0
            "general": _make_scan_result([0.0005]),  # -log10p = 3.3 < 3.0 + 3.0
        }
        bm = select_best_model(results, n_samples=100, bic_penalty=True, ploidy=4)
        # additive should win because general's penalized score < additive
        assert bm.best_model[0] == "additive"

    def test_general_wins_when_much_better(self):
        """General model wins despite BIC when it's much more significant."""
        results = {
            "additive": _make_scan_result([0.01]),  # -log10p = 2.0
            "general": _make_scan_result([1e-8]),  # -log10p = 8.0, penalized = 5.0
        }
        bm = select_best_model(results, n_samples=100, bic_penalty=True, ploidy=4)
        assert bm.best_model[0] == "general"

    def test_returns_correct_fields(self):
        results = {
            "additive": _make_scan_result([0.05, 0.01]),
            "1-dom": _make_scan_result([0.01, 0.05]),
        }
        bm = select_best_model(results, n_samples=50, bic_penalty=False)
        assert len(bm.snp) == 2
        assert len(bm.best_model) == 2
        assert bm.best_p.shape == (2,)
        assert bm.best_neglog10p.shape == (2,)
        assert "additive" in bm.all_model_p
        assert "1-dom" in bm.all_model_p

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            select_best_model({}, n_samples=100)

    def test_mismatched_snp_count_raises(self):
        results = {
            "additive": _make_scan_result([0.05, 0.01]),
            "1-dom": _make_scan_result([0.01]),
        }
        with pytest.raises(ValueError, match="SNPs"):
            select_best_model(results, n_samples=100)
