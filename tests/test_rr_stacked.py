"""Tests for RandomRegressionLMM mode='stacked' (Phase 38, Step 7).

The stacked-mode path treats the T common time points as T pseudo-traits
via MultiTraitLMM and projects the resulting Vg_T / Ve_T into basis-
coefficient space. On balanced data this is an independent-route
estimate of K_coef which should agree with the projection-mode K_coef
within tolerance — a useful end-to-end verification cross-check.
"""

import pytest
import torch

from tests.test_rr_lmm_scan import _simulate_with_planted_signal, _vmeta
from torchgenomics.models.rr_lmm import RandomRegressionLMM, RRScanResult


class TestStackedMode:
    def test_stacked_fit_runs_and_attaches_metadata(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=60, T=8, b=3, m=10, h2=0.4, seed=70
        )
        model = RandomRegressionLMM(
            basis="legendre", order=2, mode="stacked"
        )
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        # Coefficient-space Vg / Ve attached
        assert nf.Vg.shape == (3, 3)
        assert nf.Ve.shape == (3, 3)
        # K_coef alias matches Vg
        assert torch.equal(nf.K_coef, nf.Vg)
        # Symmetric and PSD
        assert torch.allclose(nf.Vg, nf.Vg.T, atol=1e-8)
        eigvals = torch.linalg.eigvalsh(nf.Vg)
        assert (eigvals >= -1e-6).all()
        # Stacked-only diagnostics
        assert hasattr(nf, "Vg_T_stacked") and nf.Vg_T_stacked.shape == (8, 8)
        assert hasattr(nf, "Ve_T_stacked") and nf.Ve_T_stacked.shape == (8, 8)

    def test_stacked_matches_projection_balanced(self):
        """Projection-mode and stacked-mode K_coef should agree on balanced
        data within a loose tolerance (both estimate the same underlying
        Φ K_coef Φ' but via two different REML factorizations)."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=120, T=10, b=3, m=10, h2=0.5, seed=71
        )
        m_proj = RandomRegressionLMM(basis="legendre", order=2, mode="projection")
        m_stack = RandomRegressionLMM(basis="legendre", order=2, mode="stacked")
        nf_proj = m_proj.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                                   t_min=t_min, t_max=t_max)
        nf_stack = m_stack.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                                     t_min=t_min, t_max=t_max)
        # K_coef agreement on the order-of-magnitude scale.
        diff = (nf_proj.Vg - nf_stack.Vg).abs().max()
        scale = nf_proj.Vg.abs().max().clamp(min=1e-6)
        assert diff / scale < 0.5, (
            f"Projection vs stacked K_coef disagree: rel_diff={diff/scale:.3f}\n"
            f"  projection Vg=\n{nf_proj.Vg}\n  stacked Vg=\n{nf_stack.Vg}"
        )

    def test_stacked_scan_returns_finite_pvalues(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=80, T=8, b=3, m=12, h2=0.4, seed=72
        )
        model = RandomRegressionLMM(basis="legendre", order=2, mode="stacked")
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        res = model.score_chunk(G, nf, _vmeta(12))
        assert isinstance(res, RRScanResult)
        for p in (res.p_joint, res.p_intercept, res.p_slope, res.p_time_varying):
            assert torch.isfinite(p).all()
            assert (p >= 0).all() and (p <= 1).all()

    def test_stacked_rejects_unbalanced(self):
        """An individual with a different number of observations should
        cause stacked mode to raise (not silently produce nonsense)."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=20, T=6, b=3, m=8, seed=73
        )
        # Drop the last observation of individual 0 → unbalanced
        keep = torch.ones_like(ids, dtype=torch.bool)
        keep[5] = False  # individual 0's row 5 (its 6th observation)
        Y_short = Y[keep]
        ids_short = ids[keep]
        t_short = t[keep]

        model = RandomRegressionLMM(basis="legendre", order=2, mode="stacked")
        with pytest.raises(ValueError, match="balanced"):
            model.fit_null(
                Y_short, X0, K, sample_ids=ids_short, time_values=t_short,
                t_min=t_min, t_max=t_max,
            )
