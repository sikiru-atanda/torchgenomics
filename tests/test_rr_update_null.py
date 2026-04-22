"""Tests for ``update_null`` warm-start dispatch on random-regression
NullFits (Phase 38, Step 10)."""

import torch

from tests.test_rr_lmm_scan import _simulate_with_planted_signal, _vmeta
from torchgwas.models.base import update_null
from torchgwas.models.rr_lmm import RandomRegressionLMM


class TestUpdateNullRandomRegression:
    def test_update_null_preserves_rr_metadata(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=60, T=8, b=3, m=10, h2=0.4, seed=100
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        nf2 = update_null(nf, max_iter=20)
        # RR-specific metadata preserved
        assert nf2.K_coef is not None
        assert nf2.basis_kind == "legendre"
        assert nf2.basis_params == nf.basis_params
        assert nf2.b == nf.b
        assert nf2.Phi_list is not None
        assert torch.equal(nf2.K_coef, nf2.Vg)
        # Updated fit is still scoreable
        res = model.score_chunk(G, nf2, _vmeta(10))
        for p in (res.p_joint, res.p_intercept, res.p_slope, res.p_time_varying):
            assert torch.isfinite(p).all()

    def test_update_null_preserves_include_pe_block(self):
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=60, T=8, b=3, m=8, seed=101
        )
        model = RandomRegressionLMM(basis="legendre", order=2, include_pe=True)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        nf2 = update_null(nf, max_iter=20)
        assert nf2.include_pe is True
        assert nf2.K_pe is not None
        assert nf2.K_pe.shape == (3, 3)
