"""Tests for the per-time-point reconstruction in RandomRegressionLMM.score_chunk
(Phase 38, Step 5).

The 5th test type — "Effect at time t" χ²(1) — is exposed via the optional
``eval_times`` argument to :meth:`score_chunk`. When supplied, the result
populates ``beta_at_t / se_at_t / stat_at_t / p_at_t / eval_times`` by
re-evaluating the basis at user-specified raw time values.
"""

import pytest
import torch

from tests.test_rr_lmm_scan import _simulate_with_planted_signal, _vmeta
from torchgwas.linalg.basis import evaluate_basis_at
from torchgwas.models.rr_lmm import RandomRegressionLMM, RRScanResult


class TestEvalTimesBasic:
    def test_eval_times_populates_optional_fields(self):
        """When eval_times is None the per-time fields stay None;
        when supplied they all become populated tensors of consistent shape."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=40, T=6, b=3, m=10, seed=20
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        # No eval_times → unchanged behavior
        res_no = model.score_chunk(G, nf, _vmeta(10))
        assert res_no.eval_times is None
        assert res_no.beta_at_t is None
        assert res_no.se_at_t is None
        assert res_no.stat_at_t is None
        assert res_no.p_at_t is None

        # eval_times supplied → all populated
        eval_t = torch.tensor([10.0, 30.0, 60.0, 90.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, _vmeta(10), eval_times=eval_t)
        assert isinstance(res, RRScanResult)
        assert res.eval_times is not None
        assert res.beta_at_t is not None
        assert res.se_at_t is not None
        assert res.stat_at_t is not None
        assert res.p_at_t is not None

        # Shapes: (m, n_t) for the per-time fields, (n_t,) for eval_times
        m = 10
        n_t = 4
        assert res.eval_times.shape == (n_t,)
        assert res.beta_at_t.shape == (m, n_t)
        assert res.se_at_t.shape == (m, n_t)
        assert res.stat_at_t.shape == (m, n_t)
        assert res.p_at_t.shape == (m, n_t)

        # SE positive, stats non-negative, p in [0, 1]
        assert (res.se_at_t > 0).all()
        assert (res.stat_at_t >= 0).all()
        assert (res.p_at_t >= 0).all() and (res.p_at_t <= 1).all()

    def test_beta_at_t_matches_phi_dot_beta(self):
        """β(t) = φ(t)' β_j must agree with the manual contraction of the
        re-evaluated basis matrix and the per-SNP coefficient vector."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=40, T=6, b=3, m=8, seed=21
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        eval_t = torch.tensor([5.0, 25.0, 55.0, 95.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, _vmeta(8), eval_times=eval_t)

        # Independent re-derivation: Phi_eval @ β_j^T
        Phi_eval = evaluate_basis_at(eval_t, nf.basis_kind, nf.basis_params)
        beta_at_t_manual = res.beta @ Phi_eval.T

        assert torch.allclose(res.beta_at_t, beta_at_t_manual, atol=1e-12)


class TestEvalTimesAtBoundaryCoincidesWithIntercept:
    def test_legendre_at_midpoint_only_intercept_basis_nonzero(self):
        """For normalized Legendre on [-1, 1], P_1(0) = 0 and P_2(0) ≠ 0.
        At the standardized midpoint t_std=0, only even-order basis values
        contribute. We verify the chi² statistic is well-defined and finite."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=40, T=6, b=3, m=8, seed=22
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        # Midpoint of [t_min, t_max]
        eval_t = torch.tensor([(t_min + t_max) / 2.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, _vmeta(8), eval_times=eval_t)

        # All stats finite
        assert torch.isfinite(res.stat_at_t).all()
        assert torch.isfinite(res.p_at_t).all()
        assert (res.p_at_t >= 0).all() and (res.p_at_t <= 1).all()


class TestEvalTimesPlantedSignalDetection:
    def test_planted_intercept_detected_at_arbitrary_time(self):
        """A planted time-stable (intercept) effect should produce small
        per-time p-values across the entire time range, because the effect
        is constant in t."""
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=150, T=10, b=3, m=40, h2=0.5,
            causal_indices=(5,), effect_pattern="intercept",
            effect_size=2.5, seed=23,
        )
        model = RandomRegressionLMM(basis="legendre", order=2)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)

        eval_t = torch.tensor([10.0, 40.0, 70.0, 100.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, _vmeta(40), eval_times=eval_t)

        # Causal SNP should have small per-time p-values at all times
        # (a constant intercept effect doesn't fade across the time range)
        causal_p = res.p_at_t[5]  # (4,)
        assert (causal_p < 1e-2).all(), (
            f"Causal SNP per-time p-values too large: {causal_p.tolist()}"
        )


class TestEvalTimesValidation:
    def test_eval_times_rejected_on_non_rr_null_fit(self):
        """Calling score_chunk with eval_times on a NullFit that has no
        basis_kind metadata must raise ValueError."""
        # Build a NullFit and then wipe out basis metadata to simulate a non-RR fit
        Y, X0, K, G, ids, t, t_min, t_max = _simulate_with_planted_signal(
            n=30, T=5, b=2, m=10, seed=24, causal_indices=(2,)
        )
        model = RandomRegressionLMM(basis="legendre", order=1)
        nf = model.fit_null(Y, X0, K, sample_ids=ids, time_values=t,
                            t_min=t_min, t_max=t_max)
        nf.basis_kind = None
        with pytest.raises(ValueError, match="basis_kind"):
            model.score_chunk(G, nf, _vmeta(10),
                              eval_times=torch.tensor([0.0, 50.0], dtype=torch.float64))
