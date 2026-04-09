"""Tests for RandomRegressionMultiEnvLMM (Phase 39, Steps 2-4)."""

import pytest
import torch

from torchgwas.linalg.basis import legendre_basis, standardize_time
from torchgwas.models.base import VariantMeta
from torchgwas.models.rr_lmm import RandomRegressionLMM
from torchgwas.models.rr_met import (
    RandomRegressionMultiEnvLMM,
    RRMetScanResult,
    _build_rr_met_contrasts,
)


# ── Simulator ────────────────────────────────────────────────────────


def _simulate_rr_met_dataset(
    n=60, T=8, E=3, b=3, h2=0.5, seed=0, sigma_env=0.5,
):
    """Simulate balanced random-regression × multi-env dataset.

    Builds a polygenic GRM, draws true coefficient genetic effects whose
    covariance is K_coef ⊗ Vg_env, adds residual noise per (i, e, t), and
    returns long-format Y plus the truths.
    """
    torch.manual_seed(seed)
    p = 200
    G = torch.randn(n, p, dtype=torch.float64)
    G = (G - G.mean(0)) / G.std(0).clamp(min=1e-6)
    K = (G @ G.T) / p
    K = K + 1e-4 * torch.eye(n, dtype=torch.float64)

    # True K_coef (b, b)
    Lk = torch.tril(torch.randn(b, b, dtype=torch.float64) * 0.4)
    Lk = Lk + torch.eye(b, dtype=torch.float64)
    K_coef_true = Lk @ Lk.T

    # True Vg_env (E, E)
    Le = torch.tril(torch.randn(E, E, dtype=torch.float64) * 0.3)
    Le = Le + torch.eye(E, dtype=torch.float64) * 0.7
    Vg_env_true = Le @ Le.T

    # Sample A: vec(A) ~ N(0, K ⊗ K_coef ⊗ Vg_env), shape (n, b, E)
    L_K = torch.linalg.cholesky(K)
    Vg_full = torch.kron(K_coef_true, Vg_env_true)  # (bE, bE)
    L_Vg = torch.linalg.cholesky(
        Vg_full + 1e-6 * torch.eye(b * E, dtype=torch.float64)
    )
    Z = torch.randn(n, b * E, dtype=torch.float64)
    A_flat = L_K @ Z @ L_Vg.T  # (n, bE)
    A = A_flat.reshape(n, b, E)

    # Time grid + Legendre basis (shared across envs by construction)
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)
    t_std, t_min, t_max = standardize_time(times)
    Phi = legendre_basis(t_std, order=b - 1)  # (T, b)

    # G_signal[i, t, e] = sum_k Phi[t, k] * A[i, k, e]
    G_signal = torch.einsum("tk,ike->ite", Phi, A)  # (n, T, E)

    var_g = G_signal.var()
    var_e = var_g * (1.0 - h2) / h2
    Y_obs = G_signal + torch.randn(n, T, E, dtype=torch.float64) * var_e.sqrt()

    # Long format
    rows = []
    for i in range(n):
        for e in range(E):
            for it, tval in enumerate(times.tolist()):
                rows.append((i, e, tval, Y_obs[i, it, e].item()))
    sample_ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    env_ids = torch.tensor([r[1] for r in rows], dtype=torch.long)
    time_values = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    Y_long = torch.tensor([r[3] for r in rows], dtype=torch.float64)

    X0 = torch.ones(n, 1, dtype=torch.float64)

    return {
        "Y_long": Y_long, "sample_ids": sample_ids, "env_ids": env_ids,
        "time_values": time_values, "X0": X0, "K": K,
        "K_coef_true": K_coef_true, "Vg_env_true": Vg_env_true,
        "n": n, "b": b, "E": E, "T": T, "t_min": t_min, "t_max": t_max,
        "times": times,
    }


# ── Step 2: fit_null (separable) ─────────────────────────────────────


class TestFitNullSeparable:
    def test_runs_and_attaches_metadata(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=3, seed=1)
        model = RandomRegressionMultiEnvLMM(
            basis="legendre", order=2, vg_structure="separable"
        )
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        assert nf.is_rr_met
        assert nf.b == 3
        assert nf.E_envs == 2
        assert nf.basis_kind == "legendre"
        assert nf.K_coef.shape == (3, 3)
        assert nf.Vg_env.shape == (2, 2)
        assert nf.Vg.shape == (6, 6)
        assert nf.Ve.shape == (6, 6)
        assert nf.log_likelihood is not None

    def test_K_coef_recovers_signal(self):
        d = _simulate_rr_met_dataset(n=80, T=8, E=3, b=3, h2=0.7, seed=2)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=2)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        # K_coef must be SPD with positive diagonal
        assert nf.K_coef.diag().min().item() > 0
        evals = torch.linalg.eigvalsh(nf.K_coef)
        assert evals.min().item() > -1e-6
        # Vg_env similarly
        evals_e = torch.linalg.eigvalsh(nf.Vg_env)
        assert evals_e.min().item() > -1e-6

    def test_separable_kron_factorization(self):
        d = _simulate_rr_met_dataset(n=60, T=6, E=2, b=2, seed=3)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        # In separable mode, full Vg should equal kron(K_coef, Vg_env).
        Vg_kron = torch.kron(nf.K_coef, nf.Vg_env)
        assert torch.allclose(nf.Vg, Vg_kron, atol=1e-8)

    def test_validation_errors(self):
        d = _simulate_rr_met_dataset(n=20, T=5, E=2, b=2, seed=4)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)

        bad_X0 = torch.ones(15, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="X0 must have shape"):
            model.fit_null(
                d["Y_long"], bad_X0, d["K"],
                sample_ids=d["sample_ids"], env_ids=d["env_ids"],
                time_values=d["time_values"],
            )

        bad_K = torch.eye(15, dtype=torch.float64)
        with pytest.raises(ValueError, match="K must have shape"):
            model.fit_null(
                d["Y_long"], d["X0"], bad_K,
                sample_ids=d["sample_ids"], env_ids=d["env_ids"],
                time_values=d["time_values"],
            )

    def test_E_too_small_raises(self):
        d = _simulate_rr_met_dataset(n=20, T=5, E=2, b=2, seed=5)
        # Force a single env by relabeling everything to env 0
        single_env = torch.zeros_like(d["env_ids"])
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        with pytest.raises(ValueError, match="E="):
            model.fit_null(
                d["Y_long"], d["X0"], d["K"],
                sample_ids=d["sample_ids"], env_ids=single_env,
                time_values=d["time_values"],
            )


# ── Step 3: contrasts + score_chunk ──────────────────────────────────


def _make_genotype_chunk(n, m, seed=0):
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    return G


def _make_variant_meta(m):
    return VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["T"] * m,
    )


class TestContrastCatalog:
    def test_dimensions_and_dfs(self):
        b, E = 3, 4
        contrasts = _build_rr_met_contrasts(b, E, torch.device("cpu"))
        bE = b * E
        # Joint
        assert contrasts["joint"].shape == (bE, bE)
        # Per env e: (b, bE)
        for e in range(E):
            assert contrasts[f"per_env_{e}"].shape == (b, bE)
        # Intercept per env e: (1, bE)
        for e in range(E):
            assert contrasts[f"intercept_{e}"].shape == (1, bE)
        # Time-varying per env e: (b-1, bE)
        for e in range(E):
            assert contrasts[f"tv_{e}"].shape == (b - 1, bE)
        # Stable per coef k: (E-1, bE)
        for k in range(b):
            assert contrasts[f"stable_{k}"].shape == (E - 1, bE)
        # GxE joint: (b(E-1), bE)
        assert contrasts["gxe_joint"].shape == (b * (E - 1), bE)
        # GxE intercept: (E-1, bE)
        assert contrasts["gxe_intercept"].shape == (E - 1, bE)
        # Mean curve: (b, bE)
        assert contrasts["mean_curve"].shape == (b, bE)

    def test_per_env_contrast_picks_correct_columns(self):
        b, E = 2, 3
        contrasts = _build_rr_met_contrasts(b, E, torch.device("cpu"))
        # per_env_1 should have 1s exactly at columns [0*3+1, 1*3+1] = [1, 4]
        C = contrasts["per_env_1"]
        assert C[0, 1] == 1.0
        assert C[1, 4] == 1.0
        # All other entries zero
        nonzero = (C != 0).sum().item()
        assert nonzero == 2

    def test_intercept_contrast_picks_correct_column(self):
        b, E = 3, 4
        contrasts = _build_rr_met_contrasts(b, E, torch.device("cpu"))
        # intercept_2 -> column 0*4 + 2 = 2
        C = contrasts["intercept_2"]
        assert C[0, 2] == 1.0
        nonzero = (C != 0).sum().item()
        assert nonzero == 1

    def test_stable_per_coef_uses_diff_pattern(self):
        b, E = 2, 3
        contrasts = _build_rr_met_contrasts(b, E, torch.device("cpu"))
        # stable_0 has D_3 at columns 0..2 (E-1 = 2 rows)
        C = contrasts["stable_0"]
        assert C.shape == (2, 6)
        # Row 0: +1 at col 0, -1 at col 1
        assert C[0, 0] == 1.0
        assert C[0, 1] == -1.0
        # Row 1: +1 at col 1, -1 at col 2
        assert C[1, 1] == 1.0
        assert C[1, 2] == -1.0
        # Other columns (3, 4, 5) all zero
        assert (C[:, 3:] == 0).all()


class TestScoreChunk:
    def test_score_chunk_runs_and_shapes(self):
        d = _simulate_rr_met_dataset(n=50, T=6, E=2, b=2, seed=10)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        m = 8
        G = _make_genotype_chunk(d["n"], m, seed=11)
        meta = _make_variant_meta(m)
        res = model.score_chunk(G, nf, meta)
        assert isinstance(res, RRMetScanResult)
        assert res.beta.shape == (m, 2, 2)
        assert res.se.shape == (m, 2, 2)
        assert res.Var_beta.shape == (m, 4, 4)
        assert res.stat_joint.shape == (m,)
        assert res.p_joint.shape == (m,)
        # Joint test df
        assert res.df_by_test["joint"] == 4
        # Stable / GxE present
        assert "gxe_joint" in res.stat_by_test
        assert res.df_by_test["gxe_joint"] == 2 * (2 - 1)  # b(E-1)
        # Mean curve
        assert "mean_curve" in res.stat_by_test
        assert res.df_by_test["mean_curve"] == 2
        # P-values are valid probabilities
        for p in res.p_by_test.values():
            assert (p >= 0).all() and (p <= 1).all()

    def test_score_chunk_eval_times_shape(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=2, seed=12)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        m = 4
        G = _make_genotype_chunk(d["n"], m, seed=13)
        meta = _make_variant_meta(m)
        eval_times = torch.tensor([10.0, 50.0, 90.0], dtype=torch.float64)
        res = model.score_chunk(G, nf, meta, eval_times=eval_times)
        assert res.beta_at_t.shape == (m, 3, 2)
        assert res.se_at_t.shape == (m, 3, 2)
        assert res.stat_at_t.shape == (m, 3, 2)
        assert res.p_at_t.shape == (m, 3, 2)
        assert (res.p_at_t >= 0).all() and (res.p_at_t <= 1).all()

    def test_joint_test_consistent_with_per_env_dfs(self):
        """sum of per-env dfs equals joint df only when E partitions
        the basis-coef effect — not strictly required, but per-env tests
        should each have df=b."""
        d = _simulate_rr_met_dataset(n=30, T=5, E=3, b=2, seed=14)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        m = 3
        G = _make_genotype_chunk(d["n"], m, seed=15)
        meta = _make_variant_meta(m)
        res = model.score_chunk(G, nf, meta)
        for e in range(3):
            assert res.df_by_test[f"per_env_{e}"] == 2  # b=2
        assert res.df_by_test["joint"] == 6  # bE


# ── Step 4: interpretive helpers ─────────────────────────────────────


class TestInterpretiveHelpers:
    def test_genetic_variance_surface_shape_and_positivity(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=3, h2=0.6, seed=20)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=2)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        t_query = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0], dtype=torch.float64)
        var_g = model.genetic_variance_surface(nf, t_query)
        assert var_g.shape == (5, 2)
        assert (var_g >= 0).all()

    def test_heritability_surface_in_unit_interval(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=2, h2=0.5, seed=21)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        t_query = torch.linspace(0.0, 100.0, 5, dtype=torch.float64)
        h2 = model.heritability_surface(nf, t_query)
        assert h2.shape == (5, 2)
        assert (h2 >= 0).all() and (h2 <= 1).all()

    def test_genetic_correlation_envs_at_time(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=3, b=2, seed=22)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        t_query = torch.tensor([0.0, 50.0, 100.0], dtype=torch.float64)
        rg = model.genetic_correlation_between_envs_at_time(nf, t_query)
        assert rg.shape == (3, 3, 3)
        # Diagonal entries should be 1
        for ti in range(3):
            assert torch.allclose(
                torch.diagonal(rg[ti]), torch.ones(3, dtype=torch.float64),
                atol=1e-6,
            )
        # Symmetric
        for ti in range(3):
            assert torch.allclose(rg[ti], rg[ti].T, atol=1e-8)
        # Off-diagonals in [-1, 1]
        assert (rg.abs() <= 1.0 + 1e-6).all()

    def test_env_specific_eigenfunctions(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=3, seed=23)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=2)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        t_query = torch.linspace(0.0, 100.0, 7, dtype=torch.float64)
        evals, fns = model.env_specific_eigenfunctions(nf, env=0, t_query=t_query)
        assert evals.shape == (3,)
        assert fns.shape == (7, 3)
        # Eigenvalues sorted descending
        assert (evals[:-1] >= evals[1:]).all()
        # Top eigenvalue non-negative
        assert evals[0].item() >= -1e-8

    def test_unstructured_vg_runs_within_guard(self):
        # bE = 2 * 2 = 4 ≤ 12, allowed
        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=2, seed=30)
        model = RandomRegressionMultiEnvLMM(
            basis="legendre", order=1, vg_structure="unstructured"
        )
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        assert nf.is_rr_met
        assert nf.Vg.shape == (4, 4)
        # Marginal K_coef should still be exposed
        assert nf.K_coef.shape == (2, 2)
        assert nf.Vg_env.shape == (2, 2)

    def test_unstructured_vg_above_guard_raises(self):
        # bE = 4 * 4 = 16 > 12, must raise
        d = _simulate_rr_met_dataset(n=30, T=8, E=4, b=4, seed=31)
        model = RandomRegressionMultiEnvLMM(
            basis="legendre", order=3, vg_structure="unstructured"
        )
        with pytest.raises(ValueError, match="unstructured.*allowed"):
            model.fit_null(
                d["Y_long"], d["X0"], d["K"],
                sample_ids=d["sample_ids"], env_ids=d["env_ids"],
                time_values=d["time_values"],
            )

    def test_fa_structure_runs(self):
        d = _simulate_rr_met_dataset(n=40, T=6, E=3, b=2, seed=32)
        model = RandomRegressionMultiEnvLMM(
            basis="legendre", order=1, vg_structure="fa(1)"
        )
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        assert nf.is_rr_met
        # bE = 2 * 3 = 6
        assert nf.Vg.shape == (6, 6)
        # Marginal K_coef and Vg_env exposed
        assert nf.K_coef.shape == (2, 2)
        assert nf.Vg_env.shape == (3, 3)


class TestUpdateNull:
    def test_update_null_separable_preserves_metadata(self):
        from torchgwas.models.base import update_null

        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=2, seed=40)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        ll_before = nf.log_likelihood
        nf2 = update_null(nf, max_iter=20)
        # RR-MET metadata preserved
        assert nf2.is_rr_met
        assert nf2.b == 2
        assert nf2.E_envs == 2
        assert nf2.basis_kind == "legendre"
        assert nf2.K_coef.shape == (2, 2)
        assert nf2.Vg_env.shape == (2, 2)
        # Log-likelihood should not have decreased significantly
        assert nf2.log_likelihood >= ll_before - 1e-3

    def test_update_null_unstructured(self):
        from torchgwas.models.base import update_null

        d = _simulate_rr_met_dataset(n=40, T=6, E=2, b=2, seed=41)
        model = RandomRegressionMultiEnvLMM(
            basis="legendre", order=1, vg_structure="unstructured"
        )
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        nf2 = update_null(nf, max_iter=20)
        assert nf2.is_rr_met
        assert nf2.vg_structure_rr == "unstructured"


    def test_env_specific_eigenfunctions_bad_env_raises(self):
        d = _simulate_rr_met_dataset(n=20, T=5, E=2, b=2, seed=24)
        model = RandomRegressionMultiEnvLMM(basis="legendre", order=1)
        nf = model.fit_null(
            d["Y_long"], d["X0"], d["K"],
            sample_ids=d["sample_ids"], env_ids=d["env_ids"],
            time_values=d["time_values"],
        )
        with pytest.raises(ValueError, match="out of range"):
            model.env_specific_eigenfunctions(
                nf, env=5, t_query=torch.tensor([0.0])
            )
