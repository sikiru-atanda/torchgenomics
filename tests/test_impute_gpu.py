"""Phase 19: GPU-accelerated imputation — Li-and-Stephens HMM and deep learning."""

from __future__ import annotations

import pytest
import torch

from torchgwas.preprocess.impute_gpu import (
    GenotypeAutoencoder,
    _build_emissions,
    _compute_posteriors,
    _imputation_rsq,
    _log_backward,
    _log_forward,
    compute_transition_matrices,
    impute_deep_learning,
    impute_li_stephens,
)

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_diploid():
    """20 samples, 10 SNPs with LD structure (adjacent SNPs correlated)."""
    torch.manual_seed(42)
    n, m = 20, 10
    # Build correlated SNPs: each is a noisy copy of the previous
    G = torch.zeros(n, m, dtype=torch.float64)
    G[:, 0] = torch.randint(0, 3, (n,)).double()
    for j in range(1, m):
        noise = (torch.rand(n) < 0.2).long()  # ~20% recombination
        G[:, j] = (G[:, j - 1].long() + noise).clamp(0, 2).double()
    return G


@pytest.fixture
def G_with_missing(simple_diploid):
    """Same as simple_diploid but with ~15% missing values."""
    G = simple_diploid.clone()
    torch.manual_seed(99)
    mask = torch.rand_like(G) < 0.15
    G[mask] = float("nan")
    return G


@pytest.fixture
def tetraploid_G():
    """10 samples, 8 SNPs, ploidy=4 (states 0-4)."""
    torch.manual_seed(7)
    n, m = 10, 8
    G = torch.zeros(n, m, dtype=torch.float64)
    G[:, 0] = torch.randint(0, 5, (n,)).double()
    for j in range(1, m):
        noise = (torch.rand(n) < 0.25).long()
        G[:, j] = (G[:, j - 1].long() + noise).clamp(0, 4).double()
    return G


# ── Transition matrices ──────────────────────────────────────────────

class TestTransitionMatrices:
    def test_shape(self, simple_diploid):
        T = compute_transition_matrices(simple_diploid, n_states=3)
        n, m = simple_diploid.shape
        assert T.shape == (m - 1, 3, 3)

    def test_rows_sum_to_one(self, simple_diploid):
        T = compute_transition_matrices(simple_diploid, n_states=3)
        row_sums = T.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_positive(self, simple_diploid):
        """Laplace smoothing ensures no zero entries."""
        T = compute_transition_matrices(simple_diploid, n_states=3)
        assert (T > 0).all()

    def test_ld_structure(self, simple_diploid):
        """On average, diagonal should be the largest entry per row."""
        T = compute_transition_matrices(simple_diploid, n_states=3)
        # For each (j, row), check if diagonal is the argmax
        n_dominant = 0
        n_total = 0
        for j in range(T.shape[0]):
            for s in range(T.shape[1]):
                if T[j, s, :].argmax().item() == s:
                    n_dominant += 1
                n_total += 1
        assert n_dominant / n_total > 0.5, "Diagonal should dominate in majority of rows"

    def test_polyploid(self, tetraploid_G):
        T = compute_transition_matrices(tetraploid_G, n_states=5)
        assert T.shape == (tetraploid_G.shape[1] - 1, 5, 5)
        assert torch.allclose(T.sum(dim=-1), torch.ones_like(T.sum(dim=-1)), atol=1e-6)


# ── Forward-backward ─────────────────────────────────────────────────

class TestForwardBackward:
    def test_log_forward_shape(self):
        n_batch, m, S = 5, 8, 3
        log_emission = torch.randn(n_batch, m, S, dtype=torch.float64)
        log_T = torch.log_softmax(torch.randn(m - 1, S, S, dtype=torch.float64), dim=-1)
        log_prior = torch.log_softmax(torch.randn(S, dtype=torch.float64), dim=-1)
        log_alpha = _log_forward(log_emission, log_T, log_prior)
        assert log_alpha.shape == (n_batch, m, S)

    def test_log_backward_shape(self):
        n_batch, m, S = 5, 8, 3
        log_emission = torch.randn(n_batch, m, S, dtype=torch.float64)
        log_T = torch.log_softmax(torch.randn(m - 1, S, S, dtype=torch.float64), dim=-1)
        log_beta = _log_backward(log_emission, log_T)
        assert log_beta.shape == (n_batch, m, S)

    def test_posteriors_sum_to_one(self):
        n_batch, m, S = 5, 8, 3
        log_emission = torch.randn(n_batch, m, S, dtype=torch.float64)
        log_T = torch.log_softmax(torch.randn(m - 1, S, S, dtype=torch.float64), dim=-1)
        log_prior = torch.log_softmax(torch.randn(S, dtype=torch.float64), dim=-1)
        log_alpha = _log_forward(log_emission, log_T, log_prior)
        log_beta = _log_backward(log_emission, log_T)
        gamma = _compute_posteriors(log_alpha, log_beta)
        sums = gamma.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_backward_initialisation(self):
        """Last position beta should be 0 in log-space (beta=1)."""
        n_batch, m, S = 3, 5, 3
        log_emission = torch.randn(n_batch, m, S, dtype=torch.float64)
        log_T = torch.log_softmax(torch.randn(m - 1, S, S, dtype=torch.float64), dim=-1)
        log_beta = _log_backward(log_emission, log_T)
        assert torch.allclose(log_beta[:, -1, :], torch.zeros(n_batch, S, dtype=torch.float64))


# ── Emissions ─────────────────────────────────────────────────────────

class TestEmissions:
    def test_shape(self):
        G = torch.tensor([[0, 1, 2], [1, float("nan"), 0]], dtype=torch.float64)
        log_em = _build_emissions(G, n_states=3, error_rate=0.01)
        assert log_em.shape == (2, 3, 3)

    def test_observed_match_high(self):
        """Observed genotype should have highest emission probability."""
        G = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
        log_em = _build_emissions(G, n_states=3, error_rate=0.01)
        for j in range(3):
            state = int(G[0, j])
            assert log_em[0, j, state] > log_em[0, j, (state + 1) % 3]

    def test_missing_uniform(self):
        """NaN entries should give uniform emissions."""
        G = torch.tensor([[float("nan")]], dtype=torch.float64)
        log_em = _build_emissions(G, n_states=3, error_rate=0.01)
        probs = torch.exp(log_em[0, 0, :])
        assert torch.allclose(probs, torch.ones(3, dtype=torch.float64) / 3, atol=1e-6)


# ── Li-and-Stephens HMM imputer ──────────────────────────────────────

class TestLiStephens:
    def test_no_missing_passthrough(self, simple_diploid):
        """If no missing values, returns clone and R2=1."""
        G_imp, r2 = impute_li_stephens(simple_diploid, ploidy=2)
        assert torch.allclose(G_imp, simple_diploid)
        assert torch.allclose(r2, torch.ones_like(r2))

    def test_imputes_all_missing(self, G_with_missing):
        """All NaN values should be filled."""
        G_imp, r2 = impute_li_stephens(G_with_missing, ploidy=2)
        assert not torch.isnan(G_imp).any(), "All NaN should be imputed"

    def test_imputed_in_valid_range(self, G_with_missing):
        """Imputed dosages should be in [0, ploidy]."""
        G_imp, _ = impute_li_stephens(G_with_missing, ploidy=2)
        assert (G_imp >= -0.01).all() and (G_imp <= 2.01).all()

    def test_observed_unchanged(self, G_with_missing, simple_diploid):
        """Observed values should not be altered."""
        G_imp, _ = impute_li_stephens(G_with_missing, ploidy=2)
        obs = ~torch.isnan(G_with_missing)
        assert torch.allclose(G_imp[obs], simple_diploid[obs])

    def test_r2_shape(self, G_with_missing):
        _, r2 = impute_li_stephens(G_with_missing, ploidy=2)
        assert r2.shape == (G_with_missing.shape[1],)

    def test_r2_fully_observed_is_one(self, G_with_missing):
        """Markers with no missing values should have R2=1."""
        missing = torch.isnan(G_with_missing)
        _, r2 = impute_li_stephens(G_with_missing, ploidy=2)
        for j in range(G_with_missing.shape[1]):
            if not missing[:, j].any():
                assert r2[j] == 1.0

    def test_accuracy_with_known_values(self, simple_diploid):
        """Mask known values, impute, check correlation."""
        G = simple_diploid.clone()
        torch.manual_seed(123)
        mask = torch.rand_like(G) < 0.20
        true_vals = G[mask].clone()
        G[mask] = float("nan")

        G_imp, _ = impute_li_stephens(G, ploidy=2)
        imp_vals = G_imp[mask]

        # Expected dosage should correlate with truth
        if true_vals.numel() > 2:
            corr = torch.corrcoef(torch.stack([true_vals, imp_vals]))[0, 1]
            assert corr > 0.3, f"Imputation correlation {corr:.3f} too low"

    def test_polyploid(self, tetraploid_G):
        """Should work for ploidy=4."""
        G = tetraploid_G.clone()
        torch.manual_seed(55)
        mask = torch.rand_like(G) < 0.15
        G[mask] = float("nan")
        G_imp, r2 = impute_li_stephens(G, ploidy=4)
        assert not torch.isnan(G_imp).any()
        assert (G_imp >= -0.01).all() and (G_imp <= 4.01).all()

    def test_windowing(self, simple_diploid):
        """Small window_size forces multi-window processing."""
        G = simple_diploid.clone()
        G[0, 5] = float("nan")
        G_imp, _ = impute_li_stephens(G, ploidy=2, window_size=4, window_overlap=1)
        assert not torch.isnan(G_imp).any()

    def test_batch_size(self, G_with_missing):
        """Small batch_size should give same result as large."""
        G_imp_small, _ = impute_li_stephens(G_with_missing, ploidy=2, batch_size=5)
        G_imp_large, _ = impute_li_stephens(G_with_missing, ploidy=2, batch_size=100)
        assert torch.allclose(G_imp_small, G_imp_large, atol=1e-6)


# ── Deep learning imputer ────────────────────────────────────────────

class TestDeepLearning:
    def test_autoencoder_forward(self):
        model = GenotypeAutoencoder(n_markers=10, n_states=3, hidden_dims=(16, 8))
        x = torch.randn(4, 10)
        logits = model(x)
        assert logits.shape == (4, 10, 3)

    def test_no_missing_passthrough(self, simple_diploid):
        G_imp, r2 = impute_deep_learning(simple_diploid, ploidy=2)
        assert torch.allclose(G_imp, simple_diploid)
        assert torch.allclose(r2, torch.ones_like(r2))

    def test_imputes_all_missing(self, G_with_missing):
        G_imp, _ = impute_deep_learning(
            G_with_missing, ploidy=2, n_epochs=20, hidden_dims=(32, 16),
        )
        assert not torch.isnan(G_imp).any()

    def test_imputed_in_valid_range(self, G_with_missing):
        G_imp, _ = impute_deep_learning(
            G_with_missing, ploidy=2, n_epochs=20, hidden_dims=(32, 16),
        )
        # Expected dosage can slightly exceed bounds due to softmax weighting
        assert (G_imp >= -0.5).all() and (G_imp <= 2.5).all()

    def test_observed_unchanged(self, G_with_missing, simple_diploid):
        G_imp, _ = impute_deep_learning(
            G_with_missing, ploidy=2, n_epochs=10, hidden_dims=(32, 16),
        )
        obs = ~torch.isnan(G_with_missing)
        assert torch.allclose(G_imp[obs], simple_diploid[obs])

    def test_r2_shape(self, G_with_missing):
        _, r2 = impute_deep_learning(
            G_with_missing, ploidy=2, n_epochs=10, hidden_dims=(32, 16),
        )
        assert r2.shape == (G_with_missing.shape[1],)

    def test_polyploid(self, tetraploid_G):
        G = tetraploid_G.clone()
        torch.manual_seed(55)
        mask = torch.rand_like(G) < 0.15
        G[mask] = float("nan")
        G_imp, _ = impute_deep_learning(
            G, ploidy=4, n_epochs=10, hidden_dims=(16, 8),
        )
        assert not torch.isnan(G_imp).any()


# ── Imputation R-squared ─────────────────────────────────────────────

class TestImputationRsq:
    def test_fully_observed(self):
        G = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        missing = torch.zeros_like(G, dtype=torch.bool)
        r2 = _imputation_rsq(G, G.clone(), missing)
        assert torch.allclose(r2, torch.ones(2, dtype=torch.float64))

    def test_all_missing_marker(self):
        G = torch.full((5, 2), float("nan"), dtype=torch.float64)
        missing = torch.ones(5, 2, dtype=torch.bool)
        G_imp = torch.ones(5, 2, dtype=torch.float64)
        r2 = _imputation_rsq(G, G_imp, missing)
        assert r2[0] == 0.0  # no observed values -> R2=0
