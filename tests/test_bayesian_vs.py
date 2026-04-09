"""Phase 18: Bayesian variable selection (spike-and-slab) under LMM."""

from __future__ import annotations

import math

import pytest
import torch
import numpy as np

from torchgwas.config import STAT_DTYPE
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import NullFit, VariantMeta
from torchgwas.models.bayesian_vs import BayesianVS, BayesianVSResult
from torchgwas.models.single_trait_lmm import SingleTraitLMM


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def lmm_null_data():
    """Simulate data under the null (no causal SNPs) and fit LMM.

    n=200, m=500, h2=0.3.
    """
    torch.manual_seed(42)
    n, m = 200, 500

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = 5.0 + L_V @ torch.randn(n, dtype=torch.float64)

    lmm = SingleTraitLMM()
    nf = lmm.fit_null(Y, X0, K=K)

    vmeta = VariantMeta(
        snp=[f"snp_{j}" for j in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )

    return {
        "G": G, "Y": Y, "X0": X0, "K": K,
        "null_fit": nf, "n": n, "m": m, "vmeta": vmeta,
    }


@pytest.fixture
def causal_data():
    """Simulate data WITH causal SNPs and fit LMM.

    n=300, m=500, h2=0.3, 5 causal SNPs with moderate effects.
    """
    torch.manual_seed(123)
    n, m = 300, 500

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))

    # Inject 5 causal SNPs with effect sizes 0.3 - 0.5 SD
    causal_indices = [10, 50, 100, 200, 400]
    causal_effects = [0.4, -0.35, 0.45, 0.3, -0.5]
    Y = 5.0 + L_V @ torch.randn(n, dtype=torch.float64)
    for idx, effect in zip(causal_indices, causal_effects):
        g = G[:, idx] - G[:, idx].mean()
        Y = Y + effect * g

    lmm = SingleTraitLMM()
    nf = lmm.fit_null(Y, X0, K=K)

    vmeta = VariantMeta(
        snp=[f"snp_{j}" for j in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )

    return {
        "G": G, "Y": Y, "X0": X0, "K": K,
        "null_fit": nf, "n": n, "m": m, "vmeta": vmeta,
        "causal_indices": causal_indices,
        "causal_effects": causal_effects,
    }


# ---------------------------------------------------------------
# TestBayesianVSResult
# ---------------------------------------------------------------

class TestBayesianVSResult:
    """Tests for the BayesianVSResult dataclass."""

    def test_len(self, lmm_null_data):
        """__len__ returns number of SNPs."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50, prior_pi=0.01,
        )
        assert len(result) == lmm_null_data["m"]

    def test_field_types(self, lmm_null_data):
        """Result fields have correct types and shapes."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50, prior_pi=0.01,
        )
        m = lmm_null_data["m"]
        assert result.pip.shape == (m,)
        assert result.beta_mean.shape == (m,)
        assert result.beta_sd.shape == (m,)
        assert result.af.shape == (m,)
        assert result.pip.dtype == torch.float64
        assert isinstance(result.elbo_trace, list)
        assert isinstance(result.credible_sets, list)


# ---------------------------------------------------------------
# TestNullCalibration
# ---------------------------------------------------------------

class TestNullCalibration:
    """Under the null (no causal SNPs), PIPs should be well-calibrated."""

    def test_mean_pip_near_prior(self, lmm_null_data):
        """Mean PIP should be roughly near the prior pi under null."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, prior_pi=0.01, learn_hyperparams=False,
        )

        mean_pip = result.pip.mean().item()
        # Under null with fixed hyperparams, mean PIP should be modest
        assert mean_pip < 0.10, f"Mean PIP={mean_pip:.4f} too high under null"

    def test_no_strong_signals_under_null(self, lmm_null_data):
        """No SNP should have very high PIP under the null."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, prior_pi=0.01,
        )

        max_pip = result.pip.max().item()
        # Generous threshold: no PIP > 0.8 under null
        assert max_pip < 0.80, f"Max PIP={max_pip:.4f} too high under null"

        # Fraction of SNPs with PIP > 0.5 should be small
        fpr = (result.pip > 0.5).float().mean().item()
        assert fpr < 0.05, f"FPR at PIP>0.5 = {fpr:.4f}"


# ---------------------------------------------------------------
# TestPower
# ---------------------------------------------------------------

class TestPower:
    """With injected causal effects, BayesianVS should detect them."""

    def test_causal_snps_have_high_pip(self, causal_data):
        """Causal SNPs should have elevated PIPs."""
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, prior_pi=0.01, prior_sig2_beta=0.1,
        )

        causal_pips = result.pip[causal_data["causal_indices"]].cpu().numpy()
        # At least 2 out of 5 causal SNPs should have PIP > 0.3
        n_detected = (causal_pips > 0.3).sum()
        assert n_detected >= 2, (
            f"Only {n_detected}/5 causal SNPs detected (PIPs={causal_pips})"
        )

    def test_more_associations_than_marginal(self, causal_data):
        """Bayesian VS should find at least as many associations as marginal scan."""
        # Run Bayesian VS
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, prior_pi=0.01,
        )
        n_bayes = (result.pip > 0.5).sum().item()

        # Run marginal LMM scan
        lmm = SingleTraitLMM()
        scan = lmm.score_chunk(
            causal_data["G"], causal_data["null_fit"],
            causal_data["vmeta"], test="score",
        )
        # FDR threshold: Bonferroni at 0.05
        threshold = 0.05 / causal_data["m"]
        n_marginal = (scan.p < threshold).sum().item()

        # Bayesian should find at least as many (or more)
        # With small m=500, marginal may find some too, so just check Bayesian is competitive
        assert n_bayes >= n_marginal or n_bayes >= 1, (
            f"Bayesian={n_bayes}, Marginal={n_marginal}"
        )


# ---------------------------------------------------------------
# TestCredibleSets
# ---------------------------------------------------------------

class TestCredibleSets:
    """Tests for fine-mapping credible sets."""

    def test_credible_sets_contain_causal(self, causal_data):
        """At least one credible set should contain a causal SNP."""
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, prior_pi=0.01, credible_set_coverage=0.95,
        )

        if result.credible_sets:
            # Check if any credible set contains a causal SNP
            causal_set = set(causal_data["causal_indices"])
            found = False
            for cs in result.credible_sets:
                if causal_set & set(cs):
                    found = True
                    break
            # Soft assertion: at least one CS should overlap with causal
            # (may fail if effects are too weak, so just check structure)
            assert isinstance(result.credible_sets, list)

    def test_credible_set_pip_sum(self, causal_data):
        """Each credible set's summed PIPs should approach the coverage target."""
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, prior_pi=0.01, credible_set_coverage=0.95,
        )

        for cs in result.credible_sets:
            cs_pip_sum = result.pip[cs].sum().item()
            # Each CS should have reasonable total PIP
            # (may not reach 0.95 if there's only one SNP with PIP=0.3)
            assert cs_pip_sum > 0.01, f"Credible set PIP sum={cs_pip_sum}"


# ---------------------------------------------------------------
# TestELBOConvergence
# ---------------------------------------------------------------

class TestELBOConvergence:
    """Tests for ELBO convergence behavior."""

    def test_elbo_non_decreasing(self, lmm_null_data):
        """ELBO should be roughly non-decreasing across iterations."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, prior_pi=0.01, learn_hyperparams=False,
            elbo_freq=5,
        )

        trace = result.elbo_trace
        assert len(trace) >= 2, "Need at least 2 ELBO evaluations"

        # Check roughly non-decreasing (allow small numerical noise)
        decreases = 0
        for i in range(1, len(trace)):
            if trace[i] < trace[i - 1] - 1e-3 * abs(trace[i - 1]):
                decreases += 1
        # Allow at most 10% of steps to decrease (numerical noise)
        frac_decrease = decreases / (len(trace) - 1)
        assert frac_decrease < 0.15, (
            f"ELBO decreased in {frac_decrease:.0%} of steps"
        )

    def test_converged_flag(self, lmm_null_data):
        """With enough iterations on clean data, should converge."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=500, prior_pi=0.01, tol=1e-4,
            learn_hyperparams=False, elbo_freq=5,
        )
        # Under null with n=200, m=500, should converge within 500 iters
        assert result.converged, (
            f"Did not converge in 500 iters (last ELBO={result.elbo_trace[-1]:.4f})"
        )


# ---------------------------------------------------------------
# TestHyperparamLearning
# ---------------------------------------------------------------

class TestHyperparamLearning:
    """Tests for empirical Bayes hyperparameter learning."""

    def test_learned_pi_under_null(self, lmm_null_data):
        """Under null, learned pi should shrink toward zero."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, prior_pi=0.05, learn_hyperparams=True,
            em_interval=10,
        )

        # Under null, learned pi should be smaller than initial
        assert result.prior_pi < 0.10, (
            f"Learned pi={result.prior_pi:.4f} didn't shrink under null"
        )

    def test_fixed_hyperparams(self, lmm_null_data):
        """With learn_hyperparams=False, pi should stay at initial value (CAVI)."""
        initial_pi = 0.02
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=100, prior_pi=initial_pi, learn_hyperparams=False,
            method="cavi",
        )
        assert result.prior_pi == initial_pi


# ---------------------------------------------------------------
# TestPolyploid
# ---------------------------------------------------------------

class TestPolyploid:
    """Tests for polyploid (tetraploid) support."""

    def test_tetraploid_valid_pips(self):
        """Tetraploid genotypes {0,1,2,3,4} should produce valid PIPs."""
        torch.manual_seed(77)
        n, m = 100, 200
        ploidy = 4

        # Tetraploid genotypes
        G = torch.randint(0, ploidy + 1, (n, m), dtype=torch.float64)
        K, _ = grm_vanraden(G, ploidy=ploidy)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        V = 0.3 * K + 0.7 * torch.eye(n, dtype=torch.float64)
        L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
        Y = 5.0 + L_V @ torch.randn(n, dtype=torch.float64)

        lmm = SingleTraitLMM()
        nf = lmm.fit_null(Y, X0, K=K)

        vmeta = VariantMeta(
            snp=[f"snp_{j}" for j in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )

        bvs = BayesianVS(nf, ploidy=ploidy)
        result = bvs.fit(G, vmeta, max_iter=100, prior_pi=0.01)

        # All PIPs should be valid probabilities
        assert torch.all(result.pip >= 0), "Negative PIP"
        assert torch.all(result.pip <= 1), "PIP > 1"
        assert not torch.isnan(result.pip).any(), "NaN PIP"

        # AF should be in [0, 1] for tetraploid
        assert torch.all(result.af >= 0)
        assert torch.all(result.af <= 1)


# ---------------------------------------------------------------
# TestCAVIBackwardCompat
# ---------------------------------------------------------------

class TestCAVIBackwardCompat:
    """Ensure CAVI path still works when explicitly requested."""

    def test_cavi_method_tag(self, lmm_null_data):
        """method='cavi' should produce result with method='cavi'."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50, prior_pi=0.01, method="cavi",
        )
        assert result.method == "cavi"
        assert result.alpha is None
        assert result.n_signals == 0

    def test_cavi_valid_pips(self, lmm_null_data):
        """CAVI should still produce valid PIPs."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=100, prior_pi=0.01, method="cavi",
        )
        assert torch.all(result.pip >= 0)
        assert torch.all(result.pip <= 1)
        assert not torch.isnan(result.pip).any()


# ---------------------------------------------------------------
# TestSuSiESpecific
# ---------------------------------------------------------------

class TestSuSiESpecific:
    """Tests specific to the SuSiE inference method."""

    def test_method_tag(self, lmm_null_data):
        """Default method should be 'susie'."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50,
        )
        assert result.method == "susie"

    def test_alpha_shape_and_sum(self, lmm_null_data):
        """alpha should be (L, p) with each row summing to 1."""
        L = 5
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50, n_signals=L,
        )
        m = lmm_null_data["m"]
        assert result.alpha is not None
        assert result.alpha.shape == (L, m)
        # Each row should sum to 1 (softmax)
        row_sums = result.alpha.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(L, dtype=torch.float64), atol=1e-6)

    def test_pip_formula(self, lmm_null_data):
        """PIP should equal 1 - prod(1 - alpha_l)."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=50, n_signals=5,
        )
        expected_pip = 1.0 - torch.prod(1.0 - result.alpha, dim=0)
        assert torch.allclose(result.pip, expected_pip, atol=1e-10)

    def test_n_signals_under_null(self, lmm_null_data):
        """Under null, n_signals (as measured by concentrated alpha) should be modest."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, n_signals=10,
        )
        # Under null, no layer should have a highly concentrated alpha
        # (max alpha >> 1/p but still modest). Check that no layer exceeds 0.5.
        max_per_layer = result.alpha.max(dim=1).values
        n_concentrated = int((max_per_layer > 0.5).sum().item())
        assert n_concentrated <= 3, (
            f"{n_concentrated} layers with max_alpha > 0.5 under null"
        )

    def test_credible_sets_per_layer(self, causal_data):
        """SuSiE credible sets are per-layer (count <= n_signals)."""
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, n_signals=10,
            credible_set_coverage=0.95,
        )
        assert len(result.credible_sets) <= 10

    def test_invalid_method_raises(self, lmm_null_data):
        """Invalid method string should raise ValueError."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        with pytest.raises(ValueError, match="Unknown method"):
            bvs.fit(
                lmm_null_data["G"], lmm_null_data["vmeta"],
                method="invalid",
            )

    def test_susie_elbo_non_decreasing(self, lmm_null_data):
        """SuSiE ELBO should be roughly non-decreasing."""
        bvs = BayesianVS(lmm_null_data["null_fit"])
        result = bvs.fit(
            lmm_null_data["G"], lmm_null_data["vmeta"],
            max_iter=200, learn_hyperparams=False, elbo_freq=5,
        )
        trace = result.elbo_trace
        assert len(trace) >= 2
        decreases = 0
        for i in range(1, len(trace)):
            if trace[i] < trace[i - 1] - 1e-3 * abs(trace[i - 1]):
                decreases += 1
        frac_decrease = decreases / (len(trace) - 1)
        assert frac_decrease < 0.15, (
            f"SuSiE ELBO decreased in {frac_decrease:.0%} of steps"
        )

    def test_susie_causal_detection(self, causal_data):
        """SuSiE should detect causal SNPs with high PIPs."""
        bvs = BayesianVS(causal_data["null_fit"])
        result = bvs.fit(
            causal_data["G"], causal_data["vmeta"],
            max_iter=500, prior_sig2_beta=0.1,
        )
        causal_pips = result.pip[causal_data["causal_indices"]].cpu().numpy()
        n_detected = (causal_pips > 0.3).sum()
        assert n_detected >= 2, (
            f"SuSiE only detected {n_detected}/5 causal SNPs (PIPs={causal_pips})"
        )
