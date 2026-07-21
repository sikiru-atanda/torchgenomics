"""Tests for the iClass FA-loading clustering module (Smith et al. 2021)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from torchgenomics.postgwas import IClassResult, iclass

# ---------------------------------------------------------------------------
# 1. Basic three-class partition from a 5x2 loading matrix
# ---------------------------------------------------------------------------

def test_basic_three_class_partition():
    """5 environments x 2 factors with polarity pattern PP/PP/PN/NP/NP.

    Expect three iClass clusters: {PP: [E1, E2], PN: [E3], NP: [E4, E5]}.
    """
    Lambda = np.array(
        [
            [0.9, 0.4],   # E1 -> "PP"
            [0.8, 0.3],   # E2 -> "PP"
            [0.7, -0.2],  # E3 -> "PN"
            [-0.6, 0.5],  # E4 -> "NP"
            [-0.5, 0.4],  # E5 -> "NP"
        ]
    )
    env_ids = ["E1", "E2", "E3", "E4", "E5"]
    result = iclass(Lambda, env_ids=env_ids)

    assert isinstance(result, IClassResult)
    assert result.n_envs == 5
    assert result.n_factors == 2
    assert result.threshold == 0.0
    assert result.cluster_labels == ["PP", "PP", "PN", "NP", "NP"]
    assert set(result.cluster_membership.keys()) == {"PP", "PN", "NP"}
    assert result.cluster_membership["PP"] == ["E1", "E2"]
    assert result.cluster_membership["PN"] == ["E3"]
    assert result.cluster_membership["NP"] == ["E4", "E5"]
    # polarity_matrix matches.
    assert result.polarity_matrix.shape == (5, 2)
    assert result.polarity_matrix[0, 0] == "P"
    assert result.polarity_matrix[2, 1] == "N"
    assert result.polarity_matrix[3, 0] == "N"


# ---------------------------------------------------------------------------
# 2. Threshold collapses small loadings to Z
# ---------------------------------------------------------------------------

def test_threshold_introduces_zero_class():
    """Loadings of magnitude 0.01 should map to "Z" once threshold=0.05."""
    Lambda = np.array(
        [
            [0.9, 0.01],   # second factor near zero -> "PZ" w/ threshold
            [0.8, -0.01],  # second factor near zero -> "PZ" w/ threshold
            [0.7, 0.5],    # both real -> "PP"
        ]
    )
    env_ids = ["E1", "E2", "E3"]

    # Without threshold: strict sign produces PP, PN, PP.
    strict = iclass(Lambda, env_ids=env_ids, threshold=0.0)
    assert strict.cluster_labels == ["PP", "PN", "PP"]

    # With threshold=0.05: the tiny entries collapse to Z.
    relaxed = iclass(Lambda, env_ids=env_ids, threshold=0.05)
    assert relaxed.cluster_labels == ["PZ", "PZ", "PP"]
    assert set(relaxed.cluster_membership.keys()) == {"PZ", "PP"}
    assert relaxed.cluster_membership["PZ"] == ["E1", "E2"]
    assert relaxed.cluster_membership["PP"] == ["E3"]
    # polarity matrix carries the Z marker.
    assert relaxed.polarity_matrix[0, 1] == "Z"
    assert relaxed.polarity_matrix[1, 1] == "Z"


# ---------------------------------------------------------------------------
# 3. Label format is exactly the concatenated sign pattern
# ---------------------------------------------------------------------------

def test_polarity_label_format():
    """Cluster labels must be exactly the per-factor P/N/Z string of length k."""
    # k = 3 factors, pick mixed signs and one exact zero.
    Lambda = np.array(
        [
            [0.3, -0.2, 0.5],   # PNP
            [-0.4, 0.6, -0.1],  # NPN
            [0.0, 0.7, -0.8],   # ZPN  (exact zero, threshold=0)
        ]
    )
    res = iclass(Lambda)
    # Default env_ids "env_0", "env_1", "env_2".
    assert res.env_ids == ["env_0", "env_1", "env_2"]
    assert res.cluster_labels == ["PNP", "NPN", "ZPN"]
    # Length of each label == n_factors.
    for label in res.cluster_labels:
        assert len(label) == res.n_factors == 3
        assert set(label).issubset({"P", "N", "Z"})


# ---------------------------------------------------------------------------
# 4. within_cluster_correlation: perfect pair vs independent third
# ---------------------------------------------------------------------------

def test_within_cluster_correlation():
    """Two perfectly correlated envs in a cluster yield rho ~= 1.0;
    a singleton cluster yields NaN."""
    rng = np.random.default_rng(0)
    n_geno = 200
    # Build BLUPs with 3 environments.
    # E0, E1 are perfectly correlated; E2 is independent.
    base = rng.standard_normal(n_geno)
    e0 = base
    e1 = base  # perfectly correlated with E0
    e2 = rng.standard_normal(n_geno)
    blups = np.stack([e0, e1, e2], axis=1)

    # Loadings: E0, E1 share polarity ("PP"); E2 has ("PN").
    Lambda = np.array(
        [
            [0.9, 0.4],   # E0 -> "PP"
            [0.8, 0.3],   # E1 -> "PP"
            [0.7, -0.2],  # E2 -> "PN"
        ]
    )
    res = iclass(Lambda, env_ids=["E0", "E1", "E2"])

    cors = res.within_cluster_correlation(blups)
    assert "PP" in cors
    assert "PN" in cors
    # PP cluster has 2 envs that are identical -> correlation ~ 1.
    assert cors["PP"] == pytest.approx(1.0, abs=1e-12)
    # PN cluster is a singleton -> NaN.
    assert math.isnan(cors["PN"])


# ---------------------------------------------------------------------------
# 5. DataFrame input infers env_ids from the index
# ---------------------------------------------------------------------------

def test_pandas_input_supported():
    """A pandas DataFrame with env IDs as index has env_ids auto-inferred."""
    Lambda_df = pd.DataFrame(
        {
            "F1": [0.9, 0.8, -0.6],
            "F2": [0.4, 0.3, 0.5],
        },
        index=["Yangco", "Bordertown", "Roseworthy"],
    )
    res = iclass(Lambda_df)

    assert res.env_ids == ["Yangco", "Bordertown", "Roseworthy"]
    assert res.n_envs == 3
    assert res.n_factors == 2
    assert res.cluster_labels == ["PP", "PP", "NP"]
    assert res.cluster_membership["PP"] == ["Yangco", "Bordertown"]
    assert res.cluster_membership["NP"] == ["Roseworthy"]
    # loadings field stores a numpy array copy.
    assert isinstance(res.loadings, np.ndarray)
    assert res.loadings.shape == (3, 2)


# ---------------------------------------------------------------------------
# 6. Validation errors
# ---------------------------------------------------------------------------

def test_negative_threshold_raises():
    with pytest.raises(ValueError, match="non-negative"):
        iclass(np.zeros((3, 2)), threshold=-0.1)


def test_env_id_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        iclass(np.zeros((3, 2)), env_ids=["only_two", "ids"])


def test_blups_shape_mismatch_raises():
    res = iclass(np.array([[1.0, 1.0], [-1.0, 1.0]]))
    with pytest.raises(ValueError, match="expected 2"):
        res.within_cluster_correlation(np.zeros((10, 5)))
