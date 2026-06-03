"""Reference / documentation copy of Pillar A coverage-test fixtures.

NOTE: These fixtures are NOT picked up by pytest from this file. pytest only
auto-discovers fixtures from files literally named ``conftest.py``. The LIVE
copies — the ones consumed by ``tests/test_coverage_*.py`` — are registered
in ``tests/conftest.py`` under the "Pillar A coverage-test fixtures" comment
block.

This file exists so that future readers can find a single, documented place
that lists every coverage-test fixture, its size, and its known-truth
property, without having to scroll through ``tests/conftest.py``.

Sizes are deliberately tiny: every Tier-1 test must complete in < 1 second
so the full coverage suite stays under the 5-minute CI budget.

If you edit a fixture's size / seed / construction here, you MUST also edit
the live copy in ``tests/conftest.py`` (or pytest will silently use the old
live version and the docs will lie).

See ``docs/superpowers/specs/2026-04-30-validation-campaign-design.md`` for
the campaign that introduced these fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def tiny_genotype_diploid():
    """100 samples x 50 SNPs, MAF ~0.3, no missing."""
    rng = np.random.default_rng(42)
    G = rng.binomial(2, 0.3, size=(100, 50)).astype(np.float64)
    return torch.from_numpy(G)


@pytest.fixture(scope="module")
def tiny_genotype_tetraploid():
    """100 samples x 50 SNPs, ploidy=4, no missing."""
    rng = np.random.default_rng(43)
    G = rng.binomial(4, 0.3, size=(100, 50)).astype(np.float64)
    return torch.from_numpy(G)


@pytest.fixture(scope="module")
def tiny_phenotype():
    """Single quantitative trait, n=100, normal."""
    rng = np.random.default_rng(44)
    return torch.from_numpy(rng.normal(0, 1, size=(100, 1)).astype(np.float64))


@pytest.fixture(scope="module")
def tiny_kinship():
    """Symmetric PSD kinship from tiny diploid genotype."""
    rng = np.random.default_rng(42)
    G = rng.binomial(2, 0.3, size=(100, 50)).astype(np.float64)
    Gc = G - G.mean(axis=0, keepdims=True)
    K = Gc @ Gc.T / 50.0
    K = K + np.eye(100) * 1e-6
    return torch.from_numpy(K)


@pytest.fixture(scope="module")
def tiny_covariates():
    """Intercept + one continuous covariate, n=100."""
    rng = np.random.default_rng(45)
    X = np.column_stack([np.ones(100), rng.normal(0, 1, 100)]).astype(np.float64)
    return torch.from_numpy(X)


@pytest.fixture(scope="module")
def stat_dtype():
    """The statistical-inference dtype TorchGenomics uses."""
    return torch.float64
