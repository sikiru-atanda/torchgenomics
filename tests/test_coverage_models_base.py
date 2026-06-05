"""Tier-1 coverage tests for ``torchgenomics.models.base`` and
``torchgenomics.config``.

Bar (Pillar A spec, foundational types):
- Constructor + attribute round-trip.
- Documented invariants verified.
- For BaseModel: protocol shape — runtime_checkable Protocol; instances
  with ``fit_null`` and ``score_chunk`` satisfy ``isinstance``.
- For NullFit / VariantMeta / TorchGenomicsConfig: dataclass smoke + defaults.

Covers 4 symbols:
- ``torchgenomics.models.base.BaseModel``
- ``torchgenomics.models.base.NullFit``
- ``torchgenomics.models.base.VariantMeta``
- ``torchgenomics.config.TorchGenomicsConfig``
"""

from __future__ import annotations

from typing import Any, Protocol

import pytest
import torch
from torch import Tensor

from torchgenomics.config import (
    AMPConfig,
    NumericalConfig,
    PloidyConfig,
    TorchGenomicsConfig,
)
from torchgenomics.models.base import BaseModel, NullFit, VariantMeta

pytestmark = pytest.mark.timeout(30)


# ---------------------------------------------------------------------------
# models.base.BaseModel (Protocol)
# ---------------------------------------------------------------------------


class _MinimalModel:
    """Minimal class satisfying the BaseModel protocol shape."""

    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor | None = None,
                 **kwargs: Any) -> NullFit:
        return NullFit()

    def score_chunk(self, G_chunk, null_fit, variant_meta, test="wald"):
        raise NotImplementedError


class _MissingFitNull:
    """Class missing fit_null — should not satisfy the protocol."""

    def score_chunk(self, G_chunk, null_fit, variant_meta, test="wald"):
        raise NotImplementedError


class TestBaseModel:
    """``BaseModel`` is a runtime-checkable Protocol with ``fit_null`` +
    ``score_chunk`` as required members."""

    def test_is_protocol(self):
        """``BaseModel`` is a typing.Protocol subclass."""
        # Protocol-derived classes have _is_protocol=True
        assert getattr(BaseModel, "_is_protocol", False) is True
        # And it inherits from Protocol
        assert issubclass(BaseModel, Protocol) or hasattr(BaseModel, "__protocol_attrs__") \
            or hasattr(BaseModel, "_is_runtime_protocol")

    def test_is_runtime_checkable(self):
        """``BaseModel`` is decorated ``@runtime_checkable``: isinstance works."""
        # If not runtime_checkable, isinstance would raise TypeError.
        instance = _MinimalModel()
        # Should not raise
        result = isinstance(instance, BaseModel)
        assert result is True

    def test_subclass_with_required_methods_passes_isinstance(self):
        """A class implementing both methods satisfies the protocol."""
        instance = _MinimalModel()
        assert isinstance(instance, BaseModel)

    def test_class_missing_method_fails(self):
        """A class without ``fit_null`` does not satisfy the protocol at
        runtime, and lacks the required attribute."""
        instance = _MissingFitNull()
        # Lacks the attribute
        assert not hasattr(instance, "fit_null")
        # And the protocol check rejects it
        assert not isinstance(instance, BaseModel)


# ---------------------------------------------------------------------------
# models.base.NullFit
# ---------------------------------------------------------------------------


class TestNullFit:
    """``NullFit`` is a dataclass holding cached null-model quantities;
    every field is optional with a documented default."""

    def test_default_construct(self):
        """Default-constructed NullFit has all fields at None / factory
        defaults: variance components None, optimizer_trace empty list,
        converged False, approximate False."""
        nf = NullFit()
        # Variance components
        assert nf.sig2_g is None
        assert nf.sig2_e is None
        assert nf.Vg is None
        assert nf.Ve is None
        # Eigendecomposition
        assert nf.eigenvalues is None
        assert nf.eigenvectors is None
        # Rotated quantities
        assert nf.Y_rot is None
        assert nf.X0_rot is None
        # Normal equations
        assert nf.M00 is None
        assert nf.b0 is None
        # Weights
        assert nf.weights is None
        # Likelihood
        assert nf.log_likelihood is None
        # Trace + flags
        assert nf.optimizer_trace == []
        assert nf.converged is False
        assert nf.device is None
        # Approximate metadata
        assert nf.approximate is False
        assert nf.approx_method is None
        assert nf.approx_config is None

    def test_optimizer_trace_is_per_instance(self):
        """``optimizer_trace`` uses ``field(default_factory=list)`` so each
        instance gets its own list (no cross-instance aliasing)."""
        nf1 = NullFit()
        nf2 = NullFit()
        nf1.optimizer_trace.append({"iter": 0, "ll": 1.0})
        # If they shared a default, nf2's trace would also be non-empty
        assert nf2.optimizer_trace == []
        assert nf1.optimizer_trace == [{"iter": 0, "ll": 1.0}]

    def test_field_assignment_round_trip(self):
        """Tensors and scalars assigned to fields read back identically."""
        ev = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        nf = NullFit(
            sig2_g=0.5,
            sig2_e=0.3,
            eigenvalues=ev,
            converged=True,
            log_likelihood=-100.5,
        )
        assert nf.sig2_g == 0.5
        assert nf.sig2_e == 0.3
        assert torch.equal(nf.eigenvalues, ev)
        assert nf.converged is True
        assert nf.log_likelihood == -100.5

    def test_repr_does_not_crash(self):
        """``repr(NullFit())`` returns a string containing the type name."""
        s = repr(NullFit())
        assert isinstance(s, str)
        assert "NullFit" in s


# ---------------------------------------------------------------------------
# models.base.VariantMeta
# ---------------------------------------------------------------------------


class TestVariantMeta:
    """``VariantMeta`` is a per-variant annotation dataclass with five list
    fields: snp, chr, pos, a1, a2. ``__len__`` returns the number of
    variants."""

    def test_construct_with_lists(self):
        """Construct with parallel lists of equal length."""
        vm = VariantMeta(
            snp=["rs1", "rs2"],
            chr=["1", "1"],
            pos=[100, 200],
            a1=["A", "C"],
            a2=["G", "T"],
        )
        assert isinstance(vm, VariantMeta)

    def test_field_round_trip(self):
        """Field values read back identically to inputs."""
        vm = VariantMeta(
            snp=["a", "b"],
            chr=["1", "2"],
            pos=[100, 200],
            a1=["A", "A"],
            a2=["G", "C"],
        )
        assert vm.snp == ["a", "b"]
        assert vm.chr == ["1", "2"]
        assert vm.pos == [100, 200]
        assert vm.a1 == ["A", "A"]
        assert vm.a2 == ["G", "C"]

    def test_len_returns_variant_count(self):
        """``len(vm)`` equals ``len(vm.snp)``."""
        vm = VariantMeta(
            snp=["a", "b", "c"],
            chr=["1"] * 3,
            pos=[1, 2, 3],
            a1=["A"] * 3,
            a2=["G"] * 3,
        )
        assert len(vm) == 3

    def test_empty_variant_meta(self):
        """Zero-length lists construct cleanly with len 0."""
        vm = VariantMeta(snp=[], chr=[], pos=[], a1=[], a2=[])
        assert len(vm) == 0

    def test_repr_does_not_crash(self):
        """``repr(VariantMeta(...))`` returns a string."""
        vm = VariantMeta(
            snp=["rs1"], chr=["1"], pos=[100], a1=["A"], a2=["G"],
        )
        s = repr(vm)
        assert isinstance(s, str)
        assert "VariantMeta" in s


# ---------------------------------------------------------------------------
# config.TorchGenomicsConfig
# ---------------------------------------------------------------------------


class TestTorchGenomicsConfig:
    """``TorchGenomicsConfig`` aggregates runtime configuration: device, AMP,
    numerical, ploidy, and streaming knobs."""

    def test_default_construct(self):
        """Defaults: deterministic=False, chunk_size=1024, n_threads=1."""
        cfg = TorchGenomicsConfig()
        assert cfg.deterministic is False
        assert cfg.chunk_size == 1024
        assert cfg.n_threads == 1

    def test_amp_field_is_AMPConfig(self):
        """``cfg.amp`` is an AMPConfig instance with documented defaults."""
        cfg = TorchGenomicsConfig()
        assert isinstance(cfg.amp, AMPConfig)
        assert cfg.amp.enabled is False  # disabled by default

    def test_numerical_field_is_NumericalConfig(self):
        """``cfg.numerical`` is a NumericalConfig instance."""
        cfg = TorchGenomicsConfig()
        assert isinstance(cfg.numerical, NumericalConfig)

    def test_ploidy_config_field_is_PloidyConfig(self):
        """``cfg.ploidy_config`` is a PloidyConfig instance with default
        ploidy=2."""
        cfg = TorchGenomicsConfig()
        assert isinstance(cfg.ploidy_config, PloidyConfig)
        assert cfg.ploidy_config.ploidy == 2

    def test_device_resolved_to_torch_device(self):
        """``cfg.device`` is a torch.device (CPU or CUDA depending on env)."""
        cfg = TorchGenomicsConfig()
        assert isinstance(cfg.device, torch.device)

    def test_ploidy_property_returns_int(self):
        """The ``ploidy`` property returns the resolved integer ploidy."""
        cfg = TorchGenomicsConfig()
        assert cfg.ploidy == 2  # default diploid

    def test_subconfigs_are_per_instance(self):
        """``amp`` / ``numerical`` / ``ploidy_config`` use
        ``default_factory`` so two TorchGenomicsConfig instances don't share
        sub-config objects."""
        cfg1 = TorchGenomicsConfig()
        cfg2 = TorchGenomicsConfig()
        assert cfg1.amp is not cfg2.amp
        assert cfg1.numerical is not cfg2.numerical
        assert cfg1.ploidy_config is not cfg2.ploidy_config

    def test_field_override(self):
        """Constructor accepts overrides for top-level scalar fields."""
        cfg = TorchGenomicsConfig(
            deterministic=True,
            chunk_size=2048,
            n_threads=4,
        )
        assert cfg.deterministic is True
        assert cfg.chunk_size == 2048
        assert cfg.n_threads == 4

    def test_repr_does_not_crash(self):
        """``repr(TorchGenomicsConfig())`` returns a string."""
        s = repr(TorchGenomicsConfig())
        assert isinstance(s, str)
        assert "TorchGenomicsConfig" in s
