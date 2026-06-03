"""Tier-1 coverage tests for ``torchgenomics.scan.prefetch`` and
``torchgenomics.scan.strategies``.

Bar (Pillar A spec, foundational types):
- Constructor + behavioral round-trip.
- Documented invariants verified.
- For PrefetchIterator: actual prefetch behavior — iteration wraps the
  underlying iterator and yields tensors on the target device.
- For FixedNullStrategy / PerSNPRefitStrategy: marker-class invariants
  (constructible, distinguishable as types).

Covers 4 symbols:
- ``torchgenomics.scan.prefetch.PrefetchIterator``
- ``torchgenomics.scan.prefetch.move_nullfit_to_device``
- ``torchgenomics.scan.strategies.FixedNullStrategy``
- ``torchgenomics.scan.strategies.PerSNPRefitStrategy``
"""

from __future__ import annotations

import pytest
import torch

from torchgenomics.models.base import NullFit, VariantMeta
from torchgenomics.scan.prefetch import PrefetchIterator, move_nullfit_to_device
from torchgenomics.scan.strategies import FixedNullStrategy, PerSNPRefitStrategy


pytestmark = pytest.mark.timeout(30)


def _make_chunk(n: int, m: int, seed: int) -> tuple[torch.Tensor, VariantMeta]:
    """Build a (G, vmeta) tuple with deterministic content."""
    torch.manual_seed(seed)
    G = torch.randn(n, m, dtype=torch.float32)
    vmeta = VariantMeta(
        snp=[f"rs{seed}_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=[1000 * seed + 10 * i for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
    )
    return G, vmeta


# ---------------------------------------------------------------------------
# scan.prefetch.PrefetchIterator
# ---------------------------------------------------------------------------


class TestPrefetchIterator:
    """``PrefetchIterator`` wraps a chunk iterator with optional async
    prefetch. On CPU device, falls back to passthrough (no thread)."""

    def test_yields_underlying_chunks(self):
        """Iterating yields the same (G, vmeta) tuples in order."""
        chunks = [_make_chunk(8, 3, seed) for seed in range(3)]
        it = PrefetchIterator(iter(chunks), device=torch.device("cpu"), n_prefetch=2)
        out = list(it)
        assert len(out) == 3
        for (G_in, vm_in), (G_out, vm_out) in zip(chunks, out):
            assert torch.equal(G_in, G_out)
            assert vm_out.snp == vm_in.snp
            assert vm_out.chr == vm_in.chr
            assert vm_out.pos == vm_in.pos
            assert vm_out.a1 == vm_in.a1
            assert vm_out.a2 == vm_in.a2

    def test_moves_to_target_device(self):
        """Yielded tensors live on the target device. CPU-only verifies the
        passthrough path; CUDA path tested only when available."""
        chunks = [_make_chunk(4, 2, seed=0)]
        target = torch.device("cpu")
        it = PrefetchIterator(iter(chunks), device=target, n_prefetch=2)
        for G_out, _vm in it:
            assert G_out.device == target

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_moves_to_cuda_when_available(self):
        """When device is CUDA, yielded tensors are on CUDA (prefetch path)."""
        chunks = [_make_chunk(4, 2, seed=0)]
        target = torch.device("cuda")
        it = PrefetchIterator(iter(chunks), device=target, n_prefetch=2)
        for G_out, _vm in it:
            assert G_out.device.type == "cuda"

    def test_handles_empty_iter(self):
        """Empty input iterator yields zero items without error."""
        it = PrefetchIterator(iter([]), device=torch.device("cpu"), n_prefetch=2)
        out = list(it)
        assert out == []

    def test_attributes_after_construction(self):
        """Construction stores device and n_prefetch; CPU disables threading."""
        it = PrefetchIterator(iter([]), device=torch.device("cpu"), n_prefetch=4)
        assert it._device == torch.device("cpu")
        assert it._n_prefetch == 4
        assert it._use_prefetch is False  # CPU path is passthrough


# ---------------------------------------------------------------------------
# scan.prefetch.move_nullfit_to_device
# ---------------------------------------------------------------------------


class TestMoveNullfitToDevice:
    """``move_nullfit_to_device`` mutates a NullFit in place, moving every
    tensor field to the target device, and returns the same instance."""

    def test_moves_tensors(self):
        """Tensor fields end up on the target device."""
        nf = NullFit(
            eigenvalues=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            eigenvectors=torch.eye(3, dtype=torch.float64),
            Y_rot=torch.zeros(3, dtype=torch.float64),
        )
        target = torch.device("cpu")
        out = move_nullfit_to_device(nf, target)
        assert out.eigenvalues.device == target
        assert out.eigenvectors.device == target
        assert out.Y_rot.device == target

    def test_returns_nullfit_instance(self):
        """Return value is the (mutated) NullFit instance."""
        nf = NullFit(eigenvalues=torch.tensor([1.0]))
        out = move_nullfit_to_device(nf, torch.device("cpu"))
        assert isinstance(out, NullFit)
        # Implementation mutates and returns the same object
        assert out is nf

    def test_none_fields_remain_none(self):
        """Fields that are None stay None after the move."""
        nf = NullFit(eigenvalues=torch.tensor([1.0]))
        # All other tensor fields default to None
        out = move_nullfit_to_device(nf, torch.device("cpu"))
        assert out.eigenvectors is None
        assert out.Y_rot is None
        assert out.X0_rot is None
        assert out.M00 is None
        assert out.b0 is None
        assert out.weights is None
        assert out.Vg is None
        assert out.Ve is None

    def test_sets_device_attribute(self):
        """After moving, ``null_fit.device`` reflects the target device."""
        nf = NullFit(eigenvalues=torch.tensor([1.0]))
        target = torch.device("cpu")
        out = move_nullfit_to_device(nf, target)
        assert out.device == target

    def test_short_circuit_when_already_on_device(self):
        """If ``null_fit.device`` already matches target, return as-is."""
        target = torch.device("cpu")
        nf = NullFit(eigenvalues=torch.tensor([1.0]), device=target)
        out = move_nullfit_to_device(nf, target)
        assert out is nf
        assert out.device == target


# ---------------------------------------------------------------------------
# scan.strategies.FixedNullStrategy
# ---------------------------------------------------------------------------


class TestFixedNullStrategy:
    """``FixedNullStrategy`` is a marker class identifying the standard GWAS
    flow: fit null once, reuse for all chunks."""

    def test_construct(self):
        """Constructs with no arguments."""
        strat = FixedNullStrategy()
        assert isinstance(strat, FixedNullStrategy)

    def test_distinct_instances(self):
        """Two instances are independent objects."""
        a = FixedNullStrategy()
        b = FixedNullStrategy()
        assert a is not b
        assert isinstance(a, FixedNullStrategy)
        assert isinstance(b, FixedNullStrategy)

    def test_repr_does_not_crash(self):
        """``repr(FixedNullStrategy())`` returns a string."""
        s = repr(FixedNullStrategy())
        assert isinstance(s, str)
        assert "FixedNullStrategy" in s


# ---------------------------------------------------------------------------
# scan.strategies.PerSNPRefitStrategy
# ---------------------------------------------------------------------------


class TestPerSnpRefitStrategy:
    """``PerSNPRefitStrategy`` is a marker class identifying per-SNP
    variance-component refit (expensive, diagnostic)."""

    def test_construct(self):
        """Constructs with no arguments."""
        strat = PerSNPRefitStrategy()
        assert isinstance(strat, PerSNPRefitStrategy)

    def test_distinct_instances(self):
        """Two instances are independent objects."""
        a = PerSNPRefitStrategy()
        b = PerSNPRefitStrategy()
        assert a is not b

    def test_repr_does_not_crash(self):
        """``repr(PerSNPRefitStrategy())`` returns a string."""
        s = repr(PerSNPRefitStrategy())
        assert isinstance(s, str)
        assert "PerSNPRefitStrategy" in s

    def test_differs_from_fixed(self):
        """``PerSNPRefitStrategy`` is a different type than
        ``FixedNullStrategy``; instances of one don't satisfy the other."""
        per_snp = PerSNPRefitStrategy()
        fixed = FixedNullStrategy()
        assert not isinstance(per_snp, FixedNullStrategy)
        assert not isinstance(fixed, PerSNPRefitStrategy)
        assert type(per_snp) is not type(fixed)
