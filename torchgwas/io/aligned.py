"""SampleAlignedReader: wraps any GenotypeReader with row reindexing.

When phenotype/covariate files have a different sample order or a subset
of the genotype samples, this wrapper applies the reindexing per chunk
without ever materializing the full genotype matrix.
"""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor

from ..models.base import VariantMeta


class SampleAlignedReader:
    """Wraps a GenotypeReader and reindexes rows to aligned sample order.

    Parameters
    ----------
    inner : GenotypeReader
        The underlying reader (PLINK, Zarr, HDF5, etc.).
    keep_indices : list[int]
        Row indices into the inner reader's sample order.  The output
        ``iter_chunks`` yields ``G_chunk[keep_indices, :]``.
    aligned_sample_ids : list[str]
        Sample IDs in the aligned (output) order.
    """

    def __init__(
        self,
        inner,
        keep_indices: list[int],
        aligned_sample_ids: list[str],
    ) -> None:
        self._inner = inner
        self._keep = keep_indices
        self._aligned_ids = aligned_sample_ids

    @property
    def n_samples(self) -> int:
        return len(self._keep)

    @property
    def n_variants(self) -> int:
        return self._inner.n_variants

    @property
    def sample_ids(self) -> list[str]:
        return self._aligned_ids

    @property
    def variant_meta(self) -> VariantMeta:
        return self._inner.variant_meta

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[tuple[Tensor, VariantMeta]]:
        """Yield (G_chunk, variant_meta) with rows reindexed to aligned order."""
        for G_chunk, vmeta in self._inner.iter_chunks(chunk_size):
            yield G_chunk[self._keep, :], vmeta
