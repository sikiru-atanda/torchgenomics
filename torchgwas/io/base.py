"""GenotypeReader protocol: the contract all format-specific readers implement."""

from __future__ import annotations

from typing import Iterator, Protocol, Tuple

from torch import Tensor

from ..models.base import VariantMeta


class GenotypeReader(Protocol):
    """Yields genotype chunks as (dosage_tensor, variant_metadata) pairs.

    Every format reader (PLINK, VCF, BGEN, HapMap, CSV, …) must implement
    this protocol so the UnifiedScanner can consume them identically.
    """

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        ...

    @property
    def n_variants(self) -> int:
        """Total number of variants."""
        ...

    @property
    def sample_ids(self) -> list[str]:
        """Ordered list of sample IDs (for alignment with phenotype)."""
        ...

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[Tuple[Tensor, VariantMeta]]:
        """Yield (G_chunk, variant_meta) where G_chunk is (n_samples, m) float dosage."""
        ...
