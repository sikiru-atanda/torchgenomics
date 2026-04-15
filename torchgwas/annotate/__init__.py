"""NCBI-backed gene annotation for GWAS hit SNPs.

Given a :class:`~torchgwas.postgwas.SumStats` and either a crop common name
(``"maize"``) or an assembly accession (``"GCF_902167145.1"``), look up the
genes overlapping a ±window around each hit SNP and attach descriptions, GO
terms, and orthologs.

This module requires network access. Responses are cached in-process per
:class:`NCBIClient` instance.

Example
-------
>>> from torchgwas.postgwas import load_sumstats
>>> from torchgwas.annotate import annotate_hits
>>> ss = load_sumstats("results.tsv")
>>> result = annotate_hits(
...     ss, crop="maize",
...     window_upstream_bp=50_000, window_downstream_bp=50_000,
...     p_threshold=5e-8, include_go=True,
... )
>>> result.to_tsv("annotated.tsv")
"""

from ._annotate import annotate_hits
from ._client import (
    AssemblyNotFoundError,
    CropNotFoundError,
    NCBIClient,
    NCBIError,
)
from ._genes import gene_details, gene_orthologs, genes_in_region
from ._resolve import fetch_chrom_map, resolve_assembly, resolve_taxid
from ._types import AnnotatedGene, AnnotatedHit, AnnotationResult

__all__ = [
    "annotate_hits",
    "NCBIClient",
    "NCBIError",
    "CropNotFoundError",
    "AssemblyNotFoundError",
    "resolve_taxid",
    "resolve_assembly",
    "fetch_chrom_map",
    "genes_in_region",
    "gene_details",
    "gene_orthologs",
    "AnnotatedGene",
    "AnnotatedHit",
    "AnnotationResult",
]
