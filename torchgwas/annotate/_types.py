"""Dataclasses for NCBI gene annotation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Status codes for AnnotatedHit.
HIT_STATUS_OK = "ok"
HIT_STATUS_NO_GENES = "no_genes"
HIT_STATUS_CHROM_UNRESOLVED = "chrom_unresolved"


@dataclass
class AnnotatedGene:
    """A single gene overlapping the window around a hit SNP.

    ``genomic_distance_to_snp`` is the **strand-agnostic** bp offset between
    the SNP position and the gene's nearest genomic edge, signed by the
    forward-strand coordinate frame: negative = SNP at lower genomic
    coordinate than the gene's start, positive = SNP at higher coordinate
    than the gene's end, zero = SNP inside the gene. **For genes on the
    minus strand this differs from the transcriptional upstream/downstream
    sense** — use ``orientation`` to interpret.
    """

    gene_id: str
    symbol: str
    description: str
    gene_type: str
    chrom: str
    start: int
    end: int
    orientation: str
    genomic_distance_to_snp: int
    go_terms: list[dict[str, Any]] = field(default_factory=list)
    orthologs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnnotatedHit:
    """A hit SNP and the genes found in its ±window.

    ``status`` is one of:
      * ``"ok"`` — the chromosome resolved and at least one gene was found.
      * ``"no_genes"`` — the chromosome resolved but no genes overlapped the window.
      * ``"chrom_unresolved"`` — the chromosome was not found in the assembly; no query was issued.
    """

    snp: str
    chrom: str
    pos: int
    p: float
    status: str = HIT_STATUS_OK
    genes: list[AnnotatedGene] = field(default_factory=list)


@dataclass
class AnnotationResult:
    """Result of `annotate_hits` — assembly metadata plus per-hit gene lists."""

    taxid: int
    assembly_accession: str
    assembly_name: str
    window_upstream_bp: int
    window_downstream_bp: int
    hits: list[AnnotatedHit] = field(default_factory=list)

    def to_dataframe(self):  # type: ignore[no-untyped-def]
        """Flatten to one-row-per-(hit, gene) pandas DataFrame.

        Hits with no overlapping genes still produce one row with empty gene
        fields — the ``status`` column distinguishes ``"no_genes"`` from
        ``"chrom_unresolved"``. GO terms and orthologs are serialised as
        semicolon-joined strings for TSV round-trippability.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for hit in self.hits:
            if not hit.genes:
                rows.append(
                    {
                        "snp": hit.snp,
                        "chrom": hit.chrom,
                        "pos": hit.pos,
                        "p": hit.p,
                        "status": hit.status,
                        "gene_id": "",
                        "symbol": "",
                        "description": "",
                        "gene_type": "",
                        "gene_chrom": "",
                        "gene_start": 0,
                        "gene_end": 0,
                        "orientation": "",
                        "genomic_distance_to_snp": 0,
                        "go_terms": "",
                        "orthologs": "",
                    }
                )
                continue
            for g in hit.genes:
                go_str = ";".join(
                    f"{t.get('go_id', '')}|{t.get('name', '')}|{t.get('namespace', '')}"
                    for t in g.go_terms
                )
                ortho_str = ";".join(
                    f"{o.get('tax_id', '')}:{o.get('symbol', '')}:{o.get('gene_id', '')}"
                    for o in g.orthologs
                )
                rows.append(
                    {
                        "snp": hit.snp,
                        "chrom": hit.chrom,
                        "pos": hit.pos,
                        "p": hit.p,
                        "status": hit.status,
                        "gene_id": g.gene_id,
                        "symbol": g.symbol,
                        "description": g.description,
                        "gene_type": g.gene_type,
                        "gene_chrom": g.chrom,
                        "gene_start": g.start,
                        "gene_end": g.end,
                        "orientation": g.orientation,
                        "genomic_distance_to_snp": g.genomic_distance_to_snp,
                        "go_terms": go_str,
                        "orthologs": ortho_str,
                    }
                )
        return pd.DataFrame(rows)

    def to_tsv(self, path: str) -> None:
        """Write the flattened result to a tab-separated file."""
        self.to_dataframe().to_csv(path, sep="\t", index=False)
