"""Annotation tier-1 API: :func:`annotate_hits`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._decorator import tool
from ._helpers import resolve_output_dir, timed
from ._results import AnnotateRun


@tool(
    name="tg_annotate",
    title="Annotate GWAS hits with nearby genes",
    description=(
        "Annotate significant GWAS hits with overlapping or flanking genes "
        "from NCBI Datasets. Specify the target organism by `crop` name "
        "(e.g. 'maize'), `taxid`, or a specific assembly accession. Window "
        "is configurable in bp. Optionally include GO terms and ortholog "
        "mappings to other species. Network-required. Returns AnnotateRun "
        "with .hits and .genes DataFrames."
    ),
    long_running=False,
    category="annotate",
    tags=["annotation", "ncbi", "genes", "go"],
)
def annotate_hits(
    sumstats: str | Path,
    *,
    crop: str | None = None,
    taxid: int | None = None,
    assembly: str | None = None,
    output: str | Path | None = None,
    p_threshold: float = 5e-8,
    window_up: int = 50_000,
    window_down: int = 50_000,
    include_go: bool = True,
    include_orthologs: bool = False,
    ortholog_taxa: list[int] | None = None,
    api_key: str | None = None,
) -> AnnotateRun:
    """Annotate GWAS hits with NCBI gene records.

    At least one of ``crop``, ``taxid``, or ``assembly`` must be provided.
    """
    if not any((crop, taxid, assembly)):
        raise ValueError("annotate_hits requires one of: crop, taxid, assembly.")

    output_dir = resolve_output_dir(output, default_prefix="torchgenomics_annotate")
    output_prefix = output_dir / "annotated"

    with timed() as elapsed:
        from ..annotate import NCBIClient
        from ..annotate import annotate_hits as _annotate
        from ..postgwas import load_sumstats

        ss = load_sumstats(str(sumstats))
        client = NCBIClient(api_key=api_key)
        result = _annotate(
            ss,
            crop=crop,
            taxid=taxid,
            assembly=assembly,
            window_upstream_bp=window_up,
            window_downstream_bp=window_down,
            p_threshold=p_threshold,
            include_go=include_go,
            include_orthologs=include_orthologs,
            ortholog_taxa=ortholog_taxa,
            client=client,
        )
        out_tsv = Path(f"{output_prefix}.genes.tsv")
        result.to_tsv(str(out_tsv))

        # Build hits + genes DataFrames inline
        hits_records: list[dict] = []
        genes_records: list[dict] = []
        for hit in result.hits:
            hits_records.append({
                "snp": hit.snp,
                "chr": hit.chr,
                "pos": hit.pos,
                "p": hit.p,
                "n_genes": len(hit.genes),
            })
            for g in hit.genes:
                genes_records.append({
                    "snp": hit.snp,
                    "gene_id": g.gene_id,
                    "gene_name": g.gene_name,
                    "chr": g.chr,
                    "start": g.start,
                    "end": g.end,
                    "strand": getattr(g, "strand", ""),
                    "distance_bp": getattr(g, "distance_bp", 0),
                })

        hits_df = pd.DataFrame(hits_records)
        genes_df = pd.DataFrame(genes_records)

        return AnnotateRun(
            runtime_s=elapsed(),
            output_files={"genes_tsv": out_tsv},
            n_input_hits=len(result.hits),
            n_genes=len({r["gene_id"] for r in genes_records}),
            crop=crop,
            assembly=result.assembly_accession,
            window_bp=window_up + window_down,
            hits=hits_df,
            genes=genes_df,
        )
