"""Top-level `annotate_hits` — take a SumStats, return per-hit gene annotations."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch

from ._client import NCBIClient
from ._genes import (
    extract_go_terms,
    gene_details,
    gene_orthologs,
    genes_in_region,
    genomic_range,
)
from ._resolve import fetch_chrom_map, normalise_chrom, resolve_assembly, resolve_taxid
from ._types import (
    HIT_STATUS_CHROM_UNRESOLVED,
    HIT_STATUS_NO_GENES,
    HIT_STATUS_OK,
    AnnotatedGene,
    AnnotatedHit,
    AnnotationResult,
)

if TYPE_CHECKING:
    from ..postgwas._sumstats import SumStats

logger = logging.getLogger("torchgwas.annotate")


def _signed_distance(snp_pos: int, gene_start: int, gene_end: int) -> int:
    """Return 0 if SNP falls inside the gene, else signed bp to the nearest edge.

    Uses the **forward-strand coordinate frame** (strand-agnostic): negative
    means ``snp_pos < gene_start``, positive means ``snp_pos > gene_end``.
    This is *not* the transcriptional upstream/downstream sense for minus-strand
    genes — callers who need that distinction should consult ``orientation``.
    """
    if gene_start <= snp_pos <= gene_end:
        return 0
    if snp_pos < gene_start:
        return snp_pos - gene_start
    return snp_pos - gene_end


def annotate_hits(
    ss: "SumStats",
    *,
    crop: str | None = None,
    taxid: int | None = None,
    assembly: str | None = None,
    window_upstream_bp: int = 50_000,
    window_downstream_bp: int = 50_000,
    p_threshold: float = 5e-8,
    include_go: bool = True,
    include_orthologs: bool = False,
    ortholog_taxa: list[int] | None = None,
    api_key: str | None = None,
    client: NCBIClient | None = None,
) -> AnnotationResult:
    """Annotate GWAS hits with nearby genes via NCBI.

    Filters SNPs in ``ss`` to those with ``p <= p_threshold``, queries NCBI
    Datasets v2 for each hit's ±window, and attaches gene descriptions plus
    optional GO terms and orthologs.

    You must provide at least one of ``crop``, ``taxid``, or ``assembly``:

    * ``crop`` → resolved to ``taxid`` via Datasets taxonomy; assembly discovered automatically.
    * ``taxid`` → skips the taxonomy round-trip (use this in CLI/scripts when you already know it).
    * ``assembly`` → used directly; still needs ``crop`` or ``taxid`` to query genes.

    Parameters
    ----------
    ss : SumStats
        Summary statistics, loaded via :func:`torchgwas.postgwas.load_sumstats`.
    crop : str or None
        Common or scientific name (e.g. ``"maize"``, ``"Zea mays"``).
    taxid : int or None
        NCBI taxonomy ID. Takes precedence over ``crop`` for taxid resolution.
    assembly : str or None
        Assembly accession (e.g. ``"GCF_902167145.1"``). Bypasses the
        reference-genome lookup if supplied.
    window_upstream_bp, window_downstream_bp : int
        Window around each hit (bp). Must be non-negative. Combined into
        ``[pos - up, pos + down]``.
    p_threshold : float
        Only SNPs with ``p <= p_threshold`` are queried.
    include_go : bool
        If True, pull GO terms per gene (one extra batched request per hit set).
    include_orthologs : bool
        If True, pull orthologs per gene (one request per unique gene; slow).
    ortholog_taxa : list[int] or None
        Optional filter passed to the ortholog endpoint.
    api_key : str or None
        NCBI API key. If None, read from ``NCBI_API_KEY``.
    client : NCBIClient or None
        Inject a pre-built / mocked client. If None, a default ``NCBIClient``
        is constructed (live network).

    Returns
    -------
    AnnotationResult
    """
    if crop is None and taxid is None and assembly is None:
        raise ValueError(
            "annotate_hits requires at least one of `crop`, `taxid`, or `assembly`."
        )
    if window_upstream_bp < 0 or window_downstream_bp < 0:
        raise ValueError("Window sizes must be non-negative.")

    if client is None:
        client = NCBIClient(api_key=api_key)

    # Resolve taxid: explicit > crop > (fail if assembly-only).
    if taxid is not None:
        resolved_taxid = int(taxid)
    elif crop is not None:
        resolved_taxid = resolve_taxid(crop, client)
    else:
        raise ValueError(
            "When passing `assembly` without `crop`, you must also provide `taxid` "
            "(NCBI gene lookup is taxid-scoped)."
        )

    if assembly is None:
        acc, asm_name = resolve_assembly(resolved_taxid, client)
    else:
        acc = assembly
        # Best-effort assembly-name lookup; failure is non-fatal.
        try:
            _, asm_name = resolve_assembly(resolved_taxid, client)
        except Exception:  # noqa: BLE001 — non-fatal enrichment
            asm_name = ""

    chrom_map = fetch_chrom_map(acc, client)

    # Filter to hit SNPs. Defensive: accept tensors, numpy arrays, or lists.
    p_tensor = torch.as_tensor(ss.p).detach().cpu()
    p_arr = p_tensor.tolist()
    hit_indices = [i for i, pv in enumerate(p_arr) if pv <= p_threshold]
    logger.info(
        "annotate_hits: %d / %d SNPs pass p <= %.3g",
        len(hit_indices),
        len(p_arr),
        p_threshold,
    )

    hits: list[AnnotatedHit] = []
    for i in hit_indices:
        snp_chr = str(ss.chr[i])
        snp_pos = int(ss.pos[i])
        snp_id = str(ss.snp[i])
        snp_p = float(p_arr[i])

        chrom = normalise_chrom(snp_chr, chrom_map)
        if chrom is None:
            warnings.warn(
                f"Chromosome '{snp_chr}' not found in assembly {acc}; skipping SNP {snp_id}.",
                stacklevel=2,
            )
            hits.append(
                AnnotatedHit(
                    snp=snp_id,
                    chrom=snp_chr,
                    pos=snp_pos,
                    p=snp_p,
                    status=HIT_STATUS_CHROM_UNRESOLVED,
                    genes=[],
                )
            )
            continue

        start = max(0, snp_pos - int(window_upstream_bp))
        stop = snp_pos + int(window_downstream_bp)

        region_genes = genes_in_region(resolved_taxid, chrom, start, stop, client)

        details_map: dict[str, dict] = {}
        if include_go and region_genes:
            ids_needing_detail = [
                str(g["gene_id"]) for g in region_genes if "gene_id" in g
            ]
            details_map = gene_details(ids_needing_detail, client)

        annotated_genes: list[AnnotatedGene] = []
        for g in region_genes:
            gid = str(g.get("gene_id", ""))
            if not gid:
                continue

            chrom_g, gstart, gend, orient = genomic_range(g)
            detail = details_map.get(gid, g)

            go = extract_go_terms(detail) if include_go else []
            ortho = (
                gene_orthologs(gid, client, taxon_filter=ortholog_taxa)
                if include_orthologs
                else []
            )

            annotated_genes.append(
                AnnotatedGene(
                    gene_id=gid,
                    symbol=str(g.get("symbol", "")),
                    description=str(g.get("description", "")),
                    gene_type=str(g.get("type", "")),
                    chrom=chrom_g or chrom,
                    start=gstart,
                    end=gend,
                    orientation=orient,
                    genomic_distance_to_snp=(
                        _signed_distance(snp_pos, gstart, gend) if gend > 0 else 0
                    ),
                    go_terms=go,
                    orthologs=ortho,
                )
            )

        status = HIT_STATUS_OK if annotated_genes else HIT_STATUS_NO_GENES
        hits.append(
            AnnotatedHit(
                snp=snp_id,
                chrom=snp_chr,
                pos=snp_pos,
                p=snp_p,
                status=status,
                genes=annotated_genes,
            )
        )

    return AnnotationResult(
        taxid=int(resolved_taxid),
        assembly_accession=acc,
        assembly_name=str(asm_name),
        window_upstream_bp=int(window_upstream_bp),
        window_downstream_bp=int(window_downstream_bp),
        hits=hits,
    )
