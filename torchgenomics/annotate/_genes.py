"""Gene-level NCBI Datasets queries: region lookup, details, orthologs."""

from __future__ import annotations

from typing import Any

from ._client import DATASETS_BASE, EUTILS_BASE, NCBIClient


def genes_in_region(
    taxid: int | str,
    chrom: str,
    start: int,
    stop: int,
    client: NCBIClient,
) -> list[dict[str, Any]]:
    """List gene records overlapping [start, stop] on the given chromosome.

    The Datasets v2 ``annotation_report`` endpoint silently ignores
    ``chromosomes``/``start``/``stop`` filters, and the former
    ``/gene/taxon/<taxid>/search`` path was removed upstream. We therefore use
    E-utilities ``esearch`` with a ``[Base Position]`` range query to resolve
    gene IDs in the window, then batch-fetch full records via the Datasets v2
    ``POST /gene`` endpoint so downstream GO / ortholog / genomic_range parsers
    keep working unchanged.
    """
    if start < 0:
        start = 0
    if stop < start:
        stop = start

    term = (
        f"{int(taxid)}[Taxonomy ID] AND "
        f"{chrom}[Chromosome] AND "
        f"{int(start)}:{int(stop)}[Base Position]"
    )
    esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
    esearch = client._get(
        esearch_url,
        params={"db": "gene", "term": term, "retmax": "1000"},
        eutils=True,
    )
    id_list = (esearch.get("esearchresult") or {}).get("idlist") or []
    if not id_list:
        return []

    details = gene_details(list(id_list), client)
    # gene_details returns a dict keyed by gene_id; preserve esearch order and
    # drop phantom records that Entrez still lists but which no longer carry
    # coordinates in the current assembly annotation.
    out: list[dict[str, Any]] = []
    for gid in id_list:
        rec = details.get(gid)
        if rec is None or not rec.get("genomic_ranges"):
            continue
        out.append(rec)
    return out


def _adapt_annotation_to_gene(ann: dict[str, Any]) -> dict[str, Any]:
    """Translate a v2 annotation_report row into the legacy ``gene`` shape."""
    gene: dict[str, Any] = {
        "gene_id": ann.get("gene_id", ""),
        "symbol": ann.get("symbol", ""),
        # legacy consumers read "description"; annotation_report uses "name".
        "description": ann.get("description") or ann.get("name", ""),
        "type": ann.get("gene_type") or ann.get("type", ""),
        "chromosomes": ann.get("chromosomes") or [],
        "orientation": ann.get("orientation", ""),
    }
    # Map genomic_regions[].gene_range -> genomic_ranges[] (legacy flat shape).
    legacy_ranges: list[dict[str, Any]] = []
    for region in ann.get("genomic_regions") or []:
        gr = region.get("gene_range") or region
        if not gr:
            continue
        legacy_ranges.append({
            "accession_version": gr.get("accession_version", ""),
            "range": gr.get("range") or [],
            "orientation": gr.get("orientation")
            or (gr.get("range") or [{}])[0].get("orientation", ""),
        })
    if legacy_ranges:
        gene["genomic_ranges"] = legacy_ranges
    return gene


def gene_details(gene_ids: list[str], client: NCBIClient) -> dict[str, dict[str, Any]]:
    """Batch-fetch full gene records (incl. GO terms) keyed by gene_id.

    Uses the POST `/gene/id` endpoint to avoid URL-length limits on large hit sets.
    """
    if not gene_ids:
        return {}
    url = f"{DATASETS_BASE}/gene"
    payload = client._post(url, {"gene_ids": [str(i) for i in gene_ids]})
    reports = payload.get("reports") or []
    out: dict[str, dict[str, Any]] = {}
    for r in reports:
        g = r.get("gene") or {}
        gid = g.get("gene_id")
        if gid is None:
            continue
        _ensure_legacy_genomic_ranges(g)
        _ensure_legacy_ontology_terms(g)
        out[str(gid)] = g
    return out


def _ensure_legacy_genomic_ranges(gene: dict[str, Any]) -> None:
    """Populate legacy ``genomic_ranges`` from ``annotations[].genomic_locations``.

    Live Datasets v2 gene records expose coordinates under
    ``annotations[0].genomic_locations[].genomic_range``. Convert to the
    legacy ``genomic_ranges[].range[]`` shape so :func:`genomic_range` works.
    """
    if gene.get("genomic_ranges"):
        return
    annotations = gene.get("annotations") or []
    if not annotations:
        return
    locs = annotations[0].get("genomic_locations") or []
    legacy: list[dict[str, Any]] = []
    for loc in locs:
        gr = loc.get("genomic_range") or {}
        if not gr:
            continue
        legacy.append(
            {
                "accession_version": loc.get("genomic_accession_version", ""),
                "orientation": gr.get("orientation", ""),
                "range": [
                    {
                        "begin": gr.get("begin", "0"),
                        "end": gr.get("end", "0"),
                        "orientation": gr.get("orientation", ""),
                    }
                ],
            }
        )
    if legacy:
        gene["genomic_ranges"] = legacy


def _ensure_legacy_ontology_terms(gene: dict[str, Any]) -> None:
    """Populate legacy ``ontology_terms`` from live ``gene_ontology`` buckets."""
    if gene.get("ontology_terms"):
        return
    go = gene.get("gene_ontology") or {}
    # Live shape: {"processes": [...], "functions": [...], "components": [...]}
    buckets = {
        "biological_process": go.get("processes") or [],
        "molecular_function": go.get("functions") or [],
        "cellular_component": go.get("components") or [],
    }
    terms: list[dict[str, Any]] = []
    for namespace, items in buckets.items():
        for t in items:
            terms.append(
                {
                    "go_id": t.get("go_id") or t.get("id") or "",
                    "name": t.get("name") or "",
                    "namespace": namespace,
                    "evidence_code": t.get("evidence_code") or "",
                }
            )
    if terms:
        gene["ontology_terms"] = terms


def extract_go_terms(gene_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull GO terms out of a Datasets gene record in a normalised shape."""
    terms = gene_record.get("ontology_terms") or []
    out: list[dict[str, Any]] = []
    for t in terms:
        out.append(
            {
                "go_id": t.get("go_id") or t.get("id") or "",
                "name": t.get("name") or "",
                "namespace": t.get("namespace") or t.get("type") or "",
                "evidence_code": t.get("evidence_code") or "",
            }
        )
    return out


def gene_orthologs(
    gene_id: str,
    client: NCBIClient,
    taxon_filter: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Fetch orthologs for a single gene_id; optionally filter to a taxid list."""
    url = f"{DATASETS_BASE}/gene/id/{gene_id}/orthologs"
    params: dict[str, str] = {}
    if taxon_filter:
        params["taxon_filter"] = ",".join(str(t) for t in taxon_filter)
    payload = client._get(url, params=params or None)
    reports = payload.get("reports") or []
    out: list[dict[str, Any]] = []
    for r in reports:
        g = r.get("gene") or {}
        if not g:
            continue
        out.append(
            {
                "gene_id": str(g.get("gene_id", "")),
                "symbol": g.get("symbol", ""),
                "tax_id": int(g.get("tax_id", 0)) if g.get("tax_id") else 0,
                "taxname": g.get("taxname", ""),
                "description": g.get("description", ""),
            }
        )
    return out


def genomic_range(gene_record: dict[str, Any]) -> tuple[str, int, int, str]:
    """Extract (chrom, start, end, orientation) from a Datasets gene record.

    Datasets returns ``genomic_ranges[].range[]`` with 1-based begin/end strings
    and an ``orientation`` of "plus" / "minus". Falls back to (empty, 0, 0, "")
    if the structure is missing — callers should treat that as "unknown" rather
    than skip the gene (description may still be useful).
    """
    ranges = gene_record.get("genomic_ranges") or []
    if not ranges:
        return "", 0, 0, ""
    r0 = ranges[0]
    chrom = r0.get("accession_version") or r0.get("chr") or ""
    inner = r0.get("range") or []
    orientation = r0.get("orientation") or ""
    if not inner:
        return str(chrom), 0, 0, str(orientation)
    begin = int(inner[0].get("begin", 0) or 0)
    end = int(inner[0].get("end", 0) or 0)
    if begin > end:
        begin, end = end, begin
    return str(chrom), begin, end, str(orientation)
