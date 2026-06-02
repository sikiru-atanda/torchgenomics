"""Resolve crop names to taxids, pick reference assemblies, build chrom maps."""

from __future__ import annotations

from ._client import (
    DATASETS_BASE,
    EUTILS_BASE,
    AssemblyNotFoundError,
    CropNotFoundError,
    NCBIClient,
    NCBIError,
)


def resolve_taxid(crop: str, client: NCBIClient) -> int:
    """Resolve a common or scientific name to an NCBI taxonomy ID.

    Tries Datasets v2 `/taxonomy/taxon/{name}` first (handles common names well),
    falls back to E-utilities `esearch` on the `taxonomy` db for edge cases.
    """
    url = f"{DATASETS_BASE}/taxonomy/taxon/{crop}"
    try:
        payload = client._get(url)
    except NCBIError:
        # Only fall back on network / API errors — let parsing bugs surface.
        payload = None

    if payload:
        nodes = payload.get("taxonomy_nodes") or []
        for node in nodes:
            tax = node.get("taxonomy") or {}
            tid = tax.get("tax_id")
            if tid is not None:
                return int(tid)

    esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
    payload = client._get(esearch_url, params={"db": "taxonomy", "term": crop}, eutils=True)
    ids = (payload.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        raise CropNotFoundError(f"Could not resolve '{crop}' to an NCBI taxid.")
    return int(ids[0])


def resolve_assembly(taxid: int, client: NCBIClient) -> tuple[str, str]:
    """Return (assembly_accession, assembly_name) for the reference genome of `taxid`.

    Queries Datasets v2 `/genome/taxon/{taxid}/dataset_report` with the
    `reference_only=true` filter. If no reference is tagged, falls back to
    the first representative / highest-quality assembly returned.
    """
    url = f"{DATASETS_BASE}/genome/taxon/{taxid}/dataset_report"
    payload = client._get(url, params={"filters.reference_only": "true"})
    reports = payload.get("reports") or []

    if not reports:
        # Retry without the reference_only filter to surface any available assembly.
        payload = client._get(url)
        reports = payload.get("reports") or []

    if not reports:
        raise AssemblyNotFoundError(f"No assemblies found for taxid {taxid}.")

    for r in reports:
        info = r.get("assembly_info") or {}
        if info.get("refseq_category") == "reference genome":
            return r["accession"], info.get("assembly_name", "")

    r = reports[0]
    info = r.get("assembly_info") or {}
    return r["accession"], info.get("assembly_name", "")


def fetch_chrom_map(accession: str, client: NCBIClient) -> dict[str, str]:
    """Return ``{chr_name: refseq_accession}`` for every chromosome in the assembly.

    The Datasets gene-search endpoint accepts chromosome names ("1", "X") rather
    than RefSeq accessions, so this map's primary role is to validate user input
    and surface a clean error for typos. It also stores the ``chr_name`` key in
    lowercase-stripped form so both "1" and "chr1" resolve.
    """
    url = f"{DATASETS_BASE}/genome/accession/{accession}/sequence_reports"
    payload = client._get(url)
    reports = payload.get("reports") or []

    out: dict[str, str] = {}
    for r in reports:
        chrom = r.get("chr_name") or r.get("assigned_molecule") or ""
        refseq = r.get("refseq_accession") or r.get("genbank_accession") or ""
        if not chrom:
            continue
        key = _strip_chr_prefix(str(chrom))
        out[key] = refseq
        out[str(chrom)] = refseq
    return out


def _strip_chr_prefix(name: str) -> str:
    """Strip a leading ``chr`` / ``CHR`` / ``Chr`` prefix; preserve the rest verbatim.

    Uses ``startswith`` on the lower-cased name then slices — ``str.lstrip("chr")``
    removes any leading characters in the *set* ``{c, h, r}`` which corrupts names
    like ``"chr"``, ``"hrc1"``, or ``"rr5"``.
    """
    lower = name.lower()
    if lower.startswith("chr"):
        return lower[3:]
    return lower


def normalise_chrom(chrom: str, chrom_map: dict[str, str]) -> str | None:
    """Return the chromosome name NCBI expects, or None if not in the map."""
    if chrom in chrom_map:
        return chrom
    key = _strip_chr_prefix(str(chrom))
    if key in chrom_map:
        return key
    return None
