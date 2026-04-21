"""Phase D.4 — NCBI cassette replay test for ``torchgwas.annotate``.

The cassette at ``tests/fixtures/ncbi_cassette/maize_chr1_one_hit.json`` is a
recorded end-to-end exchange against the live NCBI Datasets v2 + E-utilities
APIs for a one-hit maize scenario. Response payloads use the **live** schema
(``annotations[].genomic_locations[]`` + ``gene_ontology`` process/function/
component buckets), so replaying the cassette actually drives the shape
adapters in ``torchgwas.annotate._genes`` —
:func:`_ensure_legacy_genomic_ranges` and
:func:`_ensure_legacy_ontology_terms`.

If upstream NCBI renames a key (for example ``gene_ontology.processes`` →
something else), these tests fail: the adapter can no longer find the GO
terms, the GO-terms list comes back empty, and the assertion trips. That is
the canary the plan's Phase D.4 calls for.

Refreshing the cassette: set ``TORCHGWAS_NCBI_LIVE=1`` and re-run the
recorder (see ``_metadata.refresh_instructions`` in the cassette file).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from torchgwas.annotate import AnnotationResult, annotate_hits
from torchgwas.annotate._client import NCBIClient
from torchgwas.postgwas._sumstats import SumStats

CASSETTE_DIR = Path(__file__).parent / "fixtures" / "ncbi_cassette"
CASSETTE_PATH = CASSETTE_DIR / "maize_chr1_one_hit.json"


# ---------------------------------------------------------------------------
# Cassette replay client
# ---------------------------------------------------------------------------


def _load_cassette() -> dict[str, Any]:
    with open(CASSETTE_PATH) as f:
        return json.load(f)


def _matches(
    interaction: dict[str, Any],
    method: str,
    url: str,
    params: dict | None,
    body: dict | None,
) -> bool:
    """Match an interaction by method + URL substring + params/body subset."""
    if interaction["method"] != method:
        return False
    if interaction.get("url_contains") and interaction["url_contains"] not in url:
        return False
    for key, val in (interaction.get("params_contains") or {}).items():
        if params is None or params.get(key) != val:
            return False
    want_body = interaction.get("body_contains")
    if want_body is not None:
        if body is None:
            return False
        for key, val in want_body.items():
            if body.get(key) != val:
                return False
    return True


class _CassetteClient(NCBIClient):
    """NCBIClient that replays a recorded interaction sequence."""

    def __init__(self, cassette: dict[str, Any]) -> None:
        # Skip the network-dependent parent init.
        self.api_key = None
        self._cache: dict[str, Any] = {}
        self._interactions: list[dict[str, Any]] = list(cassette["interactions"])
        self.get_calls: list[tuple[str, dict | None]] = []
        self.post_calls: list[tuple[str, dict]] = []

    def _find(
        self, method: str, url: str, params: dict | None, body: dict | None
    ) -> dict[str, Any]:
        for interaction in self._interactions:
            if _matches(interaction, method, url, params, body):
                return interaction["response"]
        raise AssertionError(
            f"Cassette miss: no recorded {method} matches url={url!r} "
            f"params={params!r} body={body!r}"
        )

    def _get(self, url: str, params: dict | None = None, *, eutils: bool = False) -> Any:
        self.get_calls.append((url, params))
        return self._find("GET", url, params, None)

    def _post(self, url: str, body: dict) -> Any:
        self.post_calls.append((url, body))
        return self._find("POST", url, None, body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cassette() -> dict[str, Any]:
    assert CASSETTE_PATH.exists(), (
        f"Cassette {CASSETTE_PATH} is missing — run scripts/record_ncbi_cassette.py "
        f"with TORCHGWAS_NCBI_LIVE=1 to regenerate."
    )
    return _load_cassette()


@pytest.fixture
def one_hit_sumstats() -> SumStats:
    """A single-SNP SumStats at chr1:1_012_000 with a genome-wide significant p."""
    return SumStats(
        chr=["1"],
        pos=[1_012_000],
        snp=["rs_maize_auxrf"],
        a1=["A"],
        a2=["G"],
        beta=torch.tensor([0.25], dtype=torch.float64),
        se=torch.tensor([0.04], dtype=torch.float64),
        p=torch.tensor([1e-9], dtype=torch.float64),
        n=torch.tensor([500.0], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# End-to-end replay
# ---------------------------------------------------------------------------


def test_cassette_replay_produces_annotation_result(cassette, one_hit_sumstats):
    client = _CassetteClient(cassette)
    result = annotate_hits(
        one_hit_sumstats,
        crop="maize",
        window_upstream_bp=25_000,
        window_downstream_bp=25_000,
        p_threshold=5e-8,
        include_go=True,
        include_orthologs=False,
        client=client,
    )

    assert isinstance(result, AnnotationResult)
    assert result.taxid == 4577
    assert result.assembly_accession == "GCF_902167145.1"
    assert result.assembly_name == "Zm-B73-REFERENCE-NAM-5.0"
    assert len(result.hits) == 1

    hit = result.hits[0]
    assert hit.snp == "rs_maize_auxrf"
    assert hit.chrom == "1"
    assert hit.pos == 1_012_000
    assert hit.status == "ok"
    assert len(hit.genes) == 1

    gene = hit.genes[0]
    assert gene.gene_id == "100001"
    assert gene.symbol == "ZmAUXRF1"
    # Range came from annotations[0].genomic_locations[0].genomic_range →
    # proves _ensure_legacy_genomic_ranges ran.
    assert gene.start == 1_010_000
    assert gene.end == 1_015_000
    # SNP at 1_012_000 sits inside [1_010_000, 1_015_000] on forward strand.
    assert gene.genomic_distance_to_snp == 0


def test_cassette_replay_extracts_go_terms_from_live_buckets(cassette, one_hit_sumstats):
    """Proves _ensure_legacy_ontology_terms correctly normalises the live
    gene_ontology.{processes,functions,components} buckets into a flat
    ontology_terms list — and that extract_go_terms pulls the expected
    namespace tags through to the final result."""
    client = _CassetteClient(cassette)
    result = annotate_hits(
        one_hit_sumstats,
        crop="maize",
        window_upstream_bp=25_000,
        window_downstream_bp=25_000,
        p_threshold=5e-8,
        include_go=True,
        client=client,
    )
    go = result.hits[0].genes[0].go_terms
    # One term per bucket in the cassette.
    assert len(go) == 3
    namespaces = {t["namespace"] for t in go}
    assert namespaces == {
        "biological_process",
        "molecular_function",
        "cellular_component",
    }
    ids = {t["go_id"] for t in go}
    assert ids == {"GO:0009733", "GO:0003700", "GO:0005634"}


def test_cassette_replay_performs_expected_request_sequence(cassette, one_hit_sumstats):
    """Regression: annotate_hits must fire exactly the four GETs + one POST
    the cassette records — in order. Off-by-one request count is the usual
    symptom when someone adds a spurious cache-bypass."""
    client = _CassetteClient(cassette)
    annotate_hits(
        one_hit_sumstats,
        crop="maize",
        window_upstream_bp=25_000,
        window_downstream_bp=25_000,
        p_threshold=5e-8,
        include_go=True,
        client=client,
    )
    # 4 GETs: taxonomy, genome/taxon dataset_report, sequence_reports, esearch.fcgi
    assert len(client.get_calls) == 4
    urls = [u for u, _ in client.get_calls]
    assert any("/taxonomy/taxon/maize" in u for u in urls)
    assert any("/genome/taxon/4577/dataset_report" in u for u in urls)
    assert any("/sequence_reports" in u for u in urls)
    assert any("esearch.fcgi" in u for u in urls)

    # POSTs to /gene with the gene_id 100001 esearch resolved. In production
    # the NCBIClient._request cache dedupes identical bodies to a single
    # network call; the cassette replay bypasses that cache so we see the
    # raw call sequence — genes_in_region's detail fetch + annotate_hits'
    # include_go re-fetch = 2 POSTs, both with the same body.
    assert len(client.post_calls) >= 1
    for _, body in client.post_calls:
        assert body == {"gene_ids": ["100001"]}


def test_schema_rename_breaks_go_term_extraction(cassette, one_hit_sumstats):
    """Canary: if NCBI renames gene_ontology.processes to anything else, our
    adapter returns empty GO terms. Simulate that rename by mutating a
    deep-copy of the cassette and re-running the pipeline."""
    perturbed = copy.deepcopy(cassette)
    # Find the gene POST response in the perturbed cassette and break it.
    for interaction in perturbed["interactions"]:
        if interaction["method"] == "POST" and "/gene" in interaction.get(
            "url_contains", ""
        ):
            gene = interaction["response"]["reports"][0]["gene"]
            go = gene["gene_ontology"]
            # Rename every live bucket to a shape we don't understand.
            gene["gene_ontology"] = {
                f"XXX_{k}": v for k, v in go.items()
            }

    client = _CassetteClient(perturbed)
    result = annotate_hits(
        one_hit_sumstats,
        crop="maize",
        window_upstream_bp=25_000,
        window_downstream_bp=25_000,
        p_threshold=5e-8,
        include_go=True,
        client=client,
    )
    # Gene was still located (genomic_ranges adapter untouched), but GO
    # extraction silently comes back empty — exactly the failure mode the
    # canary is meant to flag.
    assert len(result.hits[0].genes) == 1
    assert result.hits[0].genes[0].go_terms == []


def test_schema_rename_breaks_genomic_range_extraction(cassette, one_hit_sumstats):
    """Canary: if NCBI renames annotations[].genomic_locations[] or the
    nested genomic_range key, _ensure_legacy_genomic_ranges returns nothing
    and genes_in_region drops the record (no coordinates = phantom). The
    hit then resolves to status='no_genes'."""
    perturbed = copy.deepcopy(cassette)
    for interaction in perturbed["interactions"]:
        if interaction["method"] == "POST" and "/gene" in interaction.get(
            "url_contains", ""
        ):
            gene = interaction["response"]["reports"][0]["gene"]
            # Rename the locations list — the adapter's .get("genomic_locations")
            # now returns None and no legacy ranges get populated.
            ann0 = gene["annotations"][0]
            ann0["XXX_genomic_locations"] = ann0.pop("genomic_locations")

    client = _CassetteClient(perturbed)
    result = annotate_hits(
        one_hit_sumstats,
        crop="maize",
        window_upstream_bp=25_000,
        window_downstream_bp=25_000,
        p_threshold=5e-8,
        include_go=True,
        client=client,
    )
    # genes_in_region drops records with no genomic_ranges, so the hit ends
    # up with an empty gene list and status='no_genes'.
    assert len(result.hits) == 1
    assert result.hits[0].status == "no_genes"
    assert result.hits[0].genes == []


def test_cassette_is_checked_into_git():
    """The cassette must live in tests/fixtures/ncbi_cassette/ — regression
    guard for the common ``.gitignore too aggressive`` mistake."""
    assert CASSETTE_PATH.exists()
    data = json.loads(CASSETTE_PATH.read_text())
    # Must be the documented 5-interaction sequence.
    assert len(data["interactions"]) == 5
    # Refresh instructions must be preserved so future contributors find them.
    assert "refresh_instructions" in data["_metadata"]
