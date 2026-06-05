"""LD reference metadata schema and cohort-mismatch detection.

Per the NA1 design spec (docs/superpowers/specs/2026-05-11-na1-susie-streaming-design.md)
section 3.1 and risk R-NA1-1: every LD reference file must carry metadata
stamping cohort_id, n, build, panel_provenance. At bayes-scan-rss invocation,
the LD ref's metadata is checked against the sumstats; soft mismatch
(different cohort_id) → UserWarning; hard mismatch (different build) →
MetadataMismatchError.

Reference: Zou, Carbonetto, Wang, Stephens (2022) PLOS Genet 18(7):e1010299
section 6 — LD-ref-mismatch is the primary failure mode of SuSiE-RSS.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping

SCHEMA_VERSION = 1


class MetadataMismatchError(ValueError):
    """Raised when LD reference metadata is incompatible with sumstats metadata."""


@dataclass(frozen=True)
class LDReferenceMetadata:
    """Metadata stamped into every LD reference file at build time.

    Args:
        cohort_id: Free-form identifier for the cohort the LD was built from
            (e.g., "UKB_unrelated_British", "MDP", "FinnGen_R10").
        n: Sample size used to build the LD matrix.
        build: Genome build identifier (e.g., "GRCh37", "GRCh38", "B73_v4").
        panel_provenance: Free-form description of the variant panel
            (e.g., "UKB v3 imputed", "1KG phase 3 EUR").
    """
    cohort_id: str
    n: int
    build: str
    panel_provenance: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LDReferenceMetadata":
        if d.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError(
                f"LDReferenceMetadata schema version mismatch: "
                f"expected {SCHEMA_VERSION}, got {d.get('schema_version')}"
            )
        return cls(
            cohort_id=d["cohort_id"],
            n=int(d["n"]),
            build=d["build"],
            panel_provenance=d["panel_provenance"],
        )


def check_metadata_compatibility(
    ref: LDReferenceMetadata,
    ss_meta: Mapping[str, Any],
) -> None:
    """Check LD reference metadata against sumstats metadata.

    Soft mismatch (different cohort_id, missing sumstats metadata, etc.)
    emits a UserWarning. Hard mismatch (different build) raises
    MetadataMismatchError.

    Args:
        ref: LD reference metadata.
        ss_meta: Sumstats metadata as a mapping (typically loaded from a
            sumstats sidecar file or extracted from sumstats column headers).

    Raises:
        MetadataMismatchError: on hard mismatch (different build).
    """
    if not ss_meta:
        warnings.warn(
            "sumstats has no metadata; cannot verify LD reference compatibility. "
            "Recommend stamping cohort_id and build into the sumstats sidecar.",
            UserWarning,
            stacklevel=2,
        )
        return

    ref_build = ref.build
    ss_build = ss_meta.get("build")
    if ss_build is not None and ss_build != ref_build:
        raise MetadataMismatchError(
            f"build mismatch: LD reference is {ref_build!r}, sumstats is {ss_build!r}. "
            f"Re-build the LD reference on the same genome build as the sumstats."
        )

    ref_cohort = ref.cohort_id
    ss_cohort = ss_meta.get("cohort_id")
    if ss_cohort is not None and ss_cohort != ref_cohort:
        warnings.warn(
            f"cohort_id mismatch: LD reference cohort is {ref_cohort!r}, "
            f"sumstats cohort is {ss_cohort!r}. SuSiE-RSS is sensitive to "
            f"LD reference cohort matching the GWAS cohort (Zou et al. 2022 "
            f"PLOS Genet section 6). Posteriors may be miscalibrated.",
            UserWarning,
            stacklevel=2,
        )
