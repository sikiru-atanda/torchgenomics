"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20


@dataclass
class PhasingResult:
    haplotypes: Tensor                     # (n_offspring, ploidy, m) int8 — argmax of origin_probs
    origin_probs: Tensor                   # (n_offspring, ploidy, m, n_parent_haps) float64
    parent_phased: Tensor                  # (n_parents, max_ploidy, m) int8
    offspring_ids: list[str]
    parent_ids: list[str]
    variant_ids: list[str]
    chrom: list[str]
    pos_bp: Tensor                         # (m,) int64
    pos_cm: Tensor                         # (m,) float64
    per_individual_ploidy: dict[str, int]
    map_refined: bool
    valent_diag: pd.DataFrame
    postdose_probs: Tensor                 # (n_offspring, m, max_ploidy+1) float64
    tool: str
    tool_version: str
    input_hash: str
    cmd: str
    workdir: str | None


def _build_polyorigin_pedfile(
    user_tsv: str,
    ploidy_default: int,
    sample_ids_in_probs: set[str],
    workdir: str,
) -> Path:
    """Convert user 3-col pedigree TSV to PolyOrigin's native pedfile.

    User TSV schema: ``offspring\\tparent1\\tparent2[\\tploidy]``.
    Output CSV schema: ``individual,population,motherid,fatherid,ploidy``.

    Founders: ``motherid=fatherid=0, population=0``. Offspring: integer
    ``population`` grouped by unique ``(parent1, parent2)`` pair, starting
    from 1. Per-individual ploidy from optional TSV column, else the
    ``ploidy_default`` fallback.
    """
    df = pd.read_csv(user_tsv, sep="\t")
    required = {"offspring", "parent1", "parent2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"pedigree TSV missing required column(s): {sorted(missing)}. "
            "Expected header: offspring, parent1, parent2 (optional: ploidy)."
        )
    has_ploidy_col = "ploidy" in df.columns

    dups = df[df.duplicated("offspring", keep=False)]["offspring"].unique().tolist()
    if dups:
        raise ValueError(f"pedigree TSV has duplicate offspring IDs: {dups}")

    offspring_set = set(df["offspring"])
    parents_set = set(df["parent1"]) | set(df["parent2"])

    multigen = parents_set & offspring_set
    if multigen:
        raise ValueError(
            f"PolyOrigin models F1 + founder selfings only; multi-generation "
            f"pedigrees not supported. Offending parent(s) also present as "
            f"offspring: {sorted(multigen)}."
        )

    all_ped = offspring_set | parents_set
    missing_in_probs = all_ped - sample_ids_in_probs
    if missing_in_probs:
        missing_list = sorted(missing_in_probs)[:10]
        raise ValueError(
            f"pedigree references {len(missing_in_probs)} individual(s) missing "
            f"from the probs sample set (first 10: {missing_list})."
        )

    fam_keys = list(dict.fromkeys(zip(df["parent1"], df["parent2"])))
    fam_pop = {pair: i + 1 for i, pair in enumerate(fam_keys)}

    rows: list[dict] = []
    for parent_id in sorted(parents_set):
        rows.append({
            "individual": parent_id,
            "population": 0,
            "motherid": 0,
            "fatherid": 0,
            "ploidy": ploidy_default,
        })
    for _, r in df.iterrows():
        # motherid / fatherid are the parent's individual ID (string); founders use
        # literal 0 sentinel (the only integer in these columns). PolyOrigin resolves
        # parent haplotypes by looking up the row whose `individual` == motherid.
        ploidy = (
            int(r["ploidy"])
            if has_ploidy_col and pd.notna(r.get("ploidy"))
            else ploidy_default
        )
        rows.append({
            "individual": str(r["offspring"]),
            "population": fam_pop[(r["parent1"], r["parent2"])],
            "motherid": str(r["parent1"]),
            "fatherid": str(r["parent2"]),
            "ploidy": ploidy,
        })

    out = Path(workdir) / "pedfile.csv"
    pd.DataFrame(
        rows, columns=["individual", "population", "motherid", "fatherid", "ploidy"]
    ).to_csv(out, index=False)
    return out
