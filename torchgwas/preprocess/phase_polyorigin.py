"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20


@dataclass
class PhasingResult:
    haplotypes: Tensor                     # (n_offspring, m) int64 — argmax joint-origin-combo index
    origin_probs: Tensor                   # (n_offspring, m, n_states) float64 — sparse-decoded joint origin probs
    parent_phased: Tensor                  # (n_parents, m, max_ploidy) int8 — haplotype dosages per copy
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
    state_table: Tensor              # (n_states, ploidy) int8 — joint-origin state → per-copy parental source; v = parent_id*ploidy + copy_in_parent
    haplotypes_per_copy: Tensor      # (n_off, ploidy, m) int8 — decoded per-copy parental alleles, drop-in for HaplotypeGWAS.scan


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


def _load_map_tsv(path: str, recomrate: float) -> pd.DataFrame:
    """Load marker map TSV. Synthesize cm from pos_bp if missing.

    Required columns: ``marker``, ``chrom``, ``pos_bp``. Optional: ``cm``.
    Missing cm → synthesize ``cm = pos_bp * recomrate / 1e6`` with a warning.
    Non-monotonic ``pos_bp`` within a chromosome → ``ValueError``.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"marker", "chrom", "pos_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"map TSV missing required column(s): {sorted(missing)}. "
            "Expected: marker, chrom, pos_bp (optional: cm)."
        )

    for chrom, sub in df.groupby("chrom", sort=False):
        diffs = sub["pos_bp"].diff().dropna()
        bad = diffs[diffs < 0]
        if not bad.empty:
            first_bad_idx = bad.index[0]
            first_bad_marker = df.loc[first_bad_idx, "marker"]
            raise ValueError(
                f"map TSV has non-monotonic pos_bp within chromosome {chrom!r}. "
                f"First offender: {first_bad_marker}."
            )

    if "cm" not in df.columns:
        logger.warning(
            "Map TSV lacks 'cm' column; synthesizing genetic positions via "
            "%.4f cM/Mb. Pass a linkage-map-derived 'cm' column for production runs.",
            recomrate,
        )
        df = df.copy()
        df["cm"] = df["pos_bp"].astype(float) * recomrate / 1e6

    return df[["marker", "chrom", "pos_bp", "cm"]]


def _enumerate_state_table(ploidy: int) -> Tensor:
    """Build the joint-origin state table for a 2-parent F1 at given ploidy.

    Bivalent meiosis only: each parent contributes ploidy/2 copies per
    gamete. Gametes are sorted (ploidy/2)-subsets of ``{0..ploidy-1}`` in
    lexicographic order. States are (parent1 gamete, parent2 gamete)
    Cartesian-product, flat-indexed as
    ``s = p1_gamete_idx * n_gametes + p2_gamete_idx``.

    Parameters
    ----------
    ploidy : int
        Must be even and in ``{2, 4, 6}``.

    Returns
    -------
    Tensor, shape (n_states, ploidy), int8
        Each row holds ``ploidy`` values ``v`` in ``[0, 2*ploidy)``:
        ``v = parent_id * ploidy + copy_in_parent``. Parent1 copies
        occupy the first ``ploidy/2`` slots, parent2 copies the last
        ``ploidy/2`` slots.
    """
    if ploidy not in _VALID_PLOIDIES:
        raise ValueError(
            f"_enumerate_state_table: ploidy must be in {{2, 4, 6}}; got {ploidy}."
        )
    half = ploidy // 2
    gametes = list(combinations(range(ploidy), half))

    rows: list[list[int]] = []
    for p1_gamete in gametes:
        for p2_gamete in gametes:
            row = [0 * ploidy + c for c in p1_gamete] + [1 * ploidy + c for c in p2_gamete]
            rows.append(row)

    return torch.tensor(rows, dtype=torch.int8)


def _decode_haplotypes_per_copy(
    haplotypes: Tensor,
    parent_phased: Tensor,
    state_table: Tensor,
    ploidy: int,
) -> Tensor:
    """Expand joint-origin state indices into per-chromosome-copy alleles.

    For each ``(offspring i, marker j)`` with state ``s = haplotypes[i, j]``:
    for each copy slot ``k`` in ``[0, ploidy)``::

        v = state_table[s, k]
        parent_id, copy_in_parent = divmod(int(v), ploidy)
        output[i, k, j] = parent_phased[parent_id, j, copy_in_parent]

    Vectorized via torch advanced indexing.

    Parameters
    ----------
    haplotypes : Tensor, shape (n_off, m), int64
        Joint-origin state index per offspring per marker.
    parent_phased : Tensor, shape (n_parents, m, max_ploidy), int8
        Per-copy parental alleles.
    state_table : Tensor, shape (n_states, ploidy), int8
        As produced by ``_enumerate_state_table``.
    ploidy : int
        Must match ``state_table.shape[1]``.

    Returns
    -------
    Tensor, shape (n_off, ploidy, m), int8
        Per-copy parental alleles. Drop-in for
        ``HaplotypeGWAS.scan(haplotypes=...)``.
    """
    n_off, m = haplotypes.shape

    # Look up state table rows for each offspring-marker cell.
    # state_rows shape: (n_off, m, ploidy), values v = parent_id*ploidy + copy_in_parent
    state_rows = state_table[haplotypes]  # advanced indexing

    # Decode (parent_id, copy_in_parent) per cell
    parent_id = (state_rows.to(torch.int64) // ploidy)       # (n_off, m, ploidy)
    copy_in_parent = (state_rows.to(torch.int64) % ploidy)   # (n_off, m, ploidy)

    # Gather alleles from parent_phased[parent_id, marker_j, copy_in_parent].
    # Broadcast marker index j across the (n_off, m, ploidy) grid.
    m_idx = torch.arange(m, dtype=torch.int64, device=haplotypes.device)
    m_idx_b = m_idx.view(1, m, 1).expand(n_off, m, ploidy)  # (n_off, m, ploidy)

    alleles = parent_phased[parent_id, m_idx_b, copy_in_parent]  # (n_off, m, ploidy) int8

    # Reorder to (n_off, ploidy, m)
    return alleles.permute(0, 2, 1).contiguous().to(torch.int8)


def _validate_state_table(
    state_table: Tensor,
    origin_probs: Tensor,
    parent_phased: Tensor,
    postdose_probs: Tensor,
    ploidy: int,
    atol: float = 1e-3,
    tool_version: str = "unknown",
) -> None:
    """Round-trip check: state_table-derived expected dosage must match
    PolyOrigin's emitted ``postdose_probs`` expected dosage.

    For each state ``s`` and marker ``j``, compute
    ``dose_at(j, s) = sum over the copies listed in state_table[s] of the
    corresponding parent_phased allele``. Then::

        E_state[i, j] = sum_s origin_probs[i, j, s] * dose_at(j, s)
        E_post[i, j]  = sum_d d * postdose_probs[i, j, d]

    Both should be equal (within ``atol``) iff our state enumeration's
    per-state copy-SETS match PolyOrigin's. The check catches
    dose-changing reorderings but is blind to within-parent gamete copy
    permutations (which are downstream-equivalent; see spec Section 2.3).

    Raises
    ------
    RuntimeError
        If ``max |E_state - E_post| > atol``. Diagnostic names the
        offending ``(offspring_idx, marker_idx)`` cell plus both computed
        values.
    """
    n_off, m, n_states = origin_probs.shape

    # PolyOrigin's sparse genoprob may reference fewer states than the full
    # state table (it only emits states with nonzero probability). Slice the
    # table to the first n_states rows so array dimensions stay consistent.
    st64 = state_table[:n_states].to(torch.int64)
    parent_id = st64 // ploidy           # (n_states, ploidy)
    copy_in_parent = st64 % ploidy       # (n_states, ploidy)

    # Broadcast to (n_states, ploidy, m)
    p_id_b = parent_id.unsqueeze(-1).expand(n_states, ploidy, m)
    c_in_p_b = copy_in_parent.unsqueeze(-1).expand(n_states, ploidy, m)
    m_idx = torch.arange(m, dtype=torch.int64, device=state_table.device)
    m_idx_b = m_idx.view(1, 1, m).expand(n_states, ploidy, m)
    alleles = parent_phased[p_id_b, m_idx_b, c_in_p_b].to(torch.float64)
    dose_per_state = alleles.sum(dim=1)  # (n_states, m)

    # E_state[i, j] = sum_s origin_probs[i, j, s] * dose_per_state[s, j]
    e_state = torch.einsum("ijs,sj->ij", origin_probs, dose_per_state.to(torch.float64))

    # E_post[i, j] = sum_d d * postdose_probs[i, j, d]
    kp1 = postdose_probs.shape[-1]
    d_vals = torch.arange(kp1, dtype=torch.float64, device=postdose_probs.device)
    e_post = (postdose_probs * d_vals).sum(dim=-1)

    # Compare
    abs_dev = (e_state - e_post).abs()
    max_dev = float(abs_dev.max())
    if max_dev > atol:
        flat_idx = int(abs_dev.argmax())
        off_idx, mkr_idx = divmod(flat_idx, m)
        raise RuntimeError(
            f"State-table round-trip mismatch — our enumeration disagrees "
            f"with PolyOrigin v{tool_version}. Max deviation {max_dev:.4f} at "
            f"(offspring_idx={off_idx}, marker_idx={mkr_idx}); "
            f"ours={float(e_state[off_idx, mkr_idx]):.4f}, "
            f"PolyOrigin={float(e_post[off_idx, mkr_idx]):.4f}. "
            f"Likely cause: PolyOrigin reordered states between releases. "
            f"Open an issue with the offending phase-poly inputs."
        )


def _build_polyorigin_genofile(
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    map_df: pd.DataFrame,
    parent_phased_df: pd.DataFrame | None,
    parent_ids: set[str],
    workdir: str,
) -> Path:
    """Write the merged PolyOrigin genofile CSV.

    Schema: ``marker, chromosome, pos, ind1, ind2, ..., indN``.
    Offspring cells: probability strings ``p0|p1|...|pk`` (4 decimals).
    Parents default to the same encoding; if ``parent_phased_df`` is
    given, those parents use their ``phasedgeno`` strings instead and a
    warning is logged for any parent present in both sources.

    Variant order: ``map_df`` row order. Sample columns: parents
    alphabetical, then offspring alphabetical.
    """
    if probs.dtype != torch.float64:
        probs = probs.to(torch.float64)

    n, m, _ = probs.shape
    if len(sample_ids) != n:
        raise ValueError(f"sample_ids length {len(sample_ids)} != probs.shape[0] {n}")
    if len(variant_ids) != m:
        raise ValueError(f"variant_ids length {len(variant_ids)} != probs.shape[1] {m}")

    parents_sorted = sorted(parent_ids & set(sample_ids))
    offspring_sorted = sorted(set(sample_ids) - parent_ids)
    col_order = parents_sorted + offspring_sorted

    if parent_phased_df is not None:
        both = parent_ids & set(parent_phased_df.index) & set(sample_ids)
        for pid in sorted(both):
            logger.warning(
                "Parent %s has both probs-encoded and pre-phased entries; "
                "using pre-phased.",
                pid,
            )

    sample_idx = {sid: i for i, sid in enumerate(sample_ids)}
    map_sub = map_df.set_index("marker").loc[variant_ids]

    rows: list[dict] = []
    for j, vid in enumerate(variant_ids):
        row = {
            "marker": vid,
            "chromosome": map_sub.loc[vid, "chrom"],
            "pos": map_sub.loc[vid, "cm"],
        }
        for sid in col_order:
            if (
                sid in parent_ids
                and parent_phased_df is not None
                and sid in parent_phased_df.index
            ):
                row[sid] = str(parent_phased_df.loc[sid, vid])
            else:
                p = probs[sample_idx[sid], j, :].tolist()
                row[sid] = "|".join(f"{v:.4f}" for v in p)
        rows.append(row)

    out = Path(workdir) / "genofile.csv"
    cols = ["marker", "chromosome", "pos"] + col_order
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    return out


def _validate_inputs(
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
) -> Tensor:
    """Raise on invalid input; return probs (possibly renormalized)."""
    if ploidy not in _VALID_PLOIDIES:
        raise ValueError(
            f"PolyOrigin supports ploidy 2, 4, or 6 only; got {ploidy}."
        )
    if probs.ndim != 3:
        raise ValueError(
            f"probs must be 3D (n, m, k+1); got shape {tuple(probs.shape)}."
        )
    n, m, kp1 = probs.shape
    if kp1 != ploidy + 1:
        raise ValueError(
            f"probs.shape[-1]={kp1} inconsistent with ploidy={ploidy} "
            f"(expected {ploidy + 1})."
        )
    if len(sample_ids) != n:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} != probs.shape[0] {n}."
        )
    if len(variant_ids) != m:
        raise ValueError(
            f"variant_ids length {len(variant_ids)} != probs.shape[1] {m}."
        )

    if probs.dtype != torch.float64:
        probs = probs.to(torch.float64)
    row_sums = probs.sum(dim=-1)
    max_dev = float((row_sums - 1.0).abs().max())
    if max_dev > 1e-3:
        logger.warning(
            "probs rows deviate from 1.0 by up to %.4f (tol=1e-3); renormalizing.",
            max_dev,
        )
    probs = probs / row_sums.unsqueeze(-1).clamp(min=1e-12)
    return probs


# ---------------------------------------------------------------------------
# Output CSV parsers (Task 9)
# ---------------------------------------------------------------------------

def _parse_genoprob(
    path: str,
    expected_offspring: list[str],
    ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse PolyOrigin's *_genoprob.csv (real PolyOrigin format, verified Task 14).

    Actual column scheme: ``marker, chromosome, position[, doublereduction],
    p1, p2, o1, o2, ...``

    Parent cells: ``a1|a2|...|ak`` (phased haplotype dosages, ignored here).
    Offspring cells: sparse ``idx1|idx2|...=>prob1|prob2|...`` encoding
    of joint haplotype-combination origin probabilities.

    Returns (origin_probs (n_off, m, n_states) float64, offspring_ids).
    n_states is determined from the maximum sparse index observed + 1.
    """
    df = pd.read_csv(path)
    # Detect meta columns (position / pos alias handled here)
    meta_cols = {"marker", "chromosome", "position", "pos", "doublereduction"}
    # Identify offspring columns (those in expected_offspring; rest are parents or meta)
    offspring_set = set(expected_offspring)
    off_cols = [c for c in df.columns if c in offspring_set]

    if set(off_cols) != offspring_set:
        raise RuntimeError(
            f"genoprob offspring columns {sorted(off_cols)} != expected "
            f"{sorted(expected_offspring)}."
        )
    off_order = list(expected_offspring)
    n_off = len(off_order)
    n_markers = len(df)

    # First pass: determine maximum state index across all cells
    max_state = 0
    for col in off_cols:
        for cell in df[col]:
            cell_s = str(cell)
            if "=>" in cell_s:
                idx_part = cell_s.split("=>")[0]
                for idx in idx_part.split("|"):
                    idx_i = int(idx)
                    if idx_i > max_state:
                        max_state = idx_i
            else:
                # Pipe-sep prob vector (fallback for unusual formats)
                n_vals = len(cell_s.split("|"))
                if n_vals - 1 > max_state:
                    max_state = n_vals - 1
    n_states = max_state + 1

    # Second pass: fill dense tensor
    out = torch.zeros(n_off, n_markers, n_states, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    for col in off_cols:
        oi = off_idx[col]
        for j, cell in enumerate(df[col]):
            cell_s = str(cell)
            if "=>" in cell_s:
                idx_part, prob_part = cell_s.split("=>", 1)
                indices = [int(x) for x in idx_part.split("|")]
                probs = [float(x) for x in prob_part.split("|")]
                for idx_i, p in zip(indices, probs):
                    out[oi, j, idx_i] = p
            else:
                # Fallback: treat as dense pipe-sep probs
                vals = [float(x) for x in cell_s.split("|")]
                for k, v in enumerate(vals):
                    out[oi, j, k] = v
    return out, off_order


def _parse_parentphased(
    path: str,
    expected_parents: list[str],
    max_ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse *_parentphased.csv (real PolyOrigin format, verified Task 14).

    Actual column scheme: ``marker, chromosome, position, p1, p2, o1, o2, ...``

    Parent cells: ``a1|a2|...|ak`` (integer haplotype dosages per copy).
    Offspring cells: ``d0|d1|...|dk`` (posterior dosage probs; ignored here).

    Returns (parent_phased (n_parents, m, max_ploidy) int8, parent_ids).
    """
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "position", "pos", "doublereduction"}
    parent_set = set(expected_parents)
    # Only process columns that match expected parent IDs
    parent_cols = [c for c in df.columns if c in parent_set]
    if set(parent_cols) != parent_set:
        raise RuntimeError(
            f"parentphased parent columns {sorted(parent_cols)} != expected "
            f"{sorted(expected_parents)}."
        )
    n_markers = len(df)
    out = torch.full(
        (len(expected_parents), n_markers, max_ploidy), -1, dtype=torch.int8
    )
    for i, p in enumerate(expected_parents):
        for j, cell in enumerate(df[p]):
            alleles = str(cell).split("|")
            for k, a in enumerate(alleles[:max_ploidy]):
                try:
                    out[i, j, k] = int(a)
                except ValueError:
                    raise RuntimeError(
                        f"parentphased cell at parent={p} marker_idx={j}: "
                        f"non-integer allele {a!r}."
                    )
    return out, list(expected_parents)


def _parse_postdose(
    path: str,
    expected_offspring: list[str],
    max_ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse *_postdoseprob.csv (real PolyOrigin format, verified Task 14).

    Actual column scheme: ``marker, chromosome, position[, doublereduction],
    p1, p2, o1, o2, ...``

    Offspring cells: ``d0|d1|...|dk`` (posterior dosage probabilities, ploidy+1 values).
    Parent cells: ``a1|a2|...|ak`` (ignored here).

    Returns (postdose_probs (n_off, m, max_ploidy+1) float64, offspring_ids).
    """
    df = pd.read_csv(path)
    offspring_set = set(expected_offspring)
    off_cols = [c for c in df.columns if c in offspring_set]
    if set(off_cols) != offspring_set:
        raise RuntimeError(
            f"postdose offspring columns {sorted(off_cols)} != expected "
            f"{sorted(expected_offspring)}."
        )
    off_order = list(expected_offspring)
    n_off = len(off_order)
    n_markers = len(df)
    out = torch.zeros(n_off, n_markers, max_ploidy + 1, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    for col in off_cols:
        oi = off_idx[col]
        for j, cell in enumerate(df[col]):
            vals = [float(x) for x in str(cell).split("|")]
            for d, v in enumerate(vals[: max_ploidy + 1]):
                out[oi, j, d] = v
    return out, off_order


def _parse_maprefined(
    path: str,
    expected_markers: list[str],
) -> tuple[list[str], list[str], Tensor]:
    """Return (chrom, variant_ids, pos_cm_tensor).

    Handles both ``pos`` and ``position`` column names (PolyOrigin uses either
    depending on the version).
    """
    df = pd.read_csv(path)
    got = set(df["marker"])
    if got != set(expected_markers):
        raise RuntimeError(
            f"map_refined marker set differs from input "
            f"(missing {set(expected_markers) - got}, extra {got - set(expected_markers)})."
        )
    pos_col = "pos" if "pos" in df.columns else "position"
    return (
        [str(c) for c in df["chromosome"]],
        [str(v) for v in df["marker"]],
        torch.tensor(df[pos_col].to_numpy(), dtype=torch.float64),
    )


def _parse_polyancestry(path: str) -> pd.DataFrame:
    """Return the raw DataFrame of the *_polyancestry.csv — diagnostic output.

    PolyOrigin's polyancestry file is a multi-section CSV where each section
    starts with a ``PolyOrigin-PolyAncestry,<sectionname>`` header line and
    has its own column count. Standard ``pd.read_csv`` cannot parse this;
    we skip malformed lines and return whatever pandas can read.
    """
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        # Fallback: return a minimal DataFrame so downstream code doesn't crash
        return pd.DataFrame({"marker": [], "valent": []})


# ---------------------------------------------------------------------------
# Module-level JuliaError alias (Task 10)
# Tests monkeypatch this name directly:
#   monkeypatch.setattr("torchgwas.preprocess.phase_polyorigin._JULIA_ERROR", FakeClass)
# The placeholder is a plain Exception subclass so that the module imports
# cleanly without touching juliacall (which would trigger Julia init).
# run_polyorigin replaces this in the module namespace after juliacall is
# already initialised by get_runtime(), so production catches the real class.
# ---------------------------------------------------------------------------


class _JULIA_ERROR(Exception):  # placeholder; replaced lazily inside run_polyorigin
    """Placeholder for juliacall.JuliaError — swapped out lazily at runtime."""


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def run_polyorigin(
    probs: Tensor | str,
    pedigree_tsv: str,
    map_tsv: str,
    output_path: str,
    *,
    ploidy: int,
    sample_ids: Optional[list[str]] = None,
    variant_ids: Optional[list[str]] = None,
    parent_phased_csv: Optional[str] = None,
    julia_path: Optional[str] = None,
    auto_install_julia: bool = False,
    refinemap: bool = True,
    recomrate: float = 1.0,
    delmarker: bool = True,
    keep_workdir: bool = False,
) -> PhasingResult:
    """Run PolyOrigin phasing end-to-end. See design spec Section 2.2."""
    # Resolve probs input — tensor directly, or path to a .probs.pt
    if isinstance(probs, str):
        loaded = torch.load(probs)
        if isinstance(loaded, dict):
            probs_t = loaded["probs"]
            sample_ids = sample_ids or loaded.get("sample_ids")
            variant_ids = variant_ids or loaded.get("variant_ids")
        else:
            probs_t = loaded
    else:
        probs_t = probs

    if sample_ids is None or variant_ids is None:
        raise ValueError(
            "sample_ids and variant_ids are required when probs is a "
            "tensor (or a .pt that doesn't carry them as keys)."
        )

    probs_t = _validate_inputs(probs_t, sample_ids, variant_ids, ploidy)

    # Parse map first so we know cm positions for the genofile builder
    map_df = _load_map_tsv(map_tsv, recomrate=recomrate)

    # Parent-phased escape hatch
    parent_phased_df: Optional[pd.DataFrame] = None
    if parent_phased_csv is not None:
        parent_phased_df = pd.read_csv(parent_phased_csv).set_index("individual")

    # Import the runtime late to avoid heavy import at module load
    from torchgwas.preprocess import _polyorigin_runtime as _rt

    # Build tempdir and converted files
    with tempfile.TemporaryDirectory(prefix="polyorigin_") as _tmp:
        workdir = (
            Path(tempfile.mkdtemp(prefix="polyorigin_keep_"))
            if keep_workdir
            else Path(_tmp)
        )

        # We need parent_ids from the pedigree TSV; parse once and derive.
        user_ped = pd.read_csv(pedigree_tsv, sep="\t")
        parent_ids = set(user_ped["parent1"]) | set(user_ped["parent2"])

        pedfile = _build_polyorigin_pedfile(
            pedigree_tsv,
            ploidy_default=ploidy,
            sample_ids_in_probs=set(sample_ids),
            workdir=str(workdir),
        )

        # Early same-ploidy guard: fail fast before Julia is invoked so no
        # partial workdir artifacts accumulate. Re-read the constructed pedfile
        # (already written) to get the resolved per-individual ploidy.
        _ped_early = pd.read_csv(pedfile)
        _ploidy_values_early = set(_ped_early["ploidy"].astype(int).tolist())
        if len(_ploidy_values_early) > 1:
            raise ValueError(
                f"haplotypes_per_copy decoding requires uniform ploidy across "
                f"all individuals; got {sorted(_ploidy_values_early)}. Mixed-ploidy "
                f"F1 is deferred."
            )

        genofile = _build_polyorigin_genofile(
            probs=probs_t,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            map_df=map_df,
            parent_phased_df=parent_phased_df,
            parent_ids=parent_ids,
            workdir=str(workdir),
        )

        # Compute input hash (genofile + pedfile + original map file)
        combined = (
            _sha256_file(str(genofile))
            + _sha256_file(str(pedfile))
            + _sha256_file(map_tsv)
        )
        input_hash = _sha256_str(combined)

        # Bootstrap Julia + invoke PolyOrigin
        jl, po, tool_version = _rt.get_runtime(
            julia_path=julia_path,
            auto_install_julia=auto_install_julia,
        )
        cmd_str = (
            f'polyOrigin("{genofile.name}", "{pedfile.name}", '
            f'workdir="{workdir}", isphysmap=false, refinemap={str(refinemap).lower()}, '
            f'delmarker={str(delmarker).lower()}, outstem="out")'
        )
        try:
            po.polyOrigin(
                str(genofile),
                str(pedfile),
                workdir=str(workdir),
                isphysmap=False,
                refinemap=refinemap,
                delmarker=delmarker,
                outstem="out",
            )
        except _JULIA_ERROR as e:
            raise RuntimeError(f"PolyOrigin failed: {e}") from e

        # Validate the full set of output CSVs exist
        # Note: out_maprefined.csv is only written by PolyOrigin when refinemap=True.
        expected = [
            "out_genoprob.csv",
            "out_postdoseprob.csv",
            "out_parentphased.csv",
            "out_polyancestry.csv",
        ]
        if refinemap:
            expected.append("out_maprefined.csv")
        for fname in expected:
            p = workdir / fname
            if not p.is_file():
                raise RuntimeError(
                    f"PolyOrigin returned but expected output file missing: {fname}. "
                    f"Check the Julia log at {workdir / 'out.log'}."
                )

        # Identify offspring / parents from the constructed pedfile
        ped_df = pd.read_csv(pedfile)
        parents = ped_df[ped_df["population"] == 0]["individual"].astype(str).tolist()
        offspring = ped_df[ped_df["population"] != 0]["individual"].astype(str).tolist()
        max_ploidy = int(ped_df["ploidy"].max())
        per_ind_ploidy = dict(
            zip(ped_df["individual"].astype(str), ped_df["ploidy"].astype(int))
        )

        # Parse map: use refined map if available, else fall back to input map
        if refinemap:
            chrom_ref, var_ids_ref, pos_cm_ref = _parse_maprefined(
                str(workdir / "out_maprefined.csv"),
                expected_markers=variant_ids,
            )
        else:
            # refinemap=False: PolyOrigin does not write *_maprefined.csv; use input map
            chrom_ref = [str(c) for c in map_df.set_index("marker").loc[variant_ids, "chrom"]]
            var_ids_ref = list(variant_ids)
            pos_cm_ref = torch.tensor(
                map_df.set_index("marker").loc[variant_ids, "cm"].to_numpy(),
                dtype=torch.float64,
            )
        map_refined_flag = refinemap and var_ids_ref != variant_ids
        origin_probs, _ = _parse_genoprob(
            str(workdir / "out_genoprob.csv"),
            expected_offspring=offspring,
            ploidy=max_ploidy,
        )
        postdose_probs, _ = _parse_postdose(
            str(workdir / "out_postdoseprob.csv"),
            expected_offspring=offspring,
            max_ploidy=max_ploidy,
        )
        parent_phased, _ = _parse_parentphased(
            str(workdir / "out_parentphased.csv"),
            expected_parents=parents,
            max_ploidy=max_ploidy,
        )
        valent_diag = _parse_polyancestry(str(workdir / "out_polyancestry.csv"))

        # Derive haplotypes as argmax joint-origin-combo index per offspring per marker
        # origin_probs shape: (n_off, m, n_states) → haplotypes shape: (n_off, m) int64
        haplotypes = origin_probs.argmax(dim=-1)

        # Build map-based tensors on the refined order
        map_df_ref = map_df.set_index("marker").loc[var_ids_ref]
        pos_bp = torch.tensor(map_df_ref["pos_bp"].to_numpy(), dtype=torch.int64)

        # --- Tier A #1: same-ploidy guard + state-table + validate + decode ---
        ploidy_values = set(per_ind_ploidy.values())
        if len(ploidy_values) > 1:
            raise ValueError(
                f"haplotypes_per_copy decoding requires uniform ploidy across "
                f"all individuals; got {sorted(ploidy_values)}. Mixed-ploidy "
                f"F1 is deferred."
            )
        decode_ploidy = next(iter(ploidy_values))

        state_table = _enumerate_state_table(decode_ploidy)
        _validate_state_table(
            state_table, origin_probs, parent_phased, postdose_probs,
            ploidy=decode_ploidy, tool_version=tool_version,
        )
        haplotypes_per_copy = _decode_haplotypes_per_copy(
            haplotypes, parent_phased, state_table, ploidy=decode_ploidy,
        )

        result = PhasingResult(
            haplotypes=haplotypes,
            origin_probs=origin_probs,
            parent_phased=parent_phased,
            offspring_ids=offspring,
            parent_ids=parents,
            variant_ids=var_ids_ref,
            chrom=chrom_ref,
            pos_bp=pos_bp,
            pos_cm=pos_cm_ref,
            per_individual_ploidy=per_ind_ploidy,
            map_refined=map_refined_flag,
            valent_diag=valent_diag,
            postdose_probs=postdose_probs,
            tool="polyorigin",
            tool_version=tool_version,
            input_hash=input_hash,
            cmd=cmd_str,
            workdir=str(workdir) if keep_workdir else None,
            state_table=state_table,
            haplotypes_per_copy=haplotypes_per_copy,
        )

        # Atomic persistence — only after every parse succeeds
        _persist_result(result, output_path)
        return result


def _persist_result(result: PhasingResult, output_path: str) -> None:
    """Atomically persist a PhasingResult to ``<output_path>.*``.

    Uses a temp-sibling-then-rename scheme: each artifact is written to
    ``<output_path>.<name>.tmp.<pid>`` then renamed. If any write fails,
    previously-renamed siblings are removed so the filesystem ends up
    with either all new artifacts or none.
    """
    prefix = Path(output_path)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tmpid = os.getpid()
    renamed: list[Path] = []

    def _persist(name: str, writer) -> None:
        tmp = prefix.with_suffix(prefix.suffix + f".{name}.tmp.{tmpid}")
        writer(str(tmp))
        final = prefix.with_suffix(prefix.suffix + f".{name}")
        os.replace(tmp, final)
        renamed.append(final)

    try:
        _persist("haplotypes.pt", lambda p: torch.save(result.haplotypes, p))
        _persist("origin_probs.pt", lambda p: torch.save(result.origin_probs, p))
        _persist("parent_phased.pt", lambda p: torch.save(result.parent_phased, p))
        _persist("postdose_probs.pt", lambda p: torch.save(result.postdose_probs, p))
        _persist("map_refined.tsv", lambda p: pd.DataFrame({
            "marker": result.variant_ids,
            "chrom": result.chrom,
            "pos_bp": result.pos_bp.tolist(),
            "pos_cm": result.pos_cm.tolist(),
        }).to_csv(p, index=False, sep="\t"))
        _persist("valent_diag.tsv", lambda p: result.valent_diag.to_csv(p, index=False, sep="\t"))

        meta = {
            "tool": result.tool,
            "tool_version": result.tool_version,
            "ploidy_per_individual": result.per_individual_ploidy,
            "offspring_ids": result.offspring_ids,
            "parent_ids": result.parent_ids,
            "variant_ids": result.variant_ids,
            "chrom": result.chrom,
            "pos_bp": result.pos_bp.tolist(),
            "pos_cm": result.pos_cm.tolist(),
            "map_refined": result.map_refined,
            "input_hash": result.input_hash,
            "cmd": result.cmd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _persist("meta.json", lambda p: Path(p).write_text(json.dumps(meta, indent=2)))
    except Exception:
        # Roll back any renamed artifacts so we leave no partial <output>.*
        for p in renamed:
            try:
                p.unlink()
            except OSError:
                pass
        raise
