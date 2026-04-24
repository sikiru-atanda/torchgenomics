"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass
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
    """Parse PolyOrigin's *_genoprob.csv.

    Expected column scheme (best-guess; verified at Task 14 parity run):
    ``marker, chromosome, pos`` followed by offspring × chromosome-copy ×
    parent-haplotype probability columns named like
    ``<offspring>_c<chromosome_copy>_h<parent_haplotype>``.

    Returns (origin_probs (n_off, ploidy, m, n_parent_haps), offspring_ids).
    """
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    prob_cols = [c for c in df.columns if c not in meta_cols]

    pat = re.compile(r"^(?P<off>.+)_c(?P<c>\d+)_h(?P<h>\d+)$")
    parsed: list[tuple[str, int, int, str]] = []
    for c in prob_cols:
        m = pat.match(c)
        if m is None:
            raise RuntimeError(
                f"Unexpected column in genoprob file: {c!r}. "
                "Parser expects '<offspring>_c<idx>_h<idx>' naming."
            )
        parsed.append((m["off"], int(m["c"]), int(m["h"]), c))

    offsprings = sorted({x[0] for x in parsed})
    c_vals = sorted({x[1] for x in parsed})
    h_vals = sorted({x[2] for x in parsed})
    if len(c_vals) != ploidy:
        raise RuntimeError(
            f"genoprob column copy-count {len(c_vals)} != ploidy {ploidy}."
        )
    n_parent_haps = len(h_vals)

    if set(offsprings) != set(expected_offspring):
        raise RuntimeError(
            f"genoprob offspring set {sorted(offsprings)} != expected "
            f"{sorted(expected_offspring)}."
        )
    off_order = list(expected_offspring)

    n_off = len(off_order)
    n_markers = len(df)
    out = torch.zeros(n_off, ploidy, n_markers, n_parent_haps, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    c_idx = {c: i for i, c in enumerate(c_vals)}
    h_idx = {h: i for i, h in enumerate(h_vals)}
    for off, c, h, col in parsed:
        out[off_idx[off], c_idx[c], :, h_idx[h]] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float64
        )
    return out, off_order


def _parse_parentphased(
    path: str,
    expected_parents: list[str],
    max_ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse *_parentphased.csv — one column per parent, cell = ``a1|a2|...|ak``."""
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    parent_cols = [c for c in df.columns if c not in meta_cols]
    if set(parent_cols) != set(expected_parents):
        raise RuntimeError(
            f"parentphased column set {sorted(parent_cols)} != expected "
            f"{sorted(expected_parents)}."
        )
    n_markers = len(df)
    out = torch.full(
        (len(expected_parents), max_ploidy, n_markers), -1, dtype=torch.int8
    )
    for i, p in enumerate(expected_parents):
        for j, cell in enumerate(df[p]):
            alleles = str(cell).split("|")
            for k, a in enumerate(alleles[:max_ploidy]):
                try:
                    out[i, k, j] = int(a)
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
    """Parse *_postdoseprob.csv — columns ``<offspring>_d<dosage>``."""
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    pat = re.compile(r"^(?P<off>.+)_d(?P<d>\d+)$")
    parsed: list[tuple[str, int, str]] = []
    for c in df.columns:
        if c in meta_cols:
            continue
        m = pat.match(c)
        if m is None:
            raise RuntimeError(
                f"Unexpected column in postdose file: {c!r}. "
                "Parser expects '<offspring>_d<dosage>' naming."
            )
        parsed.append((m["off"], int(m["d"]), c))

    offsprings = sorted({x[0] for x in parsed})
    if set(offsprings) != set(expected_offspring):
        raise RuntimeError(
            f"postdose offspring set {sorted(offsprings)} != expected."
        )
    kmax = max(x[1] for x in parsed)
    if kmax + 1 > max_ploidy + 1:
        raise RuntimeError(
            f"postdose dosage slots {kmax + 1} > max_ploidy+1 {max_ploidy + 1}."
        )

    off_order = list(expected_offspring)
    n_off = len(off_order)
    n_markers = len(df)
    out = torch.zeros(n_off, n_markers, max_ploidy + 1, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    for off, d, col in parsed:
        out[off_idx[off], :, d] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float64
        )
    return out, off_order


def _parse_maprefined(
    path: str,
    expected_markers: list[str],
) -> tuple[list[str], list[str], Tensor]:
    """Return (chrom, variant_ids, pos_cm_tensor)."""
    df = pd.read_csv(path)
    got = set(df["marker"])
    if got != set(expected_markers):
        raise RuntimeError(
            f"map_refined marker set differs from input "
            f"(missing {set(expected_markers) - got}, extra {got - set(expected_markers)})."
        )
    return (
        [str(c) for c in df["chromosome"]],
        [str(v) for v in df["marker"]],
        torch.tensor(df["pos"].to_numpy(), dtype=torch.float64),
    )


def _parse_polyancestry(path: str) -> pd.DataFrame:
    """Return the raw DataFrame of the *_polyancestry.csv — diagnostic output."""
    return pd.read_csv(path)


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
    nworkers: int = 1,
    seed: Optional[int] = 1234,
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
            f'nworkers={nworkers}, seed={seed}, outstem="out")'
        )
        try:
            po.polyOrigin(
                str(genofile),
                str(pedfile),
                workdir=str(workdir),
                isphysmap=False,
                refinemap=refinemap,
                nworkers=nworkers,
                seed=seed,
                outstem="out",
            )
        except _JULIA_ERROR as e:
            raise RuntimeError(f"PolyOrigin failed: {e}") from e

        # Validate the full set of output CSVs exist
        expected = [
            "out_genoprob.csv",
            "out_postdoseprob.csv",
            "out_parentphased.csv",
            "out_maprefined.csv",
            "out_polyancestry.csv",
        ]
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

        # Parse outputs
        chrom_ref, var_ids_ref, pos_cm_ref = _parse_maprefined(
            str(workdir / "out_maprefined.csv"),
            expected_markers=variant_ids,
        )
        map_refined_flag = var_ids_ref != variant_ids
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

        # Derive haplotypes as argmax of origin_probs along last axis → int8
        haplotypes = origin_probs.argmax(dim=-1).to(torch.int8)

        # Build map-based tensors on the refined order
        map_df_ref = map_df.set_index("marker").loc[var_ids_ref]
        pos_bp = torch.tensor(map_df_ref["pos_bp"].to_numpy(), dtype=torch.int64)

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
        )

        # Atomic persistence — only after every parse succeeds
        _persist_result(result, output_path)
        return result


def _persist_result(result: PhasingResult, output_path: str) -> None:
    """Placeholder — implemented in Task 11."""
    pass
