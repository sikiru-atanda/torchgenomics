#!/usr/bin/env python3
"""Compare torchgenomics' admixture-aware kinship/PCA pipeline against the
GENESIS (R: SNPRelate + GWASTools + GENESIS) golden outputs.

This is the definitive reference-equivalence gate for Phase 57 Unit A's
three estimators:

    torchgenomics.linalg.kinship_admixed.king_robust_kinship  vs snpgdsIBDKING(type="KING-robust")
    torchgenomics.linalg.kinship_admixed.pc_air                vs GENESIS::pcair()
    torchgenomics.linalg.kinship_admixed.pc_relate              vs GENESIS::pcrelate()

Every internal test in tests/test_kinship_admixed.py is explicit that it is
NOT reference-equivalence evidence; this script is what closes that gap
(spec §8 Claim 3).

Pipeline symmetry
------------------
- KING-robust: both sides run on the FULL, unpruned m=2000-marker fixture
  (``out/G.npy`` / ``out/fixture.bed``) -- KING-robust is an IBS-category
  count, not a PCA/regression step, so the natural apples-to-apples
  comparison uses the entire shared marker panel.
- PC-AiR / PC-Relate: both sides restrict to the IDENTICAL LD-pruned marker
  subset (``out/pruned_snp_ids.txt``, produced by
  ``torchgenomics.linalg.kinship_admixed.ld_prune_independent`` at export
  time and passed to GENESIS via ``snp.include=`` in ``run_genesis.R``) --
  so a PC-AiR/PC-Relate disagreement can only be attributed to the
  estimator itself, never to a marker-panel mismatch between two
  independent LD-pruning implementations.

Usage
-----
    python3 validation/external/genesis/compare.py

Runs the torchgenomics side unconditionally. If the GENESIS golden TSVs
(``out/genesis_{king_kinship,pcair_pcs,pcrelate_kinship}.tsv``) are absent
(i.e. ``run_genesis.R`` has not been executed), the GENESIS comparison is
skipped gracefully and the script exits 0 -- this lets the harness scripts
be authored and smoke-tested without R/GENESIS installed. If the TSVs ARE
present, every tolerance gate below is asserted and the script exits
nonzero on any FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REPO_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from torchgenomics.linalg.kinship_admixed import (  # noqa: E402
    king_robust_kinship,
    ld_prune_independent,
    pc_air,
    pc_relate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tolerance gates -- PLACEHOLDERS, observed-then-floored per repo convention
# (see CLAUDE.md "Validation campaign" / feedback_validation_spec.md): these
# numbers are NOT aspirational targets picked from a paper. They MUST be
# replaced with the actual first-successful-run values (floored, not
# rounded up) once run_genesis.R has been executed for real. Do not treat
# these placeholders as validated tolerances.
# ─────────────────────────────────────────────────────────────────────────────
# TODO(Task 6 rerun): set from the first successful GENESIS run.
TOL_KING_MAX_ABS_DIFF = 0.05          # PLACEHOLDER
TOL_KING_CORR = 0.95                  # PLACEHOLDER
TOL_PCAIR_ABS_CORR = 0.90             # PLACEHOLDER, per PC axis (top few PCs)
TOL_PCRELATE_MAX_ABS_DIFF = 0.05      # PLACEHOLDER
TOL_PCRELATE_CORR = 0.90              # PLACEHOLDER
TOL_PCRELATE_RELATED_PAIR_ABS_DIFF = 0.10  # PLACEHOLDER (known PO pairs)
N_PCAIR_AXES_TO_CHECK = 5             # leading PCs checked by |correlation|

N_PCS = 10
R2_PRUNE_THRESHOLD = 0.1
N_PCS_ADJUST = 3


def _load_genesis_matrix_tsv(path: Path, sample_ids: list[str]) -> np.ndarray:
    """Load a `sample_id` + (n, n) matrix TSV (KING or PC-Relate), reordered
    to `sample_ids` on both axes."""
    df = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    df = df.set_index("sample_id")
    df = df.reindex(index=sample_ids)
    df = df[sample_ids]  # reorder columns too
    return df.to_numpy(dtype=np.float64)


def _load_genesis_pcs_tsv(path: Path, sample_ids: list[str]) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    df = df.set_index("sample_id")
    df = df.reindex(index=sample_ids)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    return df[pc_cols].to_numpy(dtype=np.float64)


def _offdiag(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    g_npy = OUT / "G.npy"
    meta_npz = OUT / "meta.npz"
    if not g_npy.exists() or not meta_npz.exists():
        print(
            f"[compare] ABORT: {g_npy} / {meta_npz} not found. "
            "Run export_fixture.py first."
        )
        return 1

    G = np.load(g_npy)
    meta = np.load(meta_npz, allow_pickle=True)
    sample_ids = [str(s) for s in meta["sample_ids"]]
    related_pairs = meta["related_pairs"]  # (k, 2) int64
    n, m = G.shape
    print(f"[compare] loaded G.npy: n={n}, m={m}")

    G_t = torch.from_numpy(G).to(torch.float64)

    # ── torchgenomics side ───────────────────────────────────────────────────
    print("[compare] torchgenomics: king_robust_kinship(G) [full, unpruned panel]")
    king_tg = king_robust_kinship(G_t).numpy()

    print(f"[compare] torchgenomics: ld_prune_independent(G, r2_threshold={R2_PRUNE_THRESHOLD})")
    pruned_idx_t = ld_prune_independent(G_t, r2_threshold=R2_PRUNE_THRESHOLD)
    pruned_idx = pruned_idx_t.numpy()
    # Sanity check: reproduces the same subset export_fixture.py wrote (both
    # calls use the same default r2_threshold on the same G; a mismatch here
    # would mean the fixture is stale relative to this script/ld_prune_independent).
    if "pruned_idx" in meta.files:
        stale = not np.array_equal(pruned_idx, meta["pruned_idx"])
        if stale:
            print(
                "[compare] WARNING: freshly-computed pruned_idx differs from "
                "meta.npz's pruned_idx -- re-run export_fixture.py so the "
                "GENESIS pruned_snp_ids.txt matches this script's input."
            )
    Gp_t = G_t[:, pruned_idx_t]

    print(f"[compare] torchgenomics: pc_air(G_pruned, king) n_pcs={N_PCS}")
    king_tg_t = torch.from_numpy(king_tg)
    pcs_tg = pc_air(Gp_t, king_tg_t, n_pcs=N_PCS).numpy()

    print(f"[compare] torchgenomics: pc_relate(G_pruned, pcs) n_pcs_adjust={N_PCS_ADJUST}")
    kin_tg = pc_relate(
        Gp_t, torch.from_numpy(pcs_tg), n_pcs_adjust=N_PCS_ADJUST
    ).numpy()

    # ── GENESIS side (optional) ──────────────────────────────────────────────
    king_tsv = OUT / "genesis_king_kinship.tsv"
    pcair_tsv = OUT / "genesis_pcair_pcs.tsv"
    pcrelate_tsv = OUT / "genesis_pcrelate_kinship.tsv"

    if not (king_tsv.exists() and pcair_tsv.exists() and pcrelate_tsv.exists()):
        print(
            "\n[compare] GENESIS golden outputs not found under "
            f"{OUT} -- skipping GENESIS comparison (this is expected until "
            "install.sh + fetch_data.sh + run_genesis.R have actually been "
            "run; see README.md)."
        )
        print("[compare] torchgenomics-only path executed successfully. PASS (no GENESIS gate).")
        return 0

    print("\n[compare] GENESIS golden outputs found -- running full comparison")
    king_gn = _load_genesis_matrix_tsv(king_tsv, sample_ids)
    pcs_gn = _load_genesis_pcs_tsv(pcair_tsv, sample_ids)
    kin_gn = _load_genesis_matrix_tsv(pcrelate_tsv, sample_ids)

    results: list[tuple[str, float, float, bool]] = []  # (name, observed, floor, passed)

    # --- KING-robust ---------------------------------------------------------
    king_diff = np.abs(_offdiag(king_tg) - _offdiag(king_gn))
    king_max_abs_diff = float(np.nanmax(king_diff))
    king_corr = _corr(_offdiag(king_tg), _offdiag(king_gn))
    results.append(("KING max|diff|", king_max_abs_diff, TOL_KING_MAX_ABS_DIFF,
                     king_max_abs_diff <= TOL_KING_MAX_ABS_DIFF))
    results.append(("KING corr", king_corr, TOL_KING_CORR, king_corr >= TOL_KING_CORR))

    # --- PC-AiR: per-axis |correlation| (signs arbitrary) --------------------
    k_check = min(N_PCAIR_AXES_TO_CHECK, pcs_tg.shape[1], pcs_gn.shape[1])
    for axis in range(k_check):
        c = abs(_corr(pcs_tg[:, axis], pcs_gn[:, axis]))
        results.append((f"PC-AiR |corr| axis {axis + 1}", c, TOL_PCAIR_ABS_CORR,
                         c >= TOL_PCAIR_ABS_CORR))

    # --- PC-Relate -------------------------------------------------------------
    kin_diff = np.abs(_offdiag(kin_tg) - _offdiag(kin_gn))
    kin_max_abs_diff = float(np.nanmax(kin_diff))
    kin_corr = _corr(_offdiag(kin_tg), _offdiag(kin_gn))
    results.append(("PC-Relate max|diff|", kin_max_abs_diff, TOL_PCRELATE_MAX_ABS_DIFF,
                     kin_max_abs_diff <= TOL_PCRELATE_MAX_ABS_DIFF))
    results.append(("PC-Relate corr", kin_corr, TOL_PCRELATE_CORR,
                     kin_corr >= TOL_PCRELATE_CORR))

    # --- PC-Relate on the known related (parent-offspring) pairs ------------
    print("\n[compare] known related (parent-offspring) pairs:")
    print(f"{'(i, j)':>12}  {'tg_kin':>10}  {'genesis_kin':>12}  {'|diff|':>8}")
    pair_diffs = []
    for p, c in related_pairs.tolist():
        tg_val = float(kin_tg[p, c])
        gn_val = float(kin_gn[p, c])
        d = abs(tg_val - gn_val)
        pair_diffs.append(d)
        print(f"{f'({p}, {c})':>12}  {tg_val:>10.4f}  {gn_val:>12.4f}  {d:>8.4f}")
    pair_max_abs_diff = float(np.max(pair_diffs)) if pair_diffs else float("nan")
    results.append(("PC-Relate related-pair max|diff|", pair_max_abs_diff,
                     TOL_PCRELATE_RELATED_PAIR_ABS_DIFF,
                     pair_max_abs_diff <= TOL_PCRELATE_RELATED_PAIR_ABS_DIFF))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Metric':<38}{'Observed':>12}{'Tolerance':>12}{'Result':>10}")
    print("-" * 72)
    all_pass = True
    for name, observed, floor, passed in results:
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"{name:<38}{observed:>12.4f}{floor:>12.4f}{status:>10}")
    print("=" * 72)

    if all_pass:
        print("\n[compare] ALL GATES PASS.")
        return 0
    else:
        print(
            "\n[compare] AT LEAST ONE GATE FAILED. Tolerances above are "
            "PLACEHOLDERS pending the first real GENESIS run -- if this is "
            "that first run, replace the TOL_* constants at the top of this "
            "file with the observed values (floored, not the placeholders) "
            "before treating this as a genuine regression."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
