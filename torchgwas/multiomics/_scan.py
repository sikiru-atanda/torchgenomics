"""Genome × molecular-feature mediation scan with cis-window filtering."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..stats.multipletesting import (
    benjamini_hochberg,
    benjamini_yekutieli,
    storey_qvalue,
)
from ._mediate import _mediate_from_nullfit, fit_mediation_null
from ._types import MediationScanResult

logger = logging.getLogger("torchgwas.multiomics")


def _as_tensor(x, dtype=torch.float64) -> Tensor:
    return torch.as_tensor(x, dtype=dtype).detach().cpu()


def _build_pairs(
    snp_pos: Optional[Tensor],
    feat_pos: Optional[Tensor],
    snp_chrom: Optional[list[str]],
    feat_chrom: Optional[list[str]],
    s: int,
    f: int,
    cis_window_bp: Optional[int],
) -> list[tuple[int, int]]:
    """Return (snp_idx, feat_idx) pairs subject to cis-window + chromosome match."""
    if cis_window_bp is None:
        return [(i, j) for i in range(s) for j in range(f)]

    if snp_pos is None or feat_pos is None or snp_chrom is None or feat_chrom is None:
        raise ValueError(
            "cis_window_bp set but snp_pos / feature_pos / snp_chrom / feature_chrom missing."
        )
    if len(snp_chrom) != s or len(feat_chrom) != f:
        raise ValueError("Chromosome label length must match snp/feature counts.")

    snp_pos_np = snp_pos.cpu().numpy().astype(np.int64)
    feat_pos_np = feat_pos.cpu().numpy().astype(np.int64)

    # Bucket features by chromosome with sorted positions for searchsorted.
    by_chrom: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in set(feat_chrom):
        feat_idx = np.array([j for j, fc in enumerate(feat_chrom) if fc == c], dtype=np.int64)
        positions = feat_pos_np[feat_idx]
        order = np.argsort(positions)
        by_chrom[c] = (positions[order], feat_idx[order])

    pairs: list[tuple[int, int]] = []
    w = int(cis_window_bp)
    for i in range(s):
        c = snp_chrom[i]
        if c not in by_chrom:
            continue
        positions, feat_idx_sorted = by_chrom[c]
        lo = np.searchsorted(positions, snp_pos_np[i] - w, side="left")
        hi = np.searchsorted(positions, snp_pos_np[i] + w, side="right")
        for j in feat_idx_sorted[lo:hi]:
            pairs.append((i, int(j)))
    return pairs


def scan_mediation(
    Y,
    G,
    M,
    K,
    *,
    snp_ids: Optional[list[str]] = None,
    feature_ids: Optional[list[str]] = None,
    snp_pos: Optional[Tensor] = None,
    feature_pos: Optional[Tensor] = None,
    snp_chrom: Optional[list[str]] = None,
    feature_chrom: Optional[list[str]] = None,
    cis_window_bp: Optional[int] = 1_000_000,
    covariates=None,
    se: str = "monte-carlo",
    n_mc_draws: int = 10_000,
    fdr_method: str = "bh",
    sensitivity: bool = True,
    seed: Optional[int] = None,
) -> MediationScanResult:
    """Mediation scan over (SNP, feature) pairs filtered by a cis window.

    The null model on Y (eigendecomposition of K + variance components) is
    fit **once** and reused across all pairs.
    """
    Y_t = _as_tensor(Y).reshape(-1)
    G_t = _as_tensor(G)
    M_t = _as_tensor(M)
    K_t = _as_tensor(K)
    n = Y_t.shape[0]
    if G_t.ndim != 2 or G_t.shape[0] != n:
        raise ValueError(f"G must be (n, s); got {tuple(G_t.shape)} for n={n}.")
    if M_t.ndim != 2 or M_t.shape[0] != n:
        raise ValueError(f"M must be (n, f); got {tuple(M_t.shape)} for n={n}.")
    if K_t.shape != (n, n):
        raise ValueError(f"K must be ({n}, {n}); got {tuple(K_t.shape)}.")

    s = G_t.shape[1]
    f = M_t.shape[1]
    snp_ids = snp_ids or [f"snp_{i}" for i in range(s)]
    feature_ids = feature_ids or [f"feat_{j}" for j in range(f)]

    snp_pos_t = _as_tensor(snp_pos, dtype=torch.int64) if snp_pos is not None else None
    feat_pos_t = _as_tensor(feature_pos, dtype=torch.int64) if feature_pos is not None else None

    pairs = _build_pairs(snp_pos_t, feat_pos_t, snp_chrom, feature_chrom,
                         s, f, cis_window_bp)
    logger.info("scan_mediation: %d (snp, feature) pairs to test.", len(pairs))

    nf = fit_mediation_null(Y_t, K_t, covariates=covariates)

    rows: list[dict] = []
    for (i, j) in pairs:
        res = _mediate_from_nullfit(
            nf, G_t[:, i], M_t[:, j],
            se=se, n_mc_draws=n_mc_draws, n_boot=0,
            sensitivity=sensitivity, seed=seed,
        )
        rows.append({
            "snp": snp_ids[i],
            "feature": feature_ids[j],
            "snp_chrom": snp_chrom[i] if snp_chrom else "",
            "snp_pos": int(snp_pos_t[i]) if snp_pos_t is not None else 0,
            "feature_chrom": feature_chrom[j] if feature_chrom else "",
            "feature_pos": int(feat_pos_t[j]) if feat_pos_t is not None else 0,
            "a": res.a, "a_se": res.a_se,
            "b": res.b, "b_se": res.b_se,
            "c": res.c, "c_prime": res.c_prime,
            "indirect": res.indirect, "indirect_se": res.indirect_se,
            "indirect_pvalue": res.indirect_pvalue,
            "ci_lower": res.indirect_ci_lower,
            "ci_upper": res.indirect_ci_upper,
            "proportion_mediated": res.proportion_mediated,
            "inconsistent": res.inconsistent,
            "sensitivity_rho": res.sensitivity_rho,
        })

    if rows:
        p = torch.tensor([r["indirect_pvalue"] for r in rows], dtype=torch.float64)
        if fdr_method == "bh":
            q = benjamini_hochberg(p)
        elif fdr_method == "by":
            q = benjamini_yekutieli(p)
        elif fdr_method == "storey":
            q = storey_qvalue(p)
        else:
            raise ValueError(f"fdr_method must be 'bh', 'by', or 'storey'; got {fdr_method!r}")
        for r, qv in zip(rows, q.tolist()):
            r["q_indirect"] = float(qv)

    return MediationScanResult(
        rows=rows,
        n_pairs=len(rows),
        cis_window_bp=cis_window_bp,
        se_method=se,
        fdr_method=fdr_method,
    )
