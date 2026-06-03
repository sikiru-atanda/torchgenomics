"""Genome × molecular-feature mediation scan with cis-window filtering."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from ..stats.multipletesting import (
    benjamini_hochberg,
    benjamini_yekutieli,
    eigenmt_adjust,
    storey_qvalue,
)
from ._mediate import _mediate_from_nullfit, fit_mediation_null
from ._scan_batched import batched_scan_pairs, batched_scan_pairs_streaming
from ._types import MediationScanResult

logger = logging.getLogger("torchgenomics.multiomics")


def _as_tensor(x, dtype=torch.float64) -> Tensor:
    return torch.as_tensor(x, dtype=dtype).detach().cpu()


def _build_pairs(
    snp_pos: Tensor | None,
    feat_pos: Tensor | None,
    snp_chrom: list[str] | None,
    feat_chrom: list[str] | None,
    s: int,
    f: int,
    cis_window_bp: int | None,
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
    snp_ids: list[str] | None = None,
    feature_ids: list[str] | None = None,
    snp_pos: Tensor | None = None,
    feature_pos: Tensor | None = None,
    snp_chrom: list[str] | None = None,
    feature_chrom: list[str] | None = None,
    cis_window_bp: int | None = 1_000_000,
    covariates=None,
    se: str = "monte-carlo",
    n_mc_draws: int = 10_000,
    fdr_method: str = "bh",
    sensitivity: bool = True,
    seed: int | None = None,
    prefilter: str | None = None,
    coloc_sumstats=None,
    coloc_threshold: float = 0.5,
    device=None,
    block_size: tuple[int, int] = (256, 64),
    batched: bool | None = None,
    streaming: bool | None = None,
) -> MediationScanResult:
    """Mediation scan over (SNP, feature) pairs filtered by a cis window.

    The null model on Y (eigendecomposition of K + variance components) is
    fit **once** and reused across all pairs.

    Parameters
    ----------
    streaming : bool, optional
        When True (or None on CPU with ``len(pairs) >= 1000``), the
        SNP-block dispatch lazy-rotates one block at a time instead of
        materializing the full ``G_r = U^T @ G`` rotated genotype
        matrix. Peak memory drops from
        ``2 × n × s_variants × 8 B`` (G + G_r) to
        ``n × s_variants × 8 B + n × s_b × 8 B`` (G + one rotated
        SNP-block). Behavioral parity to float64 tolerance.
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

    # Optional coloc-based prefilter — drops (SNP, feature) pairs whose feature
    # shows no colocalisation signal between Y and M in the cis window.
    if prefilter == "coloc":
        if coloc_sumstats is None:
            raise ValueError(
                "prefilter='coloc' requires coloc_sumstats=(y_sumstats, m_sumstats)."
            )
        from ._prefilter import coloc_prefilter_pairs
        y_ss, m_ss = coloc_sumstats
        pairs, _pp_h4 = coloc_prefilter_pairs(
            pairs, y_ss, m_ss,
            snp_chrom=snp_chrom,
            snp_pos=snp_pos_t.tolist() if snp_pos_t is not None else None,
            feature_chrom=feature_chrom,
            feature_pos=feat_pos_t.tolist() if feat_pos_t is not None else None,
            cis_window_bp=int(cis_window_bp or 1_000_000),
            coloc_threshold=coloc_threshold,
        )
    elif prefilter is not None:
        raise ValueError(f"prefilter must be None or 'coloc'; got {prefilter!r}")

    logger.info("scan_mediation: %d (snp, feature) pairs to test.", len(pairs))

    nf = fit_mediation_null(Y_t, K_t, covariates=covariates)

    # Dispatch: batched path when explicitly requested, when device is CUDA,
    # or when the pair count is large enough to amortise the fixed rotation cost.
    use_batched = batched if batched is not None else _should_batch(device, pairs)

    # Streaming dispatch: default ON whenever batched is selected, since the
    # streaming variant is exact (rotation is linear, output is bit-for-bit
    # equal to the eager batched path on identical SE seeds) and saves
    # 50 % peak memory by never materializing the full rotated genotype.
    use_streaming = (
        streaming if streaming is not None
        else (use_batched and len(pairs) >= 1000)
    )

    rows: list[dict] = []
    if use_batched and pairs:
        scan_fn = batched_scan_pairs_streaming if use_streaming else batched_scan_pairs
        scan_rows = scan_fn(
            nf, G_t, M_t, pairs,
            se=se, n_mc_draws=n_mc_draws, n_boot=0,
            sensitivity=sensitivity, seed=seed,
            block_size=block_size,
        )
        for (i, j), res in zip(pairs, scan_rows):
            rows.append({
                "snp": snp_ids[i],
                "feature": feature_ids[j],
                "snp_chrom": snp_chrom[i] if snp_chrom else "",
                "snp_pos": int(snp_pos_t[i]) if snp_pos_t is not None else 0,
                "feature_chrom": feature_chrom[j] if feature_chrom else "",
                "feature_pos": int(feat_pos_t[j]) if feat_pos_t is not None else 0,
                "a": res["a"], "a_se": res["a_se"],
                "b": res["b"], "b_se": res["b_se"],
                "c": res["c"], "c_prime": res["c_prime"],
                "indirect": res["indirect"],
                "indirect_se": res["indirect_se"],
                "indirect_pvalue": res["indirect_pvalue"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "proportion_mediated": res["proportion_mediated"],
                "inconsistent": res["inconsistent"],
                "sensitivity_rho": res["sensitivity_rho"],
            })
    else:
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
        elif fdr_method == "eigenmt":
            q = _eigenmt_hierarchical_q(rows, p, G_t)
        else:
            raise ValueError(
                f"fdr_method must be 'bh', 'by', 'storey', or 'eigenmt'; got {fdr_method!r}"
            )
        for r, qv in zip(rows, q.tolist()):
            r["q_indirect"] = float(qv)

    return MediationScanResult(
        rows=rows,
        n_pairs=len(rows),
        cis_window_bp=cis_window_bp,
        se_method=se,
        fdr_method=fdr_method,
    )


def _should_batch(device, pairs: list[tuple[int, int]]) -> bool:
    """Return True when the batched path should be used."""
    if device is not None:
        try:
            d = torch.device(device) if isinstance(device, str) else device
            if d.type == "cuda":
                return True
        except Exception:
            pass
    return len(pairs) >= 10_000


def _eigenmt_hierarchical_q(
    rows: list[dict], p: Tensor, G_t: Tensor,
) -> Tensor:
    """Per-feature eigenMT + Bonferroni across genes.

    For each unique feature, collect the rows belonging to that feature, build
    the genotype LD correlation on those SNPs, call ``eigenmt_adjust`` to get a
    gene-level effective number of tests, then Bonferroni-adjust across the
    number of unique features.
    """
    feat_to_rows: dict[str, list[int]] = {}
    for idx, r in enumerate(rows):
        feat_to_rows.setdefault(r["feature"], []).append(idx)
    n_features = len(feat_to_rows)
    snp_id_to_col: dict[str, int] = {}
    q = torch.ones_like(p)
    for feat, row_idxs in feat_to_rows.items():
        # Build LD on the SNPs in this feature's rows.
        snps = [rows[idx]["snp"] for idx in row_idxs]
        # Map SNP string IDs to their G columns via order of first appearance.
        cols = []
        for s_id in snps:
            if s_id not in snp_id_to_col:
                # Fallback: SNPs were named "snp_{i}" by default; otherwise
                # recover column from the row ordering that produced this scan.
                if s_id.startswith("snp_"):
                    try:
                        snp_id_to_col[s_id] = int(s_id.split("_", 1)[1])
                    except ValueError:
                        snp_id_to_col[s_id] = len(snp_id_to_col)
                else:
                    snp_id_to_col[s_id] = len(snp_id_to_col)
            cols.append(snp_id_to_col[s_id])
        cols_t = torch.tensor(cols, dtype=torch.long)
        G_sub = G_t[:, cols_t].to(dtype=torch.float64)
        # Column-standardise, then corr.
        G_sub = G_sub - G_sub.mean(dim=0, keepdim=True)
        std = G_sub.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
        G_std = G_sub / std
        LD = (G_std.T @ G_std) / float(G_std.shape[0])
        p_feat = p[torch.tensor(row_idxs, dtype=torch.long)]
        p_adj, _m_eff = eigenmt_adjust(p_feat, LD)
        # Bonferroni across genes.
        q_feat = torch.clamp(p_adj * n_features, max=1.0)
        for idx, qv in zip(row_idxs, q_feat.tolist()):
            q[idx] = qv
    return q
