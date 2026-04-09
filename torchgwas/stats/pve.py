"""Per-marker phenotypic variance explained (PVE).

GAPIT exports a ``PVE`` column for every significant marker in its GWAS
output and — as of recent versions — only produces the file when (a) at
least one marker exceeds the user's p-value threshold and (b) fewer than
~500 markers do. Neither of those quirks is desirable; we produce a PVE
report regardless, and we compute PVE with a method that *guarantees* the
per-SNP values sum to at most the phenotypic variance (≤ 100%), even in
the presence of strong LD between the significant markers.

**Scaling at industry sizes (m ≥ 10^6 SNPs).** ``compute_pve`` never does
O(m) work beyond a single vectorized ``p < threshold`` mask walk that
selects the ``k`` significant markers. All subsequent arithmetic is on
``(k, k)`` and ``(n, k)`` matrices only, so the function remains fast
even on whole-genome scans. In the default ``method="marginal"`` path the
genotype matrix is **not touched at all** — PVE is computed from the
per-SNP effect size, allele frequency, and phenotypic variance — so the
caller can run it on a 1M-SNP ``ScanResult`` without ever materializing
a ``(n, m)`` tensor. The ``method="joint"`` path accesses only the
``G[:, sig_idx]`` slice (k columns, typically k ≪ 10³), so ``G`` can be a
memory-mapped / on-disk / chunked reader rather than a full in-memory
tensor; only the selected columns are paged in.

Two methods are supported:

* ``method="marginal"`` (default, GAPIT-compatible, fast). Per-SNP

      pve_j = beta_j^2 * var(G_j) / var(y_resid)

  with ``var(G_j) = k f_j (1 - f_j)`` under HWE for ploidy ``k``. The sum
  can exceed 1.0 when significant markers are in LD — we detect that,
  renormalize the per-SNP values to sum to 1.0, and set
  ``PVEResult.capped = True`` so the caller knows. O(k) compute, no
  genotype matrix required.

* ``method="joint"`` (accurate under LD, recommended after LD clumping).
  Residualize ``y`` and the genotypes of the selected markers against the
  covariates, fit one joint OLS regression ``y_resid ~ G_sig``, then
  report per-SNP PVE as the squared **semi-partial correlation** of each
  predictor

      pve_j = beta_j^2 / (G'G)^{-1}_{jj} / (y' y)

  which is exactly the amount by which the joint R² drops when marker
  ``j`` is removed from the model. By construction

      sum_j pve_j  <=  R²_joint  <=  1

  because a semi-partial R² only counts the *unique* variance contributed
  by each predictor. O(n·k + k³) compute; requires ``G[:, sig_idx]``
  (k columns) but not the full ``(n, m)`` matrix.

If ``method="joint"`` is selected and the joint design is rank-deficient
(e.g. perfect collinearity among the significant markers), we transparently
fall back to the marginal path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ..models.base import ScanResult

_GAPIT_SOFT_MARKER_CAP = 500


@dataclass
class PVEResult:
    """Per-marker phenotypic variance explained.

    All float fields are plain Python ``float`` / ``bool`` so the result
    is trivially serializable (JSON, Parquet, CSV).
    """

    snp: list[str]
    chr: list[str]
    pos: list[int]
    beta: Tensor                # (k,) marker effect (copied from the scan)
    pve: Tensor                 # (k,) per-marker PVE (fraction, not percent)
    pve_total: float            # sum of pve (joint R² in "joint" mode)
    phenotypic_variance: float  # var(y_resid) after covariate adjustment
    method: str                 # "joint" or "marginal"
    n_significant: int          # number of markers with p < threshold
    capped: bool                # True if marginal sum had to be renormalized
    rank_deficient: bool = False  # True if joint fit fell back to marginal

    def __len__(self) -> int:
        return len(self.snp)

    def as_table(self) -> dict[str, list]:
        """Return a column-oriented dict suitable for pandas / Parquet."""
        return {
            "snp": list(self.snp),
            "chr": list(self.chr),
            "pos": list(self.pos),
            "beta": self.beta.detach().cpu().tolist(),
            "pve": self.pve.detach().cpu().tolist(),
            "pve_percent": (self.pve * 100.0).detach().cpu().tolist(),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _residualize(
    Y: Tensor, X0: Optional[Tensor]
) -> Tensor:
    """Return ``Y - X0 (X0'X0)^{-1} X0' Y`` or ``Y`` if ``X0 is None``."""
    if X0 is None or X0.shape[1] == 0:
        return Y - Y.mean(dim=0, keepdim=True)
    # Centered covariates + an intercept absorb the grand mean.
    X0f = X0.to(torch.float64)
    ones = torch.ones(X0f.shape[0], 1, dtype=torch.float64, device=X0f.device)
    X = torch.cat([ones, X0f], dim=1)
    beta, *_ = torch.linalg.lstsq(X, Y.to(torch.float64))
    return Y.to(torch.float64) - X @ beta


def _marginal_pve(
    beta: Tensor, af: Tensor, ploidy: int, var_y: float
) -> Tensor:
    """PVE_j = beta_j^2 * k * f_j * (1 - f_j) / var(y)."""
    var_g = float(ploidy) * af * (1.0 - af)
    return (beta**2) * var_g / max(var_y, 1e-30)


def _joint_pve(
    G_sig: Tensor, y: Tensor, X0: Optional[Tensor]
) -> tuple[Tensor, float, float, bool]:
    """Joint OLS PVE via squared semi-partial correlation.

    Returns ``(pve_per_snp, pve_total, var_y_resid, rank_deficient)``.
    The per-SNP values are guaranteed to sum to ``pve_total ≤ 1``.
    """
    y_r = _residualize(y.view(-1, 1), X0).view(-1)  # (n,)
    G_r = _residualize(G_sig.to(torch.float64), X0)  # (n, k)
    tss = float((y_r * y_r).sum().item())
    if tss <= 1e-30:
        return torch.zeros(G_sig.shape[1], dtype=torch.float64), 0.0, 0.0, False

    GtG = G_r.T @ G_r          # (k, k)
    Gty = G_r.T @ y_r          # (k,)

    # Rank check via Cholesky; if it fails the design is collinear.
    try:
        L = torch.linalg.cholesky(GtG + 1e-10 * torch.eye(
            GtG.shape[0], dtype=torch.float64, device=GtG.device))
        beta_hat = torch.cholesky_solve(Gty.view(-1, 1), L).view(-1)
        GtG_inv = torch.cholesky_inverse(L)
    except Exception:
        return torch.tensor([]), 0.0, tss / max(y_r.shape[0] - 1, 1), True

    inv_diag = torch.diagonal(GtG_inv).clamp_min(1e-30)
    sp_sq = (beta_hat**2) / inv_diag / tss  # squared semi-partial correlation
    # Numerical safety: clamp at zero (tiny negatives from round-off).
    sp_sq = sp_sq.clamp_min(0.0)

    r2 = float(((beta_hat * Gty).sum() / tss).item())
    r2 = max(min(r2, 1.0), 0.0)

    # Renormalize if round-off pushed the sum above the joint R².
    sp_sum = float(sp_sq.sum().item())
    if sp_sum > r2 and sp_sum > 0:
        sp_sq = sp_sq * (r2 / sp_sum)

    var_y = tss / max(y_r.shape[0] - 1, 1)
    return sp_sq, r2, var_y, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pve(
    result: ScanResult,
    y: Tensor,
    *,
    G: Optional[Tensor] = None,
    X0: Optional[Tensor] = None,
    significance_threshold: float = 5e-8,
    method: str = "marginal",
    ploidy: int = 2,
) -> PVEResult:
    """Per-marker phenotypic variance explained.

    This function never walks the genotype matrix column-by-column at
    genome-wide scale; it only ever accesses the ``k`` columns whose
    p-values clear ``significance_threshold``. For typical GWAS scans
    that is at most a few hundred columns even when ``m`` is in the
    millions. See the module docstring for the scaling contract.

    Parameters
    ----------
    result : ScanResult
        The full GWAS scan (all markers). Only markers with
        ``result.p < significance_threshold`` contribute.
    y : (n,) Tensor
        Phenotype vector (Gaussian).
    G : (n, m) Tensor or None
        Dosage matrix aligned to ``result`` (column j corresponds to
        ``result.snp[j]``). **Required only when ``method="joint"``** —
        leave as ``None`` for ``method="marginal"``, which reads the
        allele frequency straight out of the ``ScanResult`` and never
        touches genotype data. When supplied, only the columns selected
        by the significance filter are read, so ``G`` may be a memory-
        mapped / on-disk tensor at whole-genome scale. NaNs are replaced
        by the per-marker mean.
    X0 : (n, p) Tensor or None
        Covariates used in the null fit. An intercept is always implicit.
    significance_threshold : float
        Markers with ``p < threshold`` are reported. Default 5e-8.
    method : {"marginal", "joint"}
        ``"marginal"`` (default) — GAPIT-compatible, fast, no genotype
        matrix required, sum renormalized to ≤ 1 if LD inflates it.
        ``"joint"`` — squared semi-partial R² from a single joint OLS
        fit; sum mathematically guaranteed ≤ 1.
    ploidy : int
        Organism ploidy; used in the marginal path to compute
        ``var(G_j) = k f (1 - f)``. Default 2.

    Returns
    -------
    PVEResult
    """
    if method not in ("joint", "marginal"):
        raise ValueError(f"method must be 'joint' or 'marginal'; got {method!r}")
    if method == "joint" and G is None:
        raise ValueError(
            "method='joint' requires the genotype matrix G. "
            "Pass G=... or switch to method='marginal'."
        )

    # Select significant markers from the scan.
    p_vals = result.p.to(torch.float64)
    mask = p_vals < float(significance_threshold)
    sig_idx = torch.nonzero(mask, as_tuple=False).view(-1)
    k = int(sig_idx.numel())

    beta_full = result.beta.to(torch.float64)
    if beta_full.ndim == 2:
        # Multi-trait scan: PVE is univariate-only for now — the user
        # should loop over traits and feed one column at a time.
        raise NotImplementedError(
            "compute_pve expects a single-trait ScanResult; select a trait "
            "column before calling."
        )

    if k == 0:
        empty = torch.zeros(0, dtype=torch.float64)
        var_y_only = float(y.to(torch.float64).var(unbiased=True).item())
        return PVEResult(
            snp=[], chr=[], pos=[],
            beta=empty, pve=empty,
            pve_total=0.0,
            phenotypic_variance=var_y_only,
            method=method,
            n_significant=0,
            capped=False,
        )

    if k > _GAPIT_SOFT_MARKER_CAP:
        warnings.warn(
            f"compute_pve: {k} significant markers exceeds GAPIT's soft "
            f"cap of {_GAPIT_SOFT_MARKER_CAP}; the joint fit may be "
            "numerically sensitive when many markers are in LD. "
            "Consider LD clumping first (torchgwas.postgwas.ld_clump).",
            RuntimeWarning,
        )

    # Extract the selected marker effects + metadata.
    sig_idx_list = sig_idx.tolist()
    snp_sig = [result.snp[i] for i in sig_idx_list]
    chr_sig = [result.chr[i] for i in sig_idx_list]
    pos_sig = [result.pos[i] for i in sig_idx_list]
    beta_sig = beta_full[sig_idx].contiguous()
    af_sig = result.af.to(torch.float64)[sig_idx].contiguous()

    y64 = y.to(torch.float64)

    # --- Joint mode: slice only the selected genotype columns ---
    if method == "joint":
        if G.shape[1] < len(result):
            raise ValueError(
                f"G has {G.shape[1]} markers but the ScanResult has "
                f"{len(result)}; G must be aligned to the full scan."
            )
        # Pull only the k significant columns — works with on-disk / mmapped
        # tensors that support fancy indexing.
        G_sig = G[:, sig_idx].to(torch.float64).clone()
        nan_mask = torch.isnan(G_sig)
        if nan_mask.any():
            col_means = torch.where(
                nan_mask, torch.zeros_like(G_sig), G_sig
            ).sum(dim=0) / (~nan_mask).sum(dim=0).clamp_min(1)
            G_sig = torch.where(nan_mask, col_means.expand_as(G_sig), G_sig)
        pve, pve_total, var_y, rank_def = _joint_pve(G_sig, y64, X0)
        if rank_def or pve.numel() == 0:
            # Fall back to marginal with a rank-deficient flag.
            var_y_marg = float((_residualize(y64.view(-1, 1), X0).view(-1)
                                ).var(unbiased=True).item())
            pve_m = _marginal_pve(beta_sig, af_sig, ploidy, var_y_marg)
            sum_m = float(pve_m.sum().item())
            capped = sum_m > 1.0
            if capped:
                pve_m = pve_m / sum_m
            return PVEResult(
                snp=snp_sig, chr=chr_sig, pos=pos_sig,
                beta=beta_sig, pve=pve_m,
                pve_total=float(pve_m.sum().item()),
                phenotypic_variance=var_y_marg,
                method="marginal",
                n_significant=k,
                capped=capped,
                rank_deficient=True,
            )
        return PVEResult(
            snp=snp_sig, chr=chr_sig, pos=pos_sig,
            beta=beta_sig, pve=pve,
            pve_total=float(pve.sum().item()),
            phenotypic_variance=var_y,
            method="joint",
            n_significant=k,
            capped=False,
            rank_deficient=False,
        )

    # --- Marginal mode ---
    var_y_marg = float((_residualize(y64.view(-1, 1), X0).view(-1)
                        ).var(unbiased=True).item())
    pve_m = _marginal_pve(beta_sig, af_sig, ploidy, var_y_marg)
    sum_m = float(pve_m.sum().item())
    capped = sum_m > 1.0
    if capped:
        pve_m = pve_m / sum_m
    return PVEResult(
        snp=snp_sig, chr=chr_sig, pos=pos_sig,
        beta=beta_sig, pve=pve_m,
        pve_total=float(pve_m.sum().item()),
        phenotypic_variance=var_y_marg,
        method="marginal",
        n_significant=k,
        capped=capped,
    )
