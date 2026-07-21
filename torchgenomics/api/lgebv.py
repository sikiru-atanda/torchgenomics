"""Local Genomic Estimated Breeding Values (LGEBV).

One-call API: ``torchgenomics.api.lgebv(y, G, haplo_blocks)`` returns
the per-haplo-block sum of marker effects, plus a sign-based
favorable / unfavorable classification.

Primary references
------------------
1. Endelman JB (2011). "Ridge regression and other kernels for genomic
   selection with R package rrBLUP." *The Plant Genome* 4(3):250-255.
   doi:10.3835/plantgenome2011.08.0024
2. Bernardo R (2014). "Genomewide selection when major genes are known."
   *Crop Science* 54(1):68-75. doi:10.2135/cropsci2013.05.0315
3. Pandit et al. (2026). *Theoretical and Applied Genetics* — barley leaf
   rust LGEBV application (motivating paper).
4. Lehermeier C, Schön CC, de los Campos G (2015). "Assessment of genetic
   heterogeneity in structured plant populations using multivariate
   whole-genome regression models." *Genetics* 201(1):323-337.
   doi:10.1534/genetics.115.177873

Algorithm (see the ``lgebv`` docstring for full math).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..config import STAT_DTYPE
from ._decorator import tool
from ._helpers import timed
from ._results import LGEBVResult

# --- block duck-typing -------------------------------------------------------

def _block_field(blk: Any, *names: str, default: Any = "") -> Any:
    """Return the first present attribute among ``names``.

    Handles both ``LDBlock`` (chrom/start/end nested under ``.region``)
    and ``HaplotypeBlock`` (chrom/start/end at the top level).
    """
    for n in names:
        if hasattr(blk, n):
            val = getattr(blk, n)
            if val is not None:
                return val
    return default


def _block_chrom(blk: Any) -> str:
    if hasattr(blk, "region") and getattr(blk.region, "chr", None) is not None:
        return str(blk.region.chr)
    return str(_block_field(blk, "chr", "chrom", default=""))


def _block_start(blk: Any) -> int:
    if hasattr(blk, "region") and getattr(blk.region, "start", None) is not None:
        return int(blk.region.start)
    return int(_block_field(blk, "start", default=0))


def _block_end(blk: Any) -> int:
    if hasattr(blk, "region") and getattr(blk.region, "end", None) is not None:
        return int(blk.region.end)
    return int(_block_field(blk, "end", default=0))


def _block_id(blk: Any, fallback: str) -> str:
    if hasattr(blk, "region") and getattr(blk.region, "region_id", None) is not None:
        return str(blk.region.region_id)
    if hasattr(blk, "block_id") and blk.block_id is not None:
        return str(blk.block_id)
    return fallback


# --- input coercion ----------------------------------------------------------

def _to_tensor(x: Any, name: str) -> Tensor:
    """Accept torch.Tensor / np.ndarray / pd.Series and return a STAT_DTYPE Tensor."""
    if isinstance(x, Tensor):
        return x.to(STAT_DTYPE)
    if isinstance(x, pd.Series):
        return torch.tensor(x.to_numpy(), dtype=STAT_DTYPE)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=STAT_DTYPE)
    if isinstance(x, (list, tuple)):
        return torch.tensor(list(x), dtype=STAT_DTYPE)
    raise TypeError(f"{name} must be torch.Tensor, np.ndarray, or pd.Series; got {type(x)!r}")


def _detect_ploidy(G: Tensor) -> int:
    """Round ``G.max()`` to the nearest of {2, 4, 6, 8}."""
    g_max = float(G.max().item())
    if not np.isfinite(g_max) or g_max <= 0:
        return 2
    candidates = (2, 4, 6, 8)
    # Round up so dosages like 3.7 land at 4 (tetraploid), not 2.
    for c in candidates:
        if g_max <= c + 1e-6:
            return c
    return 8


def _standardize_G(G: Tensor, ploidy: int) -> Tensor:
    """Center and ploidy-aware scale.

    For each SNP j with allele frequency ``p_j = mean(G_j) / ploidy``:

        Z[:, j] = (G[:, j] - ploidy * p_j) / sqrt(ploidy * p_j * (1 - p_j))

    Monomorphic SNPs (p_j ∈ {0, 1}) get zeroed out (their column variance
    is 0; ridge regression handles the kernel without them).
    """
    p = G.mean(dim=0) / float(ploidy)             # (m,)
    centered = G - (float(ploidy) * p).unsqueeze(0)
    scale = torch.sqrt(float(ploidy) * p * (1.0 - p))  # (m,)
    safe = scale > 1e-10
    Z = torch.zeros_like(centered)
    Z[:, safe] = centered[:, safe] / scale[safe].unsqueeze(0)
    return Z


# --- core rrBLUP fit ---------------------------------------------------------

def _fit_rrblup_marker_effects(
    y: Tensor,
    Z: Tensor,
    h2: float | None,
    device: torch.device | None = None,
) -> tuple[Tensor, float, float, float]:
    """Fit rrBLUP / G-BLUP and return per-marker effect estimates.

    Returns
    -------
    u_hat : (m,) Tensor
        BLUP marker effects in the standardized space ``Z``.
    h2_used : float
        Heritability value used for the shrinkage.
    sig2_g : float
        Genetic variance (total, on the standardized scale).
    sig2_e : float
        Residual variance.

    Notes
    -----
    rrBLUP equivalence:

        y = 1·μ + Z·u + e,  u ~ N(0, σ²_u I),  e ~ N(0, σ²_e I)
        K = Z Z^T / m,   K ≡ G-BLUP kernel
        u_hat = (σ²_u / m) · Z^T · V^(-1) · (y - X β̂)
        with V = Z Z^T · σ²_u + σ²_e I = m·K·σ²_u + σ²_e I

    When ``h2`` is supplied, we use λ = m·(1 - h²)/h² and the closed-form
    ridge solution. When ``h2`` is None, we REML-estimate σ²_g, σ²_e via
    ``SingleTraitLMM().fit_null`` on K = Z Z^T / m (single intercept), then
    recover u via the BLUP back-solve.
    """
    n, m = Z.shape
    y = y.to(STAT_DTYPE)
    Z = Z.to(STAT_DTYPE)
    if device is not None:
        y = y.to(device)
        Z = Z.to(device)

    # --- closed-form ridge if user provided h² -----------------------------
    if h2 is not None:
        if not (0.0 < float(h2) < 1.0):
            raise ValueError(f"h2 must be in (0, 1); got {h2!r}")
        h2_used = float(h2)
        y_centered = y - y.mean()
        lam = float(m) * (1.0 - h2_used) / h2_used    # = sig2_e / sig2_u
        # Use the dual form: u = Z^T (Z Z^T + λ I)^-1 (y - 1 μ)
        # This avoids an (m x m) solve when m > n.
        A = Z @ Z.T + lam * torch.eye(n, dtype=STAT_DTYPE, device=Z.device)
        alpha = torch.linalg.solve(A, y_centered)     # (n,)
        u_hat = Z.T @ alpha                            # (m,)
        # Surrogate variance components for reporting (consistent with h2)
        sig2_e = float((y_centered ** 2).mean().item()) * (1.0 - h2_used)
        sig2_g = sig2_e * h2_used / max(1.0 - h2_used, 1e-12)
        return u_hat, h2_used, sig2_g, sig2_e

    # --- REML-estimate via SingleTraitLMM ----------------------------------
    from ..models.single_trait_lmm import SingleTraitLMM

    K = (Z @ Z.T) / float(m)                          # (n, n) — rrBLUP kernel
    X0 = torch.ones((n, 1), dtype=STAT_DTYPE, device=Z.device)

    model = SingleTraitLMM()
    nf = model.fit_null(y, X0, K)

    sig2_g = float(nf.sig2_g)                         # genetic variance (kernel scale)
    sig2_e = float(nf.sig2_e)
    h2_used = sig2_g / max(sig2_g + sig2_e, 1e-300)

    # GLS estimate of the intercept under the null fit:
    # β̂ = (X0^T V^-1 X0)^-1 X0^T V^-1 y, with V = σ²_g K + σ²_e I
    # Using eigen-rotation U^T V U = σ²_g diag(λ) + σ²_e I (already cached
    # in nf.Y_rot, nf.X0_rot, nf.eigenvalues), we solve in rotated space.
    U = nf.eigenvectors                                # (n, n)
    evals = nf.eigenvalues                             # (n,)
    Y_rot = nf.Y_rot.to(STAT_DTYPE)
    X0_rot = nf.X0_rot.to(STAT_DTYPE)
    D_inv = 1.0 / (sig2_g * evals + sig2_e)            # (n,) — diag of V^-1 in rotated space

    XtVi = X0_rot.T * D_inv.unsqueeze(0)               # (1, n)
    XtViX = XtVi @ X0_rot                              # (1, 1)
    XtViy = XtVi @ Y_rot                               # (1,)
    beta_hat = torch.linalg.solve(XtViX, XtViy)        # (1,)
    mu_hat = float(beta_hat[0].item())

    # BLUP for total breeding values g = Z u in original space:
    # g_hat = σ²_g K V^-1 (y - X β̂)
    # u_hat = (σ²_g / m) Z^T V^-1 (y - X β̂)
    # Compute V^-1 (y - μ 1) via U D_inv U^T
    resid_rot = Y_rot - X0_rot @ beta_hat              # (n,)
    Vi_resid_rot = D_inv * resid_rot                   # (n,)
    Vi_resid = U @ Vi_resid_rot                        # (n,)  back to original basis
    u_hat = (sig2_g / float(m)) * (Z.T @ Vi_resid)     # (m,)

    return u_hat, h2_used, sig2_g, sig2_e


# --- public API --------------------------------------------------------------

@tool(
    name="tg_lgebv",
    title="Local Genomic Estimated Breeding Values (per haplo-block)",
    description=(
        "Sum rrBLUP / G-BLUP marker effects within each haplo-block to "
        "obtain per-block 'local' breeding values (LGEBV). Useful for "
        "identifying favorable haplotype regions in genomic selection "
        "and resistance-allele mining (Pandit et al. 2026 / Endelman 2011 / "
        "Bernardo 2014). BLUEs come in from upstream (e.g. ASReml-R MET "
        "fit); this function does NOT compute them. Haplo-blocks come "
        "from torchgenomics.ld.detect_blocks or from a HaplotypeGWAS "
        "block construction (duck-typed via .variant_indices)."
    ),
    long_running=False,
    category="pgs",
    tags=["lgebv", "rrblup", "gblup", "haplotype", "genomic-selection",
          "breeding-value", "pandit-2026"],
)
def lgebv(
    y: Any,
    G: Any,
    haplo_blocks: list,
    *,
    h2: float | None = None,
    favorable_direction: Literal["negative", "positive"] = "negative",
    standardize_G: bool = True,
    method: Literal["rrblup", "gblup"] = "rrblup",
    device: Any = None,
    return_marker_effects: bool = False,
) -> LGEBVResult:
    """Compute per-haplo-block Local Genomic Estimated Breeding Values.

    BLUEs are produced upstream (e.g. ASReml-R's FA(k) MET fit with
    genotype-as-fixed); this function does not compute them. It takes
    the pre-computed BLUE vector ``y``, the genotype dosage matrix ``G``,
    and a list of haplo-blocks, fits rrBLUP / G-BLUP, then sums marker
    effects within each block to produce per-block breeding-value
    contributions.

    Mathematical model
    ------------------
    Endelman (2011) rrBLUP / G-BLUP:

    .. math::

        \\mathbf{y} = \\mathbf{1}\\mu + \\mathbf{Z}\\mathbf{u}
        + \\mathbf{e},\\ \\mathbf{u}\\sim\\mathcal{N}(\\mathbf{0},
        \\sigma_u^2\\mathbf{I}),\\ \\mathbf{e}\\sim\\mathcal{N}(
        \\mathbf{0}, \\sigma_e^2\\mathbf{I}).

    The G-BLUP kernel is :math:`\\mathbf{K} = \\mathbf{Z}\\mathbf{Z}^\\top / m`.
    Per-marker BLUP effects:

    .. math::

        \\hat{\\mathbf{u}} = \\frac{\\sigma_u^2}{m} \\mathbf{Z}^\\top
        \\mathbf{V}^{-1}(\\mathbf{y} - \\mathbf{1}\\hat{\\mu}),

    with :math:`\\mathbf{V} = m\\mathbf{K}\\sigma_u^2 + \\sigma_e^2\\mathbf{I}`.
    Per-block LGEBV is then :math:`\\text{LGEBV}_b = \\sum_{j \\in b}
    \\hat{u}_j` (sum of marker effects in the block, on the standardized
    scale). This per-region aggregation follows Lehermeier et al. (2015,
    *Genetics*).

    Parameters
    ----------
    y : torch.Tensor | np.ndarray | pd.Series, shape (n,)
        Pre-computed BLUEs (one per genotype). BLUEs are produced upstream
        (e.g. ASReml-R's FA(k) MET fit with genotype-as-fixed);
        this function does not compute them.
    G : torch.Tensor | np.ndarray, shape (n, m)
        Genotype dosage matrix. Diploid in ``[0, 2]`` or polyploid in
        ``[0, ploidy]``. Ploidy is inferred from ``G.max()`` when
        ``standardize_G=True``.
    haplo_blocks : list[LDBlock] | list[HaplotypeBlock]
        Block partition. Duck-typed via ``.variant_indices``; accepts both
        :class:`torchgenomics.ld.LDBlock` (chrom under ``.region``) and
        :class:`torchgenomics.models.haplotype_gwas.HaplotypeBlock` (chrom
        at the top level).
    h2 : float | None, default None
        Shrinkage parameter (heritability). If ``None``, REML-estimated
        via :class:`torchgenomics.models.SingleTraitLMM` on K = Z Z^T / m.
        If ``float``, the closed-form ridge λ = m(1 - h²)/h² is used.
    favorable_direction : {"negative", "positive"}, default "negative"
        Sign convention. ``"negative"`` matches resistance traits (lower
        trait value = more favorable, as in Pandit et al. 2026 for barley
        leaf rust severity). ``"positive"`` flips the classification for
        traits where higher = better (yield, etc.).
    standardize_G : bool, default True
        If True, center and scale per-SNP by sqrt(k·p·(1-p)) (ploidy-aware
        VanRaden-style per-locus scaling). If False, ``G`` is used as-is.
    method : {"rrblup", "gblup"}, default "rrblup"
        Numerically identical fit; ``"gblup"`` is provided as an alias /
        bookkeeping flag (the per-marker back-solve is the same).
    device : torch.device | None
        Device for computation. Defaults to ``G.device`` (or CPU if numpy).
    return_marker_effects : bool, default False
        If True, populate ``result.marker_effects`` with the ``(m,)``
        per-marker BLUP effect vector.

    Returns
    -------
    LGEBVResult
        See :class:`torchgenomics.api.LGEBVResult`.

    References
    ----------
    Endelman JB (2011). Plant Genome 4(3):250-255.
    Bernardo R (2014). Crop Science 54(1):68-75.
    Pandit et al. (2026). Theoretical and Applied Genetics — barley leaf
    rust LGEBV application (motivating paper).
    Lehermeier C, Schön CC, de los Campos G (2015). Genetics 201(1):323-337.
    """
    if method not in ("rrblup", "gblup"):
        raise ValueError(f"method must be 'rrblup' or 'gblup'; got {method!r}")
    if favorable_direction not in ("negative", "positive"):
        raise ValueError(
            f"favorable_direction must be 'negative' or 'positive'; "
            f"got {favorable_direction!r}"
        )

    with timed() as elapsed:
        # --- coerce inputs ---
        y_t = _to_tensor(y, "y")
        if y_t.ndim != 1:
            if y_t.ndim == 2 and y_t.shape[1] == 1:
                y_t = y_t.squeeze(1)
            else:
                raise ValueError(
                    f"y must be 1-D (n,), got shape {tuple(y_t.shape)}"
                )

        if isinstance(G, Tensor):
            G_t = G.to(STAT_DTYPE)
        else:
            G_t = _to_tensor(G, "G")
        if G_t.ndim != 2:
            raise ValueError(f"G must be 2-D (n, m), got shape {tuple(G_t.shape)}")

        n, m = G_t.shape
        if y_t.shape[0] != n:
            raise ValueError(
                f"y has {y_t.shape[0]} samples but G has {n} rows."
            )

        if device is not None:
            y_t = y_t.to(device)
            G_t = G_t.to(device)

        if not haplo_blocks:
            raise ValueError("haplo_blocks must be a non-empty list of blocks.")

        # --- ploidy detect + standardize ---
        ploidy = _detect_ploidy(G_t) if standardize_G else 2
        if standardize_G:
            Z = _standardize_G(G_t, ploidy=ploidy)
        else:
            Z = G_t

        # --- fit ---
        u_hat, h2_used, sig2_g, sig2_e = _fit_rrblup_marker_effects(
            y_t, Z, h2=h2, device=device,
        )

        # --- per-block aggregation ---
        block_id: list[str] = []
        chrom: list[str] = []
        start: list[int] = []
        end: list[int] = []
        n_variants: list[int] = []
        lgebv_vals: list[float] = []
        block_var_vals: list[float] = []

        for b_idx, blk in enumerate(haplo_blocks):
            if not hasattr(blk, "variant_indices"):
                raise TypeError(
                    f"Block at index {b_idx} has no .variant_indices attribute. "
                    f"Pass LDBlock (torchgenomics.ld) or HaplotypeBlock "
                    f"(torchgenomics.models.haplotype_gwas) instances."
                )
            vi = list(blk.variant_indices) if blk.variant_indices is not None else []
            if len(vi) == 0:
                continue
            vi_t = torch.tensor(vi, dtype=torch.long, device=Z.device)

            u_block = u_hat[vi_t]                         # (m_b,)
            Z_block = Z[:, vi_t]                          # (n, m_b)
            g_block = Z_block @ u_block                   # (n,)

            lgebv_b = float(u_block.sum().item())
            # Population variance (1/n) to keep estimator scale-stable
            var_b = float(g_block.var(unbiased=False).item())

            block_id.append(_block_id(blk, fallback=f"block_{b_idx}"))
            chrom.append(_block_chrom(blk))
            start.append(_block_start(blk))
            end.append(_block_end(blk))
            n_variants.append(len(vi))
            lgebv_vals.append(lgebv_b)
            block_var_vals.append(var_b)

        lgebv_tensor = torch.tensor(lgebv_vals, dtype=STAT_DTYPE)
        block_var_tensor = torch.tensor(block_var_vals, dtype=STAT_DTYPE)

        if favorable_direction == "negative":
            favorable = [float(v) < 0.0 for v in lgebv_vals]
        else:
            favorable = [float(v) > 0.0 for v in lgebv_vals]

        marker_effects = u_hat.detach().cpu() if return_marker_effects else None

        return LGEBVResult(
            runtime_s=elapsed(),
            block_id=block_id,
            chrom=chrom,
            start=start,
            end=end,
            n_variants=n_variants,
            lgebv=lgebv_tensor,
            block_variance=block_var_tensor,
            favorable=favorable,
            marker_effects=marker_effects,
            h2_used=float(h2_used),
            method=method,
        )
