"""Expression-matrix normalization helpers for TWAS preprocessing.

Three opt-in transforms commonly applied to RNA-seq counts before
running an observed-expression TWAS:

- :func:`inverse_normal_transform` — Blom-style rank-INT
  (Beasley et al. 2009). The GTEx eQTL pipeline default.
- :func:`quantile_normalize` — column-wise quantile normalization
  (Bolstad et al. 2003). Standard RNA-seq batch-removal step.
- :func:`peer_residualize` — Bayesian hidden-factor adjustment
  (Stegle et al. 2012). Wraps the R ``peer`` package via subprocess,
  mirroring the convention used by :func:`run_updog` (Phase 55) and
  :func:`run_polyorigin` (Phase 56).

All three accept and return float64 :class:`torch.Tensor` objects of
shape ``(n_samples, n_genes)``. They are pure column-wise operations,
so any preprocessing order the caller chooses is valid; the convention
for GTEx-style pipelines is *quantile-normalize → INT → PEER-residualize*.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import HAS_NATIVE_EXPRESSION, _expression_native


# ---------------------------------------------------------------------------
# Rank-based inverse normal transform (Blom)
# ---------------------------------------------------------------------------

def inverse_normal_transform(x: Tensor, c: float = 3.0 / 8.0) -> Tensor:
    """Blom-style rank-based inverse normal transform.

    For a 1-D input ``x`` of length n with ranks ``r_i``::

        rint(x)_i = Phi^{-1}((r_i - c) / (n - 2c + 1))

    where ``Phi^{-1}`` is the inverse standard normal CDF and ``c =
    3/8`` is the Blom-1958 default also used by the GTEx eQTL pipeline.

    For 2-D inputs the transform is applied independently per column,
    matching the GTEx convention (one rank-INT per gene). Ties are
    broken by the average-rank rule (matching ``scipy.stats.rankdata``
    with ``method="average"``).

    Parameters
    ----------
    x : Tensor
        1-D ``(n,)`` or 2-D ``(n_samples, n_features)`` input.
    c : float
        Blom constant. Common values: ``3/8`` (Blom 1958 / GTEx default),
        ``1/2`` (Bliss 1967), ``1/3`` (Tukey 1962). Default ``3/8``.

    Returns
    -------
    Tensor
        Same shape and device as input; dtype promoted to float64.

    References
    ----------
    Beasley TM, Erickson S, Allison DB (2009). Rank-based inverse normal
    transformations are increasingly used, but are they merited?
    *Behavior Genetics* 39:580-595.

    See Also
    --------
    quantile_normalize, peer_residualize
    """
    if not (1 <= x.ndim <= 2):
        raise ValueError(
            f"inverse_normal_transform expects 1-D or 2-D input; got "
            f"shape {tuple(x.shape)}."
        )
    x64 = x.to(torch.float64)
    if x64.ndim == 1:
        return _rank_int_column(x64, c)

    # Native C++ shortcut for 2-D inputs: compute u = (rank-c)/(n-2c+1)
    # per column with average-tie ranks in one OpenMP-parallel pass, then
    # apply sqrt(2)·erfinv(2u-1) via torch on the result. The Python body
    # below is the algorithmic spec and runs unchanged on fallthrough.
    n, n_cols = x64.shape
    if (
        HAS_NATIVE_EXPRESSION
        and not native_disabled()
        and x64.device.type == "cpu"
        and n_cols * n >= 1024
        and n >= 2  # avoid degenerate single-row denom = 1 + (1 - 2c)
    ):
        x_np = x64.contiguous().numpy()
        u_np = np.empty_like(x_np)
        _expression_native.rank_int_u_columns(x_np, float(c), u_np)
        u_t = torch.from_numpy(u_np)
        return math.sqrt(2.0) * torch.erfinv(2.0 * u_t - 1.0)

    out = torch.empty_like(x64)
    for j in range(x64.shape[1]):
        out[:, j] = _rank_int_column(x64[:, j], c)
    return out


def _rank_int_column(col: Tensor, c: float) -> Tensor:
    """Apply rank-INT to a single 1-D column (float64)."""
    n = col.shape[0]
    # Average-rank tie-breaking, matching scipy.stats.rankdata(method="average").
    sorted_vals, sort_idx = torch.sort(col)
    ranks = torch.empty(n, dtype=torch.float64, device=col.device)
    # Assign 1-based ranks; resolve ties by averaging.
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based average
        ranks[sort_idx[i:j + 1]] = avg_rank
        i = j + 1

    u = (ranks - c) / (n - 2.0 * c + 1.0)
    # Map u in (0, 1) to Phi^{-1}(u); use sqrt(2) * erfinv(2u - 1).
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


# ---------------------------------------------------------------------------
# Quantile normalization (column-wise; Bolstad 2003)
# ---------------------------------------------------------------------------

def quantile_normalize(x: Tensor) -> Tensor:
    """Column-wise quantile normalization to a common reference
    distribution.

    For an input ``(n_samples, n_features)`` matrix, the reference
    distribution is the mean across columns of the sorted values; each
    column is then mapped to this reference via rank. After
    normalization every column has the same distribution (same mean,
    median, quantiles) while preserving within-column rank order.

    Ties within a column are resolved by the average-rank rule
    (matches ``scipy.stats.rankdata(method="average")``).

    Parameters
    ----------
    x : Tensor
        ``(n_samples, n_features)`` input; promoted to float64.

    Returns
    -------
    Tensor
        Quantile-normalized matrix of the same shape and device, float64.

    References
    ----------
    Bolstad BM, Irizarry RA, Astrand M, Speed TP (2003). A comparison of
    normalization methods for high density oligonucleotide array data
    based on variance and bias. *Bioinformatics* 19:185-193.

    See Also
    --------
    inverse_normal_transform, peer_residualize
    """
    if x.ndim != 2:
        raise ValueError(
            f"quantile_normalize requires a 2-D input; got shape "
            f"{tuple(x.shape)}."
        )
    x64 = x.to(torch.float64)
    n, m = x64.shape

    # Reference quantiles: mean of column-sorted values.
    sorted_x, sort_idx = torch.sort(x64, dim=0)
    reference = sorted_x.mean(dim=1)  # (n,)

    # Native C++ shortcut: column-wise tie-resolved scatter against
    # the precomputed reference, OpenMP across columns. The Python
    # body below is the algorithmic spec and runs unchanged on
    # fallthrough.
    if (
        HAS_NATIVE_EXPRESSION
        and not native_disabled()
        and x64.device.type == "cpu"
        and m * n >= 1024
    ):
        x_np = x64.contiguous().numpy()
        ref_np = reference.contiguous().numpy()
        out_np = np.empty_like(x_np)
        _expression_native.quantile_normalize_columns(x_np, ref_np, out_np)
        return torch.from_numpy(out_np)

    # Map each column's values back through the reference via rank.
    out = torch.empty_like(x64)
    for j in range(m):
        col = x64[:, j]
        # Average-rank tie-breaking, then look up reference[rank-1].
        # Equivalent: place reference[k] at the index where the k-th
        # smallest column entry sits, averaging across tied positions.
        s = sort_idx[:, j]  # positions of sorted values
        # Assign reference[i] to position s[i]; handle ties by averaging.
        vals = x64[:, j]
        result = torch.empty(n, dtype=torch.float64, device=x64.device)
        i = 0
        while i < n:
            k = i
            while k + 1 < n and vals[s[k + 1]] == vals[s[i]]:
                k += 1
            avg_ref = reference[i:k + 1].mean()
            result[s[i:k + 1]] = avg_ref
            i = k + 1
        out[:, j] = result
    return out


# ---------------------------------------------------------------------------
# PEER residualization (R-subprocess wrapper)
# ---------------------------------------------------------------------------

_PEER_AVAILABLE: dict[str, str | None] = {}


def _check_peer_available(rscript: str | None = None) -> tuple[str, str]:
    """Probe Rscript + peer; return (rscript_path, peer_version).

    Raises RuntimeError with install pointers if R or the PEER package
    is missing. Mirrors the pattern in ``preprocess.dosage_call._check_updog``.
    """
    if rscript is None:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Install R >= 4.0 and ensure Rscript is on "
            "PATH, or pass rscript_bin=<path>. PEER residualization "
            "requires the R 'peer' package (Stegle et al. 2012): "
            "see https://github.com/PMBio/peer for install instructions."
        )

    cached = _PEER_AVAILABLE.get(rscript)
    if cached is not None:
        return rscript, cached

    probe = subprocess.run(
        [rscript, "-e",
         "suppressPackageStartupMessages(library(peer)); "
         'cat(as.character(packageVersion("peer")))'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "R 'peer' package not installed. Install via:\n"
            "  remotes::install_github('PMBio/peer/R_peer')\n"
            "or follow https://github.com/PMBio/peer."
        )
    version = (probe.stdout or "").strip()
    _PEER_AVAILABLE[rscript] = version
    return rscript, version


_PEER_DRIVER_R = r"""
suppressPackageStartupMessages({
    library(peer)
})

argv <- commandArgs(trailingOnly = TRUE)
config_path <- argv[1]
config <- jsonlite::fromJSON(config_path)

# Expected fields: expression_path, covariates_path (or ""),
#                  n_factors, max_iter, bound_tol, var_tol, out_residual_path,
#                  out_summary_path
expr <- as.matrix(read.table(config$expression_path,
                              header = FALSE, sep = "\t",
                              colClasses = "numeric"))

model <- PEER()
PEER_setPhenoMean(model, expr)
PEER_setNk(model, as.integer(config$n_factors))
PEER_setNmax_iterations(model, as.integer(config$max_iter))
PEER_setTolerance(model, as.numeric(config$bound_tol))
PEER_setVarTolerance(model, as.numeric(config$var_tol))

if (nzchar(config$covariates_path)) {
    covs <- as.matrix(read.table(config$covariates_path,
                                  header = FALSE, sep = "\t",
                                  colClasses = "numeric"))
    PEER_setCovariates(model, covs)
}

PEER_update(model)
residuals <- PEER_getResiduals(model)

# Write residual matrix and summary JSON.
write.table(residuals, file = config$out_residual_path,
            row.names = FALSE, col.names = FALSE,
            sep = "\t", quote = FALSE)

summary <- list(
    n_factors_used = as.integer(config$n_factors),
    n_iterations = length(PEER_getBounds(model)),
    converged = TRUE,
    bound = tail(PEER_getBounds(model), 1)
)
writeLines(jsonlite::toJSON(summary, auto_unbox = TRUE),
           con = config$out_summary_path)
"""


def peer_residualize(
    expression: Tensor,
    n_factors: int = 15,
    covariates: Tensor | None = None,
    *,
    max_iterations: int = 1000,
    bound_tolerance: float = 1e-3,
    var_tolerance: float = 1e-5,
    keep_temp_files: bool = False,
    rscript_bin: str | None = None,
) -> tuple[Tensor, dict]:
    """Bayesian hidden-factor adjustment via PEER (Stegle et al. 2012).

    Wraps the R ``peer`` package through ``Rscript`` — same subprocess
    pattern as :func:`run_updog` and :func:`run_polyorigin`. Writes the
    expression matrix to a temporary file, calls a small R driver that
    runs ``PEER_setPhenoMean`` → ``PEER_update`` → ``PEER_getResiduals``,
    and reads back the residualised matrix.

    GTEx-recommended ``n_factors`` schedule:

    ====================  ============
    sample size           n_factors
    ====================  ============
    n < 150               15
    150 ≤ n < 250         30
    250 ≤ n < 350         45
    n ≥ 350               60
    ====================  ============

    Parameters
    ----------
    expression : Tensor
        ``(n_samples, n_genes)`` matrix of normalized expression
        (typically post-quantile-norm and rank-INT).
    n_factors : int
        Number of hidden factors to learn. Default 15.
    covariates : Tensor, optional
        ``(n_samples, q)`` known covariates passed to PEER via
        ``PEER_setCovariates``. These are jointly modelled with the
        hidden factors and removed from the residual.
    max_iterations : int
        Maximum variational E-M iterations. Default 1000.
    bound_tolerance : float
        Convergence tolerance on the variational lower bound. Default
        1e-3.
    var_tolerance : float
        Convergence tolerance on per-factor variance. Default 1e-5.
    keep_temp_files : bool
        If True, do not delete the temporary R-input / R-output files
        on success. Useful for debugging. Default False.
    rscript_bin : str, optional
        Path to ``Rscript``. Default: auto-detect via ``shutil.which``.

    Returns
    -------
    residualized : Tensor
        ``(n_samples, n_genes)`` residual matrix, float64.
    summary : dict
        ``{"n_factors_used": int, "n_iterations": int, "converged":
        bool, "bound": float}``.

    Raises
    ------
    RuntimeError
        If Rscript or the R ``peer`` package is not installed.
    ValueError
        If ``n_factors`` is non-positive, or larger than ``n_samples``,
        or if shapes are inconsistent.

    References
    ----------
    Stegle O, Parts L, Piipari M, Winn J, Durbin R (2012). Using
    probabilistic estimation of expression residuals (PEER) to obtain
    increased power and interpretability of gene expression analyses.
    *Nature Protocols* 7:500-507.

    See Also
    --------
    inverse_normal_transform, quantile_normalize
    """
    if expression.ndim != 2:
        raise ValueError(
            f"expression must be 2-D (n_samples, n_genes); got shape "
            f"{tuple(expression.shape)}."
        )
    n_samples, n_genes = expression.shape
    if n_factors <= 0:
        raise ValueError(f"n_factors must be positive; got {n_factors}.")
    if n_factors > n_samples:
        raise ValueError(
            f"n_factors ({n_factors}) cannot exceed n_samples "
            f"({n_samples}); PEER cannot fit more factors than samples."
        )
    if covariates is not None:
        if covariates.ndim != 2 or covariates.shape[0] != n_samples:
            raise ValueError(
                f"covariates must be (n_samples, q) with n_samples = "
                f"{n_samples}; got shape {tuple(covariates.shape)}."
            )

    rscript_path, _ = _check_peer_available(rscript_bin)

    expr_np = expression.detach().cpu().to(torch.float64).numpy()
    cov_np = (
        covariates.detach().cpu().to(torch.float64).numpy()
        if covariates is not None else None
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="torchgenomics_peer_"))
    try:
        expr_path = tmpdir / "expression.tsv"
        cov_path = tmpdir / "covariates.tsv"
        driver_path = tmpdir / "driver.R"
        config_path = tmpdir / "config.json"
        residual_path = tmpdir / "residuals.tsv"
        summary_path = tmpdir / "summary.json"

        np.savetxt(expr_path, expr_np, delimiter="\t")
        if cov_np is not None:
            np.savetxt(cov_path, cov_np, delimiter="\t")

        driver_path.write_text(_PEER_DRIVER_R)
        config_path.write_text(json.dumps({
            "expression_path": str(expr_path),
            "covariates_path": str(cov_path) if cov_np is not None else "",
            "n_factors": int(n_factors),
            "max_iter": int(max_iterations),
            "bound_tol": float(bound_tolerance),
            "var_tol": float(var_tolerance),
            "out_residual_path": str(residual_path),
            "out_summary_path": str(summary_path),
        }))

        run = subprocess.run(
            [rscript_path, str(driver_path), str(config_path)],
            capture_output=True, text=True, check=False,
        )
        if run.returncode != 0:
            raise RuntimeError(
                "PEER R driver failed. stderr:\n" + (run.stderr or "")
            )

        residuals = torch.from_numpy(
            np.loadtxt(residual_path, delimiter="\t").astype(np.float64)
        )
        summary = json.loads(summary_path.read_text())
    finally:
        if not keep_temp_files:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"[peer_residualize] kept temp files at {tmpdir}")

    return residuals, summary
