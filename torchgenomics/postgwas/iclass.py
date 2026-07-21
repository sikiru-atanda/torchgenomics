"""iClass clustering on factor-analytic loadings for multi-environment trials (MET).

The **iClass** procedure (Smith et al., 2015; Smith & Cullis, 2018; Smith
et al., 2021) classifies environments from a factor-analytic (FA) variance
model of a MET into groups that share a **polarity pattern** across the
estimated factors. Conceptually, after fitting

    Vg = Lambda @ Lambda.T + Psi,    Lambda in R^{E x k}, Psi diagonal

(with optional column-rotation, e.g. varimax, applied upstream -- the
inputs are taken as the *rotated* loadings), each environment ``e``
contributes a row of the loading matrix Lambda. Replacing each entry by
its sign (``+``, ``-``, ``0``) yields a length-``k`` polarity word --
encoded here as a string of the letters ``"P"`` (positive), ``"N"``
(negative) and ``"Z"`` (near zero, given a user-supplied magnitude
threshold). Environments sharing the same polarity word form an
**iClass cluster** (e.g. ``"PNN"``, ``"PNP"``, ``"PPN"``, ...).

Within a cluster, environments are expected to be **highly positively
genetically correlated** because they respond to the latent factors in
the same direction. The optional :meth:`IClassResult.within_cluster_correlation`
method estimates this directly from genotype BLUPs (the predicted random
genetic effects ``u_hat`` from the MET null fit) by averaging the
off-diagonal entries of the Pearson correlation matrix restricted to the
environments inside each cluster.

This module is a thin post-processing layer that consumes the
``(E, k)`` rotated FA loadings produced by
``torchgenomics.models.multi_env_lmm.MultiEnvLMM.fa_loadings(null_fit)``
(available when ``vg_structure="fa(k)"`` was used during
:meth:`MultiEnvLMM.fit_null`). Its placement in
``torchgenomics.postgwas`` reflects its role as a downstream
interpretation step in the ``met-scan`` -> ``iclass`` -> reporting
pipeline.

Primary sources
---------------
Smith, A. B., Ganesalingam, A., Kuchel, H., & Cullis, B. R. (2015).
Factor analytic mixed models for the provision of grower information
from national crop variety testing programs. *Theoretical and Applied
Genetics*, 128(1), 55-72.

Smith, A. B., & Cullis, B. R. (2018). Plant breeding selection tools
built on factor analytic mixed models for multi-environment trial data.
*Euphytica*, 214, 143.

Smith, A. B., Norman, A., Kuchel, H., & Cullis, B. R. (2021).
Plant variety selection using interaction classes derived from factor
analytic linear mixed models: models with independent variety effects.
*Frontiers in Plant Science*, 12, 737462. (Procedure formalized as
"iClass".)

Worked example
--------------
>>> import numpy as np
>>> from torchgenomics.postgwas import iclass
>>> # Five environments, two rotated FA factors:
>>> Lambda = np.array([
...     [ 0.9,  0.4],   # E1 -> "PP"
...     [ 0.8,  0.3],   # E2 -> "PP"
...     [ 0.7, -0.2],   # E3 -> "PN"
...     [-0.6,  0.5],   # E4 -> "NP"
...     [-0.5,  0.4],   # E5 -> "NP"
... ])
>>> result = iclass(Lambda, env_ids=["E1", "E2", "E3", "E4", "E5"])
>>> sorted(result.cluster_membership.keys())
['NP', 'PN', 'PP']
>>> result.cluster_membership["PP"]
['E1', 'E2']

Polyploid compatibility
-----------------------
The iClass procedure operates entirely on the rotated FA loadings of
the genetic variance matrix, which is ploidy-agnostic once the MET
model has been fit. All shapes and formulas are unchanged for
arbitrary ploidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_loadings(loadings: Any) -> tuple[np.ndarray, list[str] | None]:
    """Coerce ``loadings`` to a 2D float64 numpy array.

    Returns
    -------
    arr : (E, k) ndarray
    env_ids_from_input : list[str] or None
        Env IDs recovered from a pandas index, else ``None``.
    """
    env_ids_from_input: list[str] | None = None

    # Lazy import pandas to avoid hard dependency in module top-level.
    try:  # pragma: no cover - exercised when pandas absent
        import pandas as pd  # type: ignore[import-untyped]
        if isinstance(loadings, pd.DataFrame):
            env_ids_from_input = [str(x) for x in loadings.index.tolist()]
            arr = loadings.to_numpy(dtype=np.float64, copy=True)
            return arr, env_ids_from_input
    except ImportError:  # pragma: no cover
        pass

    # torch.Tensor support -- detach + cpu + numpy.
    try:  # pragma: no cover - exercised when torch present
        import torch
        if isinstance(loadings, torch.Tensor):
            arr = loadings.detach().cpu().numpy().astype(np.float64, copy=True)
            return arr, env_ids_from_input
    except ImportError:  # pragma: no cover
        pass

    arr = np.asarray(loadings, dtype=np.float64)
    return arr, env_ids_from_input


def _polarity(value: float, threshold: float) -> str:
    """Map a scalar loading to its polarity letter (``"P"``/``"N"``/``"Z"``).

    With ``threshold = 0`` (default) the mapping is strict sign:
    positive -> ``"P"``, negative -> ``"N"``, exactly zero -> ``"Z"``.
    With ``threshold > 0`` any ``|value| < threshold`` is collapsed to
    ``"Z"`` (near-zero loading -> indifferent to that factor).
    """
    if abs(value) < threshold:
        return "Z"
    if value > 0.0:
        return "P"
    if value < 0.0:
        return "N"
    return "Z"  # exact zero with threshold == 0.


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IClassResult:
    """Outcome of an iClass partition of MET environments.

    Attributes
    ----------
    cluster_labels : list[str]
        Per-environment polarity word, one entry per row of ``loadings``.
        Each label is a length-``k`` string over the alphabet
        ``{"P", "N", "Z"}``, e.g. ``"PNN"`` for three factors.
    cluster_membership : dict[str, list[str]]
        Mapping from polarity word -> list of environment IDs belonging
        to that cluster. Keys are exactly the unique values of
        ``cluster_labels``; lists preserve the original environment
        order from the input.
    loadings : numpy.ndarray
        The ``(E, k)`` rotated FA loading matrix that was classified
        (copied to a float64 ndarray for stability).
    polarity_matrix : numpy.ndarray
        ``(E, k)`` ndarray of unicode ``"P"``/``"N"``/``"Z"`` symbols
        corresponding cell-wise to ``loadings``.
    n_factors : int
        Number of factors ``k`` (columns of ``loadings``).
    n_envs : int
        Number of environments ``E`` (rows of ``loadings``).
    env_ids : list[str]
        Resolved environment IDs (either user-supplied, recovered from a
        pandas index, or default ``["env_0", "env_1", ...]``).
    threshold : float
        The magnitude threshold used; loadings with ``|x| < threshold``
        are mapped to ``"Z"``.
    """

    cluster_labels: list[str]
    cluster_membership: dict[str, list[str]]
    loadings: np.ndarray
    polarity_matrix: np.ndarray
    n_factors: int
    n_envs: int
    env_ids: list[str] = field(default_factory=list)
    threshold: float = 0.0

    # -- methods ----------------------------------------------------------

    def within_cluster_correlation(
        self,
        genotype_blups: np.ndarray,
    ) -> dict[str, float]:
        """Average within-cluster genetic correlation from genotype BLUPs.

        For each iClass cluster, this method computes the Pearson
        correlation matrix of the BLUP columns assigned to that
        cluster's environments and returns the mean of the strict
        upper-triangle (off-diagonal) entries. Singleton clusters
        return ``float("nan")`` since there is no pair to correlate.

        Formula. Let ``U`` be the ``(N, E)`` BLUP matrix (rows =
        genotypes, columns = environments), and for a cluster ``C``
        with ``|C| = c`` let ``R_C`` be the ``c x c`` Pearson
        correlation matrix of the columns of ``U`` indexed by ``C``.
        Then::

            rho_C = (2 / (c * (c - 1))) * sum_{i < j} R_C[i, j]

        Parameters
        ----------
        genotype_blups : (N, E) ndarray
            BLUPs for ``N`` genotypes across the same ``E``
            environments as ``loadings`` (and in the same column
            order). Equivalent to ``u_hat`` from
            :meth:`MultiEnvLMM.fit_null`.

        Returns
        -------
        dict[str, float]
            Mapping cluster_label -> mean off-diagonal correlation.
        """
        blups = np.asarray(genotype_blups, dtype=np.float64)
        if blups.ndim != 2:
            raise ValueError(
                f"genotype_blups must be 2D (N, E); got shape {blups.shape}"
            )
        if blups.shape[1] != self.n_envs:
            raise ValueError(
                "genotype_blups has E="
                f"{blups.shape[1]} columns; expected {self.n_envs} "
                "to match loadings."
            )

        out: dict[str, float] = {}
        for label, members in self.cluster_membership.items():
            idx = [self.env_ids.index(m) for m in members]
            if len(idx) < 2:
                out[label] = float("nan")
                continue
            sub = blups[:, idx]
            # numpy returns a (c, c) correlation matrix of the columns.
            corr = np.corrcoef(sub, rowvar=False)
            c = corr.shape[0]
            # Mean of strict upper triangle.
            iu, ju = np.triu_indices(c, k=1)
            out[label] = float(np.mean(corr[iu, ju]))
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iclass(
    loadings: Any,
    env_ids: list[str] | None = None,
    threshold: float = 0.0,
) -> IClassResult:
    """Cluster MET environments by FA loading polarity (Smith iClass).

    Parameters
    ----------
    loadings : (E, k) numpy.ndarray | torch.Tensor | pandas.DataFrame
        The **rotated** factor-analytic loading matrix. Rows index
        environments, columns index factors. For ``MultiEnvLMM`` fits
        this is the return value of
        ``MultiEnvLMM.fa_loadings(null_fit)``. A pandas DataFrame is
        accepted; its index is used as ``env_ids`` if the ``env_ids``
        argument is left ``None``.
    env_ids : list[str] or None, default None
        Optional environment identifiers (length ``E``). If ``None``
        and a DataFrame index is available it is used; otherwise
        defaults to ``["env_0", "env_1", ..., "env_{E-1}"]``.
    threshold : float, default 0.0
        Magnitude threshold below which a loading is treated as
        effectively zero (mapped to ``"Z"`` in the polarity word).
        ``threshold = 0`` is strict sign classification, matching the
        original Smith et al. (2021) formulation. Setting
        ``threshold > 0`` gives a more conservative partition that
        treats small loadings as "no opinion" on that factor.

    Returns
    -------
    IClassResult
        See class docstring for the full attribute list.

    Raises
    ------
    ValueError
        If ``loadings`` is not 2D, if ``env_ids`` is supplied but its
        length does not match the number of rows of ``loadings``, or
        if ``threshold`` is negative.

    References
    ----------
    Smith et al. (2015), *Theoretical and Applied Genetics* 128:55-72;
    Smith & Cullis (2018), *Euphytica* 214:143;
    Smith et al. (2021), *Frontiers in Plant Science* 12:737462.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative; got {threshold}.")

    arr, ids_from_input = _to_numpy_loadings(loadings)
    if arr.ndim != 2:
        raise ValueError(
            f"loadings must be 2D (E, k); got shape {arr.shape}."
        )

    n_envs, n_factors = arr.shape

    if env_ids is None:
        if ids_from_input is not None:
            env_ids = ids_from_input
        else:
            env_ids = [f"env_{i}" for i in range(n_envs)]
    else:
        env_ids = [str(x) for x in env_ids]
        if len(env_ids) != n_envs:
            raise ValueError(
                f"env_ids has length {len(env_ids)}; expected {n_envs} "
                "to match loadings rows."
            )

    # Build the polarity matrix and per-env polarity word.
    polarity_matrix = np.empty((n_envs, n_factors), dtype="<U1")
    cluster_labels: list[str] = []
    for i in range(n_envs):
        letters = [_polarity(float(arr[i, j]), threshold) for j in range(n_factors)]
        for j, ltr in enumerate(letters):
            polarity_matrix[i, j] = ltr
        cluster_labels.append("".join(letters))

    # Build cluster_membership in environment order (stable).
    cluster_membership: dict[str, list[str]] = {}
    for env_id, label in zip(env_ids, cluster_labels):
        cluster_membership.setdefault(label, []).append(env_id)

    return IClassResult(
        cluster_labels=cluster_labels,
        cluster_membership=cluster_membership,
        loadings=arr,
        polarity_matrix=polarity_matrix,
        n_factors=n_factors,
        n_envs=n_envs,
        env_ids=env_ids,
        threshold=threshold,
    )


__all__ = ["IClassResult", "iclass"]
