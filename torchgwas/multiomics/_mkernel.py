"""Multi-kernel heritability partitioning over a dict of kernels.

Thin convenience wrapper over :class:`torchgwas.models.MultiKernelLMM` that
accepts a ``{name: K}`` dict and returns a friendlier result object.
"""

from __future__ import annotations

import torch

from ..models.multi_kernel_lmm import MultiKernelLMM
from ._types import MultiKernelH2Result


def _as_tensor(x):
    return torch.as_tensor(x, dtype=torch.float64).detach().cpu()


def mkernel_h2(
    Y,
    kernels: dict,
    *,
    covariates=None,
) -> MultiKernelH2Result:
    """Partition phenotypic variance across an ordered dict of kernels.

    Parameters
    ----------
    Y : array-like (n,)
    kernels : dict[str, (n, n)]  e.g. {"snp": K_snp, "expr": K_expr}
    covariates : array-like (n, c) or None
    """
    if not kernels:
        raise ValueError("Provide at least one kernel.")
    Y_t = _as_tensor(Y).reshape(-1)
    n = Y_t.shape[0]

    intercept = torch.ones(n, 1, dtype=torch.float64)
    if covariates is None:
        X0 = intercept
    else:
        X0 = torch.cat([intercept, _as_tensor(covariates).reshape(n, -1)], dim=1)

    names = list(kernels.keys())
    K_list = [_as_tensor(K) for K in kernels.values()]
    for K in K_list:
        if K.shape != (n, n):
            raise ValueError(f"Kernel must be ({n}, {n}); got {tuple(K.shape)}.")

    model = MultiKernelLMM()
    nf = model.fit_null(Y_t, X0, kernels=K_list, kernel_names=names)

    sigma2 = {k: float(v) for k, v in nf.variance_dict.items()}
    h2_full = {k: float(v) for k, v in nf.partitioned_h2.items()}
    # h² per non-residual kernel, summing < 1 if residual variance is positive.
    h2 = {k: h2_full[k] for k in names}
    h2_total = sum(h2.values())

    return MultiKernelH2Result(
        sigma2=sigma2,
        h2=h2,
        h2_total=h2_total,
        kernel_names=names,
        n=n,
        nullfit=nf,
    )
