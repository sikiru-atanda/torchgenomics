"""`torchgenomics.multiomics` — GRM-corrected mediation + multi-kernel heritability.

Phase 49 entry points:

* :func:`mediate_lmm` — single (SNP, mediator, outcome) triple
* :func:`scan_mediation` — cis-window genome × feature scan
* :func:`mkernel_h2` — variance partition over a dict of kernels
* :func:`build_expression_kernel` — context kernel from molecular abundance
"""

from ..stats.multipletesting import eigenmt_adjust
from ._gene_set import mediate_gene_set
from ._kernels import build_expression_kernel
from ._mediate import mediate_lmm
from ._mkernel import mkernel_h2
from ._prefilter import coloc_prefilter_pairs
from ._scan import scan_mediation
from ._sensitivity import imai_rho_sensitivity
from ._types import (
    GeneSetMediationResult,
    MediationResult,
    MediationScanResult,
    MultiKernelH2Result,
)

__all__ = [
    "mediate_lmm",
    "mediate_gene_set",
    "scan_mediation",
    "mkernel_h2",
    "build_expression_kernel",
    "imai_rho_sensitivity",
    "eigenmt_adjust",
    "coloc_prefilter_pairs",
    "GeneSetMediationResult",
    "MediationResult",
    "MediationScanResult",
    "MultiKernelH2Result",
]
