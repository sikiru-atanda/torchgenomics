"""`torchgwas.multiomics` — GRM-corrected mediation + multi-kernel heritability.

Phase 49 entry points:

* :func:`mediate_lmm` — single (SNP, mediator, outcome) triple
* :func:`scan_mediation` — cis-window genome × feature scan
* :func:`mkernel_h2` — variance partition over a dict of kernels
* :func:`build_expression_kernel` — context kernel from molecular abundance
"""

from ._kernels import build_expression_kernel
from ._mediate import mediate_lmm
from ._mkernel import mkernel_h2
from ._scan import scan_mediation
from ._sensitivity import imai_rho_sensitivity
from ._types import MediationResult, MediationScanResult, MultiKernelH2Result

__all__ = [
    "mediate_lmm",
    "scan_mediation",
    "mkernel_h2",
    "build_expression_kernel",
    "imai_rho_sensitivity",
    "MediationResult",
    "MediationScanResult",
    "MultiKernelH2Result",
]
