"""Statistics module: test statistics, multiple testing correction, genomic control."""

from .adaptive_fdr import AdaPTResult, IHWResult, adapt, ihw  # noqa: F401
from .best_model import BestModelResult, select_best_model  # noqa: F401
from .cauchy import cauchy_combination  # noqa: F401
from .genomic_control import diagnose_inflation, lambda_gc  # noqa: F401
from .mixture import davies_pvalue, liu_pvalue, mixture_chi2_pvalue  # noqa: F401
from .multipletesting import (  # noqa: F401
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    holm,
    sidak,
    storey_qvalue,
)
from .peak_pruning import QTLPeak, prune_peaks  # noqa: F401
from .permutation import adaptive_permutation_maxT, permutation_maxT  # noqa: F401
from .pve import PVEResult, compute_pve  # noqa: F401
from .simplem import (  # noqa: F401
    effective_test_count,
    effective_test_count_moskvina,
    ld_correlation_eigenvalues,
)
from .spa import saddlepoint_pvalue  # noqa: F401
from .tests import chi2_sf, lrt_test, score_test, score_test_multi_df, wald_test  # noqa: F401
from .weighted_fdr import hierarchical_fdr, local_fdr, weighted_bh  # noqa: F401
