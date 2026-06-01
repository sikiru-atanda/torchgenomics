"""Optional native (C++) accelerators for torchgwas hot loops.

The compiled extensions in this package are built by ``setup.py`` only when a
C++17 toolchain is available. If the build fails or the user has no compiler,
``HAS_NATIVE_PRSCS`` is ``False`` and callers fall back to the pure-Python
reference implementations in ``torchgwas.pgs``.

Currently provided:

- ``_prscs_native.prscs_gibbs_block(beta_std, R_b, n_eff, phi, a, b,
  n_iter, n_burnin, seed) -> (mean, sd)`` — block Gibbs sampler for PRS-CS.
- ``_prscs_native.sample_gig_vec(lam, chi, psi, seed) -> ndarray`` — vectorized
  Generalized Inverse Gaussian sampler used internally and exposed for tests.
"""
from __future__ import annotations

try:  # pragma: no cover - import side-effect tested via HAS_NATIVE_PRSCS
    from . import _prscs_native  # type: ignore[attr-defined]

    HAS_NATIVE_PRSCS = True
except ImportError:  # pragma: no cover
    _prscs_native = None  # type: ignore[assignment]
    HAS_NATIVE_PRSCS = False

try:  # pragma: no cover
    from . import _ldpred2_native  # type: ignore[attr-defined]

    HAS_NATIVE_LDPRED2 = True
except ImportError:  # pragma: no cover
    _ldpred2_native = None  # type: ignore[assignment]
    HAS_NATIVE_LDPRED2 = False

try:  # pragma: no cover
    from . import _ct_native  # type: ignore[attr-defined]

    HAS_NATIVE_CT = True
except ImportError:  # pragma: no cover
    _ct_native = None  # type: ignore[assignment]
    HAS_NATIVE_CT = False

try:  # pragma: no cover
    from . import _pelt_native  # type: ignore[attr-defined]

    HAS_NATIVE_PELT = True
except ImportError:  # pragma: no cover
    _pelt_native = None  # type: ignore[assignment]
    HAS_NATIVE_PELT = False

try:  # pragma: no cover
    from . import _ess_native  # type: ignore[attr-defined]

    HAS_NATIVE_ESS = True
except ImportError:  # pragma: no cover
    _ess_native = None  # type: ignore[assignment]
    HAS_NATIVE_ESS = False

try:  # pragma: no cover
    from . import _graph_native  # type: ignore[attr-defined]

    HAS_NATIVE_GRAPH = True
except ImportError:  # pragma: no cover
    _graph_native = None  # type: ignore[assignment]
    HAS_NATIVE_GRAPH = False

try:  # pragma: no cover
    from . import _gabriel_native  # type: ignore[attr-defined]

    HAS_NATIVE_GABRIEL = True
except ImportError:  # pragma: no cover
    _gabriel_native = None  # type: ignore[assignment]
    HAS_NATIVE_GABRIEL = False

try:  # pragma: no cover
    from . import _big_ld_native  # type: ignore[attr-defined]

    HAS_NATIVE_BIG_LD = True
except ImportError:  # pragma: no cover
    _big_ld_native = None  # type: ignore[assignment]
    HAS_NATIVE_BIG_LD = False

try:  # pragma: no cover
    from . import _dp_optimize_native  # type: ignore[attr-defined]

    HAS_NATIVE_DP_OPTIMIZE = True
except ImportError:  # pragma: no cover
    _dp_optimize_native = None  # type: ignore[assignment]
    HAS_NATIVE_DP_OPTIMIZE = False

try:  # pragma: no cover
    from . import _cc_graph_native  # type: ignore[attr-defined]

    HAS_NATIVE_CC_GRAPH = True
except ImportError:  # pragma: no cover
    _cc_graph_native = None  # type: ignore[assignment]
    HAS_NATIVE_CC_GRAPH = False

try:  # pragma: no cover
    from . import _gwas_aligned_native  # type: ignore[attr-defined]

    HAS_NATIVE_GWAS_ALIGNED = True
except ImportError:  # pragma: no cover
    _gwas_aligned_native = None  # type: ignore[assignment]
    HAS_NATIVE_GWAS_ALIGNED = False

try:  # pragma: no cover
    from . import _spine_native  # type: ignore[attr-defined]

    HAS_NATIVE_SPINE = True
except ImportError:  # pragma: no cover
    _spine_native = None  # type: ignore[assignment]
    HAS_NATIVE_SPINE = False

try:  # pragma: no cover
    from . import _ld_decay_signal_native  # type: ignore[attr-defined]

    HAS_NATIVE_LD_DECAY_SIGNAL = True
except ImportError:  # pragma: no cover
    _ld_decay_signal_native = None  # type: ignore[assignment]
    HAS_NATIVE_LD_DECAY_SIGNAL = False

try:  # pragma: no cover
    from . import _greedy_mwis_native  # type: ignore[attr-defined]

    HAS_NATIVE_GREEDY_MWIS = True
except ImportError:  # pragma: no cover
    _greedy_mwis_native = None  # type: ignore[assignment]
    HAS_NATIVE_GREEDY_MWIS = False

try:  # pragma: no cover
    from . import _uncertainty_blocks_native  # type: ignore[attr-defined]

    HAS_NATIVE_UNCERTAINTY_BLOCKS = True
except ImportError:  # pragma: no cover
    _uncertainty_blocks_native = None  # type: ignore[assignment]
    HAS_NATIVE_UNCERTAINTY_BLOCKS = False

try:  # pragma: no cover
    from . import _wall_pritchard_native  # type: ignore[attr-defined]

    HAS_NATIVE_WALL_PRITCHARD = True
except ImportError:  # pragma: no cover
    _wall_pritchard_native = None  # type: ignore[assignment]
    HAS_NATIVE_WALL_PRITCHARD = False

try:  # pragma: no cover
    from . import _impute_mode_native  # type: ignore[attr-defined]

    HAS_NATIVE_IMPUTE_MODE = True
except ImportError:  # pragma: no cover
    _impute_mode_native = None  # type: ignore[assignment]
    HAS_NATIVE_IMPUTE_MODE = False

try:  # pragma: no cover
    from . import _impute_knn_native  # type: ignore[attr-defined]

    HAS_NATIVE_IMPUTE_KNN = True
except ImportError:  # pragma: no cover
    _impute_knn_native = None  # type: ignore[assignment]
    HAS_NATIVE_IMPUTE_KNN = False

try:  # pragma: no cover
    from . import _impute_ld_native  # type: ignore[attr-defined]

    HAS_NATIVE_IMPUTE_LD = True
except ImportError:  # pragma: no cover
    _impute_ld_native = None  # type: ignore[assignment]
    HAS_NATIVE_IMPUTE_LD = False

try:  # pragma: no cover
    from . import _cavi_native  # type: ignore[attr-defined]

    HAS_NATIVE_CAVI = True
except ImportError:  # pragma: no cover
    _cavi_native = None  # type: ignore[assignment]
    HAS_NATIVE_CAVI = False

try:  # pragma: no cover
    from . import _ibss_native  # type: ignore[attr-defined]

    HAS_NATIVE_IBSS = True
except ImportError:  # pragma: no cover
    _ibss_native = None  # type: ignore[assignment]
    HAS_NATIVE_IBSS = False

try:  # pragma: no cover
    from . import _mediate_native  # type: ignore[attr-defined]

    HAS_NATIVE_MEDIATE = True
except ImportError:  # pragma: no cover
    _mediate_native = None  # type: ignore[assignment]
    HAS_NATIVE_MEDIATE = False

try:  # pragma: no cover
    from . import _pcht_native  # type: ignore[attr-defined]

    HAS_NATIVE_PCHT = True
except ImportError:  # pragma: no cover
    _pcht_native = None  # type: ignore[assignment]
    HAS_NATIVE_PCHT = False

try:  # pragma: no cover
    from . import _snp_to_gene_native  # type: ignore[attr-defined]

    HAS_NATIVE_SNP_TO_GENE = True
except ImportError:  # pragma: no cover
    _snp_to_gene_native = None  # type: ignore[assignment]
    HAS_NATIVE_SNP_TO_GENE = False

try:  # pragma: no cover
    from . import _expression_native  # type: ignore[attr-defined]

    HAS_NATIVE_EXPRESSION = True
except ImportError:  # pragma: no cover
    _expression_native = None  # type: ignore[assignment]
    HAS_NATIVE_EXPRESSION = False

try:  # pragma: no cover
    from . import _hyprcoloc_native  # type: ignore[attr-defined]

    HAS_NATIVE_HYPRCOLOC = True
except ImportError:  # pragma: no cover
    _hyprcoloc_native = None  # type: ignore[assignment]
    HAS_NATIVE_HYPRCOLOC = False

try:  # pragma: no cover
    from . import _ordinal_threshold_native  # type: ignore[attr-defined]

    HAS_NATIVE_ORDINAL_THRESHOLD = True
except ImportError:  # pragma: no cover
    _ordinal_threshold_native = None  # type: ignore[assignment]
    HAS_NATIVE_ORDINAL_THRESHOLD = False

try:  # pragma: no cover
    from . import _spa_native  # type: ignore[attr-defined]

    HAS_NATIVE_SPA = True
except ImportError:  # pragma: no cover
    _spa_native = None  # type: ignore[assignment]
    HAS_NATIVE_SPA = False

try:  # pragma: no cover
    from . import _ldsc_native  # type: ignore[attr-defined]

    HAS_NATIVE_LDSC = True
except ImportError:  # pragma: no cover
    _ldsc_native = None  # type: ignore[assignment]
    HAS_NATIVE_LDSC = False

try:  # pragma: no cover
    from . import _hwe_native  # type: ignore[attr-defined]

    HAS_NATIVE_HWE = True
except ImportError:  # pragma: no cover
    _hwe_native = None  # type: ignore[assignment]
    HAS_NATIVE_HWE = False

try:  # pragma: no cover
    from . import _cross_pop_native  # type: ignore[attr-defined]

    HAS_NATIVE_CROSS_POP = True
except ImportError:  # pragma: no cover
    _cross_pop_native = None  # type: ignore[assignment]
    HAS_NATIVE_CROSS_POP = False

__all__ = [
    "HAS_NATIVE_PRSCS",
    "HAS_NATIVE_LDPRED2",
    "HAS_NATIVE_CT",
    "HAS_NATIVE_PELT",
    "HAS_NATIVE_ESS",
    "HAS_NATIVE_GRAPH",
    "HAS_NATIVE_GABRIEL",
    "HAS_NATIVE_BIG_LD",
    "HAS_NATIVE_DP_OPTIMIZE",
    "HAS_NATIVE_CC_GRAPH",
    "HAS_NATIVE_GWAS_ALIGNED",
    "HAS_NATIVE_SPINE",
    "HAS_NATIVE_LD_DECAY_SIGNAL",
    "HAS_NATIVE_GREEDY_MWIS",
    "HAS_NATIVE_UNCERTAINTY_BLOCKS",
    "HAS_NATIVE_WALL_PRITCHARD",
    "HAS_NATIVE_IMPUTE_MODE",
    "HAS_NATIVE_IMPUTE_KNN",
    "HAS_NATIVE_IMPUTE_LD",
    "HAS_NATIVE_CAVI",
    "HAS_NATIVE_IBSS",
    "HAS_NATIVE_MEDIATE",
    "HAS_NATIVE_PCHT",
    "HAS_NATIVE_SNP_TO_GENE",
    "HAS_NATIVE_EXPRESSION",
    "HAS_NATIVE_HYPRCOLOC",
    "HAS_NATIVE_ORDINAL_THRESHOLD",
    "HAS_NATIVE_SPA",
    "HAS_NATIVE_LDSC",
    "HAS_NATIVE_HWE",
    "HAS_NATIVE_CROSS_POP",
    "_prscs_native",
    "_ldpred2_native",
    "_ct_native",
    "_pelt_native",
    "_ess_native",
    "_graph_native",
    "_gabriel_native",
    "_big_ld_native",
    "_dp_optimize_native",
    "_cc_graph_native",
    "_gwas_aligned_native",
    "_spine_native",
    "_ld_decay_signal_native",
    "_greedy_mwis_native",
    "_uncertainty_blocks_native",
    "_wall_pritchard_native",
    "_impute_mode_native",
    "_impute_knn_native",
    "_impute_ld_native",
    "_cavi_native",
    "_ibss_native",
    "_mediate_native",
    "_pcht_native",
    "_snp_to_gene_native",
    "_expression_native",
    "_hyprcoloc_native",
    "_ordinal_threshold_native",
    "_spa_native",
    "_ldsc_native",
    "_hwe_native",
    "_cross_pop_native",
]
