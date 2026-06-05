"""Linear algebra module: GRM/kinship, eigendecomposition, batched solvers."""

from .basis import (  # noqa: F401
    bspline_basis,
    difference_penalty,
    evaluate_basis_at,
    legendre_basis,
    place_knots,
    pspline_2d,
    standardize_time,
)
from .eigh import auto_n_components, compute_weights, eigendecompose, rotate  # noqa: F401
from .kinship import GRMMetadata, grm_vanraden, grm_vanraden_streaming, grm_zhang  # noqa: F401
from .kinship_advanced import (  # noqa: F401
    grm_endelman_digenic,
    grm_pseudo_diploid,
    grm_slater,
    grm_su_dominance,
    grm_vitezica_dominance,
    grm_weighted,
    grm_yang_gcta,
)
from .kinship_polyploid import (  # noqa: F401
    grm_asv_transform,
    grm_epistatic_hadamard,
    grm_loco,
    grm_polyploid_gene_action,
)
from .kronecker_eed import (  # noqa: F401
    KronEED,
    diagonal_precision,
    inverse_rotate_from_ked,
    ked_reml_quantities,
    kronecker_eed,
    kronecker_eed_from_full,
    rotate_to_ked_basis,
    woodbury_fa_precision,
)
from .multi_kernel_streaming import build_multi_kernels_streaming  # noqa: F401
from .safe import safe_cholesky, safe_logdet  # noqa: F401
from .truncated_mvn import (  # noqa: F401
    bivariate_truncated_moments,
    mvn_truncated_moments,
    truncated_normal_moments,
)
from .woodbury import woodbury_inverse, woodbury_logdet  # noqa: F401
