"""Preprocessing module: imputation, QC, standardization, polyploid encoding.

Imputation methods:
- Built-in: mean, mode, KNN, LD-based (Phase 2)
- GPU-accelerated: Li-and-Stephens HMM, deep learning masked autoencoder (Phase 19)
"""

from .dosage_call import DosageCallResult, run_updog  # noqa: F401
from .dosage_uncertainty import dosage_rsq, dosage_variance, expected_dosage  # noqa: F401
from .impute import impute_knn, impute_ld, impute_mean, impute_mode  # noqa: F401
from .impute_gpu import impute_deep_learning, impute_li_stephens  # noqa: F401
from .phase_polyorigin import PhasingResult, run_polyorigin  # noqa: F401  # Phase 56
from .polyploid import detect_ploidy, list_gene_action_models, recode_gene_action  # noqa: F401
from .polyrad_wrapper import DosageProbabilities, run_polyrad  # noqa: F401
from .qc import (  # noqa: F401
    apply_qc_filters,
    compute_max_genotype_freq,
    compute_variant_qc,
    write_variant_qc_parquet,
)
from .standardize import (  # noqa: F401
    center_genotypes,
    compute_allele_frequencies,
    scale_genotypes,
)
from .expression import (  # noqa: F401
    inverse_normal_transform,
    peer_residualize,
    quantile_normalize,
)
