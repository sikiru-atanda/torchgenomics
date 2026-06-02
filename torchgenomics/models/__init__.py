"""Statistical models: GLM, LMM, mvLMM, FarmCPU, BLINK, and post-V1 extensions.

Post-V1 novel models (Phases 15-19+):
- MultiKernelLMM: Additive + dominance + epistasis variance partitioning (Phase 15)
- GxELMM / HetLMM: Gene-environment interaction (Phase 16)
- SetBasedScanner: SKAT/Burden/SKAT-O under LMM (Phase 17)
- BayesianVS: Spike-and-slab variable selection with SuSiE (IBSS) and CAVI (Phase 18)
- MultiEnvLMM: Single-trait multi-environment GWAS (MET), stable vs contingent effects
- WithinFamilyLMM: Within-family GWAS with confounding diagnostics (Phase 23)
- ConditionalLMM: LD-conditional GWAS with persistence metrics (Phase 24)
- MultiTraitMultiEnvLMM: MT-MET GWAS with separable Kronecker covariance (Phase 25)
- OCFLMM: Orthogonal Cross-Fit LMM — DML-valid debiased inference (Phase 26)
- KnockoffLMM: Group knockoff FDR-controlled GWAS under LMM (Phase 27)
- GULM: Genotype-Uncertainty LMM — dosage-variance–corrected score test (Phase 28)
- LROLMM: Leave-Region-Out LMM — block-level LOCO for proximal decontamination (Phase 29)
- SurvivalGLMM: Cox PH frailty model with PQL null fitting and martingale residual score test (Phase 35)
- RandomRegressionLMM: Longitudinal RR-LMM with Legendre / B-spline basis, projection or stacked mode (Phase 38)
- SpatioTemporalRR: RR-LMM with 2D P-spline spatial smoother (Phase 38)
"""

from .base import BaseModel, NullFit, ScanResult, VariantMeta, update_null  # noqa: F401
from .bayesian_vs import BayesianVS, BayesianVSResult  # noqa: F401
from .binary_glm import BinaryGLM  # noqa: F401
from .binary_glmm import BinaryGLMM  # noqa: F401
from .blink import BLINK  # noqa: F401
from .conditional_lmm import ConditionalLMM, ConditionalScanResult  # noqa: F401
from .farmcpu import FarmCPU  # noqa: F401
from .glm import GLM  # noqa: F401
from .glm_link import (  # noqa: F401
    CumulativeLogitLink,
    IdentityLink,
    LinkFunction,
    LogitLink,
    ProbitLink,
)
from .gu_lmm import GULM, GUResult  # noqa: F401
from .haplotype_gwas import HaplotypeBlock, HaplotypeGWAS, HaplotypeGWASResult  # noqa: F401
from .haplotype_multi import (  # noqa: F401
    HaplotypeMTMETGWAS,
    HaplotypeMTMETResult,
    HaplotypeMultiEnvGWAS,
    HaplotypeMultiEnvResult,
    HaplotypeMultiTraitGWAS,
    HaplotypeMultiTraitResult,
)
from .haplotype_novel import (  # noqa: F401
    BayesHapResult,
    BayesianHaplotypeFineMapping,
    HapGxEResult,
    HHCTResult,
    PCHTResult,
    compute_dosage_posterior_cov,
    haplotype_gxe_test,
    haplotype_similarity_kernel,
    hierarchical_haplotype_test,
    hskat_adaptive,
    hskat_test,
    pcht_score_test,
)
from .iterative import IterativeGWASLoop  # noqa: F401
from .knockoff_lmm import KnockoffLMM, KnockoffResult  # noqa: F401
from .lmm_gxe import GxELMM, GxEScanResult, HetLMM  # noqa: F401
from .lro_lmm import LROLMM, LROResult  # noqa: F401
from .multi_env_glmm import MultiEnvGLMM  # noqa: F401
from .multi_env_lmm import EnvScanResult, MultiEnvLMM  # noqa: F401
from .multi_kernel_lmm import MultiKernelLMM, build_multi_kernels  # noqa: F401
from .multi_trait_lmm import MultiTraitLMM  # noqa: F401
from .multi_trait_multi_env_lmm import MTMETScanResult, MultiTraitMultiEnvLMM  # noqa: F401
from .multinomial_glm import MultinomialGLM  # noqa: F401
from .multinomial_glmm import MultinomialGLMM  # noqa: F401
from .ocf_lmm import OCFLMM, OCFNullFit  # noqa: F401
from .ordinal_glm import OrdinalGLM  # noqa: F401
from .ordinal_glmm import OrdinalGLMM  # noqa: F401
from .rr_lmm import (  # noqa: F401
    LongitudinalProjection,
    RandomRegressionLMM,
    RRScanResult,
    longitudinal_to_wide,
)
from .rr_met import (  # noqa: F401
    MultiEnvLongitudinalProjection,
    RandomRegressionMultiEnvLMM,
    RRMetScanResult,
    project_multi_env,
)
from .rr_spatial import SpatioTemporalRR, fit_spatial_pspline  # noqa: F401
from .set_based import SetBasedResult, SetBasedScanner  # noqa: F401
from .single_trait_lmm import SingleTraitLMM  # noqa: F401
from .survival_glmm import SurvivalGLMM  # noqa: F401
from .threshold_linear import ThresholdLinearModel, ThresholdNullFit  # noqa: F401
from .within_family_lmm import WithinFamilyLMM, WithinFamilyNullFit  # noqa: F401
