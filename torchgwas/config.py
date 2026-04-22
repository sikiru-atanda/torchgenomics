"""Global configuration: dtype policy, AMP guards, numerical constants, device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Set the cuBLAS deterministic workspace config before torch initialises CUDA,
# otherwise `torch.use_deterministic_algorithms(True)` raises on CUDA >= 10.2
# matmul. Keeping this at module import means any downstream call to
# ``set_deterministic(True)`` works regardless of test ordering.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------
# FP64 is mandatory for all statistical inference paths.
# FP32/FP16/BF16 are allowed only for I/O decode, GRM GEMM, and streaming
# transforms — never for likelihoods, coefficients, SEs, test stats, or p-values.

STAT_DTYPE = torch.float64  # log-likelihood, beta, SE, p-values, eigenvalues
IO_DTYPE = torch.float32  # genotype decode, streaming chunks
GRM_ACCUM_DTYPE = torch.float64  # GRM accumulator (cast after FP32 GEMM)


# ---------------------------------------------------------------------------
# AMP (Automatic Mixed Precision)
# ---------------------------------------------------------------------------

@dataclass
class AMPConfig:
    """Controls mixed-precision behaviour.

    AMP is applied only to heavy-compute steps (GRM construction, genotype
    chunk matmuls). The statistical path is always FP64 — AMP contexts must
    never wrap likelihood evaluation, GLS solves, or test-statistic computation.
    """

    enabled: bool = False
    dtype: torch.dtype = torch.float16  # or torch.bfloat16
    # Scaler is only needed for float16; bfloat16 does not require loss scaling.
    use_scaler: bool = True


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

@dataclass
class NumericalConfig:
    """Jitter, tolerance, and convergence defaults."""

    # Diagonal jitter added before Cholesky: eps * trace(K) / n
    cholesky_jitter_factor: float = 1e-6

    # REML convergence: |ll_new - ll_old| / |ll_old| < tol
    reml_convergence_tol: float = 1e-6
    reml_max_iter: int = 100

    # PX-EM warm-start iterations before switching to AI-REML / LBFGS
    em_warmstart_iters: int = 5

    # Single-trait REML: "ai_reml" (PX-EM + AI-REML stack) or "emma" (grid search + Brent)
    reml_method: str = "ai_reml"

    # Multi-trait REML: "lbfgs" (LBFGS-autograd on Cholesky factors, default)
    #                   or "triad" (Trust-Region Inexact Autograd-Differentiated)
    #                   or "pxem_nr" (PX-EM + Newton-Raphson, GEMMA-style)
    multi_trait_reml_method: str = "lbfgs"

    # Multi-trait EM warm-start iterations (more needed than single-trait)
    multi_trait_em_iters: int = 20

    # Eigenvalue floor: clamp negative eigenvalues from eigh to this value
    eigenvalue_floor: float = 1e-10

    # P-value floor: clamp extremely small p-values to avoid log(0)
    pvalue_floor: float = 1e-300

    # --- Approximate backend defaults (Phase 13) ---
    approx_n_components: int = 100
    approx_n_oversamples: int = 10
    approx_n_power_iters: int = 2
    approx_n_landmarks: int = 500
    approx_sparse_threshold: float = 0.05
    approx_stochastic_probes: int = 30
    approx_lanczos_iters: int = 50


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(device: str | None = None) -> torch.device:
    """Resolve compute device. If *device* is None, prefer CUDA when available.

    Logs which device was selected and why, so users know whether GPU
    acceleration is active or falling back to CPU.
    """
    import logging
    _log = logging.getLogger(__name__)

    if device is not None:
        dev = torch.device(device)
        _log.info("Device: %s (explicitly requested)", dev)
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        _log.info("Device: %s (%s) — GPU acceleration active", dev, gpu_name)
        return dev
    _log.info("Device: cpu — no CUDA GPU detected, falling back to CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Deterministic mode
# ---------------------------------------------------------------------------

def set_deterministic(enabled: bool = True) -> None:
    """Toggle deterministic algorithms for regression-test reproducibility.

    When enabled, ``torch.use_deterministic_algorithms(True)`` is set so that
    GPU results are bitwise reproducible across runs (at a performance cost).
    """
    if enabled:
        # CUDA >= 10.2 requires CUBLAS_WORKSPACE_CONFIG to be set before any
        # cuBLAS call under deterministic mode, otherwise matmul raises. The
        # `:4096:8` form is recommended by NVIDIA for broad kernel coverage.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled)


# ---------------------------------------------------------------------------
# Ploidy configuration
# ---------------------------------------------------------------------------

# Crop shortcut → default ploidy mapping
CROP_PLOIDY_MAP: dict[str, int] = {
    "human": 2, "maize": 2, "rice": 2, "soybean": 2, "tomato": 2,
    "potato": 4, "alfalfa": 4, "cotton": 4,
    "wheat": 6, "oat": 6, "sweet_potato": 6,
    "sugarcane": 8, "strawberry": 8,
}


@dataclass
class PloidyConfig:
    """User-facing ploidy configuration with validation and resolution.

    Resolution order: (1) explicit ploidy integer, (2) per_variant_map,
    (3) from_vcf, (4) crop shortcut, (5) fallback P=2.
    """

    ploidy: int | str = 2
    """One of {2, 4, 6, ...} or "from_vcf" or "per_variant_map"."""

    crop: str | None = None
    """Optional shortcut (e.g., "potato" → P=4). Never overrides explicit ploidy."""

    strict_ploidy: bool = True
    """If True, ploidy mismatch triggers ERROR."""

    ploidy_mismatch_policy: str = "error"
    """One of "error" (default) or "warn+override"."""

    per_variant_map_path: str | None = None
    """Path to per-variant ploidy map when ploidy="per_variant_map"."""

    eps_dosage: float = 1e-6
    """Tolerance for dosage range validation and homozygote classification."""

    def resolve(self) -> int:
        """Resolve the effective ploidy level (scalar).

        Returns the integer ploidy. Raises if ploidy="per_variant_map" or
        "from_vcf" without the required data (those require runtime resolution).
        """
        if isinstance(self.ploidy, int):
            return self.ploidy
        if self.ploidy == "per_variant_map":
            raise ValueError(
                "ploidy='per_variant_map' requires runtime resolution with variant data. "
                "Use resolve_per_variant() instead."
            )
        if self.ploidy == "from_vcf":
            raise ValueError(
                "ploidy='from_vcf' requires runtime resolution with VCF data."
            )
        # Should not happen
        raise ValueError(f"Unknown ploidy value: {self.ploidy}")

    @property
    def ploidy_source(self) -> str:
        """Describes how ploidy was determined."""
        if isinstance(self.ploidy, int):
            if self.crop is not None and self.ploidy == CROP_PLOIDY_MAP.get(self.crop):
                return "crop"
            return "explicit"
        return str(self.ploidy)

    def __post_init__(self):
        # Apply crop shortcut only if ploidy not explicitly set as int
        if self.crop is not None and self.ploidy == 2:
            crop_lower = self.crop.lower()
            if crop_lower in CROP_PLOIDY_MAP:
                self.ploidy = CROP_PLOIDY_MAP[crop_lower]

        # Validate ploidy_mismatch_policy
        if self.ploidy_mismatch_policy not in ("error", "warn+override"):
            raise ValueError(
                f"ploidy_mismatch_policy must be 'error' or 'warn+override', "
                f"got '{self.ploidy_mismatch_policy}'"
            )


def validate_dosage_range(
    G: torch.Tensor,
    ploidy: int = 2,
    eps: float = 1e-6,
    policy: str = "error",
) -> int:
    """Validate that dosage values are in [0, P] within tolerance.

    Returns the count of out-of-range values. Raises ValueError if policy
    is "error" and violations exist.
    """
    non_nan = G[~torch.isnan(G)]
    if non_nan.numel() == 0:
        return 0

    below = (non_nan < -eps).sum().item()
    above = (non_nan > ploidy + eps).sum().item()
    n_violations = int(below + above)

    if n_violations > 0:
        msg = (
            f"Dosage range violation: {n_violations} values outside [0, {ploidy}] "
            f"(tolerance eps={eps}). {below} below 0, {above} above {ploidy}."
        )
        if policy == "error":
            raise ValueError(msg)
        else:
            import logging
            logging.getLogger(__name__).warning(msg)

    return n_violations


# ---------------------------------------------------------------------------
# Master runtime config
# ---------------------------------------------------------------------------

@dataclass
class TorchGWASConfig:
    """Aggregated runtime configuration passed through the pipeline."""

    device: torch.device = field(default_factory=lambda: resolve_device())
    amp: AMPConfig = field(default_factory=AMPConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    ploidy_config: PloidyConfig = field(default_factory=PloidyConfig)
    deterministic: bool = False
    chunk_size: int = 1024  # SNPs per streaming chunk
    n_threads: int = 1  # CPU worker threads for I/O prefetch

    @property
    def ploidy(self) -> int:
        """Convenience accessor for resolved ploidy."""
        p = self.ploidy_config.ploidy
        return p if isinstance(p, int) else 2
