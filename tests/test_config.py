"""Phase 0: Config and dataclass contract tests."""

from __future__ import annotations

import torch

from torchgenomics.config import (
    GRM_ACCUM_DTYPE,
    IO_DTYPE,
    STAT_DTYPE,
    AMPConfig,
    NumericalConfig,
    TorchGenomicsConfig,
)


class TestDtypePolicy:
    """Verify dtype constants are correct."""

    def test_stat_dtype_is_float64(self):
        assert torch.float64 == STAT_DTYPE

    def test_io_dtype_is_float32(self):
        assert torch.float32 == IO_DTYPE

    def test_grm_accum_dtype_is_float64(self):
        assert torch.float64 == GRM_ACCUM_DTYPE


class TestTorchGenomicsConfig:
    """Verify config dataclass instantiation."""

    def test_default_config(self):
        cfg = TorchGenomicsConfig(device=torch.device("cpu"))
        assert cfg.ploidy == 2
        assert cfg.chunk_size == 1024
        assert cfg.deterministic is False

    def test_amp_config_defaults(self):
        amp = AMPConfig()
        assert amp.enabled is False
        assert amp.dtype == torch.float16

    def test_numerical_config_defaults(self):
        num = NumericalConfig()
        assert num.reml_convergence_tol == 1e-6
        assert num.pvalue_floor == 1e-300
