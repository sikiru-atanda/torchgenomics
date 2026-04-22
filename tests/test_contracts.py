"""Phase 0: Protocol and dataclass contract tests — verify all contracts are importable and typed."""

from __future__ import annotations


class TestBaseModelProtocol:
    """Verify BaseModel protocol is importable and runtime-checkable."""

    def test_import_base_model(self):
        from torchgwas.models.base import BaseModel
        assert hasattr(BaseModel, "fit_null")
        assert hasattr(BaseModel, "score_chunk")

    def test_nullfit_fields(self):
        import dataclasses

        from torchgwas.models.base import NullFit
        fields = {f.name for f in dataclasses.fields(NullFit)}
        assert "sig2_g" in fields
        assert "sig2_e" in fields
        assert "eigenvalues" in fields
        assert "converged" in fields

    def test_scanresult_fields(self):
        import dataclasses

        from torchgwas.models.base import ScanResult
        fields = {f.name for f in dataclasses.fields(ScanResult)}
        assert "p" in fields
        assert "beta" in fields
        assert "se" in fields
        assert "chr" in fields

    def test_variant_meta_fields(self):
        import dataclasses

        from torchgwas.models.base import VariantMeta
        fields = {f.name for f in dataclasses.fields(VariantMeta)}
        assert "snp" in fields
        assert "chr" in fields
        assert "pos" in fields


class TestIOProtocols:
    """Verify I/O protocols are importable."""

    def test_import_genotype_reader(self):
        from torchgwas.io.base import GenotypeReader
        assert hasattr(GenotypeReader, "iter_chunks")
        assert hasattr(GenotypeReader, "n_samples")
        assert hasattr(GenotypeReader, "sample_ids")

    def test_import_preflight_report(self):
        import dataclasses

        from torchgwas.io.validate import PreflightReport
        assert dataclasses.is_dataclass(PreflightReport)

    def test_import_alignment_manifest(self):
        import dataclasses

        from torchgwas.io.phenotype import AlignmentManifest
        assert dataclasses.is_dataclass(AlignmentManifest)


class TestPreprocessContracts:
    """Verify preprocessing modules are importable."""

    def test_import_qc_config(self):
        import dataclasses

        from torchgwas.preprocess.qc import QCFilterConfig, VariantQCStats
        assert dataclasses.is_dataclass(QCFilterConfig)
        assert dataclasses.is_dataclass(VariantQCStats)

    def test_import_imputation_result(self):
        import dataclasses

        from torchgwas.preprocess.impute_external import ImputationResult
        assert dataclasses.is_dataclass(ImputationResult)

    def test_import_gene_action_models(self):
        from torchgwas.preprocess.polyploid import GENE_ACTION_MODELS
        assert "additive" in GENE_ACTION_MODELS


class TestLinalgContracts:
    """Verify linalg modules are importable."""

    def test_import_eigendecomp(self):
        import dataclasses

        from torchgwas.linalg.eigh import EigenDecomp
        assert dataclasses.is_dataclass(EigenDecomp)


class TestOptimContracts:
    """Verify optimizer contracts are importable."""

    def test_import_optimizer_mode(self):
        from torchgwas.optim.controller import OptimizerMode
        assert hasattr(OptimizerMode, "PX_EM")
        assert hasattr(OptimizerMode, "AI_REML")
        assert hasattr(OptimizerMode, "RESCUE")
