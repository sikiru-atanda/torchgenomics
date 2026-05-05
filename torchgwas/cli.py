"""CLI entry point for all TorchGWAS commands.

Subcommands: validate, convert, impute, dosage-call, phase-poly, glm-scan,
lmm-scan, mvlmm-scan, poly-scan, mklmm-scan, gxe-scan, set-scan, bayes-scan,
met-scan, farmcpu-scan, blink-scan, threshold-scan, family-scan,
conditional-scan, mtmet-scan, ocf-scan, knockoff-scan, gu-scan, lro-scan,
glmm-scan, me-glmm-scan, survival-scan, rr-scan, rr-met-scan, ld-blocks,
ldsc, ldsc-rg, meta, clump, pgs-fit, pgs-score, annotate, mediate,
mediate-scan, pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    import torch

    from .models.multi_env_lmm import EnvScanResult
    from .models.multi_trait_multi_env_lmm import MTMETScanResult

logger = logging.getLogger("torchgwas")


def main(argv: list[str] | None = None) -> int:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="torchgwas",
        description="TorchGWAS: GPU-accelerated Genome-Wide Association Studies",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Data management commands ---
    _add_validate_parser(subparsers)
    _add_convert_parser(subparsers)
    _add_impute_parser(subparsers)
    _add_dosage_call_parser(subparsers)
    _add_phase_poly_parser(subparsers)  # Phase 56

    # --- GWAS scan commands ---
    _add_glm_scan_parser(subparsers)
    _add_lmm_scan_parser(subparsers)
    _add_mvlmm_scan_parser(subparsers)
    _add_poly_scan_parser(subparsers)
    _add_mklmm_scan_parser(subparsers)
    _add_gxe_scan_parser(subparsers)
    _add_set_scan_parser(subparsers)
    _add_bayes_scan_parser(subparsers)
    _add_met_scan_parser(subparsers)
    _add_farmcpu_scan_parser(subparsers)
    _add_blink_scan_parser(subparsers)
    _add_threshold_scan_parser(subparsers)
    _add_family_scan_parser(subparsers)
    _add_conditional_scan_parser(subparsers)
    _add_mtmet_scan_parser(subparsers)
    _add_ocf_scan_parser(subparsers)
    _add_knockoff_scan_parser(subparsers)
    _add_gu_scan_parser(subparsers)
    _add_lro_scan_parser(subparsers)
    _add_glmm_scan_parser(subparsers)
    _add_me_glmm_scan_parser(subparsers)
    _add_survival_scan_parser(subparsers)
    _add_rr_scan_parser(subparsers)
    _add_rr_met_scan_parser(subparsers)

    # --- LD / haplotype block detection ---
    _add_ld_blocks_parser(subparsers)

    # --- Post-GWAS analysis ---
    _add_ldsc_parser(subparsers)
    _add_ldsc_rg_parser(subparsers)
    _add_meta_parser(subparsers)
    _add_clump_parser(subparsers)

    # --- Polygenic scores (Phase 40) ---
    _add_pgs_fit_parser(subparsers)
    _add_pgs_score_parser(subparsers)

    # --- NCBI gene annotation ---
    _add_annotate_parser(subparsers)

    # --- Multi-omics mediation (Phase 49) ---
    _add_mediate_parser(subparsers)
    _add_mediate_scan_parser(subparsers)

    # --- Full pipeline ---
    _add_pipeline_parser(subparsers)

    args = parser.parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handlers = {
        "validate": _cmd_validate,
        "glm-scan": _cmd_glm_scan,
        "lmm-scan": _cmd_lmm_scan,
        "mvlmm-scan": _cmd_mvlmm_scan,
        "poly-scan": _cmd_poly_scan,
        "mklmm-scan": _cmd_mklmm_scan,
        "gxe-scan": _cmd_gxe_scan,
        "set-scan": _cmd_set_scan,
        "bayes-scan": _cmd_bayes_scan,
        "met-scan": _cmd_met_scan,
        "farmcpu-scan": _cmd_farmcpu_scan,
        "blink-scan": _cmd_blink_scan,
        "threshold-scan": _cmd_threshold_scan,
        "family-scan": _cmd_family_scan,
        "conditional-scan": _cmd_conditional_scan,
        "mtmet-scan": _cmd_mtmet_scan,
        "ocf-scan": _cmd_ocf_scan,
        "knockoff-scan": _cmd_knockoff_scan,
        "gu-scan": _cmd_gu_scan,
        "lro-scan": _cmd_lro_scan,
        "glmm-scan": _cmd_glmm_scan,
        "me-glmm-scan": _cmd_me_glmm_scan,
        "survival-scan": _cmd_survival_scan,
        "rr-scan": _cmd_rr_scan,
        "rr-met-scan": _cmd_rr_met_scan,
        "ld-blocks": _cmd_ld_blocks,
        "ldsc": _cmd_ldsc,
        "ldsc-rg": _cmd_ldsc_rg,
        "meta": _cmd_meta,
        "clump": _cmd_clump,
        "pgs-fit": _cmd_pgs_fit,
        "pgs-score": _cmd_pgs_score,
        "annotate": _cmd_annotate,
        "mediate": _cmd_mediate,
        "mediate-scan": _cmd_mediate_scan,
        "pipeline": _cmd_pipeline,
        "convert": _cmd_convert,
        "impute": _cmd_impute,
        "dosage-call": _cmd_dosage_call,  # Phase 55
        "phase-poly": _cmd_phase_poly,   # Phase 56
    }

    handler = handlers.get(args.command)
    if handler is None:
        raise NotImplementedError(f"Command '{args.command}' not yet implemented")
    return handler(args)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    """Run pre-flight validation."""
    from .io.validate import run_preflight

    report = run_preflight(
        genotype_path=args.genotype,
        phenotype_path=args.phenotype,
        covariate_path=args.covariate,
    )

    print(f"Format:     {report.format_name}")
    print(f"Samples:    {report.n_samples_genotype} genotype, "
          f"{report.n_samples_phenotype} phenotype, "
          f"{report.n_samples_aligned} aligned")
    print(f"Variants:   {report.n_variants_total}")
    print(f"Traits:     {report.n_traits}")
    print(f"Missingness: {report.genotype_missingness_pct:.1f}% genotype, "
          f"{report.phenotype_missingness_pct:.1f}% phenotype")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
        return 1

    if args.report:
        import json
        from dataclasses import asdict
        with open(args.report, "w") as fh:
            json.dump(asdict(report), fh, indent=2)
        print(f"\nReport written to {args.report}")

    return 0


def _run_per_trait(args: argparse.Namespace, scan_fn) -> int:
    """Run a univariate scan for each trait separately, saving to per-trait folders.

    If ``--traits`` specifies multiple traits for a univariate model, this
    function iterates over each trait, creates a subdirectory ``<output>/<trait>/``
    for results, and calls the scan function once per trait.

    When only one trait is specified (or ``--traits`` is omitted), the scan
    function is called directly with no wrapper behaviour.
    """
    import copy
    from pathlib import Path

    traits_str = getattr(args, "traits", None)
    if traits_str is None:
        return scan_fn(args)

    trait_list = [t.strip() for t in traits_str.split(",")]

    if len(trait_list) <= 1:
        return scan_fn(args)

    # Multiple traits — run each sequentially with per-trait output folder
    base_output = args.output
    logger.info("Multi-trait sequential scan: %d traits — %s", len(trait_list), trait_list)

    for i, trait in enumerate(trait_list):
        logger.info("===== Trait %d/%d: %s =====", i + 1, len(trait_list), trait)
        trait_args = copy.copy(args)
        trait_args.traits = trait  # single trait for this iteration

        # Per-trait output subdirectory
        trait_dir = Path(base_output) / trait
        trait_dir.mkdir(parents=True, exist_ok=True)
        trait_args.output = str(trait_dir / "results")

        ret = scan_fn(trait_args)
        if ret != 0:
            logger.error("Trait '%s' failed with exit code %d", trait, ret)
            return ret
        logger.info("Trait '%s' complete — results in %s/", trait, trait_dir)

    logger.info("All %d traits completed. Results in %s/<trait>/", len(trait_list), base_output)
    return 0


def _cmd_glm_scan(args: argparse.Namespace) -> int:
    """Run GLM association scan (streaming — no GRM needed)."""
    return _run_per_trait(args, _cmd_glm_scan_single)


def _cmd_glm_scan_single(args: argparse.Namespace) -> int:
    """Run GLM association scan for a single trait (or auto-detected traits)."""
    import torch

    from .config import TorchGWASConfig, resolve_device
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(
        device=device,
        chunk_size=args.chunk_size,
    )

    # Streaming: align samples without materializing G
    Y, X0, aligned_reader = _align_samples(args, config)

    # Add PCs as covariates if requested (requires computing GRM)
    n_pcs = getattr(args, "n_pcs", 0)
    if n_pcs > 0:
        logger.info("Computing GRM for PC extraction (%d PCs)...", n_pcs)
        from .linalg.kinship import grm_vanraden_streaming
        K_pc, _ = grm_vanraden_streaming(
            _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
            n_samples=aligned_reader.n_samples,
            device=device,
        )
        evals, evecs = torch.linalg.eigh(K_pc)
        n_avail = evecs.shape[1]
        if n_pcs > n_avail:
            logger.warning(
                "Requested %d PCs but only %d eigenvectors available. Using %d.",
                n_pcs, n_avail, n_avail,
            )
            n_pcs = n_avail
        pc_cols = evecs[:, -n_pcs:].flip(dims=[1]).to(torch.float64)
        X0 = torch.cat([X0, pc_cols], dim=1)
        logger.info("Covariate matrix: %d columns (intercept + covariates + %d PCs)",
                     X0.shape[1], n_pcs)
        del K_pc, evals, evecs  # free memory

    # Select model based on --family
    family = getattr(args, "family", "gaussian")
    if family == "gaussian":
        from .models.glm import GLM
        model = GLM()
        test = args.test
    elif family == "binary":
        from .models.binary_glm import BinaryGLM
        firth = getattr(args, "firth", False)
        no_spa = getattr(args, "no_spa", False)
        model = BinaryGLM(firth=firth, use_spa=not no_spa)
        test = "score"
    elif family == "ordinal":
        from .models.ordinal_glm import OrdinalGLM
        n_cat = getattr(args, "n_categories", None)
        if n_cat is None:
            logger.error("--n-categories required for --family ordinal")
            return 1
        model = OrdinalGLM(n_categories=n_cat)
        test = "score"
    elif family == "multinomial":
        from .models.multinomial_glm import MultinomialGLM
        model = MultinomialGLM()
        test = "score"
    else:
        logger.error("Unknown family: %s", family)
        return 1

    # Push Y/X0 to the requested device so --device cuda actually uses CUDA
    # (the model has no internal device routing; it operates on whatever
    # device the inputs come from). Mirrors the lmm-scan device-alignment
    # fix in commit `adc7b04` (Pillar C F3).
    Y = Y.to(device)
    X0 = X0.to(device)

    null_fit = model.fit_null(Y, X0)

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_lmm_scan(args: argparse.Namespace) -> int:
    """Run single-trait LMM association scan (streaming GRM)."""
    return _run_per_trait(args, _cmd_lmm_scan_single)


def _cmd_lmm_scan_single(args: argparse.Namespace) -> int:
    """Run single-trait LMM scan for a single trait."""
    import torch

    from .config import TorchGWASConfig, resolve_device
    from .models.single_trait_lmm import SingleTraitLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(
        device=device,
        chunk_size=args.chunk_size,
    )
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    grm_method = getattr(args, "grm_method", "vanraden")
    grm_path = getattr(args, "grm", None)

    # Streaming: align samples without materializing G
    Y, X0, aligned_reader = _align_samples(args, config)

    # Load or compute GRM
    if grm_path is not None:
        # User-provided GRM — load with QC validation
        logger.info("Loading user-provided GRM from %s...", grm_path)
        K = _load_user_grm(grm_path, aligned_reader.n_samples)
    elif grm_method == "zhang":
        # Zhang GRM requires full genotype matrix
        logger.info("Computing kinship matrix (Zhang method, full materialization)...")
        from .linalg.kinship import grm_zhang
        G, _ = _load_full_genotype(aligned_reader)
        K, grm_meta = grm_zhang(G)
        del G  # free memory after GRM
        logger.info("GRM: %d samples, %d SNPs used", grm_meta.n_samples, grm_meta.n_snps_used)
    else:
        # VanRaden streaming GRM — never materializes full G
        logger.info("Computing kinship matrix (VanRaden streaming)...")
        from .linalg.kinship import grm_vanraden_streaming
        K, grm_meta = grm_vanraden_streaming(
            _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
            n_samples=aligned_reader.n_samples,
            device=device,
        )
        logger.info("GRM: %d samples, %d SNPs used", grm_meta.n_samples, grm_meta.n_snps_used)

    # Add PCs as covariates if requested
    n_pcs = getattr(args, "n_pcs", 0)
    if n_pcs > 0:
        logger.info("Extracting %d principal components from GRM for covariates...", n_pcs)
        evals, evecs = torch.linalg.eigh(K)
        # eigh returns ascending order; take the last n_pcs (largest eigenvalues)
        n_avail = evecs.shape[1]
        if n_pcs > n_avail:
            logger.warning(
                "Requested %d PCs but only %d eigenvectors available. Using %d.",
                n_pcs, n_avail, n_avail,
            )
            n_pcs = n_avail
        pc_cols = evecs[:, -n_pcs:].flip(dims=[1]).to(torch.float64)
        X0 = torch.cat([X0, pc_cols], dim=1)
        logger.info("Covariate matrix: %d columns (intercept + covariates + %d PCs)",
                     X0.shape[1], n_pcs)

    approx_method = getattr(args, "approx_method", None)

    if approx_method == "sparse":
        # Sparse GRM + PCG-REML path (score test only)
        logger.info("Using sparse GRM + PCG-REML path (score test only)...")
        from .linalg.sparse_grm import sparse_grm_streaming
        from .models.sparse_lmm import SparseLMM

        K_sparse, normalizer, n_snps = sparse_grm_streaming(
            _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
            n_samples=aligned_reader.n_samples,
            threshold=getattr(args, "sparse_threshold", 0.05),
            device=device,
        )
        model = SparseLMM()
        # Same device-alignment fix as the dense path below.
        Y_sparse_dev = Y.to(device)
        X0_sparse_dev = X0.to(device)
        null_fit = model.fit_null(Y_sparse_dev, X0_sparse_dev, K=K_sparse)
        test_type = "score"  # Only score test for sparse path
    else:
        approx_config = {}
        if approx_method is not None:
            approx_config["n_components"] = getattr(args, "approx_components", 100)
            approx_config["n_landmarks"] = getattr(args, "approx_landmarks", 500)
            if approx_method == "nystrom":
                approx_config["chunk_iter"] = _impute_chunk_iter(
                    aligned_reader.iter_chunks(config.chunk_size)
                )
                approx_config["ploidy"] = 2

        p3d = getattr(args, "p3d", True)
        model = SingleTraitLMM(p3d=p3d)
        # Move Y / X0 to the same device as K so the eigenspace rotation
        # in `single_trait_lmm.fit_null` doesn't mix CPU + CUDA tensors.
        # K already lives on `device` from `grm_vanraden_streaming(device=...)`;
        # the parallel mvlmm-scan path (line ~499) does the same `.to(device)`.
        Y_dev = Y.to(device)
        X0_dev = X0.to(device)
        K_dev = K.to(device) if hasattr(K, "to") else K
        null_fit = model.fit_null(
            Y_dev, X0_dev, K=K_dev,
            approx_method=approx_method,
            approx_config=approx_config if approx_method else None,
        )
        if not p3d:
            logger.info("P3D=FALSE: variance components will be re-estimated per marker")
        test_type = args.test

    logger.info("Null fit: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f%s",
                null_fit.sig2_g, null_fit.sig2_e,
                null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
                " [APPROXIMATE]" if null_fit.approximate else "")

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=test_type, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_mvlmm_scan(args: argparse.Namespace) -> int:
    """Run multi-trait mvLMM association scan (streaming GRM)."""
    from .config import TorchGWASConfig, resolve_device
    from .models.multi_trait_lmm import MultiTraitLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    config = TorchGWASConfig(
        device=device,
        chunk_size=args.chunk_size,
    )
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    # Parse trait names
    if args.traits is None:
        logger.error("mvlmm-scan requires --traits (comma-separated trait column names)")
        return 1
    trait_names = [t.strip() for t in args.traits.split(",")]
    if len(trait_names) < 2:
        logger.error("mvlmm-scan requires at least 2 traits (got %d)", len(trait_names))
        return 1

    grm_method = getattr(args, "grm_method", "vanraden")

    # Streaming: align samples without materializing G
    Y, X0, aligned_reader = _align_samples(args, config, trait_columns=trait_names)

    if Y.shape[1] < 2:
        logger.error("Only %d trait(s) loaded — mvlmm-scan requires >= 2.", Y.shape[1])
        return 1

    # Compute GRM — streaming or full depending on method
    if grm_method == "zhang":
        logger.info("Computing kinship matrix (Zhang method, full materialization)...")
        from .linalg.kinship import grm_zhang
        G, _ = _load_full_genotype(aligned_reader)
        K, grm_meta = grm_zhang(G)
        del G
    else:
        logger.info("Computing kinship matrix (VanRaden streaming)...")
        from .linalg.kinship import grm_vanraden_streaming
        K, grm_meta = grm_vanraden_streaming(
            _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
            n_samples=aligned_reader.n_samples,
            device=device,
        )

    logger.info("GRM: %d samples, %d SNPs used", grm_meta.n_samples, grm_meta.n_snps_used)

    # Move to device
    Y = Y.to(device)
    X0 = X0.to(device)
    K = K.to(device)

    model = MultiTraitLMM()
    null_fit = model.fit_null(Y, X0, K=K)

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_mklmm_scan(args: argparse.Namespace) -> int:
    """Run multi-kernel LMM scan (additive + dominance + epistatic variance components)."""
    from .config import TorchGWASConfig, resolve_device
    from .models.multi_kernel_lmm import MultiKernelLMM, build_multi_kernels
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter
    ploidy = getattr(args, "ploidy", 2)

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    # Parse kernel types
    kernel_types = [k.strip() for k in args.kernels.split(",")]
    include_dom = "dominance" in kernel_types
    include_epi = "epistatic" in kernel_types

    logger.info("Multi-kernel LMM: kernels=%s, ploidy=%d", kernel_types, ploidy)

    kernels, kernel_names = build_multi_kernels(
        G.to(device), ploidy=ploidy,
        include_dominance=include_dom,
        include_epistatic=include_epi,
    )
    logger.info("Built %d kernels: %s", len(kernels), kernel_names)

    Y_dev = Y.to(device)
    X0_dev = X0.to(device)

    model = MultiKernelLMM()
    null_fit = model.fit_null(
        Y_dev, X0_dev,
        kernels=kernels,
        kernel_names=kernel_names,
    )

    # Log partitioned heritability
    h2 = null_fit.partitioned_h2
    logger.info("Partitioned h²: %s", {k: f"{v:.4f}" for k, v in h2.items()})

    # Per-kernel Wald tests
    kernel_tests = model.per_kernel_wald_tests(null_fit)
    for name, (z, p) in kernel_tests.items():
        logger.info("Kernel %s: Z=%.3f, p=%.2e", name, z, p)

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_gxe_scan(args: argparse.Namespace) -> int:
    """Run gene-environment interaction LMM scan."""
    import pandas as pd
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .linalg.kinship import grm_vanraden
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    # Load environment variable
    env_df = pd.read_csv(args.env, sep="\t")
    env = torch.tensor(env_df["ENV"].values, dtype=STAT_DTYPE, device=device)

    # Build GRM
    K, _ = grm_vanraden(G.to(device))

    Y_dev = Y.to(device)
    X0_dev = X0.to(device)

    if args.gxe_model == "het":
        from .models.lmm_gxe import HetLMM

        logger.info("GxE scan: HetLMM (single-trait)")
        model = HetLMM()
        null_fit = model.fit_null(Y_dev, X0_dev, K=K, env=env)
    else:
        from .models.lmm_gxe import GxELMM

        if args.traits is None:
            raise ValueError("--traits is required for multi-trait GxE model")
        logger.info("GxE scan: GxELMM (multi-trait)")
        model = GxELMM()
        null_fit = model.fit_null(Y_dev, X0_dev, K=K, env=env)

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_set_scan(args: argparse.Namespace) -> int:
    """Run set-based association tests (SKAT/Burden/SKAT-O).

    Streaming variant: builds GRM via the streaming VanRaden path and
    accumulates per-region buffers chunk-by-chunk. Peak memory is
    bounded by ``n_samples × sum(region_size_j)`` plus one chunk —
    not the full ``(n, m)`` genotype matrix.
    """
    import pandas as pd

    from .config import TorchGWASConfig, resolve_device
    from .io.regions import load_regions
    from .linalg.kinship import grm_vanraden_streaming
    from .models.set_based import SetBasedScanner
    from .models.single_trait_lmm import SingleTraitLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    # Streaming sample alignment — never materializes G.
    Y, X0, aligned_reader = _align_samples(args, config)

    # Streaming GRM (ploidy-aware) for the LMM null fit.
    ploidy = getattr(args, "ploidy", 2)
    logger.info("Computing kinship matrix (VanRaden streaming, ploidy=%d)...", ploidy)
    K, grm_meta = grm_vanraden_streaming(
        _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
        n_samples=aligned_reader.n_samples,
        ploidy=ploidy,
        device=device,
    )
    logger.info("GRM: %d samples, %d SNPs used",
                grm_meta.n_samples, grm_meta.n_snps_used)

    # Fit null model (same as standard LMM)
    lmm = SingleTraitLMM(config=config.numerical)
    null_fit = lmm.fit_null(Y.to(device), X0.to(device), K=K)

    # Load regions. Genome-wide chr/pos lists are gathered cheaply by
    # walking iter_chunks once for metadata only — no genotype data is
    # held. (Not every reader implements `.variant_meta` as a property,
    # so we use the iterator-only contract.)
    regions = load_regions(args.regions)

    # Streaming set-based scan: SetBasedScanner.scan_regions_streaming
    # iterates chunks and copies columns into per-region buffers without
    # ever holding the full (n, m) tensor. We pre-collect the variant
    # annotation (chr/pos lists) from one metadata-only walk; this is
    # O(m) integers, not O(n*m) genotype values.
    full_chr: list[str] = []
    full_pos: list[int] = []
    for _gchunk, vm in aligned_reader.iter_chunks(config.chunk_size):
        full_chr.extend(vm.chr)
        full_pos.extend(vm.pos)
        del _gchunk  # release per-chunk dosage early

    scanner = SetBasedScanner(null_fit, ploidy=ploidy)
    result = scanner.scan_regions_streaming(
        aligned_reader.iter_chunks(config.chunk_size),
        regions,
        variant_chr=full_chr, variant_pos=full_pos,
        test=args.set_test,
        device=device,
    )

    # Save results as TSV
    output_path = args.output + "_set_based.tsv"
    rows = []
    for i in range(len(result)):
        row = {
            "REGION": result.region_id[i],
            "CHR": result.chr[i],
            "START": result.start[i],
            "END": result.end[i],
            "N_VARIANTS": result.n_variants[i],
            "Q_STAT": result.q_stat[i].item(),
            "P": result.p[i].item(),
            "TEST": result.test,
        }
        if result.rho_opt is not None:
            row["RHO_OPT"] = result.rho_opt[i].item()
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Set-based results saved to %s (%d regions)", output_path, len(result))
    return 0


def _cmd_bayes_scan(args: argparse.Namespace) -> int:
    """Run Bayesian variable selection (spike-and-slab) GWAS."""
    import pandas as pd

    from .config import TorchGWASConfig, resolve_device
    from .linalg.kinship import grm_vanraden
    from .models.bayesian_vs import BayesianVS
    from .models.single_trait_lmm import SingleTraitLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    # Build GRM (ploidy-aware)
    ploidy = getattr(args, "ploidy", 2)
    K, _ = grm_vanraden(G.to(device), ploidy=ploidy)

    # Fit null model (same as standard LMM)
    lmm = SingleTraitLMM()
    null_fit = lmm.fit_null(Y.to(device), X0.to(device), K=K)

    # Run Bayesian variable selection
    bvs = BayesianVS(null_fit, ploidy=ploidy)
    result = bvs.fit(
        G.to(device), vmeta,
        method=args.method,
        n_signals=args.n_signals,
        max_iter=args.max_iter or 1000,
        prior_pi=args.prior_pi,
        prior_sig2_beta=args.prior_sig2_beta,
        learn_hyperparams=args.learn_hyperparams,
        credible_set_coverage=args.credible_set_coverage,
        ld_threshold=args.ld_threshold,
    )

    # Save results as TSV
    output_path = args.output + "_bayesian_vs.tsv"
    rows = []
    for i in range(len(result)):
        rows.append({
            "SNP": result.snp[i],
            "CHR": result.chr[i],
            "POS": result.pos[i],
            "A1": result.a1[i],
            "A2": result.a2[i],
            "AF": result.af[i].item(),
            "PIP": result.pip[i].item(),
            "BETA_MEAN": result.beta_mean[i].item(),
            "BETA_SD": result.beta_sd[i].item(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(
        "Bayesian VS (%s) results saved to %s (%d SNPs, pi=%.4f, sig2_beta=%.4f, converged=%s)",
        result.method, output_path, len(result), result.prior_pi, result.prior_sig2_beta,
        result.converged,
    )

    # Save credible sets if present
    if result.credible_sets:
        cs_path = args.output + "_credible_sets.tsv"
        cs_rows = []
        for cs_idx, cs in enumerate(result.credible_sets):
            for snp_idx in cs:
                cs_rows.append({
                    "CS_ID": cs_idx,
                    "SNP": result.snp[snp_idx],
                    "PIP": result.pip[snp_idx].item(),
                })
        pd.DataFrame(cs_rows).to_csv(cs_path, sep="\t", index=False)
        logger.info("Credible sets saved to %s (%d sets)", cs_path, len(result.credible_sets))

    return 0


def _cmd_met_scan(args: argparse.Namespace) -> int:
    """Run multi-environment trial (MET) GWAS scan."""
    import pandas as pd
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .linalg.kinship import grm_vanraden
    from .models.multi_env_lmm import MultiEnvLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    # Detect environment columns from phenotype
    pheno_df = pd.read_csv(args.phenotype, sep="\t")
    env_cols = [c for c in pheno_df.columns if c not in ("FID", "IID", "SAMPLE")]
    if args.env_cols:
        env_cols = args.env_cols.split(",")

    if len(env_cols) < 2:
        raise ValueError(
            f"MET-scan requires >= 2 environment columns, found: {env_cols}. "
            "Specify --env-cols col1,col2,..."
        )

    # Build Y_wide (n, E)
    Y_wide = torch.tensor(
        pheno_df[env_cols].values, dtype=STAT_DTYPE, device=device,
    )

    # Build GRM (and optional additional kernels)
    ploidy = getattr(args, "ploidy", 2)
    K_add, _ = grm_vanraden(G.to(device), ploidy=ploidy)

    kernel_files = getattr(args, "kernel_files", None)
    if kernel_files:
        import numpy as np
        K = {"additive": K_add}
        for kf in kernel_files:
            name = Path(kf).stem
            if kf.endswith(".npy"):
                K[name] = torch.tensor(np.load(kf), dtype=STAT_DTYPE, device=device)
            else:
                K[name] = torch.load(kf, map_location=device).to(STAT_DTYPE)
        logger.info("Multi-kernel MET: %s", list(K.keys()))
    else:
        K = K_add

    # Fit null model
    parameterization = getattr(args, "parameterization", "per_env")
    vg_structure = getattr(args, "vg_structure", "unstructured")
    model = MultiEnvLMM(
        config=config.numerical,
        parameterization=parameterization,
        vg_structure=vg_structure,
    )
    null_fit = model.fit_null(Y_wide, X0.to(device), K, env_names=env_cols)

    # Print interpretive quantities
    rg = model.genetic_correlation(null_fit)
    h2 = model.per_env_heritability(null_fit)
    logger.info("Genetic correlations across environments:\n%s", rg)
    logger.info("Per-environment heritability: %s",
                {name: f"{h:.3f}" for name, h in zip(env_cols, h2.tolist())})

    # Scan all chunks
    results = []
    for chunk_idx, (G_c, vm) in enumerate(reader.iter_chunks(config.chunk_size)):
        G_c = G_c.to(device)
        result = model.score_chunk(G_c, null_fit, vm)
        results.append(result)

    # Merge results
    merged = _merge_env_results(results)

    # Save TSV
    _save_met_results(merged, env_cols, args)
    return 0


def _merge_env_results(results: list) -> EnvScanResult:
    """Concatenate EnvScanResult objects across chunks."""
    import torch

    from .models.multi_env_lmm import EnvScanResult

    chr_all, pos_all, snp_all, a1_all, a2_all = [], [], [], [], []
    for r in results:
        chr_all.extend(r.chr)
        pos_all.extend(r.pos)
        snp_all.extend(r.snp)
        a1_all.extend(r.a1)
        a2_all.extend(r.a2)

    # Reaction-norm fields (may be None)
    rn_kwargs = {}
    if results[0].alpha is not None:
        rn_kwargs["alpha"] = torch.cat([r.alpha for r in results])
        rn_kwargs["se_alpha"] = torch.cat([r.se_alpha for r in results])
        rn_kwargs["stat_stable"] = torch.cat([r.stat_stable for r in results])
        rn_kwargs["p_stable"] = torch.cat([r.p_stable for r in results])
        rn_kwargs["delta"] = torch.cat([r.delta for r in results])
        rn_kwargs["stat_gxe"] = torch.cat([r.stat_gxe for r in results])
        rn_kwargs["p_gxe"] = torch.cat([r.p_gxe for r in results])

    return EnvScanResult(
        chr=chr_all, pos=pos_all, snp=snp_all, a1=a1_all, a2=a2_all,
        af=torch.cat([r.af for r in results]),
        beta=torch.cat([r.beta for r in results]),
        se=torch.cat([r.se for r in results]),
        stat=torch.cat([r.stat for r in results]),
        p=torch.cat([r.p for r in results]),
        stat_homogeneity=torch.cat([r.stat_homogeneity for r in results]),
        p_homogeneity=torch.cat([r.p_homogeneity for r in results]),
        stat_marginal=torch.cat([r.stat_marginal for r in results]),
        p_marginal=torch.cat([r.p_marginal for r in results]),
        test="wald",
        env_names=results[0].env_names if results else None,
        **rn_kwargs,
    )


def _save_met_results(result, env_names: list[str], args) -> None:
    """Save MET-GWAS results as TSV with per-environment columns."""
    import pandas as pd

    has_rn = result.alpha is not None

    rows = []
    for i in range(len(result)):
        row = {
            "SNP": result.snp[i],
            "CHR": result.chr[i],
            "POS": result.pos[i],
            "A1": result.a1[i],
            "A2": result.a2[i],
            "AF": result.af[i].item(),
            "STAT_JOINT": result.stat[i].item(),
            "P_JOINT": result.p[i].item(),
            "STAT_HOM": result.stat_homogeneity[i].item(),
            "P_HOM": result.p_homogeneity[i].item(),
        }
        # Reaction-norm columns
        if has_rn:
            row["ALPHA"] = result.alpha[i].item()
            row["SE_ALPHA"] = result.se_alpha[i].item()
            row["P_STABLE"] = result.p_stable[i].item()
            row["STAT_GXE"] = result.stat_gxe[i].item()
            row["P_GXE"] = result.p_gxe[i].item()
            for e, name in enumerate(env_names):
                row[f"DELTA_{name}"] = result.delta[i, e].item()

        for e, name in enumerate(env_names):
            row[f"BETA_{name}"] = result.beta[i, e].item()
            row[f"SE_{name}"] = result.se[i, e].item()
            row[f"P_{name}"] = result.p_marginal[i, e].item()
        rows.append(row)

    output_path = args.output + "_met_scan.tsv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("MET-scan results saved to %s (%d variants)", output_path, len(result))


def _cmd_farmcpu_scan(args: argparse.Namespace) -> int:
    """Run FarmCPU multi-locus GWAS scan."""
    return _run_per_trait(args, _cmd_farmcpu_scan_single)


def _cmd_farmcpu_scan_single(args: argparse.Namespace) -> int:
    """Run FarmCPU scan for a single trait."""
    from .config import TorchGWASConfig, resolve_device
    from .models.farmcpu import FarmCPU
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    bin_sizes = [int(x) for x in args.bin_sizes.split(",")]
    max_iter = getattr(args, "max_iter", None) or 10

    model = FarmCPU(
        max_iter=max_iter,
        p_threshold=args.p_threshold,
        max_qtns=args.max_qtns,
        bin_sizes=bin_sizes,
        method_bin=args.method_bin,
        method_sub=args.method_sub,
        maf_threshold=args.maf_threshold,
    )
    logger.info(
        "FarmCPU: max_iter=%d, p_threshold=%.4f, max_qtns=%d, bins=%s",
        max_iter, args.p_threshold, args.max_qtns, bin_sizes,
    )

    null_fit = model.fit_null(
        Y.to(device), X0.to(device),
        G=G.to(device), variant_meta=vmeta,
    )

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_blink_scan(args: argparse.Namespace) -> int:
    """Run BLINK multi-locus GWAS scan."""
    return _run_per_trait(args, _cmd_blink_scan_single)


def _cmd_blink_scan_single(args: argparse.Namespace) -> int:
    """Run BLINK scan for a single trait."""
    from .config import TorchGWASConfig, resolve_device
    from .models.blink import BLINK
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    max_iter = getattr(args, "max_iter", None) or 10
    max_qtns = args.max_qtns if args.max_qtns is not None and args.max_qtns > 0 else None

    model = BLINK(
        max_iter=max_iter,
        cutoff=args.cutoff,
        max_qtns=max_qtns,
        ld_threshold=args.ld_threshold,
        ld_max_samples=args.ld_max_samples,
        method_sub=args.method_sub,
        maf_threshold=args.maf_threshold,
    )
    logger.info(
        "BLINK: max_iter=%d, cutoff=%.4f, ld_threshold=%.2f",
        max_iter, args.cutoff, args.ld_threshold,
    )

    null_fit = model.fit_null(
        Y.to(device), X0.to(device),
        G=G.to(device), variant_meta=vmeta,
    )

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_threshold_scan(args: argparse.Namespace) -> int:
    """Run threshold-linear GWAS scan for ordinal + continuous traits."""
    import numpy as np
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .models.threshold_linear import ThresholdLinearModel
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    trait_types = [t.strip() for t in args.trait_types.split(",")]
    n_categories = [int(x.strip()) for x in args.n_categories.split(",")]
    c = len(trait_types)

    # Load R and G covariance matrices
    if args.R_matrix:
        R = torch.tensor(np.loadtxt(args.R_matrix, delimiter=","), dtype=STAT_DTYPE)
    else:
        R = torch.eye(c, dtype=STAT_DTYPE)
    if args.G_matrix:
        G_cov = torch.tensor(np.loadtxt(args.G_matrix, delimiter=","), dtype=STAT_DTYPE)
    else:
        G_cov = torch.eye(c, dtype=STAT_DTYPE) * 0.5

    max_iter = getattr(args, "max_iter", None) or 100

    model = ThresholdLinearModel(
        trait_types=trait_types,
        n_categories=n_categories,
        R=R,
        G_cov=G_cov,
        solver=args.solver,
        max_iter=max_iter,
        em_warmup=args.em_warmup,
    )
    logger.info(
        "Threshold-linear: solver=%s, traits=%s, max_iter=%d",
        args.solver, args.trait_types, max_iter,
    )

    null_fit = model.fit_null(Y.to(device), X0.to(device))

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test="score", qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_conditional_scan(args: argparse.Namespace) -> int:
    """Run LD-conditional GWAS scan with persistence metrics."""
    from .config import TorchGWASConfig, resolve_device
    from .models.conditional_lmm import ConditionalLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    Y, X0, aligned_reader = _align_samples(args, config)

    # Compute GRM
    logger.info("Computing kinship matrix (VanRaden streaming)...")
    from .linalg.kinship import grm_vanraden_streaming
    K, grm_meta = grm_vanraden_streaming(
        _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
        n_samples=aligned_reader.n_samples,
        device=device,
    )

    model = ConditionalLMM(
        config=config.numerical,
        ld_method=args.ld_method,
        max_kb=args.max_kb,
        persistence_ratio=args.persistence_ratio,
        sig_threshold=args.sig_threshold,
        max_conditioning=args.max_conditioning,
    )
    null_fit = model.fit_null(Y.to(device), X0.to(device), K=K)

    logger.info("Null fit: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
                null_fit.sig2_g, null_fit.sig2_e,
                null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e))

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_mtmet_scan(args: argparse.Namespace) -> int:
    """Run multi-trait multi-environment GWAS scan."""
    from .config import TorchGWASConfig, resolve_device
    from .models.multi_trait_multi_env_lmm import MultiTraitMultiEnvLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # Parse trait and environment columns
    if args.traits is None:
        logger.error("mtmet-scan requires --traits (comma-separated trait column names)")
        return 1
    trait_names = [t.strip() for t in args.traits.split(",")]
    env_names = [e.strip() for e in args.env_cols.split(",")]
    d = len(trait_names)
    E = len(env_names)

    if d < 2:
        logger.error("mtmet-scan requires >= 2 traits (got %d)", d)
        return 1
    if E < 2:
        logger.error("mtmet-scan requires >= 2 environments (got %d)", E)
        return 1

    # Load phenotype: expect columns like trait_env or user-specified pattern
    import pandas as pd


    pheno_df = pd.read_csv(args.phenotype, sep=None, engine="python")

    # Detect ID column
    id_col = None
    for col in pheno_df.columns:
        if col.lower() in {"iid", "id", "sample", "sample_id", "sampleid"}:
            id_col = col
            break
    if id_col is None:
        id_col = pheno_df.columns[0]

    # Build Y tensor (n, d, E) from columns named {trait}_{env}
    dE_cols = []
    for t in trait_names:
        for e in env_names:
            col_name = f"{t}_{e}"
            if col_name not in pheno_df.columns:
                raise ValueError(
                    f"Expected column '{col_name}' in phenotype file. "
                    f"Columns should be named {{trait}}_{{env}}."
                )
            dE_cols.append(col_name)

    # Align with genotype
    Y, X0, aligned_reader = _align_samples(args, config, trait_columns=dE_cols)

    # Compute GRM
    logger.info("Computing kinship matrix (VanRaden streaming)...")
    from .linalg.kinship import grm_vanraden_streaming
    K, grm_meta = grm_vanraden_streaming(
        _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
        n_samples=aligned_reader.n_samples,
        device=device,
    )

    model = MultiTraitMultiEnvLMM(
        config=config.numerical,
        vg_structure=args.vg_structure,
    )
    null_fit = model.fit_null(
        Y.to(device), X0.to(device), K=K,
        n_traits=d, n_envs=E,
        trait_names=trait_names, env_names=env_names,
    )

    logger.info(
        "MT-MET null fit: ll=%.4f, h2 matrix:\n%s",
        null_fit.log_likelihood,
        MultiTraitMultiEnvLMM.per_trait_per_env_heritability(null_fit),
    )

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test="wald", qc_config=qc)

    _save_mtmet_results(result, args, trait_names, env_names)
    return 0


def _save_mtmet_results(
    result: MTMETScanResult,
    args: argparse.Namespace,
    trait_names: list[str],
    env_names: list[str],
) -> None:
    """Save MT-MET results with flattened per-trait-per-env columns."""
    import pandas as pd

    from .stats.multipletesting import (
        benjamini_hochberg,
        benjamini_yekutieli,
        bonferroni,
        holm,
        storey_qvalue,
    )

    p = result.p

    # Apply multiple testing correction on joint p-values
    correction = args.correction
    if correction == "none":
        p_adj = p
    elif correction == "bonferroni":
        p_adj = bonferroni(p)
    elif correction == "holm":
        p_adj = holm(p)
    elif correction == "bh":
        p_adj = benjamini_hochberg(p)
    elif correction == "by":
        p_adj = benjamini_yekutieli(p)
    elif correction == "storey":
        p_adj = storey_qvalue(p)
    else:
        logger.warning("Unknown correction '%s', using none", correction)
        p_adj = p

    df = pd.DataFrame({
        "CHR": result.chr,
        "POS": result.pos,
        "SNP": result.snp,
        "A1": result.a1,
        "A2": result.a2,
        "AF": result.af.cpu().numpy(),
    })

    # Flatten beta (m, d, E) and se (m, d, E) to per-trait-per-env columns
    beta_np = result.beta.cpu().numpy()  # (m, d, E)
    se_np = result.se.cpu().numpy()
    for ti, t in enumerate(trait_names):
        for ei, e in enumerate(env_names):
            df[f"BETA_{t}_{e}"] = beta_np[:, ti, ei]
            df[f"SE_{t}_{e}"] = se_np[:, ti, ei]

    # Joint test
    df["STAT_JOINT"] = result.stat.cpu().numpy()
    df["P_JOINT"] = result.p.cpu().numpy()
    df["P_JOINT_ADJ"] = p_adj.cpu().numpy()

    # Per-trait p-values
    pt_np = result.p_per_trait.cpu().numpy()
    for ti, t in enumerate(trait_names):
        df[f"P_TRAIT_{t}"] = pt_np[:, ti]

    # Per-env p-values
    pe_np = result.p_per_env.cpu().numpy()
    for ei, e in enumerate(env_names):
        df[f"P_ENV_{e}"] = pe_np[:, ei]

    # Per-trait GxE p-values
    pgxe_np = result.p_gxe_per_trait.cpu().numpy()
    for ti, t in enumerate(trait_names):
        df[f"P_GXE_{t}"] = pgxe_np[:, ti]

    out_path = args.output
    tsv_path = f"{out_path}.assoc.tsv"
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6e")
    logger.info("MT-MET results written to %s (%d variants)", tsv_path, len(df))

    n_sig_raw = int((result.p < 5e-8).sum().item())
    n_sig_adj = int((p_adj < 0.05).sum().item())
    print(f"\nResults: {len(result)} variants tested")
    print(f"  Genome-wide significant (P_JOINT < 5e-8): {n_sig_raw}")
    print(f"  Significant after {correction} (adj P < 0.05): {n_sig_adj}")
    print(f"  Output: {tsv_path}")


def _cmd_ocf_scan(args: argparse.Namespace) -> int:
    """Run Orthogonal Cross-Fit LMM (debiased) GWAS scan."""
    from .config import TorchGWASConfig, resolve_device
    from .models.ocf_lmm import OCFLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    # Load and align samples
    Y, X0, aligned_reader = _align_samples(args, config)

    # Compute GRM
    logger.info("Computing kinship matrix (VanRaden streaming)...")
    from .linalg.kinship import grm_vanraden_streaming
    K, grm_meta = grm_vanraden_streaming(
        _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
        n_samples=aligned_reader.n_samples,
        device=device,
    )

    model = OCFLMM(
        config=config.numerical,
        n_folds=args.n_folds,
        seed=args.seed,
        variance_type=args.variance_type,
        project_genotype=not args.no_genotype_projection,
    )
    null_fit = model.fit_null(Y.to(device), X0.to(device), K=K)

    logger.info(
        "OCF-LMM null fit: mean_h2=%.4f, n_folds=%d, converged=%s",
        null_fit.mean_h2, null_fit.n_folds, null_fit.converged,
    )

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test="wald", qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_knockoff_scan(args: argparse.Namespace) -> int:
    """Run knockoff FDR-controlled GWAS scan."""
    import pandas as pd

    from .config import TorchGWASConfig, resolve_device
    from .models.knockoff_lmm import KnockoffLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # Load ALL genotypes (knockoff needs full G)
    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    # Compute GRM
    logger.info("Computing kinship matrix (VanRaden)...")
    from .linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G.to(device))

    model = KnockoffLMM(
        config=config.numerical,
        target_fdr=args.fdr_level,
        ld_method=args.ld_method,
        knockoff_method=args.knockoff_method,
        aggregation=args.aggregation,
        seed=args.seed,
    )
    result = model.run(
        Y.squeeze(1).to(device), X0.to(device), K,
        G.to(device), vmeta,
        vmeta.pos, vmeta.chr,
    )

    # Save results
    output = args.output
    m = result.beta.shape[0]

    df = pd.DataFrame({
        "CHR": result.chr,
        "POS": result.pos,
        "SNP": result.snp,
        "A1": result.a1,
        "A2": result.a2,
        "AF": result.af.cpu().numpy(),
        "BETA": result.beta.cpu().numpy(),
        "SE": result.se.cpu().numpy(),
        "STAT": result.stat.cpu().numpy(),
        "P": result.p.cpu().numpy(),
        "BETA_KNOCK": result.beta_knockoff.cpu().numpy(),
        "STAT_KNOCK": result.stat_knockoff.cpu().numpy(),
        "SELECTED": result.is_selected.cpu().numpy().astype(int),
    })
    out_path = f"{output}.knockoff.tsv"
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6e")

    # Block-level summary
    block_rows = []
    for b in range(result.n_blocks):
        block_rows.append({
            "BLOCK": b,
            "N_SNPS": len(result.block_indices[b]),
            "W_STAT": result.W_stat[b].item(),
            "SELECTED": 1 if b in result.selected_blocks else 0,
        })
    df_blocks = pd.DataFrame(block_rows)
    block_path = f"{output}.knockoff_blocks.tsv"
    df_blocks.to_csv(block_path, sep="\t", index=False, float_format="%.6e")

    logger.info("Results saved: %s, %s", out_path, block_path)
    logger.info("Knockoff filter: FDR=%.3f, threshold=%.4f, %d/%d blocks selected",
                result.target_fdr,
                result.threshold if result.threshold != float("inf") else 0.0,
                result.n_selected, result.n_blocks)
    return 0


def _cmd_gu_scan(args: argparse.Namespace) -> int:
    """Run genotype-uncertainty–corrected GWAS scan."""
    import json
    import tempfile
    from pathlib import Path

    import torch

    from .config import TorchGWASConfig, resolve_device

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # --- Phase 55 `--probs` passthrough -------------------------------------
    # If the user passes `--probs out/dcall.probs.pt`, derive --genotype and
    # --dosage-var in a tempdir so the rest of the flow sees them as usual
    # .pt files. Semantics match the manual two-step recipe in
    # docs/getting-started/polyploid_dosage_call.md byte-for-byte.
    _probs_tmpdir: Path | None = None
    if args.probs:
        if args.genotype:
            raise ValueError(
                "--probs and --genotype are mutually exclusive. Pass exactly "
                "one: either a Phase 55 probs.pt (preferred) or a pre-derived "
                "dosage tensor."
            )
        from .preprocess.dosage_uncertainty import (
            dosage_variance,
            expected_dosage,
        )

        probs_path = Path(args.probs)
        probs = torch.load(str(probs_path), weights_only=True)
        meta_path = probs_path.with_suffix("").with_suffix(".meta.json")
        # Support both `<prefix>.probs.pt` and arbitrary paths whose
        # suffix doesn't terminate in `.probs.pt` — fall back to reading
        # ploidy from the probs tensor shape.
        ploidy: int
        if meta_path.is_file():
            ploidy = int(json.loads(meta_path.read_text())["ploidy"])
        else:
            ploidy = int(probs.shape[-1]) - 1
            logger.info(
                "No sibling meta.json at %s; inferring ploidy=%d from "
                "probs.shape[-1]-1.", meta_path, ploidy,
            )

        G_derived = expected_dosage(probs, ploidy)              # (n, m)
        dvar_derived = dosage_variance(probs, ploidy)            # (n, m)

        _probs_tmpdir = Path(tempfile.mkdtemp(prefix="torchgwas_guscan_"))
        g_tmp = _probs_tmpdir / "genotype.pt"
        dv_tmp = _probs_tmpdir / "dosage_var.pt"
        torch.save(G_derived, str(g_tmp))
        torch.save(dvar_derived, str(dv_tmp))
        args.genotype = str(g_tmp)
        # Only overwrite --dosage-var if the user didn't pass one. A user
        # override wins — this is an escape hatch for custom variance.
        if not args.dosage_var:
            args.dosage_var = str(dv_tmp)
        logger.info(
            "Derived genotype (%s) and dosage variance (%s) from %s "
            "(ploidy=%d).", g_tmp, dv_tmp, probs_path, ploidy,
        )
    elif not args.genotype:
        raise ValueError(
            "gu-scan requires exactly one of --genotype or --probs."
        )

    try:
        return _cmd_gu_scan_inner(args, config, device)
    finally:
        if _probs_tmpdir is not None:
            import shutil
            shutil.rmtree(_probs_tmpdir, ignore_errors=True)


def _cmd_gu_scan_inner(args, config, device) -> int:
    """The original gu-scan body after --probs derivation (if any)."""
    import torch

    from .models.gu_lmm import GULM

    # Load ALL genotypes (need full G for dosage variance)
    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    # Compute GRM
    logger.info("Computing kinship matrix (VanRaden)...")
    from .linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G.to(device))

    # Load dosage variance if provided
    dosage_var = None
    if args.dosage_var:
        dvar_path = args.dosage_var
        if dvar_path.endswith(".pt"):
            dosage_var = torch.load(dvar_path).to(device)
        else:
            import numpy as np
            dosage_var = torch.tensor(
                np.load(dvar_path), dtype=torch.float64, device=device,
            )
        logger.info("Loaded dosage variance: shape %s", tuple(dosage_var.shape))

    model = GULM(config=config.numerical)
    null_fit = model.fit_null(Y.squeeze(1).to(device), X0.to(device), K=K)

    from .preprocess.qc import QCFilterConfig
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)

    # Score all variants with uncertainty correction
    result = model.score_chunk(
        G.to(device), null_fit, vmeta,
        test="score", dosage_var=dosage_var,
    )

    _apply_correction_and_save(result, args)
    return 0


def _cmd_lro_scan(args: argparse.Namespace) -> int:
    """Run leave-region-out GWAS scan (block-level LOCO)."""
    from .config import TorchGWASConfig, resolve_device
    from .models.lro_lmm import LROLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # Load ALL genotypes (LRO needs full G)
    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    model = LROLMM(
        config=config.numerical,
        ld_method=args.ld_method,
    )
    result = model.run(
        Y.squeeze(1).to(device), X0.to(device),
        G.to(device), vmeta,
        vmeta.pos, vmeta.chr,
        test=args.test,
    )

    # Save as ScanResult-like format
    from .models.base import ScanResult
    scan_result = ScanResult(
        chr=result.chr, pos=result.pos, snp=result.snp,
        a1=result.a1, a2=result.a2, af=result.af,
        beta=result.beta, se=result.se,
        stat=result.stat, p=result.p, test=result.test,
    )
    _apply_correction_and_save(scan_result, args)
    logger.info("LRO-LMM: %d blocks (%s method), test=%s",
                result.n_blocks, result.block_method, result.test)
    return 0


def _cmd_glmm_scan(args: argparse.Namespace) -> int:
    """Run GLMM association scan (binary/ordinal with random effects)."""
    from .config import TorchGWASConfig, resolve_device

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    # Build GRM
    from .linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G.to(device))

    family = getattr(args, "family", "binary")
    if family == "binary":
        from .models.binary_glmm import BinaryGLMM
        model = BinaryGLMM(
            use_spa=not getattr(args, "no_spa", False),
            firth=getattr(args, "firth", False),
            pql_max_iter=getattr(args, "pql_max_iter", 30),
        )
    elif family == "ordinal":
        from .models.ordinal_glmm import OrdinalGLMM
        model = OrdinalGLMM(
            n_categories=args.n_categories,
            pql_max_iter=getattr(args, "pql_max_iter", 30),
        )
    elif family == "multinomial":
        from .models.multinomial_glmm import MultinomialGLMM
        model = MultinomialGLMM(
            n_classes=args.n_categories,
            pql_max_iter=getattr(args, "pql_max_iter", 30),
        )
    else:
        raise ValueError(f"Unsupported GLMM family: {family}")

    nf = model.fit_null(Y.squeeze(1).to(device), X0.to(device), K=K)
    result = model.score_chunk(G.to(device), nf, vmeta)

    _apply_correction_and_save(result, args)
    logger.info("GLMM scan complete: family=%s, %d variants", family, len(result))
    return 0


def _cmd_me_glmm_scan(args: argparse.Namespace) -> int:
    """Run multi-environment GLMM association scan."""
    from .config import TorchGWASConfig, resolve_device

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    # Y should be (n, E) — multiple phenotype columns as environments
    env_cols = getattr(args, "env_cols", None)
    if env_cols:
        env_names = [c.strip() for c in env_cols.split(",")]
    else:
        env_names = None

    # Reshape Y to (n, E) if needed
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)
    if Y.shape[1] < 2:
        raise ValueError("me-glmm-scan requires at least 2 environments in phenotype")

    # Build GRM
    from .linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G.to(device))

    family = getattr(args, "family", "binary")
    from .models.multi_env_glmm import MultiEnvGLMM
    model = MultiEnvGLMM(
        family=family,
        n_categories=getattr(args, "n_categories", None),
        parameterization=getattr(args, "parameterization", "per_env"),
        use_spa=not getattr(args, "no_spa", False),
        firth=getattr(args, "firth", False),
        pql_max_iter=getattr(args, "pql_max_iter", 30),
    )

    nf = model.fit_null(Y.to(device), X0.to(device), K=K, env_names=env_names)
    result = model.score_chunk(G.to(device), nf, vmeta)

    _apply_correction_and_save(result, args)
    logger.info("ME-GLMM scan complete: family=%s, E=%d, %d variants",
                family, Y.shape[1], len(result))
    return 0


def _cmd_survival_scan(args: argparse.Namespace) -> int:
    """Run survival GWAS scan (Cox PH frailty model)."""
    from .config import TorchGWASConfig, resolve_device

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    G, Y, X0, vmeta, aligned_reader = _load_scan_data(args, config)

    # Y should be (n, 2) with columns [time, event]
    if Y.ndim == 1:
        raise ValueError(
            "survival-scan requires (n, 2) phenotype with time and event columns. "
            "Got 1D phenotype."
        )
    if Y.shape[1] != 2:
        raise ValueError(
            f"survival-scan requires exactly 2 phenotype columns [time, event], "
            f"got {Y.shape[1]}"
        )

    # Build GRM
    from .linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G.to(device))

    from .models.survival_glmm import SurvivalGLMM
    model = SurvivalGLMM(
        use_spa=not getattr(args, "no_spa", False),
        spa_threshold=getattr(args, "spa_threshold", 2.0),
        pql_max_iter=getattr(args, "pql_max_iter", 30),
        pql_tol=getattr(args, "pql_tol", 1e-4),
        ties=getattr(args, "ties", "breslow"),
    )

    nf = model.fit_null(Y.to(device), X0.to(device), K=K)
    result = model.score_chunk(G.to(device), nf, vmeta)

    _apply_correction_and_save(result, args)
    logger.info("Survival scan complete: n=%d, %d variants", Y.shape[0], len(result))
    return 0


def _cmd_family_scan(args: argparse.Namespace) -> int:
    """Run within-family GWAS scan with confounding diagnostics."""
    import pandas as pd
    import torch

    from .config import TorchGWASConfig, resolve_device
    from .models.within_family_lmm import WithinFamilyLMM
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # Load and align samples
    Y, X0, aligned_reader = _align_samples(args, config)

    # Extract family IDs from phenotype file
    pheno_df = pd.read_csv(args.phenotype, sep=None, engine="python")
    family_col = args.family_col
    if family_col not in pheno_df.columns:
        raise ValueError(
            f"Family column '{family_col}' not found in phenotype file. "
            f"Available columns: {list(pheno_df.columns)}"
        )

    # Map family IDs to integer codes aligned to model sample order
    from .io.phenotype import load_phenotype
    pheno_data = load_phenotype(
        args.phenotype,
        genotype_sample_ids=aligned_reader.sample_ids,
    )
    # Re-read to get FID column in aligned order
    pheno_df_reread = pd.read_csv(args.phenotype, sep=None, engine="python")
    # Detect ID column
    id_col = None
    for col in pheno_df_reread.columns:
        if col.lower() in {"iid", "id", "sample", "sample_id", "sampleid"}:
            id_col = col
            break
    if id_col is None:
        id_col = pheno_df_reread.columns[0]

    pheno_df_reread[id_col] = pheno_df_reread[id_col].astype(str)
    id_to_fid = dict(zip(pheno_df_reread[id_col], pheno_df_reread[family_col]))

    fid_list = [id_to_fid.get(sid, sid) for sid in pheno_data.sample_ids]
    # Encode as integer
    unique_fids = sorted(set(fid_list))
    fid_to_int = {fid: i for i, fid in enumerate(unique_fids)}
    family_ids = torch.tensor([fid_to_int[f] for f in fid_list], dtype=torch.long)

    # Compute GRM (streaming VanRaden)
    logger.info("Computing kinship matrix (VanRaden streaming)...")
    from .linalg.kinship import grm_vanraden_streaming
    K, grm_meta = grm_vanraden_streaming(
        _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
        n_samples=aligned_reader.n_samples,
        device=device,
    )

    model = WithinFamilyLMM(
        min_family_size=args.min_family_size,
        confound_threshold=args.confound_threshold,
    )
    null_fit = model.fit_null(Y.to(device), X0.to(device), K=K, family_ids=family_ids)

    from .scan.unified import UnifiedScanner
    scanner = UnifiedScanner(aligned_reader, model, config)
    qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
    result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    _apply_correction_and_save(result, args)
    return 0


def _cmd_poly_scan(args: argparse.Namespace) -> int:
    """Run polyploid GWAS scan with ploidy-aware GRM and gene-action recoding.

    Key difference from diploid scans:
    - GRM uses grm_polyploid_gene_action (not grm_zhang) which recodes
      genotypes under the specified gene-action model before GRM construction.
    - Effective ploidy differs per model (binary models → 1, diplo-additive → k/2).
    - When --gene-action=all, scans every applicable model and outputs per-model results,
      then selects the best model per marker and performs peak pruning + optional joint QTL.
    """
    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .linalg.kinship_polyploid import grm_polyploid_gene_action
    from .models.single_trait_lmm import SingleTraitLMM
    from .preprocess.polyploid import list_gene_action_models, recode_gene_action
    from .preprocess.qc import QCFilterConfig, compute_max_genotype_freq

    device = resolve_device(args.device)
    logger.info("Device: %s", device)
    ploidy = args.ploidy
    p3d = getattr(args, "p3d", True)
    max_geno_freq = getattr(args, "max_geno_freq", None)

    config = TorchGWASConfig(
        device=device,
        chunk_size=args.chunk_size,
    )
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    G, Y, X0, vmeta, reader = _load_scan_data(args, config)

    # Determine which gene-action models to run
    gene_action = args.gene_action
    if gene_action == "all":
        models_to_run = list_gene_action_models(ploidy)
    else:
        models_to_run = [gene_action]

    logger.info(
        "Polyploid scan: ploidy=%d, gene-action models=%s, P3D=%s",
        ploidy, models_to_run, "TRUE" if p3d else "FALSE",
    )

    # Collect per-model scan results for best-model selection
    all_scan_results = {}

    for ga_model in models_to_run:
        logger.info("--- Gene-action model: %s ---", ga_model)

        # Polyploid GRM: recodes G under gene-action model, then VanRaden
        K, grm_meta = grm_polyploid_gene_action(G, model=ga_model, ploidy=ploidy)
        logger.info(
            "GRM (%s): %d samples, %d SNPs, effective ploidy implied by model",
            ga_model, grm_meta.n_samples, grm_meta.n_snps_used,
        )

        # Recode genotypes for the scan itself (same encoding as GRM)
        if ga_model == "general":
            G_scan = G
        else:
            G_scan = recode_gene_action(G, ga_model, ploidy).to(STAT_DTYPE)

        # Apply max genotype frequency filter after encoding
        if max_geno_freq is not None and G_scan.ndim >= 2:
            mgf = compute_max_genotype_freq(G_scan, ploidy)
            n_filtered = int((mgf > max_geno_freq).sum().item())
            if n_filtered > 0:
                logger.info(
                    "max.geno.freq filter (%.3f): removing %d/%d markers for model %s",
                    max_geno_freq, n_filtered, G_scan.shape[1] if G_scan.ndim == 2 else G_scan.shape[1], ga_model,
                )

        # Move to device
        Y_dev = Y.to(device)
        X0_dev = X0.to(device)
        K_dev = K.to(device)

        model = SingleTraitLMM(p3d=p3d)
        null_fit = model.fit_null(Y_dev, X0_dev, K=K_dev)
        logger.info(
            "Null fit (%s): sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
            ga_model, null_fit.sig2_g, null_fit.sig2_e,
            null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
        )
        if not p3d:
            logger.info("P3D=FALSE: variance components will be re-estimated per marker")

        from .scan.unified import UnifiedScanner
        scanner = UnifiedScanner(reader, model, config)
        qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
        result = scanner.scan(null_fit, test=args.test, qc_config=qc)

        all_scan_results[ga_model] = result

        # Save per-model output
        if len(models_to_run) > 1:
            orig_output = args.output
            args.output = f"{orig_output}_{ga_model}"
            _apply_correction_and_save(result, args)
            args.output = orig_output
        else:
            _apply_correction_and_save(result, args)

    # --- Post-scan analysis (when scanning all models) ---
    if len(models_to_run) > 1:
        _poly_scan_post_analysis(all_scan_results, G, Y, X0, args, ploidy)

    return 0


def _poly_scan_post_analysis(
    all_scan_results: dict,
    G,
    Y,
    X0,
    args: argparse.Namespace,
    ploidy: int,
) -> None:
    """Run best-model selection, peak pruning, and optional joint QTL fitting."""
    import pandas as pd

    n_samples = Y.shape[0]
    output_base = args.output

    # 1. Best-model selection
    try:
        from .stats.best_model import select_best_model
        bic_penalty = True
        best = select_best_model(
            all_scan_results, n_samples=n_samples,
            bic_penalty=bic_penalty, ploidy=ploidy,
        )
        logger.info(
            "Best-model selection: %d markers, BIC penalty=%s",
            len(best.snp), bic_penalty,
        )

        # Save best-model TSV
        df_best = pd.DataFrame({
            "SNP": best.snp,
            "CHR": best.chr,
            "POS": best.pos,
            "BEST_MODEL": best.best_model,
            "BEST_NEGLOG10P": best.best_neglog10p.cpu().numpy(),
            "BEST_P": best.best_p.cpu().numpy(),
        })
        best_path = f"{output_base}_best_model.tsv"
        df_best.to_csv(best_path, sep="\t", index=False, float_format="%.6e")
        logger.info("Best-model results written to %s", best_path)
    except Exception as e:
        logger.warning("Best-model selection failed: %s", e)
        return

    # 2. Peak pruning on best-model result
    try:
        from .stats.peak_pruning import prune_peaks

        # Use the first scan result as the template (SNP positions come from it)
        first_result = next(iter(all_scan_results.values()))
        bp_window = getattr(args, "bp_window", 1_000_000)
        peak_threshold = getattr(args, "peak_threshold", 1e-4)

        peaks = prune_peaks(
            first_result,
            bp_window=bp_window,
            p_threshold=peak_threshold,
            best_models=best.best_model,
        )
        logger.info("Peak pruning: %d QTL peaks detected (window=%d bp)", len(peaks), bp_window)

        if peaks:
            df_peaks = pd.DataFrame([
                {
                    "SNP": pk.snp, "CHR": pk.chr, "POS": pk.pos,
                    "NEGLOG10P": pk.neglog10p,
                    "BEST_MODEL": pk.best_model,
                    "WINDOW_START": pk.window_start,
                    "WINDOW_END": pk.window_end,
                    "N_MARKERS": pk.n_markers_in_window,
                }
                for pk in peaks
            ])
            peaks_path = f"{output_base}_qtl_peaks.tsv"
            df_peaks.to_csv(peaks_path, sep="\t", index=False, float_format="%.6e")
            logger.info("QTL peaks written to %s", peaks_path)
    except Exception as e:
        logger.warning("Peak pruning failed: %s", e)
        peaks = []

    # 3. Optional joint QTL fitting
    do_joint = getattr(args, "joint_qtl", False)
    if do_joint and peaks:
        try:
            from .config import resolve_device
            from .models.joint_qtl import fit_joint_qtl

            device = resolve_device(args.device)

            # Map peak SNP names to column indices in G
            first_result = next(iter(all_scan_results.values()))
            snp_to_idx = {s: i for i, s in enumerate(first_result.snp)}
            qtl_indices = []
            qtl_snps = []
            for pk in peaks:
                if pk.snp in snp_to_idx:
                    qtl_indices.append(snp_to_idx[pk.snp])
                    qtl_snps.append(pk.snp)

            if qtl_indices:
                # Use additive GRM for joint model
                from .linalg.kinship_polyploid import grm_polyploid_gene_action
                K, _ = grm_polyploid_gene_action(G, model="additive", ploidy=ploidy)

                joint_result = fit_joint_qtl(
                    Y.to(device), X0.to(device), G.to(device), K.to(device),
                    qtl_indices=qtl_indices,
                    qtl_snps=qtl_snps,
                    backward_elimination=True,
                    elimination_threshold=0.05,
                )
                logger.info(
                    "Joint QTL: %d QTL retained (%d eliminated), total R²=%.4f",
                    len(joint_result.qtl_snps), joint_result.n_eliminated,
                    joint_result.total_r_squared,
                )

                df_joint = pd.DataFrame({
                    "SNP": joint_result.qtl_snps,
                    "BETA": joint_result.beta.cpu().numpy(),
                    "SE": joint_result.se.cpu().numpy(),
                    "R_SQUARED": joint_result.r_squared.cpu().numpy(),
                    "LRT_P": joint_result.lrt_p.cpu().numpy(),
                })
                joint_path = f"{output_base}_joint_qtl.tsv"
                df_joint.to_csv(joint_path, sep="\t", index=False, float_format="%.6e")
                logger.info("Joint QTL results written to %s", joint_path)
        except Exception as e:
            logger.warning("Joint QTL fitting failed: %s", e)


def _cmd_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline: impute -> model selection -> scan -> correct.

    Orchestrates: data loading → optional imputation → GRM (if needed) →
    null fit → scan → multiple-testing correction → output.

    Models that support streaming (glm, lmm, mvlmm) never materialize
    the full genotype matrix.  FarmCPU/BLINK need full G in memory.
    """
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .preprocess.qc import QCFilterConfig

    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    ploidy = getattr(args, "ploidy", None) or 2
    config = TorchGWASConfig(
        device=device,
        chunk_size=args.chunk_size,
    )
    if getattr(args, "max_iter", None) is not None:
        config.numerical.reml_max_iter = args.max_iter

    # --- Optional imputation note ---
    impute_method = getattr(args, "impute", None)
    if impute_method and impute_method != "mean":
        logger.warning(
            "External imputation method '%s' requires pre-processing. "
            "Use `torchgwas impute` first, then run pipeline with pre-imputed data. "
            "Falling back to mean imputation.",
            impute_method,
        )

    # --- Model selection ---
    model_name = args.model
    logger.info("Pipeline: model=%s, ploidy=%d, device=%s", model_name, ploidy, device)

    # Streaming-capable models: GLM, LMM, mvLMM
    if model_name in ("glm", "lmm", "mvlmm"):
        Y, X0, aligned_reader = _align_samples(args, config)

        if model_name == "glm":
            from .models.glm import GLM
            model = GLM()
            null_fit = model.fit_null(Y, X0)

        elif model_name == "lmm":
            from .linalg.kinship import grm_vanraden_streaming
            from .models.single_trait_lmm import SingleTraitLMM
            logger.info("Computing kinship matrix (VanRaden streaming, ploidy=%d)...", ploidy)
            K, grm_meta = grm_vanraden_streaming(
                _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
                n_samples=aligned_reader.n_samples,
                ploidy=ploidy,
                device=device,
            )
            logger.info("GRM: %d samples, %d SNPs used",
                        grm_meta.n_samples, grm_meta.n_snps_used)
            model = SingleTraitLMM()
            null_fit = model.fit_null(Y.to(device), X0.to(device), K=K.to(device))
            logger.info(
                "Null fit: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
                null_fit.sig2_g, null_fit.sig2_e,
                null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
            )

        elif model_name == "mvlmm":
            from .linalg.kinship import grm_vanraden_streaming
            from .models.multi_trait_lmm import MultiTraitLMM
            if Y.shape[1] < 2:
                logger.error("mvlmm requires >= 2 traits, got %d", Y.shape[1])
                return 1
            logger.info("Computing kinship matrix (VanRaden streaming)...")
            K, grm_meta = grm_vanraden_streaming(
                _impute_chunk_iter(aligned_reader.iter_chunks(config.chunk_size)),
                n_samples=aligned_reader.n_samples,
                device=device,
            )
            logger.info("GRM: %d samples, %d SNPs used",
                        grm_meta.n_samples, grm_meta.n_snps_used)
            model = MultiTraitLMM()
            null_fit = model.fit_null(Y.to(device), X0.to(device), K=K.to(device))

        from .scan.unified import UnifiedScanner
        scanner = UnifiedScanner(aligned_reader, model, config)
        qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
        result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    # Models that need full G: FarmCPU, BLINK
    elif model_name in ("farmcpu", "blink"):
        G, Y, X0, vmeta, reader = _load_scan_data(args, config)

        if model_name == "farmcpu":
            from .models.farmcpu import FarmCPU
            model = FarmCPU()
            null_fit = model.fit_null(
                Y.to(device), X0.to(device),
                G=G.to(device), variant_meta=vmeta,
            )
        else:
            from .models.blink import BLINK
            model = BLINK()
            null_fit = model.fit_null(
                Y.to(device), X0.to(device),
                G=G.to(device), variant_meta=vmeta,
            )

        from .scan.unified import UnifiedScanner
        scanner = UnifiedScanner(reader, model, config)
        qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
        result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    # Multi-kernel LMM
    elif model_name == "mklmm":
        from .models.multi_kernel_lmm import MultiKernelLMM, build_multi_kernels
        G, Y, X0, vmeta, reader = _load_scan_data(args, config)
        kernels, kernel_names = build_multi_kernels(G.to(device), ploidy=ploidy)
        model = MultiKernelLMM()
        null_fit = model.fit_null(
            Y.to(device), X0.to(device),
            kernels=kernels, kernel_names=kernel_names,
        )
        from .scan.unified import UnifiedScanner
        scanner = UnifiedScanner(reader, model, config)
        qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
        result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    # GxE LMM (single-trait HetLMM via pipeline)
    elif model_name == "gxe":
        import pandas as pd

        from .linalg.kinship import grm_vanraden
        from .models.lmm_gxe import HetLMM
        G, Y, X0, vmeta, reader = _load_scan_data(args, config)
        K, _ = grm_vanraden(G.to(device))
        # GxE requires an environment covariate — use first covariate column
        env = torch.zeros(Y.shape[0], dtype=STAT_DTYPE, device=device)
        logger.warning(
            "Pipeline GxE uses a placeholder environment variable. "
            "For full control, use `torchgwas gxe-scan --env <file>`."
        )
        model = HetLMM()
        null_fit = model.fit_null(Y.to(device), X0.to(device), K=K, env=env)
        from .scan.unified import UnifiedScanner
        scanner = UnifiedScanner(reader, model, config)
        qc = QCFilterConfig(maf_min=args.maf_min, miss_max=args.miss_max)
        result = scanner.scan(null_fit, test=args.test, qc_config=qc)

    # Multi-environment trial (MET)
    elif model_name == "met":
        import pandas as pd

        from .linalg.kinship import grm_vanraden
        from .models.multi_env_lmm import MultiEnvLMM
        G, Y, X0, vmeta, reader = _load_scan_data(args, config)

        pheno_df = pd.read_csv(args.phenotype, sep="\t")
        env_cols = [c for c in pheno_df.columns if c not in ("FID", "IID", "SAMPLE")]
        if getattr(args, "env_cols", None):
            env_cols = args.env_cols.split(",")
        if len(env_cols) < 2:
            logger.error("MET requires >= 2 environment columns, found: %s", env_cols)
            return 1

        Y_wide = torch.tensor(pheno_df[env_cols].values, dtype=STAT_DTYPE, device=device)
        K, _ = grm_vanraden(G.to(device), ploidy=ploidy)

        parameterization = getattr(args, "parameterization", "per_env")
        vg_structure = getattr(args, "vg_structure", "unstructured")
        model = MultiEnvLMM(
            config=config.numerical,
            parameterization=parameterization,
            vg_structure=vg_structure,
        )
        null_fit = model.fit_null(Y_wide, X0.to(device), K, env_names=env_cols)

        results = []
        for chunk_idx, (G_c, vm) in enumerate(reader.iter_chunks(config.chunk_size)):
            G_c = G_c.to(device)
            res = model.score_chunk(G_c, null_fit, vm)
            results.append(res)

        merged = _merge_env_results(results)
        _save_met_results(merged, env_cols, args)
        return 0

    else:
        logger.error("Unknown model: %s", model_name)
        return 1

    _apply_correction_and_save(result, args)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Convert between genotype formats."""
    from .io.convert import convert

    convert(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
        sample_path=args.sample,
        map_path=args.map,
    )
    logger.info("Conversion complete: %s -> %s (%s)", args.input, args.output, args.format)
    return 0


def _cmd_impute(args: argparse.Namespace) -> int:
    """Impute missing genotypes."""
    method = args.method
    ploidy = getattr(args, "ploidy", 2) or 2

    # --- GPU-accelerated built-in methods ---
    if method in ("li-stephens", "deep-learning"):
        import torch

        from .preprocess.impute_gpu import impute_deep_learning, impute_li_stephens

        G = _load_genotype_matrix(args.genotype)
        logger.info("Loaded %d samples x %d markers, method=%s",
                     G.shape[0], G.shape[1], method)

        if method == "li-stephens":
            G_imp, r2 = impute_li_stephens(G, ploidy=ploidy)
        else:
            G_imp, r2 = impute_deep_learning(G, ploidy=ploidy)

        torch.save({"dosage": G_imp, "dosage_r2": r2}, args.output)
        logger.info("Imputed genotypes saved to %s", args.output)
        return 0

    # --- Simple built-in methods ---
    if method in ("mean", "mode", "knn", "ld"):
        import torch

        from .preprocess.impute import impute_knn, impute_ld, impute_mean, impute_mode

        G = _load_genotype_matrix(args.genotype)
        logger.info("Loaded %d samples x %d markers, method=%s",
                     G.shape[0], G.shape[1], method)

        if method == "mean":
            G_imp = impute_mean(G)
        elif method == "mode":
            G_imp = impute_mode(G)
        elif method == "knn":
            from .linalg.kinship import grm_vanraden
            K, _ = grm_vanraden(G, ploidy=ploidy)
            G_imp = impute_knn(G, K)
        else:  # ld
            G_imp = impute_ld(G)

        torch.save({"dosage": G_imp}, args.output)
        logger.info("Imputed genotypes saved to %s", args.output)
        return 0

    # --- External tool wrappers ---
    if method == "beagle":
        from .preprocess.impute_external import run_beagle
        result = run_beagle(
            args.genotype, args.output,
            ref_panel=getattr(args, "ref_panel", None),
            ploidy=ploidy,
        )
        logger.info("BEAGLE: %d variants imputed, mean Rsq=%.3f",
                     result.n_variants_imputed, result.mean_rsq)
        return 0

    if method == "impute5":
        from .preprocess.impute_external import run_impute5
        ref_panel = getattr(args, "ref_panel", None)
        if ref_panel is None:
            raise ValueError("--ref-panel is required for impute5")
        result = run_impute5(args.genotype, args.output, ref_panel=ref_panel)
        logger.info("IMPUTE5: %d variants imputed, mean Rsq=%.3f",
                     result.n_variants_imputed, result.mean_rsq)
        return 0

    if method == "minimac4":
        from .preprocess.impute_external import run_minimac4
        ref_panel = getattr(args, "ref_panel", None)
        if ref_panel is None:
            raise ValueError("--ref-panel is required for minimac4")
        result = run_minimac4(args.genotype, args.output, ref_panel=ref_panel)
        logger.info("Minimac4: %d variants imputed, mean Rsq=%.3f",
                     result.n_variants_imputed, result.mean_rsq)
        return 0

    raise NotImplementedError(f"impute CLI for method={method} not yet implemented")


def _cmd_dosage_call(args: argparse.Namespace) -> int:
    """Phase 55 dosage-call subcommand handler."""
    from .preprocess.dosage_call import run_updog

    result = run_updog(
        input_vcf=args.vcf,
        output_path=args.output,
        ploidy=args.ploidy,
        model=args.model,
        rscript=args.rscript,
        bias=args.bias,
        od=args.od,
        seq_error=args.seq_error,
        n_cores=args.n_cores,
        keep_tmpdir=args.keep_tmpdir,
    )
    logger.info(
        "dosage-call: %d samples × %d variants, ploidy=%d, n_missing=%d, "
        "tool_version=%s",
        len(result.sample_ids), len(result.variant_ids),
        result.ploidy, result.n_missing, result.tool_version,
    )
    return 0


def _cmd_phase_poly(args: argparse.Namespace) -> int:
    """Phase 56 phase-poly subcommand handler — polyploid phasing via PolyOrigin."""
    import json as _json

    import torch as _torch

    from .preprocess.phase_polyorigin import run_polyorigin

    # Map CLI's None sentinel for --auto-install to False (non-interactive default)
    auto = args.auto_install if args.auto_install is not None else False

    # Load probs — either a raw (n,m,k+1) tensor or a dict with sample/variant IDs.
    # Phase 55 dosage-call writes a raw tensor to <prefix>.probs.pt and puts
    # sample_ids/variant_ids in the sibling <prefix>.meta.json, so we handle
    # both the dict form (future-proofing) and the raw-tensor-plus-sidecar form.
    probs_path = Path(args.probs)
    probs_obj = _torch.load(str(probs_path), weights_only=False)

    if isinstance(probs_obj, dict):
        # Dict form: all metadata is embedded
        probs = probs_obj["probs"]
        sample_ids = probs_obj.get("sample_ids")
        variant_ids = probs_obj.get("variant_ids")
    else:
        # Raw tensor: look for the sibling .meta.json written by dosage-call
        probs = probs_obj
        sample_ids = None
        variant_ids = None
        # Derive meta.json path: strip the trailing .pt, then look for .meta.json.
        # Handles both "<prefix>.probs.pt" → "<prefix>.meta.json" and
        # "<prefix>.pt" → "<prefix>.meta.json".
        meta_candidate = probs_path.with_suffix("").with_suffix(".meta.json")
        if not meta_candidate.is_file():
            # Try stripping just one suffix (e.g. "foo.probs.pt" → "foo.probs.meta.json"
            # is wrong; try "foo.meta.json" by stripping ".probs" then ".pt")
            stem = probs_path.stem  # e.g. "dcall.probs"
            if stem.endswith(".probs"):
                stem = stem[: -len(".probs")]
            meta_candidate = probs_path.parent / (stem + ".meta.json")
        if meta_candidate.is_file():
            meta = _json.loads(meta_candidate.read_text())
            sample_ids = meta.get("sample_ids")
            variant_ids = meta.get("variant_ids")

    if sample_ids is None or variant_ids is None:
        raise SystemExit(
            "--probs must be a dict with 'sample_ids' and 'variant_ids' keys, "
            "or a raw tensor produced by 'torchgwas dosage-call' with a sibling "
            "<prefix>.meta.json. "
            "Could not find sample/variant IDs in the probs file or its sidecar."
        )

    result = run_polyorigin(
        probs=probs,
        pedigree_tsv=args.pedigree,
        map_tsv=args.map,
        output_path=args.output,
        ploidy=args.ploidy,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        parent_phased_csv=args.parent_phased,
        julia_path=args.julia_path,
        auto_install_julia=auto,
        refinemap=args.refinemap,
        recomrate=args.recomrate,
        keep_workdir=args.keep_workdir,
    )
    logger.info(
        "phase-poly: %d offspring, %d parents, %d variants, tool_version=%s",
        len(result.offspring_ids), len(result.parent_ids),
        len(result.variant_ids), result.tool_version,
    )
    return 0


def _cmd_ld_blocks(args: argparse.Namespace) -> int:
    """Detect haplotype blocks."""
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader
    from .ld import detect_blocks, save_blocks_bed

    device = torch.device(args.device) if args.device else None

    # Load genotype matrix and variant metadata
    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    chunks_dosage = []
    variant_pos = []
    variant_chr = []
    variant_ids = []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks_dosage.append(G_chunk)
        variant_pos.extend(vmeta.pos)
        variant_chr.extend(vmeta.chr)
        variant_ids.extend(vmeta.snp)

    G = torch.cat(chunks_dosage, dim=1)
    logger.info("Loaded %d samples x %d markers", G.shape[0], G.shape[1])

    # Optionally load phased haplotypes
    haplotypes = None
    if args.phased_vcf:
        from .preprocess.phase import load_haplotypes
        haplotypes = load_haplotypes(args.phased_vcf)
        logger.info("Loaded phased haplotypes: %s", haplotypes.shape)

    # Build method-specific kwargs
    method_kwargs = {}
    method = args.method
    if method == "gabriel":
        method_kwargs["ci_low"] = args.ci_low
        method_kwargs["ci_high"] = args.ci_high
    elif method == "four_gamete":
        method_kwargs["freq_threshold"] = args.freq_threshold
    elif method == "spine":
        method_kwargs["d_prime_threshold"] = args.dprime_threshold
    elif method == "r2":
        method_kwargs["r2_threshold"] = args.r2_threshold
    elif method == "gwas_aligned":
        method_kwargs["condition_penalty"] = args.condition_penalty
        method_kwargs["max_block_snps"] = args.max_block_snps
    elif method == "graphical":
        method_kwargs["l1_penalty"] = args.l1_penalty
        method_kwargs["window_size"] = args.window_size
    elif method == "changepoint":
        method_kwargs["penalty"] = args.cp_penalty
    elif method == "big_ld":
        method_kwargs["r2_threshold"] = args.r2_threshold
        method_kwargs["window_size"] = args.window_size
    elif method == "cc_graph":
        method_kwargs["r2_threshold"] = args.r2_threshold
        method_kwargs["window"] = args.ld_window
        method_kwargs["include_singletons"] = args.include_singletons
    elif method == "dp_optimize":
        method_kwargs["objective"] = args.objective
        method_kwargs["max_block_snps"] = args.max_block_snps

    blocks = detect_blocks(
        G, variant_pos, variant_chr, variant_ids,
        method=method,
        haplotypes=haplotypes,
        max_kb=args.max_kb,
        device=device,
        **method_kwargs,
    )

    # Save output in all standard formats
    from .ld import save_blocks_det, save_blocks_summary

    bed_path = f"{args.output}.bed"
    det_path = f"{args.output}.blocks.det"
    summary_path = f"{args.output}.summary.txt"

    save_blocks_bed(blocks, bed_path)
    save_blocks_det(blocks, det_path, variant_ids=variant_ids)
    save_blocks_summary(blocks, summary_path)
    logger.info(
        "Detected %d blocks (%s method) -> %s, %s, %s",
        len(blocks), args.method, bed_path, det_path, summary_path,
    )

    # Optional: compare against PLINK reference blocks
    if hasattr(args, "compare_plink") and args.compare_plink:
        from .ld import (
            compare_blocks,
            load_plink_blocks_det,
            plink_blocks_to_ldblocks,
        )

        plink_blocks = load_plink_blocks_det(args.compare_plink)
        plink_ld = plink_blocks_to_ldblocks(plink_blocks, variant_ids=variant_ids)
        result = compare_blocks(blocks, plink_ld, method_a=method, method_b="plink")

        comp_path = f"{args.output}.comparison.txt"
        with open(comp_path, "w") as f:
            f.write(f"Block Comparison: {method} vs PLINK\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Blocks ({method}):      {result.n_blocks_a}\n")
            f.write(f"Blocks (PLINK):       {result.n_blocks_b}\n")
            f.write(f"Jaccard index:        {result.jaccard_index:.4f}\n")
            f.write(f"Dice coefficient:     {result.dice_coefficient:.4f}\n")
            f.write(f"Overlap coefficient:  {result.overlap_coefficient:.4f}\n")
            f.write(f"Match rate ({method}): {result.match_rate_a:.4f}\n")
            f.write(f"Match rate (PLINK):   {result.match_rate_b:.4f}\n")
            f.write(f"Mean boundary dist:   {result.mean_boundary_dist:.1f} bp\n")
            f.write(f"Median boundary dist: {result.median_boundary_dist:.1f} bp\n")
        logger.info("Comparison with PLINK -> %s", comp_path)

    return 0


def _cmd_ldsc(args: argparse.Namespace) -> int:
    """Estimate SNP heritability via LDSC."""
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader
    from .postgwas import compute_ld_scores, ldsc_h2, load_sumstats

    ss = load_sumstats(args.sumstats)
    n = int(ss.n[~ss.n.isnan()].median().item()) if not ss.n.isnan().all() else args.n

    # Load genotypes for LD score computation
    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    chunks, var_pos, var_chr = [], [], []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks.append(G_chunk)
        var_pos.extend(vmeta.pos)
        var_chr.extend(vmeta.chr)
    G = torch.cat(chunks, dim=1)

    ld_scores = compute_ld_scores(G, var_pos, var_chr, window_kb=args.window_kb)
    result = ldsc_h2(ss.chi2, ld_scores, n, ss.m)

    out_path = f"{args.output}.ldsc.txt"
    with open(out_path, "w") as f:
        f.write(f"h2:            {result.h2:.4f} ({result.h2_se:.4f})\n")
        f.write(f"Intercept:     {result.intercept:.4f} ({result.intercept_se:.4f})\n")
        f.write(f"Mean chi2:     {result.mean_chi2:.4f}\n")
        f.write(f"Lambda GC:     {result.lambda_gc:.4f}\n")
        f.write(f"N SNPs:        {result.n_snps}\n")
    logger.info("LDSC h2 -> %s", out_path)
    print(f"h2 = {result.h2:.4f} (SE = {result.h2_se:.4f})")
    return 0


def _cmd_ldsc_rg(args: argparse.Namespace) -> int:
    """Estimate genetic correlation via cross-trait LDSC."""
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader
    from .postgwas import align_sumstats, compute_ld_scores, ldsc_rg_from_z, load_sumstats

    ss1 = load_sumstats(args.sumstats1)
    ss2 = load_sumstats(args.sumstats2)
    ss1, ss2 = align_sumstats([ss1, ss2])

    n1 = int(ss1.n[~ss1.n.isnan()].median().item()) if not ss1.n.isnan().all() else args.n1
    n2 = int(ss2.n[~ss2.n.isnan()].median().item()) if not ss2.n.isnan().all() else args.n2

    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    chunks, var_pos, var_chr = [], [], []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks.append(G_chunk)
        var_pos.extend(vmeta.pos)
        var_chr.extend(vmeta.chr)
    G = torch.cat(chunks, dim=1)

    ld_scores = compute_ld_scores(G, var_pos, var_chr, window_kb=args.window_kb)
    result = ldsc_rg_from_z(ss1.z, ss2.z, ld_scores, n1, n2, ss1.m)

    out_path = f"{args.output}.ldsc_rg.txt"
    with open(out_path, "w") as f:
        f.write(f"rg:          {result.rg:.4f} ({result.rg_se:.4f})\n")
        f.write(f"cov_g:       {result.cov_g:.6f} ({result.cov_g_se:.6f})\n")
        f.write(f"h2_1:        {result.h2_1.h2:.4f} ({result.h2_1.h2_se:.4f})\n")
        f.write(f"h2_2:        {result.h2_2.h2:.4f} ({result.h2_2.h2_se:.4f})\n")
    logger.info("LDSC rg -> %s", out_path)
    print(f"rg = {result.rg:.4f} (SE = {result.rg_se:.4f})")
    return 0


def _cmd_meta(args: argparse.Namespace) -> int:
    """Run meta-analysis across multiple GWAS results."""
    import torch

    from .postgwas import (
        align_sumstats,
        load_sumstats,
        meta_fixed_effect,
        meta_han_eskin,
        meta_random_effect,
        meta_sample_size,
    )

    ss_list = [load_sumstats(p) for p in args.input]
    ss_list = align_sumstats(ss_list)
    K = len(ss_list)

    method = args.method

    if method in ("fixed", "random", "han_eskin"):
        beta = torch.stack([ss.beta for ss in ss_list], dim=1)
        se = torch.stack([ss.se for ss in ss_list], dim=1)

        if method == "fixed":
            result = meta_fixed_effect(beta, se)
        elif method == "random":
            result = meta_random_effect(beta, se)
        else:
            result = meta_han_eskin(beta, se)
    elif method == "stouffer":
        p = torch.stack([ss.p for ss in ss_list], dim=1)
        n = torch.tensor([
            ss.n[~ss.n.isnan()].median().item() if not ss.n.isnan().all() else 1000.0
            for ss in ss_list
        ], dtype=torch.float64)
        direction = torch.stack([torch.sign(ss.beta) for ss in ss_list], dim=1)
        result = meta_sample_size(p, n, direction)
    else:
        raise ValueError(f"Unknown meta-analysis method: {method}")

    # Write output
    ref = ss_list[0]
    out_path = f"{args.output}.meta.tsv"
    with open(out_path, "w") as f:
        f.write("chr\tpos\tsnp\ta1\ta2\tbeta_meta\tse_meta\tp_meta\tz_meta\tq_stat\ti2\n")
        for j in range(ref.m):
            f.write(f"{ref.chr[j]}\t{ref.pos[j]}\t{ref.snp[j]}\t{ref.a1[j]}\t{ref.a2[j]}\t"
                    f"{result.beta_meta[j].item():.6f}\t{result.se_meta[j].item():.6f}\t"
                    f"{result.p_meta[j].item():.6e}\t{result.z_meta[j].item():.4f}\t"
                    f"{result.q_stat[j].item():.4f}\t{result.i2[j].item():.4f}\n")
    logger.info("Meta-analysis (%s, K=%d) -> %s", method, K, out_path)
    return 0


def _cmd_clump(args: argparse.Namespace) -> int:
    """LD clumping to identify independent loci."""
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader
    from .postgwas import ld_clump, load_sumstats

    ss = load_sumstats(args.sumstats)

    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    chunks, var_pos, var_chr = [], [], []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks.append(G_chunk)
        var_pos.extend(vmeta.pos)
        var_chr.extend(vmeta.chr)
    G = torch.cat(chunks, dim=1)

    result = ld_clump(
        ss.p, G, var_pos, var_chr,
        r2_threshold=args.r2,
        p_threshold=args.p_threshold,
        window_kb=args.window_kb,
    )

    out_path = f"{args.output}.clumps.tsv"
    with open(out_path, "w") as f:
        f.write("index_snp\tchr\tpos\tp\tn_clumped\n")
        for i, idx in enumerate(result.index_snps):
            f.write(f"{ss.snp[idx]}\t{ss.chr[idx]}\t{ss.pos[idx]}\t"
                    f"{result.index_p[i].item():.6e}\t{len(result.clump_members[i])}\n")
    logger.info("LD clumping: %d index SNPs -> %s", result.n_clumps, out_path)
    print(f"{result.n_clumps} independent loci identified")
    return 0


def _cmd_pgs_fit(args: argparse.Namespace) -> int:
    """Fit PGS weights from GWAS sumstats."""

    from .pgs import (
        PRSCS,
        ClumpingThresholding,
        LDpred2Auto,
        LDpred2Grid,
        LDpred2Inf,
        load_ld_reference,
        load_pgs_sumstats,
    )

    device = args.device
    ss = load_pgs_sumstats(args.sumstats, device=device)
    ld = load_ld_reference(args.ld_ref, device=device)

    method = args.method
    if method == "ct":
        m = ClumpingThresholding(device=device, seed=args.seed)
        result = m.fit(
            ss, ld,
            p_threshold=args.clump_p,
            r2_threshold=args.clump_r2,
            window_kb=args.clump_kb,
        )
    elif method == "ldpred2-inf":
        m = LDpred2Inf(device=device, seed=args.seed)
        result = m.fit(ss, ld, h2=args.h2)
    elif method == "ldpred2-grid":
        grid_p = [float(x) for x in args.grid_p.split(",")]
        grid_h2 = [float(x) for x in args.grid_h2.split(",")]
        m = LDpred2Grid(device=device, seed=args.seed)
        result = m.fit(
            ss, ld,
            grid_p=grid_p, grid_h2=grid_h2, sparse=args.grid_sparse,
            n_iter=args.n_iter, n_burnin=args.n_burnin,
        )
    elif method == "ldpred2-auto":
        m = LDpred2Auto(device=device, seed=args.seed)
        result = m.fit(
            ss, ld,
            h2_init=args.h2, p_init=args.p_causal,
            n_iter=args.n_iter, n_burnin=args.n_burnin, n_chains=args.n_chains,
        )
    elif method == "prscs":
        m = PRSCS(device=device, seed=args.seed)
        result = m.fit(
            ss, ld,
            phi=args.phi, n_iter=args.n_iter, n_burnin=args.n_burnin,
            n_chains=args.n_chains,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown PGS method: {method}")

    result.save(args.output)
    logger.info(
        "PGS fit (%s): %d SNPs weighted -> %s",
        method, result.m, args.output,
    )
    print(
        f"{method}: {result.m} SNPs | h2={result.h2} | p_causal={result.p_causal} | "
        f"converged={result.converged}"
    )
    return 0


def _cmd_pgs_score(args: argparse.Namespace) -> int:
    """Apply PGS weights to target genotypes."""
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader
    from .pgs import PGSResult, score_individuals

    result = PGSResult.load(args.weights)

    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    chunks = []
    var_snp: list[str] = []
    var_a1: list[str] = []
    var_a2: list[str] = []
    for G_chunk, vmeta in reader.iter_chunks():
        chunks.append(G_chunk)
        var_snp.extend(vmeta.snp)
        var_a1.extend(vmeta.a1)
        var_a2.extend(vmeta.a2)
    G = torch.cat(chunks, dim=1).to(args.device)

    sample_ids = getattr(reader, "sample_ids", None)
    if sample_ids is None:
        sample_ids = [f"IND{i}" for i in range(G.shape[0])]

    score_res = score_individuals(
        G, var_snp, var_a1, var_a2, result,
        standardize=args.standardize,
        handle_missing=args.handle_missing,
        chunk_size=args.chunk_size,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("FID\tIID\tPGS\n")
        for i, iid in enumerate(sample_ids):
            f.write(f"{iid}\t{iid}\t{float(score_res.pgs[i].item()):.6e}\n")
    logger.info(
        "PGS scoring: used=%d missing=%d flipped=%d -> %s",
        score_res.n_snp_used, score_res.n_snp_missing, score_res.n_snp_flipped,
        args.output,
    )
    print(
        f"Used {score_res.n_snp_used} SNPs ({score_res.n_snp_missing} missing, "
        f"{score_res.n_snp_flipped} flipped) -> {args.output}"
    )
    return 0


def _load_genotype_matrix(genotype_path: str) -> torch.Tensor:
    """Load full genotype matrix from any supported format.

    All TorchGWAS readers in :mod:`torchgwas.io` yield ``(G_chunk, vmeta)``
    tuples from ``iter_chunks()`` (not a chunk object with a ``.dosage``
    attribute). This helper accepts the canonical tuple shape used by
    every other call-site in the module.
    """
    import torch

    from .io.detect import detect_format
    from .io.validate import _open_reader

    fmt = detect_format(genotype_path)
    reader = _open_reader(genotype_path, fmt)
    chunks = []
    for chunk in reader.iter_chunks():
        # Readers may yield (G, vmeta) tuples, or for legacy paths a chunk
        # object with a `.dosage` attribute. Accept either.
        if isinstance(chunk, tuple):
            chunks.append(chunk[0])
        else:
            chunks.append(chunk.dosage)
    return torch.cat(chunks, dim=1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _align_samples(args: argparse.Namespace, config, trait_columns=None):
    """Align phenotype/covariates with genotype reader; return streaming reader.

    Returns (Y, X0, aligned_reader) where aligned_reader yields chunks with
    rows already reindexed to the aligned sample order.  The full genotype
    matrix is NOT loaded into memory.

    Parameters
    ----------
    trait_columns : list[str], optional
        If provided, select specific trait columns from phenotype file
        (used by mvlmm-scan for multi-trait selection).
    """
    from .io.aligned import SampleAlignedReader
    from .io.detect import detect_format
    from .io.phenotype import load_phenotype
    from .io.validate import _open_reader

    # Detect format and open reader
    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)
    logger.info("Format: %s, %d samples, %d variants",
                fmt, reader.n_samples, reader.n_variants)

    # Parse --traits into trait_columns if not already provided
    if trait_columns is None:
        traits_str = getattr(args, "traits", None)
        if traits_str is not None:
            trait_columns = [t.strip() for t in traits_str.split(",")]

    # Align phenotype/covariates to genotype samples
    id_column = getattr(args, "id_column", None)
    pheno_data = load_phenotype(
        args.phenotype,
        genotype_sample_ids=reader.sample_ids,
        covariate_path=getattr(args, "covariate", None),
        trait_columns=trait_columns,
        id_column=id_column,
    )

    Y = pheno_data.Y
    X0 = pheno_data.X0
    aligned_ids = pheno_data.sample_ids

    # Build sample-alignment index
    geno_id_to_idx = {s: i for i, s in enumerate(reader.sample_ids)}
    geno_idx = [geno_id_to_idx[s] for s in aligned_ids]

    aligned_reader = SampleAlignedReader(reader, geno_idx, aligned_ids)

    logger.info("Data aligned: %d samples, %d variants, %d traits (%s)",
                aligned_reader.n_samples, aligned_reader.n_variants, Y.shape[1],
                ", ".join(pheno_data.trait_names))

    return Y, X0, aligned_reader


def _load_user_grm(grm_path: str, n_samples: int) -> torch.Tensor:
    """Load a user-provided GRM matrix with validation.

    Supports NumPy (.npy, .npz), space/tab-delimited text, and CSV.
    Performs QC: symmetry, PSD check, dimension match.

    Raises
    ------
    ValueError
        If the matrix fails any validation check.
    FileNotFoundError
        If the file does not exist.
    """
    from pathlib import Path

    import numpy as np
    import torch

    p = Path(grm_path)
    if not p.is_file():
        raise FileNotFoundError(f"GRM file not found: {p}")

    ext = p.suffix.lower()
    if ext == ".npy":
        K_np = np.load(str(p))
    elif ext == ".npz":
        data = np.load(str(p))
        # Take the first array in the archive
        keys = list(data.keys())
        K_np = data[keys[0]]
        logger.info("Loaded GRM from .npz key '%s'", keys[0])
    elif ext in {".csv", ".tsv"}:
        import pandas as pd
        sep = "," if ext == ".csv" else "\t"
        K_np = pd.read_csv(str(p), sep=sep, header=None).values
    else:
        # Try space/tab-delimited text
        K_np = np.loadtxt(str(p))

    K = torch.tensor(K_np, dtype=torch.float64)

    # --- QC checks ---
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            f"GRM must be a square matrix, got shape {tuple(K.shape)}."
        )

    if K.shape[0] != n_samples:
        raise ValueError(
            f"GRM dimension ({K.shape[0]}) does not match the number of aligned "
            f"samples ({n_samples}). Ensure the GRM was computed on the same "
            f"samples as the genotype/phenotype files."
        )

    # Symmetry check (allow small numerical tolerance)
    asym = (K - K.T).abs().max().item()
    if asym > 1e-6:
        raise ValueError(
            f"GRM is not symmetric (max asymmetry = {asym:.2e}). "
            "Ensure the matrix is a valid kinship/GRM."
        )
    # Force exact symmetry
    K = (K + K.T) / 2.0

    # Check for NaN/Inf
    if torch.isnan(K).any():
        raise ValueError("GRM contains NaN values.")
    if torch.isinf(K).any():
        raise ValueError("GRM contains Inf values.")

    # Positive semi-definiteness check (warn, don't fail — small negatives are common)
    min_eval = torch.linalg.eigvalsh(K)[0].item()
    if min_eval < -0.01:
        logger.warning(
            "GRM has negative eigenvalue (min = %.4f). This may indicate a "
            "non-PSD matrix. Eigenvalues will be floored to 0 during "
            "eigendecomposition.",
            min_eval,
        )
    elif min_eval < 0:
        logger.info(
            "GRM has small negative eigenvalue (%.2e) — normal for finite-precision GRM.",
            min_eval,
        )

    logger.info(
        "User GRM loaded: %dx%d, range [%.4f, %.4f], trace = %.4f",
        K.shape[0], K.shape[1], K.min().item(), K.max().item(), K.trace().item(),
    )
    return K


def _load_full_genotype(aligned_reader):
    """Materialize the full genotype matrix from a reader.

    Used only by models that need the full G in memory (FarmCPU, BLINK).
    For LMM/mvLMM/GLM, use the streaming path instead.
    """
    from .config import STAT_DTYPE
    from .preprocess.impute import impute_mean

    chunks = list(aligned_reader.iter_chunks(chunk_size=aligned_reader.n_variants))
    G_full, vmeta = chunks[0]
    G_full = G_full.to(STAT_DTYPE)
    G_full = impute_mean(G_full)
    return G_full, vmeta


def _load_scan_data(args: argparse.Namespace, config, trait_columns=None):
    """Load genotype, phenotype, covariates; align samples; return (G, Y, X0, vmeta, reader).

    Legacy helper that materializes the full genotype matrix. Used by models
    that need the full G (FarmCPU, BLINK, polyploid).  For streaming-capable
    models, prefer _align_samples + grm_vanraden_streaming.

    Parameters
    ----------
    trait_columns : list[str], optional
        If provided, select specific trait columns from phenotype file
        (used by mvlmm-scan for multi-trait selection).
    """
    Y, X0, aligned_reader = _align_samples(args, config, trait_columns)
    G, vmeta = _load_full_genotype(aligned_reader)

    logger.info("Data loaded: %d samples, %d variants, %d traits",
                G.shape[0], G.shape[1], Y.shape[1])

    return G, Y, X0, vmeta, aligned_reader


def _impute_chunk_iter(chunk_iter):
    """Wrap a chunk iterator to mean-impute each chunk on the fly.

    Used by the streaming GRM path so that allele frequency computation
    and centering see imputed data (NaN-free).
    """
    from .preprocess.impute import impute_mean

    for G_chunk, vmeta in chunk_iter:
        yield impute_mean(G_chunk), vmeta


def _apply_correction_and_save(result, args: argparse.Namespace) -> None:
    """Apply multiple testing correction and save results.

    Handles three families of result objects:
    1. Single-trait scalar p-value (`result.p` is 1-D, `result.beta/se/stat` 1-D).
    2. Multi-output / GxE results that expose `p_joint` (and per-component
       `p_main`, `p_interact`) — we treat `p_joint` as the canonical p for FDR
       and write per-component p columns alongside.
    3. Multi-trait results (mvLMM / me-GLMM) where `result.beta/se/stat` are
       2-D `(m, d)`. We collapse to per-trait columns named `BETA_t{i}` etc.
       and require `result.p` (or `result.p_joint`) to be 1-D for correction.
    """
    import pandas as pd

    from .stats.multipletesting import (
        benjamini_hochberg,
        benjamini_yekutieli,
        bonferroni,
        holm,
        storey_qvalue,
    )

    # Phase C1: prefer joint p when models expose it (GxE: p_main / p_interact /
    # p_joint; me-GLMM: p_joint). Falls back to result.p for plain scans.
    p_attr = getattr(result, "p", None)
    p_joint = getattr(result, "p_joint", None)
    if p_attr is None and p_joint is None:
        raise AttributeError(
            f"Scan result of type {type(result).__name__} exposes neither "
            "`p` nor `p_joint`; cannot write association table."
        )
    p = p_joint if p_attr is None else p_attr

    # Apply correction
    correction = args.correction
    if correction == "none":
        p_adj = p
    elif correction == "bonferroni":
        p_adj = bonferroni(p)
    elif correction == "holm":
        p_adj = holm(p)
    elif correction == "bh":
        p_adj = benjamini_hochberg(p)
    elif correction == "by":
        p_adj = benjamini_yekutieli(p)
    elif correction == "storey":
        p_adj = storey_qvalue(p)
    elif correction == "weighted-bh":
        from .stats.weighted_fdr import weighted_bh
        weights_file = getattr(args, "weights_file", None)
        if weights_file is None:
            logger.warning("weighted-bh requires --weights-file; falling back to BH")
            p_adj = benjamini_hochberg(p)
        else:
            weights = _load_weights(weights_file, result.snp)
            p_adj = weighted_bh(p, weights)
    elif correction == "lfdr":
        from .stats.weighted_fdr import local_fdr
        p_adj = local_fdr(p)
    elif correction == "hierarchical":
        from .stats.weighted_fdr import hierarchical_fdr
        gene_map_file = getattr(args, "gene_map", None)
        if gene_map_file is None:
            logger.warning("hierarchical requires --gene-map; falling back to BH")
            p_adj = benjamini_hochberg(p)
        else:
            group_ids = _load_gene_map(gene_map_file, result.snp)
            p_adj = hierarchical_fdr(p, group_ids)
    elif correction in ("ihw", "adapt"):
        from .stats.adaptive_fdr import adapt as _adapt
        from .stats.adaptive_fdr import ihw as _ihw
        cov = _load_fdr_covariate(args, result)
        if correction == "ihw":
            res_fdr = _ihw(p, cov, q=0.05)
        else:
            res_fdr = _adapt(p, cov, q=0.05)
        p_adj = res_fdr.adjusted_p
        logger.info("%s: %d rejections at q=0.05", correction.upper(),
                    res_fdr.n_rejections)
    else:
        logger.warning("Unknown correction '%s', using none", correction)
        p_adj = p

    # Build output DataFrame.
    # For multi-trait scans (mvlmm, me-glmm) the beta / se / stat tensors are
    # 2-D (m, d); flatten by writing one column per trait/output dimension
    # (BETA_d0, BETA_d1, …) so the DataFrame columns stay 1-D.
    df_data: dict = {
        "CHR": result.chr,
        "POS": result.pos,
        "SNP": result.snp,
        "A1": result.a1,
        "A2": result.a2,
        "AF": result.af.cpu().numpy(),
    }

    def _add_per_dim(name: str, t):
        arr = t.cpu().numpy()
        if arr.ndim == 1:
            df_data[name] = arr
        else:
            for d in range(arr.shape[1]):
                df_data[f"{name}_d{d}"] = arr[:, d]

    # GxE results expose {beta,se,stat}_main / _interact instead of plain
    # beta/se/stat. Detect either schema and emit columns accordingly.
    if hasattr(result, "beta"):
        _add_per_dim("BETA", result.beta)
        _add_per_dim("SE", result.se)
        _add_per_dim("STAT", result.stat)
    else:
        if hasattr(result, "beta_main"):
            _add_per_dim("BETA_MAIN", result.beta_main)
            _add_per_dim("SE_MAIN", result.se_main)
            _add_per_dim("STAT_MAIN", result.stat_main)
        if hasattr(result, "beta_interact"):
            _add_per_dim("BETA_INTERACT", result.beta_interact)
            _add_per_dim("SE_INTERACT", result.se_interact)
            _add_per_dim("STAT_INTERACT", result.stat_interact)
        if hasattr(result, "stat_joint"):
            _add_per_dim("STAT_JOINT", result.stat_joint)

    # GxE / interaction models expose per-component p columns alongside p_joint.
    p_main = getattr(result, "p_main", None)
    p_interact = getattr(result, "p_interact", None)
    if p_main is not None:
        df_data["P_MAIN"] = p_main.cpu().numpy()
    if p_interact is not None:
        df_data["P_INTERACT"] = p_interact.cpu().numpy()

    # Canonical p (joint p when present, else result.p)
    df_data["P"] = p.cpu().numpy()
    df_data["P_ADJ"] = p_adj.cpu().numpy()
    df = pd.DataFrame(df_data)

    # Save
    out_path = args.output
    tsv_path = f"{out_path}.assoc.tsv"
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6e")
    logger.info("Results written to %s (%d variants)", tsv_path, len(df))

    # Summary
    n_sig_raw = int((p < 5e-8).sum().item())
    n_sig_adj = int((p_adj < 0.05).sum().item())
    print(f"\nResults: {len(result)} variants tested")
    print(f"  Genome-wide significant (P < 5e-8): {n_sig_raw}")
    print(f"  Significant after {correction} (adj P < 0.05): {n_sig_adj}")
    print(f"  Output: {tsv_path}")

    if getattr(result, "inference_type", None) == "post_selection":
        print("  NOTE: P-values are post-selection (conditional on selected pseudo-QTNs)")


# ---- Subparser definitions ----

def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("validate", help="Pre-flight dataset validation")
    p.add_argument("--genotype", required=True)
    p.add_argument("--phenotype", required=True)
    p.add_argument("--covariate")
    p.add_argument("--report", help="Path for JSON report output")


def _add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("convert", help="Convert between genotype formats")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", required=True, choices=["bed", "vcf", "zarr"])
    p.add_argument("--sample")
    p.add_argument("--map")


def _add_impute_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("impute", help="Impute missing genotypes")
    p.add_argument("--genotype", required=True)
    p.add_argument("--method", required=True, choices=["mean", "mode", "knn", "ld", "li-stephens", "deep-learning", "beagle", "impute5", "minimac4"])
    p.add_argument("--output", required=True)
    p.add_argument("--ploidy", type=int, default=2, help="Ploidy level (default: 2)")
    p.add_argument("--ref-panel")


def _add_dosage_call_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "dosage-call",
        help="Call polyploid allele dosages from VCF read counts via updog",
    )
    p.add_argument("--vcf", required=True,
                   help="Input VCF with AD format field (biallelic)")
    p.add_argument("--output", required=True,
                   help="Output prefix for .probs.pt / .meta.json / .snp_diag.tsv")
    p.add_argument("--ploidy", type=int, required=True,
                   help="Organism ploidy (2..8)")
    p.add_argument("--model", default="norm",
                   help="updog flexdog model name (default: norm)")
    p.add_argument("--rscript", default=None,
                   help="Path to Rscript (default: PATH lookup)")
    bias_group = p.add_mutually_exclusive_group()
    bias_group.add_argument("--bias", dest="bias", action="store_true",
                            default=True, help="Estimate allele bias (default)")
    bias_group.add_argument("--no-bias", dest="bias", action="store_false",
                            help="Fix allele bias to 1")
    od_group = p.add_mutually_exclusive_group()
    od_group.add_argument("--od", dest="od", action="store_true",
                          default=True,
                          help="Estimate overdispersion (default)")
    od_group.add_argument("--no-od", dest="od", action="store_false",
                          help="Fix overdispersion to 0")
    p.add_argument("--seq-error", type=float, default=None,
                   help="Fix sequencing error rate (default: estimate)")
    p.add_argument("--n-cores", type=int, default=1,
                   help="Parallelism passed to updog::multidog (default: 1)")
    p.add_argument("--keep-tmpdir", action="store_true",
                   help="Skip tempdir cleanup (debug aid)")


def _add_phase_poly_parser(subparsers: argparse._SubParsersAction) -> None:
    """Phase 56: polyploid F1 phasing via PolyOrigin."""
    p = subparsers.add_parser(
        "phase-poly",
        help="Polyploid F1 phasing via PolyOrigin (Phase 56)",
    )
    p.add_argument(
        "--probs", required=True,
        help=(
            "Path to .probs.pt from 'dosage-call' (raw (n,m,k+1) tensor or dict). "
            "If a raw tensor, the sibling <prefix>.meta.json is read for sample/variant IDs."
        ),
    )
    p.add_argument(
        "--pedigree", required=True,
        help="TSV with columns: offspring, parent1, parent2 (optional: ploidy)",
    )
    p.add_argument(
        "--map", required=True,
        help="TSV with columns: marker, chrom, pos_bp (optional: cm)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output prefix; writes <prefix>.haplotypes.pt etc.",
    )
    p.add_argument(
        "--ploidy", type=int, required=True, choices=[2, 4, 6],
        help="Organism ploidy (2, 4, or 6)",
    )
    p.add_argument(
        "--parent-phased", default=None,
        help="Optional CSV of pre-phased parent genotypes (escape hatch)",
    )
    p.add_argument(
        "--no-refinemap", dest="refinemap", action="store_false",
        help="Disable PolyOrigin's map refinement (default: enabled)",
    )
    p.set_defaults(refinemap=True)
    p.add_argument(
        "--recomrate", type=float, default=1.0,
        help="cM/Mb to synthesize genetic positions when --map lacks a cm column (default: 1.0)",
    )
    p.add_argument(
        "--julia-path", default=None,
        help="Path to an existing Julia binary; else discovered or auto-installed",
    )
    install_group = p.add_mutually_exclusive_group()
    install_group.add_argument(
        "--auto-install", dest="auto_install", action="store_true",
        help="Auto-install Julia if not found (non-interactive; sets consent=True)",
    )
    install_group.add_argument(
        "--no-auto-install", dest="auto_install", action="store_false",
        help="Refuse to auto-install Julia (non-interactive; sets consent=False)",
    )
    p.set_defaults(auto_install=None)  # None → interactive prompt on tty
    p.add_argument(
        "--keep-workdir", action="store_true",
        help="Do not delete the temp work directory after success (debug aid)",
    )
    p.set_defaults(func=_cmd_phase_poly)


def _add_glm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("glm-scan", help="GLM association scan")
    _add_common_scan_args(p)
    p.add_argument("--family", default="gaussian",
                   choices=["gaussian", "binary", "ordinal", "multinomial"],
                   help="Phenotype family (default: gaussian)")
    p.add_argument("--firth", action="store_true",
                   help="Use Firth's penalized likelihood for binary/ordinal")
    p.add_argument("--n-categories", type=int, default=None,
                   help="Number of ordinal categories (required for --family ordinal)")
    p.add_argument("--no-spa", action="store_true",
                   help="Disable saddlepoint approximation for binary traits")


def _add_approx_args(p: argparse.ArgumentParser) -> None:
    """Add approximate backend arguments to a parser."""
    p.add_argument("--approx-method", default=None,
                   choices=["randomized_svd", "nystrom", "lobpcg", "sparse"],
                   help="Approximate eigendecomposition method (default: exact)")
    p.add_argument("--approx-components", type=int, default=100,
                   help="Number of components for randomized SVD / LOBPCG (default: 100)")
    p.add_argument("--approx-landmarks", type=int, default=500,
                   help="Number of landmark samples for Nystrom (default: 500)")
    p.add_argument("--sparse-threshold", type=float, default=0.05,
                   help="Relatedness threshold for sparse GRM (default: 0.05)")


def _add_lmm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("lmm-scan", help="Single-trait LMM association scan")
    _add_common_scan_args(p)
    p.add_argument("--loco", action="store_true", help="Leave-One-Chromosome-Out")
    p.add_argument("--grm", default=None,
                   help="Path to pre-computed GRM matrix (NumPy .npy, .npz, or "
                        "space/tab-delimited text). If provided, --grm-method is ignored.")
    p.add_argument("--grm-method", default="vanraden", choices=["vanraden", "zhang"],
                   help="GRM method: 'vanraden' (streaming, scalable) or 'zhang' (matches GAPIT, requires full materialization)")
    p.add_argument("--p3d", dest="p3d", action="store_true", default=True,
                   help="P3D=TRUE: estimate variance components once (default)")
    p.add_argument("--no-p3d", dest="p3d", action="store_false",
                   help="P3D=FALSE: re-estimate variance components per marker (slow)")
    _add_approx_args(p)


def _add_mvlmm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("mvlmm-scan", help="Multi-trait mvLMM association scan")
    _add_common_scan_args(p)
    p.add_argument("--grm", default=None,
                   help="Path to pre-computed GRM matrix (NumPy .npy, .npz, or "
                        "space/tab-delimited text). If provided, --grm-method is ignored.")
    p.add_argument("--grm-method", default="vanraden", choices=["vanraden", "zhang"],
                   help="GRM method: 'vanraden' (streaming, scalable) or 'zhang' (matches GAPIT, requires full materialization)")


def _add_poly_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("poly-scan", help="Polyploid GWAS scan")
    _add_common_scan_args(p)
    p.add_argument("--ploidy", type=int, required=True)
    p.add_argument("--gene-action", default="additive", choices=["additive", "general", "all", "1-dom", "diplo-additive", "overdominant"])
    p.add_argument("--p3d", dest="p3d", action="store_true", default=True,
                   help="P3D=TRUE: estimate variance components once (default)")
    p.add_argument("--no-p3d", dest="p3d", action="store_false",
                   help="P3D=FALSE: re-estimate variance components per marker (slow)")
    p.add_argument("--max-geno-freq", type=float, default=None,
                   help="Max genotype class frequency filter (e.g. 0.95). Applied after gene-action encoding.")
    p.add_argument("--bp-window", type=int, default=1_000_000,
                   help="Base-pair window for LD peak pruning (default: 1000000)")
    p.add_argument("--peak-threshold", type=float, default=1e-4,
                   help="P-value threshold for peak detection (default: 1e-4)")
    p.add_argument("--joint-qtl", action="store_true", default=False,
                   help="Fit multi-QTL joint model on detected peaks")
    p.add_argument("--meff-method", default="gao", choices=["gao", "moskvina"],
                   help="Effective test count method (default: gao)")


def _add_mklmm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("mklmm-scan", help="Multi-kernel LMM scan (additive + dominance + epistatic)")
    _add_common_scan_args(p)
    p.add_argument("--kernels", default="additive,dominance,epistatic",
                   help="Comma-separated kernel types: additive,dominance,epistatic (default: all three)")
    p.add_argument("--ploidy", type=int, default=2, help="Ploidy level (default 2)")
    p.add_argument("--grm-method", default="vanraden", choices=["vanraden", "zhang"],
                   help="GRM method for additive kernel")


def _add_gxe_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("gxe-scan", help="Gene-environment interaction LMM scan")
    _add_common_scan_args(p)
    p.add_argument("--env", required=True,
                   help="Environment variable file (TSV: SAMPLE, ENV)")
    p.add_argument("--gxe-model", default="het", choices=["het", "multi"],
                   help="GxE model: 'het' (single-trait HetLMM, default) or 'multi' (multi-trait GxELMM)")


def _add_set_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("set-scan", help="Set-based association tests (SKAT/Burden/SKAT-O)")
    _add_common_scan_args(p)
    p.add_argument("--regions", required=True,
                   help="BED file defining gene/region boundaries (chr, start, end, [name])")
    p.add_argument("--set-test", default="skat", choices=["skat", "burden", "skat_o"],
                   help="Set-based test: skat (default), burden, or skat_o")
    p.add_argument("--weight-a1", type=float, default=1.0,
                   help="Beta(MAF; a1, a2) weight parameter a1 (default 1.0)")
    p.add_argument("--weight-a2", type=float, default=25.0,
                   help="Beta(MAF; a1, a2) weight parameter a2 (default 25.0)")
    p.add_argument("--ploidy", type=int, default=2,
                   help="Ploidy level (default 2 for diploid)")


def _add_bayes_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("bayes-scan", help="Bayesian variable selection (spike-and-slab) GWAS")
    _add_common_scan_args(p)
    p.add_argument("--prior-pi", type=float, default=0.01,
                   help="Prior inclusion probability (default 0.01)")
    p.add_argument("--prior-sig2-beta", type=float, default=0.1,
                   help="Prior slab variance (default 0.1)")
    p.add_argument("--method", type=str, default="susie", choices=["susie", "cavi"],
                   help="Inference method: 'susie' (default) or 'cavi'")
    p.add_argument("--n-signals", type=int, default=10,
                   help="Number of single-effect layers for SuSiE (default 10)")
    # --max-iter inherited from _add_common_scan_args; bayes-scan default is 1000
    p.add_argument("--learn-hyperparams", action="store_true", default=True,
                   help="Learn hyperparameters via empirical Bayes (default True)")
    p.add_argument("--no-learn-hyperparams", action="store_false", dest="learn_hyperparams",
                   help="Fix hyperparameters at their initial values")
    p.add_argument("--credible-set-coverage", type=float, default=0.95,
                   help="Target coverage for credible sets (default 0.95)")
    p.add_argument("--ld-threshold", type=float, default=0.5,
                   help="r^2 threshold for LD grouping in credible sets (CAVI only, default 0.5)")
    p.add_argument("--ploidy", type=int, default=2,
                   help="Ploidy level (default 2 for diploid)")


def _add_met_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "met-scan",
        help="Multi-environment trial (MET) GWAS: stable vs environment-contingent effects",
    )
    _add_common_scan_args(p)
    p.add_argument("--env-cols", type=str, default=None,
                   help="Comma-separated environment column names in phenotype file "
                        "(default: all non-ID columns)")
    p.add_argument("--ploidy", type=int, default=2,
                   help="Ploidy level (default 2 for diploid)")
    p.add_argument("--parameterization", default="per_env",
                   choices=["per_env", "reaction_norm"],
                   help="SNP effect decomposition (default: per_env)")
    p.add_argument("--vg-structure", default="unstructured", type=str,
                   help="Genetic covariance structure: 'unstructured' or 'fa(k)' "
                        "for factor analytic with k factors (default: unstructured)")
    p.add_argument("--kernel-files", nargs="*", default=None,
                   help="Additional kernel matrix files (.npy or .pt). "
                        "Used alongside the computed additive GRM for multi-kernel MET.")


def _add_farmcpu_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "farmcpu-scan",
        help="FarmCPU multi-locus GWAS via iterative FEM/REM",
    )
    _add_common_scan_args(p)
    p.add_argument("--p-threshold", type=float, default=0.01,
                   help="P-value threshold for QTN selection (default: 0.01)")
    p.add_argument("--max-qtns", type=int, default=20,
                   help="Maximum pseudo-QTNs per iteration (default: 20)")
    p.add_argument("--bin-sizes", type=str, default="500000,5000000,50000000",
                   help="Comma-separated bin sizes in bp (default: 500000,5000000,50000000)")
    p.add_argument("--method-bin", default="static",
                   choices=["static", "optimum"],
                   help="Binning method (default: static)")
    p.add_argument("--method-sub", default="reward",
                   choices=["reward", "mean", "median", "penalty"],
                   help="P-value substitution method (default: reward)")
    p.add_argument("--maf-threshold", type=float, default=0.0,
                   help="Internal MAF threshold for QTN candidacy (default: 0.0)")


def _add_blink_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "blink-scan",
        help="BLINK multi-locus GWAS via LD-clustering and BIC selection",
    )
    _add_common_scan_args(p)
    p.add_argument("--cutoff", type=float, default=0.01,
                   help="Significance cutoff (default: 0.01)")
    p.add_argument("--max-qtns", type=int, default=0,
                   help="Maximum pseudo-QTNs (0 = unlimited, default: 0)")
    p.add_argument("--ld-threshold", type=float, default=0.7,
                   help="LD correlation threshold for clustering (default: 0.7)")
    p.add_argument("--ld-max-samples", type=int, default=200,
                   help="Max samples for LD computation (default: 200)")
    p.add_argument("--method-sub", default="reward",
                   choices=["reward", "mean", "median", "penalty"],
                   help="P-value substitution method (default: reward)")
    p.add_argument("--maf-threshold", type=float, default=0.0,
                   help="Internal MAF threshold for QTN candidacy (default: 0.0)")


def _add_threshold_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "threshold-scan",
        help="Threshold-linear GWAS for ordinal + continuous traits (Bermann et al. 2026)",
    )
    _add_common_scan_args(p)
    p.add_argument("--trait-types", required=True,
                   help="Comma-separated trait types: 'ordinal' or 'continuous' (e.g., 'ordinal,ordinal,continuous')")
    p.add_argument("--n-categories", required=True,
                   help="Comma-separated number of categories per trait (use 0 for continuous, e.g., '3,2,0')")
    p.add_argument("--R-matrix", default=None,
                   help="Path to residual covariance matrix (c x c, CSV/NPY). If omitted, identity is used.")
    p.add_argument("--G-matrix", default=None,
                   help="Path to genetic covariance matrix (c x c, CSV/NPY). If omitted, identity is used.")
    p.add_argument("--solver", default="nr", choices=["nr", "em"],
                   help="Solver: 'nr' (T-EM warm-start then Newton-Raphson) or 'em' (SQUAREM-accelerated EM)")
    p.add_argument("--em-warmup", type=int, default=5,
                   help="Number of EM warm-up iterations before NR (default: 5)")


def _add_conditional_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "conditional-scan",
        help="LD-conditional GWAS with block-aware conditioning and persistence metrics",
    )
    _add_common_scan_args(p)
    p.add_argument("--ld-method", default="r2",
                   choices=["gabriel", "four_gamete", "spine", "r2",
                            "big_ld", "cc_graph", "dp_optimize"],
                   help="LD block detection method (default: r2)")
    p.add_argument("--max-kb", type=float, default=200.0,
                   help="Max distance in kb for LD blocks (default: 200)")
    p.add_argument("--persistence-ratio", type=float, default=0.5,
                   help="Signal retention ratio for persistence metric (default: 0.5)")
    p.add_argument("--sig-threshold", type=float, default=5e-8,
                   help="Significance threshold for persistence evaluation (default: 5e-8)")
    p.add_argument("--max-conditioning", type=int, default=5,
                   help="Max lead SNPs to condition on per block (default: 5)")


def _add_mtmet_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "mtmet-scan",
        help="Multi-trait multi-environment GWAS with separable Kronecker covariance",
    )
    _add_common_scan_args(p)
    p.add_argument("--env-cols", required=True,
                   help="Comma-separated environment names (e.g., E1,E2,E3)")
    p.add_argument("--vg-structure", default="separable",
                   help="Vg structure: 'separable' (default), 'unstructured', or 'fa(k)'")


def _add_ocf_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ocf-scan",
        help="Orthogonal Cross-Fit LMM scan (DML debiased inference)",
    )
    _add_common_scan_args(p)
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of cross-fitting folds (default: 5)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for fold assignment")
    p.add_argument("--variance-type", default="HC",
                   choices=["HC", "homoskedastic"],
                   help="Variance estimator: HC (sandwich) or homoskedastic (default: HC)")
    p.add_argument("--no-genotype-projection", action="store_true",
                   help="Skip projecting genotypes onto covariate space")


def _add_knockoff_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "knockoff-scan",
        help="Knockoff FDR-controlled GWAS (Sesia et al. 2020)",
    )
    _add_common_scan_args(p)
    p.add_argument("--fdr-level", type=float, default=0.05,
                   help="Target FDR level (default: 0.05)")
    p.add_argument("--ld-method", default="gabriel",
                   choices=["gabriel", "four_gamete", "spine", "r2",
                            "gwas_aligned", "uncertainty", "cross_pop",
                            "graphical", "changepoint",
                            "big_ld", "cc_graph", "dp_optimize"],
                   help="LD block detection method (default: gabriel)")
    p.add_argument("--knockoff-method", default="equicorrelated",
                   choices=["equicorrelated"],
                   help="Knockoff generation method (default: equicorrelated)")
    p.add_argument("--aggregation", default="max_stat",
                   choices=["max_stat", "sum_sq"],
                   help="Block importance aggregation (default: max_stat)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for knockoff generation")


def _add_gu_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gu-scan",
        help="Genotype-Uncertainty LMM scan (dosage-variance corrected)",
    )
    _add_common_scan_args(p)
    # Relax the common `--genotype required=True` for gu-scan so the
    # `--probs` passthrough can derive the tensor internally. Exactly one
    # of `--genotype` or `--probs` is validated in the handler.
    for action in p._actions:
        if action.dest == "genotype":
            action.required = False
            break
    p.add_argument("--dosage-var", default=None,
                   help="Path to dosage variance file (.pt or .npy, shape n×m)")
    p.add_argument("--probs", default=None,
                   help="Path to a Phase 55 <prefix>.probs.pt from "
                        "`torchgwas dosage-call`. When given, --genotype and "
                        "--dosage-var are derived internally via "
                        "expected_dosage / dosage_variance (ploidy read from "
                        "the sibling <prefix>.meta.json).")


def _add_lro_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "lro-scan",
        help="Leave-Region-Out LMM scan (block-level LOCO)",
    )
    _add_common_scan_args(p)
    p.add_argument("--ld-method", default="r2",
                   choices=["gabriel", "four_gamete", "spine", "r2",
                            "gwas_aligned", "graphical", "changepoint",
                            "big_ld", "cc_graph", "dp_optimize"],
                   help="LD block detection method (default: r2)")


def _add_family_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "family-scan",
        help="Within-family GWAS with confounding diagnostics (Young et al. 2022)",
    )
    _add_common_scan_args(p)
    p.add_argument("--family-col", default="FID",
                   help="Column name in phenotype file containing family IDs (default: FID)")
    p.add_argument("--min-family-size", type=int, default=2,
                   help="Minimum members per family for within-family scan (default: 2)")
    p.add_argument("--confound-threshold", type=float, default=0.5,
                   help="Attenuation deviation threshold for confounding flag (default: 0.5)")


def _add_glmm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "glmm-scan",
        help="GLMM association scan (binary/ordinal with random effects, SAIGE-style PQL)",
    )
    _add_common_scan_args(p)
    p.add_argument("--family", default="binary",
                   choices=["binary", "ordinal", "multinomial"],
                   help="Phenotype family (default: binary)")
    p.add_argument("--n-categories", type=int, default=3,
                   help="Number of ordinal categories (default: 3, ignored for binary)")
    p.add_argument("--firth", action="store_true", default=False,
                   help="Use Firth correction for GLM initialisation (binary only)")
    p.add_argument("--no-spa", action="store_true", default=False,
                   help="Disable SPA tail correction (binary only)")
    p.add_argument("--pql-max-iter", type=int, default=30,
                   help="Max PQL outer iterations (default: 30)")


def _add_me_glmm_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "me-glmm-scan",
        help="Multi-environment GLMM scan (binary/ordinal across environments)",
    )
    _add_common_scan_args(p)
    p.add_argument("--family", default="binary",
                   choices=["binary", "ordinal"],
                   help="Phenotype family (default: binary)")
    p.add_argument("--n-categories", type=int, default=None,
                   help="Number of ordinal categories (required for ordinal)")
    p.add_argument("--env-cols", default=None,
                   help="Comma-separated environment column names")
    p.add_argument("--parameterization", default="per_env",
                   choices=["per_env", "reaction_norm"],
                   help="Test parameterization (default: per_env)")
    p.add_argument("--firth", action="store_true", default=False,
                   help="Use Firth correction for per-env GLM init")
    p.add_argument("--no-spa", action="store_true", default=False,
                   help="Disable SPA for per-env binary marginals")
    p.add_argument("--pql-max-iter", type=int, default=30,
                   help="Max PQL outer iterations (default: 30)")


def _add_survival_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "survival-scan",
        help="Survival GWAS scan (Cox PH frailty model with martingale residual score test)",
    )
    _add_common_scan_args(p)
    p.add_argument("--time-col", default=None,
                   help="Column name for follow-up time (default: first phenotype column)")
    p.add_argument("--event-col", default=None,
                   help="Column name for event indicator 0/1 (default: second phenotype column)")
    p.add_argument("--no-spa", action="store_true", default=False,
                   help="Disable SPA for tail p-values")
    p.add_argument("--spa-threshold", type=float, default=2.0,
                   help="Chi-square threshold for SPA activation (default: 2.0)")
    p.add_argument("--pql-max-iter", type=int, default=30,
                   help="Max PQL outer iterations (default: 30)")
    p.add_argument("--pql-tol", type=float, default=1e-4,
                   help="PQL convergence tolerance (default: 1e-4)")
    p.add_argument("--ties", default="breslow", choices=["breslow"],
                   help="Tie-handling method (default: breslow)")


def _add_ld_blocks_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ld-blocks",
        help="Detect haplotype blocks from genotype data",
    )
    p.add_argument("--genotype", required=True, help="Genotype file path")
    p.add_argument("--method", default="gabriel",
                   choices=["gabriel", "four_gamete", "spine", "r2",
                            "gwas_aligned", "uncertainty", "cross_pop",
                            "graphical", "changepoint",
                            "big_ld", "cc_graph", "dp_optimize",
                            "wall_pritchard"],
                   help="Block detection method (default: gabriel)")
    p.add_argument("--phased-vcf", default=None,
                   help="Phased VCF for D'/four-gamete (optional)")
    p.add_argument("--max-kb", type=float, default=200.0,
                   help="Max distance in kb for pairwise LD (default: 200)")
    p.add_argument("--output", default="ld_blocks",
                   help="Output prefix (default: ld_blocks)")
    p.add_argument("--device", default=None, help="Compute device (cpu/cuda)")
    # Gabriel-specific
    p.add_argument("--ci-low", type=float, default=0.70,
                   help="Gabriel CI lower threshold (default: 0.70)")
    p.add_argument("--ci-high", type=float, default=0.98,
                   help="Gabriel CI upper threshold (default: 0.98)")
    # Four-gamete-specific
    p.add_argument("--freq-threshold", type=float, default=0.01,
                   help="Four-gamete frequency threshold (default: 0.01)")
    # Spine-specific
    p.add_argument("--dprime-threshold", type=float, default=0.8,
                   help="Spine D' threshold (default: 0.8)")
    # r2-specific
    p.add_argument("--r2-threshold", type=float, default=0.2,
                   help="r² block threshold (default: 0.2)")
    # GWAS-aligned
    p.add_argument("--condition-penalty", type=float, default=0.1,
                   help="Condition number penalty (gwas_aligned, default: 0.1)")
    p.add_argument("--max-block-snps", type=int, default=50,
                   help="Max SNPs per block (gwas_aligned/dp_optimize, default: 50)")
    # Graphical
    p.add_argument("--l1-penalty", type=float, default=0.1,
                   help="L1 penalty for graphical Lasso (default: 0.1)")
    # Change-point
    p.add_argument("--cp-penalty", type=float, default=0.0,
                   help="Change-point penalty (0=auto BIC, default: 0)")
    # DP optimize
    p.add_argument("--objective", default="haplotype_diversity",
                   choices=["haplotype_diversity", "tag_snp"],
                   help="DP optimization objective (default: haplotype_diversity)")
    # Big-LD
    p.add_argument("--window-size", type=int, default=200,
                   help="Window size for Big-LD / graphical (default: 200)")
    # CC-Graph
    p.add_argument("--ld-window", type=int, default=100,
                   help="Index window for cc_graph pairs (default: 100)")
    p.add_argument("--include-singletons", action="store_true", default=True,
                   help="Include singleton SNPs as size-1 blocks (cc_graph)")
    # PLINK comparison
    p.add_argument("--compare-plink", default=None,
                   help="Path to PLINK .blocks.det file for concordance comparison")


def _add_ldsc_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("ldsc", help="LDSC SNP heritability estimation")
    p.add_argument("--sumstats", required=True, help="Summary statistics TSV")
    p.add_argument("--genotype", required=True, help="Genotype file for LD score computation")
    p.add_argument("--window-kb", type=float, default=1000.0, help="LD window in kb (default 1000)")
    p.add_argument("--n", type=int, default=None, help="Sample size (if not in sumstats)")
    p.add_argument("--output", default="torchgwas_results")


def _add_ldsc_rg_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("ldsc-rg", help="LDSC genetic correlation")
    p.add_argument("--sumstats1", required=True, help="Summary stats for trait 1")
    p.add_argument("--sumstats2", required=True, help="Summary stats for trait 2")
    p.add_argument("--genotype", required=True, help="Genotype file for LD scores")
    p.add_argument("--window-kb", type=float, default=1000.0, help="LD window in kb (default 1000)")
    p.add_argument("--n1", type=int, default=None, help="Sample size trait 1")
    p.add_argument("--n2", type=int, default=None, help="Sample size trait 2")
    p.add_argument("--output", default="torchgwas_results")


def _add_meta_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("meta", help="Meta-analysis across GWAS studies")
    p.add_argument("--input", nargs="+", required=True, help="Summary stats files")
    p.add_argument("--method", default="fixed",
                   choices=["fixed", "random", "stouffer", "han_eskin"],
                   help="Meta-analysis method (default: fixed)")
    p.add_argument("--output", default="torchgwas_results")


def _add_clump_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("clump", help="LD clumping for independent loci")
    p.add_argument("--sumstats", required=True, help="Summary statistics TSV")
    p.add_argument("--genotype", required=True, help="Genotype file for r2 computation")
    p.add_argument("--r2", type=float, default=0.1, help="r2 threshold (default 0.1)")
    p.add_argument("--p-threshold", type=float, default=5e-8, help="P-value threshold (default 5e-8)")
    p.add_argument("--window-kb", type=float, default=250.0, help="Window in kb (default 250)")
    p.add_argument("--output", default="torchgwas_results")


def _add_pgs_fit_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "pgs-fit",
        help="Fit PGS weights from GWAS sumstats (C+T, LDpred2, PRS-CS)",
    )
    p.add_argument("--sumstats", required=True, help="Summary statistics TSV/CSV")
    p.add_argument("--ld-ref", required=True, help="LD reference panel (.pt, from save_ld_reference)")
    p.add_argument(
        "--method", required=True,
        choices=["ct", "ldpred2-inf", "ldpred2-grid", "ldpred2-auto", "prscs"],
        help="PGS construction method",
    )
    p.add_argument("--output", required=True, help="Output TSV path for PGS weights")
    # Shared method hyperparameters
    p.add_argument("--h2", type=float, default=0.1, help="Heritability (LDpred2-inf / init for auto)")
    p.add_argument("--n-iter", type=int, default=200, help="MCMC iterations")
    p.add_argument("--n-burnin", type=int, default=100, help="MCMC burn-in")
    p.add_argument("--n-chains", type=int, default=3, help="Number of chains (ldpred2-auto, prscs)")
    p.add_argument("--p-causal", type=float, default=0.01, help="Prior causal fraction (init)")
    p.add_argument("--phi", type=float, default=1e-2, help="PRS-CS global shrinkage parameter")
    p.add_argument("--grid-p", type=str, default="1e-3,1e-2,1e-1",
                   help="Comma-separated p grid (ldpred2-grid)")
    p.add_argument("--grid-h2", type=str, default="0.1,0.3",
                   help="Comma-separated h2 grid (ldpred2-grid)")
    p.add_argument("--grid-sparse", action="store_true",
                   help="Use sparse spike-and-slab (ldpred2-grid)")
    # C+T-specific
    p.add_argument("--clump-r2", type=float, default=0.1, help="C+T r2 threshold")
    p.add_argument("--clump-p", type=float, default=5e-8, help="C+T p threshold")
    p.add_argument("--clump-kb", type=float, default=250.0, help="C+T window in kb")
    # Harmonization
    p.add_argument("--maf-tol", type=float, default=0.20, help="MAF mismatch tolerance")
    p.add_argument("--palindromic-maf-tol", type=float, default=0.42,
                   help="Palindromic SNP MAF disambiguation cutoff")
    p.add_argument("--keep-ambiguous", action="store_true",
                   help="Keep palindromic SNPs when MAF permits (default: drop)")
    # Runtime
    p.add_argument("--device", default="cpu", help="Device (cpu / cuda)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")


def _add_pgs_score_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "pgs-score",
        help="Apply PGS weights to target genotypes",
    )
    p.add_argument("--genotype", required=True, help="Target genotype file")
    p.add_argument("--weights", required=True, help="PGS weights TSV (from pgs-fit)")
    p.add_argument("--output", required=True, help="Output per-individual scores TSV")
    p.add_argument("--standardize", action="store_true", help="Standardize SNP dosages")
    p.add_argument("--handle-missing", default="mean", choices=["mean", "drop"],
                   help="NaN handling (default: mean)")
    p.add_argument("--chunk-size", type=int, default=50000, help="SNP chunk size")
    p.add_argument("--device", default="cpu", help="Device (cpu / cuda)")


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("pipeline", help="Full pipeline: impute -> scan -> correct")
    _add_common_scan_args(p)
    p.add_argument("--impute", choices=[
        "mean", "beagle", "impute5", "minimac4", "li-stephens", "deep-learning",
    ])
    p.add_argument("--model", default="lmm", choices=[
        "glm", "lmm", "mvlmm", "farmcpu", "blink", "mklmm", "gxe", "met",
    ])
    p.add_argument("--ploidy", type=int, default=2, help="Ploidy level (default 2 for diploid)")
    # MET-specific args (used when --model met)
    p.add_argument("--env-cols", type=str, default=None,
                   help="Comma-separated environment columns (for --model met)")
    p.add_argument("--parameterization", default="per_env",
                   choices=["per_env", "reaction_norm"],
                   help="SNP effect decomposition for MET (default: per_env)")
    p.add_argument("--vg-structure", default="unstructured", type=str,
                   help="Genetic covariance structure for MET: 'unstructured' or 'fa(k)'")
    _add_approx_args(p)


def _add_rr_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rr-scan",
        help="Random Regression LMM GWAS for longitudinal / spatio-temporal data",
    )
    _add_common_scan_args(p)
    p.add_argument("--id-col", default="IID",
                   help="Phenotype column with sample identifier (default: IID)")
    p.add_argument("--time-col", default="TIME",
                   help="Phenotype column with raw time/age values (default: TIME)")
    p.add_argument("--pheno-col", default="Y",
                   help="Phenotype column with the long-format response (default: Y)")
    p.add_argument("--basis", default="legendre",
                   choices=["legendre", "bspline"],
                   help="Temporal basis (default: legendre)")
    p.add_argument("--order", type=int, default=2,
                   help="Legendre polynomial order; b = order + 1 (default: 2)")
    p.add_argument("--n-knots", type=int, default=4,
                   help="B-spline interior knot count (default: 4)")
    p.add_argument("--degree", type=int, default=3,
                   help="B-spline degree (default: 3)")
    p.add_argument("--k-coef-structure", default="unstructured",
                   help="K_coef structure: 'unstructured', 'diagonal', or 'fa(k)'")
    p.add_argument("--include-pe", action="store_true",
                   help="Decompose Ve into a permanent-environment block")
    p.add_argument("--mode", default="projection",
                   choices=["projection", "stacked"],
                   help="Reduction mode (default: projection)")
    p.add_argument("--eval-times", default=None,
                   help="Comma-separated raw time values for per-time-point χ²(1) "
                        "reconstruction. Produces an extra .rr_at_t.tsv output.")
    # Spatio-temporal extension
    p.add_argument("--row-col", default=None,
                   help="Phenotype column with field-trial row coordinate "
                        "(enables 2D P-spline spatial smoother).")
    p.add_argument("--col-col", default=None,
                   help="Phenotype column with field-trial column coordinate.")
    p.add_argument("--spatial-knots-row", type=int, default=8)
    p.add_argument("--spatial-knots-col", type=int, default=8)
    p.add_argument("--lambda-row", type=float, default=1.0)
    p.add_argument("--lambda-col", type=float, default=1.0)


def _merge_rr_results(results: list):
    """Concatenate RRScanResult objects across genotype chunks."""
    import torch

    from .models.rr_lmm import RRScanResult

    chr_all, pos_all, snp_all, a1_all, a2_all = [], [], [], [], []
    for r in results:
        chr_all.extend(r.chr)
        pos_all.extend(r.pos)
        snp_all.extend(r.snp)
        a1_all.extend(r.a1)
        a2_all.extend(r.a2)

    eval_kwargs = {}
    if results[0].eval_times is not None:
        eval_kwargs["eval_times"] = results[0].eval_times
        eval_kwargs["beta_at_t"] = torch.cat([r.beta_at_t for r in results])
        eval_kwargs["se_at_t"] = torch.cat([r.se_at_t for r in results])
        eval_kwargs["stat_at_t"] = torch.cat([r.stat_at_t for r in results])
        eval_kwargs["p_at_t"] = torch.cat([r.p_at_t for r in results])

    return RRScanResult(
        chr=chr_all, pos=pos_all, snp=snp_all, a1=a1_all, a2=a2_all,
        af=torch.cat([r.af for r in results]),
        beta=torch.cat([r.beta for r in results]),
        se=torch.cat([r.se for r in results]),
        Var_beta=torch.cat([r.Var_beta for r in results]),
        stat_joint=torch.cat([r.stat_joint for r in results]),
        p_joint=torch.cat([r.p_joint for r in results]),
        stat_intercept=torch.cat([r.stat_intercept for r in results]),
        p_intercept=torch.cat([r.p_intercept for r in results]),
        stat_slope=torch.cat([r.stat_slope for r in results]),
        p_slope=torch.cat([r.p_slope for r in results]),
        stat_time_varying=torch.cat([r.stat_time_varying for r in results]),
        p_time_varying=torch.cat([r.p_time_varying for r in results]),
        test=results[0].test,
        basis_kind=results[0].basis_kind,
        basis_params=results[0].basis_params,
        b=results[0].b,
        **eval_kwargs,
    )


def _save_rr_results(result, args) -> None:
    """Write per-SNP TSV (.rr.tsv) and optional per-time TSV (.rr_at_t.tsv)."""
    import pandas as pd

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "CHR": result.chr,
        "POS": result.pos,
        "SNP": result.snp,
        "A1": result.a1,
        "A2": result.a2,
        "AF": result.af.detach().cpu().numpy(),
        "STAT_JOINT": result.stat_joint.detach().cpu().numpy(),
        "P_JOINT": result.p_joint.detach().cpu().numpy(),
        "STAT_INTERCEPT": result.stat_intercept.detach().cpu().numpy(),
        "P_INTERCEPT": result.p_intercept.detach().cpu().numpy(),
        "STAT_SLOPE": result.stat_slope.detach().cpu().numpy(),
        "P_SLOPE": result.p_slope.detach().cpu().numpy(),
        "STAT_TIME_VARYING": result.stat_time_varying.detach().cpu().numpy(),
        "P_TIME_VARYING": result.p_time_varying.detach().cpu().numpy(),
    })
    rr_path = out.with_suffix(".rr.tsv")
    df.to_csv(rr_path, sep="\t", index=False)
    logger.info("Saved %d RR-scan rows to %s", len(df), rr_path)

    if result.eval_times is not None:
        eval_t = result.eval_times.detach().cpu().numpy()
        beta_at_t = result.beta_at_t.detach().cpu().numpy()
        p_at_t = result.p_at_t.detach().cpu().numpy()
        rows = []
        for j, snp in enumerate(result.snp):
            for k, tk in enumerate(eval_t):
                rows.append({
                    "SNP": snp,
                    "TIME": float(tk),
                    "BETA": float(beta_at_t[j, k]),
                    "P": float(p_at_t[j, k]),
                })
        at_t_df = pd.DataFrame(rows)
        at_t_path = out.with_suffix(".rr_at_t.tsv")
        at_t_df.to_csv(at_t_path, sep="\t", index=False)
        logger.info("Saved %d per-time rows to %s", len(at_t_df), at_t_path)


def _cmd_rr_scan(args: argparse.Namespace) -> int:
    """Run a Random Regression LMM GWAS scan on long-format longitudinal data."""
    import json

    import pandas as pd
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .linalg.kinship import grm_vanraden
    from .models.rr_lmm import RandomRegressionLMM
    from .models.rr_spatial import SpatioTemporalRR

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    # Load long-format phenotype TSV.
    pheno_df = pd.read_csv(args.phenotype, sep="\t")
    for col in (args.id_col, args.time_col, args.pheno_col):
        if col not in pheno_df.columns:
            raise ValueError(
                f"rr-scan: phenotype file is missing required column '{col}'. "
                f"Available columns: {list(pheno_df.columns)}"
            )

    # Open the genotype reader (without using _load_scan_data, which expects
    # one phenotype row per sample).
    from .io.detect import detect_format
    from .io.validate import _open_reader
    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)

    # Build the unique sample-id ordering from the phenotype file, then
    # restrict the reader to its intersection with the genotype samples.
    pheno_sample_ids = pheno_df[args.id_col].astype(str).tolist()
    geno_sample_ids = [str(s) for s in reader.sample_ids]
    geno_id_to_idx = {s: i for i, s in enumerate(geno_sample_ids)}
    unique_pheno = list(dict.fromkeys(pheno_sample_ids))
    aligned_ids = [s for s in unique_pheno if s in geno_id_to_idx]
    if len(aligned_ids) < 2:
        raise ValueError(
            f"rr-scan: only {len(aligned_ids)} samples overlap between phenotype "
            f"id-col '{args.id_col}' and genotype file."
        )
    geno_idx = [geno_id_to_idx[s] for s in aligned_ids]
    from .io.aligned import SampleAlignedReader
    aligned_reader = SampleAlignedReader(reader, geno_idx, aligned_ids)

    # Materialize the full aligned genotype matrix.
    G_full, vmeta = _load_full_genotype(aligned_reader)
    G_full = G_full.to(device)
    n_samples = G_full.shape[0]

    # Build long-format vectors keyed to the aligned sample order.
    id_to_int = {s: i for i, s in enumerate(aligned_ids)}
    pheno_df = pheno_df[pheno_df[args.id_col].astype(str).isin(id_to_int)].copy()
    pheno_df["__row_idx"] = pheno_df[args.id_col].astype(str).map(id_to_int)

    Y_long = torch.tensor(
        pheno_df[args.pheno_col].values, dtype=STAT_DTYPE, device=device
    )
    sample_ids_long = torch.tensor(
        pheno_df["__row_idx"].values, dtype=torch.int64, device=device
    )
    time_long = torch.tensor(
        pheno_df[args.time_col].values, dtype=STAT_DTYPE, device=device
    )
    # Per-individual covariate matrix: intercept-only by default.
    X0 = torch.ones(n_samples, 1, dtype=STAT_DTYPE, device=device)

    # Build the GRM from the aligned genotype.
    K, _ = grm_vanraden(G_full, ploidy=2)

    # Spatial mode if both row/col columns provided.
    use_spatial = bool(args.row_col) and bool(args.col_col)
    if use_spatial:
        for col in (args.row_col, args.col_col):
            if col not in pheno_df.columns:
                raise ValueError(f"rr-scan: spatial column '{col}' not in phenotype.")
        row_long = torch.tensor(
            pheno_df[args.row_col].values, dtype=STAT_DTYPE, device=device
        )
        col_long = torch.tensor(
            pheno_df[args.col_col].values, dtype=STAT_DTYPE, device=device
        )
        model = SpatioTemporalRR(
            basis=args.basis, order=args.order,
            n_interior_knots=args.n_knots, degree=args.degree,
            k_coef_structure=args.k_coef_structure,
            include_pe=args.include_pe, mode=args.mode,
            config=config.numerical,
            n_knots_row=args.spatial_knots_row,
            n_knots_col=args.spatial_knots_col,
            lambda_row=args.lambda_row, lambda_col=args.lambda_col,
        )
        null_fit = model.fit_null(
            Y_long, X0, K,
            sample_ids=sample_ids_long, time_values=time_long,
            row_coords=row_long, col_coords=col_long,
        )
    else:
        model = RandomRegressionLMM(
            basis=args.basis, order=args.order,
            n_interior_knots=args.n_knots, degree=args.degree,
            k_coef_structure=args.k_coef_structure,
            include_pe=args.include_pe, mode=args.mode,
            config=config.numerical,
        )
        null_fit = model.fit_null(
            Y_long, X0, K,
            sample_ids=sample_ids_long, time_values=time_long,
        )

    # Optional eval_times parsing.
    eval_times_t = None
    if args.eval_times:
        eval_times_t = torch.tensor(
            [float(x) for x in args.eval_times.split(",")],
            dtype=STAT_DTYPE, device=device,
        )

    # Score chunks.
    results = []
    for G_c, vm in aligned_reader.iter_chunks(config.chunk_size):
        from .preprocess.impute import impute_mean
        G_c = impute_mean(G_c.to(STAT_DTYPE)).to(device)
        res = model.score_chunk(
            G_c, null_fit, vm, eval_times=eval_times_t,
        )
        results.append(res)

    merged = _merge_rr_results(results)
    _save_rr_results(merged, args)

    # Null-model JSON: log-lik, K_coef, basis params, mode/structure.
    out = Path(args.output)
    null_meta = {
        "log_likelihood": float(null_fit.log_likelihood or 0.0),
        "converged": bool(null_fit.converged),
        "n_samples": int(n_samples),
        "n_obs_total": int(Y_long.shape[0]),
        "b": int(null_fit.b),
        "basis_kind": str(null_fit.basis_kind),
        "k_coef_structure": str(getattr(null_fit, "k_coef_structure", "unstructured")),
        "mode": str(args.mode),
        "include_pe": bool(getattr(null_fit, "include_pe", False)),
        "K_coef": null_fit.K_coef.detach().cpu().tolist(),
        "Ve": null_fit.Ve.detach().cpu().tolist(),
        "spatial": bool(use_spatial),
    }
    with open(out.with_suffix(".rr_null.json"), "w") as f:
        json.dump(null_meta, f, indent=2)
    logger.info("Saved RR null metadata to %s", out.with_suffix(".rr_null.json"))
    return 0


def _add_rr_met_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rr-met-scan",
        help="Random Regression LMM × Multi-Environment GWAS for longitudinal MET data",
    )
    _add_common_scan_args(p)
    p.add_argument("--id-col", default="IID",
                   help="Phenotype column with sample identifier (default: IID)")
    p.add_argument("--env-col", default="ENV",
                   help="Phenotype column with environment label (default: ENV)")
    p.add_argument("--time-col", default="TIME",
                   help="Phenotype column with raw time/age values (default: TIME)")
    p.add_argument("--pheno-col", default="Y",
                   help="Phenotype column with the long-format response (default: Y)")
    p.add_argument("--basis", default="legendre",
                   choices=["legendre", "bspline"],
                   help="Temporal basis (default: legendre). Pooled across all envs.")
    p.add_argument("--order", type=int, default=2,
                   help="Legendre polynomial order; b = order + 1 (default: 2)")
    p.add_argument("--n-knots", type=int, default=4,
                   help="B-spline interior knot count (default: 4)")
    p.add_argument("--degree", type=int, default=3,
                   help="B-spline degree (default: 3)")
    p.add_argument("--vg-structure", default="separable",
                   choices=["separable", "unstructured", "fa(1)", "fa(2)", "fa(3)"],
                   help="Genetic covariance structure for the (b·E) pseudo-traits "
                        "(default: separable Vg = K_coef ⊗ Vg_env)")
    p.add_argument("--eval-times", default=None,
                   help="Comma-separated raw time values for per-(time, env) χ²(1) "
                        "reconstruction. Produces an extra .rr_met_at_t.tsv output.")


def _merge_rr_met_results(results: list):
    """Concatenate RRMetScanResult objects across genotype chunks."""
    import torch

    from .models.rr_met import RRMetScanResult

    chr_all, pos_all, snp_all, a1_all, a2_all = [], [], [], [], []
    for r in results:
        chr_all.extend(r.chr)
        pos_all.extend(r.pos)
        snp_all.extend(r.snp)
        a1_all.extend(r.a1)
        a2_all.extend(r.a2)

    test_names = list(results[0].stat_by_test.keys())
    stat_by_test = {
        name: torch.cat([r.stat_by_test[name] for r in results])
        for name in test_names
    }
    p_by_test = {
        name: torch.cat([r.p_by_test[name] for r in results])
        for name in test_names
    }

    eval_kwargs = {}
    if results[0].eval_times is not None:
        eval_kwargs["eval_times"] = results[0].eval_times
        eval_kwargs["beta_at_t"] = torch.cat([r.beta_at_t for r in results])
        eval_kwargs["se_at_t"] = torch.cat([r.se_at_t for r in results])
        eval_kwargs["stat_at_t"] = torch.cat([r.stat_at_t for r in results])
        eval_kwargs["p_at_t"] = torch.cat([r.p_at_t for r in results])

    return RRMetScanResult(
        chr=chr_all, pos=pos_all, snp=snp_all, a1=a1_all, a2=a2_all,
        af=torch.cat([r.af for r in results]),
        beta=torch.cat([r.beta for r in results]),
        se=torch.cat([r.se for r in results]),
        Var_beta=torch.cat([r.Var_beta for r in results]),
        stat_joint=torch.cat([r.stat_joint for r in results]),
        p_joint=torch.cat([r.p_joint for r in results]),
        stat_by_test=stat_by_test,
        p_by_test=p_by_test,
        df_by_test=results[0].df_by_test,
        test=results[0].test,
        basis_kind=results[0].basis_kind,
        basis_params=results[0].basis_params,
        b=results[0].b,
        E=results[0].E,
        env_index=results[0].env_index,
        **eval_kwargs,
    )


def _save_rr_met_results(result, args, env_labels: list[str]) -> None:
    """Write per-SNP TSV (.rr_met.tsv) and optional per-(time, env) TSV."""
    import pandas as pd

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = {
        "CHR": result.chr,
        "POS": result.pos,
        "SNP": result.snp,
        "A1": result.a1,
        "A2": result.a2,
        "AF": result.af.detach().cpu().numpy(),
        "STAT_JOINT": result.stat_joint.detach().cpu().numpy(),
        "P_JOINT": result.p_joint.detach().cpu().numpy(),
    }
    for name in result.stat_by_test:
        upper = name.upper()
        cols[f"STAT_{upper}"] = result.stat_by_test[name].detach().cpu().numpy()
        cols[f"P_{upper}"] = result.p_by_test[name].detach().cpu().numpy()
    df = pd.DataFrame(cols)
    rr_path = out.with_suffix(".rr_met.tsv")
    df.to_csv(rr_path, sep="\t", index=False)
    logger.info("Saved %d RR-MET-scan rows to %s", len(df), rr_path)

    if result.eval_times is not None:
        eval_t = result.eval_times.detach().cpu().numpy()
        beta_at_t = result.beta_at_t.detach().cpu().numpy()  # (m, n_t, E)
        p_at_t = result.p_at_t.detach().cpu().numpy()
        stat_at_t = result.stat_at_t.detach().cpu().numpy()
        rows = []
        for j, snp in enumerate(result.snp):
            for k, tk in enumerate(eval_t):
                for e_i, e_lab in enumerate(env_labels):
                    rows.append({
                        "SNP": snp,
                        "TIME": float(tk),
                        "ENV": e_lab,
                        "BETA": float(beta_at_t[j, k, e_i]),
                        "STAT": float(stat_at_t[j, k, e_i]),
                        "P": float(p_at_t[j, k, e_i]),
                    })
        at_t_df = pd.DataFrame(rows)
        at_t_path = out.with_suffix(".rr_met_at_t.tsv")
        at_t_df.to_csv(at_t_path, sep="\t", index=False)
        logger.info("Saved %d per-(time,env) rows to %s", len(at_t_df), at_t_path)


def _cmd_rr_met_scan(args: argparse.Namespace) -> int:
    """Run a Random Regression × Multi-Environment LMM GWAS scan on long-format data."""
    import json

    import pandas as pd
    import torch

    from .config import STAT_DTYPE, TorchGWASConfig, resolve_device
    from .linalg.kinship import grm_vanraden
    from .models.rr_met import RandomRegressionMultiEnvLMM

    device = resolve_device(args.device)
    config = TorchGWASConfig(device=device, chunk_size=args.chunk_size)

    pheno_df = pd.read_csv(args.phenotype, sep="\t")
    for col in (args.id_col, args.env_col, args.time_col, args.pheno_col):
        if col not in pheno_df.columns:
            raise ValueError(
                f"rr-met-scan: phenotype file is missing required column '{col}'. "
                f"Available columns: {list(pheno_df.columns)}"
            )

    from .io.detect import detect_format
    from .io.validate import _open_reader
    fmt = detect_format(args.genotype)
    reader = _open_reader(args.genotype, fmt)

    pheno_sample_ids = pheno_df[args.id_col].astype(str).tolist()
    geno_sample_ids = [str(s) for s in reader.sample_ids]
    geno_id_to_idx = {s: i for i, s in enumerate(geno_sample_ids)}
    unique_pheno = list(dict.fromkeys(pheno_sample_ids))
    aligned_ids = [s for s in unique_pheno if s in geno_id_to_idx]
    if len(aligned_ids) < 2:
        raise ValueError(
            f"rr-met-scan: only {len(aligned_ids)} samples overlap between phenotype "
            f"id-col '{args.id_col}' and genotype file."
        )
    geno_idx = [geno_id_to_idx[s] for s in aligned_ids]
    from .io.aligned import SampleAlignedReader
    aligned_reader = SampleAlignedReader(reader, geno_idx, aligned_ids)

    G_full, vmeta = _load_full_genotype(aligned_reader)
    G_full = G_full.to(device)
    n_samples = G_full.shape[0]

    id_to_int = {s: i for i, s in enumerate(aligned_ids)}
    pheno_df = pheno_df[pheno_df[args.id_col].astype(str).isin(id_to_int)].copy()
    pheno_df["__row_idx"] = pheno_df[args.id_col].astype(str).map(id_to_int)

    # Encode env labels to integer ids in the order they appear.
    env_labels = list(dict.fromkeys(pheno_df[args.env_col].astype(str).tolist()))
    env_to_int = {e: i for i, e in enumerate(env_labels)}
    pheno_df["__env_idx"] = pheno_df[args.env_col].astype(str).map(env_to_int)

    Y_long = torch.tensor(
        pheno_df[args.pheno_col].values, dtype=STAT_DTYPE, device=device
    )
    sample_ids_long = torch.tensor(
        pheno_df["__row_idx"].values, dtype=torch.int64, device=device
    )
    env_ids_long = torch.tensor(
        pheno_df["__env_idx"].values, dtype=torch.int64, device=device
    )
    time_long = torch.tensor(
        pheno_df[args.time_col].values, dtype=STAT_DTYPE, device=device
    )
    X0 = torch.ones(n_samples, 1, dtype=STAT_DTYPE, device=device)
    K, _ = grm_vanraden(G_full, ploidy=2)

    model = RandomRegressionMultiEnvLMM(
        basis=args.basis, order=args.order,
        n_interior_knots=args.n_knots, degree=args.degree,
        vg_structure=args.vg_structure,
        config=config.numerical,
    )
    null_fit = model.fit_null(
        Y_long, X0, K,
        sample_ids=sample_ids_long,
        env_ids=env_ids_long,
        time_values=time_long,
        env_names=env_labels,
    )

    eval_times_t = None
    if args.eval_times:
        eval_times_t = torch.tensor(
            [float(x) for x in args.eval_times.split(",")],
            dtype=STAT_DTYPE, device=device,
        )

    results = []
    for G_c, vm in aligned_reader.iter_chunks(config.chunk_size):
        from .preprocess.impute import impute_mean
        G_c = impute_mean(G_c.to(STAT_DTYPE)).to(device)
        res = model.score_chunk(
            G_c, null_fit, vm, eval_times=eval_times_t,
        )
        results.append(res)

    merged = _merge_rr_met_results(results)
    _save_rr_met_results(merged, args, env_labels)

    out = Path(args.output)
    null_meta = {
        "log_likelihood": float(null_fit.log_likelihood or 0.0),
        "converged": bool(null_fit.converged),
        "n_samples": int(n_samples),
        "n_obs_total": int(Y_long.shape[0]),
        "b": int(null_fit.b),
        "E": int(null_fit.E_envs),
        "basis_kind": str(null_fit.basis_kind),
        "vg_structure": str(getattr(null_fit, "vg_structure_rr", "separable")),
        "env_labels": env_labels,
        "K_coef": null_fit.K_coef.detach().cpu().tolist(),
        "Vg_env": null_fit.Vg_env.detach().cpu().tolist(),
        "Ve": null_fit.Ve.detach().cpu().tolist(),
    }
    with open(out.with_suffix(".rr_met_null.json"), "w") as f:
        json.dump(null_meta, f, indent=2)
    logger.info("Saved RR-MET null metadata to %s", out.with_suffix(".rr_met_null.json"))
    return 0


def _add_common_scan_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all scan commands."""
    parser.add_argument("--genotype", required=True)
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--covariate")
    parser.add_argument("--test", default="wald", choices=["wald", "score", "lrt"])
    parser.add_argument("--correction", default="bh",
                        choices=["bonferroni", "holm", "bh", "by", "storey", "simplem",
                                 "weighted-bh", "lfdr", "hierarchical", "ihw", "adapt",
                                 "none"])
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--miss-max", type=float, default=0.10)
    parser.add_argument("--output", default="torchgwas_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Maximum REML optimizer iterations (default: from config)")
    parser.add_argument("--weights-file", default=None,
                        help="TSV with SNP weights for weighted-bh correction (columns: SNP, WEIGHT)")
    parser.add_argument("--gene-map", default=None,
                        help="TSV mapping SNP to gene/group for hierarchical FDR (columns: SNP, GENE)")
    parser.add_argument("--fdr-covariate", default=None,
                        help="TSV with auxiliary covariate for IHW/AdaPT (columns: SNP, COV). "
                             "If omitted, MAF from the scan result is used.")
    parser.add_argument("--n-pcs", type=int, default=0,
                        help="Number of principal components from GRM eigendecomposition "
                             "to include as covariates for population structure correction "
                             "(default: 0, meaning no PCs added)")
    parser.add_argument("--id-column", default=None,
                        help="Column name in phenotype/covariate file containing sample IDs "
                             "(default: auto-detect from column names like IID, Sample, Taxa)")
    parser.add_argument("--traits", default=None,
                        help="Comma-separated trait column names from phenotype file. "
                             "For univariate models, selects the specified trait(s). "
                             "For multivariate models, all traits are analyzed jointly.")


def _load_weights(weights_file: str, snp_ids: list[str]) -> torch.Tensor:
    """Load per-SNP weights from a TSV file (columns: SNP, WEIGHT)."""
    import pandas as pd
    import torch

    df = pd.read_csv(weights_file, sep="\t")
    weight_map = dict(zip(df["SNP"], df["WEIGHT"]))
    weights = torch.tensor(
        [weight_map.get(s, 1.0) for s in snp_ids],
        dtype=torch.float64,
    )
    return weights


def _load_fdr_covariate(args: argparse.Namespace, result) -> torch.Tensor:
    """Load auxiliary covariate for IHW/AdaPT. Falls back to MAF."""
    import pandas as pd
    import torch

    fdr_cov_file = getattr(args, "fdr_covariate", None)
    if fdr_cov_file is not None:
        df = pd.read_csv(fdr_cov_file, sep="\t")
        cov_map = dict(zip(df["SNP"], df["COV"]))
        cov = torch.tensor(
            [cov_map.get(s, 0.0) for s in result.snp],
            dtype=result.p.dtype, device=result.p.device,
        )
        return cov

    # Default: use MAF (allele frequency) from the scan result
    af = result.af
    maf = torch.min(af, 1.0 - af)
    return maf


def _load_gene_map(gene_map_file: str, snp_ids: list[str]) -> list[str]:
    """Load SNP→gene mapping from a TSV file (columns: SNP, GENE)."""
    import pandas as pd

    df = pd.read_csv(gene_map_file, sep="\t")
    gene_map = dict(zip(df["SNP"], df["GENE"]))
    return [gene_map.get(s, "unknown") for s in snp_ids]


# ---------------------------------------------------------------------------
# annotate — NCBI gene annotation for GWAS hit SNPs
# ---------------------------------------------------------------------------


def _add_annotate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "annotate",
        help="Annotate GWAS hit SNPs with nearby genes via NCBI Datasets",
    )
    p.add_argument("--sumstats", required=True, help="Summary statistics TSV/CSV")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--crop", help="Common or scientific name, e.g. 'maize' or 'Zea mays'")
    g.add_argument("--assembly", help="NCBI assembly accession, e.g. GCF_902167145.1")
    p.add_argument(
        "--taxid",
        type=int,
        help="Taxid (required when --assembly is used without --crop)",
    )
    p.add_argument("--window-up", type=int, default=50_000, help="Upstream window bp")
    p.add_argument("--window-down", type=int, default=50_000, help="Downstream window bp")
    p.add_argument("--p-threshold", type=float, default=5e-8, help="Hit p-value cutoff")
    p.add_argument("--no-go", action="store_true", help="Skip GO term lookup")
    p.add_argument("--include-orthologs", action="store_true", help="Fetch orthologs (slow)")
    p.add_argument(
        "--ortholog-taxa",
        default=None,
        help="Comma-separated taxids to filter orthologs (e.g. 9606,3702)",
    )
    p.add_argument("--api-key", default=None, help="NCBI API key (else NCBI_API_KEY env)")
    p.add_argument("--output", default="torchgwas_annotated", help="Output file prefix")


def _cmd_annotate(args: argparse.Namespace) -> int:
    """Annotate hit SNPs via NCBI."""
    from .annotate import NCBIClient, annotate_hits
    from .postgwas import load_sumstats

    ss = load_sumstats(args.sumstats)

    ortholog_taxa = None
    if args.ortholog_taxa:
        ortholog_taxa = [int(x) for x in args.ortholog_taxa.split(",") if x.strip()]

    client = NCBIClient(api_key=args.api_key)

    result = annotate_hits(
        ss,
        crop=args.crop,
        taxid=args.taxid,
        assembly=args.assembly,
        window_upstream_bp=args.window_up,
        window_downstream_bp=args.window_down,
        p_threshold=args.p_threshold,
        include_go=not args.no_go,
        include_orthologs=args.include_orthologs,
        ortholog_taxa=ortholog_taxa,
        client=client,
    )

    out_path = f"{args.output}.genes.tsv"
    result.to_tsv(out_path)
    n_hits = len(result.hits)
    n_genes = sum(len(h.genes) for h in result.hits)
    logger.info(
        "annotate: %d hits, %d genes -> %s (assembly=%s)",
        n_hits,
        n_genes,
        out_path,
        result.assembly_accession,
    )
    print(f"Wrote {n_hits} hits / {n_genes} gene rows to {out_path}")
    return 0


def _add_mediate_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "mediate",
        help="GRM-corrected single-triple causal mediation (SNP -> M -> Y)",
    )
    p.add_argument("--y", required=True, help="Phenotype vector (.npy/.pt/.tsv)")
    p.add_argument("--snp", required=True, help="SNP dosage vector (.npy/.pt/.tsv)")
    p.add_argument("--mediator", required=True, help="Mediator vector (.npy/.pt/.tsv)")
    p.add_argument("--kinship", required=True, help="GRM (n,n) (.npy/.pt)")
    p.add_argument("--covariates", default=None, help="Covariates (n,c) (.npy/.pt/.tsv)")
    p.add_argument("--se", choices=["sobel", "monte-carlo", "bootstrap"], default="monte-carlo")
    p.add_argument("--n-mc", type=int, default=10_000)
    p.add_argument("--n-boot", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-sensitivity", action="store_true", help="Skip Imai rho")
    p.add_argument("--output", default="mediation.json", help="Output JSON path")


def _add_mediate_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "mediate-scan",
        help="Genome x molecular-feature mediation scan with cis-window filtering",
    )
    p.add_argument("--y", required=True, help="Phenotype vector (.npy/.pt/.tsv)")
    p.add_argument("--genotype", required=True, help="Genotype matrix (n,s) (.npy/.pt)")
    p.add_argument("--mediator-matrix", required=True, help="Mediator matrix (n,f) (.npy/.pt)")
    p.add_argument("--kinship", required=True, help="GRM (n,n) (.npy/.pt)")
    p.add_argument("--snp-meta", default=None,
                   help="TSV with columns: id,chrom,pos (one row per SNP)")
    p.add_argument("--feature-meta", default=None,
                   help="TSV with columns: id,chrom,pos (one row per feature)")
    p.add_argument("--covariates", default=None, help="Covariates (.npy/.pt/.tsv)")
    p.add_argument("--cis-window", type=int, default=1_000_000,
                   help="Cis window in bp; -1 means no filter (all pairs)")
    p.add_argument("--se", choices=["sobel", "monte-carlo", "bootstrap"], default="monte-carlo")
    p.add_argument("--n-mc", type=int, default=10_000)
    p.add_argument("--fdr", choices=["bh", "by", "storey"], default="bh")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default="mediate_scan", help="Output prefix (.tsv + .top.tsv)")


def _load_array(path: str):
    """Load .npy / .pt / .tsv into a torch tensor (float64, CPU)."""
    import os

    import numpy as np
    import torch

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".pt", ".pth"):
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, torch.Tensor):
            return arr.double().cpu()
        arr = np.asarray(arr)
    elif ext in (".tsv", ".csv", ".txt"):
        import pandas as pd
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        arr = df.to_numpy()
        if arr.shape[1] == 1:
            arr = arr.ravel()
    else:
        raise ValueError(f"Unsupported extension {ext!r} for {path}.")
    return torch.as_tensor(arr, dtype=torch.float64)


def _cmd_mediate(args: argparse.Namespace) -> int:
    """Single-triple mediation."""
    import json

    from .multiomics import mediate_lmm

    Y = _load_array(args.y)
    snp = _load_array(args.snp)
    M = _load_array(args.mediator)
    K = _load_array(args.kinship)
    cov = _load_array(args.covariates) if args.covariates else None

    res = mediate_lmm(
        Y, snp, M, K,
        covariates=cov,
        se=args.se,
        n_mc_draws=args.n_mc,
        n_boot=args.n_boot,
        sensitivity=not args.no_sensitivity,
        seed=args.seed,
    )

    import math as _math

    def _jsonable(v):
        if v is None:
            return None
        if isinstance(v, float) and not _math.isfinite(v):
            return None
        return v

    payload = {k: _jsonable(v) for k, v in res.to_dict().items()}
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote mediation result to {args.output}")
    print(
        f"  indirect = {res.indirect:.4g} (95% CI [{res.indirect_ci_lower:.4g}, "
        f"{res.indirect_ci_upper:.4g}], p = {res.indirect_pvalue:.4g})"
    )
    if res.sensitivity_rho is not None:
        print(f"  sensitivity rho* = {res.sensitivity_rho:.3f}")
    return 0


def _cmd_mediate_scan(args: argparse.Namespace) -> int:
    """Genome x feature mediation scan."""
    import pandas as pd

    from .multiomics import scan_mediation

    Y = _load_array(args.y)
    G = _load_array(args.genotype)
    M = _load_array(args.mediator_matrix)
    K = _load_array(args.kinship)
    cov = _load_array(args.covariates) if args.covariates else None

    snp_ids = snp_chrom = snp_pos = None
    if args.snp_meta:
        meta = pd.read_csv(args.snp_meta, sep="\t")
        snp_ids = meta["id"].astype(str).tolist()
        snp_chrom = meta["chrom"].astype(str).tolist()
        snp_pos = meta["pos"].to_numpy()

    feat_ids = feat_chrom = feat_pos = None
    if args.feature_meta:
        meta = pd.read_csv(args.feature_meta, sep="\t")
        feat_ids = meta["id"].astype(str).tolist()
        feat_chrom = meta["chrom"].astype(str).tolist()
        feat_pos = meta["pos"].to_numpy()

    cis_w = None if args.cis_window < 0 else args.cis_window
    result = scan_mediation(
        Y, G, M, K,
        snp_ids=snp_ids, feature_ids=feat_ids,
        snp_pos=snp_pos, feature_pos=feat_pos,
        snp_chrom=snp_chrom, feature_chrom=feat_chrom,
        cis_window_bp=cis_w,
        covariates=cov,
        se=args.se, n_mc_draws=args.n_mc,
        fdr_method=args.fdr,
        seed=args.seed,
    )

    flat_path = f"{args.output}.tsv"
    top_path = f"{args.output}.top.tsv"
    result.to_tsv(flat_path)
    top = result.top_hits(0.05)
    top.to_csv(top_path, sep="\t", index=False)
    print(f"Wrote {result.n_pairs} pairs to {flat_path}; {len(top)} BH-significant to {top_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
