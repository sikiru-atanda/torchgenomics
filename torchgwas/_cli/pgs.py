"""CLI handlers and parsers for polygenic score commands."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("torchgwas")


def _cmd_pgs_fit(args: argparse.Namespace) -> int:
    """Fit PGS weights from GWAS sumstats."""
    from ..pgs import (
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

    from ..io.detect import detect_format
    from ..io.validate import _open_reader
    from ..pgs import PGSResult, score_individuals

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
    p.add_argument("--clump-r2", type=float, default=0.1, help="C+T r2 threshold")
    p.add_argument("--clump-p", type=float, default=5e-8, help="C+T p threshold")
    p.add_argument("--clump-kb", type=float, default=250.0, help="C+T window in kb")
    p.add_argument("--maf-tol", type=float, default=0.20, help="MAF mismatch tolerance")
    p.add_argument("--palindromic-maf-tol", type=float, default=0.42,
                   help="Palindromic SNP MAF disambiguation cutoff")
    p.add_argument("--keep-ambiguous", action="store_true",
                   help="Keep palindromic SNPs when MAF permits (default: drop)")
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
