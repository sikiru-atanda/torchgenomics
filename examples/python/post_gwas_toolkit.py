"""Post-GWAS Analysis Toolkit — a runnable tour of ``torchgenomics.api``'s
post-GWAS facade on tiny synthetic fixtures.

Every example below actually executes (no pseudo-code): each section builds
a small in-memory / on-disk fixture, calls the real ``torchgenomics.api``
function, and prints a short labeled summary of the result. Two functions
(``ldsc`` / ``ldsc_rg``) are only reachable through the ``torchgenomics.api``
module directly (they are "tier-2" CLI-bridge subcommands, not re-exported at
the top-level ``torchgenomics`` package) — see the LDSC section below for
why. One function (``annotate_hits``) needs live NCBI network access and is
shown as reference-only, never executed.

Run from the repository root::

    PYTHONPATH=. TORCHGENOMICS_DISABLE_NATIVE=1 python examples/python/post_gwas_toolkit.py

(or ``python -m examples.python.post_gwas_toolkit``). With ``torchgenomics``
properly pip-installed, plain ``python examples/python/post_gwas_toolkit.py``
works too.

Runtime: a few seconds on CPU. All fixtures are tiny (<= 60 variants) and
live under a temporary directory that is cleaned up on exit.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import torchgenomics as tg
import torchgenomics.api as tg_api  # only needed for ldsc / ldsc_rg (see §5)

SEED = 42


# =============================================================================
# 1. SETUP — build every tiny synthetic fixture the sections below need.
# =============================================================================
#
# Column convention throughout (unless noted otherwise): the lowercase
# sumstats schema TorchGenomics' `postgwas.load_sumstats` expects —
# chr, pos, snp, a1, a2, beta, se, p, n, af. `finemap` is the one exception:
# it runs the SuSiE-RSS engine (`bayes-scan-rss`), which uses the uppercase
# SNP/CHR/BP/A1/A2/BETA/SE/N schema instead — see §3.


def _write_geno_csv(path: Path, snp, chrom, pos, a1, a2, G: np.ndarray, sample_prefix="IND") -> None:
    """Write a TorchGenomics "3-column rule" structured numeric-dosage CSV.

    Header: SNP (index), Chromosome, Position_BP, A1, A2, <sample columns>.
    `G` is (n_samples, m_variants); values are written as-is (continuous
    synthetic dosages — sufficient for exercising LD/r2 machinery on toy
    data; not meant to look like real 0/1/2 genotype calls).
    """
    n_samples = G.shape[0]
    sample_ids = [f"{sample_prefix}{i:03d}" for i in range(n_samples)]
    with open(path, "w") as f:
        f.write("SNP,Chromosome,Position_BP,A1,A2," + ",".join(sample_ids) + "\n")
        for j in range(len(snp)):
            row = [snp[j], str(chrom[j]), str(pos[j]), a1[j], a2[j]]
            row += [f"{v:.4f}" for v in G[:, j]]
            f.write(",".join(row) + "\n")


def build_clump_ldsc_fixture(tmp: Path, seed: int = SEED):
    """30 SNPs on chr1 in 3 LD blocks of 10; one strongly-significant
    "causal" SNP planted at the middle of each block. Used by §2 (clump)
    and §5 (ldsc / ldsc_rg), which both need a genotype panel to compute
    r2 / LD scores from.
    """
    rng = np.random.default_rng(seed)
    m, n_blocks, block_size, n_samples = 30, 3, 10, 200
    chrom = ["1"] * m
    pos = [1000 + i * 100 for i in range(m)]
    snp = [f"rs{i}" for i in range(m)]
    a1, a2 = ["A"] * m, ["G"] * m

    # Genotype panel: 3 independent LD blocks (shared founder + noise).
    G = np.zeros((n_samples, m))
    for b in range(n_blocks):
        lo, hi = b * block_size, (b + 1) * block_size
        base = rng.normal(size=n_samples)
        for j in range(lo, hi):
            G[:, j] = 0.85 * base + 0.4 * rng.normal(size=n_samples)

    # Sumstats: one planted causal SNP per block (p=1e-12); its blockmates
    # get p drawn from a moderately-significant range so LD-clumping has
    # real tag SNPs to merge; everything else is null.
    p = rng.uniform(0.2, 0.9, size=m)
    beta = rng.normal(0, 0.02, size=m)
    causal_idx = [b * block_size + block_size // 2 for b in range(n_blocks)]
    for idx in causal_idx:
        p[idx] = 1e-12
        beta[idx] = 0.6
        lo, hi = (idx // block_size) * block_size, (idx // block_size) * block_size + block_size
        for j in range(lo, hi):
            if j != idx:
                p[j] = rng.uniform(1e-4, 1e-2)

    df = pd.DataFrame({
        "chr": chrom, "pos": pos, "snp": snp, "a1": a1, "a2": a2,
        "beta": beta, "se": np.full(m, 0.05), "p": p,
        "n": np.full(m, 5000), "af": np.full(m, 0.3),
    })
    gwas_path = tmp / "gwas_clump_ldsc.tsv"
    df.to_csv(gwas_path, sep="\t", index=False)

    geno_path = tmp / "geno_clump_ldsc.csv"
    _write_geno_csv(geno_path, snp, chrom, pos, a1, a2, G)
    return gwas_path, geno_path


def build_power_wc_fixture(tmp: Path):
    """6 SNPs: two strong hits (large |beta|, tiny p — subject to winner's
    curse) plus four null/weak SNPs. Used by §6 (power) and §7 (winners_curse).
    """
    df = pd.DataFrame({
        "chr": [1] * 6, "pos": list(range(1, 7)), "snp": [f"rs{i}" for i in range(6)],
        "a1": ["A"] * 6, "a2": ["G"] * 6,
        "beta": [0.5, 0.01, 0.285, 0.02, -0.01, 0.55],
        "se": [0.05] * 6,
        "p": [1e-20, 0.8, 1e-20, 0.7, 0.9, 1e-18],
        "n": [10000] * 6, "af": [0.3] * 6,
    })
    path = tmp / "gwas_power_wc.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


def build_finemap_fixture(tmp: Path, p: int = 5):
    """Uppercase-schema sumstats (SNP/CHR/BP/A1/A2/BETA/SE/N) + a .pt LD
    reference carrying matching SNP ids, via `save_ld_reference` +
    `LDReferenceMetadata` — exactly the pattern `bayes-scan-rss` (and
    `api.finemap`, its friendly front-end) expects.
    """
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata

    ss_path = tmp / "finemap_sumstats.tsv"
    ss = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(p)], "CHR": [22] * p,
        "BP": [1000 + i * 10 for i in range(p)], "A1": ["A"] * p, "A2": ["G"] * p,
        "BETA": [0.05, 0.03, 0.6, 0.02, 0.04][:p], "SE": [0.05, 0.04, 0.05, 0.04, 0.05][:p],
        "N": [1000] * p,
    })
    ss.to_csv(ss_path, sep="\t", index=False)

    ld_path = tmp / "finemap_ld.pt"
    R = torch.eye(p, dtype=torch.float64)
    meta = LDReferenceMetadata(cohort_id="vignette", n=1000, build="GRCh38",
                               panel_provenance="synthetic")
    save_ld_reference(ld_path, R, ss["SNP"].tolist(), meta)
    return ss_path, ld_path


def build_hess_fixture(tmp: Path, seed: int = 0):
    """40 SNPs / 4 contiguous 10-SNP regions on chr1; region 0 carries a
    heritability signal. LD matrix is the identity (uncorrelated toy LD —
    keeps the region-level h2 solve numerically simple). Used by §4 (hess),
    both the h2 mode and the local-rg mode (gwas2=same file).
    """
    rng = np.random.default_rng(seed)
    m = 40
    pos = list(range(1, m + 1))
    z = rng.normal(0, 1, size=m)
    z[0:10] += 3.0  # region 1 (SNPs 0-9) carries the signal
    df = pd.DataFrame({
        "chr": [1] * m, "pos": pos, "snp": [f"rs{i}" for i in range(m)],
        "a1": ["A"] * m, "a2": ["G"] * m,
        "beta": [z[i] * 0.02 for i in range(m)], "se": [0.02] * m,
        "p": [0.01] * m, "n": [10000] * m, "af": [0.3] * m,
    })
    gwas_path = tmp / "gwas_hess.tsv"
    df.to_csv(gwas_path, sep="\t", index=False)

    ld_path = tmp / "ld_hess.npy"
    np.save(ld_path, np.eye(m))

    regions_path = tmp / "regions_hess.tsv"
    pd.DataFrame({"chrom": [1, 1, 1, 1], "start": [1, 11, 21, 31],
                 "end": [10, 20, 30, 40]}).to_csv(regions_path, sep="\t", index=False)
    return gwas_path, ld_path, regions_path


def build_coloc_meta_fixture(tmp: Path, seed: int = 1, n_snp: int = 50):
    """Three sumstats (a, b, c) over the same 50-SNP region, all sharing one
    strong causal SNP at index 25. `a`+`b` double as §8's coloc pair AND
    §9's meta-analysis pair (two "studies" of the same trait, same SNP set —
    exactly what fixed/random-effect meta-analysis assumes); `a`+`b`+`c`
    together are the hyprcoloc (>=2 trait) fixture.
    """
    rng = np.random.default_rng(seed)
    causal = 25

    def _one():
        z = rng.normal(0, 1, size=n_snp)
        z[causal] = 6.0
        beta = z * 0.05
        se = np.full(n_snp, 0.05)
        p = 2 * norm.sf(np.abs(z))
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp, "a2": ["G"] * n_snp,
            "beta": beta, "se": se, "p": p, "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })

    a_path, b_path, c_path = tmp / "sumstats_a.tsv", tmp / "sumstats_b.tsv", tmp / "sumstats_c.tsv"
    _one().to_csv(a_path, sep="\t", index=False)
    _one().to_csv(b_path, sep="\t", index=False)
    _one().to_csv(c_path, sep="\t", index=False)
    return a_path, b_path, c_path


def build_mr_fixture(tmp: Path, seed: int = 0, effect: float = 0.5, n_snp: int = 40):
    """Two-sample MR fixture: `n_snp` instruments with a true causal
    effect (`outcome = effect * exposure + noise`) so IVW/Egger/weighted-
    median/MR-PRESSO all have a real signal to recover. Used by §10 (mr).
    """
    rng = np.random.default_rng(seed)
    bx = rng.normal(0.3, 0.1, size=n_snp)
    bx_se = np.full(n_snp, 0.02)
    by = effect * bx + rng.normal(0, 0.01, size=n_snp)
    by_se = np.full(n_snp, 0.02)

    def _df(beta, se):
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp, "a2": ["G"] * n_snp,
            "beta": beta, "se": se, "p": [2 * (1 - 0.9999)] * n_snp,
            "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })

    exp_path, out_path = tmp / "mr_exposure.tsv", tmp / "mr_outcome.tsv"
    _df(bx, bx_se).to_csv(exp_path, sep="\t", index=False)
    _df(by, by_se).to_csv(out_path, sep="\t", index=False)
    return exp_path, out_path, effect


def build_mrmega_fixture(tmp: Path, n_pops: int = 4, n_snp: int = 30, seed: int = 1):
    """`n_pops` ancestry sumstats over the same SNPs sharing a common causal
    effect plus ancestry-specific perturbation. Used by §11 (mr_mega,
    n_axes=1 needs >= 3 populations).
    """
    rng = np.random.default_rng(seed)
    true_beta = rng.normal(0.2, 0.05, size=n_snp)
    paths = []
    for pop in range(n_pops):
        beta = true_beta + rng.normal(0, 0.02, size=n_snp)
        df = pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp, "a2": ["G"] * n_snp,
            "beta": beta, "se": [0.02] * n_snp, "p": [0.001] * n_snp,
            "n": [5000] * n_snp, "af": [0.3] * n_snp,
        })
        fp = tmp / f"mrmega_pop{pop}.tsv"
        df.to_csv(fp, sep="\t", index=False)
        paths.append(fp)
    return paths


def build_smr_fixture(tmp: Path, seed: int = 0, n_snp: int = 30, n_genes: int = 3):
    """GWAS + eQTL sumstats sharing a cis causal SNP per gene, plus a long
    gene->cis-SNP map TSV (gene, snp). Used by §12 (smr).
    """
    rng = np.random.default_rng(seed)

    def _row(i):
        return {"chr": 1, "pos": i + 1, "snp": f"rs{i}", "a1": "A", "a2": "G", "n": 5000, "af": 0.3}

    gwas_rows, eqtl_rows, map_rows = [], [], []
    per = n_snp // n_genes
    for g in range(n_genes):
        causal = g * per + 1
        for j in range(per):
            i = g * per + j
            z_e = 6.0 if i == causal else rng.normal()
            z_g = 5.0 if i == causal else rng.normal()  # shared signal at the causal cis-SNP
            gwas_rows.append({**_row(i), "beta": z_g * 0.05, "se": 0.05, "p": float(2 * norm.sf(abs(z_g)))})
            eqtl_rows.append({**_row(i), "beta": z_e * 0.05, "se": 0.05, "p": float(2 * norm.sf(abs(z_e)))})
            map_rows.append({"gene": f"gene{g}", "snp": f"rs{i}"})

    gwas_path, eqtl_path, map_path = tmp / "smr_gwas.tsv", tmp / "smr_eqtl.tsv", tmp / "smr_gene_map.tsv"
    pd.DataFrame(gwas_rows).to_csv(gwas_path, sep="\t", index=False)
    pd.DataFrame(eqtl_rows).to_csv(eqtl_path, sep="\t", index=False)
    pd.DataFrame(map_rows).to_csv(map_path, sep="\t", index=False)
    return gwas_path, eqtl_path, map_path


def build_enrichment_fixture(tmp: Path, seed: int = 0):
    """6 genes x 10 SNPs; genes g0/g1 carry a strong signal, g2..g5 are
    null. Gene-set "hot" = {g0, g1} (enriched), "null" = {g2, g3, g4, g5}.
    Used by §13 (gene_set_enrichment).
    """
    rng = np.random.default_rng(seed)
    rows, ann = [], []
    for gi in range(6):
        strong = gi < 2
        gene = f"g{gi}"
        for j in range(10):
            i = gi * 10 + j
            z = rng.normal(4.0 if strong else 0.0, 1.0)
            rows.append({"chr": 1, "pos": gi * 1000 + j, "snp": f"rs{i}", "a1": "A", "a2": "G",
                        "beta": z * 0.05, "se": 0.05, "p": float(2 * norm.sf(abs(z))),
                        "n": 5000, "af": 0.3})
        ann.append({"gene": gene, "chr": 1, "start": gi * 1000, "end": gi * 1000 + 9})

    gwas_path, ann_path, sets_path = tmp / "enrich_gwas.tsv", tmp / "enrich_genes.tsv", tmp / "enrich_sets.tsv"
    pd.DataFrame(rows).to_csv(gwas_path, sep="\t", index=False)
    pd.DataFrame(ann).to_csv(ann_path, sep="\t", index=False)
    pd.DataFrame({
        "set": ["hot", "hot", "null", "null", "null", "null"],
        "gene": ["g0", "g1", "g2", "g3", "g4", "g5"],
    }).to_csv(sets_path, sep="\t", index=False)
    return gwas_path, ann_path, sets_path


def build_pgs_fixture(tmp: Path, m: int = 12, n_obs: int = 2000, n_ld: int = 300, seed: int = 0):
    """PGS sumstats (m SNPs) + a `torchgenomics.pgs.ld_ref` LD reference
    (a DIFFERENT .pt format from the postgwas fine-mapping one used in
    §3 — pgs_fit's LD reference is built by `pgs.ld_ref.build_ld_reference`,
    not `postgwas._ld_ref_loader`) + a target genotype CSV panel over the
    same SNP set for §15 (pgs_score). Used by §14 (pgs_fit / pgs_score).
    """
    g = torch.Generator().manual_seed(seed)
    betas = 0.1 * torch.randn(m, generator=g, dtype=torch.float64)
    ses = torch.full((m,), 1.0 / math.sqrt(n_obs), dtype=torch.float64)
    zs = betas / ses
    ps = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(zs.abs() / math.sqrt(2.0))))

    ss_path = tmp / "pgs_sumstats.tsv"
    with open(ss_path, "w") as f:
        f.write("chr\tpos\tsnp\ta1\ta2\tbeta\tse\tp\tn\n")
        for i in range(m):
            f.write(f"1\t{100 + i}\trs{i}\tA\tG\t{float(betas[i]):.6f}\t"
                    f"{float(ses[i]):.6f}\t{float(ps[i]):.6e}\t{n_obs}\n")

    from torchgenomics.pgs.ld_ref import build_ld_reference
    from torchgenomics.pgs.ld_ref import save_ld_reference as pgs_save_ld_reference

    g2 = torch.Generator().manual_seed(seed + 1)
    G_ld = torch.randn(n_ld, m, generator=g2, dtype=torch.float64)
    G_ld = (G_ld - G_ld.mean(0, keepdim=True)) / G_ld.std(0, keepdim=True).clamp(min=1e-12)
    ld_meta = {
        "snp": [f"rs{i}" for i in range(m)], "chr": ["1"] * m,
        "pos": list(range(100, 100 + m)), "a1": ["A"] * m, "a2": ["G"] * m,
    }
    ld = build_ld_reference(G_ld, ld_meta, mode="full")
    ld_path = tmp / "pgs_ld_ref.pt"
    pgs_save_ld_reference(ld, str(ld_path))

    # Target genotype panel for pgs_score: same SNP ids/positions/alleles
    # (so score_individuals's allele-harmonization is a straight match), a
    # fresh set of "individuals" to be scored.
    rng = np.random.default_rng(seed + 2)
    n_target = 50
    G_tgt = rng.normal(size=(n_target, m))
    geno_path = tmp / "pgs_target_geno.csv"
    _write_geno_csv(
        geno_path, [f"rs{i}" for i in range(m)], ["1"] * m, list(range(100, 100 + m)),
        ["A"] * m, ["G"] * m, G_tgt, sample_prefix="TGT",
    )
    return ss_path, ld_path, geno_path


def main() -> None:
    executed: list[str] = []
    reference_only: list[str] = []

    tmp = Path(tempfile.mkdtemp(prefix="tg_postgwas_vignette_"))
    print(f"Scratch fixtures dir: {tmp}\n")

    try:
        gwas_clump_ldsc, geno_clump_ldsc = build_clump_ldsc_fixture(tmp)
        gwas_power_wc = build_power_wc_fixture(tmp)
        finemap_ss, finemap_ld = build_finemap_fixture(tmp)
        gwas_hess, ld_hess, regions_hess = build_hess_fixture(tmp)
        sumstats_a, sumstats_b, sumstats_c = build_coloc_meta_fixture(tmp)
        mr_exposure, mr_outcome, mr_true_effect = build_mr_fixture(tmp)
        mrmega_paths = build_mrmega_fixture(tmp)
        smr_gwas, smr_eqtl, smr_gene_map = build_smr_fixture(tmp)
        enrich_gwas, enrich_genes, enrich_sets = build_enrichment_fixture(tmp)
        pgs_sumstats, pgs_ld_ref, pgs_target_geno = build_pgs_fixture(tmp)

        # ---------------------------------------------------------------
        # 2. clump — LD clumping to independent index variants.
        # ---------------------------------------------------------------
        print("=== 2. tg.clump — LD clumping ===")
        r = tg.clump(
            sumstats=str(gwas_clump_ldsc), genotype=str(geno_clump_ldsc),
            output=str(tmp / "clump_run"), p_threshold=1e-3, r2=0.1, window_kb=250.0,
        )
        print(r.summary())
        print(r.clumps.to_string(index=False))
        executed.append("clump")

        # ---------------------------------------------------------------
        # 3. finemap — SuSiE-RSS fine-mapping (credible sets).
        # ---------------------------------------------------------------
        print("\n=== 3. tg.finemap — SuSiE-RSS fine-mapping ===")
        r = tg.finemap(
            str(finemap_ss), str(finemap_ld), max_num_causal=2, threads=1,
            output=str(tmp / "finemap_out.tsv"),
        )
        print(r.summary())
        print(r.credible_sets.to_string(index=False))
        executed.append("finemap")

        # ---------------------------------------------------------------
        # 4. hess — local heritability, and local genetic correlation.
        # ---------------------------------------------------------------
        print("\n=== 4. tg.hess — local heritability (h2 mode) ===")
        r = tg.hess(str(gwas_hess), str(ld_hess), str(regions_hess),
                    output=str(tmp / "hess_h2.tsv"))
        print(r.summary())
        print(r.results.to_string(index=False))
        executed.append("hess (h2 mode)")

        print("\n=== 4b. tg.hess — local genetic correlation (rg mode, gwas2=gwas) ===")
        r = tg.hess(str(gwas_hess), str(ld_hess), str(regions_hess), gwas2=str(gwas_hess))
        print(r.summary())
        executed.append("hess (rg mode)")

        # ---------------------------------------------------------------
        # 5. ldsc / ldsc_rg — only reachable via `torchgenomics.api`
        # (tier-2 CLI-bridge subcommands, not re-exported at the
        # top-level `torchgenomics` package — see `torchgenomics/__init__.py`
        # vs. `torchgenomics/api/__init__.py`'s `__getattr__`).
        # Reuses the §2 genotype panel to compute LD scores in-process.
        # CAVEAT: with only 30 SNPs the block-jackknife falls back to a
        # single block (n_blocks=200 > m), so the printed SE is NaN — this
        # section demonstrates the call mechanics on toy data, not a
        # statistically meaningful h2/rg estimate (real runs use
        # thousands of SNPs across hundreds of jackknife blocks).
        # ---------------------------------------------------------------
        print("\n=== 5. torchgenomics.api.ldsc — SNP heritability via LD score regression ===")
        r = tg_api.ldsc(
            sumstats=str(gwas_clump_ldsc), genotype=str(geno_clump_ldsc),
            output=str(tmp / "ldsc_run"),
        )
        print(r.summary())
        print(open(str(tmp / "ldsc_run.ldsc.txt")).read())
        executed.append("ldsc")

        print("=== 5b. torchgenomics.api.ldsc_rg — genetic correlation (trait vs itself) ===")
        r = tg_api.ldsc_rg(
            sumstats1=str(gwas_clump_ldsc), sumstats2=str(gwas_clump_ldsc),
            genotype=str(geno_clump_ldsc), output=str(tmp / "ldsc_rg_run"),
        )
        print(r.summary())
        print(open(str(tmp / "ldsc_rg_run.ldsc_rg.txt")).read())
        executed.append("ldsc_rg")

        # ---------------------------------------------------------------
        # 6. power — per-variant detection power + power curve.
        # ---------------------------------------------------------------
        print("=== 6. tg.power — GWAS detection power ===")
        r = tg.power(str(gwas_power_wc), power_curve=True, output=str(tmp / "power_out.tsv"))
        print(r.summary())
        print(r.results.to_string(index=False))
        executed.append("power")

        # ---------------------------------------------------------------
        # 7. winners_curse — effect-size de-biasing (all three methods).
        # ---------------------------------------------------------------
        print("\n=== 7. tg.winners_curse — winner's-curse correction (all 3 methods) ===")
        for method in ("conditional_likelihood", "fiqt", "bootstrap"):
            kw = {"n_boot": 200, "seed": 1} if method == "bootstrap" else {}
            r = tg.winners_curse(str(gwas_power_wc), method=method, **kw)
            print(f"  [{method}] {r.summary()}")
            executed.append(f"winners_curse ({method})")

        # ---------------------------------------------------------------
        # 8. meta — fixed-effect meta-analysis of two "studies".
        # ---------------------------------------------------------------
        print("\n=== 8. tg.meta — fixed-effect meta-analysis (2 studies) ===")
        r = tg.meta([str(sumstats_a), str(sumstats_b)], method="fixed",
                    output=str(tmp / "meta_run"))
        print(r.summary())
        executed.append("meta")

        # ---------------------------------------------------------------
        # 9. mr_mega — multi-ancestry meta-regression.
        # ---------------------------------------------------------------
        print("\n=== 9. tg.mr_mega — multi-ancestry meta-regression (4 populations, n_axes=1) ===")
        r = tg.mr_mega([str(p) for p in mrmega_paths], n_axes=1, output=str(tmp / "mrmega_out.tsv"))
        print(r.summary())
        executed.append("mr_mega")

        # ---------------------------------------------------------------
        # 10. coloc — pairwise Giambartolomei coloc + hyprcoloc.
        # ---------------------------------------------------------------
        print("\n=== 10. tg.coloc — pairwise colocalization (shared causal SNP) ===")
        r = tg.coloc(str(sumstats_a), str(sumstats_b), method="pairwise",
                    output=str(tmp / "coloc_pairwise.tsv"))
        print(r.summary())
        executed.append("coloc (pairwise)")

        print("=== 10b. tg.coloc — hyprcoloc (3 traits) ===")
        r = tg.coloc([str(sumstats_a), str(sumstats_b), str(sumstats_c)], method="hyprcoloc")
        print(r.summary())
        executed.append("coloc (hyprcoloc)")

        # ---------------------------------------------------------------
        # 11. mr — two-sample Mendelian randomization (all methods).
        # ---------------------------------------------------------------
        print("\n=== 11. tg.mr — two-sample MR (method='all': IVW/Egger/weighted-median/PRESSO) ===")
        r = tg.mr(str(mr_exposure), str(mr_outcome), method="all", n_boot=200, n_perm=200,
                  seed=42, output=str(tmp / "mr_out.tsv"))
        print(r.summary())
        print(r.results.to_string(index=False))
        print(f"  (simulated causal effect = {mr_true_effect})")
        executed.append("mr")

        # ---------------------------------------------------------------
        # 12. smr — SMR + HEIDI.
        # ---------------------------------------------------------------
        print("\n=== 12. tg.smr — SMR + HEIDI (expression-mediation test) ===")
        r = tg.smr(str(smr_gwas), str(smr_eqtl), str(smr_gene_map), output=str(tmp / "smr_out.tsv"))
        print(r.summary())
        print(r.results.to_string(index=False))
        executed.append("smr")

        # ---------------------------------------------------------------
        # 13. gene_set_enrichment — MAGMA-style competitive enrichment.
        # ---------------------------------------------------------------
        print("\n=== 13. tg.gene_set_enrichment — MAGMA-style gene-set enrichment ===")
        r = tg.gene_set_enrichment(str(enrich_gwas), str(enrich_genes), str(enrich_sets),
                                   output=str(tmp / "enrich_out.tsv"))
        print(r.summary())
        print(r.results.to_string(index=False))
        executed.append("gene_set_enrichment")

        # ---------------------------------------------------------------
        # 14. pgs_fit / pgs_score — C+T PGS weights, applied to target genotypes.
        # ---------------------------------------------------------------
        print("\n=== 14. tg.pgs_fit — PGS weights (method='ct', fast for a toy fixture) ===")
        r = tg.pgs_fit(str(pgs_sumstats), str(pgs_ld_ref), str(tmp / "pgs_weights.tsv"),
                       method="ct", clump_p=0.5, clump_r2=0.9, device="cpu")
        print(r.summary())
        executed.append("pgs_fit")

        print("=== 14b. tg.pgs_score — apply weights to 50 target individuals ===")
        r = tg.pgs_score(str(pgs_target_geno), str(tmp / "pgs_weights.tsv"),
                         str(tmp / "pgs_scores.tsv"), device="cpu")
        print(r.summary())
        executed.append("pgs_score")

        # ---------------------------------------------------------------
        # 15. annotate_hits — NCBI network required — NOT executed.
        # ---------------------------------------------------------------
        print("\n=== 15. tg.annotate_hits — network-required, NOT executed ===")
        print("  Call shape (reference only; needs live NCBI Datasets/E-utilities access):")
        print("    r = tg.annotate_hits(str(gwas_power_wc), crop=\"maize\", p_threshold=5e-8,")
        print("                          window_up=50_000, window_down=50_000,")
        print("                          output=str(tmp / \"annotate_out\"))")
        if False:  # pragma: no cover — reference only, never executed
            r = tg.annotate_hits(
                str(gwas_power_wc), crop="maize", p_threshold=5e-8,
                window_up=50_000, window_down=50_000, output=str(tmp / "annotate_out"),
            )
            print(r.summary())
        reference_only.append("annotate_hits (network-required)")

        # ---------------------------------------------------------------
        # Wrap-up.
        # ---------------------------------------------------------------
        print("\n" + "=" * 72)
        print(f"EXECUTED ({len(executed)}): " + ", ".join(executed))
        print(f"REFERENCE-ONLY ({len(reference_only)}): " + ", ".join(reference_only))
        print("=" * 72)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
