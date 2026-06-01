"""Simulate a minimal PrediXcan-compatible fixture for the MetaXcan harness.

Why simulate (not download a real GTEx PrediXcan model):

- GTEx v8 PrediXcan model bundles are 1-2 GB tarballs. Network-staging them
  at every harness rerun is wasteful and brittle (the underlying URL has
  moved several times across PredictDB/Zenodo/Box releases).
- The S-PrediXcan estimator (Barbeira et al. 2018, Genome Biol 19:74,
  Eq. 4) is summary-statistic-only: given gene weights w and a SNP LD
  matrix Sigma, the test statistic
      z_g = (w^T D z) / sqrt(w^T Sigma w)
  (where D is the per-SNP standardization factor) is determined by w,
  Sigma, and the GWAS z-scores. No biological signal in the model is
  required for the numerical-equivalence check.
- The simulated fixture exercises every code path of SPrediXcan.py that
  consumes the on-disk sumstats + .db + covariance triple, which is what
  twas_sumstat targets. This is the same simulation-first strategy used
  by validation/external/twosamplemr/ (see simulate_mr.R) and validated
  by the TwoSampleMR + MRPRESSO harness.

This script produces, under --output-dir <data>:
  - model.db                  SQLite with `weights` + `extra` tables in the
                              PrediXcan schema (rsid, gene, weight,
                              ref_allele, eff_allele; gene, genename,
                              n.snps.in.model, pred.perf.R2 / pval / qval).
  - model.txt.gz              MetaXcan covariance file (header GENE RSID1
                              RSID2 VALUE, whitespace-delimited).
  - gwas_sumstats.txt.gz      GWAS sumstats with z-scores (header SNP
                              effect_allele non_effect_allele zscore beta
                              se pvalue).
  - sim_truth.json            seed / theta_true / per-gene true beta /
                              SHA256 of every fixture file.

Seed-deterministic so reruns are bit-identical.

Citations:
  Barbeira AN, ..., Im HK (2018). "Exploring the phenotypic consequences
  of tissue specific gene expression variation inferred from GWAS summary
  statistics." Nature Communications 9:1825. DOI 10.1038/s41467-018-03621-1
  (S-PrediXcan estimator).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of genes in the fixture. Five is small enough to keep the run fast
# but large enough to test multi-gene aggregation in compare.py.
N_GENES = 5
# Number of cis-SNPs per gene. The S-PrediXcan estimator is non-degenerate
# only when there are >=2 SNPs with non-zero weights per gene (otherwise
# w^T Sigma w collapses to a single variance).
SNPS_PER_GENE = 8
# Number of background (non-cis) SNPs to add to the GWAS sumstats so that
# SPrediXcan's SNP filter exercises its index-intersection path.
N_BACKGROUND_SNPS = 50
# GWAS sample size (used to convert beta/se to z-scores). 10000 is the
# canonical PrediXcan example value.
GWAS_N = 10000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate(output_dir: Path, seed: int = 42) -> dict:
    """Generate the simulated fixture and return a manifest dict."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- per-gene SNP universe -----
    # Use rs IDs in a deterministic format. We bake the gene index into the
    # rsid namespace so cis-SNPs do not collide across genes.
    snp_universe = []
    gene_snps: dict[str, list[str]] = {}
    gene_alleles: dict[str, list[tuple[str, str]]] = {}
    gene_weights: dict[str, np.ndarray] = {}
    gene_truth_beta: dict[str, float] = {}
    # Sample reasonable allele frequencies and weights.
    for g in range(N_GENES):
        gene = f"ENSG00000{1000 + g:05d}"
        snps = [f"rs{g * 1000 + j}" for j in range(SNPS_PER_GENE)]
        gene_snps[gene] = snps
        # Each SNP: pick effect and ref allele.
        alleles = [("A", "G") if (j % 2 == 0) else ("C", "T") for j in range(SNPS_PER_GENE)]
        gene_alleles[gene] = alleles
        # Weights ~ N(0, 0.3) with sparsity at 30%.
        w = rng.normal(loc=0.0, scale=0.3, size=SNPS_PER_GENE)
        sparsity = rng.random(SNPS_PER_GENE) < 0.3
        w[sparsity] = 0.0
        # Renormalize so weights are non-trivial (avoid all-zero gene).
        if np.all(w == 0):
            w[0] = 0.5
        gene_weights[gene] = w
        # Per-gene true causal beta (the regression coefficient of phenotype
        # on the true GReX). Half are non-zero so compare.py has both null
        # and signal cases.
        gene_truth_beta[gene] = float(rng.choice([0.0, 0.25])) if g >= 2 else 0.0
        snp_universe.extend(snps)

    # Add background SNPs (not assigned to any gene). These appear in the
    # GWAS sumstats but are absent from the model.db, exercising SPrediXcan's
    # SNP intersection path.
    background = [f"rs_bg_{i}" for i in range(N_BACKGROUND_SNPS)]
    snp_universe.extend(background)
    snp_universe = list(dict.fromkeys(snp_universe))  # de-dupe, keep order

    # ----- per-gene LD matrix (used by S-PrediXcan) -----
    # We construct each gene's k x k SNP correlation matrix as a band matrix
    # with off-diagonal ~ rho. This gives a well-conditioned positive-definite
    # Sigma and is the simplest non-trivial test of the w^T Sigma w denominator.
    gene_ld: dict[str, np.ndarray] = {}
    for gene, snps in gene_snps.items():
        k = len(snps)
        rho = 0.3
        Sigma = np.full((k, k), 0.0)
        for i in range(k):
            for j in range(k):
                Sigma[i, j] = rho ** abs(i - j)
        # Float-precision symmetry.
        Sigma = 0.5 * (Sigma + Sigma.T)
        gene_ld[gene] = Sigma

    # ----- write model.db -----
    db_path = output_dir / "model.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE weights (rsid TEXT, gene TEXT, weight REAL, "
        "ref_allele TEXT, eff_allele TEXT)"
    )
    cur.execute(
        'CREATE TABLE extra (gene TEXT, genename TEXT, "n.snps.in.model" INTEGER, '
        '"pred.perf.R2" REAL, "pred.perf.pval" REAL, "pred.perf.qval" REAL)'
    )
    for gene, snps in gene_snps.items():
        w = gene_weights[gene]
        alleles = gene_alleles[gene]
        for j, snp in enumerate(snps):
            if w[j] == 0.0:
                continue
            eff, ref = alleles[j]  # alleles[j] = (effect_allele, ref/non-effect_allele)
            cur.execute(
                "INSERT INTO weights (rsid, gene, weight, ref_allele, eff_allele) "
                "VALUES (?, ?, ?, ?, ?)",
                (snp, gene, float(w[j]), ref, eff),
            )
        n_nonzero = int((w != 0.0).sum())
        cur.execute(
            'INSERT INTO extra (gene, genename, "n.snps.in.model", '
            '"pred.perf.R2", "pred.perf.pval", "pred.perf.qval") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            (gene, f"GENE_{gene[-5:]}", n_nonzero, 0.15, 1e-4, 1e-3),
        )
    conn.commit()
    conn.close()

    # ----- write covariance file (MetaXcan format) -----
    # MetaXcan covariance file format (metax/MatrixManager.py): one row per
    # (gene, rsid_i, rsid_j) triple with VALUE = Sigma[i,j]. We emit the
    # upper triangle (i<=j); MetaXcan symmetrizes internally.
    cov_path = output_dir / "model.txt.gz"
    with gzip.open(cov_path, "wt", encoding="utf-8") as f:
        f.write("GENE RSID1 RSID2 VALUE\n")
        for gene, snps in gene_snps.items():
            w = gene_weights[gene]
            Sigma = gene_ld[gene]
            k = len(snps)
            for i in range(k):
                if w[i] == 0.0:
                    continue
                for j in range(i, k):
                    if w[j] == 0.0:
                        continue
                    f.write(f"{gene} {snps[i]} {snps[j]} {Sigma[i, j]:.10f}\n")

    # ----- write GWAS sumstats -----
    # For each gene with non-zero weights, the implied per-SNP causal beta
    # under independent SNPs is beta_y = theta * w. We add small noise so the
    # observed sumstats look realistic.
    gwas_path = output_dir / "gwas_sumstats.txt.gz"
    per_snp_beta: dict[str, float] = {}
    per_snp_alleles: dict[str, tuple[str, str]] = {}

    for gene, snps in gene_snps.items():
        w = gene_weights[gene]
        theta = gene_truth_beta[gene]
        for j, snp in enumerate(snps):
            if w[j] == 0.0:
                continue
            if snp in per_snp_beta:
                per_snp_beta[snp] += theta * w[j]
            else:
                per_snp_beta[snp] = theta * w[j]
                per_snp_alleles[snp] = gene_alleles[gene][j]

    # Background SNPs get null beta.
    for snp in background:
        per_snp_beta[snp] = 0.0
        per_snp_alleles[snp] = ("A", "G")

    sigma_eff = 1.0 / np.sqrt(GWAS_N)
    from scipy import stats

    snp_records = []
    for snp, beta_true in per_snp_beta.items():
        noise = rng.normal(0.0, sigma_eff)
        beta_obs = beta_true + noise
        se = sigma_eff
        z = beta_obs / se
        p = 2.0 * stats.norm.sf(abs(z))
        eff, ref = per_snp_alleles[snp]
        snp_records.append((snp, eff, ref, z, beta_obs, se, p))

    with gzip.open(gwas_path, "wt", encoding="utf-8") as f:
        f.write("SNP effect_allele non_effect_allele zscore beta se pvalue\n")
        for r in snp_records:
            f.write(
                f"{r[0]} {r[1]} {r[2]} {r[3]:.10g} {r[4]:.10g} "
                f"{r[5]:.10g} {r[6]:.10g}\n"
            )

    truth = {}
    truth["seed"] = int(seed)
    truth["n_genes"] = int(N_GENES)
    truth["snps_per_gene"] = int(SNPS_PER_GENE)
    truth["n_background_snps"] = int(N_BACKGROUND_SNPS)
    truth["gwas_n"] = int(GWAS_N)
    truth["gene_truth_beta"] = gene_truth_beta
    truth["gene_weights"] = {g: w.tolist() for g, w in gene_weights.items()}
    truth["gene_snps"] = gene_snps
    truth["gene_ld"] = {g: M.tolist() for g, M in gene_ld.items()}
    truth["gene_alleles"] = {g: a for g, a in gene_alleles.items()}
    truth["rho"] = 0.3
    truth["sigma_eff"] = float(sigma_eff)
    truth["fixture_sha256"] = dict()
    truth["fixture_sha256"]["model.db"] = _sha256(db_path)
    truth["fixture_sha256"]["model.txt.gz"] = _sha256(cov_path)
    truth["fixture_sha256"]["gwas_sumstats.txt.gz"] = _sha256(gwas_path)
    (output_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2, default=str))
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    truth = simulate(Path(args.output_dir), seed=args.seed)
    print("[simulate_fixture] wrote", args.output_dir)
    print("[simulate_fixture]   n_genes =", truth["n_genes"])
    print("[simulate_fixture] SHA256:")
    for k, v in truth["fixture_sha256"].items():
        print(" ", v, " ", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
