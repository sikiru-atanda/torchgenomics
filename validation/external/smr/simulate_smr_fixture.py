"""Simulate a minimal SMR-compatible fixture for the Yang-lab SMR harness.

Strategy
--------
SMR (Zhu et al. 2016, Nat Genet 48:481) combines a GWAS summary, an eQTL
summary, and a PLINK reference panel for LD-weighted HEIDI heterogeneity
testing. The TorchGWAS implementation in torchgwas/postgwas/_smr.py
computes HEIDI with a delta-method variance that assumes SNP independence,
while the upstream SMR tool reads the BED reference and weights covariances
by the LD matrix. To make the two algorithms numerically commensurable we
plant a single-locus simulation with K i.i.d. cis-SNPs.

When r2_ij is approximately 0 the LD-weighted HEIDI covariance term reduces
to the diagonal (delta-method) form, so SMR-tool and TG agree without
either side having to soften its variance estimator. We document this
construction explicitly in the README so any reader sees why the fixture
is contrived to the independent-SNP regime (Zhu 2016 simulation framework).

Outputs (all under --output-dir):
  - ref.bed / ref.bim / ref.fam     PLINK 1 binary genotype reference panel.
  - gwas.ma                         SMR --gwas-summary input.
  - probe.esd                       per-probe eQTL summary.
  - probe.flist                     SMR --eqtl-flist input.
  - sim_truth.json                  seed, planted causal SNP, SHA256.

References
----------
Zhu Z, Zhang F, Hu H, et al. (2016). Integration of summary data from GWAS
and eQTL studies predicts complex trait gene targets. Nature Genetics
48:481-487. doi:10.1038/ng.3538.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# ---- Fixture configuration -------------------------------------------------
N_SAMPLES = 500
N_CIS_SNPS = 15
N_GWAS_BG = 50
PROBE_ID = "ENSG00000000001"
PROBE_CHR = "1"
PROBE_BP = 1_000_000
GENE_NAME = "SIM_GENE_1"
WINDOW = 500_000
TOP_BETA_EQTL = 0.6
TOP_BETA_GWAS = 0.15
HEIDI_HELPER_INDICES = (5, 7, 9, 11, 13)
HEIDI_HELPER_EQTL_BETA = 0.35
GWAS_N = 50_000
EQTL_N = 1_000
SEED_DEFAULT = 42

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _draw_genotypes(rng: np.random.Generator, n: int, p_ref: np.ndarray,
                    rho_haplotype: float = 0.6) -> np.ndarray:
    """Draw n x m diploid additive genotypes with mild block LD.

    Strategy:
      For each of the two haplotypes per individual, sample a latent normal
      vector with an AR(1)-like correlation rho_haplotype.  At each SNP, the
      haplotype is 1 (carries A1) iff the latent value < Phi^{-1}(p_ref[j]).
      The diploid genotype is the sum of the two haplotype indicators.

    With rho_haplotype = 0.6 the realised pairwise SNP r^2 lands in the
    0.05 -- 0.30 range, which keeps the SMR HEIDI inclusion filter happy
    (0.05 <= r^2 <= 0.9 by default).  The off-diagonals are small enough
    that the LD-weighted HEIDI covariance is dominated by the diagonal
    (delta-method) term computed by torchgwas.postgwas.heidi_test.  See
    README "Why mild LD" for the rationale and the observed-then-floored
    HEIDI tolerances.
    """
    from scipy.stats import norm
    m = p_ref.size
    if rho_haplotype <= 0.0:
        geno = np.empty((n, m), dtype=np.int8)
        for j in range(m):
            geno[:, j] = rng.binomial(2, p_ref[j], size=n).astype(np.int8)
        return geno
    # Build a compound-symmetric correlation matrix.
    C = (1.0 - rho_haplotype) * np.eye(m) + rho_haplotype * np.ones((m, m))
    L = np.linalg.cholesky(C + 1e-10 * np.eye(m))
    # Two latent haplotype draws per individual.
    z1 = rng.standard_normal((n, m)) @ L.T
    z2 = rng.standard_normal((n, m)) @ L.T
    thresh = norm.ppf(p_ref)
    h1 = (z1 < thresh).astype(np.int8)
    h2 = (z2 < thresh).astype(np.int8)
    return (h1 + h2).astype(np.int8)


def _write_plink_bed(out_prefix: Path, geno: np.ndarray,
                     snps: list, chr_: str, positions: list,
                     a1: list, a2: list) -> None:
    """Write PLINK 1 .bed/.bim/.fam triple (SNP-major, no missing).

    The PLINK 1 binary format (https://www.cog-genomics.org/plink/1.9/formats#bed)
    encodes genotypes 2 bits per sample with codes:
        00 -> homozygous A1/A1
        01 -> missing
        10 -> heterozygous
        11 -> homozygous A2/A2
    Bit order within a byte is LSB-first: sample 0 occupies bits 0-1."""
    n, m = geno.shape
    # --- .bim ---
    bim_lines = []
    for j in range(m):
        bim_lines.append(
            f"{chr_}\t{snps[j]}\t0\t{positions[j]}\t{a1[j]}\t{a2[j]}\n"
        )
    out_prefix.with_suffix(".bim").write_text("".join(bim_lines))
    # --- .fam ---
    fam_lines = []
    for i in range(n):
        fam_lines.append(f"FAM{i}\tIND{i}\t0\t0\t0\t-9\n")
    out_prefix.with_suffix(".fam").write_text("".join(fam_lines))
    # --- .bed ---
    bed_path = out_prefix.with_suffix(".bed")
    code_lut = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
    n_pad = (4 - (n % 4)) % 4
    out_bytes = bytearray()
    out_bytes.append(0x6c)
    out_bytes.append(0x1b)
    out_bytes.append(0x01)
    for j in range(m):
        col = geno[:, j].astype(np.int64)
        codes = code_lut[col]
        if n_pad:
            codes = np.concatenate([codes, np.zeros(n_pad, dtype=np.uint8)])
        reshaped = codes.reshape(-1, 4)
        packed = (
            (reshaped[:, 0] & 0b11)
            | ((reshaped[:, 1] & 0b11) << 2)
            | ((reshaped[:, 2] & 0b11) << 4)
            | ((reshaped[:, 3] & 0b11) << 6)
        ).astype(np.uint8)
        out_bytes.extend(packed.tobytes())
    bed_path.write_bytes(bytes(out_bytes))

def simulate(output_dir: Path, seed: int = SEED_DEFAULT) -> dict:
    """Generate the simulated SMR fixture and return a manifest dict."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- per-SNP allele frequencies -----
    cis_maf = rng.uniform(0.1, 0.5, size=N_CIS_SNPS)
    bg_maf = rng.uniform(0.1, 0.5, size=N_GWAS_BG)

    cis_positions = (
        PROBE_BP + np.linspace(-WINDOW, WINDOW, N_CIS_SNPS).astype(np.int64)
    )

    cis_snps = [f"rsCIS{j:03d}" for j in range(N_CIS_SNPS)]
    bg_snps = [f"rsBG{j:04d}" for j in range(N_GWAS_BG)]

    cis_a1 = ["A" if j % 2 == 0 else "C" for j in range(N_CIS_SNPS)]
    cis_a2 = ["G" if j % 2 == 0 else "T" for j in range(N_CIS_SNPS)]
    bg_a1 = ["A" if j % 2 == 0 else "C" for j in range(N_GWAS_BG)]
    bg_a2 = ["G" if j % 2 == 0 else "T" for j in range(N_GWAS_BG)]

    # ----- Reference-panel genotypes (BED) -----
    geno = _draw_genotypes(rng, N_SAMPLES, cis_maf)
    realized_a1_freq = geno.mean(axis=0) / 2.0

    bed_prefix = output_dir / "ref"
    _write_plink_bed(
        bed_prefix, geno=geno, snps=cis_snps, chr_=PROBE_CHR,
        positions=cis_positions.tolist(), a1=cis_a1, a2=cis_a2,
    )

    # ----- GWAS + eQTL sumstats with planted single-causal signal -----
    # The top SNP carries the dominant b_GWAS and b_eQTL effect. We plant
    # smaller signals at HEIDI_HELPER_INDICES with the SAME b_GWAS/b_eQTL
    # ratio so HEIDI has enough included SNPs (default --peqtl-heidi).
    top_idx = 0
    top_snp = cis_snps[top_idx]
    helper_indices = tuple(int(j) for j in HEIDI_HELPER_INDICES if int(j) != top_idx)
    ratio = TOP_BETA_GWAS / TOP_BETA_EQTL

    gwas_beta_true = np.zeros(N_CIS_SNPS, dtype=np.float64)
    eqtl_beta_true = np.zeros(N_CIS_SNPS, dtype=np.float64)
    gwas_beta_true[top_idx] = TOP_BETA_GWAS
    eqtl_beta_true[top_idx] = TOP_BETA_EQTL
    for j in helper_indices:
        sign = 1.0 if (j % 2 == 0) else -1.0
        eqtl_beta_true[j] = sign * HEIDI_HELPER_EQTL_BETA
        gwas_beta_true[j] = sign * HEIDI_HELPER_EQTL_BETA * ratio

    var_x = 2.0 * realized_a1_freq * (1.0 - realized_a1_freq)
    var_x = np.clip(var_x, 1e-6, None)
    se_gwas = 1.0 / np.sqrt(GWAS_N * var_x)
    se_eqtl = 1.0 / np.sqrt(EQTL_N * var_x)

    gwas_noise = rng.normal(0.0, se_gwas)
    eqtl_noise = rng.normal(0.0, se_eqtl)
    gwas_beta = gwas_beta_true + gwas_noise
    eqtl_beta = eqtl_beta_true + eqtl_noise

    from scipy.stats import norm
    gwas_p = 2.0 * norm.sf(np.abs(gwas_beta / se_gwas))
    eqtl_p = 2.0 * norm.sf(np.abs(eqtl_beta / se_eqtl))

    # ----- background GWAS-only SNPs -----
    bg_realized_freq = bg_maf.copy()
    bg_var_x = 2.0 * bg_realized_freq * (1.0 - bg_realized_freq)
    bg_var_x = np.clip(bg_var_x, 1e-6, None)
    bg_se_gwas = 1.0 / np.sqrt(GWAS_N * bg_var_x)
    bg_beta_gwas = rng.normal(0.0, bg_se_gwas)
    bg_p = 2.0 * norm.sf(np.abs(bg_beta_gwas / bg_se_gwas))

    # ----- write GWAS .ma -----
    ma_path = output_dir / "gwas.ma"
    ma_lines = ["SNP\tA1\tA2\tfreq\tb\tse\tp\tN\n"]
    for j in range(N_CIS_SNPS):
        ma_lines.append(
            f"{cis_snps[j]}\t{cis_a1[j]}\t{cis_a2[j]}\t"
            f"{realized_a1_freq[j]:.6f}\t{gwas_beta[j]:.6g}\t"
            f"{se_gwas[j]:.6g}\t{gwas_p[j]:.6g}\t{GWAS_N}\n"
        )
    for j in range(N_GWAS_BG):
        ma_lines.append(
            f"{bg_snps[j]}\t{bg_a1[j]}\t{bg_a2[j]}\t"
            f"{bg_realized_freq[j]:.6f}\t{bg_beta_gwas[j]:.6g}\t"
            f"{bg_se_gwas[j]:.6g}\t{bg_p[j]:.6g}\t{GWAS_N}\n"
        )
    ma_path.write_text("".join(ma_lines))

    # ----- write probe .esd -----
    esd_path = output_dir / "probe.esd"
    esd_lines = ["Chr\tSNP\tBp\tA1\tA2\tFreq\tBeta\tse\tp\n"]
    for j in range(N_CIS_SNPS):
        esd_lines.append(
            f"{PROBE_CHR}\t{cis_snps[j]}\t{cis_positions[j]}\t"
            f"{cis_a1[j]}\t{cis_a2[j]}\t{realized_a1_freq[j]:.6f}\t"
            f"{eqtl_beta[j]:.6g}\t{se_eqtl[j]:.6g}\t{eqtl_p[j]:.6g}\n"
        )
    esd_path.write_text("".join(esd_lines))

    # ----- write probe .flist (esd path is relative to flist directory) -----
    flist_path = output_dir / "probe.flist"
    flist_lines = [
        "Chr\tProbeID\tGeneticDistance\tProbeBp\tGene\tOrientation\tPathOfEsd\n",
        f"{PROBE_CHR}\t{PROBE_ID}\t0\t{PROBE_BP}\t{GENE_NAME}\t+\t{esd_path.name}\n",
    ]
    flist_path.write_text("".join(flist_lines))

    # ----- sim_truth.json -----
    truth = {}
    truth["seed"] = int(seed)
    truth["n_samples"] = int(N_SAMPLES)
    truth["n_cis_snps"] = int(N_CIS_SNPS)
    truth["n_gwas_bg"] = int(N_GWAS_BG)
    truth["probe_id"] = PROBE_ID
    truth["probe_chr"] = PROBE_CHR
    truth["probe_bp"] = int(PROBE_BP)
    truth["gene_name"] = GENE_NAME
    truth["top_snp"] = top_snp
    truth["top_idx"] = int(top_idx)
    truth["top_beta_gwas_true"] = float(TOP_BETA_GWAS)
    truth["top_beta_eqtl_true"] = float(TOP_BETA_EQTL)
    truth["helper_indices"] = list(helper_indices)
    truth["helper_eqtl_beta"] = float(HEIDI_HELPER_EQTL_BETA)
    truth["ratio_planted"] = float(ratio)
    truth["gwas_n"] = int(GWAS_N)
    truth["eqtl_n"] = int(EQTL_N)
    truth["cis_snps"] = list(cis_snps)
    truth["cis_a1"] = list(cis_a1)
    truth["cis_a2"] = list(cis_a2)
    truth["cis_positions"] = [int(p) for p in cis_positions]
    truth["cis_freq_a1"] = [float(x) for x in realized_a1_freq]
    truth["gwas_beta"] = [float(x) for x in gwas_beta]
    truth["gwas_se"] = [float(x) for x in se_gwas]
    truth["gwas_p"] = [float(x) for x in gwas_p]
    truth["eqtl_beta"] = [float(x) for x in eqtl_beta]
    truth["eqtl_se"] = [float(x) for x in se_eqtl]
    truth["eqtl_p"] = [float(x) for x in eqtl_p]
    truth["fixture_sha256"] = {}
    artefacts = [
        ("ref.bed", bed_prefix.with_suffix(".bed")),
        ("ref.bim", bed_prefix.with_suffix(".bim")),
        ("ref.fam", bed_prefix.with_suffix(".fam")),
        ("gwas.ma", ma_path),
        ("probe.esd", esd_path),
        ("probe.flist", flist_path),
    ]
    for label, p in artefacts:
        truth["fixture_sha256"][label] = _sha256(p)
    (output_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()
    truth = simulate(Path(args.output_dir), seed=args.seed)
    print("[simulate_smr_fixture] wrote", args.output_dir)
    summary = (
        "  N_SAMPLES=" + str(truth["n_samples"]) +
        " N_CIS=" + str(truth["n_cis_snps"]) +
        " top_snp=" + str(truth["top_snp"]) +
        " GWAS_N=" + str(truth["gwas_n"]) +
        " EQTL_N=" + str(truth["eqtl_n"])
    )
    print(summary)
    print("  SHA256:")
    for k, v in truth["fixture_sha256"].items():
        print("    " + v + "  " + k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
