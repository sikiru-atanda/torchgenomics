"""Deterministic simulated multi-omics fixture for TorchGenomics paper Figure 4-A.

Generates ground-truth (G, expr, trait) under a known causal architecture
G -> expr -> trait, plus a non-causal coloc / SMR negative control,
seeded from numpy.random.default_rng(42).

Outputs (all under fixtures/):

* G.npy           : (n, p) genotype dosage matrix (additive coding, 0..2)
* expr.npy        : (n, m) expression / mediator matrix
* trait.npy       : (n,) outcome vector
* truth.json      : ground-truth weights, indices, variance components,
                    per-method recovery floors
* manifest.sha256 : SHA256 of every fixture file in this directory

Design references (cited in README.md):

* Barbeira et al. 2018 Nat. Comm. 9:1825 -- TWAS / S-PrediXcan z-score
  recovery via cis-eQTL weights.
* Zhu et al. 2016 Nat. Genet. 48:481-487 -- SMR (Summary Mendelian
  Randomization) beta_xy = beta_zy / beta_zx.
* Giambartolomei et al. 2014 PLoS Genet. 10:e1004383 -- Bayesian
  colocalization PP.H4 (shared causal).
* Sobel 1982 Sociol. Methodol. 13:290-312 -- product-of-coefficients
  mediation indirect effect ab.
* Wang et al. 2020 J. R. Stat. Soc. B 82:1273-1300 (SuSiE) -- inspiration
  for sparse mediator -> trait architecture.

Reproduction:
    cd validation/multiomics/sim
    python generate.py
    sha256sum -c fixtures/manifest.sha256
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = dict(
    n=500,
    p=1000,
    m=50,
    seed=42,
    maf_low=0.05,
    maf_high=0.50,
    cis_snps_per_gene=5,
    n_causal_mediators=5,
    n_negative_control_loci=2,
    h2_expr=0.60,
    h2_trait_mediated=0.30,
    h2_trait_direct=0.05,
    twas_z_corr_floor=0.95,
    smr_beta_rel_floor=0.10,
    coloc_pp_h4_floor=0.80,
    mediation_beta_rel_floor=0.20,
)


def _sample_genotypes(n, p, maf_low, maf_high, rng):
    """Sample diploid additive dosages under HWE, MAF ~ U[low, high]."""
    maf = rng.uniform(maf_low, maf_high, size=p)
    g = rng.binomial(2, maf[None, :], size=(n, p)).astype(np.float64)
    return g, maf


def _calibrate_noise(signal, h2, rng):
    """Scale Gaussian noise so signal explains exactly h2 of total variance."""
    n = signal.shape[0]
    var_sig = float(np.var(signal, ddof=1))
    if var_sig == 0.0:
        eps = rng.standard_normal(n)
        return eps / float(np.std(eps, ddof=1))
    sigma_e2 = var_sig * (1.0 - h2) / h2
    eps = rng.standard_normal(n) * float(np.sqrt(sigma_e2))
    return eps


def simulate(cfg):
    """Generate the fixture. Returns dict with arrays + truth metadata."""
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["n"]
    p = cfg["p"]
    m = cfg["m"]
    n_cis = cfg["cis_snps_per_gene"]
    n_causal_med = cfg["n_causal_mediators"]
    n_neg_loci = cfg["n_negative_control_loci"]

    # (1) Genotypes
    G, maf = _sample_genotypes(n, p, cfg["maf_low"], cfg["maf_high"], rng)
    snp_ids = [f"SNP{i:05d}" for i in range(p)]
    gene_ids = [f"G{i:03d}" for i in range(m)]

    # (2) Cis windows
    cis_block_size = p // m
    if cis_block_size < n_cis:
        raise RuntimeError(
            f"Need cis_block_size >= n_cis; got block={cis_block_size}, "
            f"n_cis={n_cis}."
        )

    cis_snp_index = dict()
    cis_snp_ids = dict()
    for j, gene in enumerate(gene_ids):
        block_start = j * cis_block_size
        idx = list(range(block_start, block_start + n_cis))
        cis_snp_index[gene] = idx
        cis_snp_ids[gene] = [snp_ids[i] for i in idx]

    # (3) beta_GE: dense within each gene's cis block, zero elsewhere
    beta_GE_sparse = np.zeros((m, n_cis), dtype=np.float64)
    for j in range(m):
        signs = rng.choice([-1.0, 1.0], size=n_cis)
        mags = np.abs(rng.standard_normal(n_cis) * 0.30) + 0.10
        beta_GE_sparse[j] = signs * mags

    beta_GE_full = np.zeros((p, m), dtype=np.float64)
    for j, gene in enumerate(gene_ids):
        for k_local, snp_idx in enumerate(cis_snp_index[gene]):
            beta_GE_full[snp_idx, j] = beta_GE_sparse[j, k_local]

    # (4) Expression: G @ beta_GE + scaled noise (target h2_expr)
    expr_signal = G @ beta_GE_full
    expr = np.zeros_like(expr_signal)
    h2_expr = cfg["h2_expr"]
    for j in range(m):
        eps_j = _calibrate_noise(expr_signal[:, j], h2_expr, rng)
        expr[:, j] = expr_signal[:, j] + eps_j
    expr = expr - expr.mean(axis=0, keepdims=True)

    expr_h2_empirical = np.zeros(m, dtype=np.float64)
    for j in range(m):
        v_sig = float(np.var(expr_signal[:, j], ddof=1))
        v_tot = float(np.var(expr[:, j], ddof=1))
        expr_h2_empirical[j] = v_sig / v_tot if v_tot > 0 else 0.0

    # (5) beta_ET: sparse mediator -> trait
    beta_ET_full = np.zeros(m, dtype=np.float64)
    causal_mediator_idx = sorted(
        rng.choice(m, size=n_causal_med, replace=False).tolist()
    )
    for j in causal_mediator_idx:
        sign = float(rng.choice([-1.0, 1.0]))
        mag = float(np.abs(rng.standard_normal()) * 0.40 + 0.20)
        beta_ET_full[j] = sign * mag

    # (6) Negative-control loci (H3 by design): pick mediators not in the
    # causal set and add a direct trait effect on a SNP that is in the
    # gene's cis block but NOT in its eQTL weight set. coloc should call
    # PP.H3 (distinct causals) not PP.H4 for these.
    causal_set = set(causal_mediator_idx)
    non_causal_pool = [j for j in range(m) if j not in causal_set]
    rng.shuffle(non_causal_pool)
    neg_control_med_idx = sorted(non_causal_pool[:n_neg_loci])

    direct_snp_indices = []
    direct_snp_betas = []
    for j in neg_control_med_idx:
        block_start = j * cis_block_size
        eqtl_set = set(cis_snp_index[gene_ids[j]])
        candidates = [
            i for i in range(block_start, block_start + cis_block_size)
            if i not in eqtl_set
        ]
        if not candidates:
            raise RuntimeError(
                f"No non-eQTL SNPs left in cis block for gene {gene_ids[j]}."
            )
        snp_choice = int(rng.choice(candidates))
        sign = float(rng.choice([-1.0, 1.0]))
        mag = float(np.abs(rng.standard_normal()) * 0.30 + 0.20)
        direct_snp_indices.append(snp_choice)
        direct_snp_betas.append(sign * mag)

    # (7) Trait: mediated + direct (pleiotropy + neg controls) + noise
    mediated = expr @ beta_ET_full
    direct_g = np.zeros(n, dtype=np.float64)
    for idx, b in zip(direct_snp_indices, direct_snp_betas):
        direct_g += G[:, idx] * b

    var_mediated = float(np.var(mediated, ddof=1))
    var_direct = float(np.var(direct_g, ddof=1))
    h2_med = cfg["h2_trait_mediated"]
    h2_dir = cfg["h2_trait_direct"]
    h2_noise = 1.0 - h2_med - h2_dir
    if h2_noise <= 0:
        raise ValueError("h2_trait_mediated + h2_trait_direct must be < 1.")

    if var_mediated > 0:
        mediated = mediated * float(np.sqrt(h2_med / var_mediated))
    if var_direct > 0:
        direct_g = direct_g * float(np.sqrt(h2_dir / var_direct))

    eps_trait = rng.standard_normal(n) * float(np.sqrt(h2_noise))
    trait = mediated + direct_g + eps_trait
    trait = trait - trait.mean()

    # (8) Ground-truth metrics
    Y = trait - trait.mean()
    twas_z_truth = np.zeros(m, dtype=np.float64)
    for j in range(m):
        ex = expr[:, j]
        ex_std = float(np.std(ex, ddof=1))
        y_std = float(np.std(Y, ddof=1))
        if ex_std == 0 or y_std == 0:
            twas_z_truth[j] = 0.0
            continue
        r = float(np.corrcoef(ex, Y)[0, 1])
        if abs(r) >= 1.0:
            twas_z_truth[j] = float(np.sign(r)) * 1e6
        else:
            twas_z_truth[j] = (
                r * float(np.sqrt(n - 2)) / float(np.sqrt(1.0 - r * r))
            )

    smr_beta_truth = beta_ET_full.copy()

    coloc_pp_h4_truth = np.zeros(m, dtype=np.float64)
    for j in causal_mediator_idx:
        coloc_pp_h4_truth[j] = 1.0

    mediation_ab_truth = np.zeros(m, dtype=np.float64)
    mediation_top_eqtl_snp = []
    for j, gene in enumerate(gene_ids):
        local_idx = int(np.argmax(np.abs(beta_GE_sparse[j])))
        snp_idx = cis_snp_index[gene][local_idx]
        a = float(beta_GE_sparse[j, local_idx])
        b = float(beta_ET_full[j])
        mediation_ab_truth[j] = a * b
        mediation_top_eqtl_snp.append(snp_ids[snp_idx])

    out = dict()
    out["G"] = G
    out["expr"] = expr
    out["trait"] = trait
    out["snp_ids"] = snp_ids
    out["gene_ids"] = gene_ids
    out["maf"] = maf
    out["beta_GE_sparse"] = beta_GE_sparse
    out["beta_GE_full"] = beta_GE_full
    out["beta_ET_full"] = beta_ET_full
    out["cis_snp_index"] = cis_snp_index
    out["cis_snp_ids"] = cis_snp_ids
    out["expr_h2_empirical"] = expr_h2_empirical
    out["causal_mediator_idx"] = causal_mediator_idx
    out["neg_control_med_idx"] = neg_control_med_idx
    out["direct_snp_indices"] = direct_snp_indices
    out["direct_snp_betas"] = direct_snp_betas
    out["twas_z_truth"] = twas_z_truth
    out["smr_beta_truth"] = smr_beta_truth
    out["coloc_pp_h4_truth"] = coloc_pp_h4_truth
    out["mediation_ab_truth"] = mediation_ab_truth
    out["mediation_top_eqtl_snp"] = mediation_top_eqtl_snp
    return out


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_truth_dict(sim, cfg):
    """Assemble the truth dictionary to be JSON-serialized.

    Built incrementally with dict() + key assignment (not literal braces)
    so the source file stays grep-friendly and free of large JSON-shaped
    literals.
    """
    truth = dict()
    truth["schema_version"] = "1.0"
    truth["description"] = (
        "Simulated multi-omics ground truth for TorchGenomics Genome Biology "
        "paper Figure 4 panel A. G -> expr -> trait with sparse beta_GE, "
        "sparse beta_ET, a pleiotropic direct-effect component, and a "
        "non-causal coloc/SMR negative control."
    )
    truth["config"] = dict(cfg)
    truth["n"] = cfg["n"]
    truth["p"] = cfg["p"]
    truth["m"] = cfg["m"]
    truth["seed"] = cfg["seed"]
    truth["snp_ids"] = sim["snp_ids"]
    truth["gene_ids"] = sim["gene_ids"]
    truth["maf"] = sim["maf"].tolist()
    truth["cis_snp_ids"] = sim["cis_snp_ids"]
    truth["beta_GE_sparse"] = sim["beta_GE_sparse"].tolist()
    truth["beta_GE_full_shape"] = list(sim["beta_GE_full"].shape)
    # Alias for the plan-section-5 acceptance gate which expects key "beta_GE"
    truth["beta_GE"] = sim["beta_GE_sparse"].tolist()
    truth["beta_ET"] = sim["beta_ET_full"].tolist()
    truth["expr_h2_empirical"] = sim["expr_h2_empirical"].tolist()
    truth["causal_mediator_idx"] = list(sim["causal_mediator_idx"])
    truth["causal_mediator_ids"] = [
        sim["gene_ids"][j] for j in sim["causal_mediator_idx"]
    ]
    truth["neg_control_mediator_idx"] = list(sim["neg_control_med_idx"])
    truth["neg_control_mediator_ids"] = [
        sim["gene_ids"][j] for j in sim["neg_control_med_idx"]
    ]
    truth["direct_pleiotropy_snp_ids"] = [
        sim["snp_ids"][i] for i in sim["direct_snp_indices"]
    ]
    truth["direct_pleiotropy_betas"] = list(sim["direct_snp_betas"])
    truth["twas_z_truth"] = sim["twas_z_truth"].tolist()
    truth["smr_beta_truth"] = sim["smr_beta_truth"].tolist()
    truth["coloc_pp_h4_truth"] = sim["coloc_pp_h4_truth"].tolist()
    truth["mediation_ab_truth"] = sim["mediation_ab_truth"].tolist()
    truth["mediation_top_eqtl_snp"] = sim["mediation_top_eqtl_snp"]
    return truth


def _build_floors_dict(cfg):
    """Recovery-floor block for truth.json (downstream methods must beat)."""
    defn = dict()
    defn["twas_z_corr_pearson"] = (
        "Pearson correlation of recovered TWAS z-scores against "
        "truth.twas_z_truth across all m genes; must be >= floor."
    )
    defn["smr_beta_relative_error"] = (
        "max over causal mediators of "
        "|b_smr_hat - smr_beta_truth| / |smr_beta_truth|; "
        "must be <= floor."
    )
    defn["coloc_pp_h4_min_for_causal"] = (
        "min over causal mediators of recovered PP.H4; must be >= "
        "floor. Negative-control mediators must have PP.H4 < 0.5."
    )
    defn["mediation_ab_relative_error"] = (
        "max over causal mediators of "
        "|ab_hat - mediation_ab_truth| / |mediation_ab_truth|; "
        "must be <= floor."
    )

    floors = dict()
    floors["twas_z_corr_pearson"] = cfg["twas_z_corr_floor"]
    floors["smr_beta_relative_error"] = cfg["smr_beta_rel_floor"]
    floors["coloc_pp_h4_min_for_causal"] = cfg["coloc_pp_h4_floor"]
    floors["mediation_ab_relative_error"] = cfg["mediation_beta_rel_floor"]
    floors["definition"] = defn
    return floors


def _build_api_dict():
    api = dict()
    api["twas"] = "torchgenomics.postgwas._twas.twas_sumstat (S-PrediXcan)"
    api["smr"] = "torchgenomics.postgwas._smr.smr_test + smr_heidi"
    api["coloc"] = "torchgenomics.postgwas._hyprcoloc.coloc_pairwise"
    api["mediation"] = "torchgenomics.multiomics._mediate.mediate_lmm"
    api["batched_mediation"] = (
        "torchgenomics.multiomics._scan_batched.batched_scan_pairs"
    )
    return api


def _build_refs_dict():
    refs = dict()
    refs["twas"] = "Barbeira et al. 2018 Nat. Comm. 9:1825 (S-PrediXcan)"
    refs["smr"] = "Zhu et al. 2016 Nat. Genet. 48:481-487 (SMR + HEIDI)"
    refs["coloc"] = "Giambartolomei et al. 2014 PLoS Genet. 10:e1004383"
    refs["mediation"] = "Sobel 1982 Sociol. Methodol. 13:290-312"
    refs["susie"] = "Wang et al. 2020 J. R. Stat. Soc. B 82:1273-1300"
    return refs


def write_fixture(out_dir, sim, cfg):
    """Persist arrays + truth.json + manifest.sha256 to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "G.npy", sim["G"].astype(np.float32))
    np.save(out_dir / "expr.npy", sim["expr"].astype(np.float64))
    np.save(out_dir / "trait.npy", sim["trait"].astype(np.float64))

    truth = _build_truth_dict(sim, cfg)
    truth["recovery_floors"] = _build_floors_dict(cfg)
    truth["downstream_api_surfaces"] = _build_api_dict()
    truth["references"] = _build_refs_dict()
    with (out_dir / "truth.json").open("w") as f:
        json.dump(truth, f, indent=2, sort_keys=False)

    manifest_files = ["G.npy", "expr.npy", "trait.npy", "truth.json"]
    lines = []
    for name in manifest_files:
        digest = _sha256(out_dir / name)
        lines.append(f"{digest}  {name}\n")
    with (out_dir / "manifest.sha256").open("w") as f:
        f.writelines(lines)


def main():
    out_dir = Path(__file__).resolve().parent / "fixtures"
    sim = simulate(CONFIG)
    write_fixture(out_dir, sim, CONFIG)

    print(f"Wrote fixture to {out_dir}")
    print(f"  G.npy      shape={sim['G'].shape} dtype=float32")
    print(f"  expr.npy   shape={sim['expr'].shape} dtype=float64")
    print(f"  trait.npy  shape={sim['trait'].shape} dtype=float64")
    print(f"  causal mediators: {sim['causal_mediator_idx']}")
    print(f"  negative-control mediators: {sim['neg_control_med_idx']}")
    print(f"  manifest:")
    for line in (out_dir / "manifest.sha256").read_text().splitlines():
        print(f"    {line}")


if __name__ == "__main__":
    main()
