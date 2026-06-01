# Simulated multi-omics ground-truth fixture (Pillar B / Tier 2 B1)

Deterministic G -> expression -> trait simulator used by the TorchGWAS
Genome Biology paper Figure 4 panel A. The fixture provides a numeric
ground truth that downstream TWAS / SMR / coloc / mediation methods are
asked to recover. No reference tool is needed: `truth.json` is the
reference.

## Architecture (causal DAG)

```
       beta_GE                beta_ET
   G ----------> expr ----------> trait
   |             (m mediators)        ^
   |                                  |
   +-------- direct pleiotropy -------+
   |                                  |
   +--- negative-control loci (H3) ---+
```

- `G` (n x p): additive dosages (0..2) under HWE.
- `expr` (n x m): expression matrix, signal from per-gene cis-eQTL plus
  i.i.d. Gaussian noise calibrated to a target h2_expr.
- `trait` (n,): outcome built from
  `mediated = expr @ beta_ET`, a small direct (horizontal-pleiotropy)
  component on `G`, and Gaussian noise. The three variance components
  add to 1.

## Parameters (`CONFIG` in `generate.py`)

| Parameter | Value | Notes |
|---|---|---|
| `n` | 500 | Individuals |
| `p` | 1000 | SNPs (additive dosage 0..2, HWE) |
| `m` | 50 | Mediators (expression features) |
| `seed` | 42 | `numpy.random.default_rng(42)` deterministic |
| `maf_low`, `maf_high` | 0.05, 0.50 | MAF ~ Uniform(low, high) |
| `cis_snps_per_gene` | 5 | Each mediator has 5 cis-eQTL SNPs |
| `n_causal_mediators` | 5 | Mediators with non-zero `beta_ET` |
| `n_negative_control_loci` | 2 | Mediators with H3 architecture |
| `h2_expr` | 0.60 | Target cis-h2 of expression |
| `h2_trait_mediated` | 0.30 | Share of trait variance via `beta_ET . expr` |
| `h2_trait_direct` | 0.05 | Horizontal-pleiotropy component (`G` direct) |
| `h2_trait_noise` | 0.65 | Residual (= 1 - mediated - direct) |

`beta_GE` is sparse by construction: zero outside each gene's cis block
of 5 SNPs. Within the block, all 5 weights are non-zero. The sparse
representation (`beta_GE_sparse`, shape `(m, n_cis)`) is persisted; the
full `(p, m)` matrix is reconstructible from `cis_snp_ids` and is also
exposed under the alias `beta_GE` to match the §5 acceptance gate.

`beta_ET` is sparse: exactly `n_causal_mediators` entries are non-zero.
The non-zero pattern is recorded in `causal_mediator_idx`.

## Negative-control loci (H3 by design)

For each of `n_negative_control_loci` non-causal mediators, a direct
trait effect is placed on a SNP that is **inside the cis block** of that
mediator but **NOT in its eQTL weight set**. The mediator's expression
points to one SNP; the trait's signal at the same locus points to a
different SNP. Per Giambartolomei 2014 this is the H3 (distinct causal
variants) configuration -- a correctly-calibrated `coloc_pairwise`
estimator should call PP.H3 (or PP.H0 if signal is too weak), not PP.H4.
SMR should likewise not be misled into a non-zero `beta_xy`.

## Reproduction

```bash
cd validation/multiomics/sim
python generate.py
sha256sum -c fixtures/manifest.sha256
```

Output:

```
G.npy: OK
expr.npy: OK
trait.npy: OK
truth.json: OK
```

Re-running is byte-identical (G/expr/trait SHAs are invariant under
re-run). The truth.json SHA may shift only if the schema is changed in
`_build_truth_dict`.

## truth.json schema (v1.0)

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | str | "1.0" |
| `config` | dict | Echo of `CONFIG` used in the run |
| `n`, `p`, `m`, `seed` | int | Convenience top-level copies |
| `snp_ids`, `gene_ids` | list[str] | Stable identifiers |
| `maf` | list[float] (p,) | Sampled MAF per SNP |
| `cis_snp_ids` | dict[gene -> list[snp]] | Cis window membership |
| `beta_GE` | list[list[float]] (m, n_cis) | eQTL weights, sparse rep |
| `beta_GE_sparse` | same as `beta_GE` | Explicit name |
| `beta_GE_full_shape` | list[int] (2,) | `[p, m]` of the full matrix |
| `beta_ET` | list[float] (m,) | Mediator -> trait effects |
| `expr_h2_empirical` | list[float] (m,) | Per-gene cis-h2 actually realised |
| `causal_mediator_idx` | list[int] | Indices of mediators with non-zero `beta_ET` |
| `causal_mediator_ids` | list[str] | Same as above but gene IDs |
| `neg_control_mediator_idx` | list[int] | H3 negative-control mediators |
| `neg_control_mediator_ids` | list[str] | Same but gene IDs |
| `direct_pleiotropy_snp_ids` | list[str] | SNPs with direct trait effects |
| `direct_pleiotropy_betas` | list[float] | Effect sizes for those SNPs |
| `twas_z_truth` | list[float] (m,) | Truth TWAS z-score per gene |
| `smr_beta_truth` | list[float] (m,) | Truth SMR beta_xy per gene (= `beta_ET`) |
| `coloc_pp_h4_truth` | list[float] (m,) | Truth PP.H4 (1.0 for causal, 0.0 otherwise) |
| `mediation_ab_truth` | list[float] (m,) | Truth Sobel a*b product per gene |
| `mediation_top_eqtl_snp` | list[str] (m,) | Top eQTL SNP per gene (anchor for a*b) |
| `recovery_floors` | dict | Per-method recovery floors (below) |
| `downstream_api_surfaces` | dict | TG modules that consume the fixture |
| `references` | dict | Primary-source citations |

## Recovery floors (Tier 4 gate)

Downstream Tier 4 methods must beat these floors when run against the
fixture. Observed-then-floored protocol: Tier 4 runs the methods once,
records the observed values, and updates the floors to the rounded
order-of-magnitude. The initial floors are conservative but realistic:

| Metric | Floor | Direction |
|---|---|---|
| TWAS z Pearson correlation vs `twas_z_truth` | >= 0.95 | higher is better |
| SMR beta relative error on causal mediators | <= 0.10 | lower is better |
| coloc PP.H4 on causal mediators (min) | >= 0.80 | higher is better |
| Mediation a*b relative error on causal mediators | <= 0.20 | lower is better |
| coloc PP.H4 on negative-control mediators | < 0.50 | higher is **bad** |

Definitions are also embedded under `truth.recovery_floors.definition`
so the figure-rendering code can grep them programmatically.

## Downstream API surfaces

Tier 4 panel-rendering code reads `truth.json` and calls:

- TWAS: `torchgwas.postgwas._twas.twas_sumstat` (S-PrediXcan; Barbeira 2018)
- SMR: `torchgwas.postgwas._smr.smr_test` + `smr_heidi` (Zhu 2016)
- coloc: `torchgwas.postgwas._hyprcoloc.coloc_pairwise`
  (Giambartolomei 2014 two-trait coloc; PP.H0..H4 decomposition)
- Mediation: `torchgwas.multiomics._mediate.mediate_lmm` (Sobel 1982);
  batched variant `torchgwas.multiomics._scan_batched.batched_scan_pairs`

## Primary sources

- Barbeira A. N. *et al.* 2018. Exploring the phenotypic consequences of
  tissue specific gene expression variation inferred from GWAS summary
  statistics. *Nature Communications* 9:1825.
- Zhu Z. *et al.* 2016. Integration of summary data from GWAS and eQTL
  studies predicts complex trait gene targets. *Nature Genetics*
  48:481-487.
- Giambartolomei C. *et al.* 2014. Bayesian test for colocalisation
  between pairs of genetic association studies using summary statistics.
  *PLoS Genetics* 10(5):e1004383.
- Sobel M. E. 1982. Asymptotic confidence intervals for indirect effects
  in structural equation models. *Sociological Methodology* 13:290-312.
- Wang G. *et al.* 2020. A simple new approach to variable selection in
  regression, with application to genetic fine mapping (SuSiE).
  *Journal of the Royal Statistical Society B* 82:1273-1300.

## Files

| File | Status | SHA256 (after first run) |
|---|---|---|
| `generate.py` | tracked | source-of-truth simulator |
| `README.md` | tracked | this file |
| `.gitignore` | tracked | excludes `fixtures/` except `truth.json` + `manifest.sha256` |
| `fixtures/G.npy` | gitignored (regenerable) | `49e2abfcf12...` |
| `fixtures/expr.npy` | gitignored (regenerable) | `b63e2e031e7...` |
| `fixtures/trait.npy` | gitignored (regenerable) | `59b2ebcb145...` |
| `fixtures/truth.json` | tracked | reference for downstream gate |
| `fixtures/manifest.sha256` | tracked | digest of all four fixture files |

Per CLAUDE.md repo conventions, the small reference artefacts
(`truth.json`, `manifest.sha256`) are committed; the regenerable bulk
arrays are not, but their SHA256 is committed so any reproducer can
verify byte-identity.
