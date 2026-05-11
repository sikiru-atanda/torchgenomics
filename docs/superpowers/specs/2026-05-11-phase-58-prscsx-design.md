# Phase 58 — PRS-CSx Multi-Population PRS (MVP) — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven)
- **Date:** 2026-05-11
- **Branch:** `modernization/specs`
- **Master spec:** [`2026-05-11-modernization-master-design.md`](2026-05-11-modernization-master-design.md)
- **Research brief:** [`../research/prscsx-research.md`](../research/prscsx-research.md)
- **Phase number:** 58
- **Scope class:** MVP (per master spec §2)
- **Sequencing constraint:** independent of X-chrom refactor (PRS-CSx is summary-stats based; no genotype matrix involved). Can be implemented in parallel with Phase 59 PolyFun per master spec §3.
- **Living spec.** Tier A is fixed; Tier B / C may be revised at implementation kickoff.

---

## 1. Goal & scope

Add multi-population polygenic-score construction via PRS-CSx (Ruan et al. 2022 [C7]) to TorchGWAS. PRS-CSx extends single-population PRS-CS (Ge et al. 2019 [C9]) by sharing a continuous-shrinkage prior across $K$ ancestry populations while keeping per-population posterior effects, enabling cross-ancestry borrowing of evidence. The modern standard for non-European PRS construction; Ruan 2022 reports a **median 52.3% R² improvement** over single-population PRS-CS in EAS targets (per [C7] Figure 3a).

Today our `pgs/prscs.py` implements only the single-population case. PRS-CSx is genuinely a different model (different state-space coupling — see §2.2), so this phase ships as a new file `pgs/prscsx.py` per master spec R-MOD-3 and the brief's §11.1 recommendation, NOT as an extension of `prscs.py`.

**In scope (MVP):**

- **Multi-population coupled-shrinkage Gibbs sampler** for $K \in \{2, 3, 4, 5\}$ populations from `{AFR, AMR, EAS, EUR, SAS}` (per [C7] Methods §"PRS-CSx" and `mcmc_gtb.py`).
- **Fixed global shrinkage parameter $\phi$** with user-supplied value (per [C7] README "Optional Flags"; recommended grid $\{10^{-6}, 10^{-4}, 10^{-2}, 1\}$ per [C7] Methods).
- **Per-(population, chromosome) posterior effect-size output** matching PRS-CSx's published file format (`<chr>  <rsid>  <bp>  <A1>  <A2>  <posterior_effect_size>`, no header, whitespace-separated; per [C7] README "Output Format").
- **HDF5 LD reference panel ingestion** for both 1000G and UK Biobank panel families (per [C7] README "LD Reference Panels & Downloads").
- **Multi-population sumstats harmonization** with shared variant universe via PRS-CSx's `snpinfo_mult_*` master file format.
- **CLI:** new subcommand `pgs-fit-csx` (parallel to existing `pgs-fit` for single-population). Mirror upstream flag names where reasonable.
- **Tier 2 parity** vs PRS-CSx upstream Python on the bundled EUR + EAS chr22 fixture per [C7] README "Test Data" — within `floor + observed × 2` tolerance per master spec §5.

**Explicitly out of scope (deferred):**

- **φ auto-learn mode** (`--phi=None` upstream code path with Gamma(1,1) hyperprior per [C7] Methods §"PRS-CSx" and `mcmc_gtb.py` step 4). Defer to a follow-on phase. MVP requires `--phi` to be specified.
- **`--meta=True` posterior averaging** (inverse-variance-weighted across-population posterior effects). Per [C7] README this is an additional output mode; explicitly excluded from MVP per master spec §2.
- **`--write_pst`** (writing per-iteration posterior trace for debugging). Defer.
- **Validation-set linear-combination workflow** (the "Approach 1" recommended in [C7] README, where users learn per-population score blending weights on a held-out set). This is downstream of the per-(pop, variant) posterior effects we produce; we document the recipe but do not implement the workflow.
- **PRS-CSx-mult / PRS-CSx-meta variants** beyond the core sampler.
- **Admixed-target ancestry inference** ([C7] Issue #73 raises this; no upstream solution; out of scope).
- **Custom user-supplied LD panels in non-1000G / non-UKBB format** ([C7] Issue #72). MVP requires user to use one of the two published panel families; custom panel ingestion deferred.

## 2. Algorithm reference

### 2.1 Single-population PRS-CS prior (for contrast)

Per Ge et al. 2019 [C9] and our existing `torchgwas/pgs/prscs.py`:

$$
\beta_j \mid \psi_j \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{N}\psi_j\right), \quad
\psi_j \mid \delta_j \sim \mathrm{Gamma}(a, \delta_j), \quad
\delta_j \mid \phi \sim \mathrm{Gamma}(b, \phi)
$$

Defaults: $a = 1$, $b = 1/2$ (Strawderman–Berger).

### 2.2 Multi-population PRS-CSx coupling

Per Ruan et al. 2022 [C7] Methods §"PRS-CSx", verbatim transcription:

> *"when SNP $j$ is available in multiple GWAS summary statistics, the continuous shrinkage prior is shared across populations (i.e., both $\phi$ and $\Psi_j$ do not depend on $k$)."*

For population $k \in \{1, \dots, K\}$ with sample size $N_k$ and standardized marginal effect $\hat\beta_{jk}$:

$$
\beta_{jk} \mid \psi_j \sim \mathcal{N}\!\left(0, \frac{\sigma_k^2}{N_k}\psi_j\right), \quad
\psi_j \mid \delta_j \sim \mathrm{Gamma}(a, \delta_j), \quad
\delta_j \mid \phi \sim \mathrm{Gamma}(b, \phi)
$$

So $\beta_{jk}$ is population-specific; $\psi_j$ and $\delta_j$ are **shared across all populations in which variant $j$ is observed**. The cross-population borrowing happens through these shared latent variables.

The posterior block update for population $k$ given $\Psi = \mathrm{diag}(\psi_1, \dots, \psi_M)$:

$$
\mathbb{E}[\beta_k \mid \hat\beta_k, \Psi] = (D_k + \Psi^{-1})^{-1}\,\hat\beta_k
$$

where $D_k$ is the per-population block-diagonal LD correlation matrix.

### 2.3 Gibbs sampler (per iteration)

Per `mcmc_gtb.py` source [C5] (verified by direct read 2026-05-11):

**Step 1 — per-population $\beta_k, \sigma_k^2$ updates.** For each population $k \in \{1, \dots, K\}$:
- Sample $\beta_k \sim \mathcal{N}(\mu_k, \Sigma_k)$ via Cholesky of $(D_k + \mathrm{diag}(1/\psi))$. Spec: same per-population precision matrix as our existing `_prscs_gibbs_block` (`pgs/prscs.py:182–195`), now per-population.
- Sample $\sigma_k^2 \sim \mathrm{Gamma}\bigl((N_k + p_k)/2,\, 1/\mathrm{err}_k\bigr)$.

**Step 2 — shared $\delta$ update.** For each variant $j$:

$$
\delta_j \sim \mathrm{Gamma}(a + b,\, 1/(\psi_j + \phi))
$$

**Step 3 — shared $\psi$ update with cross-population borrowing.** This is the key cross-pop coupling step. For each variant $j$:

$$
\psi_j \sim \mathrm{GIG}\!\left(\lambda = a - \frac{1}{2}\, n_{\text{grp},j},\quad \chi = 2\delta_j,\quad \rho = x_j\right)
$$

where $n_{\text{grp},j}$ is the **count of populations in which variant $j$ appears** (variable `n_grp[jj]` in `mcmc_gtb.py`) and $x_j = \sum_{k:\, j \in k} \beta_{jk}^2 / \sigma_k^2$.

**Step 4 — optional $\phi$ update (auto mode, deferred per scope §1):**

$$
\phi \sim \mathrm{Gamma}(M b + 1/2,\, 1/(\sum_j \delta_j + w)), \quad w \sim \mathrm{Gamma}(1, 1)
$$

After burn-in, post-burn-in $\beta_{jk}$ samples are averaged with thinning factor $T_{\text{thin}}$ (default 5, per [C5]) to produce per-population posterior mean effects $\bar\beta_{jk}$.

### 2.4 Standardized effect transformation

Per `parse_genet.parse_sumstats()` [C6]: incoming summary statistics are converted to per-SNP standardized effects:

- **with SE:** $\tilde\beta_{jk} = \beta_{jk} / (\mathrm{SE}_{jk}\sqrt{N_k})$
- **with P:** $\tilde\beta_{jk} = \mathrm{sign}(\beta_{jk})\,|\Phi^{-1}(P_{jk}/2)| / \sqrt{N_k}$

The MVP supports both schemas. The +SE schema is preferred per [C5] README's 2023-08-10 update (P-values < 1e-323 are truncated under the +P schema, losing precision at large effect sizes).

### 2.5 Dimension table

| Symbol | Meaning | Typical value |
|---|---|---|
| $K$ | discovery populations | 2–5 |
| $M$ | union variants across all populations | ~1.0–1.1 M (HapMap3) |
| $M_k$ | variants observed in population $k$ | $\le M$ |
| $N_k$ | per-population GWAS sample size | $10^4$–$10^6$ |
| $T$ | total MCMC iterations | $1000 \times K$ (default) |
| $B$ | burn-in iterations | $500 \times K$ (default) |
| $T_{\mathrm{thin}}$ | thinning factor | 5 (default) |
| $a, b$ | Strawderman–Berger prior shape | 1, 0.5 |
| $\phi$ | global shrinkage (fixed) | grid $\{10^{-6}, 10^{-4}, 10^{-2}, 1\}$ |
| $b_{\max}$ | max LD block size | ≤ 5000 (HapMap3 ldetect blocks) |

### 2.6 Default hyperparameters

| Parameter | Default | Source |
|---|---|---|
| `--n_iter` | $1000 \times K$ | [C5] README; [C5] `PRScsx.py` argparse defaults |
| `--n_burnin` | $500 \times K$ | same |
| `--thin` | 5 | same |
| `--a` | 1 | same |
| `--b` | 0.5 | same |
| `--phi` | required (no default in MVP; auto-learn deferred) | [C5] README |
| `--seed` | random by default; user-settable for reproducibility | [C5] README |

## 3. Module layout

### 3.1 New files

| Path | Purpose |
|---|---|
| `torchgwas/pgs/prscsx.py` | `PRSCSxModel` class; multi-population Gibbs sampler. Imports `sample_gig` and `_block_iter` from `prscs.py`; reuses `BasePGSMethod`, `LDReference`, `PGSResult` from `pgs/base.py`. |
| `torchgwas/pgs/ld_ref_multi.py` | `MultiAncestryLDReference` dataclass — `dict[str, LDReference]` keyed by ancestry code, plus shared SNP-info DataFrame. `load_multipop_ld_reference(ref_dir, ancestries)` constructor that ingests PRS-CSx HDF5 panels per [C6] format. |
| `torchgwas/pgs/sumstats_multi.py` | `parse_multipop_sumstats(files, populations, n_gwas)` — per-population sumstats harmonization with shared variant universe. |
| `torchgwas/pgs/snpinfo_mult.py` | Parser for PRS-CSx's `snpinfo_mult_1kg_hm3` / `snpinfo_mult_ukbb_hm3` master variant universe files (15 columns: `CHR SNP BP A1 A2 FRQ_AFR FRQ_AMR FRQ_EAS FRQ_EUR FRQ_SAS FLP_AFR FLP_AMR FLP_EAS FLP_EUR FLP_SAS`). |
| `tests/fixtures/prscsx/` | EUR + EAS chr22 fixture: per-pop sumstats (1000 SNPs), shared `.bim`, EUR + EAS chr22 LD panel slices (~200–400 MB each). Pre-downloaded via `pytest --setup-only`. |
| `tests/test_prscsx.py` | Tier 1 unit tests on the multi-pop Gibbs kernel + parse routines. |
| `tests/test_prscsx_parity_upstream.py` | Tier 2 parity vs PRS-CSx upstream Python. |
| `tests/test_prscsx_loaders.py` | LD panel + sumstats loader tests. |

### 3.2 Modified files

| Path | Change |
|---|---|
| `torchgwas/cli.py` | Add `pgs-fit-csx` subcommand. |
| `torchgwas/pgs/__init__.py` | Export `PRSCSxModel`, `MultiAncestryLDReference`, `parse_multipop_sumstats`, `load_snpinfo_mult`. |
| `torchgwas/pgs/scoring.py` | Extend `score_individuals` to optionally accept per-population posterior effects + a per-population score blend (the MVP just produces per-pop scores; the user does the blend). |
| `tests/test_streaming_memory.py` | Extend with a per-block memory regression test for PRS-CSx (see §4.3). |
| `bench/native_speedups.py` | Extend wall-time gate to include `pgs-fit-csx` per `feedback_regression_nets`. |
| `docs/cli.md` | Add `pgs-fit-csx` documentation. |
| `CLAUDE.md` | Update CLI subcommand count. Update Phase Index with Phase 58 entry. |

### 3.3 Files NOT touched (per brief §11.1)

- `torchgwas/pgs/prscs.py` — single-population; preserves V1-equivalence regression. **No edits.**
- `torchgwas/pgs/ld_ref.py` — single-ancestry container; PRS-CSx's multi-ancestry case lives in the new `ld_ref_multi.py`. **No edits.**
- `torchgwas/postgwas/_multi_ancestry.py` — MR-MEGA / MANTRA meta-analysis; genuinely separate from PRS construction (per brief §11.3). **No edits.**

### 3.4 Public API surface

```python
# torchgwas/pgs/prscsx.py
class PRSCSxModel(BasePGSMethod):
    """PRS-CSx multi-population polygenic score construction.

    Implements the coupled-shrinkage Gibbs sampler from Ruan et al. 2022 [C7].
    The continuous-shrinkage prior parameters (psi, delta, phi) are shared
    across populations; per-variant beta_k stays population-specific. This
    enables cross-ancestry borrowing of evidence at variants observed in
    multiple discovery cohorts.

    Args:
        phi: Global shrinkage (REQUIRED; no auto-learn in MVP). Recommended
            grid: 1e-6, 1e-4, 1e-2, 1 per [C7] Methods.
        n_iter: Total MCMC iterations (default 1000*K).
        n_burnin: Burn-in iterations (default 500*K).
        thin: Thinning factor (default 5).
        a, b: Strawderman-Berger prior shape (defaults 1, 0.5).
        seed: Random seed for reproducibility.

    Returns:
        PRSCSxResult with per-(population, variant) posterior mean effects
        beta_bar of shape (K, M_total), plus per-population sigma_k_squared
        trace and shared psi/delta posterior mean trace.
    """

    def fit(
        self,
        sumstats: list[SumstatsDF],          # one per population
        ld_ref: MultiAncestryLDReference,
        populations: list[str],              # ["EUR", "EAS", ...]
        n_gwas: list[int],                   # per-population GWAS sample size
        target_bim: pd.DataFrame,            # variant filter (no genotypes read)
    ) -> PRSCSxResult:
        ...

# torchgwas/pgs/ld_ref_multi.py
@dataclass
class MultiAncestryLDReference:
    """Container for per-ancestry LD reference panels."""
    panels: dict[str, LDReference]           # keyed by ancestry code
    snp_info: pd.DataFrame                   # shared variant universe
    panel_family: str                        # "1kg" or "ukbb"

def load_multipop_ld_reference(
    ref_dir: Path,
    ancestries: list[str],                   # ["EUR", "EAS", ...]
    panel_family: Literal["1kg", "ukbb"],
    chromosomes: Sequence[int] | None = None,
) -> MultiAncestryLDReference:
    """Load PRS-CSx HDF5 LD panels per ancestry.

    PRS-CSx's panels are HDF5 files (one per chromosome per ancestry) under
    {ref_dir}/ldblk_{family}_{ancestry}/. Each file has keys 'blk_N/ldblk'
    (LD correlation matrix) and 'blk_N/snplist' (variant IDs).
    See [C6] for format details.
    """
    ...
```

## 4. Data flow

### 4.1 End-to-end

```
Per-population sumstats (CSV/TSV, K files): SNP, A1, A2, BETA, SE, [P], N
   ↓ (parse_multipop_sumstats: standardize per-pop effects per §2.4)
StandardizedSumstats[K]
   +
Master snp_info file (snpinfo_mult_1kg_hm3): shared variant universe (M variants)
   +
Per-ancestry HDF5 LD panels (ref_dir/ldblk_{family}_{pop}/chr{C}.hdf5)
   ↓ (load_multipop_ld_reference)
MultiAncestryLDReference {
    panels: {"EUR": LDRef, "EAS": LDRef, ...},
    snp_info: DataFrame (M variants),
    panel_family: "1kg",
}
   +
Target .bim (variant filter only; no genotypes loaded)
   ↓ (PRSCSxModel.fit: align variants, run Gibbs sampler)
PRSCSxResult {
    beta_bar: Tensor[K × M_total],            # posterior mean effects per (pop, variant)
    sigma_k_squared_trace: Tensor[K × T_post],  # per-pop residual variance trace
    psi_posterior_mean: Tensor[M_total],      # shared psi posterior mean
    delta_posterior_mean: Tensor[M_total],    # shared delta posterior mean
    population_membership: Tensor[K × M_total],  # 0/1 mask
}
   ↓ (write per-(pop, chrom) effect-size files in PRS-CSx format)
out/<name>_{POP}_pst_eff_a{a}_b{b}_phi{phi}_chr{C}.txt
```

Downstream scoring (per-population PRS production) uses our existing `pgs/scoring.score_individuals` once per population. The per-population score linear blend on a held-out validation set is the user's responsibility (per [C7] "Recommended Analysis Approaches").

### 4.2 Multi-population variant intersection

Per `align_ldblk()` [C6]: the algorithm keeps the **union** of variants that appear in `snp_info` (the master HapMap3 panel) AND in **at least one** per-population sumstats file. Variants missing from population $k$'s sumstats are simply absent from population $k$'s posterior update — the sampler tracks per-variant population-membership counts via $n_{\text{grp},j}$ (per §2.3 step 3).

### 4.3 Memory footprint

Per brief §11.4:
- Per-block memory scales as $O(K \cdot b_{\max}^2)$ for LD matrices + $O(K \cdot b_{\max})$ for per-pop effects.
- With $b_{\max} \le 5000$ and $K = 2$: ~200 MB per block in float64.
- Streaming **across** blocks works (mirror existing `_block_iter`); streaming **within** a block would break the MCMC update.

Memory regression test (`tests/test_streaming_memory.py` extension):
- Run `pgs-fit-csx` with $K = 2$, max block size 5000, asserting peak RSS does not exceed 1.5 GB for the test fixture (gives 7.5x headroom over the per-block estimate).

### 4.4 Streaming compliance

PRS-CSx is summary-stats based, so the genotype-streaming `iter_chunks` contract from `feedback_streaming` does not literally apply. The relevant streaming concept is per-LD-block iteration (per `_block_iter` in `pgs/prscs.py`), which we mirror unchanged. Per master spec §6 and brief §11.4, this is documented in the spec docstring and confirmed by the per-block memory regression test above.

## 5. Validation gates

### 5.1 Tier 1 — math correctness (synthetic, deterministic)

Required tests in `tests/test_prscsx.py`:

- **Single-population reduction.** With $K = 1$, the multi-pop sampler must produce numerically equivalent posterior means to single-pop PRS-CS on the same input (within MCMC noise, ≤ 1e-4 relative on per-variant $\bar\beta$). Verifies the multi-pop generalization correctly nests the single-pop case. Compare against `torchgwas/pgs/prscs.py::PRSCSModel`.
- **Population-membership counting.** Hand-crafted 3-population fixture with known per-variant overlap pattern; assert `n_grp[j]` matches expected counts.
- **Shared $\psi$ update — analytical check.** Single-iteration test on a 5-variant, 2-population fixture with hand-computed $x_j = \sum_k \beta_{jk}^2 / \sigma_k^2$ and $\delta_j$; assert that the GIG sampler is called with parameters $(\lambda = a - n_{\text{grp},j}/2,\ \chi = 2\delta_j,\ \rho = x_j)$ exactly per §2.3 step 3. Use mocked GIG sampler returning a fixed value to make this deterministic.
- **Per-population $\sigma_k^2$ update.** Hand-computed Gamma shape/scale for a 50-variant, 2-pop fixture; assert sampled $\sigma_k^2$ marginal mean over 1000 draws is within 5% of analytical mean.
- **Standardized-effect transformation.** Both schemas (+SE and +P) on a 10-variant fixture; closed-form assertions to ≤ 1e-10 absolute.
- **Variant union with allele-flip.** Hand-coded 4-variant fixture where 2 variants have flipped alleles between populations; assert `align_ldblk` correctly applies sign corrections per `FLP_*` columns of `snpinfo_mult_*`.

### 5.2 Tier 2 — smoke parity vs PRS-CSx upstream Python

Required test in `tests/test_prscsx_parity_upstream.py`:

- **Fixture**: PRS-CSx bundled `test/` data (per brief §8) — EUR + EAS sumstats + shared `.bim` for 1000 SNPs on chromosome 22. ~200 KB sumstats; ~200 MB per ancestry for the chr22 LD panel slice.
- **Pre-flight install check**: PRS-CSx installable via `git clone https://github.com/getian107/PRScsx`; required Python deps `scipy + h5py + numpy`. **Pre-flight footprint check**: ≥ 30 GB free for canonical 2-ancestry install per brief §7. Memory headroom check per `feedback_preflight`.
- **PRS-CSx upstream pin**: master branch as of 2026-05-11 (latest tagged is v1.1.0 from 2023-08-11 but later in-place updates per brief §1; pin to specific commit SHA in `tests/conftest.py::PRSCSX_PINNED_SHA`).
- **Reference run** (per [C7] README "Test Data"):

  ```
  python PRScsx.py --ref_dir=$REF --bim_prefix=$BIM/test \
    --sst_file=$SS/EUR_sumstats.txt,$SS/EAS_sumstats.txt \
    --n_gwas=200000,100000 --pop=EUR,EAS --chrom=22 --phi=1e-2 \
    --seed=42 --out_dir=$OUT --out_name=test
  ```

  Estimated runtime: ~1 minute on 8 GB RAM per [C7] README.

- **Our run**: same inputs through `torchgwas pgs-fit-csx ...` with matching flag mappings + identical seed.
- **Comparison**: per-(POP, variant) posterior effect-size files compared row-by-row.
- **Tolerance** (per master spec §5; MVP scope): `floor + observed × 2`. Floor: 1e-5 absolute on posterior $\bar\beta_{jk}$ (effects are very small numbers, so absolute floor is generous). Observed divergence captured at first run.
- **Expected sources of divergence** to verify against tolerance:
  - GIG sampler: scipy's `geninvgauss` (existing `pgs/prscs.py` path) vs PRS-CSx's `gigrnd.py`. Both are rejection samplers; per-variant draws will differ by RNG sequence even at the same seed (different PRNG advancement). Tolerance must absorb this.
  - LD panel SVD projection (per `parse_ldblk` 2021-04-06 change in [C5]): if our HDF5 loader applies the same nearest-PSD projection, results match; if not, expect block-diagonal divergence.

### 5.3 Tier 2 — single-population reduction parity

Run our `PRSCSxModel` with $K = 1$ AND existing `PRSCSModel` on the same EUR sumstats fixture; assert per-variant $\bar\beta$ agrees within 1e-4 relative. Validates the multi-pop generalization correctly nests the single-pop case (also a Tier 1 sanity check; this is the Tier-2-against-existing-code variant).

### 5.4 Tier 3 — full-scale empirical (deferred per NA3)

Documented assertion (not run): on a 5-population HapMap3 dataset (~1M variants × {EUR, EAS, AFR, AMR, SAS}), our outputs match PRS-CSx upstream within `floor + observed × 2`. Per-population R² in a held-out target cohort matches the published median 52.3% improvement over single-pop PRS-CS (per [C7] Figure 3a) within Monte-Carlo error. Deferred to NA3.

## 6. CLI integration

### 6.1 New subcommand

```bash
torchgwas pgs-fit-csx \
    --ref-dir /path/to/PRScsx_panels/ \
    --bim-prefix data/target \
    --sst-file EUR_ss.tsv,EAS_ss.tsv \
    --n-gwas 200000,100000 \
    --pop EUR,EAS \
    --phi 1e-2 \
    --seed 42 \
    --n-iter 2000 \           # default = 1000 * K
    --n-burnin 1000 \         # default = 500 * K
    --thin 5 \
    --chrom 1-22 \
    --panel-family 1kg \      # or "ukbb"
    --out out/csx
```

### 6.2 Flag mapping (ours → PRS-CSx upstream)

| Ours | Upstream | Notes |
|---|---|---|
| `--ref-dir` | `--ref_dir` | identical |
| `--bim-prefix` | `--bim_prefix` | identical |
| `--sst-file` | `--sst_file` | identical (comma-separated, order matches `--pop` and `--n-gwas`) |
| `--n-gwas` | `--n_gwas` | identical |
| `--pop` | `--pop` | identical; allowed values `{AFR, AMR, EAS, EUR, SAS}` |
| `--phi` | `--phi` | identical (REQUIRED in MVP; auto-learn deferred) |
| `--seed` | `--seed` | identical |
| `--n-iter` | `--n_iter` | identical (default `1000 * K`) |
| `--n-burnin` | `--n_burnin` | identical (default `500 * K`) |
| `--thin` | `--thin` | identical (default 5) |
| `--chrom` | `--chrom` | identical (range syntax `1-22` or comma list) |
| `--a` | `--a` | identical (default 1) |
| `--b` | `--b` | identical (default 0.5) |
| `--out` | `--out_dir` + `--out_name` (combined) | we split internally by parsing the trailing path component |
| `--panel-family` | (implicit in `--ref_dir` choice) | we make it explicit to avoid filesystem ambiguity |

### 6.3 Output file format

Per [C7] README "Output Format":

```
out/csx_EUR_pst_eff_a1_b0.5_phi1e-2_chr22.txt
out/csx_EAS_pst_eff_a1_b0.5_phi1e-2_chr22.txt
```

Each file:
```
<chr>  <rsid>  <bp>  <A1>  <A2>  <posterior_effect_size>
22     rs5746647  16051249  C  G  -1.234e-05
```

No header row; whitespace-separated. Direct round-trip with PLINK 1.9 `--score ... sum` for downstream scoring.

### 6.4 Streaming compliance

Per `feedback_streaming` and brief §11.4: PRS-CSx is summary-stats based, so genotype-streaming via `iter_chunks` does not literally apply. The relevant per-block iteration uses `_block_iter` (mirror existing `pgs/prscs.py`). Per-block memory regression test in §5.1 enforces ceiling of 1.5 GB.

## 7. Native acceleration scope (deferred)

Per `Python-as-spec, native-as-shortcut` convention. Pure-torch path is the spec.

Candidate hot loops for a future Phase-41-style sub-phase (per brief §11.5):

- **GIG sampler** (`sample_gig` in `pgs/prscs.py:108`). Per-variant scalar draws via scipy's `geninvgauss`. PRS-CSx hits this $M$ times per iteration ($M \approx 10^6$ at HapMap3 scale across ancestries). The native `_prscs_native.prscs_sample_psi_delta` C++ entry already exists for single-pop; could be generalized to accept the per-variant population count $n_{\text{grp},j}$ as an extra argument. Defer to a sub-phase; MVP ships on scipy path.
- **Per-population block-Cholesky in step 1.** Vectorizable across populations in torch; native unlikely to outperform existing torch GPU path. Verify at profiling time.
- **Shared $\psi$ / $\delta$ updates** (steps 2–3). Vectorizable in torch; low compute relative to GIG sampling.

Speedup targets not committed; deferred to native sub-phase.

## 8. F3 severity application

Per master spec §7:

| Event | Action |
|---|---|
| Tier 1 test failure | Halt; fix the bug |
| Tier 2 parity divergence within `floor + observed × 2` | Pass; no findings entry |
| Tier 2 parity divergence outside tolerance | Document in `docs/validation_findings.md`; investigate (most likely RNG sequence — see §5.2 expected sources of divergence); does not block phase |
| Single-pop reduction (Tier 2 §5.3) divergence > 1e-4 relative | **Halt** — single-pop reduction failure means the multi-pop generalization is incorrect, not just numerically different |
| Per-block memory regression test exceeds 1.5 GB ceiling | **Halt**; investigate |
| ≥ 3 unrelated divergences during Tier B | C4 emergency stop |

## 9. Risks & open questions

### 9.1 Risks (from brief §11)

| ID | Risk | Mitigation |
|---|---|---|
| R-P58-1 | RNG sequence divergence between scipy `geninvgauss` and upstream `gigrnd.py` produces per-variant differences exceeding tolerance | Tolerance per §5.2 absorbs; if exceeded, port `gigrnd.py` algorithm verbatim into our `pgs/prscs.py::sample_gig` |
| R-P58-2 | LD panel SVD projection (nearest-PSD) differs between our HDF5 loader and upstream `parse_ldblk` | Mirror upstream's projection step in `load_multipop_ld_reference`; assert in unit test |
| R-P58-3 | Brief Issue #75 alleges upstream `main()` doesn't loop over populations correctly | Re-derive multi-pop behavior from the paper [C7] Methods, not from upstream's code organization. If our parity test diverges from upstream output specifically because upstream is broken, document and pin a specific upstream commit known to work. |
| R-P58-4 | scipy GIG sampler becomes a wall-time bottleneck at HapMap3 scale | Document expected slowdown; defer native acceleration to sub-phase |
| R-P58-5 | Per-variant population-membership $n_{\text{grp},j}$ accounting bug | Tier 1 unit test on hand-crafted overlap pattern catches this |
| R-P58-6 | User supplies discovery population whose super-pop label is unmapped | Validate `--pop` against `{AFR, AMR, EAS, EUR, SAS}` early with `ValueError` |
| R-P58-7 | User supplies admixed target cohort | Document warning in `pgs-fit-csx --help` per brief Issue #73; recommend ancestry-stratified analysis |
| R-P58-8 | LD panel ancestry mismatch with discovery cohort (e.g., 1000G EUR panel for a Finnish-only discovery) | Document in user guide; not a hard error; note that Ruan 2022 [C7] Methods uses super-population matching |
| R-P58-9 | Cold-start cost (per brief Issue #68) — first iteration appears to hang | Add log message at iteration 0 estimating total runtime; document expected ~1 min for chr22 fixture |

### 9.2 Open questions (deferred to implementation kickoff)

- **OQ-P58-1**: Should we accept user-supplied custom HDF5 panels (not from PRS-CSx's published distribution)? Default lean: no, MVP requires one of the two published panel families; custom panels deferred per brief Issue #72.
- **OQ-P58-2**: Should the per-population posterior effects be written as separate files (matching upstream) or as a single Parquet/HDF5 (more efficient)? Default lean: matching upstream for round-trip compatibility; opt-in alternative format via `--output-format parquet`.
- **OQ-P58-3**: Should `pgs-fit-csx` automatically run the downstream `score_individuals` when given a target genotype matrix? Default lean: no; keep PRS construction and scoring as separate CLI commands per existing `pgs-fit` / `pgs-score` convention.
- **OQ-P58-4**: Should we expose the auto-φ Gamma-prior path (currently deferred) behind an `--phi auto` flag for users who want it? Default lean: defer to follow-on phase as per scope §1; document the deferral in `pgs-fit-csx --help`.
- **OQ-P58-5**: Should `pgs-fit-csx` support multi-chain MCMC for R-hat convergence diagnostics (per brief §10 — upstream is single-chain)? Default lean: yes, expose `--n-chains 1` (default 1 for parity) but allow `--n-chains 4` for diagnostic mode using existing `pgs/diagnostics.py`. Document that `n_chains > 1` deviates from upstream for a useful reason.

## 10. Implementation phasing

### 10.1 Tier A — minimum viable shipping unit

Shippable when: `pgs-fit-csx` runs end-to-end on the bundled EUR + EAS chr22 fixture, produces output matching PRS-CSx upstream within tolerance, single-pop reduction parity passes.

| Step | Files | Test |
|---|---|---|
| **A1** Master SNP info loader | `pgs/snpinfo_mult.py` | `tests/test_prscsx_loaders.py` |
| **A2** HDF5 LD panel loader (per ancestry, per chromosome) | `pgs/ld_ref_multi.py` | `tests/test_prscsx_loaders.py` |
| **A3** Multi-population sumstats harmonization | `pgs/sumstats_multi.py` | `tests/test_prscsx_loaders.py` |
| **A4** Multi-population Gibbs kernel (steps 1–3 of §2.3; fixed φ) | `pgs/prscsx.py` | `tests/test_prscsx.py` Tier 1 unit tests |
| **A5** Per-(pop, chrom) output writer in PRS-CSx format | `pgs/prscsx.py` | `tests/test_prscsx.py` writer test |
| **A6** CLI subcommand | `cli.py` | `tests/test_cli.py` smoke |
| **A7** Tier 2 parity vs upstream on bundled fixture | `tests/test_prscsx_parity_upstream.py` | this test |
| **A8** Single-pop reduction parity (`K=1` matches `PRSCSModel`) | `tests/test_prscsx.py` | this test |
| **A9** Per-block memory regression test | `tests/test_streaming_memory.py` extension | this test |
| **A10** Wall-time regression net | `bench/native_speedups.py` extension | this test |

### 10.2 Tier B — parity-tightening + scaling

Shippable when: full 5-ancestry runs work, full HapMap3-scale fixtures pass, multi-chain R-hat available.

| Step | Files | Test |
|---|---|---|
| **B1** 5-ancestry support (AFR + AMR + EAS + EUR + SAS) | `pgs/prscsx.py` | parity test on 5-pop fixture |
| **B2** UKB panel family support | `pgs/ld_ref_multi.py` | parity test with `--panel-family ukbb` |
| **B3** Multi-chain wrapper for R-hat diagnostics | `pgs/prscsx.py` + `pgs/diagnostics.py` | `tests/test_prscsx_diagnostics.py` |
| **B4** φ grid search support (run multiple φ values, output per-φ files) | `cli.py` extension | parity test |
| **B5** HapMap3-scale fixture parity (1M variants × 2 ancestries) | `tests/test_prscsx_parity_upstream.py` extension | this test |
| **B6** Reproducibility hardening (MKL/OMP thread pinning per brief §7 caveat) | `pgs/prscsx.py` | wall-time consistency test |

### 10.3 Tier C — nice-to-haves & follow-on

| Step | Files | Test |
|---|---|---|
| **C1** Native acceleration of GIG sampler with `n_grp` arg | `csrc/prscs_native_multi.cpp` | parity + bench |
| **C2** φ auto-learn (Gamma(1,1) hyperprior, per §2.3 step 4) | `pgs/prscsx.py` | parity test with `--phi auto` |
| **C3** `--meta=True` posterior averaging output | `pgs/prscsx.py` extension | parity test |
| **C4** `--write_pst` posterior trace output for debugging | `pgs/prscsx.py` extension | smoke test |
| **C5** Validation-set linear-combination workflow helper | new `pgs/multipop_blend.py` | unit test on synthetic per-pop scores |
| **C6** Custom user-supplied LD panel ingestion | `pgs/ld_ref_multi.py` extension | unit test |
| **C7** Documented preparation for cross-ancestry MR / fine-mapping integration | docs only | n/a |

## 11. Citations

- **[C1] PRS-CSx GitHub repository.** <https://github.com/getian107/PRScsx>. README + version history.
- **[C2] PRS-CSx LICENSE (MIT).** <https://github.com/getian107/PRScsx/blob/master/LICENSE>.
- **[C3] PRS-CSx release v1.1.0 (2023-08-11).** <https://github.com/getian107/PRScsx/releases/tag/v1.1.0>.
- **[C4] PRS-CSx `PRScsx.py` main entry script.** <https://raw.githubusercontent.com/getian107/PRScsx/master/PRScsx.py>. Argument parsing, default scaling of `n_iter` / `n_burnin` to $1000K$ / $500K$.
- **[C5] PRS-CSx `mcmc_gtb.py` Gibbs sampler source.** <https://raw.githubusercontent.com/getian107/PRScsx/master/mcmc_gtb.py>. Per-population $\beta_k$, $\sigma_k^2$ updates; shared $\delta$ and $\psi$ updates with `n_grp[jj]` cross-population count; auto-φ Gamma update.
- **[C6] PRS-CSx `parse_genet.py` I/O source.** <https://raw.githubusercontent.com/getian107/PRScsx/master/parse_genet.py>. `parse_ref`, `parse_bim`, `parse_sumstats`, `parse_ldblk`, `align_ldblk` signatures and column layouts.
- **[C7] Ruan, Y. et al. (2022).** *Improving polygenic prediction in ancestrally diverse populations.* *Nature Genetics* 54:573–580. DOI: <https://doi.org/10.1038/s41588-022-01054-7>. Methods §"PRS-CSx" prior hierarchy; Figures 3a/3b/4a validation results.
- **[C8] Open-access mirror of [C7]: PMC9117455.** <https://pmc.ncbi.nlm.nih.gov/articles/PMC9117455/>.
- **[C9] Ge, T. et al. (2019).** *Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors.* *Nature Communications* 10:1776. <https://doi.org/10.1038/s41467-019-09718-0>. Original PRS-CS prior hierarchy that PRS-CSx generalizes; the prior our existing `pgs/prscs.py` implements verbatim.
- **[C10] Ge, T. et al. (2022).** *Development and validation of a trans-ancestry polygenic risk score for type 2 diabetes in diverse populations.* *Genome Medicine* 14:70. Applied PRS-CSx reference cited by [C1].
- **[C11] Kachuri, L. et al. (2024).** *Principles and methods for transferring polygenic risk scores across global populations.* *Nature Reviews Genetics* 25:8–25. Best-practices review covering PRS-CSx vs MultiPRS / PolyPred.
- **[C12] Issue #67 — small non-EUR GWAS sample sizes.** <https://github.com/getian107/PRScsx/issues/67>.
- **[C13] Issue #68 — PRSCSx taking so long (cold-start cost).** <https://github.com/getian107/PRScsx/issues/68>.
- **[C14] Issue #72 — independent LD blocks for custom panels.** <https://github.com/getian107/PRScsx/issues/72>.
- **[C15] Issue #73 — PRS-CSx ancestry selection in admixed samples.** <https://github.com/getian107/PRScsx/issues/73>.
- **[C16] Issue #75 — alleged single-population logic regression.** <https://github.com/getian107/PRScsx/issues/75>. UNVERIFIED (per brief §10).
- **[C17] Issue #76 — validation set for tuning and weight estimation.** <https://github.com/getian107/PRScsx/issues/76>.
- **[C18] PRS-CSx posterior weights data release (Dropbox).** <https://www.dropbox.com/sh/5v1bzlukxoor9fi/AACA580wl_gNKapqWvx3siOza?dl=0>.
- **[C19] PRS-CSx Broad mirror for LD panels.** <https://personal.broadinstitute.org/hhuang//public//PRS-CSx/Reference>.

## 12. Approval & next step

This sub-spec is committed once approved alongside the master and the other three sub-specs. Implementation is DEFERRED until validation campaign Pillars A–D and the three NA tasks (NA1, NA2, NA3) close per master spec §3. Independent of X-chrom refactor sequencing (PRS-CSx is summary-stats based; no genotype matrix involved). Can be implemented in parallel with Phase 59 PolyFun per master spec §3, subject to `feedback_parallel_agents_no_shortcuts` constraint.

At implementation kickoff, this spec is fed to `superpowers:writing-plans` to produce the per-Tier implementation plan.
