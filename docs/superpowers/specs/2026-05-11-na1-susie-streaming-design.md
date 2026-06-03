# NA1 — `bayes-scan-rss` (SuSiE-RSS sumstats fine-mapping) — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven)
- **Date:** 2026-05-11
- **Branch:** `research/na1-susie-streaming` (worktree at `.claude/worktrees/na1-susie-streaming/`)
- **Research brief:** [`../research/na1-susie-streaming-research.md`](../research/na1-susie-streaming-research.md)
- **Upstream task definition:** `docs/superpowers/SESSION_HANDOFF.md` § "Task NA1 — bayes-scan SuSiE streaming"
- **Sequencing:** independent of X-chrom refactor; gated on either rebase onto `efficiency/streaming-scan-audit` OR merge of that branch to master (per A0 sequencing decision below)
- **Living spec.** Tier A is fixed; Tier B / C may be revised at implementation kickoff.

---

## 1. Goal & scope

Add SuSiE-RSS (Zou, Carbonetto, Wang, Stephens 2022 [C4]) — the summary-statistics-input variant of SuSiE — to TorchGenomics as a new CLI subcommand `bayes-scan-rss`. SuSiE-RSS operates on per-locus z-scores + an LD reference matrix, replacing the raw-genotype $X^\top X / n$ in the per-layer Single Effect Regression (SER) update with the LD reference $R$. Memory bound becomes per-locus $O(p^2)$ for $R$ + $O(L \cdot p)$ for posteriors — typically ~200 MB per LD block at $p \le 5000$, vs $O(n \cdot p)$ for raw-G SuSiE which is up to 800 GB at UKB chr22 scale.

**Why now**: the existing `bayes-scan` CLI (raw-G `BayesianVS.fit`) is the only remaining materialized scan path per `docs/efficiency/streaming_audit.md`. SuSiE-RSS:
- Aligns with the canonical biobank fine-mapping workflow (sumstats → fine-map per locus on sumstats + LD).
- Composes cleanly with Phase 59 PolyFun (which is RSS-based per `docs/superpowers/specs/2026-05-11-phase-59-polyfun-design.md`).
- Reuses TorchGenomics' existing `postgwas` LD infrastructure.
- Ships with bounded scope (~1 week of focused work) per the brainstorming Decision 1 (Path D only; Path A chunked IBSS deferred to follow-on).

**Brainstorming decisions locked** (all from 2026-05-11 brainstorming session with the user):

| # | Decision | Resolution |
|---|---|---|
| 1 | Path priority | **Path D (SuSiE-RSS) only** for this spec. Path A (chunked IBSS for raw-G `bayes-scan`) deferred to a follow-on phase per OQ-NA1-1. Paths B (SVI) and C (sparse-G) deferred indefinitely (research-grade per brief §7.3). |
| 2 | CLI shape | **New subcommand `bayes-scan-rss`** (mirrors Phase 40 `pgs-fit` precedent: separate subcommands when input shape changes). |
| 3 | Fidelity tolerances | Brief proposal + add **β_mean / β_sd Pearson correlation ≥ 0.999** per scientific-rigor mandate. |
| 4 | LD reference source | **`--ld-ref` (pre-built file) AND `--geno` (in-sample LD per locus)** — both modes, matches Phase 59 PolyFun. |
| 5 | Materialized `bayes-scan` guardrail | **Soft `UserWarning` when `p > 10000`** pointing at `bayes-scan-rss`. Non-blocking; doesn't break plant-breeding-scale workflows (user's domain per `user_role` memory). |

**In scope (Path D MVP):**

- New class `BayesianVSRss` in new file `torchgenomics/models/bayesian_vs_rss.py` (Approach 1 from brainstorming — separate file preserves all 30+ existing `tests/test_bayesian_vs.py` parity tests with zero modification).
- New CLI subcommand `bayes-scan-rss` accepting `--sumstats` + (`--ld-ref` OR `--geno`) + `--regions` + `--max-num-causal` + `--coverage` + `--purity` + `--prior-pi` (D3 shim).
- Block decomposition as **Tier A**: auto-detect via `torchgenomics.ld.detect_blocks` if `--regions` not passed; per-block IBSS (one block of $R$ in RAM at a time).
- Multi-format LD ref loader: `.pt` (existing TorchGenomics), `.npz` (PolyFun) at Tier A; `.bcor` (FINEMAP) deferred to Tier B.
- LD reference metadata schema + cohort-mismatch detector (warn on mismatch, refuse on hard mismatch e.g. different genome build).
- D3 backward-compat shim: `prior_pi_per_snp: Optional[Tensor] = None` keyword preserves uniform-prior default; per-SNP vector triggers PolyFun-style prior injection.
- External validation harness `validation/external/susieR/` mirroring Pillar B layout.
- Tier 2 parity vs `susieR::susie_rss()` on MDP-derived per-locus fixture within tolerances of §5.

**Explicitly out of scope (deferred):**

- **Path A (chunked IBSS for raw-G `bayes-scan`)** — separate follow-on phase per OQ-NA1-1; tracked as "NA1-followon".
- **Path B (Stochastic VI)** — research-grade per brief §3; no published SVI-on-SuSiE algorithm; defer indefinitely.
- **Path C (Sparse-G representation)** — 2–4 week refactor of centering + rotation algebra per brief §4; defer indefinitely.
- **Multi-trait fine-mapping** (`mvSusieR`-equivalent) — Tier C C4.
- **Cross-ancestry SuShiE** [C9] — Tier C C5.
- **FINEMAP backend integration** — Tier B B2 (LD-ref reader only); shelling out to FINEMAP binary deferred indefinitely.
- **Hard deprecation of `bayes-scan`** — only soft warning per Decision 5.
- **Deletion of `BayesianVS.fit()` raw-G path** — preserved for plant-breeding-scale workflows.

## 2. Algorithm reference

Per Zou, Carbonetto, Wang, Stephens (2022) [C4] *PLOS Genetics* — the canonical SuSiE-RSS derivation. Notation matches their §2.

### 2.1 Inputs

- $z \in \mathbb{R}^p$ — per-variant z-scores. Spec accepts `BETA + SE + N` columns and converts internally as $z_j = \hat\beta_j / \hat{se}_j$.
- $R \in \mathbb{R}^{p \times p}$ — LD correlation matrix from a reference cohort matched to the GWAS cohort. Per [C4] §6, the dominant SuSiE-RSS failure mode is cohort mismatch.
- $n$ — GWAS sample size used in computing the sumstats.
- $L$ — number of single-effect layers (default 10 per [C4]; matches [C1] SuSiE default).
- $\boldsymbol\pi \in \mathbb{R}^p$ — per-variant prior inclusion probabilities. Default uniform $1/p$. The Phase 59 PolyFun spec D3 shim allows per-SNP priors via `--prior-pi` accepting either scalar or vector.

### 2.2 Single Effect Regression (SER) on summary data

Per [C4] §2 eq. 8–10, for each layer $l$ and variant $j$:

$$
\sigma_{l,j}^2 = \frac{1}{R_{jj}/\sigma_{l,prior}^2 + n}
$$

$$
\mu_{l,j} = \sigma_{l,j}^2 \cdot z_j \cdot \sqrt{n}
$$

$$
\log\mathrm{BF}_{l,j} = \tfrac{1}{2} \log\frac{\sigma_{l,j}^2}{\sigma_{l,prior}^2} + \tfrac{1}{2} \frac{\mu_{l,j}^2}{\sigma_{l,j}^2}
$$

$$
\alpha_{l,j} = \frac{\pi_j \exp(\log\mathrm{BF}_{l,j})}{\sum_{j'} \pi_{j'} \exp(\log\mathrm{BF}_{l,j'})}
$$

(softmax over BFs weighted by per-SNP priors — exactly where the Phase 59 D3 `prior_pi_per_snp` plugs in.)

### 2.3 IBSS update

Per [C4] §2 eq. 11. After each layer $l$ updates, the running fitted "effect" used by the *other* layers is recomputed as $\mathbf{b}_l = \boldsymbol\alpha_l \odot \boldsymbol\mu_l$, and the partial residual for layer $l$ at iteration $t+1$ is:

$$
\tilde{z}_l^{(t+1)} = z - R \sum_{l' \neq l} \mathbf{b}_{l'}^{(t)}
$$

**$R$ replaces $X^\top X / n$** vs raw-G SuSiE — the algorithmic substitution that makes RSS work on summary statistics.

### 2.4 ELBO for convergence diagnosis

Per [C4] §A.2 supplementary. Monotone non-decreasing per IBSS iteration — same property as raw-G SuSiE per [C1]. Existing `bayesian_vs.py::_susie_compute_elbo` is **not** directly reusable (uses `X^\top X` form); we re-derive for RSS in `bayesian_vs_rss.py::_compute_elbo_rss`.

### 2.5 Credible-set construction

Per [C1] §3.4. For each layer $l$:
1. Sort $\boldsymbol\alpha_l$ descending.
2. Take the smallest set of variants whose cumulative $\alpha$ ≥ `coverage` threshold (default 0.95).
3. Enforce a "purity" check via pairwise $|R_{jk}|$ ≥ `purity` threshold (default 0.5; matches `susieR::susie_rss(min_abs_corr=0.5)`).

Pairwise $|R_{jk}|$ comes directly from $R$ — no need to re-read genotypes (a meaningful win vs raw-G SuSiE which needs `X[:, candidates]`).

### 2.6 Block decomposition (Tier A per Section 3 design lean)

When the locus exceeds a memory threshold (default `--block-size-threshold 5000` SNPs, mirroring PolyFun's [C5] convention), the LD ref is decomposed into LD blocks. Block boundaries auto-detected via `torchgenomics.ld.detect_blocks` if user doesn't pass `--regions`. Per-block IBSS is mathematically equivalent to dense IBSS on the full locus when the block boundaries are at near-zero-LD positions (this is the standard fine-mapping practice; PolyFun [C10] uses ldetect blocks averaging ~1500 SNPs).

**Block decomposition correctness invariant** (Tier 1 §5 test): block decomp on `p = 100` fixture (4 blocks of 25) matches dense `p = 100` run to ≤ 1e-10 absolute. If blocks are correctly chosen, this is exact, not approximation.

### 2.7 Dimension table

| Symbol | Meaning | Typical biobank value |
|---|---|---|
| $p$ | variants in locus | 100–10,000 |
| $L$ | single-effect layers | 10 (default) |
| $n$ | GWAS sample size | 10K–500K |
| $K$ | credible sets returned | ≤ $L$ |
| $p_{block,max}$ | max block size | ≤ 5000 (default; ldetect-derived in practice) |

### 2.8 Memory bound

Per locus: $O(p_{block,max}^2)$ for $R_{block}$ + $O(L \cdot p)$ for $\boldsymbol\alpha, \boldsymbol\mu, \boldsymbol\sigma^2$ + $O(p)$ for $z$. At $p = 200{,}000$ with $p_{block,max} = 5000$: ~200 MB per block in float64 + ~16 MB for posteriors. Total ~216 MB — fits commodity RAM. Versus the raw-G `bayes-scan` 800 GB at the same locus.

## 3. Module layout

### 3.1 New files

| Path | Purpose |
|---|---|
| `torchgenomics/models/bayesian_vs_rss.py` | `BayesianVSRss` class implementing SuSiE-RSS per §2. Pure-torch path is the spec body per `Python-as-spec, native-as-shortcut`. |
| `torchgenomics/postgwas/_ld_ref_loader.py` | Multi-format LD reference loader (`.pt` + `.npz` at Tier A; `.bcor` at Tier B). In-sample LD via `compute_in_sample_ld(genotype_path, locus)` for the `--geno` mode per Decision 4. |
| `torchgenomics/postgwas/_ld_ref_metadata.py` | LD reference metadata schema + cohort-mismatch detector. Stamps `cohort_id`, `n`, `build`, `panel_provenance` into LD ref files at build time; refuses-with-warning when `bayes-scan-rss` sees a mismatch with the sumstats cohort. |
| `tests/test_bayesian_vs_rss.py` | Tier 1 unit tests for the RSS algorithm (closed-form / scipy comparisons). |
| `tests/test_ld_ref_loader.py` | Tier 1 tests for the loader + metadata. |
| `validation/external/susieR/install.sh` | Install `susieR` via CRAN; idempotent; pre-flight per `feedback_preflight`. Falls back to containerized R per R-NA1-2 mitigation. |
| `validation/external/susieR/fetch_data.sh` | Provision the MDP-derived per-locus `(z, R, n)` triple from `benchmark/data/`. |
| `validation/external/susieR/run_susieR.sh` | R script invokes `susieR::susie_rss(z, R, n, L=10, coverage=0.95)`. Captures credible sets, PIPs, posterior $\beta_{mean}$ + $\beta_{sd}$, ELBO at convergence, wall-time, peak RSS. |
| `validation/external/susieR/run_torchgenomics.sh` | Same fixture through our `torchgenomics bayes-scan-rss` with identical flags. |
| `validation/external/susieR/compare.py` | Compute the 6 tolerance metrics from §5; emit findings ledger row per F3 severity. |
| `validation/external/susieR/README.md` | Operations doc; how to run; expected runtime; failure modes. |
| `tests/test_external_susieR.py` | `pytestmark = [pytest.mark.external, pytest.mark.golden]` — invokes `compare.py` flow as a Tier 2 test. |

### 3.2 Modified files

| Path | Change | Backward compat |
|---|---|---|
| `torchgenomics/models/bayesian_vs.py` | Add `UserWarning` in `fit()` when `p > 10000` per Decision 5. NO algorithmic change. | YES — warning only |
| `torchgenomics/cli.py` | Add `bayes-scan-rss` subcommand handler. | YES — additive |
| `docs/cli.md` | Document new subcommand. | YES |
| `tests/test_streaming_memory.py` | Add memory regression for `bayes-scan-rss` (peak ≤ `O(p_max_block^2 + L * p_total)`). | YES — additive |
| `bench/native_speedups.py` | Add wall-time gate for `bayes-scan-rss` per `feedback_regression_nets`. | YES — additive |
| `CLAUDE.md` | Update CLI subcommand count (37 → 38); add new module to `models` list. | YES |

### 3.3 Files NOT touched

- `torchgenomics/models/bayesian_vs.py` `BayesianVS` class internals — preserves V1 parity tests; only the warning in `fit()` is added.
- `torchgenomics/postgwas/_finemapping.py` — credible-set utilities reused as-is.
- `torchgenomics/pgs/ld_ref.py` (existing single-population container) — unrelated; SuSiE-RSS LD ref is per-locus, not per-ancestry.

### 3.4 Public API surface

```python
# torchgenomics/models/bayesian_vs_rss.py
class BayesianVSRss:
    """SuSiE-RSS fine-mapping on summary statistics.

    Implements the algorithm of Zou, Carbonetto, Wang, Stephens (2022) [C4].
    Replaces the raw-genotype X^T X / n in raw-G SuSiE with an LD reference
    matrix R, enabling fine-mapping from sumstats + LD without ever loading
    the original genotype matrix.

    Args:
        max_num_causal: L (single-effect layers, default 10 per [C4]).
        coverage: credible-set coverage threshold (default 0.95).
        purity: minimum |R_jk| within a credible set (default 0.5,
            matches susieR::susie_rss(min_abs_corr=0.5)).
        prior_pi_per_snp: optional per-SNP prior inclusion probabilities
            (D3 shim from Phase 59 PolyFun spec; default None = uniform 1/p).
        block_size_threshold: max p per block before decomposition kicks in
            (default 5000 per PolyFun [C5] / ldetect convention).
    """

    def fit_rss(
        self,
        z: Tensor,                                    # (p,) z-scores
        R: Tensor | LDReference,                      # (p, p) or block-decomposed
        n: int,
        variant_meta: VariantMeta,
        *,
        regions: Optional[list[Region]] = None,       # block boundaries
        prior_pi_per_snp: Optional[Tensor] = None,    # D3 shim
    ) -> BayesianVSRssResult:
        ...
```

## 4. Data flow

```
Per-trait sumstats (CHR, BP, SNP, A1, A2, BETA, SE, [Z], N)
   ↓ (CLI: --sumstats)
Internal canonical: (chr, bp, snp, a1, a2, z, n) DataFrame
   +
Either: --ld-ref ld_chr22.pt    OR    --geno panel.bed
   ↓ (postgwas/_ld_ref_loader.py)
LD reference R (per-locus or per-block; metadata-stamped per §3.1)
   +
Either: --regions regions.tsv    OR    auto-detected via torchgenomics.ld.detect_blocks
   ↓ (BayesianVSRss.fit_rss per locus, per block)
Per-locus posteriors: alpha (L × p), mu (L × p), sigma^2 (L × p), credible_sets, PIPs, beta_mean, beta_sd, ELBO
   ↓ (output writer: PolyFun-compatible columns)
out/finemap.tsv: SNP, CHR, BP, A1, A2, Z, N, PIP, BETA_MEAN, BETA_SD, CREDIBLE_SET
```

**Streaming compliance** (per `feedback_streaming` and master spec §6):
- `bayes-scan-rss` is summary-stats based; the genotype `iter_chunks` streaming concept doesn't literally apply.
- Block decomposition (Tier A per §2.6) IS the streaming primitive — one block of $R$ in RAM at a time.
- Memory regression test in `tests/test_streaming_memory.py`: assert peak RSS during `bayes-scan-rss` ≤ `O(p_max_block^2 + L * p_total)`.

## 5. Validation gates

### 5.1 Tier 1 — math correctness (synthetic, deterministic)

Tests in `tests/test_bayesian_vs_rss.py`:

- **SER posterior closed-form on synthetic z** — hand-crafted $p = 50$ fixture with known $z$, $R$, $n$, $\sigma^2_{prior}$; assert $(\sigma_{l,j}^2, \mu_{l,j}, \log\mathrm{BF}_{l,j})$ to ≤ 1e-10 absolute against §2.2 equations.
- **Softmax over BFs with per-SNP priors** — extreme prior on one SNP ($\pi_j = 0.99$) drives $\alpha_{l,j} \approx 0.99$ regardless of moderate $z_j$; uniform prior recovers exact softmax-over-BFs. Validates D3 shim wiring (§3.4 `prior_pi_per_snp`).
- **IBSS update equation** — single-iteration step on $L = 2$ fixture; manually compute $\tilde{z}_l$ per §2.3 and assert match.
- **Block-decomposition correctness** — split $p = 100$ fixture into 4 blocks of 25; assert posterior matches dense $p = 100$ run to ≤ 1e-10 absolute (block decomp must be algorithmically equivalent, not approximation).
- **ELBO monotone non-decreasing** — assert `elbo[t+1] >= elbo[t] - 1e-9` over 50 iterations on 5 random fixtures (per [C4] §A.2).
- **PIP formula** — `pip[j] = 1 - prod_l(1 - alpha[l, j])` per [C1] eq. 12; closed-form check.
- **Credible-set construction** — hand-crafted $\alpha$ on $p = 20$ fixture with known purity matrix; assert credible-set membership matches expected.

Tests in `tests/test_ld_ref_loader.py`:

- **Multi-format round-trip** — write `.pt`, `.npz`; read back; assert tensor equality.
- **In-sample LD computation** — synthetic genotype `G`; assert `compute_in_sample_ld(G)` matches `np.corrcoef(G.T)` to ≤ 1e-10.
- **Cohort-mismatch detector** — LD ref stamped with `cohort_id="UKB"` + sumstats with `cohort_id="FinnGen"` → assert `UserWarning` raised; LD ref with `build="GRCh38"` + sumstats with `build="GRCh37"` → assert hard `ValueError` raised.
- **Block auto-detection** — synthetic LD with planted block boundaries; assert `detect_blocks` recovers them.

### 5.2 Tier 2 — smoke parity vs `susieR::susie_rss()` (golden marker)

Test in `tests/test_external_susieR.py` (marker: `pytestmark = [pytest.mark.external, pytest.mark.golden]`):

- **Pre-flight**: `validation/external/susieR/install.sh` installs `susieR` via CRAN if not present; ≥ 2 GB free disk, ≥ 4 GB RAM check per `feedback_preflight`. Falls back to containerized R (`docker run --rm rocker/r-ver:latest R -e 'install.packages("susieR")'`) per R-NA1-2 mitigation.
- **Fixture**: MDP-derived per-locus `(z, R, n)` triple. Build via `lmm-scan` on MDP genotype + phenotype → take a window of $p = 500$ SNPs around the strongest hit → `compute_pairwise_ld` for $R$. Reproducible from `validation/external/susieR/fetch_data.sh`.
- **Reference run**: R script invokes `susieR::susie_rss(z, R, n, L=10, coverage=0.95)`; captures credible sets, PIPs, posterior $\beta_{mean}$ + $\beta_{sd}$, ELBO at convergence, wall-time, peak memory (RSS via `/usr/bin/time -v`).
- **Our run**: same inputs through `torchgenomics bayes-scan-rss --max-num-causal 10 --coverage 0.95`; same captures.
- **Comparison** in `compare.py`:

| Metric | Threshold | Source |
|---|---|---|
| Credible-set Jaccard | **≥ 0.95** (hard) | `SESSION_HANDOFF.md` NA1 success criteria |
| PIP correlation (PIP > 0.1 in either) | ≥ 0.99 | brief §8.2 |
| β_mean correlation (Pearson) | ≥ 0.999 | Decision 3 (rigor mandate addition) |
| β_sd correlation (Pearson) | ≥ 0.999 | Decision 3 (rigor mandate addition) |
| ELBO relative diff at convergence | ≤ 1e-4 | brief §8.2 |
| Wall-time (p ≤ 10K) | ≤ 2× susieR | brief §8.2 |

- **Failure mode**: any threshold violation → log to `docs/validation_findings.md` per F3 severity; phase does NOT block on documented divergences (post-V1).
- **Observed-then-floored caveat** (per `feedback_validation_spec`): the thresholds above are **floors**. First Tier 2 run captures observed agreement; if observed is dramatically tighter (e.g., Jaccard = 1.000 every time), the spec records that for transparency but the threshold floor stays as documented.

### 5.3 Tier 3 — full-scale empirical (deferred per NA3)

Documented assertion (not run): on a UKB-derived per-locus `(z, R, n)` for a known fine-mapped locus (e.g., FTO for BMI per Yang et al. 2007), `bayes-scan-rss` recovers the published credible set within Jaccard ≥ 0.95 of the canonical PolyFun result. Deferred to NA3 task per `project_next_agent_tasks` memory.

### 5.4 Phase 59 PolyFun §5.5 backward-compat regression net

Per Phase 59 §5.5 contract: all existing `tests/test_bayesian_vs.py` tests run unchanged. NA1 spec preserves this by (a) not modifying `BayesianVS.fit()` algorithmic body (only adding the `UserWarning`), and (b) adding the new `BayesianVSRss` class in its own file. Verified by full-suite run as part of Tier A step A12.

## 6. CLI integration

```bash
torchgenomics bayes-scan-rss \
    --sumstats hits.tsv \              # Z, BETA, SE, A1, A2, CHR, BP, SNP, N columns
    --ld-ref ld_chr22.pt \             # PRE-BUILT LD ref (.pt or .npz) — OR --geno
    --geno panel.bed \                 # alt: compute in-sample LD per locus
    --regions regions.tsv \            # locus list (chr, start, end); auto-detected if omitted
    --max-num-causal 10 \              # L
    --coverage 0.95 \                  # credible-set coverage threshold
    --purity 0.5 \                     # credible-set purity threshold (susieR min_abs_corr)
    --prior-pi 0.01 \                  # OR a path to a per-SNP prior file (D3 shim)
    --block-size-threshold 5000 \      # block decomposition trigger
    --output out/finemap.tsv \
    --threads 4
```

**Flag mapping (ours → susieR):**

| Ours | susieR | Notes |
|---|---|---|
| `--sumstats` | (z arg) | We accept BETA+SE+N; convert internally to z |
| `--ld-ref` | (R arg) | Multi-format loader from `_ld_ref_loader.py` |
| `--geno` | (X arg, in `susie()`) | Triggers in-sample LD per locus (Phase 59 PolyFun parity per Decision 4) |
| `--regions` | (per-call locus) | Block decomposition Tier A per §2.6 |
| `--max-num-causal` | `L` | identical |
| `--coverage` | `coverage` | identical |
| `--purity` | `min_abs_corr` | susieR's name; we use `--purity` (clearer) |
| `--prior-pi` | `prior_weights` | D3 shim — accepts scalar OR per-SNP vector path |
| `--output` | (write to file) | TSV with PolyFun-compatible columns |

**Output schema** (matches Phase 59 PolyFun §4.4 for downstream tool compatibility):

```
SNP   CHR   BP   A1   A2   Z   N   PIP   BETA_MEAN   BETA_SD   CREDIBLE_SET
```

`CREDIBLE_SET` = integer index of credible set (0 = not in any CS); matches PolyFun convention so Phase 59 aggregation utilities work without translation.

**Soft warning on materialized `bayes-scan` (Decision 5):**

In `torchgenomics/models/bayesian_vs.py::BayesianVS.fit()`:

```python
if G.shape[1] > 10000:
    warnings.warn(
        f"BayesianVS.fit called on locus with p={G.shape[1]} > 10000. "
        f"This will materialize ~{G.shape[0] * G.shape[1] * 8 / 1e9:.1f} GB. "
        f"For large loci, prefer 'torchgenomics bayes-scan-rss' which operates "
        f"on summary statistics + LD reference (per-locus memory ~O(p^2)). "
        f"See docs/cli.md#bayes-scan-rss.",
        UserWarning,
        stacklevel=2,
    )
```

CLI handler also emits this warning before invoking the model (so users see it whether they use Python API or CLI).

## 7. Native acceleration scope (deferred)

Per `Python-as-spec, native-as-shortcut` convention. Pure-torch path is the spec.

Candidate hot loops for a future Phase-41-style sub-phase:

- **Per-block softmax + log-BF computation** — vectorizable in torch via batched `torch.logsumexp` and `torch.softmax`; native unlikely to outperform existing torch path. Verify at profiling time.
- **Per-block IBSS update (`R @ b_l` matmul)** — already covered by torch BLAS; native path probably moot.
- **Credible-set purity check (pairwise `|R_jk|` lookup)** — small workload (~|CS|² pairs); not a bottleneck.

Speedup targets not committed; deferred to native sub-phase if profiling identifies a hot loop.

## 8. F3 severity application

Per master spec §7 — post-V1 → divergences documented, regressions in existing paths halt:

| Event | Action |
|---|---|
| Tier 1 test failure | **Halt**; bug in our RSS implementation |
| Tier 2 vs `susieR::susie_rss` divergence within tolerances (§5.2 table) | Pass; no findings entry |
| Tier 2 divergence outside tolerance | **Document** in `docs/validation_findings.md`; investigate (likely LD ref mismatch or numeric solver difference); does NOT block phase |
| Existing `bayes-scan` parity tests (`tests/test_bayesian_vs.py`, ~30 tests) regress | **HALT** the phase; revert. Same invariant as Phase 57/58/59 R-MOD-1. |
| Phase 59 PolyFun §5.5 backward-compat regression net fails (`prior_pi_per_snp` shim) | **HALT**; spec contract with Phase 59 |
| Block-decomposition equivalence test (§5.1 Tier 1) fails | **HALT**; block-decomp is supposed to be exact, not approximation |
| ≥ 3 unrelated divergences during Tier B | **C4 emergency stop**; regroup with user |

## 9. Risks & open questions

### 9.1 Risks (mitigations made concrete)

| ID | Risk | Mitigation |
|---|---|---|
| R-NA1-1 | LD ref mismatch (cohort divergence between sumstats and LD ref) — primary failure mode per [C4] §6 | `_ld_ref_metadata.py` stamps `cohort_id`/`n`/`build` at LD-ref build time; `bayes-scan-rss` warns on soft mismatch and refuses on hard mismatch (different `build`) |
| R-NA1-2 | `susieR` install fails on the validation host (R toolchain quirks) | `install.sh` falls back to a containerized R (`docker run --rm rocker/r-ver:latest`) if local install fails; both paths satisfy `feedback_preflight` |
| R-NA1-3 | Block decomposition produces different posteriors than dense (numerical, not algorithmic) | Tier 1 §5.1 test catches this — block-decomp must match dense to ≤ 1e-10. If observed gap is real, it's a bug, halt. |
| R-NA1-4 | `feedback_streaming` invariant on raw-G `bayes-scan` carve-out — Path D doesn't close it; warning is the placeholder | Documented in `feedback_streaming` update at spec-merge time; explicit "Path A deferred to follow-on (OQ-NA1-1)" entry |
| R-NA1-5 | D3 shim from Phase 59 — `prior_pi_per_snp` keyword must propagate correctly through `bayes-scan-rss` | Tier 1 unit test in §5.1 + Phase 59 §5.5 regression net catches this |
| R-NA1-6 | UNVERIFIED items in research brief: SuSiE+ naming [C5], Quickdraws DOI [C8] | Both refer to Path B (deferred) and Path A (follow-on) — not in this spec's scope, so UNVERIFIED markers don't bite Path D |
| R-NA1-7 | Tests on this branch can't see `tests/test_streaming_memory.py` (lives on `efficiency/streaming-scan-audit`, 113 commits ahead of master) | Implementation kickoff MUST first either rebase NA1 onto `efficiency/streaming-scan-audit` OR merge that branch to master. **Both options acceptable per Section 6 design lean (a)**; document the chosen path in the implementation plan. |
| R-NA1-8 | `susieR` future API churn (e.g., `susie_rss` arg renames between minor versions) | Pin `susieR` version in `validation/external/susieR/install.sh` (use `remotes::install_version("susieR", "0.12.x")`); document the pin in the spec |
| R-NA1-9 | `--purity` 0.5 default differs from PolyFun's wrapper which uses 0.1 in some configs | OQ-NA1-4 below; default lean is 0.5 (susieR default) for parity test; expose `--purity` flag for users to override |

### 9.2 Open questions deferred to implementation kickoff

- **OQ-NA1-1**: Path A (chunked IBSS for raw-G `bayes-scan`) follow-on — when? After Phase 57 lands, in parallel with Phase 58/59, or a standalone effort? Default lean: standalone effort tracked separately as "NA1-followon" once NA1 ships. **Per Section 6 design lean (b), this OQ is documented here for the master-spec-style decisions ledger**.
- **OQ-NA1-2**: For the `--ld-ref` multi-format loader, do we ship `.bcor` (FINEMAP format) reader in Tier A or defer to Tier B? Default lean: Tier B — `.pt` (TorchGenomics) and `.npz` (PolyFun) cover the immediate needs; FINEMAP integration is a Phase-59-Tier-C concern.
- **OQ-NA1-3**: ELBO at convergence — does `bayes-scan-rss` write the per-iteration trace by default (for diagnostic) or only the final value? Default lean: final + max diff, with a `--write-elbo-trace` flag for deeper diagnostics.
- **OQ-NA1-4**: `--purity` threshold default. `susieR` defaults to 0.5; PolyFun's wrapper uses 0.1 in some configurations. Pick one and document. Default lean: **0.5** (susieR default) for the parity test; expose `--purity` flag for users.

## 10. Implementation phasing

### 10.1 Tier A — minimum viable shipping unit

Shippable when: `bayes-scan-rss` runs end-to-end on the MDP-derived per-locus fixture, produces output matching `susieR::susie_rss()` within tolerances (§5.2 table), block decomposition works, D3 shim wired, existing `bayes-scan` parity tests untouched.

| Step | Files | Test |
|---|---|---|
| **A0** Pre-flight: confirm `tests/test_streaming_memory.py` infrastructure available (R-NA1-7 — rebase or merge `efficiency/streaming-scan-audit`) | (sequencing gate) | n/a |
| **A1** LD-ref loader: `.pt` + `.npz` formats, in-sample LD via `--geno`, metadata schema + cohort-mismatch detection | `postgwas/_ld_ref_loader.py`, `postgwas/_ld_ref_metadata.py` | `tests/test_ld_ref_loader.py` |
| **A2** Block-decomposition primitive: split `R` by regions; auto-detect via `torchgenomics.ld.detect_blocks` if `--regions` not passed | `postgwas/_ld_ref_loader.py` extension | block-equivalence test |
| **A3** SuSiE-RSS algorithm: SER posterior + IBSS update + ELBO + credible sets — pure-torch implementation per §2 | `models/bayesian_vs_rss.py` | Tier 1 unit tests in `tests/test_bayesian_vs_rss.py` |
| **A4** D3 shim: `prior_pi_per_snp` keyword — accepts scalar OR per-SNP vector, normalizes per locus | `models/bayesian_vs_rss.py` | per-SNP-prior unit test |
| **A5** Output writer: TSV with PolyFun-compatible columns (§6 schema) | `models/bayesian_vs_rss.py` | output-format unit test |
| **A6** CLI subcommand `bayes-scan-rss` | `cli.py` | `tests/test_cli.py` smoke |
| **A7** Soft warning on materialized `bayes-scan` when `p > 10000` | `models/bayesian_vs.py` (warning only) | warning capture test |
| **A8** External validation harness: `validation/external/susieR/` mirroring Pillar B layout | `validation/external/susieR/{install,fetch_data,run_susieR,run_torchgenomics}.sh` + `compare.py` + `README.md` | manual smoke first, then `tests/test_external_susieR.py` |
| **A9** Tier 2 parity test vs `susieR::susie_rss` on MDP fixture | `tests/test_external_susieR.py` (`@pytest.mark.external @pytest.mark.golden`) | this test |
| **A10** Memory regression test for `bayes-scan-rss` | `tests/test_streaming_memory.py` extension | this test |
| **A11** Wall-time regression net | `bench/native_speedups.py` extension | this test |
| **A12** Phase 59 PolyFun §5.5 backward-compat: existing `tests/test_bayesian_vs.py` runs unchanged | full test suite | implicit |

### 10.2 Tier B — parity-tightening + scaling

Shippable when: 1KG-derived secondary fixture passes Tier 2; `.bcor` (FINEMAP) LD-ref format supported; per-iteration ELBO trace optional; `--purity` threshold tunable.

| Step | Files | Test |
|---|---|---|
| **B1** Secondary Tier 2 fixture: 1KG-derived per-locus `(z, R, n)` (~1000 SNPs) | `validation/external/susieR/` extension | parity test |
| **B2** `.bcor` (FINEMAP format) LD-ref reader | `postgwas/_ld_ref_loader.py` extension | format round-trip test |
| **B3** Per-iteration ELBO trace + `--write-elbo-trace` flag | `models/bayesian_vs_rss.py` extension | trace-format test |
| **B4** Configurable `--purity` threshold + cross-tool sweep at multiple values | `cli.py` extension | parity sweep test |
| **B5** PolyFun integration: confirm `bayes-scan-rss` accepts the `prior_pi_per_snp` from PolyFun's S-LDSC step end-to-end (gated on Phase 59 implementation landing) | (none — integration test only) | new `tests/test_polyfun_bayes_scan_rss_integration.py` |
| **B6** Memory regression at full-genome scale (UKB chr22 simulated `(z, R, n)`) | `tests/test_streaming_memory.py` extension | this test |

### 10.3 Tier C — nice-to-haves & follow-on

| Step | Files | Test |
|---|---|---|
| **C1** Native acceleration of per-block softmax + IBSS update | `csrc/susie_rss.cpp` + dispatcher | parity + bench |
| **C2** GPU dispatch for block IBSS loop | torch GPU body | bench |
| **C3** **Path A follow-on phase** — chunked IBSS for raw-G `bayes-scan` (per OQ-NA1-1) | (separate sub-spec; out of NA1 scope) | (separate phase) |
| **C4** Multi-trait extension (joint fine-mapping across correlated traits) | new `models/bayesian_vs_rss_multi.py` | parity vs SuSiE-mvSusie |
| **C5** Cross-ancestry extension via SuShiE [C9] (research-grade, longer horizon) | new module | per-ancestry parity |
| **C6** `feedback_streaming` memory update — remove the raw-G `bayes-scan` carve-out once Path A ships (C3) | docs only | n/a |

## 11. Citations

- **[C1] Wang, G., Sarkar, A., Carbonetto, P., Stephens, M. (2020).** *A simple new approach to variable selection in regression, with application to genetic fine-mapping.* JRSS-B 82(5): 1273–1300. <https://doi.org/10.1111/rssb.12388>. SuSiE / IBSS algorithm; original credible-set construction §3.4; PIP formula eq. 12.
- **[C2] Stephens Lab.** *susieR: Sum of Single Effects (SuSiE) Regression* (R package documentation). <https://stephenslab.github.io/susieR/>. `susie()` and `susie_rss()` reference; vignettes "Fine-mapping a region with SuSiE" and "Fine-mapping with summary statistics".
- **[C4] Zou, Y., Carbonetto, P., Wang, G., Stephens, M. (2022).** *Fine-mapping from summary data with the "Sum of Single Effects" model.* PLOS Genetics 18(7): e1010299. <https://doi.org/10.1371/journal.pgen.1010299>. SuSiE-RSS canonical derivation; §2 equations 7–11; §6 LD-ref-mismatch failure mode; §A.2 ELBO derivation.
- **[C5] PolyFun documentation.** Wiki §1 — "Computing prior causal probabilities with PolyFun". <https://github.com/omerwe/polyfun/wiki/1.-Computing-prior-causal-probabilities-with-PolyFun>. Block-size convention (`block_size_threshold` ≤ 5000).
- **[C7] Benner, C., Spencer, C. C. A., Havulinna, A. S., Salomaa, V., Ripatti, S., Pirinen, M. (2016).** *FINEMAP: efficient variable selection using summary data from genome-wide association studies.* Bioinformatics 32(10): 1493–1501. <https://doi.org/10.1093/bioinformatics/btw018>. `.bcor` format reference for Tier B B2.
- **[C9] Lu, Z., Cao, J., et al. (2024).** *SuShiE: Sum of Shared Single Effects, ancestry-aware fine-mapping using GWAS summary statistics.* Nature Communications 15. <https://www.nature.com/articles/s41467-024-54571-w>. Cross-ancestry extension reference for Tier C C5.
- **[C10] Weissbrod, O., et al. (2020).** *Functionally informed fine-mapping and polygenic localization of complex trait heritability.* Nature Genetics 52: 1355–1363. <https://doi.org/10.1038/s41588-020-00735-5>. PolyFun; uses SuSiE-RSS internally; ldetect-block convention.
- **[C11] PolyFun GitHub repository.** <https://github.com/omerwe/polyfun>. Reference Python implementation; documents the LD-block-per-locus workflow Path D depends on.
- **[C-PHASE-59] TorchGenomics Phase 59 PolyFun design spec.** `docs/superpowers/specs/2026-05-11-phase-59-polyfun-design.md`. D3 shim definition (§2.3); §5.5 backward-compat regression net.
- **[C-NA1-BRIEF] NA1 research brief.** `docs/superpowers/research/na1-susie-streaming-research.md`. 19 primary citations; comparison matrix §6; recommended path §7.
- **[C-SESSION-HANDOFF] SESSION_HANDOFF.md NA1 task definition.** `docs/superpowers/SESSION_HANDOFF.md` § "Task NA1 — bayes-scan SuSiE streaming". Source of credible-set Jaccard ≥ 0.95 success criterion.

## 12. Approval & next step

This spec is committed once approved. Implementation MAY begin once approved (does NOT need to wait for validation campaign + NA1–NA3 to close, since this IS NA1). Implementation kickoff blockers:
1. A0 sequencing gate per R-NA1-7 — rebase NA1 onto `efficiency/streaming-scan-audit` OR merge that branch to master.
2. Phase 59 PolyFun spec status — D3 shim definition in §2.3 of Phase 59 spec is the contract NA1 D3 implementation must match. If Phase 59 is being implemented in parallel, coordinate the shim signature.
3. Decision on OQ-NA1-1 (Path A follow-on timing) at implementation plan time, not now.

At implementation kickoff, this spec is fed to `superpowers:writing-plans` to produce the per-Tier implementation plan.
