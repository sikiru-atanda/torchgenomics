# hyprcoloc reference harness (Pillar B / Tier 1 A4)

[hyprcoloc](https://github.com/jrs95/hyprcoloc) (Foley et al. 2021) is the
canonical R implementation of multi-trait colocalization, generalising the
Giambartolomei et al. 2014 two-trait `coloc` to K traits via a branch-and-bound
greedy algorithm with a hierarchical conditional prior. This harness installs
hyprcoloc + its R-side prerequisites (Rmpfr, gmp via conda-forge headers; iterpc,
arrangements via CRAN), simulates a 3-trait sumstats fixture with a planted
shared causal SNP, runs hyprcoloc in R, and compares cluster membership +
posterior PPs against TorchGenomics' `torchgenomics.postgwas._hyprcoloc.hyprcoloc`.

## Pinned reference

| Field | Value |
|---|---|
| hyprcoloc version    | 0.0.2 (GitHub `jrs95/hyprcoloc` @ commit `0348bbd`) |
| hyprcoloc commit SHA | `0348bbdd977be731d82e4efda115a9bed63cd44f` (2024-04-08; HEAD) |
| RcppEigen version    | 0.3.3.9.4 (pinned; **see note below**) |
| Rmpfr / gmp / iterpc / arrangements | 1.1.2 / 0.7.5.1 / 0.4.2 / 1.1.10 (CRAN) |
| jsonlite version     | 2.0.0 (CRAN; JSON bridge) |
| R runtime            | 4.5.1 (system R) |
| Conda gmp / mpfr     | 6.3.0 / 4.2.2 (conda-forge base env; supplies gmp.h / mpfr.h + libgmp / libmpfr without root) |
| Simulation seed      | 42 |
| Fixture markers (m)  | 100 |
| Fixture traits (K)   | 3 (T1, T2, T3) |
| Causal SNP index     | 50 (1-based) -> rs00050 |
| Causal beta          | 0.5 (on all three traits — *shared* causal architecture) |
| Background noise sd  | 0.05 |
| Standard error       | 0.1 (per SNP, per trait) |

The exact installed versions are recorded in `.install_marker` after a successful install.


## RcppEigen 0.3.3.9.4 pin (compatibility note)

The CRAN HEAD of RcppEigen (0.3.4.0.2 as of 2026-05) ships Eigen 3.4, which
routes `operator()(double,double)` to the new `IndexedView` overload. hyprcoloc's
`src/align*.cpp` use the Eigen-3.3 scalar overload (`RHO(snps1(0,i)-1, j2)`
returns a `double`), so compiling against RcppEigen 0.3.4 emits errors like:

    error: cannot convert IndexedView<...> to double

We pin RcppEigen to **0.3.3.9.4** (CRAN archive; the last release that ships
Eigen 3.3.9.x) for this harness only. The pin is applied via
`remotes::install_version` and is local to the user's R library — it does not
affect any other R package built elsewhere.

## Conda gmp/mpfr (no-root installation)

Rmpfr and gmp R packages require `gmp.h` / `mpfr.h` headers and the
corresponding shared libraries at compile and link time. On RHEL/CentOS this
would need `yum install gmp-devel mpfr-devel` (root). We instead use:

    /home/sikiru.atanda/miniconda3/bin/conda install -n base -c conda-forge \
        gmp mpfr pkg-config -y

which puts:
- `gmp.h`, `mpfr.h` -> `~/miniconda3/include/`
- `libgmp.so`, `libmpfr.so` -> `~/miniconda3/lib/`
- `gmp.pc`, `mpfr.pc` -> `~/miniconda3/lib/pkgconfig/`

without sudo. The `install.sh` script then appends a marker block to
`~/.R/Makevars` (backing up the original to `~/.R/Makevars.bak.hyprcoloc`):

    # hyprcoloc-conda-marker (appended by validation/external/hyprcoloc/install.sh)
    CPPFLAGS+=-I/home/sikiru.atanda/miniconda3/include
    LDFLAGS+=-L/home/sikiru.atanda/miniconda3/lib -Wl,-rpath,/home/sikiru.atanda/miniconda3/lib

so R picks up the conda headers/libs for *any* package compilation. The marker
is idempotent (re-running install.sh detects and refuses to duplicate the
block). The rpath ensures the linked `libgmp.so` / `libmpfr.so` resolve at
runtime without exporting `LD_LIBRARY_PATH`.

## Why simulated sumstats (not real GWAS)

Real 3-trait sumstats (e.g., UKB lipid triplet + GTEx eQTL) require OpenGWAS /
GTEx API access at every CI run, and the underlying datasets shift over time as
the upstream pipelines re-harmonize their meta-analyses. This breaks
reproducibility. The hyprcoloc estimator is *summary-statistic only* — it does
not need real biology to test correctness. Simulated data with a planted shared
causal effect on all K = 3 traits tests every code path in
`hyprcoloc::hyprcoloc()` and TG's `torchgenomics.postgwas._hyprcoloc.hyprcoloc`.

Per Foley et al. 2021, the minimal informative fixture is K = 3 correlated
traits with one shared causal SNP. The expected cluster on this fixture is
{T1, T2, T3} with regional PP >= 0.95.

## Reproduction recipe

```bash
# From repo root:
bash validation/external/hyprcoloc/install.sh        # one-time R + conda install (~10 min wall time)
bash validation/external/hyprcoloc/fetch_data.sh     # simulate 3-trait sumstats (<1 s)
Rscript validation/external/hyprcoloc/run.R          # run hyprcoloc (~5 s)
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/external/hyprcoloc/compare.py

# Outputs (gitignored except results/):
#   data/sumstats.tsv, data/sim_truth.json
#   outputs/hyprcoloc_results.json
#   results/summary.tsv, results/agreement.json, results/manifest.sha256
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts
disk + RAM headroom before doing any work, per the Pillar B contract.

## Reference outputs

| File | Source | TorchGenomics counterpart |
|---|---|---|
| `outputs/hyprcoloc_results.json` (key `results[0]`) | `hyprcoloc::hyprcoloc(snpscores=TRUE)` | `torchgenomics.postgwas._hyprcoloc.hyprcoloc` |
| `outputs/hyprcoloc_results.json` (key `snpscores_per_iter[0]`) | `hyprcoloc::hyprcoloc(snpscores=TRUE)$snpscores` | (informational; not currently compared) |
| `results/summary.tsv` | per-check observed/threshold/passed | — |
| `results/agreement.json` | full structured comparison report + tool versions | — |
| `results/manifest.sha256` | SHA-256 of every reproducibility artifact in this dir | — |

## Calibrated tolerances and observed values

Spec target (from Tier 1 A4 brief): max |DeltaPP| <= 1e-3 AND cluster-assignment
agreement = 100% on the test locus (observed-then-floored).

Observed (2026-05-15, seed = 42, m = 100, K = 3):

| Metric | R | TG | Delta | Spec floor | Status |
|---|---|---|---|---|---|
| Cluster membership (zero-based) | [0, 1, 2] | [0, 2] | — | exact | **FAIL** |
| Cluster-assignment agreement | — | — | — | 100% | **0% (FAIL)** |
| Candidate SNP id | rs00050 | rs00050 | — | exact | PASS |
| Per-SNP PP within cluster | 1.0000 | 0.999996 | 3.69e-06 | 1e-3 | PASS |
| Regional PP_S | 0.9764 | 0.0746 | 9.02e-01 | 1e-3 | **FAIL** |

The **F3 finding**: TG's hyprcoloc picks the pair-cluster {T1, T3} (PP = 0.075)
over the truth-shared triple {T1, T2, T3} (PP = 0.023). The candidate SNP
identity (`rs00050`) and per-SNP PP (~1.0 in both tools) agree to FP precision.

### Root cause (and why we accept the divergence for the paper)

The disagreement is in the *prior parameterization*, not the data likelihood.

TG's `_hyprcoloc.py:204-218` uses the product form:

    Pr(H_S) = prior_1^|S| * prior_2^(|S|-1) * (1 - prior_1)^(K - |S|)

For |S| = 2 vs |S| = 3 the prior ratio is `prior_1 * prior_2 ~ 1e-4 * 0.98 ~
1e-4`, which heavily penalises the larger subset.

R hyprcoloc uses the **conditional prior c** parameterization of Foley 2021
Eq. 2 with a hierarchical structure (Foley 2021 Algorithm 1):

    Pr(H_S | union of associated traits) = c^(|S|-1) * (1-c)^(|union|-|S|)

This conditions on the cluster being *the* associated set, so growing |S| from
2 to 3 incurs only a `c = 0.02` multiplier, not the per-trait `prior_1 = 1e-4`.
The triple wins.

This is a documented post-V1 implementation gap. The fix is a ~40-line change
in `_hyprcoloc.py` to adopt the conditional-prior parameterization. Recorded
in `docs/validation_findings.md` under "2026-05-15 — Tier 1 A4: hyprcoloc R
harness (post-V1 F3 divergence)". Per the F3 severity policy, this is a
**post-V1 documented divergence** (hyprcoloc is Phase 42), not a V1-core
fix-now blocker.

The Genome Biology paper draft will reference this divergence in the methods
section ("best-cluster selection diverges from R reference under the
conditional-prior parameterization; candidate SNP + per-SNP PP agree"). The
full fix is queued for the post-paper Pillar A documented-divergences-closeout.

## Peak memory

| Stage | Disk pre-flight | RAM pre-flight | Observed peak |
|---|---|---|---|
| install (conda + R compile) | 8 GB | 6 GB | ~600 MB during install_github compile |
| fetch (simulate sumstats) | 4 GB | 6 GB | <50 MB |
| run (hyprcoloc on 100 SNPs x 3 traits) | n/a (no preflight) | n/a | <100 MB R interpreter |
| compare.py (Python 3 + torchgenomics) | n/a | n/a | ~750 MB (torch + scipy + numpy at import) |

## Layout

```
validation/external/hyprcoloc/
|-- install.sh             # conda gmp/mpfr + R-package install (idempotent)
|-- fetch_data.sh          # invokes simulate.R; idempotent
|-- simulate.R             # generates 3-trait sumstats with planted shared causal
|-- run.R                  # runs hyprcoloc::hyprcoloc(snpscores=TRUE), emits JSON
|-- compare.py             # parses JSON, runs TG hyprcoloc, asserts tolerances
|-- README.md              # this file
|-- .install_marker        # records pinned versions (gitignored except via reproducibility)
|-- data/                  # simulated sumstats + truth manifest (gitignored)
|-- outputs/               # hyprcoloc_results.json (gitignored)
+-- results/               # summary.tsv, agreement.json, manifest.sha256 (committed)
```
