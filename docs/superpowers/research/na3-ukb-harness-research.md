# NA3 UKB-scale validation harness — research brief

**Date:** 2026-05-11
**Branch:** `feat/na3-ukb-harness` (worktree at
`.claude/worktrees/na3-ukb-harness/`)
**Scope:** scaffold-only research; no actual UKB run.
**Authoritative spec:** Task NA3 in
`/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/docs/superpowers/SESSION_HANDOFF.md`
(lines 91-104).

## 1. The empirical gap NA3 closes

Per SESSION_HANDOFF lines 93-94:

> all of the campaign's biobank-scale memory math (40 TB → 4 GB) is
> *projected* from chunk-size arithmetic. No actual run on UKB-scale
> (n ≈ 500K samples, m ≈ 10M variants) data has been done. The
> streaming-memory regression tests verify peak ∝ chunk_size on
> n=200/m=2000 fixtures — that's the *invariant*, not the
> *biobank-scale measurement*.

The 40 TB number is the projected peak for a re-materialized
`(n=500K) × (m=10M)` float64 dosage matrix:

```
500_000 × 10_000_000 × 8 B = 4.0 × 10^13 B = 40 TB
```

The 4 GB number is the per-chunk peak under streaming
(`n × chunk_size × 8 B` with the default chunk_size = 1024):

```
500_000 × 1024 × 8 B = 4.1 × 10^9 B ≈ 4 GB
```

Both numbers are documented in
`/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/docs/efficiency/streaming_audit.md`
(lines 27-30) on the validation branch. The streaming-memory regression
tests at
`/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/tests/test_streaming_memory.py`
verify the **invariant** (peak grows ∝ chunk_size, not ∝ m) on
synthetic n=200/m=2000 fixtures. They do not verify the **measurement**
at n=500K.

NA3 closes that gap with a single empirical chr22 LMM + LDSC h² run.

## 2. Pillar B harness mirror — what we mirror, what we adapt

The eleven existing Pillar B reference-tool harnesses on the
`validation/pillar-A-coverage` branch follow a tight convention:

```
validation/external/<tool>/
├── install.sh            # download + verify + pre-flight; idempotent
├── fetch_data.sh         # stage / fetch input data; idempotent
├── run_<tool>.sh         # produce reference outputs; idempotent
├── compare.py            # parse outputs; assert tolerances; emit JSON
├── README.md             # operations doc
├── .env_marker           # records pinned versions (gitignored)
├── .cache/               # download cache (gitignored)
├── data/                 # staged inputs (gitignored)
└── outputs/              # tool outputs (gitignored)
```

Shared helper: `validation/external/_lib/preflight.sh` (sourced by every
script; provides `preflight_check`, `preflight_check_with_data_size`,
`verify_checksum`).

### What we mirror byte-for-byte

| Component | Source path | Why |
|---|---|---|
| `_lib/preflight.sh` | `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/_lib/preflight.sh` | The user-mandated pre-flight library; copying preserves the disk + RAM gate exactly. |
| Idempotence pattern in `install.sh` | `validation/external/regenie/install.sh` | "Skip if already installed at pinned version" is the same lattice; we just check both v4.2 and v4.1 acceptable. |
| Conda-env idempotence in `install.sh` | `validation/external/ldsc/install.sh` lines 45-58 | LDSC bioconda env detection logic is identical (subshell + `set +eu` for conda hooks). |
| `/usr/bin/time -v` peak RSS pattern | not in Pillar B (new convention for NA3) | Pillar B harnesses didn't measure peak RSS; we add it for NA3 because peak RAM is the third comparison gate. |
| Tolerance gates structure (`CheckResult` + `ComparisonReport`) | `validation/external/{regenie,ldsc}/compare.py` | Same dataclasses; same `_check_min` / `_check_max` helpers. |
| LDSC log parser regex | `validation/external/ldsc/compare.py` lines 121-176 | Parses `Total Observed scale h2:`, `Intercept:`, `Mean Chi^2:`, `Lambda GC:` — applies unchanged to the UKB-scale run. |

### What we adapt for biobank scale

| Adaptation | Rationale |
|---|---|
| REGENIE pin: v4.2 → v4.1 fallback (vs v3.3 in Pillar B) | The user instruction cites Phase 57 modernization spec for v4.x; v4.2 release availability is UNVERIFIED at scaffold time (knowledge cutoff Jan 2026), so the fallback to v4.1 is built into `install.sh`. |
| `--bsize 1000` (vs `--bsize 100/200` in Pillar B) | Mbatchou 2021 §"Computational cost" recommends `--bsize 1000` for biobank-scale Step 1; Pillar B used 100 because the MDP fixture had m=3093. |
| `--threads ${UKB_THREADS:-4}` (vs `--threads 1` in Pillar B) | Pillar B forced single-thread for determinism in the small-fixture comparison; at UKB scale we let REGENIE parallelize per host. |
| `fetch_data.sh` symlinks UKB (vs MDP staging in Pillar B) | UKB is DUA-controlled and lives wherever the user keeps it; we never duplicate it (could be hundreds of GB). |
| `run_torchgenomics.sh` is new (no Pillar-B equivalent) | Pillar B harnesses only ran the *reference* tool; TG was invoked from `compare.py`. NA3 needs the TG side as a separate process so we can measure its peak RSS independently for the third gate. |
| Quant-only (no Firth path) | Phase 57 modernization MVP scope; Pillar B regenie ran both qt + binary because the MDP fixture had EarHT_bin too. |
| Three-comparison structure (β corr, h² Δ, peak RAM) | Pillar B harnesses had per-tool comparisons (3 each for regenie + ldsc). NA3 compresses them into the three NA3 success-criteria axes (SESSION_HANDOFF line 102). |

## 3. Reference tool versions + pin justifications

### REGENIE: try v4.2, fall back to v4.1

- **Why v4.x and not v3.x.** Per the Task NA3 spec, "REGENIE pin = v4.2
  if available, else v4.1 (per Phase 57 modernization spec)". The
  Pillar B regenie harness used v3.3 because RHEL 9.6's glibc 2.34
  could not load v3.4+ static binaries (those link against glibc 2.35).
  The Phase 57 modernization track presumably either (a) bumps the host
  baseline to a newer glibc, or (b) adopts a v4.x release that links
  against an older glibc. **UNVERIFIED**: I do not have visibility into
  Phase 57's modernization spec from this worktree (it lives on the
  validation branch); the user-supplied instruction is the authoritative
  pin guidance.
- **Why fallback to v4.1.** v4.2 release availability is UNVERIFIED
  (Anthropic knowledge cutoff Jan 2026); the install script tries
  v4.2 first, falls back to v4.1 if the GitHub release URL 404s.
- **Where the install record lives.** `validation/external/ukb/.env_marker`
  records `REGENIE_VERSION_USED` after install; `compare.py` reads it
  for ledger-row provenance.

### LDSC: 1.0.1 (bioconda)

- Same pin as Pillar B: bioconda's `ldsc=1.0.1=pyhdfd78af_2`.
  Justification mirrored from
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/ldsc/install.sh`
  lines 13-25:

  > Upstream LDSC (https://github.com/bulik/ldsc) is Python 2.7 with
  > numpy<1.17 / scipy<0.19 / pandas<0.21 pinned. Modern pip + venv
  > cannot install these against a current Python. The maintained
  > Python-3 fork (belowlab/ldsc) is more recent but its numerical
  > output drifts from the canonical reference; the bioconda 1.0.1
  > build is the closest match to the published method and is what
  > S-LDSC papers continue to cite.

- LDSC `--two-step 99999` (single-pass IRWLS) is used to match
  TorchGenomics' `ldsc_h2(n_iter=2)` IRWLS port — see the resolved
  finding in `docs/validation_findings.md` (B2 row, validation-branch
  ledger):

  > Phase 37 IRWLS port resolved B2; intercept agreement now ~3e-5
  > absolute, gated at 5e-3.

## 4. Comparison metric thresholds + literature basis

### β Pearson r > 0.999 (TorchGenomics vs REGENIE)

- **Source:** Task NA3 success criteria (SESSION_HANDOFF line 102):
  "β corr > 0.999".
- **Literature basis:** REGENIE and TorchGenomics' SingleTraitLMM both
  implement a per-SNP score test on the same null model
  (intercept + covariates + LOCO-corrected GRM-leveraged variance
  components). When both tools see the same dosage, same covariates,
  and the same trait, β disagreement should be at the BLAS-precision
  floor — well below 1e-3 absolute. Pearson r > 0.999 is the
  observable consequence on a chr22-scale m ~ 1.6M variant set.
- **Validation-branch precedent:**
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/regenie/compare.py`
  observed β corr 0.821 at MDP scale (N=281), well below 0.999. The
  small-N noise dominated; the line cited in the
  validation-branch README is "aspirational gates from the harness
  spec (β corr > 0.99, -log10p > 0.95) are calibrated to N≥5000"
  (`validation/external/regenie/README.md` line 67-72). At UKB scale
  (N ≥ 50K) the design regime supports the > 0.999 gate.

### \|Δ h²\| < 5e-3 (TorchGenomics vs LDSC)

- **Source:** Task NA3 success criteria (SESSION_HANDOFF line 102):
  "h² Δ < 5e-3".
- **Literature basis:** LDSC h² has a published per-replicate jackknife
  SE that on chr22-only sumstats with ~17K HM3 SNPs typically sits
  ~0.01-0.03 (per Bulik-Sullivan 2015 Table S2). Two implementations
  of the same IRWLS regression on the same chi² + LD scores should
  agree to ~5e-3 or tighter — that's the LDSC tolerance literature's
  "indistinguishable" floor.
- **Validation-branch precedent:** TG ↔ LDSC IRWLS port observed
  \|Δ h²\| = 1.0e-5 on the simulated chr22 fixture (17,489 SNPs); see
  `validation/external/ldsc/README.md` lines 53-67 on the validation
  branch. The 5e-3 NA3 gate leaves head-room for biobank-scale
  jackknife block-boundary drift.

### Peak RAM ≤ 10% of `n × m × 8 B`

- **Source:** Streaming-first contract from `feedback_streaming` memory
  + the streaming audit
  (`/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/docs/efficiency/streaming_audit.md`).
- **Why 10% (not the projected 0.06% per-chunk fraction):** the K
  matrix is a model-architectural always-resident allocation
  (n × n × 8 B). At n=50K, K ≈ 20 GB; at n=500K, K ≈ 1 TB. The 10%
  ceiling comfortably accommodates K + per-chunk + Python overhead at
  n ≤ 50K subset-UKB scale (TG peak ≈ 24 GB vs full-mat
  ≈ n=50K × m=1.6M × 8 B = 640 GB → 3.75%) and gates against
  re-materialization regressions.

## 5. Benchmark-against-installed-tools

Per `feedback_benchmark_against_installed_tools`: this harness IS the
benchmark. The reference tools used are:

| Tool | Pin | Where the install record lives |
|---|---|---|
| REGENIE | v4.2 (try) → v4.1 (fallback) | `.env_marker` written by `install.sh` (line `REGENIE_VERSION_USED=...`); also `bin/regenie --version` |
| LDSC | 1.0.1 (bioconda `pyhdfd78af_2`) | `.env_marker` (`LDSC_VERSION=1.0.1`); `conda list -n ldsc_env ldsc` |

Pin justifications (above) are the canonical record of why these
versions specifically.

## 6. Risks

- **UKB data access prerequisite.** UKB is application-controlled. The
  user must already have UKB data on the host. The harness verifies the
  paths but never downloads. If the user does not have UKB access yet,
  the entire NA3 task is blocked at `install.sh` step 2 (env-var
  verification) — the script aborts with a clear message documenting
  what to set.
- **Estimated CPU hours per run.** Per Task NA3 spec ("a single LMM scan
  + a single LDSC h² estimate at UKB chr22 scale takes ~few CPU-hours").
  At full n=500K, REGENIE Step 1 dominates and can stretch to 2-4 hours
  on a 4-core host; the LMM scan can take 1-3 hours. Total: ~one
  CPU-day worst case. Subsetting to n=50K cuts this to ~1-2 hours
  total.
- **K-matrix scale ceiling.** TG holds the n × n kinship matrix in
  RAM throughout the scan. At n=500K, K ≈ 1 TB — out of scope for any
  single-host run. The MVP harness expects the user to subsample to
  n ≤ 50K (≈ 20 GB K) or supply pre-computed K via `--kinship`. This
  is a model-architectural cost (not a streaming-axis cost), and is
  documented in README.md as a known limitation.
- **REGENIE v4.x release availability.** UNVERIFIED. v4.2 may not exist
  yet at run time; the install script falls back to v4.1. If neither
  is available, the script aborts with instructions to manually drop a
  binary at `bin/regenie` and re-run.
- **Phenotype-table schema.** The harness assumes the user-supplied
  phenotype TSV has columns `FID  IID  <trait>  [covariates ...]`.
  Default expected covariates: `AGE,SEX,PC1,..,PC10`. Override via
  `UKB_COVAR_COLS`. If the user's actual schema differs, REGENIE +
  TG both abort at the column-resolution step; no silent miscompare.
- **Failure modes that need user intervention.**
  1. UKB data path verification → user re-exports env vars.
  2. REGENIE OOM at user-specified n → user subsamples and re-runs.
  3. β Pearson r < 0.999 → user inspects dosage convention (PlinkBedReader counts BIM A2; REGENIE counts ALLELE1) and trait-column alignment.
  4. \|Δ h²\| > 5e-3 → user inspects LD-score panel population matching (1000G EUR vs UKB EUR-subset).
  5. TG peak RSS over budget → user inspects whether a non-streaming code path was hit (e.g., `--grm-method zhang` opts in to materialization per the streaming audit).

## 7. No-shortcuts citations

Per `feedback_parallel_agents_no_shortcuts`, every claim about NA3
scope is anchored to SESSION_HANDOFF line numbers, and every "we
mirror" claim is anchored to a specific Pillar B harness file:

- "Single LMM scan + LDSC h² on chr22 is the right first target":
  SESSION_HANDOFF lines 98-100.
- "β corr > 0.999, h² Δ < 5e-3, peak memory observed vs projected
  invariant" success criteria: SESSION_HANDOFF lines 100-102.
- "Few CPU-hours per run": SESSION_HANDOFF line 104.
- "Mirror the Pillar B layout": SESSION_HANDOFF line 99.
- Pre-flight library copied verbatim from
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/_lib/preflight.sh`.
- REGENIE install pattern adapted from
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/regenie/install.sh`.
- LDSC install pattern adapted from
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/ldsc/install.sh`.
- LDSC `--two-step 99999` rationale + IRWLS port reference adapted
  from
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/validation/external/ldsc/run_ldsc.sh`
  lines 79-90 + `validation/external/ldsc/compare.py` lines 1-30.
- 40 TB → 4 GB projection (the claim NA3 empirically validates):
  `/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/docs/efficiency/streaming_audit.md`
  lines 27-30.
- F3 severity classifications: `feedback_f3` memory.
- Streaming-first invariants: `feedback_streaming` memory.
- Pre-flight hard-rule (abort, never partial-execute):
  `feedback_preflight` memory.
- Local commits only, no auto-push: `feedback_no_autonomous_push`
  memory (referenced; the harness scaffold ships zero `git push`
  invocations).

## 8. Out-of-scope (explicit non-goals for the scaffold)

- Binary phenotype + Firth path (Phase 57 MVP is quant-only).
- Multi-chromosome run (chr22 only for NA3).
- GPU lmm-scan (NA2 covers GPU CI infra).
- Auto-running the harness (the user — or future-Claude with user
  approval — runs).
- Modifying `torchgenomics/` source (harness consumes the CLI as-is).
- A separate UKB K-builder script (out of scope; user supplies K via
  --kinship if they need to break the n × n RAM ceiling, or
  subsamples to n ≤ 50K).

## 9. Verification done at scaffold time

- `bash -n` passes on all four shell scripts (install.sh, fetch_data.sh,
  run_torchgenomics.sh, run_reference.sh).
- `python -m py_compile` passes on `compare.py`.
- All shell scripts marked executable (`chmod +x`).
- Pre-flight calls in `install.sh` and run scripts use real
  `preflight_check` / `preflight_check_with_data_size` from the
  copied `_lib/preflight.sh` — not stubs.
- `docs/validation_findings.md` row schema verified against the canonical
  ledger header on the validation branch
  (`/home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/validation-pillar-A-coverage/docs/validation_findings.md`,
  lines 1-15: 10-column markdown table format).

## 10. Open follow-ups (post-NA3)

- Bump REGENIE pin to v4.2 explicitly once verified released; remove
  the v4.1 fallback path.
- Add a binary + Firth comparison once Phase 57 MVP closes.
- Extend to multi-chromosome (chr1-22) once chr22 baseline lands.
- Wire a `pytest -m external` test cell in `tests/test_external_ukb.py`
  that imports `compare.py` (currently the harness is shell-driven
  only; a pytest gate would auto-skip in normal CI but allow opt-in
  on a UKB-staged host).
- Add the GPU lmm-scan branch once NA2 (GPU CI infra) lands a
  self-hosted GPU runner.
