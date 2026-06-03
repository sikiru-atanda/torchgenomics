# Phase 55 — Polyploid Allele Dosage Assignment

**Status**: Design approved 2026-04-22. Awaiting implementation plan.
**Scope**: One phase. Polyploid *phasing* is deferred to a separate phase (56+), scoped later.
**Authors**: Sikiru Atanda + Claude Code (brainstorming session 2026-04-22).

---

## 1. Motivation and scope boundary

TorchGenomics already contains the *downstream* half of a polyploid dosage pipeline:

- `torchgenomics.preprocess.dosage_uncertainty` — consumes posterior `P(dosage = d)` of shape `(n, m, k+1)` and produces `expected_dosage`, `dosage_variance`, `dosage_rsq`.
- `torchgenomics.models.GULM` — consumes `dosage_var` for a dosage-variance-corrected score test (Phase 28, "GU-LMM").
- `torchgenomics.preprocess.phase.load_haplotypes` — ploidy-generic, reads a phased polyploid VCF.
- `torchgenomics.models.haplotype_gwas` — Phase 46/47 haplotype scans, ploidy-generic on the input side.

What is *missing* is the **producer** side. `torchgenomics/preprocess/phase.py::phase_beagle` falsely advertised a `ploidy` knob against BEAGLE 5.x (which is diploid-only). That was corrected in the same commit that introduces this design document. There is no tool inside TorchGenomics that converts raw polyploid sequencing read counts to posterior dosages.

Phase 55 delivers **exactly one** piece of that producer side: a thin external-wrapper module, `torchgenomics.preprocess.dosage_call`, that shells out to the R package `updog` to call posterior dosages from per-sample allele-depth read counts. Downstream modules consume the output without modification.

### In scope

- One wrapper function `run_updog(...)` around `updog::multidog()`.
- One dataclass `DosageCallResult` describing the call output.
- One CLI subcommand `torchgenomics dosage-call`.
- Tests: ~15 always-on Python tests (subprocess stubbed) plus ~5 release-time end-to-end tests that run real `updog`.
- Documentation: CLI help surface, one getting-started recipe, one memory file.

### Out of scope (explicitly deferred)

- **Polyploid phasing** — to be re-brainstormed as a separate phase once Phase 55 lands.
- **`polyRAD` / `fitPoly` wrappers** — same wrapper pattern applies; queued as Phase 55 follow-ups.
- **GPU acceleration** — `updog` runs on R/C; we only parse its output.
- **Changes to `dosage_uncertainty.py`, `GULM`, `poly-scan`** — already correct downstream consumers.
- **Integration with the `torchgenomics pipeline` subcommand** — cleanup concern, not Phase 55.
- **A pure-Python reimplementation of `updog`** — the repo's "Python-as-spec, native-as-shortcut" convention applies to *accelerated inference* primitives, not to *producer* steps. We follow the existing precedent in `torchgenomics.preprocess.impute_external` (BEAGLE / IMPUTE5 / Minimac4 are wrapped, not reimplemented).

### Integration point

```
calls.vcf.gz  →  run_updog()  →  probs (n, m, k+1)
                                 │
                                 ├── expected_dosage(probs, ploidy)  →  (n, m)
                                 ├── dosage_variance(probs, ploidy)  →  (n, m)
                                 └── → GULM.score_chunk(..., dosage_var=...)
```

No code changes to the downstream path.

---

## 2. Components and interfaces

Three components, all living in a single new file `torchgenomics/preprocess/dosage_call.py`.

### 2.1 `DosageCallResult` dataclass

Mirrors `ImputationResult` in `torchgenomics/preprocess/impute_external.py` for conceptual symmetry.

```python
@dataclass
class DosageCallResult:
    probs: Tensor                 # (n, m, k+1), float64; rows sum to ~1
    sample_ids: list[str]         # length n
    variant_ids: list[str]        # length m
    ploidy: int                   # k; 2 <= k <= 8
    tool: str                     # "updog"
    tool_version: str             # e.g. "2.0.2" (the supported floor)
    model: str                    # flexdog model name used
    mean_dosage_var: Tensor       # (m,) — mean Var[d] per marker (quality)
    allele_freq: Tensor           # (m,) — from expected dosages
    n_missing: int                # sample×marker slots updog couldn't score
    input_hash: str               # sha256 of input VCF
    cmd: str                      # exact Rscript invocation for reproducibility
```

### 2.2 `run_updog(...)` function

```python
def run_updog(
    input_vcf: str,                    # VCF with AD field
    output_path: str,                  # prefix for disk artifacts
    *,
    ploidy: int,                       # required; 2..8
    model: str = "norm",               # updog flexdog "model" arg
    rscript: str | None = None,        # path to Rscript; auto-detected if None
    bias: bool = True,                 # estimate allele bias
    od: bool = True,                   # estimate overdispersion
    seq_error: float | None = None,    # fix sequencing error, or None to estimate
    n_cores: int = 1,
    keep_tmpdir: bool = False,         # debug aid; skip cleanup
) -> DosageCallResult: ...
```

**Preconditions (raised before R is spawned)**:

- `shutil.which("Rscript")` must succeed (or `rscript=` must be valid) → else `RuntimeError`.
- `updog` must be available to `Rscript` (probed once and cached on a module-level flag) → else `RuntimeError`.
- `2 <= ploidy <= 8` → else `ValueError`.
- `model` in the documented `flexdog` set → else `ValueError`.
- VCF header declares `AD` → else `ValueError` (points to `bcftools +setGT` / GATK `HaplotypeCaller`).
- All variants biallelic → else `ValueError` on the first multi-allelic site (points to `bcftools norm -m -any`).

**Postconditions**: three files at `<output_path>.*` — `probs.pt`, `meta.json`, `snp_diag.tsv` — written atomically (i.e., only after `multidog` succeeds). Partial failures leave no output artifacts.

### 2.3 R driver script

A ~40-line `_UPDOG_DRIVER_R` string constant at module level in `dosage_call.py`. Written to the tempdir at each call. Reads two input TSVs (`ref.tsv`, `size.tsv`), runs `multidog(refmat, sizemat, ploidy, model, bias, od, seq, nc)`, writes `k+1` output TSVs (`pr_0.tsv` ... `pr_k.tsv`) via `format_multidog(mout, varname="Pr_d")` plus `snp_diag.tsv` from `mout$snpdf`.

**Why `format_multidog()` + `k+1` wide TSVs, not a single long-format table + Python pivot**: the long-form output of `mout$inddf` has no contract on row ordering, so a `.reshape(m, n, k+1)` in Python would silently misassign if `multidog` changes iteration order across versions. `format_multidog(mout, varname="Pr_d")` returns a `(m × n)` matrix keyed by snp/ind names — Python reads each file with `pandas.read_csv(..., index_col=0)`, transposes to `(n, m)`, and stacks.

### 2.4 Directory

```
torchgenomics/preprocess/
├── dosage_call.py          # NEW — run_updog, DosageCallResult, _UPDOG_DRIVER_R
└── dosage_uncertainty.py   # unchanged
```

---

## 3. Data flow

```
USER INPUT                                                 WRITTEN BY
─────────────────────────────────────────────────────────────────────
calls.vcf.gz               (biallelic, AD field)           user
│
▼ cyvcf2 iterator — per-variant extract AD[:,0] and AD[:,1]
│   missing AD (negative sentinel) → set ref=0, size=0
│
refmat:  (m, n) int32                                       run_updog (RAM)
sizemat: (m, n) int32    = refmat + altmat                  run_updog (RAM)
│
▼ write as TSVs with rownames=variant_ids, colnames=sample_ids
│
tmp/ref.tsv, tmp/size.tsv                                   run_updog (disk)
│
▼ subprocess [Rscript, driver.R, ref.tsv, size.tsv, ploidy, model, ...]
│
tmp/pr_0.tsv ... tmp/pr_k.tsv       (m × n) per file        R driver
tmp/snp_diag.tsv                    from mout$snpdf         R driver
│
▼ for d in 0..k: pandas.read_csv(pr_d.tsv).T.values → stack last axis
│
probs: (n, m, k+1) float64 torch tensor                     run_updog (RAM)
snp_diag: DataFrame                                         run_updog (RAM)
│
▼ compute quality metrics; hash input VCF
│
DosageCallResult                                            run_updog (return)
│
▼ persist
│
<output>.probs.pt        torch.save                         run_updog (disk)
<output>.meta.json       json.dump                          run_updog (disk)
<output>.snp_diag.tsv    DataFrame.to_csv                   run_updog (disk)
│
▼ downstream (existing code, unchanged)
│
expected_dosage(probs, ploidy) → (n, m)
dosage_variance(probs, ploidy) → (n, m)
GULM.fit_null(...).score(G=dosages, dosage_var=dosage_var)
```

**Tempdir lifecycle**: `tempfile.TemporaryDirectory()` scoped to the call. Cleanup is automatic on exit (including exceptions). `keep_tmpdir=True` suppresses cleanup for debugging.

**Atomicity**: persistent outputs written only *after* `multidog` succeeds. Partial R failure leaves no user-visible output on disk.

**Variant and sample ordering**: the `m × n` matrices written to R have `rownames=variant_ids` (from `cyvcf2.variant.ID`) and `colnames=sample_ids` (from `cyvcf2.VCF.samples`). `format_multidog()` preserves this keying. Python's pandas reads with `index_col=0`, so we never rely on positional ordering.

**Ploidy handling**: `ploidy` is a required kwarg. The R driver takes it as an argv parameter, `multidog()` consumes it. The output column set `Pr_0 ... Pr_k` is generated from `ploidy`.

---

## 4. Error handling and environmental dependencies

Four failure classes, each with a deterministic response:

### 4.1 Environment missing

| Failure | Response |
| --- | --- |
| `Rscript` not on `PATH` and no `rscript=` override | `RuntimeError("Rscript not found. Install R >= 4.0 and ensure Rscript is on PATH, or pass rscript=.")` |
| `updog` R package not installed | `RuntimeError("updog R package not installed. Install with: Rscript -e 'install.packages(\"updog\")'")` |
| Above probes succeeded once | Cache result on a module-level `_UPDOG_CHECKED` flag — subsequent `run_updog` calls in the same process skip the probe. |

No silent fallback to a Python reimplementation — by design (Section 0 decision). The wrapper is always-external-or-error.

### 4.2 Input validation

Raised *before* the R subprocess is spawned:

- `AD` not in VCF header → `ValueError` with a pointer to `bcftools +setGT` / GATK `HaplotypeCaller`.
- Any multi-allelic site → `ValueError` naming the first offender + `bcftools norm -m -any` pointer.
- `ploidy < 2` or `ploidy > 8` → `ValueError`.
- `model` not in the documented `flexdog` set (`"norm"`, `"hw"`, `"bb"`, `"s1"`, `"f1"`, `"s1pp"`, `"f1pp"`, `"flex"`, `"uniform"`) → `ValueError`.

### 4.3 R subprocess failure

- `subprocess.run(..., capture_output=True, text=True, check=False)`.
- Non-zero exit → `RuntimeError(f"updog failed (exit {rc}). stderr:\n{stderr}")`.
- Exit 0 but expected TSVs missing → `RuntimeError("updog exited 0 but produced no output — check stderr above.")`.
- On success: R stderr is forwarded through `logger.info`. On failure: forwarded through `logger.error`. Matches `run_beagle` / `run_impute5` convention.

### 4.4 Output validation

After stacking `probs`:

- `probs.shape == (n, m, k+1)` — else `RuntimeError`.
- `probs.sum(dim=-1)` is within `atol=1e-4` of 1.0 elementwise — `updog` rounds output before writing, so tolerance is loose.
- Any `probs[i, j, :].sum() == 0` → slot treated as missing: set to uniform `1/(k+1)`, `n_missing += 1`, no fail.
- Negative probs → clamp to 0, renormalize, warn via `logger.warning`.

### 4.5 Dependency matrix summary

| Dependency | Status | Reason |
| --- | --- | --- |
| `cyvcf2` | already optional (used by `torchgenomics.io.vcf.VCFReader`, `load_haplotypes`) | VCF + AD extraction |
| R ≥ 4.0 + `updog` ≥ 2.0.2 | NEW, opt-in, user-installed | the wrapped tool |
| `pandas` | hard dep | read `k+1` output TSVs |

No new Python dependency beyond `pyproject.toml`. R + `updog` is documented as an opt-in external requirement, same tier as BEAGLE / IMPUTE5 / Minimac4 already are.

---

## 5. Testing and validation gate

Two tiers. Tier 1 always runs. Tier 2 runs locally and in a release-time job where R + `updog` is provisioned.

### 5.1 Tier 1 — always-on Python tests (CI on every push)

File: `tests/test_dosage_call.py`. No R involved; the subprocess boundary is monkeypatched. Ships reference canned TSVs under `tests/fixtures/dosage_call/` (a toy `pr_*.tsv` family plus a small VCF).

| Category | Test |
| --- | --- |
| Input validation | Missing AD → `ValueError` |
| | Multi-allelic site → `ValueError` |
| | `ploidy=1` and `ploidy=9` → `ValueError` |
| | Bad model name → `ValueError` |
| | Missing `Rscript` → `RuntimeError` |
| Subprocess stubbing | Mock R writes valid canned TSVs → correct `DosageCallResult` shape, IDs, ploidy, tool |
| Output normalization | Rows summing to 0.99 → accepted |
| | Rows summing to 0 → flagged missing, set uniform, `n_missing` incremented |
| | Negative probs → clamped to 0 + renormalized + warning |
| Metadata | `sha256(vcf)` matches `meta.json` field |
| | `cmd` string contains exact Rscript invocation |
| Atomicity | Mock R writes only 2 of `k+1` TSVs → `RuntimeError`, no partial output files |
| Missing-AD handling | Negative AD sentinel values → set to `ref=0, size=0` before writing to R |

~15 tests total. Must pass on every CI job (Linux + Windows × 3.10/3.11/3.12).

### 5.2 Tier 2 — end-to-end with real `updog` (release-time + local)

File: `tests/test_dosage_call_updog_e2e.py`. Module-level `pytest.mark.skipif` gates on `shutil.which("Rscript")` and a one-shot `updog` import probe. Pattern matches `run_beagle` tests.

| Test | Pass criterion |
| --- | --- |
| `test_parity_uitdewilligen_tetraploid` | Run `run_updog(vcf, keep_tmpdir=True)`, capture tempdir. Reference R script reads the **same** `ref.tsv`/`size.tsv` that our wrapper wrote, calls `multidog()` directly, writes its own `pr_*.tsv`. Compare. `torch.allclose(probs_ours, probs_ref, atol=1e-6)`. The parity contract is **our R driver vs. bare `multidog()` on identical input**, not cross-VCF-parser equivalence. |
| `test_parity_ploidy_6` | Same contract on simulated hexaploid data (generated in-test). |
| `test_simulated_recovery_highdepth` | Simulate tetraploid reads at 30× depth, 1% seq error, 5% bias, 0.01 overdispersion. Mode-recovery ≥ (calibrated rate − 2%). |
| `test_simulated_recovery_lowdepth` | Same simulation at 8× depth. Mode-recovery ≥ (calibrated rate − 2%). |
| `test_gulm_end_to_end` | Simulate 500 markers × 200 samples. Push through wrapper → `dosage_variance` → `GULM.score_chunk(..., dosage_var=...)`. KS p-value > 0.01 (smoke test for integration, not a tight calibration; see `test_gu_lmm.py` for the full GULM calibration battery). |

### 5.3 Calibration helper (dev-time, not CI)

File: `bench/calibrate_updog_recovery.py`. Not picked up by pytest. Run once at phase sign-off to measure the recovery rate under the `test_simulated_recovery_*` simulation parameters, then paste the observed rate + a `# source: bench/calibrate_updog_recovery.py run 2026-XX-XX` comment into the Tier 2 assertions. `updog`'s paper gives oracle error rates by depth but not directly-comparable mode-accuracy thresholds, so self-calibration is the most honest path.

**Why 2% safety floor, not a statistical bound**: `flexdog` has no stochastic elements (deterministic EM with a fixed vector of bias-parameter inits — confirmed against `updog` NEWS.md). Same version + same input produces bit-identical output. The 2% margin therefore covers only **cross-updog-version drift** (algorithmic tweaks between releases moving the EM optimum slightly). A bootstrap-based statistical bound would be misleading because there's no stochasticity to bootstrap over — it would just measure numerical jitter. 2% is a pragmatic cross-version slack that can be tightened if the phase-sign-off calibration shows the observed rate is already close to 100%.

### 5.4 The gate

Phase 55 is complete when **all** of the following hold:

1. All Tier 1 tests pass on every CI matrix job (Linux + Windows × 3.10/3.11/3.12).
2. All Tier 2 tests pass locally on a machine with `R >= 4.0` and `updog >= 2.0.2` (the floor where `format_multidog()`'s SNP-dimension reorder bug was fixed upstream).
3. The Phase 55 entry in `docs/ROADMAP.md` is **removed** (it's shipped, no longer pending). The separate Phase 56 entry for polyploid phasing stays untouched.
4. `torchgenomics dosage-call --help` is captured in `docs/cli.md`.
5. `docs/getting-started/` has one recipe walking VCF → `dosage-call` → `gu-scan` on a toy dataset.
6. A memory file `project_dosage_call.md` is added under `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\` with scope, test count, and any gotchas discovered during implementation (mirrors the other per-phase memories).
7. No new C++ kernels were added → no `bench/native_speedups.md` entry required.

---

## 6. CLI

New subcommand `torchgenomics dosage-call`, wired into `torchgenomics/cli.py`. No decomposition of `cli.py` in this phase — that's a standing `docs/ROADMAP.md` cleanup.

```bash
torchgenomics dosage-call \
  --vcf calls.vcf.gz \
  --output out/dosage \
  --ploidy 4 \
  --model norm \
  [--bias/--no-bias] \
  [--od/--no-od] \
  [--seq-error FLOAT] \
  [--n-cores N] \
  [--rscript /path/to/Rscript] \
  [--keep-tmpdir]
```

Writes:
- `<output>.probs.pt` — torch tensor `(n, m, k+1)`, float64.
- `<output>.meta.json` — sample IDs, variant IDs, ploidy, tool + version, input hash, cmd, `n_missing`, timestamp.
- `<output>.snp_diag.tsv` — per-marker `updog` diagnostics (`bias`, `seq`, `od`, `prop_mis`, ...).

**Defaults**:
- `--model norm` (updog's Hardy-Weinberg-free default).
- `--bias` and `--od` on (updog's recommendation for sequencing data).
- `--n-cores 1` (safe default; parallelism is an explicit opt-in).

**Documented pipeline recipes** (both in `docs/getting-started/`):

```bash
# 1. Dosage-call → GU-LMM (GP-aware score test, primary use case)
torchgenomics dosage-call --vcf calls.vcf.gz --output out/dcall --ploidy 4
python -c "
import torch
from torchgenomics.preprocess.dosage_uncertainty import expected_dosage, dosage_variance
probs = torch.load('out/dcall.probs.pt')
torch.save(expected_dosage(probs, 4), 'out/dosage.pt')
torch.save(dosage_variance(probs, 4), 'out/dosage_var.pt')
"
torchgenomics gu-scan --genotype out/dosage.pt --phenotype pheno.txt \
    --dosage-var out/dosage_var.pt

# 2. Dosage-call → standard polyploid scan (ignoring uncertainty)
torchgenomics dosage-call --vcf calls.vcf.gz --output out/dcall --ploidy 4
# (same python one-liner as above to produce out/dosage.pt)
torchgenomics poly-scan --genotype out/dosage.pt --phenotype pheno.txt --ploidy 4
```

### Deferred cleanup (not Phase 55)

- Teach `gu-scan` to accept `--probs <probs.pt>` directly so the python one-liner goes away. Small argparse + dispatch change; queue as a follow-up.
- Wiring `dosage-call` into the `torchgenomics pipeline` subcommand — depends on `pipeline` learning about ploidy-aware dosage QC, not a narrow Phase 55 concern.

---

## 7. Non-goals and explicit decisions

Decisions made during the brainstorm, recorded so the implementation plan doesn't re-litigate:

1. **Split dosage assignment from phasing into two phases.** Different validation datasets, different user populations, different tools. Phase 55 = dosage. Phase 56 (TBD brainstorm) = phasing.
2. **Pure external-wrapper, no Python reimplementation of `updog`.** Producer step, not an inference primitive; matches `impute_external.py` precedent.
3. **`updog` only in the first cut.** `polyRAD` and `fitPoly` are follow-ups that reuse the `DosageCallResult` pattern established here.
4. **Both pass-through parity *and* simulated recovery in the validation gate.** Parity alone doesn't exercise our input-format conversion; simulated recovery alone doesn't prove wrapper fidelity. Both together catch both classes of bug.
5. **Standalone module + standalone CLI** (not merged into `impute_external.py`, not CLI-deferred). Dosage calling has a distinct output type (`(n, m, k+1)` posterior, not imputed VCF) that would break `ImputationResult` semantics.
6. **`format_multidog()` + `k+1` wide TSVs as the R→Python handoff**, not a long-format pivot, not an R object serialization. Avoids row-ordering contracts, avoids a `pyreadr` dependency.
7. **`cyvcf2` for VCF parsing on the Python side, not `vcfR` on the R side.** `cyvcf2` is already the repo's canonical VCF reader.
8. **TSV as the tempdir interchange format**, not feather / parquet. Dosage calling is one-time preprocessing, not a hot path; TSV keeps the dep graph simple.
9. **`keep_tmpdir=True` as a debug knob**, not a default. Tempdir cleanup is automatic under normal use.
10. **Recovery-rate thresholds are self-calibrated**, not cited from the updog paper. `bench/calibrate_updog_recovery.py` is a one-time dev helper; the observed rate minus a 2% safety floor goes into the Tier 2 assertions with a source comment.

---

## 8. Open questions that do not block implementation

None that affect this phase. The following are flagged for the follow-up phase brainstorm, not for Phase 55:

- **Phase 56 scoping.** Which polyploid phaser to wrap first — `PolyOrigin` (pedigree-aware autotetraploid), `TetraOrigin` (F1), `hapCON`, or `PolyPhase` — and whether to cover a single one or establish an orchestration pattern from the start.
- **`gu-scan --probs` passthrough.** Small, independent, can land any time after Phase 55 without blocking.

---

## Appendix: file-level impact summary

**New files**:
- `torchgenomics/preprocess/dosage_call.py` — ~400 LOC including the R driver string constant.
- `tests/test_dosage_call.py` — Tier 1, ~15 tests.
- `tests/test_dosage_call_updog_e2e.py` — Tier 2, ~5 tests.
- `tests/fixtures/dosage_call/` — toy VCF, canned TSVs.
- `bench/calibrate_updog_recovery.py` — dev-time calibration helper.
- `docs/getting-started/polyploid_dosage_call.md` — one recipe extending existing material.
- `~/.claude/projects/.../memory/project_dosage_call.md` — phase memory.

**Modified files**:
- `torchgenomics/cli.py` — one new subcommand (`dosage-call`) + argparse entry.
- `torchgenomics/preprocess/__init__.py` — re-export `run_updog`, `DosageCallResult`.
- `docs/ROADMAP.md` — update the Phase 55 entry to mark the dosage half as shipped and the phasing half as still-open.
- `docs/cli.md` — `torchgenomics dosage-call --help` capture.
- `CLAUDE.md` — optional: add `dosage-call` to the CLI commands list; bump subcommand count from 35 to 36.

**Unchanged**:
- `torchgenomics/preprocess/dosage_uncertainty.py`, `torchgenomics/models/gu_lmm.py`, `torchgenomics/preprocess/phase.py`, `torchgenomics/models/haplotype_gwas.py` — already correct as downstream / peer-producer code.
