# User-Friendly Programmatic API — Design Spec

**Status:** APPROVED 2026-08-03 (user confirmed: name `tg.gwas`; wire the whole model registry in scope).
**Branch:** `feat/friendly-api` (off `origin/master` `f8986e3`).
**Scope:** Python / R / CLI **programmatic surface** ergonomics. **No frontend / GUI / web** — purely the shape of the code users call.

## 1. Motivation

TorchGenomics is powerful and broad (40+ CLI scans, dozens of models, post-GWAS,
PGS, LD, viz) — but that breadth makes it hard to approach. The `torchgenomics.api`
facade exists but is **thin and uneven**: it exports only **`lmm_scan` + `glm_scan`**
as scan functions (farmcpu/blink/mvlmm/glmm/gxe/… have **no `api` function** — a
user must drop to the low-level `models`/`scan` layer for them), there is a CLI
**`pipeline`** command (impute→model-select→scan→correct) with **no Python
equivalent**, users run one model per call, must know which model/QC/kinship/PCA/
correction to use, call plotting themselves, and get no guidance. The user's goal
(2026-08-03): make the package **very user-friendly** — "simple by default,
powerful on demand" — across all four friction points they identified: (1) getting
started / "just works", (2) **freely choosing** among many options (esp. running
the model(s) they want, one or several), (3) consistency across the API,
(4) understanding and acting on results.

GAPIT was cited only as a reference for *ease of use*, NOT as a feature target —
TorchGenomics already far exceeds GAPIT's capability. The aim is ergonomics.

## 2. Goal & non-goals

**Goal:** a single obvious entry point + a self-explaining unified result + a
guidance layer, so a domain scientist goes from raw data to a trustworthy,
well-diagnosed result in one call — while experts keep the untouched low-level
modules.

**Non-goals:** no frontend/GUI/web; no new statistical methods (this is thin
orchestration over existing `api`/`models`/`linalg`/`viz`); the low-level API
(`models`, `scan`, `linalg`, per-scan `api.*_scan`) is **unchanged** and remains
the power-user path; LLM/MCP surface is a separate later effort.

**Design principle:** *simple by default, powerful on demand.* The default path
makes every reasonable choice automatically AND states what it chose (nothing
hidden); every choice is overridable.

## 3. Three components

### 3a. `tg.gwas(...)` — the one obvious entry point

```python
res = tg.gwas(
    phenotype,               # path | DataFrame | array | Series
    genotype,                # path (BED/VCF/CSV/…) | array   — format auto-detected
    covariates=None,         # path | DataFrame | array
    kinship=None,            # path | array | "auto" (default) | False (no random effect)
    pcs=None,                # int (#PCs to compute) | array | "auto" (default)
    trait=None,              # column name / index; default = first/only phenotype column
    trait_type=None,         # "continuous"|"binary"|"categorical"|None(auto-detect)
    models="auto",           # USER'S CHOICE (primary): "lmm" | "glm" | "farmcpu" | ... ;
                             #   a LIST runs several in one call -> GwasComparison (GAPIT model=c(...));
                             #   default "auto" picks a sensible model from the trait type (always
                             #   printed + overridable) so a bare call just works — never forced
    qc=True,                 # True(default QC) | False | dict(overrides)
    correction="bh",         # "bh"|"bonferroni"|"none"|…
    output=None,             # dir to auto-write the full report folder; None = no files
    device=None,             # None(auto) | "cpu" | "cuda"
    verbose=True,            # print the decisions it made
) -> GwasResult | GwasComparison
```

**Behavior (thin orchestration over existing code):**
1. **Load + align** phenotype/genotype/covariates (reuse `io` auto-detection +
   the existing three-way sample alignment). Accept in-memory objects by writing
   a tiny adapter (array/DataFrame → the reader interface) — no new formats.
2. **Auto trait-type detection** (if `trait_type=None`): binary if exactly 2
   distinct non-missing values; categorical if a small integer set (≤ ~10 and
   non-continuous); else continuous. Always report what was detected.
3. **Auto-QC** (if `qc=True`): MAF, call-rate, HWE at documented defaults (reuse
   the existing QC + variant-QC-Parquet path). `qc=dict(...)` overrides thresholds.
4. **Auto-kinship / PCA** (if `kinship="auto"`/`pcs="auto"`): VanRaden GRM
   (`linalg.kinship.grm_vanraden`) + top-N PCs (sensible default, e.g. 10) unless supplied.
5. **Model(s): the user's choice, first-class.** `models=` is how the user picks
   exactly what to run — one model, or a **list to run several in one call**
   (returns a `GwasComparison`), mirroring GAPIT's `model=c("GLM","MLM","FarmCPU",…)`.
   The full TorchGenomics model registry is available by name (§4). `models="auto"`
   is an *optional convenience fallback* (the §4 tree) for users who'd rather not
   choose — it is never forced, and whatever it picks is printed + overridable.
6. **Scan** via a single internal dispatch (`_run_model(alias, …)`) — the shared
   core the api, CLI, and R bridge all call. **Reality check:** today only `lmm`
   and `glm` have `api.*_scan` functions; the rest must be reached through the
   low-level `models` + `scan.UnifiedScanner`. So this effort **adds a thin
   `api`-level scan wrapper for every model in the §4 registry** (or a generic
   `api.scan(model=…)`), so the whole registry is genuinely runnable through one
   friendly surface — not just lmm/glm. No new statistics; just wiring.
7. **Multiple testing** via `correction`.
8. **Warnings** (§5) surfaced during the run.
9. Return a `GwasResult` (or `GwasComparison`); if `output=` given, call
   `.report(output)`.
10. If `verbose`, print a concise decision log (trait type, QC survivors, model +
    rationale, #PCs, correction, λ_GC).

**Friendly errors ("just works" — or fails helpfully):** every user-facing entry
validates inputs early and raises **actionable** messages, never a deep torch/
pandas traceback. E.g. unmatched sample IDs → "0 samples shared between phenotype
(n=500) and genotype (n=480); check the ID column"; unknown model → lists valid
names; a binary trait passed to `lmm` → suggests `glmm`. This is a first-class
requirement, not polish.

**Relationship to the existing CLI `pipeline`:** `tg.gwas` is the Python face of
the same orchestration the CLI `pipeline` already performs (impute→model-select→
scan→correct). They share one core. The new `torchgenomics gwas` CLI (§5) and the
existing `pipeline` become **aliases over that shared core** (keep `pipeline`
working; `gwas` is the discoverable name). Reuse `pipeline`'s existing
model-selection logic rather than writing a second one.

### 3b. `GwasResult` — consistency + understanding

Standardize on the existing `ScanRun` (already has `top_hits`, `lambda_gc`,
`output_files`, `summary()`, `manhattan()`, `qq()`, `to_dict/json`). Enhance so
**every** `api` scan returns this same shape:

- `.hits` — tidy DataFrame of significant/top variants (alias/companion to `top_hits`).
- `.summary()` — **plain-language** block: trait + type, n aligned, #variants after
  QC, model + one-line rationale, correction, **λ_GC with interpretation**
  ("✓ well-calibrated" / "⚠ inflated — consider more PCs"), #genome-wide-significant
  loci + top locus, any warnings, and a **"Next:" suggestions** line
  (e.g. `res.report("out/")`, `tg.annotate(res)`).
- `.diagnostics` — structured dict/obj: λ_GC, MAF spectrum summary, n, model config,
  QC survivors, warnings list.
- `.manhattan()` / `.qq()` — unchanged (already present), now uniformly available.
- `.report(dir)` — writes the full **publication folder**: `manhattan.png`,
  `qq.png`, `pca.png` (if PCs computed), `kinship.png` (if GRM computed),
  `results.tsv`/`.parquet`, and `summary.txt` (the plain-language summary).
  Reuses `viz` for all plots.

`GwasComparison` (multi-model): holds a `GwasResult` per model, with
`.summary()` (side-by-side λ_GC + #hits per model), `.results["lmm"]`, a combined
`.report(dir)` (per-model subfolders + an overlap/consistency table).

### 3c. Guidance layer

- `tg.recommend(phenotype, genotype, covariates=None) -> Recommendation` — runs the
  same load + auto-detection + choice logic as `gwas()` but **stops before scanning**;
  prints/returns the plan + rationale (detected trait type, suggested model, #PCs,
  QC that would apply, expected runtime ballpark). The "what should I use?" helper.
- **Warnings** (emitted during `gwas()` and reported in `.diagnostics.warnings`),
  a small high-value set: genomic inflation (λ_GC > ~1.10), excessive deflation,
  low post-QC variant count, extreme case/control imbalance (binary), high mean
  relatedness (suggests the mixed model / more care), phenotype with many missing.
  Each warning is one plain sentence + a suggested action.

## 4. Model choice — user-driven (primary), with an optional auto fallback

### 4a. Model registry (the names the user passes to `models=`)

The user is free to run any of these, singly or as a **list** (multi-model in one
call → `GwasComparison`). Names are short, memorable aliases over the existing
`models`/`api` layer (this is the discoverability surface — `tg.models()` lists
them with one-line descriptions):

| name | model | trait types | notes |
|---|---|---|---|
| `glm` | `GLM` | cont/binary/ordinal/multinomial | fixed-effects; PCs as covariates |
| `lmm` | `SingleTraitLMM` | continuous | GRM mixed model (GEMMA-equivalent) |
| `mvlmm` | `MultiTraitLMM` | multi continuous | multi-trait |
| `farmcpu` | `FarmCPU` | continuous | iterative, multi-locus |
| `blink` | `BLINK` | continuous | multi-locus, fast |
| `glmm` | `Binary/Ordinal/MultinomialGLMM` | binary/ordinal | PQL, SAIGE-style |
| `mklmm` | `MultiKernelLMM` | continuous | additive+dominance kernels |
| `gxe` | `GxELMM` | continuous | genotype × environment |
| `set` | `SetBasedScanner` | any | region/gene-based (SKAT…) |
| `bayes` | `BayesianVS` | continuous | SuSiE fine-mapping scan |
| `met` | multi-env models | continuous | multi-environment |
| … | (the full model list) | | surfaced via `tg.models()` |

`models=` accepts any alias, a list of aliases, or `"auto"`. Unknown names raise a
clear error listing valid options. (GAPIT-name synonyms — `MLM→lmm`, `GLM→glm`,
`Blink→blink`, `FarmCPU→farmcpu` — are accepted for migrants.)

### 4b. `models="auto"` — optional convenience fallback (never forced)

Only when the user explicitly asks for `"auto"` (or omits a choice and opts into
the default), pick a sensible model from the trait type — always printed + fully
overridable:

```
continuous:  kinship auto/present -> lmm      ;  kinship=False -> glm
binary:      kinship auto/present -> glmm     ;  kinship=False -> glm(firth)
ordinal:     -> glmm (or glm)   [small #categories]
```
The chosen model + the one-line reason are printed and stored in `.diagnostics`.
Auto is a helper for users who don't want to choose — it never overrides an
explicit `models=`.

## 5. Cross-surface consistency (Python / R / CLI)

The same concept, three faithful surfaces (consistent names + semantics):

- **Python (primary):** `tg.gwas(...)`, `tg.recommend(...)`, `GwasResult`.
- **R (`rTorchGenomics`, reticulate bridge):** `tg_gwas(...)`, `tg_recommend(...)`
  → an S4 `GwasResult` class mirroring the Python attrs/methods (`summary()`,
  `top_hits`, `manhattan()`, `report()`), following the existing `tg_*` /
  `bridge_call` pattern; add to `NAMESPACE` + `_pkgdown.yml`.
- **CLI:** a `torchgenomics gwas` subcommand mirroring the same flags
  (`--phenotype --genotype --covariates --kinship auto --pcs auto --models
  lmm,farmcpu,blink --correction --output`), which writes the report folder and
  prints the decision log + summary. It shares the same core as the existing
  `pipeline` command (which stays as an alias); `--models` accepts a comma-list
  for multi-model. Plus `torchgenomics recommend` (dry-run) and `torchgenomics
  models` (list runnable models). Same auto-behavior as the library.

Consistency requirements (apply to the whole `api` surface, not just `gwas`):
- Uniform argument names everywhere: `phenotype`, `genotype`, `covariates`,
  `kinship`, `pcs`, `correction`, `output`, `device`, `verbose`.
- **Every `api.*_scan` returns a `GwasResult`** with the same attrs/methods (audit
  the existing scans; fix any that diverge).

## 6. Backward compatibility

- Existing `tg.lmm_scan(...)` / `glm_scan(...)` etc. keep working; they may gain the
  richer `summary()`/`.report()` (additive). No breaking changes.
- Low-level `models`/`scan`/`linalg` untouched.
- `tg.gwas` is purely additive orchestration.

## 7. Scope / MVP boundary

**MVP (this effort):**
- `tg.gwas` — one entry: auto trait-type, auto-QC/kinship/PCA, **user-chosen
  model(s)** (single or list → `GwasComparison`), `"auto"` fallback, friendly
  errors, `verbose` decision log.
- **Wire the full model registry through the friendly surface** — add the thin
  `api` scan wrappers (or a generic `api.scan(model=…)`) for every §4 model, so
  `tg.gwas(models=…)` genuinely runs all of them (not just lmm/glm today).
- Enriched `GwasResult` (`.summary` plain-language, `.diagnostics`, `.report`
  folder, `.hits`) + `GwasComparison`; **every `api` scan returns this shape**.
- `tg.recommend` (dry-run) + `tg.models` (registry listing) + the core warnings.
- CLI `gwas` + `recommend` + `models` (sharing the `pipeline` core); R
  `tg_gwas`/`tg_recommend`/`tg_models` wrappers.

**Deferred (not now):** `preset=` bundles (fast/standard/thorough) — sensible
defaults + explicit params suffice first; interactive/HTML reports; ML-based
model suggestion; biobank-scale streaming of the orchestrator's own loads
(the underlying scans already stream); LLM/MCP wiring; auto-fine-mapping/
annotation chaining beyond a suggestion.

## 8. Risks

- **In-memory input adapter**: arrays/DataFrames → reader interface must align
  samples/variants correctly; reuse the existing alignment path, don't reinvent.
- **Auto-detection wrong call**: trait-type/model heuristics must be conservative
  + always visible + overridable; document the rules; test the boundaries
  (2-value continuous vs binary, small-int categorical).
- **Consistency audit scope**: making every existing scan return `GwasResult` may
  surface divergent result schemas (gxe/mvlmm/me-glmm produce multi-output) —
  handle these explicitly (a `GwasResult` that carries multiple sub-results).
- **CLI/R parity**: keep the three surfaces in lockstep; a shared core function
  the CLI + api + bridge all call prevents drift.

## 9. References
- Existing facade: `torchgenomics/api/` (`scans.py`, `_results.py` `ScanRun`,
  `plotting.py`, `data.py`).
- The "three audiences, one engine" note in `CLAUDE.md`.
- GAPIT (Lipka et al. 2012) — cited only as an ease-of-use reference, not a target.
