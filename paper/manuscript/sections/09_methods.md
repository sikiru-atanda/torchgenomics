# 9 Methods (compressed)

This section summarises four architectural mechanisms — streaming I/O, the six-mode optimizer stack, native dispatch, and the observed-then-floored tolerance protocol — and two operational rules (the pre-flight resource gate and the F3 severity policy) that together make every numerical result reproducible from a single shell command.

## 9.1 Streaming I/O surface

Every format reader in `torchgwas.io` implements the `GenotypeReader` protocol (`torchgwas/io/base.py`): three read-only properties (`n_samples`, `n_variants`, `sample_ids`) and one iterator,

```
def iter_chunks(self, chunk_size: int = 1024)
        -> Iterator[tuple[Tensor, VariantMeta]]
```

Each call yields `(G_chunk, vmeta)` where `G_chunk` is an `(n_samples, m)` float dosage tensor for `m <= chunk_size` consecutive variants. Production paths (PLINK BED/PGEN, VCF/BCF, BGEN, HapMap, Zarr, HDF5, CSV) read from file-backed storage and never hold the full genotype matrix; the chunk tensor is the only live allocation, freed before the next yield. The `UnifiedScanner` consumes any reader identically, so adding a format requires only a new `iter_chunks`. The contract is gated by `tests/test_streaming_memory.py`, a tracemalloc-instrumented per-PR regression net that asserts peak resident memory scales with `chunk_size`, not `n_variants`. As of v0.3.8, 36 of 40 CLI scan subcommands stream end-to-end (40 TB on disk → 1–4 GB peak).

## 9.2 Six-mode optimizer fallback stack

REML in `torchgwas.optim` is a six-mode controller (`OptimizerController`, `controller.py`) that escalates between solvers. Each layer reports diagnostics (gradient norm, REML log-likelihood, Hessian conditioning, iteration count) that drive the next-layer choice:

1. **PX-EM** (parameter-expanded EM; `em_warmstart.py`) — primary path for the diploid LMM; globally convergent at the cost of a slow tail.
2. **AI-REML** (average-information REML; `ai_reml.py`) — fast quadratic convergence near the optimum; receives the PX-EM warm start.
3. **LBFGS-autograd** (`lbfgs_reml.py`) — PyTorch autograd quasi-Newton; engaged when AI-REML hits a non-positive-definite information matrix.
4. **MM algorithm** (`mm_reml.py`) — minorize-maximize surrogate; monotone ascent when AI fails.
5. **PCG iterative** — preconditioned conjugate gradient on the mixed-model equations; sparse-GRM biobank path (UKB n = 500 K, Section 8).
6. **Derivative-free rescue** — Brent / golden-section on the variance ratio when all gradient-based paths stall.

`fit_null()` returns the mode that converged together with its diagnostics, so the dispatcher escalates cleanly without losing reproducibility. Adapted from the ASReml-class charter (Section 13).

## 9.3 Native-dispatch protocol

Every hot loop with a C++ accelerator funnels through `torchgwas._dispatch.select_path`, which consults four signals: (a) whether the kernel's `torchgwas._native._<name>` pybind11 extension was built (the `csrc/` extensions are `optional=True` in `setup.py`, so `pip install` succeeds without a C++ toolchain); (b) whether CUDA is available and the input lives on a CUDA device; (c) whether a dedicated GPU kernel exists; (d) the environment overrides `TORCHGWAS_DISABLE_NATIVE=1` and `TORCHGWAS_DISABLE_GPU=1`. The dispatcher returns one of `"gpu"`, `"native"`, `"python"` and never moves data between devices: a CUDA tensor with no dedicated GPU kernel falls through to the generic torch path on-device rather than incurring a PCIe round-trip. The Python / torch body is the algorithmic specification; the C++ shortcut is wired at the top of each function. This is the **Python-as-spec, native-as-shortcut** invariant — when a divergence surfaces, correctness is debugged in the Python body first.

## 9.4 Observed-then-floored tolerance protocol

Numerical-equivalence gates are observed, not aspirational (`memory/feedback_validation_spec.md`). The procedure: (i) run the head-to-head once on a controlled fixture against the installed upstream tool; (ii) measure per-metric divergence (effect size, standard error, p-value, posterior); (iii) floor the gate at the next rounded order of magnitude above the observed value (e.g., 4.7 × 10⁻³ → 1 × 10⁻²); (iv) commit the floor with an inline comment naming the observed value and the source of slack (FP roundoff, parameterization mismatch, finite-difference resolution). Future runs that beat the floor pass silently; runs that exceed it trip the regression net and are routed through `docs/validation_findings.md`. Spec §16 aspirational targets appear in the supplement but are never used as test gates.

## 9.5 Pre-flight contract

Every download, install, and run sources `validation/external/_lib/preflight.sh` and calls `preflight_check_with_data_size <tool> <data_gb> <peak_ram_gb>` before any I/O. Disk floor: `data_gb * 4` (download + working set + 2× headroom). RAM floor: `peak_ram_gb * 1.5 + 4 GB headroom`. On failure the script aborts before touching network or disk — no partial executions on insufficient resources. Mandatory across every external-tool harness, figure-rendering stage, and the top-level driver.

## 9.6 F3 severity policy

Per `memory/feedback_f3.md`: V1-core math error → fix in-tier; V1-core regression on a golden test → C4 halt-pillar emergency stop; post-V1 divergence beyond tolerance → document with an `xfail(strict=True)` reason; **three or more unrelated post-V1 divergences in a single tier → C4 emergency stop** on suspicion of a systemic bug. Each F3 ships as three artefacts: a production-code fix commit (separate, before the test commit), a regression test, and a row in `docs/validation_findings.md`. The campaign triggered one C4 stop during Tier 1 (four post-V1 F3s plus one F2 silent bug). The F2 (LD-aware haplotype pruning) was fixed inline; the four F3 divergences (hyprcoloc Foley-2021 conditional prior, coloc_pairwise H3 outer-minus-diagonal, heidi_test LD-weighted variance, OCFLMM `nuisance_learner='ridge_quadratic'`) were initially documented as post-V1 limitations and have **all been closed on master** as of 2026-05-21 with backward-compatible patches (V1 defaults preserved; corrected forms opt-in via a new parameter where applicable). Each closure ships its harness re-run with the upstream-tool agreement number persisted next to the original F3 row.

## 9.7 GWAS↔TWAS integration dispatcher

`torchgwas.postgwas.combine_gwas_twas` dispatches a 14-way combination kernel against a single `(SumStats, TWASResult)` pair. The GWAS sumstats are first aggregated per gene via the MAGMA-style `snp_to_gene` mean-χ² test (Phase 44), giving a per-gene $p_{GWAS}$; the matched $p_{TWAS}$ comes from any of the three TWAS workflows. For the eight classical kernels the per-gene combination is the scalar two-input form of each method (Fisher's $-2\sum \log p$, Stouffer's $\sum w_i z_i / \|w\|$, Cauchy's $\sum \tan(\pi(0.5 - p_i))$, Brown's effective-df correction via Kost-McDermott covariance, Empirical Brown with Spearman-r-derived covariance, Wilson's HMP with $L \approx \log K + \gamma$ asymptotic scale, Zaykin's truncated product with the closed-form null over $m = s, \ldots, k$, and Tippett's Šidák-weighted min-p). The six novel methods route through dedicated helpers that consume additional inputs: $r^2_{eQTL}$ tensors (R²-Stouffer), an LD r matrix (LD-aware Brown via simpleM / eigenMT), a 2-D $(\mathrm{SNP}, \mathrm{gene\_action})$ p-value tensor (polyploid Fisher), a tissue×gene matrix plus a lead-SNP p (multi-tissue ACAT+lead), a kinship matrix for `ConditionalLMM` (conditional COJO), and a PPFC value from a prior hyprcoloc run (hyprcoloc-gated). The dispatcher emits a `CombinedResult` dataclass with `gene_id` / `chr` / `start` / `end` / `p_gwas` / `p_twas` / `p_combined` / `method` / `direction_concordance` / `r2_model` / `p_adj` per gene; multiple-testing correction (BH / Bonferroni / BY / Storey) is applied across the gene set.

## 9.8 Reproducibility infrastructure

All numerical claims regenerate from `bash paper/reproducibility/reproduce_paper.sh`. The driver runs the eight-stage pipeline `preflight → install references → stage fixtures → run references → run TorchGWAS → bench streaming → bench native → run multi-omics → render figures`, writing a `manifest.json` that records every numeric result, its source script, and the commit SHA. Pre-flight aborts on insufficient disk, RAM, GPU, or network before any download begins.

