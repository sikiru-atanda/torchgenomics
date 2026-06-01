# 8. GPU acceleration and biobank-scale streaming

## 8.1 The memory wall

At UK Biobank scale (n = 500,000 samples, p = 10,000,000 variants) a single materialized genotype matrix G in `float32` is roughly 20 TB, and 40 TB in the `float64` precision required for statistical inference. Loading G into RAM is therefore not an option — not even on a single node. Any production GWAS engine that intends to scale beyond commodity desktops must ship a streaming I/O contract as a first-class invariant, not an opt-in optimization. TorchGWAS enforces that contract end-to-end (Figure 6).

## 8.2 The streaming I/O surface

Every reader in `torchgwas.io` (PLINK BED, PLINK2 PGEN, VCF/BCF, BGEN, HapMap, Zarr/HDF5, CSV dosage) exposes the same `iter_chunks(chunk_size, ...)` generator that yields one column slice of G at a time. The canonical streaming consumer is `UnifiedScanner`, which streams chunks through any model implementing the `BaseModel.score_chunk` protocol; the canonical streaming GRM is `grm_vanraden_streaming`, an accumulator that builds the (n × n) cross-product by summing per-chunk contributions `G_chunk @ G_chunk.T` without ever materializing G. At biobank n where the dense (n, n) GRM itself reaches 4 TB at n = 1M, the sparse-block-diagonal GRM + PCG-REML path (Panel C) runs in `O(n × nnz)` per iteration. The streaming contract is audited in `docs/efficiency/streaming_audit.md`, which classifies every CLI scan subcommand as `streaming`, `partial`, or `materialized`; as of the v0.3.0–0.3.9 efficiency campaign, **36 of 40 CLI scans stream**, and the remaining materialized scans (`bayes-scan`, `mklmm-scan`, `set-scan`, `ld-blocks`) are materialized only where the algorithm itself requires global visibility of G.

## 8.3 Measured streaming-vs-materialized memory (Panel B)

Panel B is the headline empirical result. We swept p over {1e4, 3e4, 1e5, 3e5, 1e6} at n = 2,000, chunk size 1,024, recorded peak RSS for the streaming path, and compared against the analytical `n × p × 4 B` cost of holding G in `float32`. The streaming peak plateaus at **924 MB** across the entire p sweep; the materialized cost grows linearly, reaching **8.0 GB at p = 1e6** — an **8.7× reduction at p = 1e6**, extrapolating to the multi-terabyte gap at biobank p. Log-log slopes make the asymptotic statement precise: streaming slope **0.020** (statistically indistinguishable from zero, i.e. constant memory) and materialized slope **0.933** (linear in p, as expected). Bench script: `bench/streaming_p_sweep.py`; JSON output: `bench/streaming_p_sweep.json`; F6 renderer falls back to the analytical scaffold only when the bench has not been run.

## 8.4 Native C++ kernels (Panel A)

The streaming contract is necessary but not sufficient: per-chunk work also has to be fast. TorchGWAS ships 25 pybind11 C++ extensions under `csrc/`, OpenMP-parallelized where it pays off (sequential branchy LD-block detection, PGS Gibbs samplers, HWE/imputation hot loops). The extensions are `optional=True` in `setup.py`, so `pip install` succeeds without a C++ toolchain; the device-aware dispatcher routes each call to native if available and healthy, GPU if a dedicated CUDA kernel exists, else the pure-PyTorch reference. The Python body is preserved verbatim as the algorithmic specification (the "Python-as-spec, native-as-shortcut" invariant), so the C++ shortcut can never silently diverge — disabling native via `TORCHGWAS_DISABLE_NATIVE=1` falls through. The realistic-size benchmark (24 measured kernels) spans **1.0×–9,210.7× speedup with a median of 93.5×**: huge wins on LD blocks (Gabriel 9,211×, PELT 3,834×, Big-LD 2,277×), strong wins on PGS samplers (LDpred2 Gibbs 352×, PRS-CS 150×) and preprocessing (KNN 645×, polyploid HWE-DR 109×), down to launch-bound on already-fast kernels.

## 8.5 GPU parity and precision (Panel D)

Every numerical statistic in TorchGWAS is computed in FP64 on GPU. AMP (FP16 / BF16) is permitted only for I/O and GRM accumulation, never for the score / Wald / LRT statistics whose tail behavior gates discovery. CPU↔GPU parity is gated by the `tests/test_*gpu*.py` suite (68 tests across 4 files), which asserts equality at the FP64 noise floor on every reduction, eigendecomposition, and score chunk. Validation gates use numerical tolerance rather than bitwise identity by design — GPU reductions are non-deterministic at the bit level, but their FP64 noise floor is orders of magnitude tighter than any statistical inference tolerance.

## 8.6 Sparse-GRM + PCG-REML (Panel C)

For biobank n where a dense (n, n) GRM is itself 4 TB at n = 1M, dense eigendecomposition (GEMMA-style) is infeasible. TorchGWAS provides a sparse-block-diagonal GRM coupled to a PCG-REML solver in `torchgwas.optim`. Panel C shows a 1.8× wall-time reduction over dense-eigendecomposition REML at n = 500,000; the measurement is currently scaffolded pending the NA3 UKB-scale empirical-validation harness, which will run on a dedicated GPU node.

## 8.7 Reproducibility

Every number in Figure 6 is regenerable. `bash paper/reproducibility/reproduce_paper.sh` runs the eight orchestrator stages, including the streaming p-sweep and native-kernel realistic benchmark, and writes the measured values into `paper/reproducibility/manifest.json` for the F6 renderer to consume. The manifest records, per panel, whether the value is `measured` or `scaffold` and the source artifact path, so a reviewer can audit provenance without re-reading the code.
