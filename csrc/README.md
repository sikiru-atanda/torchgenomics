# csrc — native C++ accelerators

Hand-written C++17 kernels exposed to Python via [pybind11](https://github.com/pybind/pybind11).
These accelerators are **optional**: the build is wired with `optional=True` in
`setup.py`, so an install on a machine without a C++ toolchain still succeeds
and the pure-Python reference implementations under `torchgenomics/pgs/` are used at
runtime.

## What is built

| Extension                              | Sources                                | Replaces                                  |
| -------------------------------------- | -------------------------------------- | ----------------------------------------- |
| `torchgenomics._native._prscs_native`      | `csrc/pgs/prscs_gibbs.cpp` + `gig_sampler.cpp` | `_prscs_gibbs_block` and `sample_gig` in `torchgenomics/pgs/prscs.py` |
| `torchgenomics._native._ldpred2_native`    | `csrc/pgs/ldpred2_gibbs.cpp`           | `_ldpred2_gibbs_block` in `torchgenomics/pgs/ldpred2.py` (used by both `LDpred2Grid` and `LDpred2Auto`) |
| `torchgenomics._native._ct_native`         | `csrc/pgs/ct_clump.cpp`                | `_clump_with_ld_reference` in `torchgenomics/pgs/ct.py` (greedy C+T clumping, full + block modes); also reused by `ld_clump` in `torchgenomics/postgwas/_clump.py` (one chromosome = one block) |
| `torchgenomics._native._pelt_native`       | `csrc/ld/pelt_changepoint.cpp`         | inner PELT DP loop of `dp_changepoint` in `torchgenomics/ld/_changepoint.py` (Gaussian + Poisson cost) |
| `torchgenomics._native._ess_native`        | `csrc/stats/ess_geyer.cpp`             | Geyer initial-positive ESS loop inside `ess` in `torchgenomics/pgs/diagnostics.py` |
| `torchgenomics._native._graph_native`      | `csrc/ld/graph_utils.cpp`              | `connected_components` in `torchgenomics/ld/_graph_utils.py` (dense BFS over thresholded adjacency) |
| `torchgenomics._native._gabriel_native`    | `csrc/ld/gabriel_blocks.cpp`           | `detect_blocks_gabriel` in `torchgenomics/ld/_blocks.py` (2D-prefix-sum O(n²) candidate scan) |
| `torchgenomics._native._big_ld_native`     | `csrc/ld/big_ld.cpp`                   | candidate-interval loop of `detect_blocks_big_ld` in `torchgenomics/ld/_blocks_literature.py` (2D-prefix-sum mean-r² queries) |
| `torchgenomics._native._dp_optimize_native`| `csrc/ld/dp_optimize.cpp`              | per-chromosome DP + `_block_cost` (haplotype_diversity / tag_snp) in `detect_blocks_dp_optimize` (`torchgenomics/ld/_blocks_literature.py`) |
| `torchgenomics._native._cc_graph_native`   | `csrc/ld/cc_graph_adj.cpp`             | windowed r² adjacency build inside `detect_blocks_cc_graph` (`torchgenomics/ld/_blocks_literature.py`) |
| `torchgenomics._native._gwas_aligned_native`| `csrc/ld/gwas_aligned.cpp`            | DP + Jacobi-eigvalsh inner loop of `detect_blocks_gwas_aligned` in `torchgenomics/ld/_blocks_novel.py` |
| `torchgenomics._native._spine_native`      | `csrc/ld/spine.cpp`                    | greedy spine-of-LD partition loop of `detect_blocks_spine` in `torchgenomics/ld/_blocks.py` |
| `torchgenomics._native._ld_decay_signal_native`| `csrc/ld/ld_decay_signal.cpp`      | per-variant top-k loop of `ld_decay_signal` in `torchgenomics/ld/_changepoint.py` |
| `torchgenomics._native._greedy_mwis_native`| `csrc/ld/greedy_mwis.cpp`              | greedy interval-graph MWIS scan of `greedy_mwis` in `torchgenomics/ld/_graph_utils.py` |
| `torchgenomics._native._uncertainty_blocks_native`| `csrc/ld/uncertainty_blocks.cpp` | per-chromosome adjacent-r² break + block-stat loop of `detect_blocks_uncertainty` in `torchgenomics/ld/_blocks_novel.py` |
| `torchgenomics._native._wall_pritchard_native`| `csrc/ld/wall_pritchard_perm.cpp` | permutation loop of `compute_wall_pritchard_diagnostics` in `torchgenomics/ld/_blocks_diagnostics.py` (statistically equivalent — uses `std::mt19937_64` rather than `torch.randperm`) |
| `torchgenomics._native._impute_mode_native`| `csrc/preprocess/impute_mode.cpp`     | per-column mode histogram of `impute_mode` in `torchgenomics/preprocess/impute.py` |
| `torchgenomics._native._impute_knn_native`| `csrc/preprocess/impute_knn.cpp`       | per-missing-entry weighted-KNN fill of `impute_knn` in `torchgenomics/preprocess/impute.py` |
| `torchgenomics._native._impute_ld_native` | `csrc/preprocess/impute_ld.cpp`        | per-missing-entry windowed-correlation fill of `impute_ld` in `torchgenomics/preprocess/impute.py` |
| `torchgenomics._native._cavi_native`      | `csrc/models/cavi_sweep.cpp`           | per-SNP coordinate-update inner sweep of `BayesianVS._cavi_loop` in `torchgenomics/models/bayesian_vs.py` |
| `torchgenomics._native._spa_native`       | `csrc/stats/spa_lugannani_rice.cpp`    | per-extreme-SNP saddlepoint Newton + Lugannani-Rice tail body of `saddlepoint_pvalue` in `torchgenomics/stats/spa.py` |
| `torchgenomics._native._ldsc_native`      | `csrc/postgwas/ldsc_jackknife.cpp`     | n_blocks=200 leave-one-block-out WLS loop of `_block_jackknife_se` in `torchgenomics/postgwas/_ldsc.py` |
| `torchgenomics._native._hwe_native`       | `csrc/preprocess/hwe.cpp`              | per-SNP chi-squared HWE test loop of `_compute_hwe_pvalue` in `torchgenomics/preprocess/qc.py` (diploid + polyploid; hand-rolled regularized upper incomplete gamma); also `_compute_hwe_double_reduction` (tetraploid Haldane double-reduction model with method-of-moments alpha) |
| `torchgenomics._native._cross_pop_native` | `csrc/ld/cross_pop_stability.cpp`      | per-cluster, per-population, per-pair r² stability loop of `detect_blocks_cross_pop` in `torchgenomics/ld/_blocks_novel.py` (CSR-encoded clusters; returns `min(per-pop mean r²) / max(per-pop mean r²)` per cluster) |

`csrc/pgs/linalg.hpp` provides a tiny header-only Cholesky / triangular-solve
used by the PRS-CS Gibbs sampler. We deliberately avoid Eigen / BLAS to keep
the build self-contained — PRS-CS LD blocks are typically 50–2000 SNPs and the
hand-rolled routines are fast enough at that size.

## Building

```bash
pip install -e .
```

If a C++17 compiler is missing, the install proceeds and prints a warning. You
can verify the native module is loaded with:

```bash
python -c "from torchgenomics._native import HAS_NATIVE_PRSCS; print(HAS_NATIVE_PRSCS)"
```

On Windows, the install requires Microsoft Visual Studio Build Tools 2019 or
later with the "C++ build tools" workload.

## Forcing the pure-Python path

The pure-Python reference implementations remain in-tree as the algorithmic
spec — iteration on correctness happens in Python first. To force the Python
path even when the native extension is available (useful for debugging,
regression testing, or comparing numerics):

```bash
export TORCHGENOMICS_DISABLE_NATIVE=1   # bash / zsh
$env:TORCHGENOMICS_DISABLE_NATIVE = "1" # PowerShell
```

The pytest suite under `tests/test_pgs_prscs.py` is run with this variable
both set and unset to guard against drift between the two implementations.

## Device-aware dispatch

Hot-loop dispatchers in TorchGenomics go through `torchgenomics._dispatch.select_path`,
which picks one of three execution paths based on the input tensor's
**device**, **problem size**, and **build capability**:

| Path | Used when |
|---|---|
| `gpu`     | A dedicated torch-CUDA kernel exists *and* the tensor is on CUDA *and* the problem is large enough to amortize launch overhead. |
| `native`  | The C++ extension is built, `TORCHGENOMICS_DISABLE_NATIVE` is unset, the tensor is on CPU, and the problem exceeds the small-input cut-off (default 1 000 elements). |
| `python`  | Everything else — including CUDA tensors that have no dedicated GPU kernel. |

The dispatcher **never** copies a CUDA tensor to host memory just to use a
C++ extension: PCIe round-trips almost always cost more than they save.
CUDA tensors instead fall through to the generic torch path, which runs on
whatever device the data already lives on. To force the CPU path even on a
GPU machine, set `TORCHGENOMICS_DISABLE_GPU=1`.

## Benchmarks

A reproducible benchmark of every native kernel against its pure-Python
reference lives at `bench/native_speedups.py`. Run with:

```bash
python bench/native_speedups.py            # full ~5 repeats
TORCHGENOMICS_BENCH_QUICK=1 python bench/native_speedups.py  # 2 repeats, fast
```

The script writes a markdown table to `bench/native_speedups.md`. Most
kernels show **50× to 12 000×** speedup at realistic problem sizes;
the two exceptions on the current bench (`ESS Geyer`, `Spine blocks`)
are launch-overhead-bound at the small input sizes used and would
break even on larger inputs. The benchmark is informational — it does
not gate any test — but the *relative ordering* informs which kernels
are worth further optimization (e.g. OpenMP).

## Determinism

The native and Python paths are **statistically equivalent** but **not
bit-for-bit identical** — they use different RNGs (`std::mt19937_64` on the
C++ side, `torch.Generator` + scipy on the Python side). Within each path,
fixing the seed produces deterministic results.

## Licensing of native code

All C++ sources under `csrc/` are part of TorchGenomics and inherit the project's
MIT license. There are no vendored third-party headers; the only external
dependency is pybind11 (BSD-3-Clause), which is a build-time-only requirement.
