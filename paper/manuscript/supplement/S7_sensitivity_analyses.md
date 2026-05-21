# S7 — Sensitivity analyses

## Pre-flight gate behaviour

Every external-tool installer, fetch script, and run script in the validation campaign sources `validation/external/_lib/preflight.sh` and calls `preflight_check_with_data_size <tool> <data_gb> <peak_ram_gb>` BEFORE any download, install, or run. The gate enforces the spec section 5.3 contract:

- Disk: `df --output=avail /home` must report >= `data_gb * 2 + working_set * 2`.
- RAM: `free -g` must report available >= `peak_ram_gb * 1.5 + 4 GB headroom`.

On insufficient resources the gate aborts with an explicit shortfall message naming the missing GB of disk or RAM. The script does not download / install / run partially. Per `memory/feedback_preflight.md` this is a hard user rule: a partially-downloaded reference panel or partially-completed harness with insufficient memory is worse than aborting cleanly. The library exposes `preflight_check`, `preflight_check_with_data_size` (auto-applies the multipliers above), and `verify_checksum` for downloads.

Documented infra-blocker fallbacks: SAIGE on RHEL 9 without docker (Bioconductor R fallback); BOLT-LMM on hosts with glibc < 2.35 (binary loader failure, captured into `.infra_blocker`). Both abort cleanly; pytest tests then skip with the captured shortfall reason as the skip message.

## Native-off parity scans (`TORCHGWAS_DISABLE_NATIVE=1`)

The 26 C++ accelerators in `csrc/` are wired at the top of every accelerated function via `torchgwas._dispatch.select_path`; when the extension is missing or `TORCHGWAS_DISABLE_NATIVE=1` is set, execution falls through to the pure-Python / torch body which is the algorithmic reference (Python-as-spec invariant, recorded in `memory/feedback_validation_spec.md` and CLAUDE.md Repo Conventions). The Python body is mathematically identical to the native shortcut; expected parity is FP64 floor (max |Delta| < 1e-12 across every kernel-paired test in `tests/test_native_*.py`). Regression net: every native test file pair (`tests/test_native_*.py`) runs both modes and asserts parity; the gate is wired in CI workflow `.github/workflows/native-parity.yml`.

## OpenMP-off parity scans (`TORCHGWAS_DISABLE_OPENMP=1`)

OpenMP parallelisation is opt-in per platform; setting `TORCHGWAS_DISABLE_OPENMP=1` at build time produces serial native extensions. Expected parity vs the OpenMP-on build: FP64 floor (within thread-scheduling noise for reductions). Some kernels (parallel reductions over n samples) exhibit non-determinism at the last bit due to FMA ordering; the gate tolerance for OpenMP-off-vs-on parity is `max |Delta| < 1e-10`. Regression net: the same `tests/test_native_*.py` files are re-run under the OpenMP-off build in CI; failure under the tighter 1e-12 gate (relative to Python-as-spec) is acceptable; failure under 1e-10 (relative to the OpenMP-on build) is a regression.

## GPU-off parity scans (`TORCHGWAS_DISABLE_GPU=1`)

Device-aware routing in `torchgwas._dispatch.select_path` is centralised; setting `TORCHGWAS_DISABLE_GPU=1` forces CPU-only dispatch even when CUDA tensors arrive. Expected parity vs the GPU path: max |Delta| < 1e-10 (limit is FP64 non-determinism in GPU reductions; per the validation-gates convention in CLAUDE.md, validation gates use numerical tolerance not bitwise identity). Regression net: the GPU-marked subset of `tests/` (manifest F6 panel D notes 68 tests across 4 test files; e.g. `tests/test_gpu_*.py`) runs under both modes in CI workflow `.github/workflows/gpu.yml`. GPU CI artefacts are scaffold at the paper branch HEAD (`F6.numbers.panels.D.scaffold=true`); promotion to live is part of the NA2 GPU CI infra task in the next-agent task queue.

## CI wiring

Five GitHub workflows wire the regression nets above:

1. `.github/workflows/tests.yml` — full pytest suite (2801 pass / 544 skip at v0.3.8).
2. `.github/workflows/native-parity.yml` — `TORCHGWAS_DISABLE_NATIVE=1` vs default.
3. `.github/workflows/perf.yml` — wall-time gate driven by `bench/native_speedups.py`; flags > 10 percent regressions.
4. `.github/workflows/streaming.yml` — memory regression gate driven by `tests/test_streaming_memory.py`; flags re-materialisation of G.
5. `.github/workflows/gpu.yml` — GPU-marked test subset under both `TORCHGWAS_DISABLE_GPU=0/1`.
