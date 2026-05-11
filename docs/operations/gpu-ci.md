# GPU CI on PRs (NA2)

This document describes the self-hosted GPU CI used to validate `--device cuda`
code paths in TorchGWAS. It covers the security model, the runner architecture,
the runbook for adding/labeling/debugging PRs, capacity planning, and failure
modes.

For the workflow itself, see [`.github/workflows/gpu.yml`](../../.github/workflows/gpu.yml).
For runner setup, see [`scripts/setup_gpu_runner.sh`](../../scripts/setup_gpu_runner.sh).

---

## 1. Why GPU CI

TorchGWAS is GPU-accelerated end-to-end: every tensor op routes through
`torchgwas._dispatch.select_path` and runs on whichever device the user
requested. The default CI matrix (`.github/workflows/ci.yml`) only exercises
the CPU path because GitHub-hosted runners do not have CUDA hardware. This
leaves a class of bugs that are completely silent on CPU and only manifest
when the user passes `--device cuda`.

The Pillar C / E1 device-alignment audit (April–May 2026) caught five such
bugs that had landed on master without anyone noticing:

| Bug class | Example commit | Symptom |
|---|---|---|
| Tensor-on-CPU passed to CUDA op | `adc7b04` | `RuntimeError: Expected all tensors to be on the same device, but got mat is on cuda:0, different from other tensors on cpu` |
| `--device cuda` silently falls back to CPU | `1e27d81` | No error; user thinks they got GPU acceleration but didn't |
| `chunk.dosage` against a `(G_chunk, vmeta)` tuple | `adc7b04` | `AttributeError: 'tuple' object has no attribute 'dosage'` on every committed format under `impute` |

`adc7b04` and `1e27d81` are the canonical historical references — both fixes
are documented in `docs/validation_findings.md`. The audit was reactive
("we found these by running everything by hand"); the GPU CI defined here
is the proactive replacement.

> **Per `feedback_benchmark_against_installed_tools`:** these tests catch
> CUDA correctness regressions. They do not cover head-to-head numerical
> agreement against installed reference tools (GEMMA, GAPIT, GWASpoly) —
> that is what `pytest -m golden` does, and those tests are CPU-only by
> design. See section 7 below.

---

## 2. Security model

### The threat

Self-hosted runners are physical hardware owned by the maintainer. If the
runner executes attacker-controlled code, the attacker has code execution
on the maintainer's workstation with whatever privileges the runner process
has. For a public / community-PR repository, this is a critical risk.

GitHub explicitly warns against using `pull_request` triggers with
self-hosted runners on public repos:

> "We recommend that you only use self-hosted runners with private
> repositories. This is because forks of your public repository can
> potentially run dangerous code on your self-hosted runner machine by
> creating a pull request that executes the code in a workflow."
>
> — *GitHub Docs, "About self-hosted runners"*, [docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security)

GitHub Security Lab documents the specific exploit pattern (a PR rewriting
the workflow file to run arbitrary commands) under "Pwn Requests":
[securitylab.github.com/research/github-actions-preventing-pwn-requests/](https://securitylab.github.com/research/github-actions-preventing-pwn-requests/)

### The mitigation

The workflow uses three layered controls:

1. **`pull_request_target` instead of `pull_request`.**
   `pull_request_target` runs the workflow definition from the BASE
   branch (master), not the PR head. A community PR cannot rewrite the
   workflow to add malicious steps — only changes that have already been
   merged to master can change what the workflow does.
   See [docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#pull_request_target](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#pull_request_target).

2. **Label gating.** The job only runs if the PR carries the `gpu-tested`
   label, which only repository maintainers can apply. Community
   contributors cannot self-label. The label is applied AFTER a maintainer
   reviews the diff for anything obviously hostile (a malicious
   `setup.py`, a poisoned test file, etc.).

3. **`--ephemeral` runner registration.** The runner processes ONE job
   and then deregisters. The systemd unit re-registers a fresh runner
   for the next job. This bounds the blast radius of a compromised job
   to a single workflow run; persistent state poisoning across PRs (a
   key escalation in the Pwn-Requests writeup) is impossible.
   See [docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/autoscaling-with-self-hosted-runners#using-ephemeral-runners-for-autoscaling](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/autoscaling-with-self-hosted-runners#using-ephemeral-runners-for-autoscaling).

### Residual risk

Even with all three controls, the PR head code DOES run on the runner
once the maintainer applies the label. The maintainer review is the
last line of defense. The workflow does not inject any secrets into the
test step (`persist-credentials: false` on checkout, no `secrets.*`
references downstream of checkout). A compromised PR can therefore:

- Exhaust the GPU's compute and VRAM for the duration of the run.
- Read/write files inside `_work/` (cleaned per ephemeral run).
- Make outbound network requests from the runner host.

It cannot:

- Persist state across runs (ephemeral).
- Read repository secrets (none are mounted).
- Modify the workflow on master (only `pull_request_target` semantics
  allow workflow changes, which require their own approval).

If the threat model later requires sandboxing the test process itself
(e.g., a hardened container or an unprivileged user namespace), the
`runs-on:` step can be replaced with a `container:` step that pins a
read-only image. That is out of scope for the initial scaffold.

---

## 3. Runner architecture

### Hardware

| Item | Value |
|---|---|
| Vendor / model | NVIDIA RTX 2000 Ada Generation |
| VRAM | 16 GB GDDR6 |
| Driver (current host) | 580.82.07 |
| CUDA runtime visible | 13.0 (forward-compatible with cu121 wheels) |
| Host | Maintainer workstation, Linux x86_64 |

The 16 GB VRAM ceiling shapes what tests can run. Today's GPU test
files (`tests/test_gpu*.py`, `tests/test_impute_gpu*.py`) all fit
inside ~2 GB peak. UKB-scale tests (NA3) would exceed this and need
either chunking or a different runner.

### Runner labels

The runner registers with three labels: `self-hosted`, `gpu`,
`rtx-2000-ada`. The workflow requires all three (`runs-on: [self-hosted,
gpu, rtx-2000-ada]`) so a misconfigured CPU-only self-hosted runner
cannot accidentally pick up the job.

### Isolation

Each job runs on a freshly registered ephemeral runner. The systemd
unit (`actions.runner.<org>.<host>.service`) restarts after each job,
calls `config.sh remove`, then `config.sh` again with a new
registration token (managed by GitHub's runner-group token rotation).

Inside the runner, the job uses a clean `_work/` directory, a clean
Python venv (created by `actions/setup-python`), and re-installs
TorchGWAS from the PR head every time. There is no caching across PRs.

---

## 4. Runbook

### Adding the `gpu-tested` label

The label exists on the repo (created out-of-band). To apply it:

1. Open the PR in the GitHub UI.
2. In the right-hand "Labels" panel, click the gear icon.
3. Select `gpu-tested`. The workflow triggers within ~30s on `labeled`.

CLI equivalent:
```
gh pr edit <PR-NUMBER> --add-label gpu-tested
```

The label can be removed with `--remove-label`. Removing it does NOT
cancel an in-flight run — only future pushes won't trigger.

### Interpreting failures

The workflow uploads three artifacts under
`gpu-test-artifacts-<PR-or-run-id>`:

- `junit-gpu.xml` — pytest JUnit XML, importable into any CI viewer.
- `pytest-gpu.log` — full verbose pytest output with traceback.
- `vram-before.csv` / `vram-after.csv` — `nvidia-smi` snapshots.

Common failure shapes:

| Failure | Likely cause | Action |
|---|---|---|
| `CUDA out of memory` | Test grew past 16 GB | Add `@pytest.mark.skipif(torch.cuda.get_device_properties(0).total_memory < THRESHOLD, ...)` or chunk the test |
| `Expected all tensors to be on the same device` | Device-alignment regression | This is exactly the class of bug NA2 exists to catch — fix in the same PR |
| `nvidia-smi not found on PATH` | Runner deregistered or driver borked | Inspect `journalctl -u actions.runner.* --since '1 hour ago'`; re-run `setup_gpu_runner.sh` |
| Test passes locally, fails in CI | Driver / CUDA toolkit version skew | Compare `nvidia-smi` output between local and CI |

### Debugging locally

To reproduce a CI failure on the maintainer's workstation:

```
# Force the same dispatch path the CI uses (no native, GPU-only).
export TORCHGWAS_DISABLE_NATIVE=1
export CUDA_VISIBLE_DEVICES=0

pytest -m gpu \
    tests/test_gpu.py \
    tests/test_gpu_model_parity.py \
    tests/test_impute_gpu.py \
    tests/test_impute_gpu_kernels.py \
    -v --tb=short
```

If a single test is failing, narrow with `-k <pattern>` and add `--pdb`
to drop into a debugger on first failure. Captured artifacts from the
CI run are usually enough to identify the failing assertion without
re-running.

---

## 5. Capacity planning

### Per-PR cost (estimated)

| Phase | Wall time | GPU-minutes |
|---|---|---|
| Checkout + Python setup | ~30 s | 0 |
| Install torch (cu121) | ~90 s | 0 |
| Install torchgwas + deps | ~60 s | 0 |
| `pytest` (current 68-test suite) | ~3–5 min | ~3–5 |
| Artifact upload + cleanup | ~15 s | 0 |
| **Total** | **~6–8 min** | **~3–5** |

A typical PR triggers the workflow 1–3 times (initial label, plus
`synchronize` on each push). Average GPU-minutes per PR is therefore
~10–15 minutes of single-GPU contention.

The runner is the maintainer's interactive workstation — the GPU is
shared with desktop compositing (~700 MiB resident from `gnome-shell` /
Xwayland) and any active interactive workloads. The 16 GB VRAM ceiling
is comfortable for the current test suite (peak observed: well under
2 GB).

### Quiet-hours scheduling

The nightly cron is set to **04:00 UTC** = 23:00 US Central. This is
deliberately outside the maintainer's active working hours so a long
nightly run (5–10 min) does not contend with interactive use.

If the test suite grows past 30 minutes per run, consider:

- Splitting parity tests (cheap) from full kernel tests (expensive)
  into separate workflow jobs with different schedules.
- Demoting the nightly cron to weekly and relying on label-gated PR
  runs as the primary regression net.

---

## 6. Failure modes

### Runner offline

Symptom: PR sits with the workflow "Queued" indefinitely.

- Check the runner status:
  `cd ~/actions-runner && ./svc.sh status`
- Check systemd logs:
  `journalctl -u actions.runner.* --since '1 hour ago'`
- Re-register if needed: `cd ~/actions-runner && ./config.sh remove
  --token <REMOVAL_TOKEN> && bash scripts/setup_gpu_runner.sh`.

The workflow does not auto-fall-back to another runner. A `gpu-tested`
PR with no available GPU runner will simply queue.

### CUDA OOM

Symptom: pytest fails with `torch.cuda.OutOfMemoryError`.

The runner does not auto-restart on OOM. Two recovery actions:

- Reduce the test's memory footprint (preferred).
- Add a `pytest.mark.skipif` gating on `total_memory`, with a TODO to
  fix or move the test to a larger-VRAM runner.

The `vram-before.csv` artifact captures the starting state — if a
previous job leaked VRAM into a non-ephemeral runner, this will be
visible. (With `--ephemeral` registration, this should not happen.)

### Test exceeds timeout

Each pytest invocation has `--timeout=900` (15 min per test) and the
overall job has `timeout-minutes: 60`. A test that exceeds 900 s will
fail with `pytest-timeout`'s thread-dump traceback in the artifact.
A job that exceeds 60 min will be killed by GitHub.

If a legitimate slow test exceeds the per-test cap, mark it
`@pytest.mark.slow` and adjust the workflow to `--timeout=1800` for
that tier. Don't blanket-bump the cap — a runaway test should fail
loudly, not block the queue.

### PR with `gpu-tested` label, but author force-pushes hostile commit

This is the threat model `pull_request_target` + label-gating defends
against. The label is NOT auto-cleared on `synchronize`; if a malicious
contributor pushes new code to a labeled PR, the new code WILL run.

Mitigation: when reviewing a labeled PR, treat any subsequent push as
requiring re-review before re-applying the label. A pre-commit hook on
the runner side could enforce "remove label on each push", but that is
not currently implemented — it is a maintainer-discipline gate.

---

## 7. Scope: what GPU CI does and does not validate

### Does

- CPU↔GPU numerical parity (`tests/test_gpu_model_parity.py`).
- Dedicated GPU imputation kernels (`tests/test_impute_gpu_kernels.py`).
- CUDA-only correctness and shape contracts (`tests/test_gpu.py`).
- That `--device cuda` does not silently fall back to CPU.

### Does not

- Head-to-head numerical agreement against GEMMA / GAPIT / GWASpoly.
  Those reference tools are CPU-only. Their parity tests live in
  `pytest -m golden` and run in the main CI.
- UKB-scale empirical validation (NA3 — not yet implemented).
- Multi-GPU correctness (the runner has one GPU).
- AMD / ROCm correctness (no test fixtures).

A PR that touches a CUDA path AND a `golden` reference test will need
both `pytest -m golden` (default CI) and `pytest -m gpu` (this CI) to
go green.
