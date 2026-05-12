# GPU CI runbook — self-hosted runner setup

**Goal**: register the local RTX 2000 Ada (16 GB) as a self-hosted GitHub
Actions runner so `.github/workflows/gpu.yml` can execute on PRs labeled
`gpu-ci` and on every push to master.

**Audience**: NA2 setup. One-time install.

## What this gives you

- GPU CI runs on **push to master** (always) and on **labeled PRs** (opt-in).
- Workflow validates 40+ GPU tests in ~10s of compute (currently 7.1s wall on
  this machine — `tests/test_gpu.py`, `tests/test_gpu_model_parity.py`, and
  `tests/test_impute_gpu_kernels.py`).
- Catches the class of bug Pillar C surfaced 5 of: silent CUDA fallback (a
  CUDA tensor is moved to CPU mid-pipeline because of a missing `.to(device)`).

## Security model

The well-known risk with self-hosted runners + PRs is that anyone who can
open a PR can run arbitrary code on your machine. The workflow mitigates
this with three layered controls:

1. **`pull_request_target` not `pull_request`** — the workflow YAML used is
   the one from `master` (trusted), not from the PR branch. So a malicious
   PR cannot rewrite the workflow itself to disable security.
2. **Label gating (`labeled` event only)** — the job runs only when a
   maintainer adds the `gpu-ci` label. The maintainer is implicitly
   reviewing the code before allowing it on the runner.
3. **No `synchronize` subscription** — once labeled, the job runs ONCE on
   that commit. New commits to the PR don't auto-rerun. To rerun, the
   maintainer removes and re-adds the label (forcing a fresh review).

This is the GitHub-documented pattern. See
<https://securitylab.github.com/research/github-actions-preventing-pwn-requests/>
for the full threat model.

## One-time setup

### 1. Create the runner registration token

In a browser: navigate to the GitHub repo → Settings → Actions → Runners →
"New self-hosted runner" → choose Linux/x64. Copy the registration token
(it's a one-shot, expires in ~1 hour).

### 2. Install the runner agent

```bash
# Pick a directory (NOT inside the repo, to avoid any chance of the runner's
# files getting committed). $HOME/actions-runner is the GitHub-recommended
# default.
mkdir -p $HOME/actions-runner && cd $HOME/actions-runner

# Download the latest runner (check
# https://github.com/actions/runner/releases for the current version)
RUNNER_VER=2.319.1   # CHECK FOR LATEST
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/download/v${RUNNER_VER}/actions-runner-linux-x64-${RUNNER_VER}.tar.gz
tar xzf ./actions-runner-linux-x64.tar.gz

# Configure with the token from step 1. CRITICAL: assign the labels
# `linux,gpu` so the workflow's `runs-on: [self-hosted, linux, gpu]` matches.
./config.sh \
  --url https://github.com/<owner>/<repo> \
  --token <REGISTRATION_TOKEN> \
  --name "rtx2000-ada-host" \
  --labels "linux,gpu" \
  --work _work \
  --runasservice
```

### 3. Install + start as a systemd service

```bash
# From the actions-runner directory
sudo ./svc.sh install $USER
sudo ./svc.sh start
sudo ./svc.sh status   # should report "active (running)"
```

The runner now polls GitHub for jobs targeted at
`[self-hosted, linux, gpu]`.

### 4. Verify CUDA visibility for the runner user

```bash
# As the user the runner runs under
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns False, ensure the runner user is in
the `video` group or whichever group owns `/dev/nvidia*`. On Ubuntu/RHEL:

```bash
sudo usermod -a -G video,render $USER
# logout/login (or restart the runner service) for group membership to take effect
sudo ./svc.sh stop && sudo ./svc.sh start
```

### 5. Create the `gpu-ci` repo label

In a browser: GitHub repo → Issues → Labels → "New label" → name `gpu-ci`,
description "Trigger GPU CI on this PR (maintainer-only)".

Restrict who can apply the label via repo permissions (Settings →
Collaborators) — only users with Triage / Write access can add labels by
default, which is the right gate.

### 6. Trigger a smoke run

```bash
# From a workstation, with the gh CLI
gh workflow run gpu.yml --ref master
gh run list --workflow=gpu.yml --limit 1
```

Or push a no-op commit to master and watch the GPU job run automatically.

## Routine maintenance

- **Runner updates**: GitHub auto-updates the agent. If a release breaks
  something, `cd $HOME/actions-runner && ./svc.sh stop && ./config.sh remove`,
  then re-register with a fresh token.
- **Driver updates**: when bumping NVIDIA driver, re-test
  `python -c "import torch; print(torch.cuda.is_available())"` and re-run
  the `gpu.yml` workflow manually before merging anything.
- **Disk hygiene**: `_work/` accumulates checkout copies. `du -sh
  $HOME/actions-runner/_work` periodically; prune with `./svc.sh stop &&
  rm -rf _work && ./svc.sh start` if it grows past 10 GB.

## How a maintainer triggers GPU CI on a PR

1. Open or comment on the PR.
2. Apply the `gpu-ci` label.
3. The `gpu-tests` job appears in the PR's Checks tab and runs on the local
   runner.
4. To re-run after a new push to the PR, **remove and re-add the label**
   (the workflow does not subscribe to `synchronize` — see security model).

## How to remove a runner

If the runner is decommissioned:

```bash
cd $HOME/actions-runner
sudo ./svc.sh stop
sudo ./svc.sh uninstall
./config.sh remove --token <REMOVAL_TOKEN>  # token from GitHub Settings
```

This deregisters the runner and stops the systemd service.
