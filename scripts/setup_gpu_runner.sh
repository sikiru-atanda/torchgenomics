#!/usr/bin/env bash
# scripts/setup_gpu_runner.sh
#
# NA2 — Self-hosted GitHub Actions runner setup for the on-machine
# NVIDIA RTX 2000 Ada Generation. This script PREPARES the environment
# but DOES NOT register the runner — registration requires a one-shot
# admin token from
#   https://github.com/<org>/<repo>/settings/actions/runners/new
# which only the repo admin (the user) can mint. The script writes out
# the exact `./config.sh` invocation you should run with that token.
#
# Hard contract — what this script does NOT do:
#
#   * It does NOT call `./config.sh` (that requires the admin token).
#   * It does NOT install systemd units or run sudo commands by default.
#     Use `--install-systemd` to opt in; it will print every sudo command
#     before running it and abort on `--dry-run`.
#   * It does NOT modify firewall, network, or GPU driver state.
#   * It does NOT push to any remote, change git config, or fetch from
#     anywhere except api.github.com (runner release metadata) and
#     github.com (the runner tarball download).
#
# Idempotency:
#
#   Re-running on a host with an existing actions-runner directory will
#   detect it and offer to upgrade, skip, or abort. It will NEVER blow
#   away an existing registered runner — you must `./config.sh remove`
#   first if you want a clean slate.
#
# Usage:
#
#   bash scripts/setup_gpu_runner.sh                  # default: just prepare
#   bash scripts/setup_gpu_runner.sh --dry-run        # show what would happen
#   bash scripts/setup_gpu_runner.sh --install-systemd  # offer systemd setup
#   bash scripts/setup_gpu_runner.sh --runner-dir DIR # custom install location
#
# References:
#   https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners
#   https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-self-hosted-runners

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults & arg parsing
# ---------------------------------------------------------------------------

RUNNER_DIR="${HOME}/actions-runner"
DRY_RUN=0
INSTALL_SYSTEMD=0
RUNNER_LABELS="self-hosted,gpu,rtx-2000-ada"
RUNNER_NAME="$(hostname -s)-gpu"
RUNNER_GROUP="default"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --install-systemd) INSTALL_SYSTEMD=1; shift ;;
        --runner-dir)     RUNNER_DIR="$2"; shift 2 ;;
        --labels)         RUNNER_LABELS="$2"; shift 2 ;;
        --name)           RUNNER_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 64
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

err()   { printf '\033[31m[ERR]\033[0m  %s\n' "$*" >&2; }
warn()  { printf '\033[33m[WARN]\033[0m %s\n' "$*" >&2; }
info()  { printf '\033[36m[INFO]\033[0m %s\n' "$*" >&2; }
ok()    { printf '\033[32m[OK]\033[0m   %s\n' "$*" >&2; }
step()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*" >&2; }

# Run a command (or print it under --dry-run). If sudo is requested,
# prompt the operator before executing.
run_cmd() {
    local desc="$1"; shift
    info "${desc}"
    printf '  $ %s\n' "$*" >&2
    if (( DRY_RUN )); then
        warn "(dry-run) skipped"
        return 0
    fi
    "$@"
}

run_sudo() {
    local desc="$1"; shift
    info "${desc} (requires sudo)"
    printf '  $ sudo %s\n' "$*" >&2
    if (( DRY_RUN )); then
        warn "(dry-run) skipped"
        return 0
    fi
    if ! sudo -n true 2>/dev/null; then
        warn "sudo will prompt for your password."
    fi
    sudo "$@"
}

# ---------------------------------------------------------------------------
# Pre-flight: host environment checks
# ---------------------------------------------------------------------------

step "Pre-flight: host environment"

# 1. Linux x86_64 (the runner tarball we download is linux-x64).
if [[ "$(uname -s)" != "Linux" ]]; then
    err "This script supports Linux hosts only (got: $(uname -s))."
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    err "Expected x86_64 (got: $(uname -m)). Update RUNNER_PKG below for arm64."
    exit 1
fi
ok "Host is Linux x86_64."

# 2. CUDA driver visible.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found on PATH. Install the NVIDIA driver before"
    err "registering this host as a GPU runner."
    exit 1
fi
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
ok "NVIDIA driver detected: ${DRIVER_VERSION}"

# 3. RTX 2000 Ada is the expected device. We warn (don't abort) if it's
#    something else — the workflow will run on any CUDA device, but the
#    label `rtx-2000-ada` is now misleading.
DEVICE_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ -z "${DEVICE_NAME}" ]]; then
    err "nvidia-smi reports no devices. The driver is installed but no GPU is"
    err "visible — check 'lspci | grep -i nvidia' and your kernel module."
    exit 1
fi
info "Detected GPU: ${DEVICE_NAME}"
if ! echo "${DEVICE_NAME}" | grep -qi "RTX 2000 Ada"; then
    warn "GPU is not 'NVIDIA RTX 2000 Ada Generation'. The default labels"
    warn "include 'rtx-2000-ada' — pass --labels to override."
fi

# 4. sudo availability (only required if --install-systemd).
if (( INSTALL_SYSTEMD )); then
    if ! command -v sudo >/dev/null 2>&1; then
        err "--install-systemd requested but sudo is not available."
        exit 1
    fi
    info "sudo is available; systemd install will prompt as needed."
fi

# 5. curl / tar / sha256sum present.
for tool in curl tar sha256sum; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        err "Required tool not found: ${tool}"
        exit 1
    fi
done
ok "Required tools present: curl, tar, sha256sum."

# 6. Disk headroom (runner + repo workspace + pip cache).
#    Conservatively require 20 GB free in the runner-dir parent.
RUNNER_DIR_PARENT="$(dirname "${RUNNER_DIR}")"
mkdir -p "${RUNNER_DIR_PARENT}"
AVAIL_GB="$(df --output=avail -BG "${RUNNER_DIR_PARENT}" | tail -1 | tr -d 'G ')"
if (( AVAIL_GB < 20 )); then
    err "Insufficient disk in ${RUNNER_DIR_PARENT}: ${AVAIL_GB} GB free (need >= 20)."
    exit 1
fi
ok "Disk headroom in ${RUNNER_DIR_PARENT}: ${AVAIL_GB} GB free."

# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

step "Idempotency check: existing runner?"

if [[ -d "${RUNNER_DIR}" && -f "${RUNNER_DIR}/config.sh" ]]; then
    info "Found existing runner installation at ${RUNNER_DIR}"
    if [[ -f "${RUNNER_DIR}/.runner" ]]; then
        info "Runner is already registered with GitHub:"
        cat "${RUNNER_DIR}/.runner" | sed 's/^/    /'
        warn "This script will NOT modify the registered runner."
        warn "To re-register: cd ${RUNNER_DIR} && ./config.sh remove --token <REMOVAL_TOKEN>"
        warn "then re-run this script."
        echo
        info "If you only want to upgrade the runner binary in place, run:"
        info "  cd ${RUNNER_DIR} && ./run.sh --once   # confirm current version"
        info "  ./svc.sh stop && ... # then re-extract the tarball"
        exit 0
    else
        warn "Runner directory exists but is unregistered. Will refresh binaries."
    fi
fi

# ---------------------------------------------------------------------------
# Download runner tarball
# ---------------------------------------------------------------------------

step "Download GitHub Actions runner"

# Look up the latest release. We hit the un-authenticated public API,
# which is rate-limited but sufficient for one-shot setup.
LATEST_JSON="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest)"
LATEST_TAG="$(echo "${LATEST_JSON}" | grep -oE '"tag_name":\s*"[^"]+"' | head -1 | sed 's/.*"v\?\([^"]*\)"/\1/')"

if [[ -z "${LATEST_TAG}" ]]; then
    err "Could not determine latest runner version from api.github.com."
    err "Check connectivity or set RUNNER_VERSION explicitly in this script."
    exit 1
fi
info "Latest actions/runner release: v${LATEST_TAG}"

RUNNER_PKG="actions-runner-linux-x64-${LATEST_TAG}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${LATEST_TAG}/${RUNNER_PKG}"

mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"

if [[ -f "${RUNNER_PKG}" ]]; then
    info "${RUNNER_PKG} already present; skipping download."
else
    run_cmd "Downloading runner tarball" \
        curl -fsSLo "${RUNNER_PKG}" "${RUNNER_URL}"
fi

# We do NOT pin a checksum in this script because the upstream checksum
# is published per-release on the release page. The right contract is:
# operator verifies the .tar.gz matches the SHA256 listed at
#   https://github.com/actions/runner/releases/tag/v${LATEST_TAG}
# before extracting.
warn "Verify the tarball checksum manually before extraction:"
warn "  Expected SHA256: see https://github.com/actions/runner/releases/tag/v${LATEST_TAG}"
warn "  Local SHA256:    $(sha256sum "${RUNNER_PKG}" | awk '{print $1}')"

if [[ ! -f "config.sh" ]]; then
    run_cmd "Extracting runner tarball" \
        tar xzf "${RUNNER_PKG}"
else
    info "config.sh already present; tarball already extracted."
fi

ok "Runner binaries ready in ${RUNNER_DIR}"

# ---------------------------------------------------------------------------
# Print the registration command (operator runs this manually)
# ---------------------------------------------------------------------------

step "Manual step: register runner with GitHub"

cat <<EOF >&2

The runner binaries are installed. The next two steps require YOUR admin
access to the repository — this script cannot do them for you.

  1. Open in a browser:
       https://github.com/<ORG>/<REPO>/settings/actions/runners/new
     Choose "Linux x64". GitHub will display a one-shot registration
     token of the form 'A...' (valid for ~1 hour).

  2. Run, in this shell, with that token (substitute YOUR_TOKEN):

       cd ${RUNNER_DIR}
       ./config.sh \\
         --url    https://github.com/<ORG>/<REPO> \\
         --token  YOUR_TOKEN \\
         --name   ${RUNNER_NAME} \\
         --labels ${RUNNER_LABELS} \\
         --runnergroup ${RUNNER_GROUP} \\
         --work   _work \\
         --ephemeral \\
         --unattended

The --ephemeral flag is REQUIRED for the security model documented in
docs/operations/gpu-ci.md. Without it, a compromised job could persist
state into subsequent jobs.

EOF

# ---------------------------------------------------------------------------
# Optional: install systemd unit
# ---------------------------------------------------------------------------

if (( INSTALL_SYSTEMD )); then
    step "Install systemd unit (optional, requires sudo)"

    if [[ ! -f "${RUNNER_DIR}/.runner" ]]; then
        warn "Runner is not yet registered (no .runner file)."
        warn "Skip systemd install for now; re-run with --install-systemd"
        warn "AFTER you have run config.sh per the manual step above."
    else
        info "Will run the upstream svc.sh installer to create a systemd unit."
        info "This wraps `systemctl enable --now actions.runner.<org>.<host>.service`."
        if [[ ! -x "${RUNNER_DIR}/svc.sh" ]]; then
            err "${RUNNER_DIR}/svc.sh missing. Re-extract the tarball."
            exit 1
        fi

        # svc.sh internally calls sudo. We inherit its prompt.
        run_cmd "Installing systemd service" \
            "${RUNNER_DIR}/svc.sh" install
        run_cmd "Starting systemd service" \
            "${RUNNER_DIR}/svc.sh" start
        run_cmd "Verifying service status" \
            "${RUNNER_DIR}/svc.sh" status

        ok "Systemd unit installed. Logs: journalctl -u actions.runner.* -f"
    fi
else
    info "Systemd install skipped (pass --install-systemd to enable)."
    info "To start the runner manually for testing: cd ${RUNNER_DIR} && ./run.sh"
fi

step "Setup complete"
ok "Runner directory: ${RUNNER_DIR}"
ok "Runner labels:    ${RUNNER_LABELS}"
ok "Runner name:      ${RUNNER_NAME}"
info "Next step: register the runner per the printed instructions."
