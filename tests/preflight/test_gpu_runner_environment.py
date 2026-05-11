"""Meta-tests for the NA2 GPU CI scaffold.

These tests do NOT require a GPU. They validate that the CI scaffold
itself is well-formed:

* the workflow YAML parses and contains the expected security-critical
  configuration (label gating, ephemeral runner labels, no secrets
  injected post-checkout);
* the setup script exists, is executable, and has no syntax errors;
* the operations doc exists and references the workflow + script.

Run from the repo root:

    pytest tests/preflight/test_gpu_runner_environment.py -v

This is the regression net for the scaffold itself: if someone deletes
the label-gate condition by mistake, this test will fail in the default
CI (which does not require GPU).
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gpu.yml"
SCRIPT = REPO_ROOT / "scripts" / "setup_gpu_runner.sh"
OPS_DOC = REPO_ROOT / "docs" / "operations" / "gpu-ci.md"


# ---------------------------------------------------------------------------
# Workflow YAML
# ---------------------------------------------------------------------------


def test_workflow_file_exists() -> None:
    assert WORKFLOW.is_file(), f"Missing workflow file: {WORKFLOW}"


def test_workflow_yaml_parses() -> None:
    yaml = pytest.importorskip("yaml")
    with WORKFLOW.open() as fh:
        parsed = yaml.safe_load(fh)
    assert isinstance(parsed, dict), "workflow YAML did not parse to a mapping"
    assert "jobs" in parsed, "workflow has no `jobs:` key"
    # PyYAML parses the bare key `on:` as Python True (YAML 1.1 boolean).
    # Accept either the boolean key or the string key for forward compat.
    assert ("on" in parsed) or (True in parsed), "workflow has no `on:` key"


def test_workflow_uses_pull_request_target_not_pull_request() -> None:
    """Security-critical: must use pull_request_target, never pull_request.

    `pull_request` would let a fork rewrite this workflow and execute
    arbitrary commands on the maintainer's GPU. See docs/operations/gpu-ci.md
    section 2 ("Security model") and the GitHub Security Lab "Pwn Requests"
    writeup cited there.
    """
    text = WORKFLOW.read_text()
    assert "pull_request_target" in text, (
        "workflow must trigger on `pull_request_target`, not `pull_request`. "
        "See docs/operations/gpu-ci.md section 2."
    )
    # Look for `pull_request:` at the start of a line (not as a substring of
    # pull_request_target). Allow indentation.
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("pull_request:"):
            pytest.fail(
                "workflow uses bare `pull_request:` trigger — this is a "
                "self-hosted-runner pwn vector. Use `pull_request_target`."
            )


def test_workflow_has_label_gate() -> None:
    """Job must require the `gpu-tested` label before running on PRs."""
    text = WORKFLOW.read_text()
    assert "gpu-tested" in text, (
        "workflow must reference the `gpu-tested` label in its `if:` gate."
    )
    assert "github.event.pull_request.labels" in text, (
        "workflow must consult `github.event.pull_request.labels` in its `if:` gate."
    )


def test_workflow_runs_on_dedicated_runner() -> None:
    """Job must require all three runner labels (defense in depth)."""
    text = WORKFLOW.read_text()
    # `runs-on` may be either an inline list or a YAML flow seq.
    for required in ("self-hosted", "gpu", "rtx-2000-ada"):
        assert required in text, (
            f"workflow must include runner label '{required}' so a "
            f"misconfigured CPU runner cannot pick up the job."
        )


def test_workflow_has_nightly_cron() -> None:
    text = WORKFLOW.read_text()
    assert "schedule:" in text, "workflow missing `schedule:` for nightly drift detection"
    assert "cron:" in text, "workflow missing `cron:` line under schedule"


def test_workflow_uploads_artifacts() -> None:
    text = WORKFLOW.read_text()
    assert "actions/upload-artifact" in text, (
        "workflow must upload pytest JUnit XML + nvidia-smi snapshots as artifacts"
    )
    assert "junit-gpu.xml" in text, "workflow must produce JUnit XML for downstream tooling"
    assert "nvidia-smi" in text, "workflow must capture nvidia-smi output"


def test_workflow_disables_persist_credentials() -> None:
    """When checking out PR head under pull_request_target, the GitHub
    token must not be persisted to disk for downstream untrusted steps.

    See: https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#using-secrets
    """
    text = WORKFLOW.read_text()
    assert "persist-credentials: false" in text, (
        "checkout step must set `persist-credentials: false` when running "
        "untrusted PR head code."
    )


# ---------------------------------------------------------------------------
# Setup script
# ---------------------------------------------------------------------------


def test_setup_script_exists() -> None:
    assert SCRIPT.is_file(), f"Missing setup script: {SCRIPT}"


def test_setup_script_is_executable() -> None:
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"{SCRIPT} is not executable. Run: chmod +x {SCRIPT}"
    )


def test_setup_script_has_valid_shell_syntax() -> None:
    """`bash -n` does parse-only validation; no commands execute."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this host")
    result = subprocess.run(
        [bash, "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n {SCRIPT} reported syntax errors:\n{result.stderr}"
    )


def test_setup_script_does_not_auto_register() -> None:
    """The script must NOT call `./config.sh` directly.

    Per the NA2 hard contract: registration requires the user's admin
    token, which the script cannot fetch. It should print the command
    for the operator to run, never execute it.

    We walk the script line by line, tracking heredoc state, and only
    flag `config.sh` invocations that appear in actual executable
    context (not inside `cat <<EOF` instruction blocks).
    """
    text = SCRIPT.read_text()
    in_heredoc = False
    heredoc_terminator: str | None = None

    for line in text.splitlines():
        stripped = line.strip()

        # Track heredoc state. Match `<<EOF`, `<<'EOF'`, `<<"EOF"`, etc.
        if not in_heredoc:
            # Detect a heredoc opener of the form `<<TAG` or `<<'TAG'`.
            for marker in ("<<'EOF'", '<<"EOF"', "<<EOF"):
                if marker in stripped:
                    in_heredoc = True
                    heredoc_terminator = "EOF"
                    break
        else:
            if stripped == heredoc_terminator:
                in_heredoc = False
                heredoc_terminator = None
            # Anything inside a heredoc is text, not code.
            continue

        # Skip pure-comment lines.
        if stripped.startswith("#"):
            continue

        # Look for an actual invocation of config.sh in executable
        # context: a command starting with `./config.sh` or with
        # `${RUNNER_DIR}/config.sh`.
        if stripped.startswith("./config.sh"):
            pytest.fail(
                f"setup script appears to invoke config.sh directly "
                f"in executable context: {line!r}. Per NA2 hard contract "
                f"this must be a printed instruction, not an executed "
                f"command."
            )
        if stripped.startswith("${RUNNER_DIR}/config.sh") or stripped.startswith(
            "$RUNNER_DIR/config.sh"
        ):
            pytest.fail(
                f"setup script appears to invoke config.sh directly: {line!r}"
            )


def test_setup_script_supports_dry_run() -> None:
    text = SCRIPT.read_text()
    assert "--dry-run" in text, "setup script must support a --dry-run mode"


# ---------------------------------------------------------------------------
# Operations doc
# ---------------------------------------------------------------------------


def test_ops_doc_exists() -> None:
    assert OPS_DOC.is_file(), f"Missing ops doc: {OPS_DOC}"


def test_ops_doc_addresses_security_model() -> None:
    text = OPS_DOC.read_text()
    # Must explicitly discuss pull_request_target risk + mitigation.
    for required in (
        "pull_request_target",
        "Security model",
        "ephemeral",
        "label",
    ):
        assert required.lower() in text.lower(), (
            f"ops doc missing required section/term: {required!r}"
        )


def test_ops_doc_references_pillar_c_history() -> None:
    text = OPS_DOC.read_text()
    # The two canonical commits cited in the NA2 brief.
    assert "adc7b04" in text, "ops doc must cite commit adc7b04 (Pillar C device-alignment fix)"
    assert "1e27d81" in text, "ops doc must cite commit 1e27d81 (post-campaign device-alignment fix)"


def test_ops_doc_references_workflow_and_script() -> None:
    text = OPS_DOC.read_text()
    assert "gpu.yml" in text, "ops doc must reference the workflow file by name"
    assert "setup_gpu_runner.sh" in text, "ops doc must reference the setup script by name"


# ---------------------------------------------------------------------------
# Cross-cutting: file layout exists end-to-end
# ---------------------------------------------------------------------------


def test_scaffold_complete() -> None:
    """One assertion per scaffold artifact, so a missing file is obvious."""
    artifacts = {
        "workflow": WORKFLOW,
        "setup script": SCRIPT,
        "ops doc": OPS_DOC,
        "preflight test (this file)": Path(__file__),
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    assert not missing, f"Scaffold incomplete; missing: {missing}"


def test_repo_root_resolution_is_sane() -> None:
    """Sanity: REPO_ROOT should look like a TorchGWAS checkout."""
    assert (REPO_ROOT / "pyproject.toml").is_file(), (
        f"REPO_ROOT does not look like a TorchGWAS checkout: {REPO_ROOT}"
    )
    assert (REPO_ROOT / "torchgwas").is_dir(), (
        f"REPO_ROOT missing torchgwas/ package: {REPO_ROOT}"
    )


# ---------------------------------------------------------------------------
# Smoke: the GPU test files the workflow targets actually exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_file",
    [
        "tests/test_gpu.py",
        "tests/test_gpu_model_parity.py",
        "tests/test_impute_gpu.py",
        "tests/test_impute_gpu_kernels.py",
    ],
)
def test_gpu_test_file_exists(test_file: str) -> None:
    """Workflow names these files explicitly; if any is renamed/removed,
    the workflow will silently miss coverage. Fail loud here instead.
    """
    path = REPO_ROOT / test_file
    assert path.is_file(), (
        f"GPU test file referenced by workflow is missing: {test_file}. "
        f"Either restore it or update .github/workflows/gpu.yml."
    )


def test_gpu_marker_registered_in_pyproject() -> None:
    """The workflow uses `pytest -m gpu`; the marker must be registered
    so pytest does not warn-and-skip under `--strict-markers`.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    text = pyproject.read_text()
    assert "gpu:" in text, "pyproject.toml must register the `gpu` pytest marker"
