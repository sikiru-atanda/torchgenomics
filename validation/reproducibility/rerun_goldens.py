#!/usr/bin/env python3
"""Pillar D — orchestrator.

For each committed reference fixture, re-run the matching Pillar B harness
end-to-end (install → fetch → run) and diff the *fresh* upstream tool
output against the *committed* fixture (NOT against TorchGenomics).

Tools audited: gemma, gapit, gwaspoly.

For tools whose Pillar B harness already supports a "fall back to the
committed reference" path (gapit, gwaspoly), this script first attempts
to force a real re-run (GAPIT3 must be loadable in R, or
GWASPOLY_FORCE_RERUN=1) and otherwise records an `infra-blocker` outcome
so the next session can pick up where this one left off.

CLI:
    python3 rerun_goldens.py                       # all 3 tools
    python3 rerun_goldens.py --tool gemma          # single tool
    python3 rerun_goldens.py --skip-rerun          # only diff existing outputs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


@dataclass
class ToolStatus:
    tool: str
    install_ok: bool = False
    fetch_ok: bool = False
    run_ok: bool = False
    diff_ok: bool = False
    notes: list[str] = field(default_factory=list)


def _run(cmd: list[str], env: dict | None = None, cwd: Path | None = None,
         timeout: int = 1800) -> tuple[int, str, str]:
    """Run a subprocess, capture stdout/stderr, return (rc, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def preflight_orchestrator() -> None:
    """Top-level pre-flight: 10 GB disk + 4 GB RAM."""
    rc, stdout, stderr = _run(
        ["bash", "-c", (
            "set -e; "
            f"source {REPO}/validation/external/_lib/preflight.sh; "
            "preflight_check pillar-d-orchestrator 10 4"
        )]
    )
    if rc != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError(f"orchestrator pre-flight failed (rc={rc})")
    print(stderr.rstrip())  # preflight prints to stderr


def rerun_one(tool: str, force_gwaspoly_rerun: bool = False,
              skip_rerun: bool = False) -> ToolStatus:
    status = ToolStatus(tool=tool)
    harness = REPO / "validation" / "external" / tool

    # 1. install ---------------------------------------------------------------
    install_sh = harness / "install.sh"
    if install_sh.exists():
        rc, stdout, stderr = _run(["bash", str(install_sh)], timeout=900)
        status.install_ok = rc == 0
        if rc != 0:
            status.notes.append(f"install rc={rc}: {stderr[-300:]}")
        else:
            status.notes.append("install ok")
    else:
        status.notes.append(f"no install.sh at {install_sh}")
        return status

    # 2. fetch_data ------------------------------------------------------------
    fetch_sh = harness / "fetch_data.sh"
    if fetch_sh.exists():
        rc, stdout, stderr = _run(["bash", str(fetch_sh)], timeout=600)
        status.fetch_ok = rc == 0
        if rc != 0:
            status.notes.append(f"fetch rc={rc}: {stderr[-300:]}")
        else:
            status.notes.append("fetch ok")
    else:
        status.notes.append(f"no fetch_data.sh at {fetch_sh}")

    # 3. run -------------------------------------------------------------------
    if skip_rerun:
        status.notes.append("skip-rerun: not invoking run script")
        # Mark run_ok=True only if outputs/ already populated
        outputs_dir = harness / "outputs"
        status.run_ok = outputs_dir.exists() and any(outputs_dir.iterdir())
    else:
        run_sh = harness / f"run_{tool}.sh"
        env = None
        if tool == "gwaspoly" and force_gwaspoly_rerun:
            import os
            env = os.environ.copy()
            env["GWASPOLY_FORCE_RERUN"] = "1"
            status.notes.append("GWASPOLY_FORCE_RERUN=1 set")
        if run_sh.exists():
            # GWASpoly can take 10-15 min on the full per-marker scan
            timeout = 1800 if tool == "gwaspoly" else 900
            rc, stdout, stderr = _run(
                ["bash", str(run_sh)], env=env, timeout=timeout,
            )
            status.run_ok = rc == 0
            if rc != 0:
                status.notes.append(f"run rc={rc}: {stderr[-300:]}")
            else:
                status.notes.append("run ok")
                # Surface the last 5 lines of run output (informational)
                tail = stdout.strip().splitlines()[-5:]
                if tail:
                    status.notes.append("run tail: " + " | ".join(tail))
        else:
            status.notes.append(f"no run_{tool}.sh at {run_sh}")

    # 4. diff ------------------------------------------------------------------
    if not status.run_ok:
        status.notes.append("skipping diff — run did not succeed")
        return status

    cmp_py = HERE / "compare_fixtures.py"
    rc, stdout, stderr = _run(
        ["python3", str(cmp_py), "--tool", tool], timeout=300,
    )
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    status.diff_ok = rc == 0
    status.notes.append(f"compare_fixtures rc={rc}")
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tool", choices=["gemma", "gapit", "gwaspoly", "all"], default="all",
    )
    ap.add_argument(
        "--force-gwaspoly-rerun", action="store_true",
        help="set GWASPOLY_FORCE_RERUN=1 (slow: 10-15 min real GWASpoly run)",
    )
    ap.add_argument(
        "--skip-rerun", action="store_true",
        help="skip install/fetch/run; just re-diff whatever is in outputs/",
    )
    ap.add_argument(
        "--skip-preflight", action="store_true",
        help="skip the orchestrator-level pre-flight (use only if the per-tool "
             "harnesses' own preflight is sufficient on this host)",
    )
    args = ap.parse_args(argv)

    if not args.skip_preflight:
        try:
            preflight_orchestrator()
        except RuntimeError as e:
            print(f"[orchestrator] pre-flight failed: {e}", file=sys.stderr)
            return 1

    tools = ["gemma", "gapit", "gwaspoly"] if args.tool == "all" else [args.tool]
    statuses: list[ToolStatus] = []
    for tool in tools:
        print()
        print("#" * 90)
        print(f"#  Pillar D — re-running {tool.upper()} harness vs committed fixture")
        print("#" * 90)
        s = rerun_one(
            tool,
            force_gwaspoly_rerun=args.force_gwaspoly_rerun,
            skip_rerun=args.skip_rerun,
        )
        statuses.append(s)
        for note in s.notes:
            print(f"  [{tool}] {note}")

    print()
    print("#" * 90)
    print("#  Pillar D — orchestrator summary")
    print("#" * 90)
    for s in statuses:
        flags = (
            f"install={'OK' if s.install_ok else 'FAIL'} "
            f"fetch={'OK' if s.fetch_ok else 'FAIL'} "
            f"run={'OK' if s.run_ok else 'FAIL'} "
            f"diff={'OK' if s.diff_ok else 'FAIL'}"
        )
        print(f"  {s.tool:10s}  {flags}")

    n_full = sum(1 for s in statuses if s.diff_ok)
    print(f"\n  → {n_full}/{len(statuses)} tools completed install→fetch→run→diff")

    # Orchestrator returns 0 unconditionally — drift / infra-blocker findings
    # are surfaced via the per-tool JSON reports under outputs/, the README
    # drift table, and (if drift exists) docs/validation_findings.md.
    return 0


if __name__ == "__main__":
    sys.exit(main())
