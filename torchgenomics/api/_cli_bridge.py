"""Run tier-2 CLI subcommands programmatically for Python/R parity.

The friendly api exposes rich, typed functions for tier-1 workflows. The many
tier-2 CLI subcommands (bayes-scan, ldsc, mediate, ...) don't each have a typed
api facade; this bridge runs them via ``torchgenomics.cli.main`` and returns a
uniform :class:`CliRun` so R (and Python) callers can reach them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import timed
from ._results import CliRun


def _kwargs_to_argv(kwargs: dict[str, Any]) -> list[str]:
    """Convert ``run_cli_subcommand`` keyword arguments to an argv flag list.

    ``key`` -> ``--key-with-dashes``. ``True`` becomes a bare flag (for
    ``store_true`` switches); ``False``/``None`` are omitted entirely (so
    callers can pass a keyword without forcing the flag on); a ``list``/
    ``tuple`` becomes the flag followed by each stringified element (for
    ``nargs``-style options); everything else becomes ``flag, str(value)``.
    """
    argv: list[str] = []
    for key, val in kwargs.items():
        flag = "--" + str(key).replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv += [str(v) for v in val]
        else:
            argv += [flag, str(val)]
    return argv


def _collect_output_files(kwargs: dict[str, Any]) -> dict[str, Path]:
    """Best-effort discovery of output paths a subcommand wrote.

    Tier-2 subcommands write their own outputs (paths named via
    ``--output``/``--output-dir``/``--output-prefix``); this bridge doesn't
    know each command's exact output schema, so it only reports what it can
    verify exists on disk after the run — the directory/file at the output
    path itself, plus the common ``.assoc.tsv`` / ``.tsv`` suffix variants.
    """
    out: dict[str, Path] = {}
    for key in ("output", "output_dir", "output_prefix"):
        base = kwargs.get(key)
        if not base:
            continue
        p = Path(str(base))
        if p.exists():
            out[key] = p
        for suffix in (".assoc.tsv", ".tsv"):
            cand = Path(str(base) + suffix)
            if cand.exists():
                out["tsv"] = cand
    return out


def run_cli_subcommand(subcommand: str, **kwargs) -> CliRun:
    """Run a tier-2 CLI ``subcommand`` programmatically and return a ``CliRun``.

    ``kwargs`` are converted to CLI flags (``key`` -> ``--key-with-dashes``;
    ``True`` -> bare flag; ``False``/``None`` -> omitted; list -> repeated
    values for ``nargs``). Raises a friendly :class:`RuntimeError` (never a bare
    ``SystemExit`` or an internal exception type) — naming ``subcommand`` and
    the underlying cause — when argparse rejects the arguments, the command
    exits non-zero, or the command's handler raises.
    """
    from ..cli import main

    argv = [subcommand, *_kwargs_to_argv(kwargs)]
    with timed() as elapsed:
        try:
            rc = main(argv)
        except SystemExit as e:  # argparse error / explicit sys.exit
            code = e.code if isinstance(e.code, int) else 2
            raise RuntimeError(
                f"torchgenomics {subcommand} failed (exit {code}); check the "
                f"arguments and input paths."
            ) from e
        except Exception as e:  # handler raised (bad path, bad data, ...)
            raise RuntimeError(
                f"torchgenomics {subcommand} failed: {e}"
            ) from e

        # A handler returning None on success is treated as exit code 0 —
        # `main`'s type signature promises int, but individual handlers are
        # plain functions and a missing `return 0` at the end is an easy typo
        # to make; don't let that surface as a `TypeError` here.
        rc = rc if rc is not None else 0

        if rc != 0:
            raise RuntimeError(
                f"torchgenomics {subcommand} failed (exit {rc}); check the "
                f"arguments and input paths."
            )
        return CliRun(
            runtime_s=elapsed(),
            output_files=_collect_output_files(kwargs),
            command=subcommand,
            exit_code=int(rc),
            args={str(k): (str(v) if isinstance(v, Path) else v) for k, v in kwargs.items()},
        )
