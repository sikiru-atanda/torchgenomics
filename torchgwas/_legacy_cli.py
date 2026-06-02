"""Deprecated CLI alias: print one-shot banner then proxy to ``torchgenomics``."""
from __future__ import annotations

import sys

_BANNER = (
    "[deprecation] The `torchgwas` CLI has been renamed to `torchgenomics`. "
    "Use `torchgenomics <subcommand>` going forward; the `torchgwas` alias will "
    "be removed in v1.0.0.\n"
)


def main(argv=None) -> int:
    sys.stderr.write(_BANNER)
    sys.stderr.flush()
    from torchgenomics.cli import main as _real_main

    return _real_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
