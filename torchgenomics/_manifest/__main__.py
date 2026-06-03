# torchgenomics/_manifest/__main__.py
"""Console entry: write the manifest as JSON to stdout."""
from __future__ import annotations

import json
import sys

from . import build_manifest


def main() -> None:
    json.dump(build_manifest(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
