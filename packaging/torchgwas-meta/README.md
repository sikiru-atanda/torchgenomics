# torchgwas — deprecation shim

This package is the **deprecated alias** for
[`torchgenomics`](https://pypi.org/project/torchgenomics/). It exists only
to give existing `torchgwas` v0.3.x users a smooth migration path:

```bash
pip install --upgrade torchgwas    # pulls in torchgenomics==0.4.0 automatically
```

After upgrade, your existing code keeps working:

```python
import torchgwas                          # DeprecationWarning fires once
from torchgwas.models import SingleTraitLMM
```

…but every `torchgwas[.X]` import is silently redirected to
`torchgenomics[.X]` through a meta-path finder. Same module objects, same
behaviour.

**This shim is removed in v1.0.0.** Migrate to `torchgenomics` at your
convenience:

```diff
- import torchgwas
+ import torchgenomics
- from torchgwas.models import SingleTraitLMM
+ from torchgenomics.models import SingleTraitLMM
```

## What you get

- A tiny `torchgwas` Python package (just `__init__.py` + `_legacy_cli.py`)
- The `torchgwas` CLI entry point — prints a deprecation banner once,
  then forwards every argument to `torchgenomics`.
- A hard pin on `torchgenomics==0.4.0` so this shim and the real package
  stay version-locked.

## Why a separate distribution?

Up through v0.3.10, `torchgwas` was a real PyPI package. In v0.4.0 the
project was renamed to `torchgenomics`. To keep `pip install --upgrade
torchgwas` working for downstream users, we publish this thin shim that
depends on the renamed package. Anyone who pins `torchgwas` in their
own dependency lists keeps getting upgrades; once they're ready, they
swap their import.

For details see the [TorchGenomics CHANGELOG](https://github.com/sikiru-atanda/torchgenomics/blob/master/CHANGELOG.md#040--2026-06-03).
