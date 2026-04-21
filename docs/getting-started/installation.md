# Installation

TorchGWAS supports Python 3.10, 3.11, and 3.12 on Linux, Windows, and macOS.
PyTorch is the only heavy dependency; everything else is standard scientific
Python.

## From PyPI (recommended)

```bash
pip install torchgwas
```

This pulls the **source distribution** and compiles the 24 native C++ extensions
if a compiler is available. If compilation fails, the pure-Python / torch fall-
backs still work — every accelerator is optional.

## With optional dependencies

| Extra      | What it adds                                            |
|------------|---------------------------------------------------------|
| `[all]`    | `zarr`, `h5py`, `pyarrow`, `seaborn`                    |
| `[dev]`    | `all` + `pytest`, `pytest-cov`, `ruff`, `mypy`          |
| `[docs]`   | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`     |

```bash
pip install "torchgwas[all]"
pip install "torchgwas[dev]"
```

## From source

```bash
git clone https://github.com/sikiru-atanda/torchgwas.git
cd torchgwas
pip install -e ".[dev]"
```

Editable installs are the path of choice for contributors. Tests run with:

```bash
pytest tests/ -v
```

## GPU support

TorchGWAS is device-agnostic — every tensor operation works on CPU or CUDA.
To use a GPU, install PyTorch with the matching CUDA toolchain **before** in-
stalling TorchGWAS:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchgwas
```

See the [PyTorch install matrix](https://pytorch.org/get-started/locally/) for
the exact wheel URL for your CUDA version. The native C++ extensions are linked
against CPU PyTorch but run correctly against GPU PyTorch at runtime — GPU
kernels for imputation dispatch automatically via
`torchgwas._dispatch.select_path`.

## Windows users

The sdist compiles on Windows with Visual Studio Build Tools installed. If you
don't have a C++ toolchain, the pure-Python path still works — just with
reduced performance on a handful of hot loops. A pre-built Windows wheel is
planned for v0.3.0 (see the [roadmap](https://github.com/sikiru-atanda/torchgwas/blob/master/docs/ROADMAP.md)).

## Verifying your install

```bash
python -c "import torchgwas; print(torchgwas.__version__)"
```

Should print `0.1.1`.

```bash
torchgwas --help
```

Should list all 35 subcommands.

## Disabling native accelerators

If you hit a compile issue or want to pin to the pure-Python path for debugging:

```bash
export TORCHGWAS_DISABLE_NATIVE=1
# or to disable just GPU kernels
export TORCHGWAS_DISABLE_GPU=1
# or to disable OpenMP
export TORCHGWAS_DISABLE_OPENMP=1
```

Every function falls back transparently to its torch / Python body.
