# Installation

TorchGenomics supports Python 3.10, 3.11, and 3.12 on Linux, Windows, and macOS.
PyTorch is the only heavy dependency; everything else is standard scientific
Python.

## From PyPI (recommended)

```bash
pip install torchgenomics
```

Pip uses a pre-built wheel where one is available and falls back to the source
distribution otherwise. Source installs compile the 24 native C++ extension
modules if a compiler is available. If compilation fails, the pure-Python /
torch fallbacks still work — every accelerator is optional.

## With optional dependencies

| Extra      | What it adds                                            |
|------------|---------------------------------------------------------|
| `[all]`    | `zarr`, `h5py`, `pyarrow`, `seaborn`                    |
| `[dev]`    | `all` + `pytest`, `pytest-cov`, `ruff`, `mypy`          |
| `[docs]`   | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`     |
| `[polyploid-phase]` | `juliacall` for Julia-backed polyploid phasing |

```bash
pip install "torchgenomics[all]"
pip install "torchgenomics[dev]"
pip install "torchgenomics[polyploid-phase]"
```

## From source

```bash
git clone https://github.com/sikiru-atanda/torchgenomics.git
cd torchgenomics
pip install -e ".[dev]"
```

Editable installs are the path of choice for contributors. Tests run with:

```bash
LC_ALL=C.UTF-8 LANG=C.UTF-8 TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/ -v --tb=short -x -q --timeout=300
LC_ALL=C.UTF-8 LANG=C.UTF-8 TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/ -m golden -v --tb=short --timeout=600
```

See [Validation](../validation.md) for the supported local validation tiers.

## GPU support

TorchGenomics has CUDA-qualified tensor paths for core PyTorch operations and
selected GPU imputation kernels. Unsupported native kernels fall back through
the package dispatch layer. To use a GPU, install PyTorch with the matching CUDA
toolchain **before** installing TorchGenomics:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torchgenomics
```

See the [PyTorch install matrix](https://pytorch.org/get-started/locally/) for
the exact wheel URL for your CUDA version. The native C++ extensions are linked
against CPU PyTorch but run correctly against GPU PyTorch at runtime. GPU
coverage is validated by tests marked `gpu` when CUDA is available:

```bash
pytest tests/ -m gpu -v --tb=short
```

## Polyploid phasing

The `polyploid-phase` extra installs `juliacall`. End-to-end PolyOrigin-backed
phasing additionally requires Julia 1.10 or newer and the Julia packages
declared in `torchgenomics/preprocess/juliapkg.json`. Tests that require the Julia
bridge are skipped automatically when those external tools are unavailable.

## Windows users

Pre-built `win_amd64` wheels are built for supported CPython releases. If pip
falls back to the sdist, it compiles on Windows with Visual Studio Build Tools
installed. Without a C++ toolchain, the pure-Python path still works — just with
reduced performance on a handful of hot loops.

## Verifying your install

```bash
python -c "import torchgenomics; print(torchgenomics.__version__)"
```

Should print `0.2.0`.

```bash
torchgenomics --help
```

Should list all 40 subcommands.

## Disabling native accelerators

If you hit a compile issue or want to pin to the pure-Python path for debugging:

```bash
export TORCHGENOMICS_DISABLE_NATIVE=1
# or to disable just GPU kernels
export TORCHGENOMICS_DISABLE_GPU=1
# or to disable OpenMP
export TORCHGENOMICS_DISABLE_OPENMP=1
```

Every function falls back transparently to its torch / Python body.
