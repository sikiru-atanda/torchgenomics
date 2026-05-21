# 11. Availability

**License.** Apache 2.0 (subject to confirmation; flagged in `paper/manuscript/metadata.yaml` as `Apache 2.0 (subject to confirmation)` pending final author sign-off).

**Installation.** `pip install torchgwas` from PyPI. Optional GPU + native (pybind11/C++) build via `pip install torchgwas[gpu,native]`. A Bioconda recipe is planned but not yet published; the `conda install -c bioconda torchgwas` channel will be wired in only after the recipe is accepted upstream.

**Source code.** `https://github.com/sikiru-atanda/torchgwas` (placeholder URL — to be confirmed with corresponding author before submission). Master branch is the canonical release branch; every phase ships as a `Phase N: ...` commit whose body is the authoritative changelog.

**Documentation.** mkdocs-material site deployed from `master`, with full public API reference, end-to-end tutorials, CLI reference (38 subcommands), and the validation-findings ledger (`docs/validation_findings.md`) all publicly browsable.

**Reproducibility.** From a clean checkout, `bash paper/reproducibility/reproduce_paper.sh` regenerates every number in every figure and table; outputs land at `paper/reproducibility/output/F{1..7}.pdf` together with `paper/reproducibility/manifest.json` (every numeric result + the commit SHA the run was based on). Each harness ships an idempotent `install.sh` + `fetch_data.sh` + `run.sh` + `compare.py`; SHA256 manifests pin each fixture. Companion reproducibility repository: `https://github.com/sikiru-atanda/torchgwas-paper-reproducibility` (placeholder URL — to be confirmed).

**Validation-findings ledger.** `docs/validation_findings.md` carries the full F2 finding + 4 F3 divergences with classification and proposed fixes.

**Contact.** Corresponding author: Sikiru Atanda, sikiruandfriends@gmail.com.

