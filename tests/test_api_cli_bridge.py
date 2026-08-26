import pytest


def _tiny_bed_fixture():
    """Reuse the repo's tiny PLINK fixture for a real CLI subcommand run."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    bed = root / "tests" / "fixtures" / "tiny.bed"
    pheno = root / "tests" / "fixtures" / "tiny_pheno.txt"
    return str(bed), str(pheno)


def test_run_cli_subcommand_validate(tmp_path):
    from torchgenomics.api import CliRun, run_cli_subcommand
    bed, pheno = _tiny_bed_fixture()
    r = run_cli_subcommand("validate", genotype=bed, phenotype=pheno)
    assert isinstance(r, CliRun) and r.command == "validate" and r.exit_code == 0


def test_api_getattr_exposes_tier2_and_preserves_handcrafted():
    import torchgenomics.api as a
    # a tier-2 CLI subcommand with no hand-crafted api fn is now reachable
    assert callable(a.bayes_scan)
    assert hasattr(a, "bayes_scan")
    # hand-crafted api fns still resolve to the real function, not the CLI runner
    from torchgenomics.api._cli_bridge import run_cli_subcommand
    assert a.gwas is not run_cli_subcommand
    assert getattr(a.gwas, "__name__", "") == "gwas"
    # a genuinely unknown name still raises AttributeError
    with pytest.raises(AttributeError):
        _ = a.definitely_not_a_command


def test_run_cli_subcommand_friendly_error_on_failure():
    from torchgenomics.api import run_cli_subcommand
    with pytest.raises(RuntimeError) as e:
        run_cli_subcommand("lmm-scan", genotype="/no/such.bed", phenotype="/no/such.txt")
    assert "lmm-scan" in str(e.value)   # friendly, names the subcommand; not a SystemExit


def test_api_tier2_wrapper_runs_end_to_end(tmp_path):
    """The api CLI-runner reachable via __getattr__ actually runs a tier-2
    scan to completion and reports the output file in the CliRun.

    Uses a single-trait phenotype (the shipped fixture has two traits Y1/Y2;
    single-trait scan models need one) and farmcpu-scan, which writes the
    standard ``<output>.assoc.tsv`` the CliRun output-file collector recognizes.
    """
    import pandas as pd
    import torchgenomics.api as a
    from torchgenomics.api import CliRun

    bed, pheno = _tiny_bed_fixture()
    ph = pd.read_csv(pheno, sep="\t")
    single = tmp_path / "pheno1.txt"
    ph[[ph.columns[0], "Y1"]].to_csv(single, sep="\t", index=False)

    out = tmp_path / "farmcpu_out"
    r = a.farmcpu_scan(genotype=bed, phenotype=str(single), output=str(out))
    assert isinstance(r, CliRun)
    assert r.command == "farmcpu-scan" and r.exit_code == 0
    assert "tsv" in r.output_files          # <output>.assoc.tsv captured
