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
