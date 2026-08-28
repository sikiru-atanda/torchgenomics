import pandas as pd


def test_finemaprun_shape_and_summary():
    from torchgenomics.api import FineMapRun
    r = FineMapRun(
        n_variants=5, n_credible_sets=1, n_variants_in_credible_sets=2,
        results=pd.DataFrame({"SNP": ["rs0"], "PIP": [0.9], "CREDIBLE_SET": [1]}),
        credible_sets=pd.DataFrame({"credible_set": [1], "n_snps": [2],
                                    "lead_snp": ["rs0"], "lead_pip": [0.9],
                                    "sum_pip": [1.1]}),
    )
    s = r.summary()
    assert "Fine-mapping" in s and "SuSiE-RSS" in s
    d = r.to_dict()
    assert d["n_credible_sets"] == 1 and d["n_variants_in_credible_sets"] == 2


def _write_finemap_fixture(tmp_path, p=5):
    """Sumstats TSV (uppercase schema) + a .pt LD reference carrying SNP ids."""
    import torch
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata
    sumstats_path = tmp_path / "sumstats.tsv"
    sumstats = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(p)], "CHR": [22] * p,
        "BP": [1000 + i * 10 for i in range(p)], "A1": ["A"] * p, "A2": ["G"] * p,
        "BETA": [0.05, 0.03, 0.6, 0.02, 0.04][:p], "SE": [0.05, 0.04, 0.05, 0.04, 0.05][:p],
        "N": [1000] * p,
    })
    sumstats.to_csv(sumstats_path, sep="\t", index=False)
    ld_path = tmp_path / "ld.pt"
    R = torch.eye(p, dtype=torch.float64)
    meta = LDReferenceMetadata(cohort_id="test", n=1000, build="GRCh38",
                               panel_provenance="synthetic")
    save_ld_reference(ld_path, R, sumstats["SNP"].tolist(), meta)
    return str(sumstats_path), str(ld_path)


def test_api_finemap_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import FineMapRun
    ss, ld = _write_finemap_fixture(tmp_path)
    out = tmp_path / "fm.tsv"
    r = tg.finemap(ss, ld, max_num_causal=2, threads=1, output=str(out))
    assert isinstance(r, FineMapRun)
    assert r.n_variants == 5
    assert list(r.results.columns) == [
        "SNP", "CHR", "BP", "A1", "A2", "Z", "N", "PIP", "BETA_MEAN", "BETA_SD",
        "CREDIBLE_SET"]
    assert list(r.credible_sets.columns) == [
        "credible_set", "n_snps", "lead_snp", "lead_pip", "sum_pip"]
    assert r.n_variants_in_credible_sets == int((r.results["CREDIBLE_SET"] > 0).sum())
    assert out.exists()


def test_api_finemap_no_output_temp(tmp_path):
    import torchgenomics as tg
    ss, ld = _write_finemap_fixture(tmp_path)
    r = tg.finemap(ss, ld, max_num_causal=2, threads=1)  # output=None
    assert r.n_variants == 5 and not r.results.empty
    assert r.output_files == {}  # temp dir cleaned; nothing persisted


def test_api_finemap_needs_ld(tmp_path):
    import torchgenomics as tg, pytest
    ss, _ = _write_finemap_fixture(tmp_path)
    with pytest.raises(ValueError):
        tg.finemap(ss)  # no ld_ref, no geno
