import pandas as pd


def test_smrrun_shape_and_summary():
    from torchgenomics.api import SMRRun
    r = SMRRun(n_genes_tested=5, n_significant_smr=2, n_pass_heidi=1,
               results=pd.DataFrame({"gene_id": ["g1", "g2"], "p_smr": [1e-9, 0.2]}))
    assert r.n_genes_tested == 5 and r.n_significant_smr == 2
    s = r.summary()
    assert "SMR" in s and "5" in s
    assert r.to_dict()["n_pass_heidi"] == 1


def test_mrmegarun_shape_and_summary():
    from torchgenomics.api import MRMegaRun
    r = MRMegaRun(method="mr_mega", n_axes=1, n_populations=3, n_snps=40,
                  min_p_meta=1e-8,
                  results=pd.DataFrame({"p_meta": [1e-8, 0.3], "p_ancestry": [0.5, 0.6]}))
    s = r.summary()
    assert "MR-MEGA" in s and "3" in s and "40" in s
    assert r.to_dict()["n_populations"] == 3


def _write_smr_fixture(tmp_path, seed=0, n_snp=30, n_genes=3):
    """GWAS + eQTL sumstats sharing a cis causal SNP per gene + a gene-map TSV."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def _row(i):
        return {"chr": 1, "pos": i + 1, "snp": f"rs{i}", "a1": "A", "a2": "G", "n": 5000, "af": 0.3}

    gwas_rows, eqtl_rows, map_rows = [], [], []
    per = n_snp // n_genes
    for g in range(n_genes):
        causal = g * per + 1
        for j in range(per):
            i = g * per + j
            z_e = 6.0 if i == causal else rng.normal()
            z_g = 5.0 if i == causal else rng.normal()   # shared signal at the causal cis-SNP
            gwas_rows.append({**_row(i), "beta": z_g * 0.05, "se": 0.05, "p": 2 * (1 - 0.9999)})
            eqtl_rows.append({**_row(i), "beta": z_e * 0.05, "se": 0.05, "p": 2 * (1 - 0.9999)})
            map_rows.append({"gene": f"gene{g}", "snp": f"rs{i}"})
    import pandas as pd
    gwas = tmp_path / "gwas.tsv"; eqtl = tmp_path / "eqtl.tsv"; gmap = tmp_path / "map.tsv"
    pd.DataFrame(gwas_rows).to_csv(gwas, sep="\t", index=False)
    pd.DataFrame(eqtl_rows).to_csv(eqtl, sep="\t", index=False)
    pd.DataFrame(map_rows).to_csv(gmap, sep="\t", index=False)
    return str(gwas), str(eqtl), str(gmap), n_genes


def test_api_smr_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import SMRRun
    gwas, eqtl, gmap, n_genes = _write_smr_fixture(tmp_path)
    r = tg.smr(gwas, eqtl, gmap, output=str(tmp_path / "smr.tsv"))
    assert isinstance(r, SMRRun)
    assert r.n_genes_tested == n_genes
    assert "gene_id" in r.results.columns
    assert (tmp_path / "smr.tsv").exists()


def test_api_smr_bad_gene_map(tmp_path):
    import torchgenomics as tg, pandas as pd, pytest
    gwas, eqtl, _, _ = _write_smr_fixture(tmp_path)
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError) as e:
        tg.smr(gwas, eqtl, str(bad))
    assert "gene" in str(e.value) and "snp" in str(e.value)


def _write_mrmega_fixture(tmp_path, n_pops=4, n_snp=30, seed=1):
    """n_pops ancestry sumstats over the same SNPs with a shared causal effect."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    true = rng.normal(0.2, 0.05, size=n_snp)          # shared causal effects
    paths = []
    for p in range(n_pops):
        beta = true + rng.normal(0, 0.02, size=n_snp)  # ancestry-perturbed
        df = pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": [0.02] * n_snp,
            "p": [0.001] * n_snp, "n": [5000] * n_snp, "af": [0.3] * n_snp,
        })
        fp = tmp_path / f"pop{p}.tsv"; df.to_csv(fp, sep="\t", index=False)
        paths.append(str(fp))
    return paths


def test_api_mr_mega_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import MRMegaRun
    paths = _write_mrmega_fixture(tmp_path, n_pops=4)
    r = tg.mr_mega(paths, n_axes=1, output=str(tmp_path / "mrmega.tsv"))
    assert isinstance(r, MRMegaRun)
    assert r.n_populations == 4 and r.n_snps > 0
    assert "p_meta" in r.results.columns
    assert (tmp_path / "mrmega.tsv").exists()


def test_api_mr_mega_too_few_populations(tmp_path):
    import torchgenomics as tg, pytest
    paths = _write_mrmega_fixture(tmp_path, n_pops=2)
    with pytest.raises(ValueError) as e:
        tg.mr_mega(paths, n_axes=1)   # needs >= n_axes + 2 = 3
    assert "population" in str(e.value).lower() or "sumstats" in str(e.value).lower()
