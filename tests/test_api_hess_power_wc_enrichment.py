import pandas as pd


def test_result_classes_shape_and_summary():
    from torchgenomics.api import PowerRun, WinnersCurseRun, EnrichmentRun, HessRun
    p = PowerRun(alpha=5e-8, n=5000, target_power=0.8, n_variants=3, n_powered=1,
                 results=pd.DataFrame({"snp": ["a"], "power": [0.9]}))
    assert "Power" in p.summary() and p.to_dict()["n_powered"] == 1
    w = WinnersCurseRun(method="conditional_likelihood", n_corrected=2, n_variants=10,
                        results=pd.DataFrame({"snp": ["a"], "beta_adjusted": [0.1]}))
    assert "curse" in w.summary().lower() and w.to_dict()["n_corrected"] == 2
    e = EnrichmentRun(n_genes_total=100, n_gene_sets=5, n_significant=1,
                      results=pd.DataFrame({"gene_set_name": ["s1"], "p": [1e-3]}),
                      genes=pd.DataFrame({"gene_id": ["g1"], "p": [1e-4]}))
    assert "nrichment" in e.summary() and e.to_dict()["n_gene_sets"] == 5
    h = HessRun(mode="h2", h2_total=0.3, h2_total_se=0.05, n_regions=4, n_snps_total=400,
                results=pd.DataFrame({"region_id": ["chr1:1-2"], "h2_local": [0.1]}))
    assert "HESS" in h.summary() and h.to_dict()["n_regions"] == 4


def _write_power_fixture(tmp_path):
    import pandas as pd, numpy as np
    # variant 0: large effect + moderate AF (high power); variant 1: tiny effect (low power)
    df = pd.DataFrame({
        "chr": [1, 1], "pos": [10, 20], "snp": ["rsA", "rsB"],
        "a1": ["A", "A"], "a2": ["G", "G"],
        "beta": [0.5, 0.01], "se": [0.05, 0.05], "p": [1e-20, 0.8],
        "n": [10000, 10000], "af": [0.3, 0.3],
    })
    fp = tmp_path / "gwas.tsv"; df.to_csv(fp, sep="\t", index=False)
    return str(fp)


def test_api_power_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import PowerRun
    fp = _write_power_fixture(tmp_path)
    r = tg.power(fp, power_curve=True, output=str(tmp_path / "power.tsv"))
    assert isinstance(r, PowerRun)
    assert r.n_variants == 2
    res = r.results.set_index("snp")
    # high-power variant: power near 1 and SMALLER required_n than the low-power one
    assert res.loc["rsA", "power"] > 0.9
    assert res.loc["rsA", "required_n"] < res.loc["rsB", "required_n"]
    assert r.curve is not None and len(r.curve) >= 2
    assert (tmp_path / "power.tsv").exists()


def test_api_power_needs_af(tmp_path):
    import torchgenomics as tg, pandas as pd, pytest
    df = pd.DataFrame({"chr": [1], "pos": [1], "snp": ["a"], "a1": ["A"], "a2": ["G"],
                       "beta": [0.1], "se": [0.05], "p": [0.1], "n": [1000]})
    fp = tmp_path / "no_af.tsv"; df.to_csv(fp, sep="\t", index=False)
    with pytest.raises(ValueError):
        tg.power(str(fp))


def test_api_winners_curse_all_methods(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import WinnersCurseRun
    import pandas as pd
    # a strong hit (should be shrunk) + noise
    df = pd.DataFrame({
        "chr": [1] * 5, "pos": list(range(5)), "snp": [f"rs{i}" for i in range(5)],
        "a1": ["A"] * 5, "a2": ["G"] * 5,
        "beta": [0.285, 0.02, -0.01, 0.03, 0.55], "se": [0.05] * 5,
        "p": [1e-20, 0.7, 0.8, 0.6, 1e-18], "n": [10000] * 5, "af": [0.3] * 5,
    })
    fp = tmp_path / "gwas.tsv"; df.to_csv(fp, sep="\t", index=False)
    for method in ("conditional_likelihood", "fiqt", "bootstrap"):
        kw = {"n_boot": 200, "seed": 1} if method == "bootstrap" else {}
        r = tg.winners_curse(str(fp), method=method, **kw)
        assert isinstance(r, WinnersCurseRun) and r.n_variants == 5
        res = r.results.set_index("snp")
        # rs0 is marginally significant (z≈5.7, just past the alpha=5e-8 threshold),
        # so the winner's-curse bias is real and large relative to bootstrap MC noise —
        # all three methods must shrink |beta| strictly toward zero.
        assert abs(res.loc["rs0", "beta_adjusted"]) < abs(res.loc["rs0", "beta_original"])
