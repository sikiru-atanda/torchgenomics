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
