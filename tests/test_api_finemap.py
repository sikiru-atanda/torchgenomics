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
