import pandas as pd


def test_mrrun_shape_and_summary():
    from torchgenomics.api import MRRun
    r = MRRun(method="all", n_instruments=25, primary_beta=0.42, primary_p=1e-6,
              results=pd.DataFrame({"method": ["ivw", "egger"], "beta": [0.42, 0.40],
                                    "pval": [1e-6, 2e-3]}))
    assert r.primary_beta == 0.42 and r.n_instruments == 25
    s = r.summary()
    assert "MR" in s and "ivw" in s.lower()
    d = r.to_dict()
    assert d["method"] == "all"


def test_colocrun_shape_and_summary():
    from torchgenomics.api import ColocRun
    r = ColocRun(method="pairwise", pp={"h0": 0.01, "h1": 0.02, "h2": 0.02,
                                        "h3": 0.03, "h4": 0.92},
                 candidate_snp=137, headline=0.92)
    s = r.summary()
    assert "0.92" in s and ("H4" in s or "h4" in s or "shared" in s.lower())
    assert r.headline == 0.92
