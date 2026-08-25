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


def _write_mr_fixture(tmp_path, seed=0, effect=0.5, n_snp=40):
    """Two-sample MR fixture: instruments with a true causal effect + noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bx = rng.normal(0.3, 0.1, size=n_snp)          # instrument-exposure
    bx_se = np.full(n_snp, 0.02)
    by = effect * bx + rng.normal(0, 0.01, size=n_snp)  # instrument-outcome
    by_se = np.full(n_snp, 0.02)
    def _df(beta, se):
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": se,
            "p": 2 * (1 - 0.9999), "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })
    exp = tmp_path / "exp.tsv"; out = tmp_path / "out.tsv"
    _df(bx, bx_se).to_csv(exp, sep="\t", index=False)
    _df(by, by_se).to_csv(out, sep="\t", index=False)
    return str(exp), str(out), effect


def test_api_mr_all_recovers_effect(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import MRRun
    exp, out, effect = _write_mr_fixture(tmp_path)
    r = tg.mr(exp, out, method="all", output=str(tmp_path / "mr.tsv"))
    assert isinstance(r, MRRun)
    methods = set(r.results["method"])
    assert {"ivw", "egger", "weighted_median", "presso"}.issubset(methods)
    ivw_beta = float(r.results.loc[r.results["method"] == "ivw", "beta"].iloc[0])
    assert abs(ivw_beta - effect) < 0.15          # recovers the simulated effect
    assert (tmp_path / "mr.tsv").exists()


def test_api_mr_single_method_and_bad_method(tmp_path):
    import torchgenomics as tg, pytest
    exp, out, _ = _write_mr_fixture(tmp_path)
    r = tg.mr(exp, out, method="ivw")
    assert r.method == "ivw" and len(r.results) == 1
    with pytest.raises(ValueError) as e:
        tg.mr(exp, out, method="nope")
    assert "nope" in str(e.value) and "ivw" in str(e.value)
