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


def test_api_mr_all_na_columns_are_nan(tmp_path):
    import math
    import torchgenomics as tg
    exp, out, _ = _write_mr_fixture(tmp_path)
    r = tg.mr(exp, out, method="all")
    df = r.results.set_index("method")

    egger_row = df.loc["egger"]
    assert not pd.isna(egger_row["egger_intercept_p"])
    assert not pd.isna(egger_row["egger_intercept"])
    for m in ("ivw", "weighted_median"):
        row = df.loc[m]
        assert math.isnan(row["egger_intercept"])
        assert math.isnan(row["egger_intercept_p"])

    presso_row = df.loc["presso"]
    assert not pd.isna(presso_row["beta_corrected"])
    assert not pd.isna(presso_row["pval_corrected"])
    for m in ("ivw", "egger", "weighted_median"):
        row = df.loc[m]
        assert math.isnan(row["beta_corrected"])
        assert math.isnan(row["pval_corrected"])
        assert math.isnan(row["n_outliers"])


def _write_coloc_fixture(tmp_path, shared=True, seed=1, n_snp=50):
    """Two sumstats over the same region; a shared causal SNP if shared=True."""
    import numpy as np
    rng = np.random.default_rng(seed)
    causal = 25
    def _df(offset):
        z = rng.normal(0, 1, size=n_snp)
        idx = causal if shared else (causal + offset)
        z[idx] = 6.0                     # strong signal at the (shared?) SNP
        beta = z * 0.05
        se = np.full(n_snp, 0.05)
        from scipy.stats import norm
        p = 2 * norm.sf(np.abs(z))
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": se, "p": p,
            "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })
    a = tmp_path / "a.tsv"; b = tmp_path / "b.tsv"
    _df(0).to_csv(a, sep="\t", index=False)
    _df(10).to_csv(b, sep="\t", index=False)
    return str(a), str(b)


def test_api_coloc_pairwise_shared_signal(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import ColocRun
    a, b = _write_coloc_fixture(tmp_path, shared=True)
    r = tg.coloc(a, b, method="pairwise", output=str(tmp_path / "coloc.tsv"))
    assert isinstance(r, ColocRun) and r.method == "pairwise"
    assert r.headline == r.pp["h4"]
    assert r.pp["h4"] > 0.5              # shared causal -> high PP.H4
    assert (tmp_path / "coloc.tsv").exists()


def test_api_coloc_pairwise_requires_second(tmp_path):
    import torchgenomics as tg, pytest
    a, _ = _write_coloc_fixture(tmp_path)
    with pytest.raises(ValueError) as e:
        tg.coloc(a, method="pairwise")
    assert "sumstats2" in str(e.value) or "second" in str(e.value).lower()


def test_api_coloc_hyprcoloc_three_traits(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import ColocRun
    a, b = _write_coloc_fixture(tmp_path, shared=True, seed=2)
    c, _ = _write_coloc_fixture(tmp_path, shared=True, seed=3)
    r = tg.coloc([a, b, c], method="hyprcoloc")
    assert isinstance(r, ColocRun) and r.method == "hyprcoloc"
    assert r.headline is not None

    import pytest
    with pytest.raises(ValueError):
        tg.coloc([a], method="hyprcoloc")   # <2 traits
