"""Tests for the friendly-API GWAS result object (`GwasResult` = `ScanRun`).

Task 3 of the ``tg.gwas(...)`` friendly-API plan: enriches `ScanRun` with
`.trait_type`, `.hits`, `.diagnostics`, a plain-language `.summary()`, and
`.report(dir)`.
"""
import pandas as pd

from torchgenomics.api import GwasResult


def test_gwasresult_shape_and_summary_and_report(tmp_path):
    r = GwasResult(model="SingleTraitLMM", test="wald", correction="bh",
                   n_variants=1000, n_significant=2, lambda_gc=0.99, n_samples=300,
                   top_hits=pd.DataFrame({"SNP":["s1"],"CHR":[1],"BP":[10],"P":[1e-9]}))
    r.trait_type = "continuous"
    assert isinstance(r.hits, pd.DataFrame) and not r.hits.empty     # .hits alias
    s = r.summary()
    assert "continuous" in s and "λ_GC" in s and "0.99" in s
    assert "well-calibrated" in s.lower() or "calibrat" in s.lower() # plain-language λ interpretation
    assert "Next" in s                                              # next-step suggestions
    d = r.diagnostics
    assert d["lambda_gc"] == 0.99 and d["n_significant"] == 2
    # report writes a folder (summary.txt at minimum; plots best-effort)
    out = r.report(tmp_path / "rep")
    assert (tmp_path / "rep" / "summary.txt").exists()


def test_run_model_lmm_and_a_lowlevel_model(tmp_path):
    # reuse an existing tiny GWAS fixture if the repo has one; else synth a small BED
    from torchgenomics.api._inputs import load_inputs
    from torchgenomics.api._dispatch import run_model, RunOptions
    import numpy as np, pandas as pd
    ids=[f"s{i}" for i in range(150)]
    # Hardy-Weinberg-consistent genotypes: draw a per-SNP minor-allele
    # frequency in [0.2, 0.8] (clears the MAF>=0.01 QC filter with margin),
    # then genotypes ~ Binomial(2, p) per sample — HWE-consistent by
    # construction, so all 300 variants survive standard QC (MAF +
    # missingness + HWE, which is ON by default in tg.gwas; see Fix 1).
    # A uniform-random 0/1/2 matrix is NOT HWE-consistent and would fail the
    # HWE filter on ~1/5 of variants, making n_variants==300 a flaky assertion.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, size=300)
    G = rng.binomial(2, p, size=(150, 300)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=150), index=ids, name="y")
    inp=load_inputs(phenotype=y, genotype=G)
    r=run_model("lmm", inp, RunOptions(kinship="auto", pcs=0, correction="bh", verbose=False))
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult) and r.n_variants==300
    # a low-level model also returns the same GwasResult shape
    r2=run_model("blink", inp, RunOptions(kinship=False, pcs=0, verbose=False))
    assert isinstance(r2, GwasResult) and r2.n_variants>0


def test_tg_gwas_single_auto_and_multimodel(capsys):
    import numpy as np, pandas as pd
    import torchgenomics as tg
    ids=[f"s{i}" for i in range(150)]
    G=np.random.default_rng(0).integers(0,3,(150,300)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=150), index=ids, name="yield")
    # bare call: auto model, prints decisions, returns one GwasResult
    r=tg.gwas(y, G, kinship="auto", pcs=0)
    assert isinstance(r, tg.GwasResult)
    out=capsys.readouterr().out
    assert "yield" in out and ("lmm" in out.lower() or "SingleTraitLMM" in out)  # decision log
    # user picks multiple models -> GwasComparison
    cmp=tg.gwas(y, G, models=["lmm","blink"], kinship="auto", pcs=0, verbose=False)
    assert isinstance(cmp, tg.GwasComparison)
    assert set(cmp.results) == {"lmm","blink"}
    assert "lmm" in cmp.summary() and "blink" in cmp.summary()
    # tg.models() lists the registry
    assert "lmm" in set(tg.models()["alias"])


def _hwe_consistent_fixture(n_samples=150, n_variants=300, seed=0):
    """Shared HWE-consistent (float) genotype + continuous phenotype fixture.

    Mirrors the fixture used by ``test_run_model_lmm_and_a_lowlevel_model``:
    per-SNP MAF drawn in [0.2, 0.8], genotypes ~ Binomial(2, p) — HWE-
    consistent by construction so all variants survive standard QC
    (MAF + missingness + HWE, ON by default). See that test for the
    rationale on why a uniform-random 0/1/2 matrix would be flaky here.
    """
    import numpy as np, pandas as pd
    ids = [f"s{i}" for i in range(n_samples)]
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, size=n_variants)
    G = rng.binomial(2, p, size=(n_samples, n_variants)).astype(float)
    y = pd.Series(np.random.default_rng(seed + 1).normal(size=n_samples), index=ids, name="y")
    return y, G


def test_tg_gwas_list_vs_str_return_type():
    """Fix 2 regression: models= return type is decided by INPUT TYPE, not count.

    A one-element list (``models=["lmm"]``) must return a GwasComparison
    (the caller explicitly asked for the comparison shape), while a bare
    str (``models="lmm"``) must return a plain GwasResult -- even though
    both resolve to exactly one alias.
    """
    import torchgenomics as tg
    from torchgenomics.api.gwas import GwasComparison

    y, G = _hwe_consistent_fixture()

    cmp = tg.gwas(y, G, models=["lmm"], kinship="auto", pcs=0, verbose=False)
    assert type(cmp) is GwasComparison
    assert set(cmp.results) == {"lmm"}

    r = tg.gwas(y, G, models="lmm", kinship="auto", pcs=0, verbose=False)
    assert type(r) is tg.GwasResult


def test_tg_gwas_comparison_runtime_s_populated():
    """Fix 3 regression: GwasComparison.runtime_s is the sum of per-model runtimes."""
    import torchgenomics as tg

    y, G = _hwe_consistent_fixture()
    cmp = tg.gwas(y, G, models=["lmm", "blink"], kinship="auto", pcs=0, verbose=False)
    assert cmp.runtime_s == sum(r.runtime_s for r in cmp.results.values())
    assert cmp.runtime_s > 0
    assert f"Runtime: {cmp.runtime_s:.1f}s" in cmp.summary()


def test_auto_model_never_picks_unwired_glmm_for_binary_trait():
    """Fix 1 regression: auto-selection must never pick an unwired model.

    A binary trait with default kinship="auto" used to resolve to "glmm",
    which run_model does not wire (NotImplementedError) -- crashing the
    most common non-continuous default path. It must now resolve to the
    wired "glm" alias, with a rationale that still mentions glmm as the
    scientifically-preferable (but not-yet-wired) alternative.
    """
    import numpy as np, pandas as pd
    from torchgenomics.api._inputs import load_inputs
    from torchgenomics.api.gwas import _auto_model

    ids = [f"s{i}" for i in range(150)]
    y = pd.Series(np.random.default_rng(2).integers(0, 2, size=150), index=ids, name="case")
    _, G = _hwe_consistent_fixture(seed=3)
    inputs = load_inputs(phenotype=y, genotype=G)
    assert inputs.trait_type == "binary"

    alias, rationale = _auto_model(inputs, kinship="auto")
    assert alias == "glm"
    assert "glmm" in rationale.lower()
