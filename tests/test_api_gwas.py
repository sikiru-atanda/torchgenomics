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
    # tg.list_models() lists the registry
    assert "lmm" in set(tg.list_models()["alias"])


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
    """Fix 1 regression: auto-selection must prefer the lighter wired model.

    A binary trait with default kinship="auto" resolves to "glm", not the
    (now wired, but heavier -- GRM + PQL null fit) "glmm": auto-selection
    is a documented convenience for the common case, not a statistical
    recommendation engine, so the zero-argument default stays the cheap
    option. The rationale string still names glmm as the scientifically
    more appropriate, and directly usable (models="glmm"), alternative for
    related samples.
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


def test_recommendation_to_dict_is_json_safe_and_matches_explain():
    """Task 9 (R wrappers): Recommendation.to_dict() is additive and must
    round-trip every field `recommend()` sets, plus a "plan" key holding
    the same text as `.explain()` -- this is what `bridge_call` on the R
    side actually serializes (it auto-detects and calls `to_dict()`)."""
    import json
    import numpy as np, pandas as pd, torchgenomics as tg
    ids = [f"s{i}" for i in range(120)]
    y = pd.Series(np.random.default_rng(0).normal(size=120), index=ids, name="y")
    G = np.random.default_rng(0).integers(0, 3, (120, 200)).astype(float)
    rec = tg.recommend(y, G)

    d = rec.to_dict()
    assert d["trait_name"] == rec.trait_name
    assert d["trait_type"] == rec.trait_type == "continuous"
    assert d["n_samples"] == rec.n_samples
    assert d["suggested_model"] == rec.suggested_model
    assert d["rationale"] == rec.rationale
    assert d["n_pcs"] == rec.n_pcs
    assert d["qc_summary"] == rec.qc_summary
    assert d["qc_summary"] is not rec.qc_summary  # copy, not the same mutable dict
    assert d["plan"] == rec.explain()

    # JSON-safe: every value round-trips through json.dumps/loads unchanged.
    json.loads(json.dumps(d))


def test_recommend_dry_run_and_warnings():
    import numpy as np, pandas as pd, torchgenomics as tg
    ids=[f"s{i}" for i in range(120)]
    yb=pd.Series(np.r_[np.ones(5), np.zeros(115)], index=ids, name="disease")  # imbalanced binary
    G=np.random.default_rng(0).integers(0,3,(120,200)).astype(float)
    rec=tg.recommend(yb, G)
    assert rec.trait_type=="binary"
    assert rec.suggested_model in {"glmm","glm"}
    assert "binary" in rec.explain().lower()          # human-readable plan, nothing run
    # a run on the imbalanced binary emits a case/control-imbalance warning
    r=tg.gwas(yb, G, models="glm", kinship=False, pcs=0, verbose=False)
    warns=" ".join(r.diagnostics.get("warnings", []))
    assert "imbalance" in warns.lower() or "case" in warns.lower()


def test_existing_scans_return_gwasresult():
    """Task 7 consistency audit: every friendly-API scan path (here, the
    ``api_lmm`` route reached via ``tg.gwas(models="lmm")``) returns the
    enriched ``GwasResult`` with ``.hits`` / ``.diagnostics`` / callable
    ``.summary()`` -- not a divergent object. See
    ``test_run_model_lmm_and_a_lowlevel_model`` above for the equivalent
    check on the low-level ``blink`` route via ``run_model`` directly.
    """
    import numpy as np, pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    # lmm_scan / glm_scan return the enriched GwasResult with .hits/.summary/.diagnostics
    ids=[f"s{i}" for i in range(120)]
    G=np.random.default_rng(0).integers(0,3,(120,200)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=120), index=ids, name="y")
    r=tg.gwas(y, G, models="lmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult)
    assert hasattr(r,"hits") and hasattr(r,"diagnostics") and callable(r.summary)


def test_api_lmm_n_samples_backfilled():
    """Regression test: GwasResult.n_samples is backfilled from inputs on api lmm route.

    The friendly-API lmm route reads n_samples from DataFrame.attrs, which doesn't
    survive the CSV/Parquet round-trip in lmm_scan. This test verifies the backfill
    restores it so that .summary() and .diagnostics["n_samples"] report the true count.
    """
    import numpy as np, pandas as pd
    import torchgenomics as tg
    from torchgenomics.api import GwasResult

    n_samples, n_variants = 120, 200
    ids = [f"s{i}" for i in range(n_samples)]
    # HWE-consistent fixture: per-SNP MAF in [0.2, 0.8], Binomial(2, p).
    rng = np.random.default_rng(42)
    p = rng.uniform(0.2, 0.8, size=n_variants)
    G = rng.binomial(2, p, size=(n_samples, n_variants)).astype(float)
    y = pd.Series(np.random.default_rng(43).normal(size=n_samples), index=ids, name="y")

    r = tg.gwas(y, G, models="lmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult)
    assert r.n_samples == n_samples, f"Expected n_samples={n_samples}, got {r.n_samples}"
    assert r.diagnostics["n_samples"] == n_samples


def test_toplevel_exports():
    """Task 10 finalization test: the five friendly-API names are exported
    at the top level (``import torchgenomics as tg; tg.<name>``), not just
    reachable via ``torchgenomics.api``.

    This is the last item of the 10-task plan; Tasks 5/6 already added
    ``gwas``/``recommend``/``list_models``/``GwasResult``/``GwasComparison``
    to ``torchgenomics/__init__.py``'s imports and ``__all__``, so this test
    is expected to PASS immediately -- it exists to make that invariant
    explicit and regression-tested going forward, not to drive new
    implementation.

    Note: the model-listing function is ``list_models``, not ``models`` --
    see ``test_list_models_does_not_shadow_models_subpackage`` below for why
    (the final-review fix that renamed it).
    """
    import torchgenomics as tg

    for name in ("gwas", "recommend", "list_models", "GwasResult", "GwasComparison"):
        assert hasattr(tg, name), name


def test_list_models_does_not_shadow_models_subpackage():
    """Final-review regression test (Fix 1): ``tg.list_models`` must not
    occupy the ``torchgenomics.models`` top-level attribute.

    Before this fix, the friendly-API listing function was named
    ``tg.models()`` and was re-exported at ``torchgenomics.models``,
    shadowing the :mod:`torchgenomics.models` subpackage's top-level
    attribute -- so ``import torchgenomics.models; torchgenomics.models.SingleTraitLMM``
    raised ``AttributeError`` (the subpackage import itself still worked and
    populated ``sys.modules["torchgenomics.models"]``, but the *attribute*
    ``torchgenomics.models`` on the already-imported ``torchgenomics``
    package object pointed at the shadowing function, not the module).
    Renaming the function to ``list_models`` frees the ``models`` name so it
    resolves to the subpackage again.

    This test asserts BOTH halves of the fix in one place: the subpackage
    attribute access works, AND the renamed listing function is callable
    and returns the expected registry.
    """
    import torchgenomics
    import torchgenomics as tg

    # The subpackage resolves via attribute access on the already-imported
    # `torchgenomics` package (not just via `sys.modules` after a submodule
    # import) -- this is exactly the access pattern that broke before Fix 1.
    assert torchgenomics.models.SingleTraitLMM.__name__ == "SingleTraitLMM"

    # The friendly-API function moved to `list_models` and still works.
    assert callable(tg.list_models)
    assert "lmm" in set(tg.list_models()["alias"])


def test_run_model_temp_workdir_cleaned_up_on_gc():
    """Final-review regression test (Fix 2): `run_model`'s temp working
    directory must not leak.

    Before this fix, `run_model` created `tempfile.mkdtemp(prefix="tg_gwas_")`
    for every call and never removed it -- one leaked directory (plus a
    materialized genotype.csv for in-memory input) per call, unbounded in
    the documented "loop over many traits" pattern (see `RunOptions.output`
    / `gwas()`'s docstring). The fix attaches a `weakref.finalize` callback
    to the returned `GwasResult` that removes the directory once the result
    is garbage-collected.

    This test runs two sequential `tg.gwas` calls (in-memory genotype, so a
    temp genotype.csv is actually materialized each time), captures the
    first result's temp workdir from its `output_files` before dropping the
    reference, forces a GC pass, and asserts the directory is gone -- while
    the second (still-referenced) result's own workdir is untouched.
    """
    import gc

    import torchgenomics as tg

    y1, G1 = _hwe_consistent_fixture(seed=10)
    r1 = tg.gwas(y1, G1, models="lmm", kinship="auto", pcs=0, verbose=False)
    tsv1 = r1.output_files["tsv"]
    # output_files["tsv"] == <workdir>/lmm_out/results.assoc.tsv (the "lmm_out"
    # output prefix from torchgenomics.api._dispatch._run_api_lmm is used as a
    # directory by the underlying lmm-scan CLI route); <workdir> is the
    # tempfile.mkdtemp(prefix="tg_gwas_") directory from run_model.
    workdir1 = tsv1.parent.parent
    assert workdir1.exists()
    assert str(workdir1.name).startswith("tg_gwas_")

    y2, G2 = _hwe_consistent_fixture(seed=11)
    r2 = tg.gwas(y2, G2, models="lmm", kinship="auto", pcs=0, verbose=False)
    tsv2 = r2.output_files["tsv"]
    workdir2 = tsv2.parent.parent
    assert workdir2.exists()
    assert workdir1 != workdir2

    del r1
    gc.collect()

    assert not workdir1.exists(), "first result's temp workdir should be removed after GC"
    # The still-referenced second result's workdir must survive -- cleanup
    # is per-result, not a blanket sweep.
    assert workdir2.exists()
    assert tsv2.exists()

    del r2
    gc.collect()
    assert not workdir2.exists()


def test_model_options_to_argv_conversion():
    from torchgenomics.api._dispatch import _model_options_to_argv
    assert _model_options_to_argv({}) == []
    assert _model_options_to_argv({"kernels": "additive,dominance"}) == ["--kernels", "additive,dominance"]
    assert _model_options_to_argv({"n_categories": 3}) == ["--n-categories", "3"]
    # bool True -> bare store_true flag; False -> omitted
    assert _model_options_to_argv({"firth": True}) == ["--firth"]
    assert _model_options_to_argv({"firth": False}) == []


def test_runoptions_has_model_options_default_none():
    from torchgenomics.api._dispatch import RunOptions
    assert RunOptions().model_options is None


def _binary_fixture(n=160, m=200, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    y = pd.Series(rng.integers(0, 2, size=n), index=ids, name="disease")
    return y, G

def test_glmm_extra_argv_infers_family_from_trait_type():
    import pandas as pd
    from torchgenomics.api._dispatch import _glmm_extra_argv
    from torchgenomics.api._inputs import GwasInputs
    # binary trait -> --family binary
    inp_b = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([0, 1, 1, 0]),
                       covariates=None, trait_type="binary", n_samples=4, trait_name="t")
    assert _glmm_extra_argv(inp_b, {}) == ["--family", "binary"]
    # categorical -> --family multinomial --n-categories <#distinct>
    inp_c = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([0, 1, 2, 1, 2]),
                       covariates=None, trait_type="categorical", n_samples=5, trait_name="t")
    argv = _glmm_extra_argv(inp_c, {})
    assert argv[:2] == ["--family", "multinomial"]
    assert "--n-categories" in argv and "3" in argv
    # continuous -> friendly ValueError
    inp_q = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([1.1, 2.2, 3.3]),
                       covariates=None, trait_type="continuous", n_samples=3, trait_name="height")
    import pytest
    with pytest.raises(ValueError) as e:
        _glmm_extra_argv(inp_q, {})
    assert "glmm" in str(e.value).lower() and "lmm" in str(e.value).lower()
    # user override wins (family explicitly set)
    assert _glmm_extra_argv(inp_b, {"family": "ordinal", "n_categories": 3}) == [
        "--family", "ordinal", "--n-categories", "3"]

def test_glmm_runs_through_tg_gwas_on_binary_trait():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _binary_fixture()
    r = tg.gwas(y, G, models="glmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0

def test_glmm_on_continuous_trait_friendly_error():
    import numpy as np, pandas as pd, pytest, torchgenomics as tg
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(120)]
    p = rng.uniform(0.2, 0.8, size=150)
    G = rng.binomial(2, p, size=(120, 150)).astype(float)
    y = pd.Series(rng.normal(size=120), index=ids, name="height")
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="glmm", kinship="auto", pcs=0, verbose=False)
    assert "glmm" in str(e.value).lower() and "lmm" in str(e.value).lower()


def _quant_fixture(n=140, m=180, seed=2):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    y = pd.Series(rng.normal(size=n), index=ids, name="yield")
    return y, G

def test_mklmm_extra_argv_default_and_override():
    from torchgenomics.api._dispatch import _mklmm_extra_argv
    assert _mklmm_extra_argv({}) == ["--kernels", "additive,dominance"]
    assert _mklmm_extra_argv({"kernels": "additive,dominance,epistatic"}) == [
        "--kernels", "additive,dominance,epistatic"]

def test_mklmm_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture()
    r = tg.gwas(y, G, models="mklmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0

def test_gwas_bogus_model_options_key_raises_friendly_value_error():
    """Fix 2 regression: an unrecognized model_options flag must not crash via SystemExit.

    argparse.parse_args() prints usage and raises SystemExit(2) on an
    unrecognized flag; SystemExit is a BaseException, so it slips past
    `except Exception` and reads as a raw crash in a notebook. tg.gwas
    must instead raise a friendly ValueError naming the offending alias and
    tokens.
    """
    import pytest
    import torchgenomics as tg
    y, G = _binary_fixture()
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="glmm", model_options={"glmm": {"nonsense": 1}},
                kinship="auto", pcs=0, verbose=False)
    msg = str(e.value)
    assert "nonsense" in msg
    assert "model_options" in msg


def test_glmm_result_test_label_is_score_not_wald():
    """Fix 4 regression: glmm-scan runs a PQL score test, not Wald.

    _run_lowlevel_cli previously read getattr(args, "test", "wald") for every
    low-level model, including glmm -- which mislabeled the GwasResult.test
    field as "wald" even though the GLMM null model is fit via PQL and scored
    with a score test.
    """
    import torchgenomics as tg
    y, G = _binary_fixture()
    r = tg.gwas(y, G, models="glmm", kinship="auto", pcs=0, verbose=False)
    assert r.test == "score"


def test_multimodel_comparison_includes_glmm():
    import torchgenomics as tg
    from torchgenomics.api import GwasComparison
    y, G = _binary_fixture()
    cmp = tg.gwas(y, G, models=["glm", "glmm"], kinship=False, pcs=0, verbose=False)
    assert isinstance(cmp, GwasComparison)
    assert set(cmp.results) == {"glm", "glmm"}

def test_gwas_model_options_reaches_runner(monkeypatch):
    """Fix 3: non-vacuous threading test.

    The previous version of this test passed {"kernels": "additive,dominance"}
    -- identical to mklmm's own default -- so it could not detect a dropped
    option (the assertion would pass even if model_options were silently
    ignored). This version overrides with a *distinguishing* non-default
    value ("additive" only, no dominance) and captures the RunOptions that
    actually reaches torchgenomics.api._dispatch._run_lowlevel_cli by
    monkeypatching it with a recording wrapper that still delegates to the
    real implementation -- proving the full gwas() -> RunOptions ->
    run_model -> _run_lowlevel_cli threading, in one real end-to-end call.
    """
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    from torchgenomics.api import _dispatch

    captured: dict = {}
    real_run_lowlevel_cli = _dispatch._run_lowlevel_cli

    def _recording_run_lowlevel_cli(inputs, opts, **kwargs):
        captured["model_options"] = opts.model_options
        return real_run_lowlevel_cli(inputs, opts, **kwargs)

    monkeypatch.setattr(_dispatch, "_run_lowlevel_cli", _recording_run_lowlevel_cli)

    y, G = _quant_fixture()
    r = tg.gwas(y, G, models="mklmm", kinship="auto", pcs=0, verbose=False,
                model_options={"mklmm": {"kernels": "additive"}})
    assert isinstance(r, GwasResult) and r.n_variants > 0
    assert captured["model_options"] == {"mklmm": {"kernels": "additive"}}


def test_lowlevel_cli_entries_are_4_tuples_with_default_readback():
    from torchgenomics.api._dispatch import _LOWLEVEL_CLI, _read_scan_output
    for alias, entry in _LOWLEVEL_CLI.items():
        assert len(entry) == 4, f"{alias} entry must be a 4-tuple (subcommand, runner, label, readback)"
        # existing models use the default read-back (None) or _read_scan_output explicitly
        assert entry[3] is None or callable(entry[3])


def test_gxe_requires_env_friendly_error():
    import torchgenomics as tg
    y, G = _quant_fixture()          # helper already in this file
    import pytest
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="gxe", kinship="auto", pcs=0, verbose=False)
    assert "gxe" in str(e.value).lower() and "env" in str(e.value).lower()

def test_gxe_runs_with_inmemory_env():
    import numpy as np, pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    ids = list(y.index)
    env = pd.Series(np.random.default_rng(7).normal(size=len(ids)), index=ids, name="ENV")
    r = tg.gwas(y, G, models="gxe", env=env, kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0
    # Confirms the P_JOINT read-back actually produced a value (not a
    # p_col=None path that would leave lambda_gc unset).
    assert r.lambda_gc is not None


def test_materialize_env_reindexes_and_errors_on_missing(tmp_path):
    """Review-fix regression: the pandas.Series branch of _materialize_env
    must ALWAYS reindex to the aligned sample order (inputs.phenotype.index)
    and raise a friendly ValueError if any aligned sample id is missing from
    the env Series — never silently fall through to the Series' own
    (possibly mismatched/shuffled) order. gxe-scan consumes env positionally
    by row order, so a silent misorder would corrupt GxE p-values.
    """
    import numpy as np
    import pandas as pd
    import pytest
    from torchgenomics.api._dispatch import _materialize_env
    from torchgenomics.api._inputs import load_inputs

    n = 20
    ids = [f"s{i}" for i in range(n)]
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, size=30)
    G = rng.binomial(2, p, size=(n, 30)).astype(float)
    y = pd.Series(rng.normal(size=n), index=ids, name="y")
    inputs = load_inputs(phenotype=y, genotype=G)

    # 1) env Series indexed by the same ids but in SHUFFLED order must be
    #    written out in inputs.phenotype.index order, not its own order.
    shuffled_ids = list(ids)
    rng.shuffle(shuffled_ids)
    env = pd.Series(rng.normal(size=n), index=shuffled_ids, name="ENV")
    path = _materialize_env(env, inputs, tmp_path)
    written = pd.read_csv(path, sep="\t")["ENV"].to_numpy()
    expected = env.reindex(inputs.phenotype.index).to_numpy()
    assert np.allclose(written, expected)

    # 2) env Series missing one aligned sample id must raise a friendly
    #    ValueError mentioning the problem, not silently fall back.
    incomplete_ids = [i for i in ids if i != "s5"]
    env_missing = pd.Series(rng.normal(size=n - 1), index=incomplete_ids, name="ENV")
    with pytest.raises(ValueError) as e:
        _materialize_env(env_missing, inputs, tmp_path)
    msg = str(e.value).lower()
    assert "missing" in msg and "sample" in msg


def test_materialize_env_id_keyed_aligns_under_reorder(tmp_path):
    """CRITICAL regression: env must be transported BY SAMPLE ID, not by row
    position, so it survives a downstream sort-order change between where it
    is materialized (``inputs.phenotype.index`` order) and where it is read
    back (``_cmd_gxe_scan``'s ``aligned_reader.sample_ids``, which for a PATH
    genotype is lexicographically sorted by ``load_phenotype`` regardless of
    ``inputs.phenotype``'s original order).

    In-memory genotypes route through ``load_inputs`` -> ``_align_by_ids``,
    which itself sorts ids, so building an *unsorted* ``inputs.phenotype``
    that way is impossible; instead we hand-build a ``GwasInputs`` with a
    phenotype Series whose index is deliberately NOT sorted (``s2, s0, s1``)
    -- the same shape a path-genotype run produces internally -- and prove
    ``_materialize_env`` + ``_load_env_vector`` round-trip correctly even
    when the read-back sample order differs from the write-time order.
    """
    import numpy as np
    import pandas as pd
    import torch

    from torchgenomics import cli
    from torchgenomics.api._dispatch import _materialize_env
    from torchgenomics.api._inputs import GwasInputs

    unsorted_ids = ["s2", "s0", "s1"]
    phenotype = pd.Series([10.0, 20.0, 30.0], index=unsorted_ids, name="y")
    inputs = GwasInputs(
        genotype_path_or_reader=None,
        phenotype=phenotype,
        covariates=None,
        trait_type="continuous",
        n_samples=3,
        trait_name="y",
    )

    # env Series indexed in the SAME unsorted order as inputs.phenotype ---
    # distinct, identifiable-by-id values.
    env = pd.Series([100.0, 200.0, 300.0], index=unsorted_ids, name="ENV")
    path = _materialize_env(env, inputs, tmp_path)

    # _materialize_env must have written a SAMPLE id column (not ENV-only).
    written = pd.read_csv(path, sep="\t")
    assert "SAMPLE" in written.columns and "ENV" in written.columns

    # Read back with a DIFFERENT (sorted) sample order -- as _cmd_gxe_scan
    # does for a path genotype via aligned_reader.sample_ids.
    sorted_ids = ["s0", "s1", "s2"]
    result = cli._load_env_vector(path, sorted_ids, dtype=torch.float64, device="cpu")

    expected = env.reindex(sorted_ids).to_numpy()
    assert np.allclose(result.numpy(), expected), (
        "env must align BY SAMPLE ID: expected "
        f"{expected} (id-reordered) but got {result.numpy()} -- a positional "
        "read would incorrectly return [100, 200, 300] (env's own write-time "
        "order) instead."
    )


def test_read_bayes_output_pip_sorted(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _read_bayes_output
    prefix = str(tmp_path / "run")
    pd.DataFrame({
        "SNP": ["s1", "s2", "s3"], "CHR": [1, 1, 1], "POS": [10, 20, 30],
        "A1": ["A", "A", "A"], "A2": ["B", "B", "B"], "AF": [0.3, 0.4, 0.5],
        "PIP": [0.1, 0.9, 0.4], "BETA_MEAN": [0.0, 1.0, 0.2], "BETA_SD": [0.1, 0.1, 0.1],
    }).to_csv(prefix + "_bayesian_vs.tsv", sep="\t", index=False)
    pd.DataFrame({"SNP": ["s2"], "CS": [1]}).to_csv(prefix + "_credible_sets.tsv", sep="\t", index=False)
    r = _read_bayes_output(prefix, model_label="BayesianVS", test="susie", correction="none",
                           trait_type="continuous", n_samples=100, significance_threshold=5e-8,
                           top_k=50, runtime_s=0.0)
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult)
    assert r.lambda_gc is None and r.model == "BayesianVS"
    assert list(r.top_hits["SNP"])[0] == "s2"        # highest PIP first
    assert r.n_significant == 1                        # one variant in a credible set

def test_read_bayes_output_missing_pip_column_raises_clear_error(tmp_path):
    """MINOR fix: a bayes output missing PIP must raise a clear RuntimeError
    (symmetric with _read_set_output's "if 'P' in df.columns" guard), not a
    bare KeyError from df.sort_values('PIP')."""
    import pandas as pd
    import pytest
    from torchgenomics.api._dispatch import _read_bayes_output

    prefix = str(tmp_path / "run")
    pd.DataFrame({"SNP": ["s1"], "CHR": [1], "POS": [10]}).to_csv(
        prefix + "_bayesian_vs.tsv", sep="\t", index=False
    )
    with pytest.raises(RuntimeError) as e:
        _read_bayes_output(prefix, model_label="BayesianVS", test="susie", correction="none",
                            trait_type="continuous", n_samples=100, significance_threshold=5e-8,
                            top_k=50, runtime_s=0.0)
    assert "PIP" in str(e.value)


def test_bayes_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    r = tg.gwas(y, G, models="bayes", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "BayesianVS"
    assert r.lambda_gc is None and "PIP" in r.top_hits.columns


def test_read_set_output_region_level(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _read_set_output
    prefix = str(tmp_path / "run")
    pd.DataFrame({
        "REGION": ["geneA", "geneB"], "CHR": [1, 1], "START": [1, 100], "END": [50, 150],
        "N_VARIANTS": [10, 8], "P": [1e-9, 0.2],
    }).to_csv(prefix + "_set_based.tsv", sep="\t", index=False)
    r = _read_set_output(prefix, model_label="SetBasedScanner", test="skat", correction="bh",
                         trait_type="continuous", n_samples=100, significance_threshold=5e-8,
                         top_k=50, runtime_s=0.0)
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult) and r.model == "SetBasedScanner"
    assert r.n_variants == 2 and r.n_significant == 1        # geneA below threshold
    assert list(r.top_hits["REGION"])[0] == "geneA"          # smallest P first

def test_set_requires_regions_friendly_error():
    import torchgenomics as tg, pytest
    y, G = _quant_fixture()
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="set", kinship="auto", pcs=0, verbose=False)
    assert "set" in str(e.value).lower() and "region" in str(e.value).lower()


def test_set_runs_end_to_end_with_regions_df():
    import pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    # One region spanning the first 40 synthesized variants. ArrayReader
    # (torchgenomics/api/_inputs.py) synthesizes chr="0" (not "1") and
    # pos=0..m-1 for an in-memory ndarray with no VariantMeta of its own --
    # CHR must be "0" here to actually match those variants (see
    # map_regions_to_variants' exact chr string match in
    # torchgenomics/io/regions.py).
    regions = pd.DataFrame({"CHR": ["0"], "START": [0], "END": [39], "REGION": ["blockA"]})
    r = tg.gwas(y, G, models="set", regions=regions, kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "SetBasedScanner" and r.n_variants >= 1


def test_mvlmm_via_gwas_requires_traits():
    # mvlmm is now wired through _LOWLEVEL_CLI (Task 2); with a single-trait
    # phenotype and no traits=, it raises the friendly "needs >=2 traits"
    # ValueError from _mvlmm_extra_argv rather than the old NotImplementedError.
    # Full traits= support in tg.gwas() itself lands in Task 3.
    import torchgenomics as tg, pytest
    y, G = _quant_fixture()
    with pytest.raises(ValueError, match="(?i)mvlmm.*trait"):
        tg.gwas(y, G, models="mvlmm", kinship="auto", pcs=0, verbose=False)


def _multitrait_fixture(n=140, m=160, seed=3):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    base = rng.normal(size=n)
    pheno = pd.DataFrame(
        {"Y1": base + rng.normal(scale=0.5, size=n),
         "Y2": base + rng.normal(scale=0.5, size=n),
         "Y3": rng.normal(size=n)},
        index=ids,
    )
    return pheno, G

def test_load_inputs_multitrait():
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    assert inp.trait_names == ["Y1", "Y2"]
    assert inp.phenotypes is not None and inp.phenotypes.shape[1] == 2
    assert list(inp.phenotypes.columns) == ["Y1", "Y2"]
    assert inp.phenotype.name == "Y1"          # first trait as the Series
    assert inp.n_samples == inp.phenotypes.shape[0]
    assert inp.trait_type == "continuous"

def test_load_inputs_multitrait_missing_column_errors():
    from torchgenomics.api._inputs import load_inputs
    import pytest
    pheno, G = _multitrait_fixture()
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "NOPE"])
    assert "NOPE" in str(e.value)

def test_load_inputs_multitrait_bare_series_errors():
    from torchgenomics.api._inputs import load_inputs
    import pandas as pd, numpy as np, pytest
    ids = [f"s{i}" for i in range(20)]
    y = pd.Series(np.arange(20.0), index=ids, name="Y1")
    G = pd.DataFrame(np.zeros((20, 10)), index=ids)
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=y, genotype=G, traits=["Y1", "Y2"])
    assert "multi-trait" in str(e.value).lower() or "dataframe" in str(e.value).lower()

def test_mvlmm_extra_argv_traits():
    # CRITICAL fix: mvlmm-scan's CLI subparser has no --ploidy flag (only
    # poly-scan does); emitting --ploidy made every mvlmm run fail with
    # "Unrecognized model_options for 'mvlmm': ['--ploidy', ...]" from the
    # unknown-arg guard in _run_lowlevel_cli. _mvlmm_extra_argv must emit
    # only --traits (+ any real user-supplied model_options).
    from torchgenomics.api._dispatch import _mvlmm_extra_argv, RunOptions
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    argv = _mvlmm_extra_argv(inp, RunOptions(), {})
    assert "--traits" in argv and "Y1,Y2" in argv
    assert "--ploidy" not in argv
    # a real mvlmm-scan flag (--test is a common-scan arg mvlmm-scan accepts)
    # passes through untouched
    argv2 = _mvlmm_extra_argv(inp, RunOptions(), {"test": "score"})
    assert "--test" in argv2 and argv2[argv2.index("--test") + 1] == "score"
    assert "--ploidy" not in argv2

def test_mvlmm_requires_two_traits_friendly_error():
    from torchgenomics.api._dispatch import _mvlmm_extra_argv, RunOptions
    from torchgenomics.api._inputs import load_inputs
    import pytest
    pheno, G = _multitrait_fixture()
    inp1 = load_inputs(phenotype=pheno, genotype=G, traits=["Y1"])   # only 1 trait
    with pytest.raises(ValueError) as e:
        _mvlmm_extra_argv(inp1, RunOptions(), {})
    assert "mvlmm" in str(e.value).lower() and "trait" in str(e.value).lower()

def test_write_multitrait_phenotype_tsv(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _write_multitrait_phenotype_tsv
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    p = tmp_path / "ph.tsv"
    _write_multitrait_phenotype_tsv(inp, p)
    back = pd.read_csv(p, sep="\t")
    assert "sample_id" in back.columns and "Y1" in back.columns and "Y2" in back.columns
    assert len(back) == inp.n_samples


def test_mvlmm_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    pheno, G = _multitrait_fixture()
    r = tg.gwas(pheno, G, models="mvlmm", traits=["Y1", "Y2"],
                kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "MultiTraitLMM" and r.n_variants > 0


def test_mvlmm_without_traits_friendly_error():
    import torchgenomics as tg, pytest
    pheno, G = _multitrait_fixture()
    with pytest.raises(ValueError) as e:
        tg.gwas(pheno, G, models="mvlmm", kinship="auto", pcs=0, verbose=False)
    assert "mvlmm" in str(e.value).lower() and "trait" in str(e.value).lower()


def test_mvlmm_runs_via_run_model():
    # CRITICAL regression proof: before the fix, this call raised
    # ValueError: "Unrecognized model_options for 'mvlmm': ['--ploidy', '2']"
    # because _mvlmm_extra_argv emitted a --ploidy flag mvlmm-scan's CLI
    # subparser doesn't accept. mvlmm must now run end-to-end via run_model.
    from torchgenomics.api._dispatch import run_model, RunOptions
    from torchgenomics.api._inputs import load_inputs
    from torchgenomics.api import GwasResult

    pheno, G = _multitrait_fixture(n=140, m=160)
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    r = run_model("mvlmm", inp, RunOptions(kinship="auto", pcs=0))
    assert isinstance(r, GwasResult)
    assert r.model == "MultiTraitLMM"
    assert r.n_variants > 0
