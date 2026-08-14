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
