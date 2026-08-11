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
