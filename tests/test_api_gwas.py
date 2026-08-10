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
