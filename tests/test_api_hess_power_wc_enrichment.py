import pandas as pd


def test_result_classes_shape_and_summary():
    from torchgenomics.api import PowerRun, WinnersCurseRun, EnrichmentRun, HessRun
    p = PowerRun(alpha=5e-8, n=5000, target_power=0.8, n_variants=3, n_powered=1,
                 results=pd.DataFrame({"snp": ["a"], "power": [0.9]}))
    assert "Power" in p.summary() and p.to_dict()["n_powered"] == 1
    w = WinnersCurseRun(method="conditional_likelihood", n_corrected=2, n_variants=10,
                        results=pd.DataFrame({"snp": ["a"], "beta_adjusted": [0.1]}))
    assert "curse" in w.summary().lower() and w.to_dict()["n_corrected"] == 2
    e = EnrichmentRun(n_genes_total=100, n_gene_sets=5, n_significant=1,
                      results=pd.DataFrame({"gene_set_name": ["s1"], "p": [1e-3]}),
                      genes=pd.DataFrame({"gene_id": ["g1"], "p": [1e-4]}))
    assert "nrichment" in e.summary() and e.to_dict()["n_gene_sets"] == 5
    h = HessRun(mode="h2", h2_total=0.3, h2_total_se=0.05, n_regions=4, n_snps_total=400,
                results=pd.DataFrame({"region_id": ["chr1:1-2"], "h2_local": [0.1]}))
    assert "HESS" in h.summary() and h.to_dict()["n_regions"] == 4
