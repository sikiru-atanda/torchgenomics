import numpy as np, pandas as pd, pytest
from torchgenomics.api._inputs import detect_trait_type, load_inputs

def test_trait_type_detection():
    assert detect_trait_type(pd.Series([0,1,1,0,1])) == "binary"
    assert detect_trait_type(pd.Series([1.2,3.4,0.1,9.9,2.2])) == "continuous"
    assert detect_trait_type(pd.Series([0,1,2,1,2,0,1])) == "categorical"

def test_load_inputs_in_memory_and_alignment(tmp_path):
    # in-memory phenotype Series + genotype array with matching sample ids
    ids = [f"s{i}" for i in range(20)]
    y = pd.Series(np.random.default_rng(0).normal(size=20), index=ids, name="yield")
    G = np.random.default_rng(1).integers(0,3,(20,50)).astype(float)  # (n, m)
    inp = load_inputs(phenotype=y, genotype=G)
    assert inp.n_samples == 20 and inp.trait_name == "yield"
    assert inp.trait_type == "continuous"

def test_friendly_error_on_no_shared_samples():
    y = pd.Series([1.0,2.0,3.0], index=["a","b","c"], name="t")
    G = pd.DataFrame(np.zeros((2,10)), index=["x","y"])   # disjoint ids
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=y, genotype=G)
    assert "shared" in str(e.value).lower()   # actionable, not a KeyError/traceback
