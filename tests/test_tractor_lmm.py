import torch
from tests.fixtures.tractor.make_synth import make_synth

def test_make_synth_shapes_and_determinism():
    d1 = make_synth(n=200, m=50, K=2, seed=7)
    d2 = make_synth(n=200, m=50, K=2, seed=7)
    assert d1["dosages"].shape == (2, 200, 50)
    assert d1["y_cont"].shape == (200,)
    assert d1["K_grm"].shape == (200, 200)
    # ancestry dosages sum to total genotype dosage in [0, 2]
    tot = d1["dosages"].sum(dim=0)
    assert tot.min() >= 0 and tot.max() <= 2.0 + 1e-6
    # GRM is symmetric PSD
    assert torch.allclose(d1["K_grm"], d1["K_grm"].T, atol=1e-6)
    assert torch.linalg.eigvalsh(d1["K_grm"]).min() > -1e-6
    # determinism
    assert torch.equal(d1["dosages"], d2["dosages"])
