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

def test_ancestry_dosages_container_and_chunking(tmp_path):
    import numpy as np
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages, AncestryDosages
    # write two Tractor-style dosage files: rows=variants, cols=samples
    n, m = 6, 10
    for k in range(2):
        arr = np.arange(m * n).reshape(m, n).astype(float) + k
        np.savetxt(tmp_path / f"anc{k}.dosage.txt", arr)
    ad = read_ancestry_dosages(str(tmp_path / "anc"), ancestry_names=["AFR", "EUR"])
    assert isinstance(ad, AncestryDosages)
    assert ad.n_ancestries() == 2
    assert ad.dosages.shape == (2, n, m)          # (K, n, m) after transpose
    chunks = list(ad.iter_chunks(chunk_size=4))
    assert len(chunks) == 3                        # 10 variants / 4 -> 3 chunks
    assert chunks[0][0].dosages.shape == (2, n, 4)

def test_read_ancestry_dosages_npy_roundtrip(tmp_path):
    import numpy as np
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages
    arr = np.arange(2 * 5 * 3).reshape(2, 5, 3).astype(np.float64)
    p = tmp_path / "stack.npy"
    np.save(p, arr)
    ad = read_ancestry_dosages(str(p), fmt="npy")
    assert ad.dosages.shape == (2, 5, 3)
    assert ad.dosages.dtype == torch.float64
    assert torch.equal(ad.dosages, torch.as_tensor(arr, dtype=torch.float64))
    # auto-detect by extension too
    ad2 = read_ancestry_dosages(str(p))
    assert ad2.dosages.shape == (2, 5, 3)

def test_read_ancestry_dosages_missing_files_raises(tmp_path):
    import pytest
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages
    with pytest.raises(FileNotFoundError):
        read_ancestry_dosages(str(tmp_path / "nope"))

def test_read_ancestry_dosages_name_length_mismatch_raises(tmp_path):
    import pytest
    import numpy as np
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages
    n, m = 4, 3
    for k in range(2):
        arr = np.arange(m * n).reshape(m, n).astype(float) + k
        np.savetxt(tmp_path / f"anc{k}.dosage.txt", arr)
    with pytest.raises(ValueError):
        read_ancestry_dosages(str(tmp_path / "anc"), ancestry_names=["AFR", "EUR", "AMR"])

def test_read_ancestry_dosages_numeric_sort_ge_10(tmp_path):
    import numpy as np
    from torchgenomics.io.ancestry_dosage import read_ancestry_dosages
    n, m = 3, 2
    # write anc0..anc11 with a marker value equal to k so we can check order
    for k in range(12):
        arr = np.full((m, n), float(k))
        np.savetxt(tmp_path / f"anc{k}.dosage.txt", arr)
    ad = read_ancestry_dosages(str(tmp_path / "anc"))
    assert ad.n_ancestries() == 12
    # dosages[k] should be filled with value k (numeric order, not lexicographic)
    for k in range(12):
        assert torch.allclose(ad.dosages[k], torch.full((n, m), float(k), dtype=torch.float64))

def test_iter_chunks_preserves_sliced_variant_meta():
    from torchgenomics.io.ancestry_dosage import AncestryDosages
    from torchgenomics.models.base import VariantMeta
    K, n, m = 2, 3, 7
    dosages = torch.zeros(K, n, m, dtype=torch.float64)
    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )
    ad = AncestryDosages(dosages, ["AFR", "EUR"], variant_meta=vmeta)
    chunks = list(ad.iter_chunks(chunk_size=3))
    assert len(chunks) == 3
    offset = 0
    for chunk, sl in chunks:
        expected_len = sl.stop - sl.start
        assert chunk.variant_meta is not None
        assert len(chunk.variant_meta) == expected_len
        assert chunk.variant_meta.snp == vmeta.snp[sl]
        assert chunk.variant_meta.pos == vmeta.pos[sl]
        offset += expected_len

def test_iter_chunks_variant_meta_none_stays_none():
    from torchgenomics.io.ancestry_dosage import AncestryDosages
    dosages = torch.zeros(2, 3, 5, dtype=torch.float64)
    ad = AncestryDosages(dosages, ["AFR", "EUR"])
    chunks = list(ad.iter_chunks(chunk_size=2))
    for chunk, _sl in chunks:
        assert chunk.variant_meta is None

def test_tractor_null_projection_matches_bruteforce():
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    d = make_synth(n=120, m=30, K=2, seed=3)
    Y, X0, Kg = d["y_cont"], d["X0"], d["K_grm"]
    nf = TractorLMM(family="gaussian").fit_null(Y, X0, Kg)
    # brute-force P y from the fitted variance components
    V = nf.sigma2 * torch.eye(len(Y), dtype=torch.float64) + nf.tau2 * Kg
    Vi = torch.linalg.inv(V)
    P = Vi - Vi @ X0 @ torch.linalg.inv(X0.T @ Vi @ X0) @ X0.T @ Vi
    assert torch.allclose(nf.resid, P @ Y, atol=1e-8)
    g = d["dosages"][0, :, 0]
    assert torch.allclose(nf.Py(g.unsqueeze(1)).squeeze(), P @ g, atol=1e-8)
