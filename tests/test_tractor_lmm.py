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

def test_joint_score_recovers_causal_ancestry():
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=400, m=40, K=2, seed=11)
    m = TractorLMM(family="gaussian", test="score", ancestry_names=["AFR", "EUR"])
    nf = m.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(40)],
        chr=["1"] * 40,
        pos=list(range(40)),
        a1=["A"] * 40,
        a2=["G"] * 40,
    )
    res = m.score_chunk(nf, d["dosages"], meta)
    df = res.to_dataframe()
    # causal variants (ancestry-0 effect) reach smaller joint p than non-causal
    causal = set(d["causal_idx"])
    jp = df["joint_p"].to_numpy()
    causal_p = [jp[i] for i in range(40) if i in causal]
    null_p = [jp[i] for i in range(40) if i not in causal]
    assert min(causal_p) < 1e-3
    assert sum(p < 1e-3 for p in null_p) <= 1        # few/no null hits
    # ancestry-0 effect sign is positive at the top causal variant
    top = min(causal, key=lambda i: jp[i])
    assert df["beta_AFR"].to_numpy()[top] > 0

def test_allele_count_threshold_drops_rare_ancestry():
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=300, m=5, K=2, seed=5)
    # force ancestry-1 (EUR) dosages to zero at variant 0 -> allele count 0 < 50
    d["dosages"][1, :, 0] = 0.0
    mdl = TractorLMM(family="gaussian", min_allele_count=50, ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(5)],
        chr=["1"] * 5,
        pos=list(range(5)),
        a1=["A"] * 5,
        a2=["G"] * 5,
    )
    res = mdl.score_chunk(nf, d["dosages"], meta)
    df = res.to_dataframe()
    # ancestry-1 (EUR) effect at variant 0 is NaN (dropped); ancestry-0 still tested
    assert df["p_EUR"].isna()[0] or df["se_EUR"].isna()[0]
    assert not df["joint_p"].isna()[0]  # AFR survived -> marginal joint p present


def test_allele_count_threshold_all_dropped_gives_nan_joint_p():
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    d = make_synth(n=300, m=5, K=2, seed=5)
    # force BOTH ancestries below threshold at variant 0
    d["dosages"][0, :, 0] = 0.0
    d["dosages"][1, :, 0] = 0.0
    mdl = TractorLMM(family="gaussian", min_allele_count=50, ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(5)],
        chr=["1"] * 5,
        pos=list(range(5)),
        a1=["A"] * 5,
        a2=["G"] * 5,
    )
    res = mdl.score_chunk(nf, d["dosages"], meta)
    df = res.to_dataframe()
    assert df["joint_p"].isna()[0]
    assert df["se_AFR"].isna()[0] and df["se_EUR"].isna()[0]


def test_binary_null_calibrated_under_null():
    """Binary (logistic GLMM, PQL null) Tractor-Mix score test must be
    well-calibrated when genotype is independent of phenotype.

    Permuting samples in the ancestry dosages breaks any genotype-phenotype
    association, so the joint score p-values should follow Uniform(0,1): the
    fraction below 0.05 must sit near the nominal 5% (slack for finite m).
    A miscalibrated variance construction (e.g. the wrong working-covariance
    sketch) fails this gate.
    """
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=500, m=200, K=2, seed=9)
    # NULL: permute samples in the dosages so genotype _|_ phenotype.
    perm = torch.randperm(500, generator=torch.Generator().manual_seed(1))
    Gnull = d["dosages"][:, perm, :]
    mdl = TractorLMM(family="binary", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(200)],
        chr=["1"] * 200,
        pos=list(range(200)),
        a1=["A"] * 200,
        a2=["G"] * 200,
    )
    res = mdl.score_chunk(nf, Gnull, meta)
    jp = res.to_dataframe()["joint_p"].to_numpy()
    jp = jp[~torch.isnan(torch.as_tensor(jp)).numpy()]
    frac = (jp < 0.05).mean()
    assert 0.01 < frac < 0.15, f"joint_p not calibrated under null: frac={frac}"


def test_binary_reduces_to_binary_glmm_single_ancestry():
    """A single-ancestry (K=1) Tractor-Mix binary test must reproduce the
    validated 1-df BinaryGLMM score test exactly (same U, V, chi2_1 p-value),
    since the K-column construction is BinaryGLMM's score test generalized
    from a scalar SNP to an ancestry-dosage matrix.
    """
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.binary_glmm import BinaryGLMM
    from torchgenomics.models.base import VariantMeta
    from scipy.stats import chi2 as chi2_dist

    d = make_synth(n=300, m=6, K=2, seed=4)
    Y, X0, Kg = d["y_bin"], d["X0"], d["K_grm"]
    # collapse the two ancestry tracks to a single total-dosage track so the
    # Tractor test has K=1 (one ancestry column == the ordinary SNP dosage).
    G1 = d["dosages"].sum(dim=0, keepdim=True)  # (1, n, m)

    mdl = TractorLMM(family="binary", min_allele_count=0, ancestry_names=["A0"])
    nf = mdl.fit_null(Y, X0, Kg)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(6)], chr=["1"] * 6, pos=list(range(6)),
        a1=["A"] * 6, a2=["G"] * 6,
    )
    res = mdl.score_chunk(nf, G1, meta)

    # reference: BinaryGLMM 1-df score test (SPA off for a bare chi2_1 tail)
    bg = BinaryGLMM(use_spa=False)
    bnf = bg.fit_null(Y, X0, Kg)
    G2d = G1[0]  # (n, m)
    ref = bg.score_chunk(G2d, bnf, meta)
    # joint chi2_1 p (rank 1) must match BinaryGLMM's chi2_1 p to high precision
    assert torch.allclose(res.joint_p, ref.p, atol=1e-9, rtol=0)
    _ = chi2_dist  # referenced for intent


def test_rank_deficient_surviving_var_uses_reduced_df():
    """Two surviving ancestry dosage columns that are exactly collinear make
    Var = Gv^T P Gv rank-deficient (rank 1, not rank 2). The chi2 tail must
    use df = matrix_rank(Var), not the raw surviving-ancestry count, or the
    p-value is miscalibrated (too conservative -- using df=2 on a stat that
    is effectively 1-df understates significance).
    """
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta
    from scipy.stats import chi2 as chi2_dist

    d = make_synth(n=300, m=3, K=2, seed=9)
    # Make ancestry-1 dosage at variant 0 an exact copy of ancestry-0's
    # dosage at variant 0 -> Gv columns collinear -> Var singular (rank 1).
    d["dosages"][1, :, 0] = d["dosages"][0, :, 0].clone()
    # Ensure both columns clear the allele-count threshold so neither is
    # dropped by the min_allele_count filter -- the rank deficiency must be
    # caught independently of that filter.
    assert d["dosages"][0, :, 0].sum() >= 50
    assert d["dosages"][1, :, 0].sum() >= 50

    mdl = TractorLMM(family="gaussian", min_allele_count=50, ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(3)],
        chr=["1"] * 3,
        pos=list(range(3)),
        a1=["A"] * 3,
        a2=["G"] * 3,
    )
    res = mdl.score_chunk(nf, d["dosages"], meta)

    stat0 = float(res.joint_stat[0])
    p0 = float(res.joint_p[0])
    # rank of the (singular) 2x2 Var at variant 0 must be 1, not 2
    Gv0 = d["dosages"][:, :, 0].T.to(__import__("torch").float64)
    PGv0 = nf.Py(Gv0)
    Var0 = Gv0.T @ PGv0
    rank0 = int(__import__("torch").linalg.matrix_rank(Var0))
    assert rank0 == 1
    # the p-value actually used must match the reduced-df tail, not df=2
    expected_reduced = float(chi2_dist.sf(stat0, df=rank0))
    expected_full = float(chi2_dist.sf(stat0, df=2))
    assert abs(p0 - expected_reduced) < 1e-9
    assert p0 != expected_full
    assert 0.0 <= p0 <= 1.0
