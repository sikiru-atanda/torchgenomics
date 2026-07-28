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


def test_wald_matches_score_at_moderate_signal():
    """Wald full-model test (Task 7) must closely track the Rao score test
    at moderate effect size, since both estimate the same GLS ancestry
    effect at the fixed null variance components (FWL theorem: the score
    estimate Vinv @ T and the Wald GLS block estimate are algebraically
    equivalent when V is held fixed rather than re-estimated per variant).
    """
    import numpy as np
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=400, m=20, K=2, seed=21)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(20)],
        chr=["1"] * 20,
        pos=list(range(20)),
        a1=["A"] * 20,
        a2=["G"] * 20,
    )
    sc = TractorLMM(test="score", ancestry_names=["AFR", "EUR"])
    wd = TractorLMM(test="wald", ancestry_names=["AFR", "EUR"])
    nfs = sc.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    nfw = wd.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    res_s = sc.score_chunk(nfs, d["dosages"], meta)
    res_w = wd.score_chunk(nfw, d["dosages"], meta)
    assert res_s.test == "score"
    assert res_w.test == "wald"
    ds = res_s.to_dataframe()
    dw = res_w.to_dataframe()
    assert dw["joint_p"].to_numpy().shape == (20,)
    assert np.corrcoef(ds["beta_AFR"], dw["beta_AFR"])[0, 1] > 0.98


def test_wald_binary_not_implemented():
    """Wald is gaussian-only for now (Task 7 scope); binary + wald must
    raise a clear NotImplementedError rather than silently doing something
    wrong or crashing deep in the score-test machinery.
    """
    import pytest
    from torchgenomics.models.tractor_lmm import TractorLMM

    with pytest.raises(NotImplementedError):
        TractorLMM(family="binary", test="wald")


def test_spa_off_is_identical_and_on_changes_tail():
    """Opt-in SPA (Task 8) for the per-ancestry binary score component.

    (1) SPA-off identity guard: the default (``use_spa=False``) path must be
        completely inert to ``spa_threshold`` -- proving the gated SPA block
        never executes (no import, no call) when the feature is off, which
        protects the faithful normal-approx equivalence claim.
    (2) SPA-on changes at least one tail per-ancestry p-value relative to the
        normal approximation, demonstrating the SPA correction actually wires
        through to ``p_anc`` when enabled.
    """
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=600, m=10, K=2, seed=31)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(10)],
        chr=["1"] * 10,
        pos=list(range(10)),
        a1=["A"] * 10,
        a2=["G"] * 10,
    )
    names = ["AFR", "EUR"]
    off = TractorLMM(family="binary", use_spa=False, ancestry_names=names)
    on = TractorLMM(family="binary", use_spa=True, spa_threshold=1.0, ancestry_names=names)
    # default-off must be INERT to SPA settings (proves the faithful path is
    # pure normal-approx): flipping spa_threshold with use_spa=False changes
    # nothing.
    off_lowthr = TractorLMM(family="binary", use_spa=False, spa_threshold=0.0,
                             ancestry_names=names)
    nf1 = off.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    nf2 = on.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    nf3 = off_lowthr.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    a = off.score_chunk(nf1, d["dosages"], meta).to_dataframe()
    b = on.score_chunk(nf2, d["dosages"], meta).to_dataframe()
    c = off_lowthr.score_chunk(nf3, d["dosages"], meta).to_dataframe()
    # (1) SPA-off identity guard: default path unaffected by SPA settings
    assert (a["p_AFR"].to_numpy() == c["p_AFR"].to_numpy()).all()
    # (2) SPA-on changes at least one tail p-value
    assert not (a["p_AFR"].to_numpy() == b["p_AFR"].to_numpy()).all()


def test_streaming_scan_matches_full_chunk_joint_p():
    """TractorLMM.scan() streamed in two chunk sizes must reproduce the
    same joint_p as a single non-chunked score_chunk call (streaming
    invariance) -- score_chunk already scores variants independently, so
    chunking must not change any per-variant result.
    """
    import numpy as np
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=300, m=24, K=2, seed=41)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(24)],
        chr=["1"] * 24,
        pos=list(range(24)),
        a1=["A"] * 24,
        a2=["G"] * 24,
    )
    mdl = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])

    full = mdl.score_chunk(nf, d["dosages"], meta).to_dataframe()["joint_p"].to_numpy()

    # Streamed via the dedicated scan() driver on the raw (K, n, m) tensor,
    # with a chunk_size that does not evenly divide m (24 / 7 -> 4 chunks).
    streamed_res = mdl.scan(nf, d["dosages"], meta, chunk_size=7)
    streamed = streamed_res.to_dataframe()["joint_p"].to_numpy()
    assert np.allclose(full, streamed, atol=1e-10, equal_nan=True)
    assert streamed_res.snp == list(meta.snp)
    assert streamed_res.chr == list(meta.chr)
    assert streamed_res.pos == list(meta.pos)

    # Manual two-half-chunk equivalence (mirrors the brief's smoke check).
    h1 = mdl.score_chunk(
        nf, d["dosages"][:, :, :12],
        VariantMeta(snp=[f"v{i}" for i in range(12)], chr=["1"] * 12,
                    pos=list(range(12)), a1=["A"] * 12, a2=["G"] * 12),
    )
    h2 = mdl.score_chunk(
        nf, d["dosages"][:, :, 12:],
        VariantMeta(snp=[f"v{i}" for i in range(12, 24)], chr=["1"] * 12,
                    pos=list(range(12, 24)), a1=["A"] * 12, a2=["G"] * 12),
    )
    manual_streamed = np.concatenate(
        [h1.to_dataframe()["joint_p"], h2.to_dataframe()["joint_p"]]
    )
    assert np.allclose(full, manual_streamed, atol=1e-10, equal_nan=True)


def test_streaming_scan_over_ancestry_dosages_container():
    """TractorLMM.scan() must also work when handed an AncestryDosages
    container (not just a raw tensor), iterating its own iter_chunks and
    adopting its ancestry_names when the model has none set.
    """
    import numpy as np
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.io.ancestry_dosage import AncestryDosages
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=200, m=17, K=2, seed=13)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(17)],
        chr=["1"] * 17,
        pos=list(range(17)),
        a1=["A"] * 17,
        a2=["G"] * 17,
    )
    mdl = TractorLMM(family="gaussian")  # no ancestry_names -> adopt from AD
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])

    full = mdl.score_chunk(nf, d["dosages"], meta).to_dataframe()["joint_p"].to_numpy()

    ad = AncestryDosages(d["dosages"], ["AFR", "EUR"], variant_meta=meta)
    res = mdl.scan(nf, ad, meta, chunk_size=5)
    df = res.to_dataframe()
    assert np.allclose(full, df["joint_p"].to_numpy(), atol=1e-10, equal_nan=True)
    assert res.ancestry_names == ["AFR", "EUR"]
    assert "beta_AFR" in df.columns and "beta_EUR" in df.columns
    assert len(res) == 17


def test_conditional_joint_p_present_only_with_local_ancestry():
    """Passing L_chunk to score_chunk must add a conditional_joint_p
    column; without it, no such column exists (unchanged output).
    """
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=300, m=10, K=2, seed=41)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(10)],
        chr=["1"] * 10,
        pos=list(range(10)),
        a1=["A"] * 10,
        a2=["G"] * 10,
    )
    mdl = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])

    no_l = mdl.score_chunk(nf, d["dosages"], meta)
    assert no_l.conditional_joint_p is None
    assert "conditional_joint_p" not in no_l.to_dataframe().columns

    # Stand-in local ancestry: use the (admixture-derived) ancestry
    # dosages themselves as a smoke check of the conditional pathway.
    L = d["dosages"].clone()
    res = mdl.score_chunk(nf, d["dosages"], meta, L_chunk=L)
    assert res.conditional_joint_p is not None
    df = res.to_dataframe()
    assert "conditional_joint_p" in df.columns
    import numpy as np
    cp = df["conditional_joint_p"].to_numpy()
    valid = cp[~np.isnan(cp)]
    assert len(valid) > 0
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_conditional_joint_p_binary_raises_not_implemented():
    """family='binary' + L_chunk given must raise NotImplementedError --
    the conditional analysis is gaussian-only (uses null.Py / null.resid,
    which only the continuous null fit populates)."""
    import pytest
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=300, m=6, K=2, seed=4)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(6)], chr=["1"] * 6, pos=list(range(6)),
        a1=["A"] * 6, a2=["G"] * 6,
    )
    mdl = TractorLMM(family="binary", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_bin"], d["X0"], d["K_grm"])
    L = d["dosages"].clone()
    with pytest.raises(NotImplementedError):
        mdl.score_chunk(nf, d["dosages"], meta, L_chunk=L)


def test_conditional_joint_p_bruteforce_matches_formula():
    """Cross-check the conditional stat at one variant against a direct
    (non-vectorized) re-derivation of the brief's formula, using a
    synthetic local-ancestry track independent of the tested dosages so
    the conditional test is exercised on a nontrivial Lv.
    """
    import torch
    from scipy.stats import chi2 as chi2_dist
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=250, m=4, K=2, seed=55)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(4)], chr=["1"] * 4, pos=list(range(4)),
        a1=["A"] * 4, a2=["G"] * 4,
    )
    mdl = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])

    g = torch.Generator().manual_seed(99)
    L = torch.rand(2, 250, 4, generator=g, dtype=torch.float64)

    res = mdl.score_chunk(nf, d["dosages"], meta, L_chunk=L)
    j = 0
    Gv = d["dosages"][:, :, j].T.to(torch.float64)
    T, Var = nf.score_moments(Gv)
    Lv = L[:-1, :, j].T.to(torch.float64)
    PLv = nf.Py(Lv)
    TL = Lv.T @ nf.resid
    LtPL = Lv.T @ PLv
    LtPL_inv = torch.linalg.pinv(LtPL)
    GtPL = Gv.T @ PLv
    cond_mean = GtPL @ (LtPL_inv @ TL)
    cond_var = Var - GtPL @ LtPL_inv @ GtPL.T
    dvec = T - cond_mean
    cond_stat = float(dvec @ torch.linalg.pinv(cond_var) @ dvec)
    rank = int(torch.linalg.matrix_rank(cond_var))
    expected_p = float(chi2_dist.sf(cond_stat, df=rank))

    got_p = float(res.to_dataframe()["conditional_joint_p"].to_numpy()[j])
    assert abs(got_p - expected_p) < 1e-9


def test_tractor_scan_result_concat_preserves_order_and_fields():
    """TractorScanResult.concat stitches per-chunk results back together,
    preserving variant order and every tensor/list field, including
    conditional_joint_p when present on every chunk.
    """
    import torch
    from tests.fixtures.tractor.make_synth import make_synth
    from torchgenomics.models.tractor_lmm import TractorLMM, TractorScanResult
    from torchgenomics.models.base import VariantMeta

    d = make_synth(n=200, m=9, K=2, seed=61)
    meta = VariantMeta(
        snp=[f"v{i}" for i in range(9)], chr=["1"] * 9, pos=list(range(9)),
        a1=["A"] * 9, a2=["G"] * 9,
    )
    mdl = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
    nf = mdl.fit_null(d["y_cont"], d["X0"], d["K_grm"])
    L = d["dosages"].clone()

    chunks = []
    for start in range(0, 9, 4):
        end = min(start + 4, 9)
        sl = slice(start, end)
        cmeta = VariantMeta(
            snp=meta.snp[sl], chr=meta.chr[sl], pos=meta.pos[sl],
            a1=meta.a1[sl], a2=meta.a2[sl],
        )
        chunks.append(
            mdl.score_chunk(nf, d["dosages"][:, :, sl], cmeta, L_chunk=L[:, :, sl])
        )

    merged = TractorScanResult.concat(chunks)
    full = mdl.score_chunk(nf, d["dosages"], meta, L_chunk=L)

    assert merged.snp == full.snp
    assert merged.chr == full.chr
    assert merged.pos == full.pos
    assert merged.ancestry_names == full.ancestry_names
    assert torch.allclose(merged.joint_p, full.joint_p, atol=1e-10, equal_nan=True)
    assert torch.allclose(merged.beta, full.beta, atol=1e-10, equal_nan=True)
    assert torch.allclose(merged.se, full.se, atol=1e-10, equal_nan=True)
    assert torch.allclose(merged.p_anc, full.p_anc, atol=1e-10, equal_nan=True)
    assert merged.conditional_joint_p is not None
    assert torch.allclose(
        merged.conditional_joint_p, full.conditional_joint_p,
        atol=1e-10, equal_nan=True,
    )
