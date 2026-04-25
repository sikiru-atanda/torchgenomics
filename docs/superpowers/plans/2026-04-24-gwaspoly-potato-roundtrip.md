# GWASpoly Potato Round-Trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin a stable public mirror for a tetraploid potato F1 dataset, capture its SHA-256, and fill out `tests/test_phase_polyorigin_e2e.py::test_gwaspoly_potato_roundtrip` so the roadmap-named integration gate (`dosage-call → phase-poly → HaplotypeGWAS`) actually runs end-to-end against real published data.

**Architecture:** No new modules; everything lands in `tests/test_phase_polyorigin_e2e.py`. The lazy-download cache + sha-verify scaffolding is already in place (`_ensure_potato_dataset` helper). Two constants (`_POTATO_URL`, `_POTATO_SHA256`) need real values, and the test body needs to (1) parse the dataset into the per-individual VCF / dosage tensors that Phase 56's `run_polyorigin` consumes, (2) run the full pipeline, (3) compare Phase 46 haplotype p-values against a bare-`PolyOrigin.polyOrigin()` reference run.

**Tech Stack:** Python 3.10+, PyTorch, pandas, urllib, hashlib. Real Julia 1.11.9 + PolyOrigin v1.0.3 (Tier 2 only — already gated by the file's module-level `pytestmark`). `cyvcf2` if the dataset comes as VCF; `pandas.read_csv` if it ships as a tabular dosage matrix.

**Spec parent:** `docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md` Section 5.2 ("Tier 2 — end-to-end with real PolyOrigin") + Section 5.4 #3 ("`test_gwaspoly_potato_roundtrip` specifically passes — the roadmap's named integration target"). Adapter contract from `2026-04-24-haplotypegwas-shape-adapter-design.md`.

---

## What you inherit from prior sessions

Read this carefully before starting — the environment is unusual.

### Memory (MANDATORY pre-read)

- `~/.claude/projects/C--Users-Sikiru-Documents-GWAS-Expert/memory/MEMORY.md`
- `project_handoff_state.md` — current local repo state, push-policy
- `project_polyploid_phasing.md` — Phase 56 module surface, the four upstream-reality gotchas (state count 100 with double-reduction, 1-based CSV indices, 1=ref/2=alt allele coding, prefer `*_parentphased_corrected.csv`)
- `project_post_phase56_backlog.md` — Tier A #1 and #2 are CLOSED; this plan is for Tier A #3.
- `project_dosage_call.md` — Phase 55 (updog wrapper) details
- `user_sole_contributor.md` — local commits only, no push without explicit instruction

### Current repo state

- HEAD: `6ee4caa` — `Tier A #2: fill recovery calibration helper + unskip recovery tests`. (The plan-write commit will land on top of this.)
- Branch: `master`. Working tree should be clean.
- Tier 1 suite: 53 passed locally. Tier 2 e2e parity + scan + recovery_30x + recovery_8x: passing on this machine when Julia is available.
- Total commits ahead of `origin/master`: ~46. **NOT PUSHED.**

### Confirmed-installed tooling

- Python 3.11 with editable torchgwas (`pip install -e ".[dev,polyploid-phase]"`).
- Julia 1.11.9, juliacall 0.9.31, PolyOrigin v1.0.3 — same as Task 7 of the shape-adapter plan.
- R 4.3.0 + updog 2.1.7 (installed during Tier A #2).
- `juliapkg.executable()` works for Julia discovery.
- Likely missing: `cyvcf2` (it has been spotty — gate on availability and use a pandas fallback if not).

### Key code surface used by this plan

- `torchgwas.preprocess.phase_polyorigin.run_polyorigin(...)` — Phase 56 wrapper.
- `torchgwas.preprocess.phase_polyorigin.PhasingResult` — has `state_table`, `haplotypes_per_copy` (Tier A #1).
- `torchgwas.preprocess.phase_polyorigin._parse_genoprob`, `_parse_parentphased` — for the reference-run parsing (used in `test_parity_tetraploid_f1`).
- `torchgwas.preprocess.dosage_uncertainty.expected_dosage(probs, ploidy)` — `(n,m,k+1) → (n,m)` dosage.
- `torchgwas.models.HaplotypeGWAS(method, test, ploidy)` + `.scan(Y, G, haplotypes=...)` — Phase 46.
- `torchgwas.preprocess.dosage_call.run_updog(...)` — Phase 55 wrapper for VCF → posterior dosage probs (only needed if the published dataset is raw read counts; many published datasets ship pre-called dosages).
- `tests/test_phase_polyorigin_e2e.py::_ensure_potato_dataset` and `_CACHE` — already in place at `tests/fixtures/_cache/` (gitignored).

---

## File Structure

**Modified files:**
- `tests/test_phase_polyorigin_e2e.py` — pin `_POTATO_URL` / `_POTATO_SHA256`; replace the `pytest.skip(...)` body of `test_gwaspoly_potato_roundtrip` with the real pipeline; add a small `_load_potato_dataset` helper specific to whichever published format the chosen dataset ships in.

**New files (likely none):**
- If the dataset is large enough to justify a separate loader, add `tests/_potato_helpers.py`. Otherwise inline.

**Unchanged files:** everything else. No production-code edits.

---

## Task 1 — Choose and pin the dataset

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py` (the `_POTATO_URL` and `_POTATO_SHA256` constants)

This task is the research-y one. Take it carefully — the URL needs to be stable for years, not a one-off blob.

- [ ] **Step 1: Survey candidate datasets via the web**

Three known candidates for a public tetraploid potato F1 dataset:

1. **Bourke et al. 2018** — *PolyOrigin* / *polymapR* companion data; biparental F1 of "Atlantic" × "B1829-5". Often shipped as a tarball on the Wageningen UR mirror or as a Zenodo deposit. Search "Bourke 2018 potato polymapR Zenodo".
2. **Endelman lab GWASpoly companion data** — example dataset shipped with the `GWASpoly` R package itself (file `TableS1.csv` etc.). The R package is on CRAN and GitHub. The example dataset is small and stable.
3. **Massa et al. 2015** / **PotatoMASH** — broader datasets, may not be F1 structured.

Use `WebSearch` and `WebFetch` to find the canonical landing page. Prefer Zenodo DOIs over institutional URLs because Zenodo is forever-stable and SHA-pinned.

Recommended search queries:
- `"polyOrigin" tetraploid potato F1 dataset zenodo`
- `Bourke 2018 polymapR companion data`
- `GWASpoly example dataset tetraploid potato F1`

**Acceptance criteria for the chosen dataset:**
- Tetraploid (ploidy=4).
- F1 design (one or two parents, ≥ 50 progeny is ideal; ≥ 20 is acceptable).
- At least one chromosome's worth of biallelic SNP markers (≥ 100, ideally ≥ 500).
- Ships either (a) per-sample VCF with `AD` field or (b) a posterior-dosage matrix in tabular form.
- Persistent URL (Zenodo DOI, OSF.io DOI, or stable institutional path; **not** a Google Drive link).

- [ ] **Step 2: Download the chosen dataset and capture its SHA-256**

```bash
mkdir -p tests/fixtures/_cache
curl -L "<chosen URL>" -o tests/fixtures/_cache/gwaspoly_potato_f1.tar.gz
sha256sum tests/fixtures/_cache/gwaspoly_potato_f1.tar.gz
```

Note both the size (MB) and SHA-256.

- [ ] **Step 3: Inspect the dataset's internal structure**

Unpack into a scratch directory and document what's inside:

```bash
mkdir -p /tmp/potato-inspect && tar -xzf tests/fixtures/_cache/gwaspoly_potato_f1.tar.gz -C /tmp/potato-inspect
ls -la /tmp/potato-inspect
```

Note:
- Are samples in a VCF, or a flat dosage matrix CSV?
- Pedigree info — is there a separate file naming parents and progeny?
- Marker map — bp positions present? cM map present?
- Phenotype data — is there a Y vector for HaplotypeGWAS to scan against?

This shapes Task 2's loader.

- [ ] **Step 4: Pin the constants in `tests/test_phase_polyorigin_e2e.py`**

Replace:

```python
_POTATO_URL = "PLACEHOLDER — implementer pins before first commit of this test"
_POTATO_SHA256 = "PLACEHOLDER — implementer pins before first commit of this test"
```

with:

```python
# Source: <DOI / paper citation>. Captured 2026-04-24.
_POTATO_URL = "<chosen URL>"
_POTATO_SHA256 = "<sha256 hex>"
```

- [ ] **Step 5: Verify the pinning works via the existing helper**

```bash
pytest tests/test_phase_polyorigin_e2e.py::test_gwaspoly_potato_roundtrip -v
```

Expected: the test will still SKIP (because the body is still `pytest.skip(...)`), but `_ensure_potato_dataset` should succeed silently on its first call — meaning the SHA matched.

- [ ] **Step 6: Commit**

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Tier A #3: pin GWASpoly potato F1 dataset URL + SHA-256

Source: <citation>.
URL: <url>
SHA-256: <hex>"
```

---

## Task 2 — Implement the dataset loader

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py`

The loader converts the unpacked dataset into the inputs `run_polyorigin` and `HaplotypeGWAS` need: posterior dosage probs `(n_total, m, ploidy+1)` + a pedigree TSV + a marker map TSV + a phenotype `Y` if available (else synthetic).

The loader's exact code depends on Task 1 Step 3's findings. Below is the skeleton — fill in the concrete file paths and parsing once you know the dataset's structure.

- [ ] **Step 1: Add `_load_potato_dataset` helper**

Append to `tests/test_phase_polyorigin_e2e.py`:

```python
def _load_potato_dataset(tmp_path: Path) -> tuple[
    torch.Tensor,           # probs (n_total, m, ploidy+1) float64
    list[str],              # sample_ids (parents first)
    list[str],              # variant_ids
    Path,                   # pedigree_tsv path
    Path,                   # map_tsv path
    torch.Tensor,           # Y (n_offspring,) — phenotype or None-equivalent synthetic
    int,                    # ploidy (= 4 for this dataset)
]:
    """Unpack the cached potato dataset and convert it into Phase 56 inputs.

    The cached tarball is at `_CACHE / "gwaspoly_potato_f1.tar.gz"`. Unpacks
    to a scratch directory under `tmp_path`. Reads:
      - <whatever the dataset ships> → probs tensor
      - <pedigree file> → 3-col TSV (offspring, parent1, parent2)
      - <marker map> → 3-col TSV (marker, chrom, pos_bp [, cm])
      - <phenotype if any> → torch tensor Y (or random-seeded synthetic)
    Returns the inputs in the order ``run_polyorigin`` expects.
    """
    import tarfile
    archive = _ensure_potato_dataset()
    unpack_dir = tmp_path / "potato_unpacked"
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(unpack_dir)

    # ----- Concrete parsing — fill in based on Task 1 Step 3 inspection -----
    # Example A: dataset ships per-sample VCF with AD →
    #   from torchgwas.preprocess.dosage_call import run_updog
    #   probs_obj = run_updog(str(unpack_dir / "calls.vcf.gz"),
    #                         str(tmp_path / "dcall"), ploidy=4)
    #   probs = probs_obj["probs"]  (or torch.load if it persisted)
    #   sample_ids = probs_obj["sample_ids"]
    #   variant_ids = probs_obj["variant_ids"]
    #
    # Example B: dataset ships a flat dosage-prob CSV →
    #   df = pd.read_csv(unpack_dir / "dosages.csv")
    #   probs = ... # build (n, m, 5) from columns
    #   sample_ids = list(df.columns)
    #   variant_ids = list(df["marker"])
    #
    # Build pedigree TSV from the dataset's own pedigree file:
    ped = tmp_path / "ped.tsv"
    # ped.write_text("offspring\tparent1\tparent2\n" + ...)

    # Build marker map TSV:
    mp = tmp_path / "map.tsv"
    # mp.write_text("marker\tchrom\tpos_bp\n" + ...)

    # Phenotype (or synthetic):
    # If the dataset has a phenotype column, use it; else build a synthetic Y
    # by drawing N(0,1) noise on top of mean-dosage-of-marker-1.
    Y = ...

    return probs, sample_ids, variant_ids, ped, mp, Y, 4
```

- [ ] **Step 2: Sanity-check the loader interactively**

Before committing, run from the repo root:

```bash
python - <<'PY'
from pathlib import Path
import tempfile, torch
import sys; sys.path.insert(0, "tests")
from test_phase_polyorigin_e2e import _load_potato_dataset

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    probs, sample_ids, variant_ids, ped, mp, Y, ploidy = _load_potato_dataset(tmp)
    print(f"probs shape {tuple(probs.shape)}, dtype {probs.dtype}")
    print(f"n samples={len(sample_ids)}, n markers={len(variant_ids)}, ploidy={ploidy}")
    print(f"Y shape {tuple(Y.shape) if Y is not None else 'None'}")
    print(f"pedigree first 5 lines:\n{ped.read_text()[:200]}")
PY
```

Expected: clean output showing reasonable shapes (e.g. `(60, 1500, 5)` for 2 parents + 58 offspring × 1500 markers × 5 dosage slots).

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Tier A #3: dataset loader for tetraploid potato F1 round-trip"
```

---

## Task 3 — Fill the round-trip test body

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py`

This is the actual integration gate.

- [ ] **Step 1: Replace the `pytest.skip(...)` body**

Find `test_gwaspoly_potato_roundtrip(tmp_path)` and replace its body:

```python
def test_gwaspoly_potato_roundtrip(tmp_path):
    """The roadmap's integration gate.

    dosage-call (or pre-called dosages) → phase-poly → HaplotypeGWAS.
    Compares Phase 46 haplotype p-values from `result.haplotypes_per_copy`
    against a bare PolyOrigin.polyOrigin() reference run on the same
    genofile/pedfile inputs (parsing the reference output via our standalone
    parsers).

    Pass criteria (smoke-level, not power-level):
      - run_polyorigin completes without raising
      - result.haplotypes_per_copy has shape (n_off, ploidy, m)
      - HaplotypeGWAS.scan(Y, G, haplotypes=result.haplotypes_per_copy)
        returns at least one block and all p-values are valid floats in [0, 1]
      - Direct re-parse of the reference workdir's _genoprob.csv and
        _parentphased.csv yields tensors that match result.origin_probs
        and result.parent_phased (atol=1e-4 / exact equality respectively).
    """
    from torchgwas.preprocess.phase_polyorigin import (
        run_polyorigin, _parse_genoprob, _parse_parentphased,
    )
    from torchgwas.preprocess.dosage_uncertainty import expected_dosage
    from torchgwas.models import HaplotypeGWAS

    probs, sample_ids, variant_ids, ped, mp, Y, ploidy = _load_potato_dataset(tmp_path)
    n_off = len(sample_ids) - 2  # parents are first 2

    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    auto = bool(os.environ.get("TORCHGWAS_ALLOW_AUTO_INSTALL"))
    result = run_polyorigin(
        probs=probs,
        pedigree_tsv=str(ped),
        map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=ploidy,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        auto_install_julia=auto,
        keep_workdir=True,
        # leave refinemap default; potato F1 typically benefits from refinement
    )

    # 1. Shape sanity
    assert result.haplotypes_per_copy.shape[1] == ploidy
    assert result.haplotypes_per_copy.shape[0] == n_off
    assert result.haplotypes_per_copy.shape[2] == len(result.variant_ids)

    # 2. Re-parse parity (mirrors test_parity_tetraploid_f1's pattern).
    # PolyOrigin is non-deterministic without a seed parameter, so we re-parse
    # the same workdir CSVs through our standalone parsers and verify they
    # match what run_polyorigin already gave us — the same wrapper-vs-parser
    # parity contract used in Phase 56's parity test.
    workdir = Path(result.workdir)
    ref_origin, _ = _parse_genoprob(
        str(workdir / "out_genoprob.csv"),
        expected_offspring=result.offspring_ids,
        ploidy=ploidy,
    )
    parent_csv = workdir / "out_parentphased_corrected.csv"
    if not parent_csv.exists():
        parent_csv = workdir / "out_parentphased.csv"
    ref_parent, _ = _parse_parentphased(
        str(parent_csv),
        expected_parents=result.parent_ids,
        max_ploidy=ploidy,
    )
    assert torch.allclose(result.origin_probs, ref_origin, atol=1e-4)
    assert torch.equal(result.parent_phased, ref_parent)

    # 3. Feed haplotypes_per_copy into HaplotypeGWAS.scan
    G = expected_dosage(result.postdose_probs, ploidy=ploidy).to(torch.float64)
    # Use 'window' method — block detection on real potato F1 data with
    # hundreds of markers tends to be sparse; window method always
    # produces blocks at the configured size.
    scanner = HaplotypeGWAS(
        method="window", test="f_test", ploidy=ploidy,
        window_size=20, step=10,
    )
    scan_result = scanner.scan(
        Y=Y.to(torch.float64),
        G=G,
        haplotypes=result.haplotypes_per_copy,
    )
    assert len(scan_result.start) > 0, "expected at least one window block"
    for pv in scan_result.p_global.tolist():
        assert isinstance(pv, float)
        assert not math.isnan(pv)
        assert 0.0 <= pv <= 1.0
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_phase_polyorigin_e2e.py::test_gwaspoly_potato_roundtrip -v
```

Expected: PASS. Runtime ~5-15 minutes (dataset unpack + Julia bootstrap + JIT + run + parse + scan).

If it FAILS, the most likely causes:
- Loader produces probs with wrong shape — check `(n_total, m, ploidy+1)` and the dosage slots sum to ~1.
- Pedigree parents aren't both founders — ensure the dataset's parent IDs appear in the `sample_ids` list and not as offspring.
- `_validate_state_table` fires — would mean PolyOrigin's state ordering on this real dataset disagrees with our enumeration. Cross-check Tier A #1 Task 7's parity finding (we already saw 100 states with double-reduction; this test exercises the same enumeration).
- HaplotypeGWAS can't find blocks — switch from `method="block"` to `method="window"` (already in the example above).

Report any failure with the specific traceback before tweaking.

- [ ] **Step 3: Run the full Tier 2 suite to confirm no regressions**

```bash
pytest tests/test_phase_polyorigin_e2e.py -v --tb=short
```

Expected: parity passes, scan-consumes-per-copy passes, recovery_30x + recovery_8x pass, potato_roundtrip passes. Total: 5 passing.

- [ ] **Step 4: Ruff check + commit**

```bash
ruff check tests/test_phase_polyorigin_e2e.py
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Tier A #3: GWASpoly potato F1 round-trip integration test"
```

---

## Task 4 — Close-out: backlog + memory

**Files:**
- Modify: `~/.claude/projects/C--Users-Sikiru-Documents-GWAS-Expert/memory/project_post_phase56_backlog.md`
- Modify: `~/.claude/projects/C--Users-Sikiru-Documents-GWAS-Expert/memory/project_polyploid_phasing.md`
- Modify: `~/.claude/projects/C--Users-Sikiru-Documents-GWAS-Expert/memory/MEMORY.md`

These memory files live OUTSIDE the repo. Edits are NOT committed.

- [ ] **Step 1: Mark Tier A #3 closed in `project_post_phase56_backlog.md`**

Find the bullet `3. **Pin GWASpoly potato URL + SHA.** ...`. Strike it through and add a CLOSED marker:

```markdown
3. **~~Pin GWASpoly potato URL + SHA.~~** **CLOSED 2026-04-24** (commits ... ). Pinned to `<URL>` (`<sha256>`). `test_gwaspoly_potato_roundtrip` exercises the full chain (probs → run_polyorigin → HaplotypeGWAS) on the published dataset and validates wrapper-vs-parser parity, shape compatibility with Phase 46, and valid p-values in [0, 1]. ~10 min runtime on a Julia-equipped machine.
```

Update the "Recommended next action" footer:

```markdown
### Recommended next action

All Tier A items now CLOSED. **Next session: pick a Tier C candidate phase brainstorm** (Phase 50 GWAS-by-imputation, 51 sex-chromosome, 52 admixture-aware LMM, 53 rare-variant beyond SKAT, or 54 R wrapper to CRAN). Or address Tier B (CI billing block + push) if that's now resolved.
```

- [ ] **Step 2: Update `project_polyploid_phasing.md`**

Append a one-line note under the Tier-A #1 close-out section: "Tier A #3 (GWASpoly potato F1 round-trip integration gate) closed 2026-04-XX — `test_gwaspoly_potato_roundtrip` now runs end-to-end against the published dataset."

- [ ] **Step 3: Update `MEMORY.md`**

Find the line pointing to `project_post_phase56_backlog.md` and update its description:

```markdown
- [Post-Phase-56 Action Backlog](project_post_phase56_backlog.md) — All Tier A items CLOSED (shape adapter, calibration fill, potato roundtrip). Remaining: Tier B (CI billing + push), Tier C (Phases 50–54 candidates), Tier D (maintenance).
```

- [ ] **Step 4: Update the handoff state**

Open `project_handoff_state.md`. Update the "Recommended next action" near the bottom to reflect that Tier A is fully closed; suggest a Tier C brainstorm or Tier B push.

---

## Self-review notes

**Spec coverage check** (against the original Phase 56 spec):

- Section 5.4 #3 ("`test_gwaspoly_potato_roundtrip` specifically passes — the roadmap's named integration target") → Task 3 ✓
- Section 5.2 (Tier 2 e2e gating) → already in place from prior work ✓
- Section 5.5 (test fixture footprint, lazy-download cache) → already in place ✓

**Type / signature consistency**:

- `_load_potato_dataset(tmp_path) -> 7-tuple` matches the kwargs `run_polyorigin` consumes.
- `result.haplotypes_per_copy` shape `(n_off, ploidy, m)` matches `HaplotypeGWAS.scan(haplotypes=...)` contract.

**Placeholder scan**:

- `<chosen URL>` and `<sha256 hex>` and `<DOI / paper citation>` in Task 1 are placeholders the implementer fills with real values. They are NOT plan failures — the research IS the work of Task 1.
- `_load_potato_dataset` body has `# Example A` / `# Example B` / `...` placeholders that are filled in based on the actual dataset's structure (Task 1 Step 3 inspection). The full code can't be written until the dataset is chosen.

**Risk register** (things that may complicate execution):

1. **No suitable public mirror** — if all candidate datasets fail the acceptance criteria (especially "persistent URL"), the plan stalls. Fallback: ship a small synthetic F1 simulation as a tarball in `tests/fixtures/` (≤ 50 KB) and pin THAT instead. The integration gate becomes "synthetic F1, not real published data" — weaker but still useful.
2. **Dataset is too large** (> 100 MB) — `_CACHE` is gitignored so it doesn't bloat the repo, but slow CI / first-runtime experience. Consider a smaller subset of the published data or a chromosome-1-only filter.
3. **Pedigree structure mismatch** — published datasets sometimes have multi-generational structure or unrelated controls mixed in. The loader needs to filter to a clean F1 subset; document this in the loader's docstring.
4. **PolyOrigin runtime > 15 min on real-size data** — if the test takes > 15 min, mark it `pytest.mark.slow` and add an env-var gate so CI can skip it by default; full runs still happen at release time on a Julia-equipped machine.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-24-gwaspoly-potato-roundtrip.md`. Next agent: pick this up via `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.
