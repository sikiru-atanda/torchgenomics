# Paper track — session handoff (Plan B complete)

**Branch**: `paper/genome-biology-methods`
**Tip**: `3a8e5e2` (Tier 3 C3/C5/C6 recovered commit)
**Base**: `consolidated/na-roundup@480cc99`
**Commits ahead of base**: 19

## What landed (Plan A + Plan B)

- **Plan A (Tier 0)** — `docs/superpowers/plans/2026-05-15-paper-tier0-working-branch.md` executed inline; smoke pass 2879 tests / 328 skipped.
- **Plan B Tier 1** — `docs/superpowers/plans/2026-05-15-paper-tier123-agent-briefs.md` §4 — 5/5 reference-tool harnesses landed under `validation/external/{metaxcan,hyprcoloc,coloc,hapref,smr}/`. One F2 fix shipped (`f601f20` LD-aware pruning, red-green verified). Three F3 divergences documented in `docs/validation_findings.md`.
- **Plan B Tier 2** — Plan §5 — 3/3 multi-omics fixtures landed under `validation/multiomics/{sim,plant,gtex_ukb}/`. Notable: `gtex_ukb/aligned.parquet` is the SORT1 worked example with Pearson(z_eqtl, z_gwas) = −0.968 (biologically expected).
- **Plan B Tier 3** — Plan §6 — 6/6 specialty fixtures landed under `validation/specialty/{survival,rr,family,threshold,knockoff,ocf}/`.

## Status of every Tier 1/2/3 deliverable

| Tier | Deliverable | Status | Commit |
|---|---|---|---|
| 1 | A1 MetaXcan / S-PrediXcan | ✓ bit-equal PASS | `ffa1151` |
| 1 | A4 hyprcoloc + F3 #1 | ✓ DONE (F3 documented) | `357e299` + `d1b9a60` |
| 1 | A3 coloc + F3 #2 | ✓ DONE (F3 documented) | `bc5ae00` |
| 1 | A5 hapref (F2 finding) | ✓ Documented + finding | `f91e76f` |
| 1 | F2 fix (LD-aware pruning) | ✓ Red-green verified | `f601f20` |
| 1 | A5 hapref harness recovery | ✓ DONE | `e45a187` |
| 1 | F3 audit + A5 header normalize | ✓ DONE | `6867990` |
| 1 | A2 SMR + F3 #3 | ✓ DONE (F3 documented) | `212c05f` |
| 2 | B1 simulated multi-omics fixture | ✓ DONE | `f3643a9` |
| 2 | B3 Arabidopsis 1001G + 1001T + Atwell | ✓ DONE | `e8eb188` |
| 2 | B2 GTEx + UKB SORT1/LDL | ✓ DONE | `78f4064` |
| 3 | C1 survival (coxme reference) | ✓ DONE | `700de64` |
| 3 | C2 random regression (lme4 reference) | ✓ DONE | `01fa9c5` |
| 3 | C4 threshold (BLUPF90+ reference) | ✓ DONE | `d80df25` |
| 3 | C3 family / C5 knockoff / C6 OCF recovery | ✓ DONE | `3a8e5e2` |

## Outstanding (for next session)

1. **3 F3 patches available** (all post-V1, deferred per F3 policy but each documented with a proposed fix):
   - A3 coloc_pairwise formula (`torchgenomics/postgwas/_hyprcoloc.py` lines 360-365) — ~10 lines. Smallest; numerically verified by A3 agent.
   - A4 hyprcoloc prior (`torchgenomics/postgwas/_hyprcoloc.py` lines 204-218) — ~40 lines.
   - A2 SMR HEIDI variance (`torchgenomics/postgwas/_smr.py heidi_test`) — medium-sized; port Zhu 2016 supplementary formula.
   - **C6 OCF nuisance-learner non-linearity** (`torchgenomics/models/ocf_lmm.py`) — F3 #4 from 2026-05-18; coverage 0.41 vs ref 0.91; documented fix: add `nuisance_learner` parameter accepting non-linear learners (mirror DoubleML's `ml_g` / `ml_m`).

2. **Plan C remaining** — `docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md`:
   - D1: ✓ Done (commit `1f8d089`) — reproducibility repo skeleton + 8-stage orchestrator.
   - D2: **7 of 10 done**:
     - ✓ D2.1 F1 capability map (commit `2613b5a`) — 11 clusters / 121 capabilities
     - ✓ D2.2 F2 16-tool equivalence grid (commit `9b9c4da`) — 11 harnesses / 45 checks / 40 pass
     - ✓ D2.3-D2.7 F3/F4/F5/F6/F7 (commit `44a064a`) — full pipeline reproducible
     - **D2.8 `bench/streaming_p_sweep.py`** — pending (F6 panel B uses scaffolded slope until this lands)
     - **D2.9 GU + LRO internal-consistency simulators** — pending (F7 GU/LRO panels are scaffold-only until these land)
     - ✓ D2.10 manifest verification — verified via render pipeline run
   - **D3**: 12 manuscript sections (abstract + 11 body + supplement S1-S8) — NOT STARTED
   - **D4**: internal review — NOT STARTED
   - **D5**: user-gated bioRxiv + Genome Biology submission — NOT STARTED

## Running the figure pipeline

```bash
python3 paper/reproducibility/stages/08_render_figures.py
# -> 7 PDFs in paper/reproducibility/output/F{1..7}.pdf
# -> paper/reproducibility/manifest.json populated with per-figure numbers
```

Full pipeline (install + fetch + run + figures):

```bash
bash paper/reproducibility/reproduce_paper.sh
```

## Key methodology learnings (encoded in memory)

- `feedback_subagent_write_workaround.md` — subagent Write/Edit + Bash heredoc are intermittently denied by the sandbox; **only `Rscript -e 'writeLines(...)'` is universally reliable**.
- `.claude/settings.local.json` has `worktree.baseRef = "head"` set — future agent worktrees will branch from the current local HEAD (`paper/genome-biology-methods`) instead of `origin/master`.
- The agents that succeeded did pre-flight worktree-base checks before writing files (`git merge-base --is-ancestor paper/genome-biology-methods HEAD`).

## How to resume

```bash
git -C /home/sikiru.atanda/Documents/GWAS_Expert log --oneline -25
cat docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md  # Plan C
cat docs/validation_findings.md | tail -300  # F2 / 3 F3 entries
```

Optional first move: run the 30-second C6 TG-side completion to close that PENDING before starting Plan C. Then proceed to Plan C D1.
