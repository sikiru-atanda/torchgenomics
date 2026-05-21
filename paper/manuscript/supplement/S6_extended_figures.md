# S6 — Extended figures

Each main-text figure (F1-F7) has a corresponding supplementary panel inventory enumerated below. Live panels are renderable today from the manifest + raw `agreement.json` artefacts via `paper/reproducibility/stages/08_render_figures.py`; scaffold panels are wired in the renderer but draw from placeholder data because their upstream fixture is gated on a still-blocking infra step (GPU CI artefacts, SAIGE docker on RHEL 9, PolyOrigin Julia subprocess, etc.).

## F1 extended panels

F1 is the capability map (11 clusters / 121 capabilities). No supplementary breakdown beyond the full enumeration in S2 (the capability inventory); F1 itself does not subdivide.

## F2 extended panels

F2 is the reference-tool equivalence grid (11 harnesses, 45 checks, 40 passing). The full per-check ledger is S1; no extra panels — every cell of F2 is enumerated in S1.B.

## F3 extended panels (haplotype-method internals)

F3 panels A / B / C cover LD blocks (13 methods), haplotype GWAS methods (9 methods), and one window-level worked example. Supplementary expansions S3.1-S3.9 will hold per-method internals at extended scale.

| Panel | Method | Status | Data source |
|---|---|---|---|
| S3.1 | LD block per-method clustering (4 classical + 5 novel + 3 literature + 1 diagnostic, against PLINK 1.9 `--blocks`) | scaffold (F3 panel A scaffold_only=true in manifest) | `validation/external/plink2/results/agreement.json` |
| S3.2 | Haplotype HTR-block (F-test) internals at extended n | live (1 concrete row at n=279 MDP) | `validation/external/hapref/results/agreement.json` |
| S3.3-S3.9 | HTR-block (LRT), HTR-window (F-test), SKAT-block, PCHT, HHCT, HSKAT, HapGxE, BayesHap | scaffold (per-method internals pending SoyMD-anchored harness build-out) | manifest `F3.numbers.panels.B.scaffold_pending` |

## F4 extended panels (multi-omics internals at extended n)

F4 panels A / B / C cover simulated multi-omics (n=50/m=50 mediators), GTEx + UKB SORT1/LDL (n=48 variants), and a plant fixture (n=50/m=100 SNPs/20 expr cols).

| Panel | Topic | Status | Data source |
|---|---|---|---|
| S4.1 | TWAS sensitivity to z-score scale; per-gene Pearson r vs MetaXcan at extended panel | live (5 gene; 4 metrics) | `validation/external/metaxcan/results/agreement.json` |
| S4.2 | SMR + HEIDI HEIDI-statistic divergence audit (5 vs 6 flanking SNPs) | live | `validation/external/smr/results/agreement.json` |
| S4.3 | coloc per-scenario PP breakdown (shared / distinct / null) | live | `validation/external/coloc/results/agreement.json` |
| S4.4 | hyprcoloc per-trait cluster-membership stability | live | `validation/external/hyprcoloc/results/agreement.json` |
| S4.5 | Mediation recovery at extended n (causal mediator ID; SoyMD anchor) | scaffold pending TG-side recovery run | `validation/multiomics/sim/fixtures/truth.json` |
| S4.6 | Multi-kernel h2 partitioning vs LDSC h2 anchor | scaffold | `validation/external/ldsc/` |

## F5 extended panels (per-ploidy parity at k=4, 6, 8 + simulated hexaploid/octoploid)

F5 panels A / B / C / D cover updog dosage calling, PolyOrigin F1 phasing, GWASpoly per-model -log10p, and a k-sweep at k=4/6/8.

| Panel | Topic | Status | Data source |
|---|---|---|---|
| S5.1 | Dosage call accuracy at k=4 (HW posterior surrogate vs simulator truth) | scaffold (Phase 55 updog wrapper gated on R+updog) | manifest `F5.numbers.panels.A` |
| S5.2 | PolyOrigin F1 phasing on simulated potato pedigree | scaffold (PolyOrigin Julia wrapper gated; Phase 56 placeholders) | manifest `F5.numbers.panels.B` |
| S5.3 | GWASpoly per-gene-action -log10p scatter at k=4 (8 models) | scaffold (gwaspoly data not staged on the paper branch) | manifest `F5.numbers.panels.C` (correlation_logp 0.9983) |
| S5.4 | k-sweep Manhattan at k=4 / 6 / 8 (synthesised at render-time) | live | manifest `F5.numbers.panels.D` |
| S5.5 | Per-ploidy GRM eigenvalue spectrum at k=4 / 6 / 8 | scaffold | (k-aware `linalg.grm` extension) |

## F6 extended panels (GPU + streaming)

F6 panels A / B / C / D cover the 24-kernel speedup table, the streaming-vs-materialised peak-memory p-sweep, the n=500K UKB scan speedup, and the GPU parity scatter.

| Panel | Topic | Status | Data source |
|---|---|---|---|
| S6.1 | Per-kernel speedup at realistic vs micro sizes | live | `bench/native_speedups_realistic.md` |
| S6.2 | Streaming peak memory at p in {1e4, 3e4, 1e5, 3e5, 1e6}, n=2000, chunk=1024 | live (ratio 8.71x at p=1e6) | `bench/streaming_p_sweep.json` |
| S6.3 | Biobank-scale GPU speedup (n=500K, chr22) | scaffold (NA3 UKB GPU-node run pending) | manifest `F6.numbers.panels.C` |
| S6.4 | GPU parity scatter (68 GPU-marked tests across 4 test files) | scaffold (GPU CI JUnit XML not committed) | manifest `F6.numbers.panels.D` |

## F7 extended panels (specialty model internals)

F7 has 6 external-reference panels (C1-C6) + 2 internal-only panels (GU, LRO). All 8 are enumerated in S1.B / S1.C with their full per-check tables; no further breakdown is required for F7.

## Status summary

Per the manifest at HEAD: F3 panel A is scaffold; F3 panel B has 1 of 9 concrete rows; F4 panels A-C are live (with mediation recovery pending); F5 has 3 of 4 panels scaffold + 1 live (the k-sweep); F6 panels A and B are live; F6 panels C and D are scaffold pending UKB GPU + GPU CI artefacts; F7 is all live (6 external + 2 internal). Scaffold panels are explicitly labelled in `paper/reproducibility/manifest.json` with `scaffold: true` and a `scaffold_reason` string.
