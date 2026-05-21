# D4 — Internal spec-alignment review (Tier 4)

**Status:** scaffold authored 2026-05-19; review-pass results to be filled in
*after* D3 sections (00-11) and supplements (S1-S8) land on
`paper/genome-biology-methods` and `paper/manuscript/full_draft.md` is
assembled (D3.14).

**Spec:** `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
**Plan:** `docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md` § 5 D4

This review is a non-optional gate before D5 submission. The reviewer
(main session, not a subagent) walks the assembled draft through four
passes and records findings here. Any open finding blocks submission.

---

## Pass 1 — Spec-alignment (per-claim anchor)

**Rule:** every numerical claim in the manuscript must be backed by:

1. A figure or supplement table that contains the number, AND
2. A `manifest.json` entry produced by `reproduce_paper.sh`, AND
3. A reference-tool harness output that backs it (where applicable; pure
   internal-consistency claims point at the simulator instead).

Any claim missing one of these is logged below and either fixed (claim
corrected) or removed.

### Claim checklist (populated post-assembly)

| # | Claim text (excerpt) | Manuscript section | Figure / table | manifest.json entry | Harness `results/agreement.json` | Status |
|---|---|---|---|---|---|---|
| 1 | "121 capabilities across 11 clusters" | Background + Architecture + Abstract | F1 | `F1.numbers.n_capabilities = 121` + `n_clusters = 11` | — (F1 is hand-curated; no harness) | TODO |
| 2 | "15 reference tools" | Title + Background + Abstract | F2 | `F2.numbers.n_harnesses` (currently 11; need to confirm framing for 15 algorithmic refs) | each of `validation/external/{15 tools}/results/agreement.json` | TODO |
| 3 | "40 of 45 numerical checks PASS" | Background + Equivalence + Abstract | F2 | `F2.numbers.n_pass_checks = 40` + `n_total_checks = 45` | aggregated from 11 harnesses | TODO |
| 4 | "MetaXcan/TWAS bit-equal at \|Δz\|=3.07e-08" | Equivalence + Multi-omics | F2 + F4 | `F2.numbers.per_harness[MetaXcan].checks[0].observed` | `validation/external/metaxcan/results/agreement.json` | TODO |
| 5 | "SORT1 Pearson(z_eqtl,z_gwas) = -0.97" | Multi-omics + Abstract | F4 panel B | `F4.numbers.panels.B.sort1_pearson_z` | `validation/multiomics/gtex_ukb/` | TODO |
| 6 | "8.7× memory-slope reduction at p=1e6" | GPU + streaming + Abstract | F6 panel B | `F6.numbers.panels.B.slope_ratio = 8.71` | `bench/streaming_p_sweep.json` | TODO |
| 7 | "streaming log-log slope 0.020 vs materialized 0.933" | GPU + streaming | F6 panel B | `F6.numbers.panels.B.{streaming,materialized}_log_log_slope` | `bench/streaming_p_sweep.json` | TODO |
| 8 | "25 native C++ kernel speedups 1.5×-9210×" | Architecture + GPU + streaming | F6 panel A | `F6.numbers.panels.A.speedup_range` (renderer reports 24 measured kernels; reconcile vs spec's 25) | `bench/native_speedups.json` | TODO |
| 9 | "Cox PH frailty β Pearson 0.9956" | Specialty | F7 C1 | `F7.numbers.panels.C1.checks[0].observed` | `validation/specialty/survival/results/agreement.json` | TODO |
| 10 | "Random regression β agreement to 10 sig-figs" | Specialty | F7 C2 | `F7.numbers.panels.C2.checks[*]` | `validation/specialty/rr/results/agreement.json` | TODO |
| 11 | "Within-family β_direct agreement 1.73e-10" | Specialty | F7 C3 | `F7.numbers.panels.C3.checks[0]` | `validation/specialty/family/results/agreement.json` | TODO |
| 12 | "OCF coverage 0.41 (TG) vs 0.91 (ref) — F3 #4" | Equivalence + Specialty + Discussion | F7 C6 | `F7.numbers.panels.C6.checks[*]` | `validation/specialty/ocf/results/agreement.json` | TODO |
| 13 | "GU 4/4 internal-consistency gates PASS" | Specialty | F7 GU panel | `F7.numbers.panels.GU.{n_pass,n_total}` | `validation/specialty/gu/results/agreement.json` | TODO |
| 14 | "LRO p_LRO 1.8e-27 vs p_LMM 3.0e-10 (17 orders of magnitude)" | Specialty | F7 LRO panel | `F7.numbers.panels.LRO.checks[2].observed` | `validation/specialty/lro/results/agreement.json` | TODO |
| 15 | "13 LD-block-design methods" | Haplotype | F3 panel A | `F3.numbers.n_ld_block_methods = 13` | `torchgwas.ld` module enumeration | TODO |
| 16 | "9 haplotype GWAS methods" | Haplotype | F3 panel B | `F3.numbers.n_haplotype_gwas_methods = 9` | `torchgwas.models.haplotype_*` enumeration | TODO |
| 17 | "F2 fix in commit f601f20 (LD-aware pruning)" | Haplotype | — | (cite git log) | `tests/test_haplotype_gwas.py::test_ld_aware_pruning_keeps_high_freq_haplotype_under_tight_ld` red-green | TODO |
| 18 | "Arabidopsis 1001G + 1001T + Atwell flowering-time, n=50 (3-way intersection)" | Multi-omics | F4 panel C | `F4.numbers.panels.C.{n_samples,n_snps,n_expr_cols}` | `validation/multiomics/plant/` | TODO |
| 19 | "4 post-V1 F3 divergences, all documented + proposed fixes" | Discussion + Equivalence | — | — | `docs/validation_findings.md` (F3 #1-4) | TODO |
| 20 | "5 documented limitations: Windows GPU, phase-coherence, UKB n=500K, multi-ancestry head-to-head, F3 patches" | Discussion | — | — | — | TODO |

Add rows as the assembled draft surfaces additional numerical claims.

---

## Pass 2 — Capability-coverage review

**Rule:** every cluster in the spec's lead thesis (variant + haplotype +
multi-omics + polyploid + biobank-streaming + reproducibility) must be
surfaced in the assembled draft in three places:

1. The structured Abstract.
2. A dedicated Results section.
3. At least one main figure.

### Cluster checklist

| Cluster | Abstract mention? | Results section | Main figure | Status |
|---|---|---|---|---|
| Variant-level GWAS | ✓ implied by "15 reference tools" | Equivalence (§3) | F2 | TODO |
| Haplotype layer | ✓ "haplotype" in title | Haplotype (§4) | F3 | TODO |
| Multi-omics integration | ✓ Pearson z-vs-z claim | Multi-omics (§5) | F4 | TODO |
| Polyploid pipeline | ✓ "polyploid-first" in title | Polyploid (§6) | F5 | TODO |
| Biobank streaming | ✓ "biobank scale" in title + 8.7× claim | GPU + streaming (§8) | F6 | TODO |
| Specialty models | ✓ implied | Specialty (§7) | F7 | TODO |
| Reproducibility | ✓ "one shell command" claim | Availability (§11) | manifest.json | TODO |

---

## Pass 3 — Tolerance-policy review

**Rule:** every reported agreement number must be observed-then-floored
(matches `results/agreement.json` from the relevant harness; not
aspirational; cushion factor between observed and floor documented).

For each numerical claim cited in the manuscript that references a
tolerance gate, verify:

1. The observed number in the manuscript matches the harness's
   `agreement.json` exactly (no rounding errors, no off-by-one).
2. The gate (threshold) in the harness is at-or-above the observed
   value with a sensible cushion (typically 5-30× observed).
3. The cushion factor is mentioned in the supplement S5 table.

### Tolerance audit (populated post-assembly)

| Claim | Observed (manuscript) | Observed (agreement.json) | Floor (agreement.json) | Cushion | Status |
|---|---|---|---|---|---|
| MetaXcan \|Δz\| max | 3.07e-08 | 3.07e-08 | 1e-06 | 32× | TODO verify match |
| hapref \|Δfreq\| max | 1.14e-4 | 1.14e-4 | 2e-4 | 1.75× | TODO verify match |
| GU stat conservativeness fraction | 1.0 | 1.0 | 0.95 | exceeds | TODO verify match |
| LRO p_LRO at causal | 1.8e-27 | 1.8e-27 | (rank gate, not abs) | rank | TODO verify match |
| ... (extend per Pass 1) | | | | | |

---

## Pass 4 — Novelty-claim review

**Rule:** every "first", "novel", "to our knowledge", or unqualified
superiority claim must:

1. Carry the literal qualifier "to our knowledge" (per repo convention,
   `CLAUDE.md` Repo Conventions).
2. Be substantiated by a concrete claim about the prior literature (a
   citation or a documented absence).

### Novelty audit

| Claim text | Section | Qualifier present? | Substantiated by | Status |
|---|---|---|---|---|
| "first single open-source toolkit to unify variant-level GWAS, haplotype-level inference, and multi-omics integration in one GPU-accelerated runtime" | Background contribution (i) | "to our knowledge" | reviews `regenie`/`SAIGE`/`GWASpoly`/`SuSiE`/`MetaXcan` as single-regime tools | TODO |
| "13 LD-block-design methods (4 classical + 5 novel)" | Haplotype | "to our knowledge" | the 5 novel methods cited as TG-introduced | TODO |
| "Polyploid LD blocks at arbitrary ploidy k" | Polyploid | (verify) | `GWASpoly` does not implement LD-block detection | TODO |
| "GU dosage-variance-corrected score test internal consistency" | Specialty (GU panel) | — (Phase 28 acknowledged as new) | spec §10.6 "no widely-used external reference" | TODO |
| "LRO block-level LOCO with leave-this-block-out GRM" | Specialty (LRO panel) | (verify) | proximal-decontamination not widely implemented | TODO |
| ... | | | | |

---

## Open findings (populated during review)

When a pass surfaces a finding that requires manuscript edits, log it
here with severity:

- **BLOCK** — submission cannot proceed (e.g., claim unsupported by any
  manifest entry; novelty claim unqualified).
- **EDIT** — manuscript text needs revision but the claim is sound
  (e.g., wrong digit in a cited number; missing qualifier).
- **NOTE** — observation worth recording but no action required.

| # | Pass | Severity | Section | Finding | Resolution |
|---|---|---|---|---|---|
| (none yet) | | | | | |

---

## Sign-off

After all four passes are green:

- [ ] Pass 1 spec-alignment: all 20+ claims have anchors
- [ ] Pass 2 capability-coverage: all 7 clusters covered
- [ ] Pass 3 tolerance-policy: every cited number matches `agreement.json`
- [ ] Pass 4 novelty-claim: every claim qualified + substantiated
- [ ] D3.14 assembled full draft passes `wc -w` budget (target 8000-9000)
- [ ] All open findings resolved (zero BLOCK; all EDIT applied)
- [ ] Tag the draft (`git tag -a paper-draft-v1`)

Sign-off requires the reviewer (main session) to mark each box and
commit this file. Only then does D5 unblock.
