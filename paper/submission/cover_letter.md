# Cover letter — *Genome Biology — Methods (Software)* submission

**To:** Editorial board, *Genome Biology*
**Re:** Submission of "TorchGWAS: a GPU-accelerated, polyploid-first
toolkit unifying variant, haplotype, and multi-omics GWAS at biobank
scale, with end-to-end numerical equivalence to fifteen reference tools"

---

Dear Editors,

We are pleased to submit *TorchGWAS*, an open-source Python/PyTorch
toolkit, to *Genome Biology* (Methods — Software). The work addresses
three concrete gaps in contemporary GWAS infrastructure: the
fragmentation of single-purpose reference tools, the structural
second-class treatment of autopolyploid genomes, and the trade-off
between biobank-scale memory budgets and methodological breadth.

To our knowledge, *TorchGWAS* is the first open-source toolkit to unify
variant-level GWAS (LMM/GLMM/fine-mapping), haplotype-level inference
(13 LD-block-design methods + 9 haplotype GWAS methods), multi-omics
integration (TWAS / SMR / coloc / GRM-corrected causal mediation),
polyploid analysis (dosage calling + F1 phasing + polyploid GWAS at
arbitrary ploidy), and the standard post-GWAS suite (LDSC / MR / PGS /
fine-mapping / TWAS / SMR / HESS) in a single GPU-portable runtime. We
document end-to-end numerical equivalence to fifteen reference tools
(`BOLT-LMM`, `SAIGE`, `regenie`, `GEMMA`, `GAPIT`, `GWASpoly`, `PLINK
2.0`, `LDSC`, `susieR`, `TwoSampleMR`, `MetaXcan` / `S-PrediXcan`, `SMR`
+ `HEIDI`, R `coloc`, R `hyprcoloc`, and `haplo.stats`) under
observed-then-floored tolerances, with 40 of 45 numerical checks
passing.

The differentiators we believe are most relevant for *Genome Biology*
readers are:

1. **Polyploid-first design.** Genotypes are stored as float tensors
   in `[0, k]` for arbitrary ploidy `k`; the gene-action model space
   (additive, `j-dom` for `j ∈ [1, k-1]`, diplo-additive, overdominant)
   is tested per variant. Diploid is `k = 2`. This makes contemporary
   plant- and animal-breeding GWAS first-class instead of having to
   pass through `GWASpoly` or round to diploid.

2. **Biobank-scale streaming on commodity GPUs.** A measured 8.7×
   reduction in peak memory at `p = 10⁶` (n=2000 sample sweep,
   `bench/streaming_p_sweep.json`), with log-log slope 0.020
   (streaming, near-constant) vs 0.933 (materialized, near-linear).
   The streaming contract is gated on every reader by a tracemalloc
   regression net.

3. **Multi-omics integration in one runtime.** The SORT1 cis-eQTL ×
   UKB-style LDL worked example (GTEx v8 liver × GLGC 2021 LDL)
   shows `Pearson(z_eqtl, z_gwas) = -0.97` across 48 harmonized
   variants — the canonical strong-signal locus, processed end-to-end
   in one shell command. Per-tool equivalence is documented against
   `MetaXcan`, `SMR`/`HEIDI`, R `coloc`, R `hyprcoloc`.

4. **Honest disclosure of limitations.** Five post-V1 divergences
   are documented in the public validation-findings ledger
   (`docs/validation_findings.md`): one F2-class silent bug (since
   patched and red-green-verified) and four post-V1 F3 divergences
   (each with a proposed ~10-40-line patch). We chose to leave the
   F3 gates at the spec floors in this submission so reviewers can
   audit the gaps directly rather than have them hidden under
   tolerated tolerances.

5. **Reproducibility from one shell command.** Every figure and
   table in the paper is regenerable via `bash
   paper/reproducibility/reproduce_paper.sh` from a clean checkout.
   The 8-stage pipeline installs the 15 reference tools (each pinned
   to a specific version + SHA256), stages every fixture, runs each
   head-to-head, and renders the 7 main figures into a single
   `manifest.json` carrying every numeric value cited in the paper.

The work has not been submitted elsewhere. A bioRxiv preprint is in
preparation for concurrent submission (priority-date claim only; not
double submission). We confirm that no author has competing interests
that would affect the impartiality of this submission. All software
and data fixtures are open source under the Apache 2.0 license; the
companion reproducibility repository (forthcoming) will carry the
full version-pinned harness suite.

We suggest the following reviewers who have relevant expertise in
mixed-model GWAS, polyploid genetics, and statistical software
benchmarking: [reviewer suggestions to be added before submission].

Thank you for considering this submission. We look forward to the
reviewers' feedback.

Sincerely,

Sikiru Atanda
PulseSmartLab Innovations
sikiruandfriends@gmail.com
(Corresponding author)
