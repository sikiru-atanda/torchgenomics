# *Genome Biology* submission checklist

Use this checklist when filling in the journal's online submission
portal. The user runs through it after D4 sign-off; this file is the
artifact, not the submission itself.

## Article type

- [x] **Methods (Software)** — original-paper track for new software
      tools. See: https://genomebiology.biomedcentral.com/submission-guidelines/preparing-your-manuscript/methods
- Word target: ~7000-9000 main text + supplements + figures.
- Required structured-abstract sections: Background / Results /
  Conclusions / Availability and implementation / Contact.

## Manuscript components

- [ ] Title page (title + authors + affiliations + corresponding email)
- [ ] Structured abstract (≤ 250 words) — see `paper/manuscript/sections/00_abstract.md`
- [ ] Background — see `01_background.md`
- [ ] Results sections (Architecture, Equivalence, Haplotype, Multi-omics,
      Polyploid, Specialty, GPU + streaming) — see `02_*` through `08_*`
- [ ] Methods (compressed) — see `09_methods.md`
- [ ] Discussion + limitations + roadmap — see `10_discussion.md`
- [ ] Availability — see `11_availability.md`
- [ ] References (Vancouver style; tooling: pandoc or Manubot to follow
      `paper/manuscript/references.bib` once authored)
- [ ] Figure captions (one per F1-F7) — pull from each renderer's
      docstring
- [ ] Supplementary materials S1-S7 + S8 demo notebooks (future)

## Pre-submission gates

- [ ] D4 review completed (all four passes green; sign-off line in
      `paper/manuscript/D4_review.md` ticked)
- [ ] D3.14 assembled `full_draft.md` exists and `wc -w` is within
      [8000, 9000]
- [ ] `bash paper/reproducibility/reproduce_paper.sh` runs end-to-end on
      a clean checkout (no errors); 7 PDFs + populated `manifest.json`
      produced
- [ ] Every numerical claim in the manuscript has an anchor in the
      manifest (per D4 Pass 1 checklist)
- [ ] Every novelty claim carries "to our knowledge" or equivalent
      qualifier (per D4 Pass 4)
- [ ] License selected (Apache 2.0 — confirm in
      `paper/manuscript/metadata.yaml`)
- [ ] Author list + affiliations finalized (per Open Decision 1 in the
      design spec)
- [ ] Reviewer-suggestions list completed (per Open Decision 2)
- [ ] Bioconda recipe state confirmed (per Open Decision 3) — if not
      ready, remove bioconda mention from Availability section
- [ ] User explicitly authorizes a remote push of the working branch
      (per `memory/feedback_no_autonomous_push.md`)

## Submission portal fields

| Field | Source |
|---|---|
| Title | `metadata.yaml.title` |
| Article type | "Methods" (software subcategory) |
| Subject area | Bioinformatics / Genomics |
| Keywords | per `biorxiv_metadata.yaml` |
| Manuscript file | `paper/manuscript/full_draft.pdf` (export from `full_draft.md` via pandoc / LaTeX) |
| Figures | `paper/reproducibility/output/F1.pdf` ... `F7.pdf` |
| Supplements | `paper/manuscript/supplement/S{1..7}_*.md` (export to PDF) |
| Cover letter | `paper/submission/cover_letter.md` (export to PDF) |
| Author response to reviewers (revisions only) | TBD if applicable |
| Funding statement | per author |
| Competing interests | None declared (per cover letter) |
| Data availability | github.com/sikiru-atanda/torchgwas + reproducibility repo |
| Ethics approval | Not applicable (software paper; no human or animal subjects) |

## Post-submission

- [ ] Save assigned manuscript number
- [ ] Save reviewer assignments (when known)
- [ ] On reviewer feedback, log every requested change in this file
      under "Revisions"

## Revisions (populated post-review)

| Reviewer | Comment | Section affected | Response + manuscript change | Status |
|---|---|---|---|---|
| (none yet) | | | | |
