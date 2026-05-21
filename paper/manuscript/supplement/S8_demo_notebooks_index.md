# S8 — Demo notebooks index (placeholder)

This supplement is a placeholder index for the future per-figure demonstration notebooks under `paper/manuscript/supplement/notebooks/F{1..7}_demo.ipynb`. The notebooks are out of scope for the current D3.13 dispatch (which authors only S1-S7 plus this index); they are scaffolded here so reviewers and downstream consumers know where the future end-to-end demos will live and what they will do.

## Future location

```
paper/manuscript/supplement/notebooks/
  F1_demo.ipynb   # Capability map; renders F1 from manifest
  F2_demo.ipynb   # Reference-tool equivalence grid; renders F2 from agreement.json artefacts
  F3_demo.ipynb   # Haplotype + LD blocks; renders F3 from plink2 + hapref + soymd
  F4_demo.ipynb   # Multi-omics; renders F4 from sim + GTEx/UKB + plant fixtures
  F5_demo.ipynb   # Polyploid; renders F5 panels A/B/C/D end-to-end
  F6_demo.ipynb   # GPU + streaming; renders F6 from bench artefacts + GPU CI XML
  F7_demo.ipynb   # Specialty models; renders F7 from validation/specialty agreement.json
```

## What each demo will do

Each notebook will be a single end-to-end pipeline matching the figure it backs:

1. `pip install torchgwas` (no editable install required; targets the published package).
2. Fetch the staged fixture for the figure from a pinned URL (each fixture has a SHA-256 captured at paper-submission time; the notebook verifies before consuming).
3. Run TorchGWAS end-to-end on the fixture (every CLI subcommand or programmatic API needed for the figure).
4. Render the figure inline (matplotlib-only, identical output to the paper-figure PDF).
5. Print the manifest entry that corresponds to the rendered figure, so a reviewer can verify the live numbers match the published cell.

## Why scaffold only

The notebook scaffolds will be created in a follow-up dispatch (D4 or later) after the figure-rendering driver is feature-frozen on the paper branch. The driver itself (`paper/reproducibility/stages/08_render_figures.py`) already provides the same end-to-end pipeline today, with the same fixture-staging guarantee — the notebooks will be a presentation layer atop that driver. Until then, reviewers should run:

```bash
bash paper/reproducibility/reproduce_paper.sh
```

and inspect the regenerated figures under `paper/reproducibility/output/F{1..7}.pdf`. Per-figure agreement values are in `paper/reproducibility/manifest.json` (the same manifest the figures and S1 / S2 / S6 are computed from).

## Status

**Future work, scaffolded.** No notebooks are committed at the paper branch HEAD. The figure-rendering driver in `paper/reproducibility/stages/08_render_figures.py` provides the same end-to-end pipeline today; the demo notebooks are a future presentation layer that wraps it for reviewer convenience.
