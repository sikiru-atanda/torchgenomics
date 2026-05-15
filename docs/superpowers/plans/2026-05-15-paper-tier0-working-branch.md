# Paper Tier 0 — Working-branch setup implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this small serial plan) or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `paper/genome-biology-methods` working branch from `consolidated/na-roundup` with the design spec cherry-picked, infrastructure verified, and a green smoke pass — so Tiers 1-3 parallel agents have a place to write.

**Architecture:** Use `consolidated/na-roundup` (tip `480cc99`) as the base because it already merges NA1+NA2+NA3+Pillar B+streaming-scan-audit. Cherry-pick the design spec from `modernization/specs`. Skip the legacy App Note draft (`paper/`). Verify infrastructure with `ls`/`find`. Smoke-pass the test suite. All work is local-only — no push without explicit user approval.

**Tech Stack:** git (worktree-friendly), pytest, bash.

---

### Task 1: Pre-flight — confirm bases exist locally

**Files:**
- Read: git refs only

- [ ] **Step 1: Confirm `consolidated/na-roundup` exists and is at commit `480cc99`**

Run: `git rev-parse consolidated/na-roundup`
Expected: prints a commit SHA starting with `480cc99`

- [ ] **Step 2: Confirm `modernization/specs` exists and carries the design spec**

Run: `git show modernization/specs:docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md | head -5`
Expected: prints the spec header `# Genome Biology Methods paper — design spec`

- [ ] **Step 3: Confirm we are not in a dirty state on the current branch**

Run: `git status --short`
Expected: empty output (clean working tree)

---

### Task 2: Cut the working branch

**Files:**
- Create: new branch ref `paper/genome-biology-methods`

- [ ] **Step 1: Create the working branch from `consolidated/na-roundup`**

Run: `git checkout -b paper/genome-biology-methods consolidated/na-roundup`
Expected: `Switched to a new branch 'paper/genome-biology-methods'`

- [ ] **Step 2: Verify branch state**

Run: `git rev-parse --abbrev-ref HEAD && git rev-parse HEAD`
Expected: prints `paper/genome-biology-methods` then a SHA starting with `480cc99`

---

### Task 3: Cherry-pick the design spec onto the working branch

**Files:**
- Create on working branch: `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`

- [ ] **Step 1: Identify the spec's authoring commit**

Run: `git log modernization/specs --oneline -- docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
Expected: lists at minimum three commits (`0a71207` initial spec, `6beb027` pass-1 fixes, `4172878` pass-2 fixes); record the most recent SHA

- [ ] **Step 2: Cherry-pick the latest spec commit**

Run: `git cherry-pick 4172878`
Expected: cherry-pick succeeds; if a conflict appears on an unrelated file, abort with `git cherry-pick --abort` and instead use `git checkout modernization/specs -- docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md && git commit -m "Cherry-pick design spec from modernization/specs@4172878"`

- [ ] **Step 3: Verify spec is present**

Run: `ls -la docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md && head -5 docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
Expected: file exists, first line is `# Genome Biology Methods paper — design spec`

---

### Task 4: Decision — skip the legacy App Note draft

**Files:**
- Create: `paper/README.md` (decision marker)

- [ ] **Step 1: Confirm `paper/` is absent on the working branch**

Run: `ls paper/ 2>&1 | head -3`
Expected: `ls: cannot access 'paper/': No such file or directory`

- [ ] **Step 2: Create the decision marker**

```bash
mkdir -p paper
cat > paper/README.md <<'EOF'
# paper/

Target venue: **Genome Biology — Methods (Software)**.

Authoritative design spec: `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`.

The earlier Bioinformatics Application Note draft (`PLAN.md`, `drafts/abstract.md`) lives on the `research/na1-susie-streaming` worktree and was intentionally NOT cherry-picked here because (a) it targets a different venue with a different word budget, and (b) it under-represented the package's capability surface. The current paper is authored fresh against the GB Methods spec.

Subdirectories under `paper/` are populated by Tier 4 of the implementation plan (`docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md`).
EOF
```

- [ ] **Step 3: Commit the decision**

Run: `git add paper/README.md && git commit -m "Tier 0: paper/ scaffold marker; legacy App Note draft intentionally skipped"`
Expected: commit succeeds

---

### Task 5: Verify infrastructure presence

**Files:** read-only verification

- [ ] **Step 1: Verify the 14 external-tool / fixture directories**

Run: `ls validation/external/ | sort`
Expected output (alphabetical): `_lib bolt_lmm gapit gemma gwaspoly ldsc plink2 regenie saige soymd soynam susieR twosamplemr ukb`

- [ ] **Step 2: Verify native C++ sources (25 files)**

Run: `find csrc/ -name "*.cpp" | wc -l`
Expected: `25`

- [ ] **Step 3: Verify the benchmark + streaming + audit + ledger files**

```bash
ls bench/native_speedups.py
ls tests/test_streaming_memory.py
ls docs/efficiency/streaming_audit.md
ls docs/validation_findings.md
ls validation/external/_lib/preflight.sh
ls .github/workflows/gpu.yml
```
Expected: all six commands exit 0 and print the path

- [ ] **Step 4: Verify the spec is at the correct path on the working branch**

Run: `ls docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
Expected: prints the path

---

### Task 6: Smoke-pass the test suite

**Files:** read-only

- [ ] **Step 1: Confirm pytest is installed in the venv**

Run: `python -c "import pytest; print(pytest.__version__)"`
Expected: prints a version (e.g., `8.x.x`); if `ModuleNotFoundError`, run `pip install -e ".[dev]"` first and re-try

- [ ] **Step 2: Run the smoke pass (excludes the streaming memory regression and external opt-in suites)**

Run: `pytest -x --ignore=tests/test_streaming_memory.py -m "not slow and not external and not cli_matrix and not reproducibility and not gpu" -q`
Expected: exits 0; summary like `N passed, M skipped`

- [ ] **Step 3: Record the smoke-pass result in a tier-0 log**

```bash
cat > docs/superpowers/plans/2026-05-15-paper-tier0-RESULT.md <<EOF
# Tier 0 result log

Working branch: \`paper/genome-biology-methods\`
Base: \`consolidated/na-roundup@$(git rev-parse consolidated/na-roundup)\`

## Smoke pass

\`\`\`
$(pytest -x --ignore=tests/test_streaming_memory.py -m "not slow and not external and not cli_matrix and not reproducibility and not gpu" --tb=no -q 2>&1 | tail -10)
\`\`\`

## Verified infrastructure

- \`validation/external/\` — 14 entries
- \`csrc/\` — 25 .cpp files
- \`bench/native_speedups.py\` — present
- \`tests/test_streaming_memory.py\` — present
- \`docs/efficiency/streaming_audit.md\` — present
- \`docs/validation_findings.md\` — present
- \`validation/external/_lib/preflight.sh\` — present
- \`.github/workflows/gpu.yml\` — present

## Next

Proceed to Plan B (\`docs/superpowers/plans/2026-05-15-paper-tier123-agent-briefs.md\`).
EOF
```

- [ ] **Step 4: Commit the result log**

Run: `git add docs/superpowers/plans/2026-05-15-paper-tier0-RESULT.md && git commit -m "Tier 0: result log + verified infrastructure + smoke pass"`
Expected: commit succeeds

---

### Task 7: Stop — no autonomous push

**Files:** none

- [ ] **Step 1: Confirm we did NOT push**

Run: `git log @{u}..HEAD --oneline 2>/dev/null || echo "no upstream configured for this branch — OK; remains local"`
Expected: either lists the new commits as ahead of upstream, OR prints the "no upstream" line; either way, the branch is local

- [ ] **Step 2: Report to user**

Print the final status:
```
Tier 0 complete.
Working branch: paper/genome-biology-methods (local only).
Smoke pass: PASS.
Infrastructure: verified.
Ready for Plan B (Tiers 1-3 parallel agent dispatch).
No push without explicit approval (per memory/feedback_no_autonomous_push.md).
```

---

## Acceptance for Tier 0

- [ ] Branch `paper/genome-biology-methods` exists locally
- [ ] Design spec present at canonical path on the branch
- [ ] `paper/README.md` decision marker present
- [ ] All 8 infrastructure paths verified (Task 5)
- [ ] Smoke pass green (Task 6)
- [ ] Result log committed
- [ ] No autonomous push
