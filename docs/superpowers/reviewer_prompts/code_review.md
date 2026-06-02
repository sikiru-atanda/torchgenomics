# Code-Review Reviewer Prompt — Pillar {{PILLAR}} Tier {{TIER}}

You are reviewing tier {{TIER}} of pillar {{PILLAR}} of the TorchGenomics validation campaign.

## What was just done

The campaign spec is at `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`. The pillar plan is at `docs/superpowers/plans/2026-04-30-pillar-{{PILLAR}}-plan.md`.

This tier's bar (from §4.1 of the spec):

{{TIER_BAR_QUOTE}}

The diff under review is:

```bash
git diff {{TIER_START_SHA}}..HEAD -- tests/test_coverage_*.py scripts/ pyproject.toml
```

## What I want you to verify

1. **Bar conformance** — every new test in this tier meets the tier's bar. Tier 1 tests must verify math correctness (closed-form / scipy / Monte-Carlo). Tier 2 tests must cover golden + edge + error. Tier 3 tests must be a happy-path smoke check.
2. **Tolerance contract** — tolerances in new tests match `docs/validation.md` Section 16 where applicable. New tolerances introduced are justified (in a comment) by an observed-then-floored measurement, not aspirational.
3. **F3 severity classification** — every commit that fixes production code is correctly classified per the F3 table in §9 of the spec. V1-core fixes are real; post-V1 should be xfail'd or ledgered, not silently fixed.
4. **Test isolation** — tests do not depend on global state; fixtures use `conftest_coverage.py`; no committed-data dependencies beyond what the spec allows.
5. **Findings ledger** — every divergence found is a row in `docs/validation_findings.md` with the documented schema.

## What I do NOT want you to do

- Do not propose refactors to TorchGenomics production code beyond what F3 already drove.
- Do not propose adding test cases beyond the tier's bar.
- Do not propose new dependencies.

## Output

Return three sections:

- **Blockers** — issues that must be fixed before the tier closes.
- **Nits** — non-blocking improvements.
- **Out-of-scope** — items that should be deferred to later pillars.

Under 500 words.
