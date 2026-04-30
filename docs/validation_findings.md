# TorchGWAS Validation Findings Ledger

Append-only ledger of every divergence found during the validation
campaign. Schema and policy: spec §11 + §9.

Classifications per F3:
- **V1-core / fix-now** — Phases 0–13 math error; commit fix in same tier.
- **V1-core / regression** — golden test that previously passed; halt pillar.
- **post-V1 / documented** — Phase ≥14 divergence; xfail with reason.
- **post-V1 / open issue** — divergence exposes missing-feature charter claim.
- **infra-blocker** — install / data / env failure; documented; comparison skipped.

| Date | Pillar | Tier | Module / Function | Reference | Dataset | Δ observed | Tolerance | F3 class | Resolution |
|------|--------|------|-------------------|-----------|---------|------------|-----------|----------|------------|
| _no entries yet_ | | | | | | | | | |
