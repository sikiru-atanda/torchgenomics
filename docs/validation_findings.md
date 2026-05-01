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
| 2026-04-30 | A | 1 | `torchgwas.stats.calibrate.compare_pvalues` | self-paired p-vector | synthetic | function raised `NotImplementedError` (Phase 4 stub) | impl required for Tier 1 coverage | V1-core / fix-now | implemented compare_pvalues with `n / mean_abs_diff / max_abs_diff / frac_within_tolerance / corr_neglog10 / mean_abs_diff_neglog10 / tolerance` keys; -log10 corr is NaN-safe; commit on this branch |
| 2026-04-30 | A | 1 | `torchgwas.models.lmm_single` / `lmm_multi` / `lmm_multi_fit` | canonical-path import | n/a | three modules carried Phase 4/5 `NotImplementedError` stubs that diverged from the actual Phase-4/5 implementations in `single_trait_lmm` / `multi_trait_lmm` / `optim.pxem_nr_mvreml` / `optim.lbfgs_reml` | impl required for Tier 1 coverage | V1-core / fix-now | converted `lmm_single` and `lmm_multi` to thin re-export modules and replaced `lmm_multi_fit.fit_mvlmm_null_{ai_reml,lbfgs}` with wrappers around `pxem_nr_mvreml` / `lbfgs_reml` returning a populated `NullFit`; commit on this branch |
