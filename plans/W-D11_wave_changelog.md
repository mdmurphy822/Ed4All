# W-D11 Wave Changelog — `evidence_quote` + `char_span` per-claim adjudication

- **Wave SHA range**: `92d1bf8` (T11.0) → `b460a78` (T11.6)
- **Branch**: `dev-v0.3.0`
- **Date**: 2026-05-08

## Summary

Wave W-D11 landed cleanly across seven sub-tasks (T11.0 → T11.6), wiring an additive `evidence_quote` + `char_span` projection through the schema, validators, generators, aggregators, and integration-test surfaces. During execution a parallel-agent race produced three commits that all share the same subject line — the literal text of T11.5's subject — but whose actual contents are T11.3, T11.4, and T11.5 respectively. This changelog documents the SHA→content mapping so future readers running `git log --grep` or `git blame` can navigate the wave correctly without rewriting history.

## SHA → content mapping (the three "T11.5"-subject commits)

| SHA | Subject line says | Content actually is | Files |
|-----|-------------------|---------------------|-------|
| `60bc1e9` | "W-D11 T11.5 — aggregators..." | **T11.3** — synthesis-side `evidence_quote` emit + leakage exclusion | 5 generator files (`Trainforge/generators/_anthropic_provider.py`, `_base_synthesis_provider.py`, `_claude_session_provider.py`, `_local_provider.py`, `_together_provider.py`) + 2 new test files (`test_provider_evidence_quote_emit.py`, `test_synthesis_leakage_evidence_quote_excluded.py`) |
| `2d3e6ee` | "W-D11 T11.5 — aggregators..." | **T11.4** — `claim_support` + `pair_claim_support` decision-rationale extension | 2 validator files (`lib/validators/claim_support.py`, `lib/validators/pair/claim_support.py`) + 2 test files (`test_claim_support_evidence_quote.py`, `test_pair_claim_support_evidence_quote.py`) |
| `8e0c372` | "W-D11 T11.5 — aggregators..." | **T11.5** (correct) — `promotion_chain_report` + `trainforge_assessment_quality_report` aggregator wiring + schema bump | 3 source files (`lib/aggregators/promotion_chain_report.py`, `lib/aggregators/trainforge_assessment_quality_report.py`, `schemas/governance/promotion_chain.schema.json`) + 2 test files |

The subject mismatch is cosmetic — every commit's content is internally coherent and validated by its own tests. Only the subject string drifted.

## Sub-task → canonical SHA index (inverse view)

| Sub-task | Canonical SHA | Notes |
|----------|--------------|-------|
| T11.0 | `92d1bf8` | additive `evidence_quote` + `char_span` schema fields |
| T11.1 | `58650de` | `claim_support` evidence_quote + char_span enforcement |
| T11.2 | `5b9445e` | `pair_claim_support` evidence_quote + char_span enforcement |
| T11.3 | `60bc1e9` | (mislabeled subject — see table above) |
| T11.4 | `2d3e6ee` | (mislabeled subject — see table above) |
| T11.5 | `8e0c372` | (subject is correct here) |
| T11.6 | `b460a78` | evidence_quote fixtures + end-to-end integration test |

## Dangling revert (`64ff491`) in the reflog

Mid-wave a worker created a `git revert` of `60bc1e9` that lives in the reflog at `HEAD@{18}`. The revert was wrong — it deleted T11.3's synthesis-side work — and was undone via `git reset --hard HEAD~1` to restore `60bc1e9` as the tip. The revert SHA is now reflog-only and not in linear history. Future operators running `git fsck` may surface it as an orphan; that is expected and harmless.

## Rationale for NOT rebasing

Rewriting the wave to clean up the three mislabeled subjects would re-SHA 14 commits (`60bc1e9` through `7b840ad`). Several already-landed commit bodies cite SHAs that fall inside that range:

- W-D11.D (`4f7b85e`) cites the W-D11 wave SHAs.
- W-D14 (`f28efa7`) cites `285dfec` (W-D12) — IN range.
- W-D15 (`7b840ad`) cites `285dfec` and `f28efa7` — both IN range.
- W-D13 T13.2 (`68bf1a4`) cites `2aeb043` (W-D13 T13.1) — IN range.

(W-D11.E `082a826` cites `58650de` / `5b9445e` and `plans/GPTFEEDBACK3_response.md` cites `2a5556e` / `5235b3c` — those are PRE-range and would survive a rebase intact.)

The rebase introduces more cross-reference rot than the disease (subject mislabel) causes. The mislabel is cosmetic; no code or behavior is wrong. A rebase remains an operator option later if the audit-trail cleanup ever outweighs cross-reference stability — at which point this changelog itself becomes the canonical SHA→content map for the rewrite.

## Follow-up waves landed on top of W-D11

| Wave | SHA | Summary |
|------|-----|---------|
| W-D11.A | `b1d9f18` | pin `trainforge_assessment_quality_report` shape in `schemas/aggregators/`. |
| W-D11.B | `82f8acc` | credit `claim_support` + `pair_claim_support` to instructional + trainable cohorts in `lib/governance/course_status.py`. |
| W-D11.C | `cef3c44` | refresh `docs/validation/gates.md` claim_support rows for evidence_quote enforcement. |
| W-D11.D | `4f7b85e` | note Arrow 2 schema-vs-validation seam in `lib/aggregators/`. |
| W-D11.E | `082a826` | note `evidence_quote_*` signals in `claim_support_check` decision-event rationale fields. |
| W-D11.F | `0730a14` | document license-clean run configuration + opt-in deployment recipe. |
| W-D12 | `285dfec` | generic OpenAI-compatible `LLMBackend` + provider registry on the MCP orchestrator. |
| W-D13 T13.1 | `2aeb043` | vision content-block support in `OpenAICompatibleClient`. |
| W-D13 T13.2 | `68bf1a4` | `DART_PROVIDER` + `DART_VISION_PROVIDER` plumbing + vision-mode OpenAI-compatible backend. |
| W-D14 | `f28efa7` | `COURSEPLANNER_PROVIDER` + course-outliner in-process provider. |
| W-D15 | `7b840ad` | `TRAINFORGE_ASSESSMENT_PROVIDER` + assessment-generator in-process provider. |

W-D11.A through W-D11.F are cleanup waves built on the W-D11 substrate; W-D12 through W-D15 are the license-clean engineering waves that extend per-subsystem provider pinning across DART, courseforge course planning, and Trainforge assessment generation.
