# Courseforge Two-Pass Architecture — Phase Plans

This directory holds the planning artifacts for the Courseforge two-pass /
LLM-agnostic generation rewrite. Each phase has its own plan file authored
by a dedicated planning worker before any execution work runs.

**These plans are tracked in git** (by explicit override of the historical
`plans/` gitignore rule, applied for this work stream) so the design intent
travels with the branch.

## Phase index

| File | Phase | Scope size |
|------|-------|------------|
| `courseforge_architecture_roadmap.md` | **Cross-cutting roadmap (Phases 1-7)** — authoritative architectural framing; the *why* | large |
| `phase1_tos_unblock.md` (+ `_detailed`) | ToS unblock via env-var provider swap | small (DONE) |
| `phase2_intermediate_format.md` (+ `_detailed`) | Stable intermediate format + Block dataclass | medium (DONE) |
| `phase3_two_pass_router.md` (+ `_detailed`) | Two-pass execution + per-block router | medium (~75% landed) |
| (planned) `phase3_5_post_rewrite_validation.md` | Symmetric-validation post-rewrite gate + retry budget bump 3→10 + remediation builder generalization + Phase 3a env-var fixes | small-medium |
| `phase4_statistical_tier.md` | Statistical-tier validators + SHACL wire-up + BERT ensemble + k-reranker | medium |
| `phase5_independent_stages.md` | CLI subcommands + per-block re-execution | small |
| (planned) `phase6_abcd_concept_extractor.md` | ABCD authorship + concept extractor decoupling + verb-level mismatch validator (folded into ABCD) | medium |
| (planned) `phase7_chunker_dual_chunkset.md` | `ed4all-chunker` package + dual-chunkset architecture + LibV2 dual-chunkset manifest | medium |

The cross-cutting roadmap is the authoritative architectural framing
for Phases 1-7; per-phase plans are the implementation contracts.
Phases 3.5, 6, and 7 are introduced by the roadmap and gain detailed
plans authored by a follow-on investigation worker (Worker B in the
parent conversation) running the formal `investigate → plan → execute
→ validate` loop.

## Source context

The audit summary that drove these plans lives in the parent conversation
transcript (4-agent parallel audit covering Courseforge architecture, LLM
provider abstractions, schema/SHACL/CURIE/validator landscape, and
licensing posture). Each plan should re-cite the exact `file:line`
references it depends on so it stays useful even after the conversation
context is gone.
