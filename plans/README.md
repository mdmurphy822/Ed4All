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
| `phase1_tos_unblock.md` | ToS unblock via env-var provider swap | small |
| `phase2_intermediate_format.md` | Stable intermediate format + Block dataclass | medium |
| `phase3_two_pass_router.md` | Two-pass execution + per-block router | medium |
| `phase4_statistical_tier.md` | Statistical-tier validators + SHACL wire-up | medium |
| `phase5_independent_stages.md` | CLI subcommands + per-block re-execution | small |

## Source context

The audit summary that drove these plans lives in the parent conversation
transcript (4-agent parallel audit covering Courseforge architecture, LLM
provider abstractions, schema/SHACL/CURIE/validator landscape, and
licensing posture). Each plan should re-cite the exact `file:line`
references it depends on so it stays useful even after the conversation
context is gone.
