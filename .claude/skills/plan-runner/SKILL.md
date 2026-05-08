---
name: plan-runner
description: Report phase-status of a plans/*.md file by cross-referencing git log. Use when a plan amendment is needed or to gauge progress on a multi-phase plan.
disable-model-invocation: true
---

# Plan Runner

Report the phase status of an Ed4All `plans/*.md` file by cross-referencing
the git commit log. Useful when amending a plan, scoping the next wave, or
deciding whether a phase is actually done.

## Inputs

The user names a plan file (relative or absolute path), e.g.

```
plan-runner plans/rdf-shacl-enrichment-2026-04-26.md
```

If the path doesn't exist, ask the user for clarification. Otherwise proceed.

## Workflow

### 1. Read the plan, extract phase identifiers

```bash
PLAN="plans/rdf-shacl-enrichment-2026-04-26.md"   # whatever was passed in
grep -nE '^#{2,4}.*Phase [0-9]+(\.[0-9]+)?' "$PLAN"
```

Parse out unique `Phase N` and `Phase N.M` identifiers. Preserve order of
first appearance.

### 2. Capture each phase's status keyword from the body

For each phase, scan the lines until the next phase heading:

```bash
grep -niE '\b(shipped|in flight|in progress|deferred|pending|todo|blocked|done|landed|completed)\b'
```

Pick the strongest signal in the phase body (deferred > shipped > in-flight >
pending). If the plan uses checkboxes (`- [x]`), treat them as evidence for
"done".

### 3. Match commits to phases

For each phase, search `git log` with multiple heuristics:

```bash
# By phase number
git log --oneline --all | grep -iE "phase[ _-]?<num>"
# By wave number, if the phase mentions a wave
git log --oneline --all | grep -iE "wave[ _-]?<wave>"
# By a distinctive file path mentioned in the phase body
git log --oneline --all -- <path>
```

Collect all matching commit hashes (deduped, short form).

### 4. Classify status

| Status      | Rule |
|-------------|------|
| `[done]`    | ≥1 matching commit found |
| `[in-flight]` | no matching commit, but plan body says shipped/in progress/landed |
| `[pending]` | no matching commit, no shipped-ish keyword |
| `[deferred]` | plan body explicitly says deferred / out of scope / parked |

A claimed-shipped phase with **no** matching commits is a coherence bug —
flag it explicitly in the Notes column ("plan claims shipped, no commit
found"). Defer the deeper audit to the `plan-coherence-reviewer` subagent.

### 5. Report

Output a markdown table:

```
| Phase | Status | Matching commits | Notes |
|-------|--------|------------------|-------|
| Phase 1 | [done] | 0579d7b, 993cdf2 | Wave 81 enrichment baked in |
| Phase 1.5 | [in-flight] | — | plan says "in progress" |
| Phase 2.5 | [done] | abe155c | |
| Phase 2.6 | [pending] | — | |
| Phase 3 | [deferred] | — | plan body marks "deferred to post-v0.3" |
```

Then a one-line summary:

```
Summary: 2 of 5 phases complete, 1 in-flight, 1 pending, 1 deferred.
```

## Reference example

`plans/rdf-shacl-enrichment-2026-04-26.md` had 7 numbered phases with
sub-phases (Phase 1, 1.5, 2, 2.5, 2.6, 3, 4). The Wave 81 commits
(`fc2675d`, `993cdf2`, `0579d7b`, `07c3f26`) match Phases 1 and 1.5; later
phases were in-flight or deferred at audit time.

## What this skill does NOT do

- Does not amend the plan. Use `plan-coherence-reviewer` to surface bugs and
  let the human (or another agent) author the amendment.
- Does not run pytest. Use the `pytest-targeted` or `corpus-rebuild` skill
  for verification.
- Does not commit anything.
