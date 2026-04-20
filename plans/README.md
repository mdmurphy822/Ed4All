# `plans/` — durable planning artifacts

This directory holds medium-to-long-lived planning artifacts: evaluations, reviews, synthesis documents, and implementation plans that feed upcoming work but are not themselves user-facing reference documentation or pipeline runtime state.

## When to use `plans/` vs `docs/` vs `state/`

| Directory | Purpose | Audience | Lifetime |
|---|---|---|---|
| `plans/` | Forward-looking evaluations, proposals, multi-worker syntheses; inputs to future implementation sessions | Planners, reviewers, future workers | Medium-long (retained for audit trail even after superseded) |
| `docs/` | User-facing reference — how to use the system, validation reports that are authoritative, architectural decision records | Operators, contributors, external readers | Long (evergreen or versioned) |
| `state/` | Pipeline runtime data — progress files, locks, per-run captures, status tracking | The orchestrator + agents at runtime | Short (per-run / per-batch) |

Rule of thumb: if the document answers "what should we do next and why," it belongs in `plans/`. If it answers "what is the shape of X today" as durable reference, it belongs in `docs/`. If it answers "what is the current pipeline run doing right now," it belongs in `state/`.

## Layout pattern

Each planning effort is its own kebab-cased, date-stamped sub-directory:

```
plans/
├── README.md                           ← this file (convention + active-effort index)
└── <topic>-YYYY-MM/                    ← one directory per effort
    ├── README.md                       ← effort overview: goal, timeline, workers, status, deliverables
    ├── discovery/                      ← optional: per-worker discovery artifacts
    │   ├── a-<slug>.md
    │   ├── b-<slug>.md
    │   └── …
    └── review.md                       ← the synthesis (or plan.md for implementation plans)
```

Effort-level `README.md` should always include: one-sentence goal, start date, status (`planning` | `in-progress` | `complete` | `superseded`), workers/authors, list of deliverables with links, and a next-step pointer (where the output feeds).

## Naming conventions

- Top-level effort dirs: kebab-case topic + `YYYY-MM` month stamp (e.g. `kg-quality-review-2026-04/`, `v0.3-emit-hardening-2026-05/`).
- Per-worker discovery artifacts: `<letter>-<slug>.md` where `<letter>` is the worker id used in the effort plan (A, B, C, …) and `<slug>` is the scope.
- The final synthesis is named `review.md` for evaluative efforts or `plan.md` for implementation plans.
- Ephemeral scratch (inline notes, transient drafts) must not live at repo root — nest under the effort directory (e.g. `<topic>-YYYY-MM/notes.md`) or discard after promotion.

## Gitignore policy

`plans/` is committed by default. Individual efforts may choose to gitignore `discovery/` (keeping worker artifacts local-only) via a per-directory `.gitignore`, but the effort `README.md` and final `review.md` / `plan.md` are committed to serve as an audit trail.

For the initial effort (`kg-quality-review-2026-04/`), all artifacts — including `discovery/*.md` — are committed, because the worker-level evidence supports the synthesis citations.

## Active efforts

| Effort | Status | Description | Entry point |
|---|---|---|---|
| [`kg-quality-review-2026-04/`](./kg-quality-review-2026-04/) | complete | KG-quality evaluation of the unified ontology + cross-system contracts on `dev-v0.2.0` (post-Workers R/S/T/U). 5 discovery workers + 1 synthesis worker. | [`kg-quality-review-2026-04/README.md`](./kg-quality-review-2026-04/README.md) |

## Superseded / archived

_None yet._ When an effort is superseded by a newer plan or its recommendations have landed, add it to this section with a pointer to the follow-up effort.

## Relationship to the master `/home/mdmur/.claude/plans/` tree

Claude's per-session planning tree (`~/.claude/plans/`) holds the orchestration plan for a specific session (which workers run, in what order, with what constraints). The repo-level `plans/` tree holds the deliverables those sessions produce. They are complementary: the session plan names `plans/<effort>/review.md` as its final output; the repo-level tree hosts that output durably.
