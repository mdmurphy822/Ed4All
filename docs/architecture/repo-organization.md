# Repo Organization Schema

**Status: ADOPTED** — Phase 1 landed 2026-07-29 (this document, the four-bucket
`docs/` taxonomy, `ci/layout_guard.py`); the runtime-collapse phase landed the
same day (owner decision): `state/`, `training-captures/`, `seats/`, `demo/`,
`extracted/`, `testruns/`, `scratchpad/`, and `shots/` all live under the
single gitignored `runtime/` root, with compat symlinks at `state`,
`training-captures`, and `seats` until the paused run finishes and operator
env files are updated. Phase 2 (`scripts/` re-taxonomy) is pending; the
import-root moves stay **rejected** (§5). This document is the placement authority: when a new file or
directory doesn't obviously fit a rule below, that is a design question, not a
formatting one.

## 1. Diagnosis (surveyed 2026-07-29, dev-v0.4.0, ~3,000 tracked files)

- **24 top-level dirs** in four unlabeled roles: packaged platform code
  (`lib/ MCP/ cli/ gui/` — exactly the `pyproject` `packages.find.include`
  set), subsystem products (`SemantiK/ Courseforge/ Trainforge/ LibV2/`),
  contracts/infra (`config/ schemas/ ci/ seats/ scripts/ docs/ tests/`), and
  gitignored data roots (`state/ runtime/ inputs/ training-captures/
  extracted/ testruns/ scratchpad/ plans/ demo/`). The roles exist; nothing
  declared them, so new dirs landed by vibes.
- **`scripts/` is a junk drawer**: ~23 loose top-level entries spanning three
  lifetimes — durable operator entry points, reusable measurement harnesses,
  and one-shot pilots — while the good precedent (`archive/`, `codegen/`,
  `integration/`, `tests/`) already exists underneath.
- **`lib/` mixes ~30 flat modules with 20+ subpackages.** TECH_DEBT D19
  documents the sanctioned migration pattern (subpackage +
  `PendingDeprecationWarning` shim); the rule below stops *new* flat modules.
- **`docs/` had six single-file dirs** beside three real ones (collapsed in
  Phase 1).
- **The data roots are load-bearing**: `.gitignore` encodes a defensive
  contract (`state/*/*` catch-all, `.gitkeep` re-includes, weight/pair
  globs), MCP's write sandbox is path-confined to `runtime/` + `state/`, and
  `ED4ALL_HOME` (`lib/paths.py`) already relocates the data roots by
  basename. Renaming them touches sandbox + gitignore + checkpoints + ~55
  modules simultaneously — hence §5.

## 2. The schema: four zones, one allowlist

Every top-level path belongs to exactly one zone. **The top level is closed**:
adding a top-level dir requires editing `ci/layout_allowlist.txt` — that diff
is the design review.

| Zone | Dirs | Invariant |
|---|---|---|
| **CODE-PLATFORM** | `lib/ MCP/ cli/ gui/` | Ships in the wheel. Importable, tested, no data. TitleCase forbidden except `MCP`. |
| **CODE-SUBSYSTEM** | `SemantiK/ Courseforge/ Trainforge/ LibV2/` | One product each, own `CLAUDE.md`, own `tests/fixtures/`. TitleCase names are **reserved** for this zone. |
| **CONTRACTS & INFRA** | `config/ schemas/ ci/ seats/ scripts/ docs/ tests/` + root deploy files (`Dockerfile*`, `docker-compose*`, `Makefile`, `pyproject.toml`, `run-gui.*`) | Tracked, hand-authored, no generated data. |
| **VAR (gitignored)** | `runtime/ inputs/ plans/` | `runtime/` holds ALL mutable data (`state/`, `training-captures/`, `seats/`, `demo/`, `extracted/`, `testruns/`, `scratchpad/`, `shots/`, ...). Only `.gitkeep` sentinels tracked. Nothing here is ever a git dependency of the build. |

Per-dir placement rules (purpose / belongs / **never**):

- `lib/` — cross-cutting Python. New code goes in a **subpackage**
  (`lib/<topic>/`); a new flat `lib/*.py` is a guard violation (ratchet frozen
  at today's set). Never: subsystem-specific logic (belongs in the subsystem),
  scripts whose primary purpose is `__main__`.
- `MCP/` — orchestrator + tool surface. Never: validators (→
  `lib/validators/`), schemas.
- `cli/`, `gui/` — thin surfaces over `lib`/`MCP`. Never: business logic
  reachable nowhere else.
- Subsystem dirs — everything for that product incl. its `tests/fixtures/`.
  Never: another subsystem's fixtures (root `CLAUDE.md` § Test fixtures
  stands), imports from a sibling subsystem's private internals.
- `config/` — orchestration YAML + constraint files. Never: per-machine values
  (those are `seats/*.sh` or `run-env.*.sh`, both gitignored).
- `schemas/` — JSON Schema / SHACL / taxonomies +
  `schemas/tests/fixtures/<wave>/`. Never: generated instance data.
- `ci/` — repo guards + their tests + allowlists. Never: runbooks (→
  `docs/operations/`).
- `runtime/seats/` — gitignored real seat scripts; the tracked reference is
  `docs/operations/seat-scripts.md` + `docs/operations/launch-seat.example.sh`.
- `scripts/` — see §3. Never: anything imported by shipped code paths (if
  `lib`/`MCP` needs it at runtime, it graduates into the package).
- `docs/` — see §4. Never: machine-specific values or gitignored-local-path
  references (existing hygiene contract; an intrinsically operator-local
  runbook goes untracked, like `spark-profile.md` / `dgx-spark.md`),
  generated manifests (`docs/MANIFEST.md` stays gitignored).
- `tests/` — cross-project integration only; single-subsystem tests live with
  the subsystem (unchanged).
- `runtime/` — the single gitignored data root: `runtime/state/` (workflow
  state, checkpoints, caches), `runtime/training-captures/`,
  `runtime/seats/`, `runtime/demo/`, `runtime/extracted/`,
  `runtime/testruns/`, `runtime/scratchpad/`, `runtime/shots/`. `inputs/`
  stays a sibling (corpus staging, `ED4ALL_HOME`-adjacent). Never: tracked
  files other than `.gitkeep`; never new top-level siblings — an operator
  experiment that needs a dir gets `runtime/<name>/`, never a new root.
  Under `ED4ALL_HOME` the relocated basenames are unchanged (`state`,
  `training-captures`, ...) — the `runtime/` nesting is the in-repo default
  layout only (`lib/paths.py`).
- `plans/` — local-only, unchanged.

**Naming conventions.** Dirs: lowercase (kebab or snake, match siblings);
TitleCase only for subsystem products. Docs: kebab-case `.md`. Env flags: the
one-owner-per-prefix contract (root `CLAUDE.md` § Opt-In Behavior Flags)
unchanged. One-off scripts: date- or wave-stamped and born in
`scripts/archive/` or `runtime/` — a script whose name contains `pilot`/`ab`/
`wave` does not belong at `scripts/` root.

## 3. `scripts/` taxonomy (Phase 2)

```
scripts/
  ops/        # durable operator entry points, documented in docs/operations/
              # (bootstrap-training-env.sh, ed4all-training, gpu_guard.sh,
              #  mailbox_servicer.py, repair_partial_resume_state.py, ...)
  harness/    # reusable measurement/QA harnesses (re-runnable, versioned)
              # (calibration_harness.py, gold_compare.py, structure_scorecard.py,
              #  code_index.py, ocr_recall_ab.py, ...)
  integration/  # (exists, unchanged)
  codegen/      # (exists, unchanged)
  archive/      # (exists) one-shots after their campaign ends; pilots retire here
  tests/        # (exists) tests move only when their subject moves
```

Placement question for any new script: *documented operator procedure* →
`ops/`; *produces a measurement you'll want again* → `harness/`; *one
campaign* → `archive/` (or `runtime/` if truly scratch). Known references to sweep
when Phase 2 executes: `MCP/tests/test_mailbox_servicer_stop.py` /
`test_repair_partial_resume_state.py` (`scripts.` imports),
`tests/test_prepare_fresh_training_synthesis.py`,
`tests/test_stratified_synthesis_pilot.py`, `scripts/tests/*` imports,
docs/CLAUDE.md mentions, assistant campaign-tool fixed argvs, (the shots
output dir already moved to `runtime/shots/`).

## 4. `docs/` taxonomy (Phase 1 — DONE)

Four buckets, hard rule "no single-file dirs":

- `docs/architecture/` — design + ADRs (absorbed `chunk-schema-v4.md`,
  `typed-edges.md`, `workers.md`).
- `docs/operations/` — runbooks + behavior-flag references (absorbed
  `flow-metrics.md`, `learner-ui-manual-pass.md`).
- `docs/validation/` — gates/validators (unchanged).
- `docs/reference/` — compliance + subsystem deep-dives (absorbed
  `audit-trail.md`, `cross-package-index.md`, `reference-retrieval.md`).
- Root-level `docs/{LICENSING,TECH_DEBT,FILE_MANIFEST,file-audit-cleanup}.md`
  stay at `docs/` root (registers, not topics). CLAUDE.md family unchanged:
  CLAUDE.md is *navigation*, docs are *content*.

## 5. What we deliberately do NOT do (Phase 3 — REJECTED)

- **No `src/` or `packages/` layout.** ~2,600 files across 8 import roots;
  `pyproject`, coverage, ruff, CI, `AGENT_TOOL_MAPPING`, and every checkpoint
  alias assume current paths. Cost is weeks of churn; benefit is aesthetic.
- ~~No renaming `state/ training-captures/`~~ — **superseded 2026-07-29 by
  owner decision**: both now nest under `runtime/` (executed with a full
  tracked-reference sweep + compat symlinks; `ED4ALL_HOME` basenames
  unchanged). `inputs/` keeps its name and root position.
- **No collapsing `Courseforge/exports/`, `SemantiK/output/`,
  `LibV2/courses/` into `runtime/`.** They are `ED4ALL_HOME` basename keys with
  dual-read legacy handling; moving them re-opens the D1 migration class for
  zero operator benefit.
- **No mass `lib/` flat-module consolidation.** Ratchet + as-touched migration
  via the D19 shim pattern.

Revisit any of these only on a repo split or a 2.0 packaging change.

## 6. Enforcement

1. **This document** is the placement authority; root `CLAUDE.md` carries a
   pointer.
2. **`ci/layout_guard.py`** (pattern: `ci/course_slug_guard.py` — guard +
   allowlist + `ci/tests/` test), registered in `ci/integrity_check.py`.
   Checks, all against `git ls-files` (tracked only, so VAR contents never
   trip it):
   - top-level entries ⊆ `ci/layout_allowlist.txt`;
   - no new flat `lib/*.py` beyond the frozen snapshot list;
   - `docs/` subdirs ⊆ {architecture, operations, validation, reference,
     vendor} and no single-file subdirs;
   - no new loose `scripts/*` files beyond the current snapshot (shrinks to
     the §3 taxonomy in Phase 2).
3. **Ratchet semantics**: allowlists may only shrink; adding a line requires
   the same PR to justify it.
