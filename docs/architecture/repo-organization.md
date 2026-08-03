# Repo Organization Schema

**Status: ADOPTED** — Phase 1 landed 2026-07-29 (this document, the four-bucket
`docs/` taxonomy, `ci/layout_guard.py`); the runtime-collapse phase landed the
same day (owner decision): `state/`, `training-captures/`, `seats/`, `demo/`,
`extracted/`, `testruns/`, `scratchpad/`, and `shots/` all live under the
single gitignored `runtime/` root, with compat symlinks at `state`,
`training-captures`, and `seats` until the paused run finishes and operator
env files are updated. Phase 2 (`scripts/` re-taxonomy) landed 2026-08-02;
one campaign-specific root wrapper remains temporarily ratcheted pending its
separate public-release disposition. The
import-root moves stay **rejected** (§5). Phase 4 extended the schema
*inward* on 2026-08-01 — the subsystem interiors now carry a declared shape
and a flat-file cap (§7); its reorgs are pending. This document is the
placement authority at every level: when a new file or directory doesn't
obviously fit a rule below, that is a design question, not a formatting one.

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
- **`lib/` mixes ~30 flat modules with 20+ subpackages.** The sanctioned
  migration pattern is a subpackage plus a
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
  runbook stays untracked),
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

## 3. `scripts/` taxonomy (Phase 2 — DONE)

```
scripts/
  ops/        # durable operator entry points, documented in docs/operations/
              # (bootstrap-training-env.sh, ed4all-training, gpu_guard.sh,
              #  mailbox_servicer.py, repair_partial_resume_state.py, ...)
  harness/    # reusable measurement/QA harnesses (re-runnable, versioned)
              # (calibration_harness.py, gold_compare.py, structure_scorecard.py,
              #  code_index.py, ocr_recall_ab.py, ...)
  integration/  # cross-subsystem integration tools
  codegen/      # generated-contract maintenance
  archive/      # one-shots after their campaign ends; pilots retire here
  tests/        # tests move only when their subject moves
```

Placement question for any new script: *documented operator procedure* →
`ops/`; *produces a measurement you'll want again* → `harness/`; *one
campaign* → `archive/` (or `runtime/` if truly scratch). The Phase 2 move
updated imports, tests, documented commands, code comments, fixed argument
vectors, and repo-root derivations together; the loose-file ratchet now retains
only the wrapper awaiting separate disposition.

## 4. `docs/` taxonomy (Phase 1 — DONE)

Four buckets, hard rule "no single-file dirs":

- `docs/architecture/` — design + ADRs (absorbed `chunk-schema-v4.md`,
  `typed-edges.md`, `workers.md`).
- `docs/operations/` — runbooks + behavior-flag references (absorbed
  `flow-metrics.md`, `learner-ui-manual-pass.md`).
- `docs/validation/` — gates/validators (unchanged).
- `docs/reference/` — compliance + subsystem deep-dives (absorbed
  `audit-trail.md`, `cross-package-index.md`, `reference-retrieval.md`).
- Root-level `docs/LICENSING.md` is the public licensing register. Generated
  manifests, cleanup evidence, technical-debt ledgers, campaign histories,
  and machine-specific runbooks stay local and gitignored. CLAUDE.md family
  remains navigation; public docs remain durable product content.

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
4. **Recursive source-release policy** —
   [`repository-layout.json`](repository-layout.json) classifies every root,
   directory, and publishable file by role. Its companion
   [`repository-layout.schema.json`](repository-layout.schema.json) validates
   the policy shape without duplicating policy values. The standalone
   `ci/repository_policy_guard.py` applies it to tracked files plus untracked,
   non-ignored release candidates; ignored input and runtime trees are never
   traversed. It rejects unclassified or ambiguous paths, illegal parent/child
   role combinations, non-sentinel tracked content in external-data roots,
   generated or oversized artifacts, common secret shapes, and nested source
   repositories. It also scans path segments for private run/export shapes.
   An operator can provide additional private course names and slugs through
   `ED4ALL_PRIVATE_TOKEN_FILE`; that vocabulary remains local, and an in-repo
   token file must itself be gitignored. The check is registered in
   `ci/integrity_check.py` as `repository_policy`.

The JSON policy is the executable classification contract; this document is
the placement and migration rationale. A path that does not resolve to exactly
one policy role is a design failure. Tests and fixtures remain exempt from the
flat-file *count* heuristic, but not from recursive role, privacy, or release
classification.

## 7. Subsystem interior schema (Phase 4)

**Status: ratchet ADOPTED 2026-08-01; recursive policy adopted 2026-08-02;
reorgs in progress.** The original §§ 2–4 checks stopped at depth 1. The
machine-readable policy in § 6 now classifies every depth, while the flat-file
caps below remain migration ratchets for the largest existing containers.
The subsystem roots keep their names and positions —
`SemantiK/ Courseforge/ Trainforge/ LibV2/ MCP/` are unchanged; their interiors
must now resolve to declared roles as well as respect the caps.

### 7.1 The interior rule

Every non-test code directory is one of two things, and must be able to say
which:

- **A cohesive package** — a flat set of modules that genuinely belong at one
  level because they are peers implementing one contract (`lib/validators/`,
  `lib/ontology/`, `lib/retrieval/`). Flatness here is correct; the module
  list *is* the registry.
- **A container** — holds material of mixed lifetime or purpose
  (`SemantiK/scripts/`, `Trainforge/eval/`). A container must carry a
  taxonomy of subdirs. Flatness here is debt.

The failure mode is a container pretending to be a package: nobody declares
it, so it accretes until nothing can be found. Both shapes are legitimate;
the schema only demands that growth be a decision.

### 7.2 Enforcement: the flat-file cap (check 5)

Each interior directory holding ≥8 loose code files carries a frozen
`flatcap:<dir>=<count>` in `ci/layout_allowlist.txt`, seeded 2026-08-01 at
36 directories. Adding a loose file past the cap fails the guard.

It caps a **count**, not a **name set** — deliberately different from the
`libflat:` / `script:` ratchets in § 6. Those cover ~57 files, where freezing
names is cheap; the interior covers ~600, where name-freezing would make
every legitimate rename a 120-line allowlist diff while adding nothing. The
guarded failure mode is a directory *growing* a 71st loose script, not a file
changing its name.

Three exclusions, each for its own reason:

| Not counted | Why |
|---|---|
| any path with a `tests` segment | a flat test dir mirroring a flat module set is correct; capping it would punish adding a test |
| `__init__.py` | mandatory package marker, never a choice |
| `*.md` | `CLAUDE.md` / `README.md` / `architecture.md` are mandated *at* the dir root by § 4 |

Interior roots are `SemantiK Courseforge Trainforge LibV2 MCP lib gui cli ci`.
`config/` and `schemas/` are deliberately **out of scope**: they are
CONTRACTS, where the flat shape *is* the contract. VAR zones stay invisible to
the guard (tracked-files-only, § 6).

A cap naming a vanished directory is itself a violation — otherwise a reorg
leaves a dead cap behind and the ratchet silently stops enforcing that path.
`ci/tests/test_layout_guard.py::test_every_seeded_flatcap_matches_the_real_tree_exactly`
additionally pins each cap to the *exact* live count, so a cap seeded above
reality (slack the ratchet could never recover) fails the suite.

**Ratchet doctrine, restated for caps: numbers only ever go down.** Raising a
cap is a real exception and must be justified in the same PR. The intended
motion is downward — every reorg below tightens the number it frees.

### 7.3 Reorg backlog (not yet executed)

The cap freezes the problem; it does not fix it. Ordered by ratio of pain to
risk. None of these are started.

1. **`SemantiK/scripts/` — 70 loose, zero subdirs.** The worst container in
   the tree, and 30 of the 70 are referenced by nothing else in the tracked
   tree at all. Proposed taxonomy: `training/` (the three trainers
   `SemantiK/CLAUDE.md` already documents as intentional exceptions, plus
   `register_qwen_adapter.py`, `qwen_lora_to_gguf.py`), `eval/`, `smoke/`,
   `calibration/`, `datasets/`, `analysis/`. `run_cascade_json.py` stays flat
   — it is the cascade subprocess entry point, named at `SemantiK/CLAUDE.md`
   §§ 174 and 195. **Sweep every tracked `SemantiK/scripts/` reference before
   any `git mv`.**
2. **`Trainforge/` root — 19 loose modules** with one obvious cluster: nine
   `synthesis_*.py` plus `synthesize_training.py` want to be
   `Trainforge/synthesis/`. Costs a compatibility shim per moved module
   because `MCP/` dispatches into several by dotted path.
3. **`Trainforge/eval/` — 39 loose**, three subdirs already exist; the flat
   remainder should join them.
4. **`SemantiK/data/` — 25 loose** dataset builders, several unreferenced.
5. **`lib/validators/` — 115 loose.** Largest number in the tree but the
   *weakest* case: it is a genuine package whose flat module list is the
   registry `docs/validation/gates.md` maps onto. Listed for completeness;
   the recommendation is to leave it flat and let the cap hold the line.

Per § 5, mass consolidation stays rejected — these land as-touched, each with
its reference sweep, never as one big move.
