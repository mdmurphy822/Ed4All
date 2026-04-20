# Worker X — Wave 6 Consolidation Sub-Plan

Final consolidation before user pipeline testing. Four atomic actions:

1. Commit `plans/` tree (untracked on dev-v0.2.0).
2. Delete `PLAN_NOTES.md` at repo root (legacy scratch; never tracked in git but still present on disk).
3. Refresh `README.md` (v0.2.0 status).
4. Refresh 8 `CLAUDE.md` files (spot edits for v0.2.0 state).

**Authoritative source** for Waves 1–6 changes: `schemas/ONTOLOGY.md` § 12. **Do not duplicate** — reference it.

---

## 1. Files to commit under `plans/`

Verified against `git status` on `chore/wave6-consolidation` branched from `dev-v0.2.0` (worker sub-plans are already tracked; only these are untracked):

- `plans/README.md` — plans/ directory convention (tracked convention doc)
- `plans/kg-quality-review-2026-04/README.md` — effort overview
- `plans/kg-quality-review-2026-04/review.md` — 848-line KG-quality review
- `plans/kg-quality-review-2026-04/discovery/a-bloom-verbs.md`
- `plans/kg-quality-review-2026-04/discovery/b-controlled-vocabularies.md`
- `plans/kg-quality-review-2026-04/discovery/c-jsonld-contract.md`
- `plans/kg-quality-review-2026-04/discovery/d-taxonomy-propagation.md`
- `plans/kg-quality-review-2026-04/discovery/e-cross-system-contracts.md`

Total: 8 files.

Worker sub-plans under `plans/kg-quality-review-2026-04/worker-*-subplan.md` (F through W, plus this X) are all already tracked — confirmed via `git ls-files plans/` returning 18 entries.

## 2. PLAN_NOTES.md

Status: **never tracked in git** (`git log --all -- PLAN_NOTES.md` returns empty). Present on disk at `/home/mdmur/Projects/Ed4All/PLAN_NOTES.md` in the main worktree.

Action: physically remove the file from disk in the main worktree at the end (after PR creation). In my worktree, it does not exist. `git rm PLAN_NOTES.md` would fail (not tracked), so no git action is needed — just an `rm` at the end.

## 3. README.md updates (spot edits)

Current state (264 lines) is reasonably accurate post-unification. Issues:

- **Line 20**: Claims concept graph has "typed semantic relationships (prerequisite, is-a, related-to)". Actually: 8 edge types now (3 taxonomic + 5 pedagogical). Fix: say "8 typed relationships" or similar small correction.
- **Missing**: No mention of v0.2.0 state, `schemas/ONTOLOGY.md`, or `plans/` directory. Add a short "Status" or "Current state" callout near the top (under the first paragraph).

Planned edits:

1. Update the concept-graph sentence on line 20 to reflect 8 edge types (not 3) and point at the ontology map.
2. Add one 5-line status block near the top referencing:
   - v0.2.0 state (KG-quality hardening complete, 6 waves)
   - `schemas/ONTOLOGY.md` § 12 for the change summary
   - `plans/kg-quality-review-2026-04/review.md` for the review

No restructure. No rewrite of the who/why/what sections.

## 4. CLAUDE.md updates (8 files)

Per-file plan, strictly spot edits, target ~15–30 lines per file.

### 4.1 `/CLAUDE.md` (root, 595 lines)

Edits:

- **Line 214 area — "Courseforge Metadata Output"**: add a note that the canonical `data-cf-*` table lives in `Courseforge/CLAUDE.md`; leave the summary sentence but do not duplicate the full table.
- **Active Gates table (lines 510–524)**: add 3 rows — `course_generation.page_objectives` (PageObjectivesValidator, Wave 2 Worker L), `batch_dart.dart_markers` (DartMarkersValidator, Wave 6 Worker V), `textbook_to_course.dart_markers` (same validator, Wave 6 Worker V).
- **New subsection under "Configuration Files"**: add a short "Opt-in behavior flags" table listing the 7 env vars from ONTOLOGY § 12, plus a pointer to ONTOLOGY for detail. Keep to ~12 lines.
- **"Individual Project Guides" (line 542)**: add a bullet pointing at `plans/kg-quality-review-2026-04/review.md` + `schemas/ONTOLOGY.md`.
- **Decision Capture section**: no change needed (examples still accurate; `decision_type` enum expansion is descriptive, does not break current code).

Target: ~25–30 lines of additions/modifications.

### 4.2 `/Courseforge/CLAUDE.md` (361 lines)

Edits:

- **data-cf-* attribute table (lines 244–255)**:
  - Remove `data-cf-objectives-count` (not in current table — already gone). Confirm via re-read.
  - Add row for `data-cf-role` (`<body>`, role classification e.g. `template-chrome`).
  - Add row for `data-cf-teaching-role` (Wave 2 Worker K, `<section>` / component wrappers).
  - Add row for `data-cf-bloom-range` (`<section>` / heading, emit-only).
  - Add row for `data-cf-term` (key-term spans, emit-only).
- **Scripts table (around line 343)**: update `generate_course.py` row to note "emits `course_metadata.json` + page-level JSON-LD + prerequisite page refs (Wave 2)". Update `package_multifile_imscc.py` row to note "default-on IMSCC validation + auto-bundles `course_metadata.json`".
- **JSON-LD Structured Metadata (line 258)**: add a short note that the authoritative shape is `schemas/knowledge/courseforge_jsonld_v1.schema.json`.

Target: ~20 lines.

### 4.3 `/Trainforge/CLAUDE.md` (257 lines)

Edits:

- **Decision Capture Protocol section (around line 44)**: add a one-line note that `DECISION_VALIDATION_STRICT=true` (Wave 1 Worker G) opt-in fails closed on unknown `decision_type` values. Do not expand the section — leave examples intact.
- **Metadata Extraction (line 168)**: add note that canonical chunk contract is `schemas/knowledge/chunk_v4.schema.json` (opt-in enforcement via `TRAINFORGE_VALIDATE_CHUNKS=true`).
- **Enriched Chunk Fields table (line 176)**: add rows for `run_id` and `created_at` provenance fields (always-emitted, Wave 4.1 Worker P).
- **New 4-line note** near Metadata Extraction pointing at ONTOLOGY § 12 for: concept node `occurrences[]` back-references, 8 edge types in concept graph, opt-in flags.

Target: ~20 lines.

### 4.4 `/LibV2/CLAUDE.md` (238 lines)

Edits:

- Lines 138–139 already point at `../schemas/library/` and `../schemas/taxonomies/` correctly — no change needed.
- Line 223 already says `schemas/taxonomies/` — no change.
- **Common Tasks / Advanced Commands (around line 186)**: add a one-line note that `content_type_label` validates when `TRAINFORGE_ENFORCE_CONTENT_TYPE=true` (Wave 5 Worker T).
- **Ontology Mappings (line 219)**: no change needed.

Target: ~5–10 lines (this file is already mostly current).

### 4.5 `/DART/CLAUDE.md` (139 lines)

Edits:

- **MCP Tools table (line 50)**: add a row for `validate_dart_markers` — gate-wired as `dart_markers` on `batch_dart` + `textbook_to_course` in Wave 6 (Worker V).
- No other stale content detected.

Target: ~3 lines.

### 4.6 `/Courseforge/agents/CLAUDE.md` (158 lines)

Scope check: file covers agent scratchpad / batching / template protocols. Grep for `data-cf-`, `schemas/`, `bloom`: no stale refs found. Leave untouched unless a pass finds something.

Target: ~0–5 lines (likely no edits).

### 4.7 `/Courseforge/imscc-standards/CLAUDE.md` (400 lines)

Scope check: pure IMSCC / QTI / D2L spec reference. No schema paths that Wave 1 unified would apply. No stale KG refs. Leave untouched.

Target: ~0 lines.

### 4.8 `/Trainforge/agents/CLAUDE.md` (73 lines)

Scope check: purely agent hand-off contracts. No stale schema paths. Add optional one-line pointer to ONTOLOGY § 12 for the canonical chunk / edge / node contracts that agents produce.

Target: ~2 lines (or 0).

---

## Scope discipline

- Spot edits only. No section rewrites. No tone changes.
- Do NOT duplicate content from `schemas/ONTOLOGY.md` § 12 — reference it.
- Do NOT modify code, schemas, tests, or configs.
- If any file's edits approach 50 lines, STOP and scope down.

## Commit structure

Two commits for review clarity:

1. `chore: commit plans/ tree (KG-quality review working artifacts)` — adds the 8 untracked plans files.
2. `docs: refresh README + CLAUDE.md files for v0.2.0 state (post-Waves 1–6)` — all doc spot edits in one commit.

PLAN_NOTES.md removal lives outside the git tree (it was never tracked), so no commit touches it. I will `rm` the file from the main worktree disk at the end.

## Verification

```bash
python3 -m ci.integrity_check
source venv/bin/activate && pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ LibV2/tools/libv2/tests/ -q
rg 'schemas/academic-metadata/' --glob '!plans/*' --glob '!.claude/*'
rg 'schemas/learning-objectives/' --glob '!plans/*' --glob '!.claude/*'
rg 'LibV2/ontology/taxonomy.json' --glob '!plans/*' --glob '!.claude/*'
rg 'data-cf-objectives-count' --glob '!plans/*' --glob '!.claude/*'
```

All greps must return zero hits in the live tree. `plans/` and `.claude/` contain historical references that are expected to stay.
