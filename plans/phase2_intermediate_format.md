# Phase 2 — Intermediate Block Format Plan

## 1. Goal & scope

Introduce a stable in-memory `Block` intermediate that sits between the course-data dicts loaded by `Courseforge/scripts/generate_course.py:2120` (`generate_course`) and the HTML/JSON-LD emitted by `_wrap_page` at `Courseforge/scripts/generate_course.py:754`. The Block must:

- Be the single source of truth for `data-cf-*` attribute and JSON-LD field emission.
- Carry per-block provenance (`touched_by[]`) so a later "outline tier → validation → rewrite tier" pipeline can route each block to a different model.
- Round-trip losslessly: emit → parse → re-emit byte-stable.
- Preserve current consumer contract for `Trainforge/process_course.py` (HTML attribute parsing) while opening a cleaner consumption path via JSON-LD `blocks[]`.

## 2. Audit results — emit-call sites in `generate_course.py`

49 occurrences of the literal `data-cf-` across `Courseforge/scripts/generate_course.py`. All emit sites cluster in 8 string-interpolating renderer functions plus the JSON-LD builders. Distinct emit functions and their current line ranges:

| # | Function | Lines | What it emits |
|---|---|---|---|
| 1 | `_wrap_page` | `:754-809` | `<body>` chrome `data-cf-role="template-chrome"`, JSON-LD `<script>` block |
| 2 | `_source_attr_string` | `:812-830` | `data-cf-source-ids`, `data-cf-source-primary` |
| 3 | `_render_objectives` | `:833-873` | `<li data-cf-objective-id data-cf-bloom-level data-cf-bloom-verb data-cf-cognitive-domain>`; `.objectives` wrapper source-ids |
| 4 | `_render_flip_cards` | `:876-895` | `data-cf-component data-cf-purpose data-cf-teaching-role data-cf-term` |
| 5 | `_render_self_check` | `:898-951` | `data-cf-component data-cf-purpose data-cf-teaching-role data-cf-bloom-level data-cf-objective-ref` + source attrs |
| 6 | `_render_content_sections` | `:990-1104` | `<section>`/heading `data-cf-content-type data-cf-key-terms data-cf-bloom-range`, callout `data-cf-content-type`, source attrs |
| 7 | `_render_activities` | `:1107-1146` | `data-cf-component data-cf-purpose data-cf-teaching-role data-cf-bloom-level data-cf-objective-ref` + source attrs |
| 8 | `generate_week` (open `<section data-cf-source-ids>` wrappers around overview/application/self-check/summary/discussion bodies) | `:1761-2081` | inline string concat at `:1834,1907,1945,2009,2015,2052` |

JSON-LD builders (parallel emit surface, currently de-coupled from HTML emit):

| # | Function | Lines | Output |
|---|---|---|---|
| 9 | `_build_objectives_metadata` | `:1331-1421` | `learningObjectives[]` entries |
| 10 | `_build_sections_metadata` | `:1455-1490` | `sections[]` entries |
| 11 | `_build_bloom_distribution` | `:1493-1527` | `bloomDistribution{}` |
| 12 | `_build_misconceptions_metadata` | `:1530-1579` | `misconceptions[]` entries |
| 13 | `_build_page_metadata` | `:1582-1638` | top-level `CourseModule` JSON-LD |

Total: **8 HTML-attribute emit sites + 5 JSON-LD builders + 6 inline `<section data-cf-source-ids>` wrappers in `generate_week`**. Plan effort estimate: ~3 weeks (one engineer) for the refactor proper, ~1 week for migration tooling and tests.

The current code path is: course-data dict → `_render_*` (string concat HTML, reading dict keys directly) **and in parallel** course-data dict → `_build_*_metadata` (assembling JSON-LD). The two paths share the same input dict but each independently picks fields, which is the structural reason a block-shaped intermediate is needed: today `data-cf-bloom-level` on a heading and `bloomLevel` in the JSON-LD entry can drift because nothing forces them to come from the same Block.

## 3. Block dataclass design

New module: `Courseforge/scripts/blocks.py` (sibling to `generate_course.py`). A single `@dataclass` plus a small registry of valid `block_type` literals.

```python
@dataclass(frozen=True)
class Block:
    # --- Identity ---
    block_id: str                          # stable: f"{page_id}#{type}_{slug}_{idx}"
    block_type: str                        # one of BLOCK_TYPES below
    page_id: str                           # back-pointer to the owning page
    sequence: int                          # position within page (for stable diffs)

    # --- Content ---
    content: Union[str, Dict[str, Any]]    # str for prose; structured dict for
                                           # objectives/self_check/activities/flip_cards
    template_type: Optional[str] = None    # data-cf-template-type (Wave 79):
                                           # real_world_scenario | problem_solution
                                           # | common_pitfall | procedure | None
    key_terms: Tuple[str, ...] = ()        # canonical slugs; data-cf-key-terms

    # --- Pedagogical metadata (mirrors current data-cf-* + JSON-LD axes) ---
    objective_ids: Tuple[str, ...] = ()    # data-cf-objective-id / data-cf-objective-ref
    bloom_level: Optional[str] = None      # primary; serialises to data-cf-bloom-level + bloomLevel
    bloom_verb: Optional[str] = None
    bloom_range: Optional[Tuple[str, ...]] = None  # data-cf-bloom-range
    bloom_levels: Tuple[str, ...] = ()     # Wave 58 multi-verb support
    bloom_verbs: Tuple[str, ...] = ()
    cognitive_domain: Optional[str] = None
    teaching_role: Optional[str] = None    # data-cf-teaching-role
    content_type_label: Optional[str] = None  # SectionContentType enum value
    purpose: Optional[str] = None          # data-cf-purpose
    component: Optional[str] = None        # data-cf-component (flip-card | self-check | activity)

    # --- Source attribution ---
    source_ids: Tuple[str, ...] = ()       # data-cf-source-ids (canonical "dart:slug#blockid" strings)
    source_primary: Optional[str] = None
    source_references: Tuple[Dict[str, Any], ...] = ()
                                           # full SourceReference dicts when JSON-LD-grade detail exists

    # --- Provenance (NEW) ---
    touched_by: Tuple[Touch, ...] = ()     # see below

    # --- Stability hash ---
    content_hash: str = ""                 # sha256(canonical_json(content + tier-1 metadata))
                                           # Used to detect re-execution drift across tiers.
```

```python
@dataclass(frozen=True)
class Touch:
    model: str            # e.g. "claude-sonnet-4-6", "qwen2.5:14b"
    provider: str         # "anthropic" | "local" | "claude_session" | "deterministic"
    tier: str             # "outline" | "validation" | "rewrite"
    timestamp: str        # ISO-8601 UTC
    decision_capture_id: str   # links into training-captures (Wave 112 invariant — never empty)
    purpose: str          # short tag, e.g. "draft", "rewrite", "validate"
```

`BLOCK_TYPES` (frozenset, validated at construction):

```
{"objective", "concept", "example", "assessment_item", "explanation",
 "prereq_set", "activity", "misconception", "callout", "flip_card_grid",
 "self_check_question", "summary_takeaway", "reflection_prompt",
 "discussion_prompt", "chrome", "recap"}
```

`prereq_set` is a single-block aggregate per page (carrying `prerequisitePages` IDs from the course data). `chrome` covers the body header/footer/skip-link the validator already excludes via `data-cf-role="template-chrome"`.

### Helpers on Block

```python
def to_html_attrs(self) -> str:
    """Render the same data-cf-* attribute string Courseforge emits today.
    Single source of truth — every renderer calls this instead of inlining
    f-strings. Preserves attribute order so existing snapshot tests stay
    byte-stable."""

def to_jsonld_entry(self) -> Dict[str, Any]:
    """Render the JSON-LD entry shape this block contributes to (LearningObjective
    / Section / Misconception / KeyTerm). Structural-only; the page-level
    builder still composes them into _build_page_metadata's payload."""

def with_touch(self, t: Touch) -> "Block":
    """Return a new Block with the touch appended (frozen dataclass)."""

def stable_id(self) -> str: ...
def compute_content_hash(self) -> str: ...
```

Why `frozen=True` and tuples instead of lists: the two-pass pipeline will mutate blocks repeatedly across tiers; immutability + `with_*` constructors make the touch chain auditable and prevent accidental shared-list mutation across model dispatches.

## 4. Schema strategy — extend additively to `courseforge_jsonld_v1`

Recommendation: **extend additively**, do NOT fork to v2 in this phase.

Why:
- `schemas/knowledge/courseforge_jsonld_v1.schema.json` is currently `additionalProperties: false` at root and on every `$def`. We will need to add new optional top-level fields, which is a non-breaking schema change as long as we keep existing required keys intact.
- A v2 fork would force migration of every existing course in `Courseforge/exports/`, the `LibV2/courses/` corpora, and `validate_page_jsonld_shacl_at_emit` (Wave 68). That cost belongs to a later phase, not here.
- The `@context` IRI (`https://ed4all.dev/ns/courseforge/v1`) and the `lib/ontology/jsonld_context_loader.py` path stay stable; pyld + SHACL keep working unchanged.

Concrete schema additions (to `schemas/knowledge/courseforge_jsonld_v1.schema.json`):

1. New top-level optional `blocks[]` array. Each entry is a new `$defs/Block` matching the dataclass above. Required: `block_id`, `block_type`, `sequence`. All other fields optional. `additionalProperties: false`.
2. New top-level optional `provenance` object with `runId`, `pipelineVersion`, `tiers[]` summary. Per-block `touched_by[]` lives inside each `Block`.
3. New top-level optional `contentHash` (string, sha256 hex) on the page so `course_metadata.json` can list per-page hashes.
4. Existing `learningObjectives[] / sections[] / misconceptions[]` stay. They continue to be the **canonical** projection of `blocks[]` for back-compat — emit code populates both. The schema gains a `$comment` saying the redundancy is intentional during the migration window.

Course-level wrapper: extend `course_metadata.json` (currently emitted by `generate_course` at `Courseforge/scripts/generate_course.py:2179-2200`) with a new optional `blocks_summary` object: `{total_blocks, by_type{}, hash_root}`. Author a sibling `schemas/knowledge/course_metadata.schema.json` for it (currently the stub is unschematised — verify by inspection at `:2181`).

### Provenance storage decision

Pick **embed-in-JSON-LD** for `touched_by[]`, NOT inline `data-cf-touched-*` HTML attributes nor a separate JSON sidecar. Justification:

- Per-block touch arrays grow with each tier (3+ entries per block × ~50 blocks/page × ~80 pages = 12k entries per course). Inline HTML attributes would inflate every page by ~30-50%, hurt the existing snapshot tests, and require parsing on read.
- A sidecar file means consumers must coordinate two paths and can drift; the JSON-LD `<script>` block is already self-contained and carried into IMSCC by the packager.
- JSON-LD has the entire SHACL/pyld validation scaffolding wired (`lib/validators/shacl_runner.py:53`); SHACL shapes for `Block.touched_by[]` come for free.
- Reserve `data-cf-block-id` as the only new HTML attribute we add (so a parser walking HTML can cross-reference into JSON-LD `blocks[]` by ID). Touch metadata stays JSON-LD only.

## 5. Refactor map for `generate_course.py`

Recommended approach: **bottom-up incremental migration with an internal seam**, not a top-down rewrite.

Step A — introduce the `Block` class and `BlockEmitter` without changing emit. Add `Courseforge/scripts/blocks.py` and `Courseforge/scripts/block_emitter.py`. `BlockEmitter` exposes:

```python
class BlockEmitter:
    def emit_block_html(self, block: Block) -> str: ...
    def emit_block_jsonld(self, block: Block) -> Dict[str, Any]: ...
```

Step B — convert `_render_*` functions to **build a `List[Block]` first, then call `emit_block_html` on each.** One renderer per commit:

| Order | Renderer | Block type produced | Risk |
|---|---|---|---|
| B1 | `_render_objectives` (`:833-873`) | `objective` | Low — small, well-tested |
| B2 | `_render_flip_cards` (`:876-895`) | `flip_card_grid` | Low |
| B3 | `_render_self_check` (`:898-951`) | `self_check_question` × N | Medium — per-question `source_references` override pattern |
| B4 | `_render_activities` (`:1107-1146`) | `activity` × N | Low |
| B5 | `_render_content_sections` (`:990-1104`) | `explanation` / `example` / `procedure` / `comparison` / `definition` / `overview` / `summary` / `exercise` per heading + nested `flip_card_grid` + `callout` | **High** — most complex emit site, has the Wave 35 ancestor-walk wrapper logic |
| B6 | `generate_week` inline `<section>` wrappers (`:1834,1907,1945,2009,2015,2052`) | folded into the B5 wrapper logic by adding a "page body sections" Block sequence | High — this is where Wave 35/41/43 grounding contracts live |

Step C — convert `_build_*_metadata` to call `Block.to_jsonld_entry()` instead of re-reading the source dicts. Keep the function signatures stable so `_build_page_metadata` stays the public seam. After this step, `learningObjectives[]` and `sections[]` are projections of `blocks[]`; `_build_page_metadata` adds the new `blocks[]` and `provenance` fields.

Step D — add the new emit fields `blocks[]`, `provenance`, `contentHash` behind a single env flag `COURSEFORGE_EMIT_BLOCKS` (default `false` for one wave) so existing snapshot tests stay green during the migration window.

Step E — flip `COURSEFORGE_EMIT_BLOCKS` default to `true` after Trainforge gains a consumer; remove the flag in a follow-up phase.

Why incremental over top-down: the file is 2296 lines with ~30 callers across `MCP/tools/_content_gen_helpers.py`, `cli/`, and the test corpus at `Courseforge/scripts/tests/`. A top-down rewrite would force every test to update simultaneously and lose the byte-stable snapshot guarantee that's been catching regressions since Wave 9.

## 6. Consumer compatibility — Trainforge

Today's path (`Trainforge/process_course.py:1370` and `Trainforge/parsers/html_content_parser.py:255`):

1. `HTMLContentParser.parse()` extracts a JSON-LD dict via `_extract_json_ld` (parser.py:445).
2. It also walks the DOM via regex to populate `ContentSection` dataclasses (parser.py:608) reading `data-cf-content-type`, `data-cf-key-terms`, `data-cf-teaching-role`, `data-cf-objective-ref`, `data-cf-source-ids`, `data-cf-template-type`.
3. `process_course._extract_section_metadata` then merges JSON-LD `sections[]` with `data-cf-*` per-section data (process_course.py:2266 onward).

Strategy:

1. **Phase 2 introduces no breakage.** Continue emitting all current `data-cf-*` attributes byte-identically (Block.to_html_attrs() is the new single source). Trainforge consume is unchanged.
2. **Add a parallel JSON-LD `blocks[]` consumer in `html_content_parser.py`.** New method `_extract_blocks_from_jsonld(json_ld) -> List[Block]` that prefers `blocks[]` when present and falls back to the current `sections[]` + DOM scan when absent. Land this as PR-on-the-side; Block construction is preferred path, regex DOM scan becomes secondary fallback (still kept for non-Courseforge IMSCC packages).
3. **Deprecation path for `data-cf-*` attributes**: keep them as redundant emit for one full phase (Phase 2 + Phase 3). Trainforge's regex consumer continues working. After Phase 3 ships and consumers migrated, schedule a Phase 4 follow-up to drop attribute emit for blocks fully covered by JSON-LD `blocks[]` (target: keep `data-cf-block-id`, `data-cf-role="template-chrome"`, `data-cf-source-ids` for ancestor-walk; drop the rest). Document deprecation in `Courseforge/CLAUDE.md`'s "HTML Data Attributes" table.
4. **Contract test** (new): `tests/test_block_contract_trainforge.py`. For each block_type, emit a Block via Courseforge, parse via Trainforge, assert key fields (`bloom_level`, `content_type_label`, `objective_ids`, `source_ids`, `template_type`) are equal across both consume paths (legacy regex DOM walk + new JSON-LD blocks[] walk). Run on every Courseforge wave and every Trainforge wave.

## 7. Outline-only mode for the packager

Wire as a new flag on `Courseforge/scripts/package_multifile_imscc.py` plus a parallel "outline emit" mode in `generate_course.py`:

- New CLI flag on `generate_course.py`: `--emit-mode {full|outline}` (default `full`). In `outline` mode, only blocks whose `block_type ∈ {"objective", "prereq_set", "summary_takeaway", "chrome"}` are rendered to HTML. Content/example/explanation/assessment blocks are emitted to JSON-LD `blocks[]` only, not into HTML body.
- New CLI flag on `package_multifile_imscc.py`: `--outline-only`. When set, the packager uses a per-week skeleton: only `week_NN_overview.html` and `week_NN_summary.html` are zipped; the manifest's `<item>` tree drops content/application/self_check/discussion entries; `learningObjectives` validation is unchanged; `course_metadata.json` is included as before but augmented with `blocks_summary.outline_only=true`.
- Trainforge handling: `process_course.py` reads `course_metadata.blocks_summary.outline_only` if present; when true, downstream synthesis skips the `instruction_pair` extraction loop (no full content to extract from) but keeps `kg_metadata` / `schema_translation` deterministic generators (they read the property manifest, not chunks). New `decision_type="trainforge_outline_only_input"` enum addition required in `schemas/events/decision_event.schema.json`. Without that key in `course_metadata.json`, behavior is unchanged (back-compat).

Implementation locus:
- `generate_course.py:1761-2081` `generate_week`: add an early branch that filters blocks before each `_wrap_page` call when `--emit-mode=outline`.
- `package_multifile_imscc.py:116-213` `build_manifest`: skip non-overview/summary `html_files` when `--outline-only` is true, and tag the LOM description with `[OUTLINE]` so a human inspecting the IMSCC sees the mode.

## 8. Migration / backward compatibility

Single-flag rollout:

1. Land Block dataclass, BlockEmitter, and the bottom-up renderer migration without touching emit output (Steps A-C above). Snapshot tests confirm zero diff.
2. Add `COURSEFORGE_EMIT_BLOCKS` env flag. When truthy, JSON-LD payload includes the new `blocks[]`, `provenance`, `contentHash` fields. Schema adds them as optional. Default off.
3. Land Trainforge consumer (`_extract_blocks_from_jsonld`) reading the new fields when present.
4. Flip the default to on after one wave of co-existence; declare the contract test green.
5. Drop the flag after a second wave with no regressions.

Regression tests must include byte-stable snapshots of every `Courseforge/scripts/tests/test_*_emit.py` fixture. The fixtures live in `Courseforge/scripts/tests/fixtures/` and form the regression net for Block migration.

## 9. Sequencing of subtasks

```
Week 1 — foundations (serial)
  T1. blocks.py: Block + Touch dataclasses + BLOCK_TYPES enum + tests (1 day)
  T2. block_emitter.py: to_html_attrs / to_jsonld_entry implementations + tests (1.5 days)
  T3. JSON Schema additions (Block $def + blocks[] property) (0.5 day)
  T4. SHACL shape additions for Block (0.5 day)
  T5. course_metadata.schema.json author (0.5 day)

Week 2 — renderer migration (parallelisable in pairs)
  T6.  Migrate _render_objectives        (B1)  ┐
  T7.  Migrate _render_flip_cards        (B2)  │  Two engineers
  T8.  Migrate _render_self_check        (B3)  │  can split B1-B4
  T9.  Migrate _render_activities        (B4)  ┘
  T10. Migrate _render_content_sections  (B5)  ── solo, week-long, highest risk
  T11. Migrate generate_week section wrappers (B6) ── follows T10

Week 3 — JSON-LD + provenance + outline-only + consumers (parallelisable)
  T12. Migrate _build_*_metadata to call Block.to_jsonld_entry (1.5 days)
  T13. Add blocks[] / provenance / contentHash emit behind flag (1 day)
  T14. Add --emit-mode flag to generate_course.py (1 day)
  T15. Add --outline-only flag to package_multifile_imscc.py (0.5 day)
  T16. Add _extract_blocks_from_jsonld consumer in html_content_parser.py (1.5 days)
  T17. Land contract test test_block_contract_trainforge.py (0.5 day)
  T18. Migration docs + CLAUDE.md attribute-table update (0.5 day)
```

T6-T9 can run in parallel (independent renderers, independent fixture files). T10 must be serialised because it touches the Wave 35/41/43 ancestor-walk wrapping logic that the validators depend on.

## 10. Testing plan

- **Unit (new)**: `tests/test_block_dataclass.py` covers identity, frozen-immutability, `with_touch` chain growth, `compute_content_hash` stability across reorderings, BLOCK_TYPES validation.
- **Unit (new)**: `tests/test_block_emitter_html.py` and `tests/test_block_emitter_jsonld.py` — for each block_type, assert `to_html_attrs` and `to_jsonld_entry` produce the exact substring the legacy renderer would.
- **Snapshot regression** (existing, must stay green): every test under `Courseforge/scripts/tests/test_*_emit.py` runs against the migrated renderers. Byte-stable diff is the migration gate.
- **Round-trip integration (new)**: `tests/test_block_roundtrip.py` — emit page → parse via Trainforge → emit again from re-constructed Blocks → assert HTML and JSON-LD byte-equal. Idempotency is the key invariant.
- **Contract test (new)**: `tests/test_block_contract_trainforge.py` (described in §6). Runs in both Courseforge and Trainforge CI matrix.
- **JSON-LD schema validation**: extend `Courseforge/scripts/tests/test_generate_course_jsonld_validation.py` with cases asserting `blocks[]` validates against the extended schema; existing required-field cases unchanged.
- **SHACL validation**: extend the Wave 68 SHACL emit-time check (`generate_course.py:508-530`) to include `Block.touched_by[]` cardinality assertions.
- **Provenance audit**: new `tests/test_block_provenance_chain.py` — drive a synthetic 3-tier run, assert each Block carries 3 `Touch` entries with monotonically-increasing timestamps and never-empty `decision_capture_id` (mirrors Wave 112 invariant).

## 11. Risks & rollback

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Performance regression** from frozen-dataclass instantiation across ~50 blocks/page × ~80 pages = ~4k Block instances per course | Medium | Medium (CI wall-time) | Profile with `cProfile` after T11. Use `slots=True` if instantiation cost > 5%. The current `_render_content_sections` already string-allocates similar volume; net impact should be neutral. |
| **Trainforge consumer breakage** during co-existence window | Medium | High (blocks training-data pipeline) | Keep `data-cf-*` attribute emit byte-identical for the entire Phase 2; consumer test suite gates the Block emit flag flip. |
| **JSON-LD schema drift** between Courseforge emit and SHACL shape | Medium | Medium (CI failures, no shipped damage) | Land schema additions in the same commit as emit additions; the existing SHACL emit-time check (`generate_course.py:508`) catches drift at emit-time. |
| **Wave 35 ancestor-walk grounding contract** breaks under B5/B6 | Medium | High (every page flagged ungrounded) | T10 includes a test specifically asserting that for every non-trivial `<p>`/`<li>`/`<figcaption>` in emit output, an ancestor `<section data-cf-source-ids>` exists. Mirror `ContentGroundingValidator`'s walk in the test. |
| **`@context` IRI semantics shift** when `blocks[]` lands | Low | High (pyld parse failures) | Reuse the existing `https://ed4all.dev/ns/courseforge/v1` IRI; add Block term mappings to the in-repo context document at `lib/ontology/jsonld_context_loader.py`. No version bump on `@context` itself. |

**Rollback**: every change is gated on `COURSEFORGE_EMIT_BLOCKS=false` (default). If a regression ships, set the env var false in CI and runtime; emit reverts to current shape because the migrated renderers still produce byte-stable HTML/JSON-LD when no Block-extension fields are added. Hard rollback (revert the renderer migration commits) is also possible per-renderer because each landed in its own commit.

## 12. Open questions for the user

1. **Block ID stability across re-runs**: should `block_id` be content-hash-based (matching the `TRAINFORGE_CONTENT_HASH_IDS` Worker N pattern) or position-based (simpler, but reorder-fragile)? Position-based is the cheap default; hash-based costs nothing but couples re-execution semantics.
2. **`touched_by[]` retention policy**: keep the full chain forever (cumulative across runs) or last-N-tiers-only? Full chain inflates JSON-LD over course lifetime; last-3 is enough for two-pass debugging.
3. **Outline-only IMSCC**: should we ship a separate manifest type (`<schema>IMS Common Cartridge Outline</schema>`) or reuse the standard schema and tag via metadata only? Brightspace tolerates the latter; LMS-side pedagogical clarity prefers the former.
4. **Schema fork timing**: do you want `courseforge_jsonld_v2.schema.json` reserved as a follow-up phase deliverable, or should we authoritatively commit to staying on v1 indefinitely? My plan assumes v1-additive forever; v2 only forks if `additionalProperties: false` becomes painful for some future axis.
5. **`prereq_set` granularity**: one block per page (carrying `prerequisitePages[]` array) or one block per prerequisite? Current JSON-LD emits the array as a single `prerequisitePages` field; one-block-per-page matches that and is simpler.
6. **`Touch.decision_capture_id` link semantics**: should the ID be a path into `training-captures/` JSONL, a UUID stored in a separate ledger, or both? Trainforge Wave 112 requires it non-empty but the format isn't fixed.
7. **Trainforge consumer drop semantics**: when we eventually drop redundant `data-cf-*` attributes (Phase 4 follow-up), do we need to support reading legacy IMSCC packages without `blocks[]`? If yes the regex DOM walk stays as a permanent fallback path — fine, but worth stating.

### Critical Files for Implementation

- `/home/user/Ed4All/Courseforge/scripts/generate_course.py`
- `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json`
- `/home/user/Ed4All/Trainforge/parsers/html_content_parser.py`
- `/home/user/Ed4All/Courseforge/scripts/package_multifile_imscc.py`
- `/home/user/Ed4All/Trainforge/process_course.py`
