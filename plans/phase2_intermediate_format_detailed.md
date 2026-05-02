# Phase 2 Detailed Execution Plan — Intermediate Block Format

Refines `/home/user/Ed4All/plans/phase2_intermediate_format.md` into atomic, individually-verifiable subtasks. Each subtask has a unique, deterministic verification command. The execution worker should NOT need to re-explore the codebase. Mirrors the granularity of `/home/user/Ed4All/plans/phase1_tos_unblock_detailed.md`.

---

## Investigation findings (locked)

- **`generate_course.py` is 2,296 lines and the line ranges in the high-level plan have NOT drifted.** Verified: `_wrap_page` `:754-809`; `_source_attr_string` `:812-830`; `_render_objectives` `:833-873`; `_render_flip_cards` `:876-895`; `_render_self_check` `:898-951`; `_render_content_sections` `:990-1104`; `_render_activities` `:1107-1146`; `generate_week` `:1761-2081` with inline `<section>` source-id wrappers at `:1834,1907,1945,2009,2015,2052`; `_build_objectives_metadata` `:1331-1421`; `_build_sections_metadata` `:1455-1490`; `_build_bloom_distribution` `:1493-1527`; `_build_misconceptions_metadata` `:1530-1579`; `_build_page_metadata` `:1582-1638`; `generate_course` `:2084` (entry; data load `:2120`; `course_metadata.json` stub `:2179-2201`).
- **The `course_metadata.json` stub IS schematised** (the high-level plan said unschematised). Schema lives at `/home/user/Ed4All/schemas/academic/course_metadata.schema.json` (Draft-07, 1.0.0). It uses MIT-OCW-style top-level keys (`courseIdentification`, etc.), NOT the keys the Courseforge stub at `:2181-2198` actually emits (`course_code`, `course_title`, `classification`, `ontology_mappings`). The Phase 2 stub the planner expected is therefore **schemaless in practice** — the academic schema doesn't match. Action: Phase 2 **authors a new schema** at `schemas/knowledge/course_metadata.schema.json` for the Courseforge stub shape (so it doesn't collide with the academic one), and the existing academic schema is left alone.
- **JSON-LD schema** lives at `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json` (255 lines, Draft 2020-12, `additionalProperties: false` at every level, top-level required = `["@context", "@type", "courseCode", "weekNumber", "moduleType", "pageId"]`). Adding optional properties (`blocks[]`, `provenance`, `contentHash`) and a new `$defs/Block` is a non-breaking additive change.
- **`@context` document** at `/home/user/Ed4All/schemas/context/courseforge_v1.jsonld`. Block term mappings (`blocks`, `blockId`, `blockType`, `sequence`, `touchedBy`, `provenance`, `contentHash`) slot in alongside the existing terms; canonical IRI `https://ed4all.dev/ns/courseforge/v1` does NOT bump.
- **SHACL shapes** at `/home/user/Ed4All/schemas/context/courseforge_v1.shacl.ttl` (411 lines). Class targeting is on `ed4all:CourseModule` / `ed4all:LearningObjective` etc. Adding `ed4all:Block` and a `BlockShape` (with `sh:property` for `touched_by[]` cardinality) extends the Wave-67 pattern.
- **Schema validation entry point** for emit-time SHACL is `/home/user/Ed4All/Courseforge/scripts/generate_course.py:508-530` `_validate_page_jsonld_shacl_at_emit`, gated on `COURSEFORGE_ENFORCE_SHACL`. JSON-Schema validation is `_validate_page_jsonld` (called from `_wrap_page` at `:780-781`).
- **Trainforge consumer** has TWO read paths:
  1. `Trainforge/parsers/html_content_parser.py::_extract_json_ld` (`:445-456`) — JSON-LD parse.
  2. `Trainforge/parsers/html_content_parser.py::_extract_sections` (`:608-720`) — DOM regex walk reading `data-cf-content-type`, `data-cf-key-terms`, `data-cf-teaching-role`, `data-cf-objective-ref`, `data-cf-source-ids`, `data-cf-template-type`. Populates `ContentSection` dataclass at `:29-74`.
  3. The merge happens in `Trainforge/process_course.py::_extract_section_metadata` (`:2259-2417`) — JSON-LD `cf_meta["sections"]` is checked first, falls back to DOM-walk `ContentSection`s. Phase 2 must keep both paths working.
- **The Phase 1 `ContentGeneratorProvider.generate_page`** at `/home/user/Ed4All/Courseforge/generators/_provider.py:259-317` returns `str` and explicitly carries the comment `# Phase 2: will return a Block dataclass`. Phase 2 widens this to return `Block`. The Phase 1 wire-in at `MCP/tools/_content_gen_helpers.py:1860-1905` regex-parses the returned HTML via `_parse_provider_page_html` to extract `(heading, paragraphs[])` — Phase 2 deletes that parse and consumes the `Block` directly.
- **No existing `@dataclass Block` shaped class anywhere under `Courseforge/`, `lib/ontology/`, `lib/validators/`** — confirmed via `grep -rn "class.*Block" lib/ Courseforge/scripts/`. The high-level plan's "no precedent" claim is correct.
- **Snapshot regression suite**: `/home/user/Ed4All/Courseforge/scripts/tests/` contains 23 test files (~20 emit-related). Critical regression tests for byte-stable migration: `test_template_chrome_emit.py`, `test_misconception_bloom_tag_emit.py`, `test_lo_multi_verb_emit.py`, `test_lo_targeted_concepts_emit.py`, `test_lo_hierarchy_edges_emit.py`, `test_bloom_distribution_emit.py`, `test_generate_course_jsonld_validation.py`, `test_generate_course_shacl_validation.py`, `test_generate_course_sourcerefs.py`, `test_content_type_enum_validation.py`, `test_callout_content_type_enum_validation.py`. Fixtures live in `Courseforge/scripts/tests/fixtures/` (just `sample_html/` + `sample_imscc/`). Most snapshot tests build their own minimal `course_data` dict in-test and compare emitted HTML strings.
- **Phase 1 wire-in** has `content_provider` already threaded through `MCP/tools/_content_gen_helpers.py::build_week_data` (`:1579`) into `_build_content_modules_dynamic` (`:1796`). The provider call at `:1888` passes a `page_context` dict and currently consumes `str` HTML.
- **`COURSEFORGE_PROVIDER` env-flag table row** is at root `/home/user/Ed4All/CLAUDE.md:729`. The new `COURSEFORGE_EMIT_BLOCKS` row sorts alphabetically BEFORE that row, between `COURSEFORGE_ENFORCE_SHACL` (current ordering — verify locally) and `COURSEFORGE_PROVIDER`.
- **`COURSEFORGE_ENFORCE_SHACL`** env-flag is referenced at `generate_course.py:352`. Constant is `_ENFORCE_SHACL_ENV` (look near line 369). The new `COURSEFORGE_EMIT_BLOCKS` env-flag follows the same opt-in pattern.
- **`html_content_parser.py::ContentSection`** dataclass (`:29-74`) is the Trainforge-side mirror. Phase 2 adds a sibling `Block` consumer that can populate `ContentSection` from JSON-LD `blocks[]` instead of the regex DOM walk; the regex walk stays as fallback for non-Courseforge IMSCC.

## Pre-resolved decisions

1. **Block ID stability:** position-based — `f"{page_id}#{type}_{slug}_{idx}"` (`type` is the block_type, `slug` is `_slugify(heading or term or first 30 chars of content)`, `idx` is the 1-indexed position within page). Hash-based IDs deferred to a later phase. Rationale: bottom-up migration produces stable orderings per renderer; reorder churn is rare; hash-based IDs would couple re-execution semantics to an unstable hashing convention.
2. **`touched_by[]` retention:** keep the full chain (cumulative). Per-course budget ≈ 12k entries × ~80 bytes = ~1 MB JSON before gzip — well within IMSCC payload budgets. Audit value of full chain outweighs the size cost.
3. **Outline-only IMSCC manifest type:** reuse the standard IMS CC v1.3 schema. Tag via `course_metadata.blocks_summary.outline_only=true` only; LMS-side pedagogical clarity stays via metadata. No separate manifest type.
4. **Schema fork timing:** stay on `courseforge_jsonld_v1.schema.json` indefinitely (additive only). v2 fork reserved for a future phase if `additionalProperties: false` becomes painful.
5. **`prereq_set` granularity:** one block per page carrying `prerequisitePages[]` array. Matches the existing JSON-LD field shape.
6. **`Touch.decision_capture_id` semantics:** string `"{capture_file_basename}:{event_index}"` — points into the JSONL written by `_emit_decision` at `Courseforge/generators/_provider.py:_emit_decision`. Path is `training-captures/courseforge/{course_code}/phase_content-generator/decisions_{timestamp}.jsonl`. Non-empty per Wave 112 invariant. No separate ledger.
7. **Trainforge consumer drop semantics:** the regex DOM walk stays as a permanent fallback path for non-Courseforge IMSCC packages. Phase 2 does NOT drop any `data-cf-*` emit; the eventual drop of redundant attributes is a Phase-2-followup, not Phase 2 itself.
8. **`COURSEFORGE_EMIT_BLOCKS` default:** `false` (off) for Wave N; flip to `true` after byte-stable confirmation; drop the flag in a follow-up wave.

---

## Atomic subtasks

Estimated total LOC across all subtasks: ~1,900 (350 dataclass + 220 emitter + 80 schema + 60 SHACL + 80 course_metadata schema + 350 renderer migration + 180 JSON-LD builder migration + 140 new emit + 90 outline-mode + 60 packager + 200 consumer + 200 tests + 80 docs).

### A. Block + Touch dataclass

#### Subtask 1: Create `Courseforge/scripts/blocks.py` skeleton with `BLOCK_TYPES` enum
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** none
- **Estimated LOC:** ~40
- **Change:** Module docstring stating "Phase 2 intermediate block format". Imports: `dataclasses` (`dataclass`, `field`), `typing` (`Any, Dict, List, Optional, Tuple, Union`), `hashlib`, `json`, `re`. Define `BLOCK_TYPES: frozenset[str] = frozenset({"objective", "concept", "example", "assessment_item", "explanation", "prereq_set", "activity", "misconception", "callout", "flip_card_grid", "self_check_question", "summary_takeaway", "reflection_prompt", "discussion_prompt", "chrome", "recap"})`. Add `__all__ = ["Block", "Touch", "BLOCK_TYPES"]`. Define stub `@dataclass(frozen=True) class Touch: pass` with field declarations only (no body). Same for `Block`. Helpers (`to_html_attrs`, `to_jsonld_entry`, `with_touch`, `compute_content_hash`, `stable_id`) declared as `def ... raise NotImplementedError` so the module imports cleanly.
- **Verification:** `python -c "from Courseforge.scripts.blocks import Block, Touch, BLOCK_TYPES; assert 'objective' in BLOCK_TYPES; assert len(BLOCK_TYPES) == 16"` exits 0.

#### Subtask 2: Implement `Touch` dataclass
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~25
- **Change:** Replace the `Touch` stub with `@dataclass(frozen=True)` with fields `model: str`, `provider: str`, `tier: str`, `timestamp: str`, `decision_capture_id: str`, `purpose: str`. All required (no defaults). `__post_init__` validates: empty `decision_capture_id` raises `ValueError("Touch.decision_capture_id required (Wave 112 invariant)")`; `tier not in {"outline", "validation", "rewrite"}` raises `ValueError`; `provider not in {"anthropic", "local", "together", "claude_session", "deterministic"}` raises `ValueError`. Add `to_jsonld() -> Dict[str, Any]` method returning `{"model": ..., "provider": ..., "tier": ..., "timestamp": ..., "decisionCaptureId": ..., "purpose": ...}` (camelCase wire keys).
- **Verification:** `python -c "from Courseforge.scripts.blocks import Touch; t=Touch(model='m',provider='local',tier='outline',timestamp='2026-05-02T00:00Z',decision_capture_id='x:0',purpose='draft'); assert t.to_jsonld()['decisionCaptureId']=='x:0'"` exits 0.

#### Subtask 3: Implement `Block` dataclass with all fields
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** Subtask 2
- **Estimated LOC:** ~70
- **Change:** Replace the `Block` stub with `@dataclass(frozen=True)`. Fields per high-level plan §3 PLUS the two new feedback-driven fields (post-feedback amendment): `block_id, block_type, page_id, sequence, content, template_type, key_terms, objective_ids, bloom_level, bloom_verb, bloom_range, bloom_levels, bloom_verbs, cognitive_domain, teaching_role, content_type_label, purpose, component, source_ids, source_primary, source_references, touched_by, content_hash, validation_attempts: int = 0, escalation_marker: Optional[str] = None`. The two new fields support Phase 3's per-block regeneration budget + escalation-on-fail primitive: `validation_attempts` is incremented by Phase 3's outline-tier router on every failed validator pass; `escalation_marker` is set to a non-empty string (e.g. `"outline_budget_exhausted"`) when the budget is exhausted and the block is escalated to the rewrite tier. Both stay `0` / `None` for blocks emitted by the deterministic / Phase-1-provider paths in Phase 2. Use `Tuple[str, ...] = ()` instead of lists; `Optional[X] = None` for nullables; `content: Union[str, Dict[str, Any]]` no default. `__post_init__` validates `block_type in BLOCK_TYPES` (raises `ValueError`), `sequence >= 0`, `page_id` non-empty, `validation_attempts >= 0`. Add an `_ESCALATION_MARKERS` frozenset (`{"outline_budget_exhausted", "structural_unfixable", "validator_consensus_fail"}`) and validate `escalation_marker` against it when non-None (extensible — new markers added as Phase 3+ identifies failure modes).
- **Verification:** `python -c "from Courseforge.scripts.blocks import Block; b=Block(block_id='p#x_0',block_type='objective',page_id='p',sequence=0,content='c'); assert b.block_type=='objective'"` exits 0; bad `block_type` raises: `python -c "from Courseforge.scripts.blocks import Block; Block(block_id='x',block_type='bogus',page_id='p',sequence=0,content='c')" 2>&1 | grep -q ValueError`.

#### Subtask 4: Implement `Block.with_touch`, `stable_id`, `compute_content_hash`
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** Subtask 3
- **Estimated LOC:** ~40
- **Change:** `with_touch(self, t: Touch) -> "Block"`: returns `dataclasses.replace(self, touched_by=self.touched_by + (t,))`. `stable_id(page_id: str, block_type: str, slug: str, idx: int) -> str`: classmethod returning `f"{page_id}#{block_type}_{slug}_{idx}"`. `compute_content_hash(self) -> str`: sha256-hex of `json.dumps({"content": self.content, "block_type": self.block_type, "key_terms": list(self.key_terms), "bloom_level": self.bloom_level, "objective_ids": list(self.objective_ids)}, sort_keys=True, ensure_ascii=False).encode()`. Excludes `touched_by` and `sequence` so a touch-only revision keeps a stable hash. Add module-level `_slugify(text: str) -> str` helper (mirror Courseforge's: lowercase, non-alnum → `_`, strip, max 40 chars).
- **Verification:** `python -c "from Courseforge.scripts.blocks import Block, Touch; b=Block(block_id='x',block_type='objective',page_id='p',sequence=0,content='c'); h1=b.compute_content_hash(); t=Touch(model='m',provider='local',tier='outline',timestamp='t',decision_capture_id='x:0',purpose='p'); b2=b.with_touch(t); assert b2.compute_content_hash()==h1; assert len(b2.touched_by)==1"` exits 0.

#### Subtask 5: Add `tests/test_block_dataclass.py` covering identity, immutability, touch chain, hash stability
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/tests/test_block_dataclass.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~120
- **Change:** Tests: `test_block_type_validates_against_enum` (bogus → ValueError); `test_block_is_frozen` (assert `dataclasses.FrozenInstanceError` on `b.content = ...`); `test_with_touch_appends_and_returns_new_instance` (id() differs, len grows); `test_with_touch_chain_grows_three_tiers` (chain three Touches with `tier in {"outline","validation","rewrite"}`); `test_compute_content_hash_is_stable_across_touch_chain`; `test_compute_content_hash_changes_when_content_changes`; `test_compute_content_hash_excludes_sequence`; `test_stable_id_format` (`"week_01_overview#objective_TO-01_0"`); `test_touch_validates_decision_capture_id_non_empty`; `test_touch_validates_tier_enum`; `test_touch_validates_provider_enum`. Mirror import-path pattern from `Courseforge/scripts/tests/test_template_chrome_emit.py:1-15`.
- **Verification:** `pytest Courseforge/scripts/tests/test_block_dataclass.py -v` reports ≥11 PASSED, 0 FAILED.

### B. BlockEmitter — to_html_attrs / to_jsonld_entry

#### Subtask 6: Implement `Block.to_html_attrs()` for all 16 block types
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~120
- **Change:** Method on `Block` returning the same `data-cf-*` attribute string the legacy renderers emit, dispatched by `block_type`. Reproduce the exact attribute order / quoting / HTML-escape used today: `objective` → `' data-cf-objective-id="…" data-cf-bloom-level="…" data-cf-bloom-verb="…" data-cf-cognitive-domain="…"'` (matching `_render_objectives:854-860`); `flip_card_grid` → `' data-cf-component="flip-card" data-cf-purpose="term-definition" data-cf-teaching-role="reinforce" data-cf-term="<slug>"'` (matching `_render_flip_cards:887-889`); `self_check_question` → `' data-cf-component="self-check" data-cf-purpose="formative-assessment" data-cf-teaching-role="assess" data-cf-bloom-level="…" data-cf-objective-ref="…"'` plus source attrs (matching `_render_self_check:929-944`); `activity` → analogous (matching `_render_activities:1126-1140`); `explanation`/`example`/`procedure`/`comparison`/`definition`/`overview`/`summary`/`exercise` (heading blocks under `_render_content_sections`) → `' data-cf-content-type="…" data-cf-key-terms="…" data-cf-bloom-range="…"' + source attrs` (matching `:1018-1035`); `callout` → `' data-cf-content-type="…"'` (matching `:1071-1073`); `chrome` → `' data-cf-role="template-chrome"'` (matching `_wrap_page:796-797,804`); `prereq_set`, `summary_takeaway`, `reflection_prompt`, `discussion_prompt` → wrapper-only `' data-cf-source-ids="…" data-cf-source-primary="…"'` (the inline `<section>` wrappers in `generate_week`). NEW Phase-2 attribute: `' data-cf-block-id="<self.block_id>"'` is APPENDED to every emit (the only new HTML attribute Phase 2 adds, gated behind `COURSEFORGE_EMIT_BLOCKS`). Source-ids/primary use `_source_attr_string`'s exact format (defined at `:812-830` of `generate_course.py`); reuse that helper inline (string format `' data-cf-source-ids="<comma-joined-escaped>"' + (' data-cf-source-primary="<escaped>"' if primary else "")`).
- **Verification:** `python -c "from Courseforge.scripts.blocks import Block; b=Block(block_id='p#objective_TO-01_0',block_type='objective',page_id='p',sequence=0,content='Define X',objective_ids=('TO-01',),bloom_level='remember',bloom_verb='define',cognitive_domain='factual'); s=b.to_html_attrs(); assert 'data-cf-objective-id=\"TO-01\"' in s and 'data-cf-bloom-level=\"remember\"' in s"` exits 0.

#### Subtask 7: Implement `Block.to_jsonld_entry()` for all 16 block types
- **Files:** `/home/user/Ed4All/Courseforge/scripts/blocks.py`
- **Depends on:** Subtask 6
- **Estimated LOC:** ~100
- **Change:** Method on `Block` returning a `Dict[str, Any]` matching the JSON-LD entry shape the existing `_build_*_metadata` helpers emit. Dispatched by `block_type`: `objective` → entry shape from `_build_objectives_metadata:1364-1420` (`id`, `statement`, `bloomLevel`, `bloomVerb`, `cognitiveDomain`, optional `bloomLevels[]`, `bloomVerbs[]`, `keyConcepts[]`, `targetedConcepts[]`, `assessmentSuggestions[]`, `prerequisiteObjectives[]`, `hierarchyLevel`, `parentObjectiveId`); section blocks → `_build_sections_metadata:1467-1490` shape (`heading`, `contentType`, optional `keyTerms[]`, `teachingRole[]`, `bloomRange[]`, `sourceReferences[]`); misconception → `_build_misconceptions_metadata:1571-1578` shape; flip_card_grid → contributes to `keyTerms[]` via parent section; self_check_question / activity / chrome / prereq_set / summary_takeaway / reflection_prompt / discussion_prompt → emit `{"blockId": ..., "blockType": ..., "sequence": ...}` PLUS NEW Phase-2 fields (`touchedBy`, `contentHash`) for the new top-level `blocks[]` array. Add helper `_render_touched_by(self) -> List[Dict[str, Any]]` returning `[t.to_jsonld() for t in self.touched_by]`.
- **Verification:** `python -c "from Courseforge.scripts.blocks import Block; b=Block(block_id='x',block_type='objective',page_id='p',sequence=0,content='Define X',objective_ids=('TO-01',),bloom_level='remember',bloom_verb='define',cognitive_domain='factual'); e=b.to_jsonld_entry(); assert e['id']=='TO-01' and e['bloomLevel']=='remember' and e['statement']=='Define X'"` exits 0.

#### Subtask 8: Add `Courseforge/scripts/tests/test_block_emitter_html.py` and `test_block_emitter_jsonld.py`
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/tests/test_block_emitter_html.py` and `/home/user/Ed4All/Courseforge/scripts/tests/test_block_emitter_jsonld.py`
- **Depends on:** Subtasks 6, 7
- **Estimated LOC:** ~200 across both files
- **Change:** For each of the 16 block_types, build a representative `Block` instance and assert `to_html_attrs()` produces the exact substring the legacy renderer would emit for the equivalent input. Cross-check against the literal string format in `_render_objectives:861-862`, `_render_flip_cards:887-889`, `_render_self_check:945-950`, `_render_activities:1141-1145`, `_render_content_sections:1047-1083`. JSON-LD test asserts `to_jsonld_entry()` keys match the exact camelCase the existing `_build_*_metadata` helpers emit. ONE specific test per renderer-emit-site so a regression points at the right block_type. Include `test_to_html_attrs_includes_data_cf_block_id_when_emit_blocks_flag_set` (env-gated; uses `monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS","true")`).
- **Verification:** `pytest Courseforge/scripts/tests/test_block_emitter_html.py Courseforge/scripts/tests/test_block_emitter_jsonld.py -v` reports ≥32 PASSED.

#### Subtask 9: Add snapshot-byte-stable test in `test_block_emitter_html.py` against `_render_objectives` legacy output
- **Files:** `/home/user/Ed4All/Courseforge/scripts/tests/test_block_emitter_html.py` (extend)
- **Depends on:** Subtask 8
- **Estimated LOC:** ~50
- **Change:** New test `test_block_to_html_attrs_byte_equal_to_legacy_render_objectives`: build a fixture objectives list (`[{"id":"TO-01","statement":"Define X","bloom_level":"remember","bloom_verb":"define"}]`); call legacy `_render_objectives(objs)` and capture string; build the equivalent Block; emit `b.to_html_attrs()`; assert the legacy output's `<li ...>` substring contains exactly the bytes returned by `b.to_html_attrs()`. Repeat for one representative case per renderer (objectives / flip_card / self_check / activity / content_section heading / callout). Skips if `COURSEFORGE_EMIT_BLOCKS` is set (the new `data-cf-block-id` attribute would break byte equality).
- **Verification:** `pytest Courseforge/scripts/tests/test_block_emitter_html.py::test_block_to_html_attrs_byte_equal_to_legacy_render_objectives -v` PASSES.

### C. JSON Schema additions

#### Subtask 10: Add `$defs/Block` and `$defs/Touch` to `courseforge_jsonld_v1.schema.json`
- **Files:** `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json:73-255` (extend `$defs`)
- **Depends on:** Subtask 3
- **Estimated LOC:** ~80
- **Change:** Add `"Touch"` $def: `{type:"object", required:["model","provider","tier","timestamp","decisionCaptureId","purpose"], additionalProperties:false, properties:{model:{type:"string"}, provider:{enum:["anthropic","local","together","claude_session","deterministic"]}, tier:{enum:["outline","validation","rewrite"]}, timestamp:{type:"string",format:"date-time"}, decisionCaptureId:{type:"string",minLength:1}, purpose:{type:"string"}}}`. Add `"Block"` $def: `{type:"object", required:["blockId","blockType","sequence"], additionalProperties:false, properties:{blockId:{type:"string",minLength:1}, blockType:{enum:[…16 types…]}, sequence:{type:"integer",minimum:0}, contentHash:{type:"string",pattern:"^[a-f0-9]{64}$"}, touchedBy:{type:"array",items:{$ref:"#/$defs/Touch"}}, pageId:{type:"string"}, bloomLevel:{$ref:"…BloomLevel…"}, contentTypeLabel:{type:"string"}, objectiveIds:{type:"array",items:{type:"string"}}, sourceIds:{type:"array",items:{type:"string"}}, sourcePrimary:{type:"string"}, templateType:{type:"string"}, keyTerms:{type:"array",items:{type:"string"}}, teachingRole:{type:"string"}, purpose:{type:"string"}, component:{type:"string"}, validationAttempts:{type:"integer",minimum:0,description:"Phase-3 regeneration-budget counter; incremented per failed validator pass."}, escalationMarker:{type:"string",enum:["outline_budget_exhausted","structural_unfixable","validator_consensus_fail"],description:"Phase-3 escalation tag; non-null when block was escalated to rewrite tier."}}}`. Both new fields stay optional (not in `required` list); legacy emit without these fields keeps validating.
- **Verification:** `python -c "import json; d=json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); assert 'Block' in d['\$defs'] and 'Touch' in d['\$defs']; assert 'blockId' in d['\$defs']['Block']['required']"` exits 0.

#### Subtask 11: Add top-level optional `blocks[]`, `provenance`, `contentHash` to schema
- **Files:** `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json:38-72` (top-level properties)
- **Depends on:** Subtask 10
- **Estimated LOC:** ~30
- **Change:** Insert `"blocks": {type:"array", items:{$ref:"#/$defs/Block"}, description:"Phase 2: stable in-memory Block intermediate; canonical projection of learningObjectives[]/sections[]/misconceptions[]"}`. Insert `"provenance": {type:"object", additionalProperties:false, properties:{runId:{type:"string"}, pipelineVersion:{type:"string"}, tiers:{type:"array",items:{type:"object",properties:{tier:{enum:["outline","validation","rewrite"]},model:{type:"string"},provider:{type:"string"}}}}}}`. Insert `"contentHash": {type:"string", pattern:"^[a-f0-9]{64}$", description:"sha256 hex of canonical Block payload for re-execution drift detection"}`. Add `$comment` on the top-level object stating the redundancy between `blocks[]` and `learningObjectives[]/sections[]/misconceptions[]` is intentional during the Phase-2 migration window. Required keys list at line 8 stays unchanged.
- **Verification:** `python -c "import json; d=json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); assert 'blocks' in d['properties']; assert 'provenance' in d['properties']; assert 'contentHash' in d['properties']; assert d['additionalProperties']==False; assert d['required']==['@context','@type','courseCode','weekNumber','moduleType','pageId']"` exits 0.

#### Subtask 12: Extend `test_generate_course_jsonld_validation.py` with Block validation cases
- **Files:** `/home/user/Ed4All/Courseforge/scripts/tests/test_generate_course_jsonld_validation.py` (extend)
- **Depends on:** Subtask 11
- **Estimated LOC:** ~80
- **Change:** New tests: `test_jsonld_blocks_array_validates_against_extended_schema` (build a payload with one Block, run JSON-Schema validation, assert valid); `test_jsonld_block_missing_block_id_fails_validation`; `test_jsonld_block_invalid_block_type_fails_validation`; `test_jsonld_provenance_object_validates`; `test_jsonld_content_hash_pattern_enforced` (non-hex string fails); `test_jsonld_legacy_payload_without_blocks_still_validates` (regression — existing pages keep validating). Reuse the test harness pattern from existing tests in this file.
- **Verification:** `pytest Courseforge/scripts/tests/test_generate_course_jsonld_validation.py -k "blocks or provenance or content_hash or legacy_payload" -v` reports ≥6 PASSED.

### D. SHACL shape additions

#### Subtask 13: Add `BlockShape` and `TouchShape` to `courseforge_v1.shacl.ttl`
- **Files:** `/home/user/Ed4All/schemas/context/courseforge_v1.shacl.ttl` (append)
- **Depends on:** Subtask 11
- **Estimated LOC:** ~60
- **Change:** Append two NodeShapes targeting `ed4all:Block` and `ed4all:Touch`. `BlockShape` declares `sh:property` for `ed4all:blockId` (datatype `xsd:string`, `sh:minCount 1`, `sh:maxCount 1`), `ed4all:blockType` (`sh:in` over the 16-value enum), `ed4all:sequence` (datatype `xsd:integer`, `sh:minInclusive 0`), `ed4all:contentHash` (datatype `xsd:string`, `sh:pattern "^[a-f0-9]{64}$"`), `ed4all:touchedBy` (`sh:node ed4all:Touch`, `sh:minCount 0`), `ed4all:validationAttempts` (datatype `xsd:nonNegativeInteger`, `sh:minCount 0`, `sh:maxCount 1` — Phase-3 regeneration-budget counter), `ed4all:escalationMarker` (`sh:in` over `("outline_budget_exhausted" "structural_unfixable" "validator_consensus_fail")`, `sh:maxCount 1` — Phase-3 escalation tag). `TouchShape` declares `sh:property` for `ed4all:tier` (`sh:in` over `outline`/`validation`/`rewrite`), `ed4all:provider` (sh:in over the 5 providers), `ed4all:decisionCaptureId` (`sh:minLength 1`). Mirror the Wave-67 `cfshapes:` namespace conventions at `:413-end`. Add a `@context` mapping for `blockId/blockType/sequence/touchedBy/provenance/contentHash/validationAttempts/escalationMarker` in `schemas/context/courseforge_v1.jsonld` (no `@context` IRI bump) — the existing IRI `https://ed4all.dev/ns/courseforge/v1` resolves them via the `ed4all:` prefix.
- **Verification:** `python -c "from rdflib import Graph; g=Graph(); g.parse('schemas/context/courseforge_v1.shacl.ttl', format='turtle'); from rdflib.namespace import Namespace; sh=Namespace('http://www.w3.org/ns/shacl#'); ed4all=Namespace('https://ed4all.dev/ns/courseforge/v1#'); shapes=set(g.subjects(sh.targetClass, ed4all.Block)); assert shapes, 'BlockShape not registered'"` exits 0.

#### Subtask 14: Extend `test_generate_course_shacl_validation.py` with Block touched_by cardinality test
- **Files:** `/home/user/Ed4All/Courseforge/scripts/tests/test_generate_course_shacl_validation.py` (extend)
- **Depends on:** Subtask 13
- **Estimated LOC:** ~40
- **Change:** New test `test_block_touched_by_cardinality_validated_by_shacl`: emit a payload with a Block carrying empty `touched_by[]` (valid, minCount=0); emit a payload with a Block carrying a Touch missing `decision_capture_id` (invalid, minLength violation); confirm pyshacl `conforms` is True for the valid case and False with a clear violation message for the invalid case. Skip when pyshacl/rdflib unavailable (mirror existing skip pattern in this file).
- **Verification:** `pytest Courseforge/scripts/tests/test_generate_course_shacl_validation.py::test_block_touched_by_cardinality_validated_by_shacl -v` PASSES (or SKIPS cleanly without pyshacl).

### E. course_metadata.schema.json authoring

#### Subtask 15: Author `schemas/knowledge/course_metadata.schema.json` for the Courseforge stub shape
- **Files:** create `/home/user/Ed4All/schemas/knowledge/course_metadata.schema.json`
- **Depends on:** Subtask 11
- **Estimated LOC:** ~80
- **Change:** Draft 2020-12 schema. `$id: "https://ed4all.dev/ns/courseforge/v1/CourseMetadata.schema.json"`. Top-level required: `["course_code", "course_title", "classification", "ontology_mappings"]`. `additionalProperties: false`. Properties match `generate_course.py:2181-2198` literally. NEW optional `blocks_summary` object: `{total_blocks: integer ≥0, by_type: object[block_type → count], hash_root: string ^[a-f0-9]{64}$, outline_only: boolean}`. Add a comment noting the existing `schemas/academic/course_metadata.schema.json` is for MIT-OCW-shaped academic metadata (different surface) and explicitly is NOT this schema. Add wired loading via the same pattern as `_validate_page_jsonld` (lazy `jsonschema` import) for emit-time validation of the stub.
- **Verification:** `python -c "import json,jsonschema; s=json.load(open('schemas/knowledge/course_metadata.schema.json')); jsonschema.Draft202012Validator.check_schema(s); jsonschema.validate({'course_code':'X_101','course_title':'T','classification':{'division':'STEM','primary_domain':'cs','subdomains':[],'topics':[]},'ontology_mappings':{'acm_ccs':[],'lcsh':[]}}, s)"` exits 0.

### F. Renderer migration B1-B6

#### Subtask 16: Migrate `_render_objectives` to build `List[Block]` then emit (B1)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:833-873`
- **Depends on:** Subtasks 6, 9
- **Estimated LOC:** ~30
- **Change:** Refactor `_render_objectives(objectives, *, source_ids, source_primary)` to: (1) build `blocks = [Block(block_id=Block.stable_id(page_id,"objective",o["id"],i), block_type="objective", page_id=<from caller>, sequence=i, content=o["statement"], objective_ids=(o["id"],), bloom_level=…, bloom_verb=…, cognitive_domain=…) for i,o in enumerate(objectives)]`; (2) for each block, emit `<li{block.to_html_attrs()}>...</li>` using `block.content` for the statement; (3) wrap in `.objectives <div>` carrying `_source_attr_string(source_ids, source_primary)` (unchanged). Add a new optional kwarg `page_id: str = ""` so the caller (`generate_week`) can pass it through; default empty so legacy callers (rare — most call from `generate_week`) keep working. The string-emit shape stays byte-identical when `COURSEFORGE_EMIT_BLOCKS=false` (no `data-cf-block-id` appended).
- **Verification:** `pytest Courseforge/scripts/tests/test_lo_multi_verb_emit.py Courseforge/scripts/tests/test_lo_targeted_concepts_emit.py Courseforge/scripts/tests/test_lo_hierarchy_edges_emit.py -v` all PASS (snapshot regression).

#### Subtask 17: Migrate `_render_flip_cards` (B2)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:876-895`
- **Depends on:** Subtask 16
- **Estimated LOC:** ~25
- **Change:** Build one `Block(block_type="flip_card_grid", content={"terms":[{"term":t["term"],"definition":t["definition"]} for t in terms]}, key_terms=tuple(_slugify(t["term"]) for t in terms), …)`; emit the wrapping `.flip-card-grid <div>` and per-term `.flip-card` children using the legacy literal shape but reading attribute strings from sub-blocks (one per term — block_type="flip_card_grid" parent, plus per-card emit handled inline since flip-card attribute set is identical for every card under the same parent purpose). Keep byte-identical with `COURSEFORGE_EMIT_BLOCKS=false`.
- **Verification:** `pytest Courseforge/scripts/tests/ -k "flip_card or content_type" -v` reports all PASS.

#### Subtask 18: Migrate `_render_self_check` (B3)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:898-951`
- **Depends on:** Subtask 17
- **Estimated LOC:** ~40
- **Change:** Build `blocks = [Block(block_type="self_check_question", page_id=…, sequence=i, content={"question":q["question"],"options":q["options"]}, bloom_level=q.get("bloom_level","remember"), objective_ids=(q.get("objective_ref",""),) if q.get("objective_ref") else (), source_ids=tuple(...), source_primary=…) for i,q in enumerate(questions, 1)]`. Emit `.self-check <div>` wrappers using `block.to_html_attrs()` for the data-cf-* attributes. Per-question `source_references` override pattern preserved. Byte-stable with flag off.
- **Verification:** `pytest Courseforge/scripts/tests/test_generate_course_sourcerefs.py -v` reports all PASS.

#### Subtask 19: Migrate `_render_activities` (B4)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1107-1146`
- **Depends on:** Subtask 18
- **Estimated LOC:** ~30
- **Change:** Same pattern as B3: build `blocks = [Block(block_type="activity", …) for i,a in enumerate(activities,1)]`. Emit `.activity-card <div>` carrying `block.to_html_attrs()`. Preserve the per-activity `source_references` override.
- **Verification:** `pytest Courseforge/scripts/tests/ -k "activity or sourcerefs" -v` reports all PASS.

#### Subtask 20: Migrate `_render_content_sections` (B5 — high-risk, Wave-35 ancestor-walk wrapper)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:990-1104`
- **Depends on:** Subtask 19
- **Estimated LOC:** ~120
- **Change:** Most complex renderer. Refactor to: (1) build a `List[Block]` per section — heading block + body paragraphs (held inside `block.content` as the joined paragraphs string) + optional flip_card_grid sub-block + optional callout block; (2) emit the Wave-35 `<section data-cf-source-ids="…">` wrapper exactly as today (`:1037-1046, :1098-1103`) using `_source_attr_string(section_ids, section_primary)`; (3) emit the heading `<{tag}{block.to_html_attrs()}>{heading}</{tag}>` using the heading block; (4) emit `<p>` children (no attributes per Wave-9 P2 decision); (5) emit nested flip_card_grid via `_render_flip_cards`; (6) emit callout via the inline shape (lines `:1064-1083`) using `block.to_html_attrs()` on the `<div class="callout">` wrapper. CRITICAL: every existing test in `test_template_chrome_emit.py`, `test_generate_course_sourcerefs.py`, `test_misconception_bloom_tag_emit.py` MUST stay green at every commit boundary.
- **Verification:** `pytest Courseforge/scripts/tests/ -v` reports the FULL test suite all PASS (regression gate). Single-file gate: `pytest Courseforge/scripts/tests/test_generate_course_sourcerefs.py Courseforge/scripts/tests/test_template_chrome_emit.py -v`.

#### Subtask 21: Migrate `generate_week` inline `<section>` wrappers (B6)
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1834,1907,1945,2009,2015,2052`
- **Depends on:** Subtask 20
- **Estimated LOC:** ~80
- **Change:** Six inline emit sites (`overview_body_attrs`, `app_body_attrs`, `sc_body_attrs`, summary recap + Key Takeaways, discussion `disc_attrs`). Refactor each to construct a `Block(block_type="summary_takeaway"|"discussion_prompt"|"chrome"|…, source_ids=(…), source_primary=…)` and emit the wrapper using `block.to_html_attrs()`. The Wave-43 "Chapter Recap" path at `:2004-2013` becomes a `Block(block_type="recap", content=" ".join(recap_paragraphs))`. Preserve every existing back-compat contract: empty `source_ids` → no wrapper emitted (test_no_map_no_emit).
- **Verification:** `pytest Courseforge/scripts/tests/test_generate_course_sourcerefs.py -v` PASSES; manually diff one fixture's emitted HTML pre-vs-post: `diff <(python -m Courseforge.scripts.generate_course tests/fixtures/sample_course_data.json /tmp/pre/) <(git stash && python -m Courseforge.scripts.generate_course tests/fixtures/sample_course_data.json /tmp/post/ && git stash pop) | wc -l` reports 0 (only meaningful in dev workflow; CI gate is the snapshot tests).

### G. JSON-LD builder migration

#### Subtask 22: Migrate `_build_objectives_metadata` to call `Block.to_jsonld_entry()`
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1331-1421`
- **Depends on:** Subtask 16, 7
- **Estimated LOC:** ~30
- **Change:** Refactor to construct `Block(block_type="objective", …)` per LO and call `[b.to_jsonld_entry() for b in blocks]`. Keep the public signature `_build_objectives_metadata(objectives) -> List[Dict[str, Any]]` stable so `_build_page_metadata` doesn't change. The same Block instances built here can later be reused by Subtask 27 to populate `blocks[]`.
- **Verification:** `pytest Courseforge/scripts/tests/test_lo_multi_verb_emit.py Courseforge/scripts/tests/test_bloom_distribution_emit.py -v` all PASS (regression).

#### Subtask 23: Migrate `_build_sections_metadata` and `_build_misconceptions_metadata`
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1455-1490, 1530-1579`
- **Depends on:** Subtask 22
- **Estimated LOC:** ~50
- **Change:** Same pattern as Subtask 22 for sections and misconceptions. `_build_sections_metadata` builds Block per section heading (block_type chosen from `_infer_content_type`); `_build_misconceptions_metadata` builds `Block(block_type="misconception", content={"misconception":…,"correction":…}, bloom_level=…)`. Public signatures unchanged.
- **Verification:** `pytest Courseforge/scripts/tests/test_misconception_bloom_tag_emit.py Courseforge/scripts/tests/test_content_type_enum_validation.py -v` all PASS.

#### Subtask 24: Migrate `_build_bloom_distribution` to consume Block list
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1493-1527`
- **Depends on:** Subtask 22
- **Estimated LOC:** ~10
- **Change:** Optional refactor — change the input from `objectives_metadata: List[Dict]` to allow either the dict list or `List[Block]`; aggregate from `block.bloom_level` when blocks are passed. Simpler alternative: leave the signature unchanged and call after `_build_objectives_metadata`. Pick simpler-alternative for Phase 2; document the dual-input idea as a deferred improvement. (Net: 0-line code change; just verify the existing flow stays green.)
- **Verification:** `pytest Courseforge/scripts/tests/test_bloom_distribution_emit.py -v` PASSES.

### H. New emit fields behind `COURSEFORGE_EMIT_BLOCKS` flag

#### Subtask 25: Add `_courseforge_emit_blocks_enabled()` helper and env-flag constant
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py` (insert near `:369` alongside `_ENFORCE_SHACL_ENV`)
- **Depends on:** none
- **Estimated LOC:** ~15
- **Change:** Module constant `_EMIT_BLOCKS_ENV = "COURSEFORGE_EMIT_BLOCKS"`. Helper `_courseforge_emit_blocks_enabled() -> bool` mirroring `_shacl_enforcement_enabled` semantics: returns True if env var is set to `"true" / "1" / "yes"` (case-insensitive). Default off.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_EMIT_BLOCKS']='true'; from Courseforge.scripts.generate_course import _courseforge_emit_blocks_enabled; assert _courseforge_emit_blocks_enabled()==True"` exits 0.

#### Subtask 26: Extend `_build_page_metadata` to emit `blocks[]`, `provenance`, `contentHash` when flag on
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1582-1638`
- **Depends on:** Subtasks 22, 23, 25
- **Estimated LOC:** ~40
- **Change:** Add new optional kwarg `blocks: Optional[List[Block]] = None` to `_build_page_metadata`. After the existing field assembly (`:1610-1637`), if `_courseforge_emit_blocks_enabled() and blocks`: append `meta["blocks"] = [b.to_jsonld_entry() for b in blocks]`; append `meta["provenance"] = {"runId": os.environ.get("COURSEFORGE_RUN_ID",""), "pipelineVersion":"phase2", "tiers":[]}` (deterministic baseline; tiers populated only when a multi-tier provider has authored the page); append `meta["contentHash"] = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()).hexdigest()` AFTER all other fields are set. Hash excludes itself.
- **Verification:** `python -c "import os; os.environ['COURSEFORGE_EMIT_BLOCKS']='true'; from Courseforge.scripts.generate_course import _build_page_metadata; from Courseforge.scripts.blocks import Block; b=Block(block_id='x',block_type='objective',page_id='p',sequence=0,content='Define X',objective_ids=('TO-01',),bloom_level='remember',bloom_verb='define',cognitive_domain='factual'); m=_build_page_metadata('C_101',1,'overview','p',objectives=[{'id':'TO-01','statement':'Define X','bloom_level':'remember','bloom_verb':'define'}], blocks=[b]); assert 'blocks' in m and 'contentHash' in m and 'provenance' in m"` exits 0.

#### Subtask 27: Thread `blocks` into `generate_week` and pass to `_build_page_metadata`
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:1761-2081` (the six `_build_page_metadata` call sites)
- **Depends on:** Subtask 26
- **Estimated LOC:** ~60
- **Change:** At each `_build_page_metadata` call site (`:1848-1855, 1878-1886, 1917-1925, 1958-1966, 2030-2037, 2064-2071`), build the corresponding `List[Block]` (objectives + sections + activities + self_check + misconceptions etc. for that page) and pass `blocks=…`. The Block list is constructed by reusing the same Block instances already built by the migrated renderer functions (Subtasks 16-21) — store them in a local variable inside `generate_week` instead of throwing away after emit. Per-page `blocks_summary` is collected by `generate_course` and folded into `course_metadata.json` (Subtask 30).
- **Verification:** `COURSEFORGE_EMIT_BLOCKS=true python -m Courseforge.scripts.generate_course Courseforge/scripts/tests/fixtures/<sample>.json /tmp/blocks_out/ && grep -l '"blocks":' /tmp/blocks_out/week_*/*.html | head -3` returns ≥1 file. With flag off: `python -m Courseforge.scripts.generate_course Courseforge/scripts/tests/fixtures/<sample>.json /tmp/legacy_out/ && ! grep -l '"blocks":' /tmp/legacy_out/week_*/*.html` returns nothing (legacy path byte-stable).

### I. `--emit-mode {full|outline}` on `generate_course.py`

#### Subtask 28: Add `--emit-mode` CLI flag and outline filter
- **Files:** `/home/user/Ed4All/Courseforge/scripts/generate_course.py:2207-2265` (CLI parser) and `:1761-2081` (generate_week)
- **Depends on:** Subtask 27
- **Estimated LOC:** ~50
- **Change:** New CLI arg `--emit-mode`, choices `["full","outline"]`, default `"full"`. In `generate_course`, accept new kwarg `emit_mode: str = "full"`; pass to `generate_week`. In `generate_week`, when `emit_mode == "outline"`: only blocks with `block_type ∈ {"objective","prereq_set","summary_takeaway","chrome","recap"}` are rendered to HTML (filter the block list before emit); content/example/explanation/assessment_item/activity/self_check_question blocks are still emitted to JSON-LD `blocks[]` but their HTML body is empty. Stamp `blocks_summary.outline_only=true` on `course_metadata.json` when emit_mode=="outline".
- **Verification:** `python -m Courseforge.scripts.generate_course Courseforge/scripts/tests/fixtures/<sample>.json /tmp/outline_out/ --emit-mode outline && grep -c "data-cf-content-type" /tmp/outline_out/week_01/*content*.html | sort -u` returns `0` (content-type attrs absent in outline mode); `grep -l "outline_only.*true" /tmp/outline_out/course_metadata.json` returns the file.

### J. `--outline-only` on `package_multifile_imscc.py`

#### Subtask 29: Add `--outline-only` CLI flag to packager
- **Files:** `/home/user/Ed4All/Courseforge/scripts/package_multifile_imscc.py:116-213` (build_manifest) and `:334-355` (CLI parser)
- **Depends on:** Subtask 28
- **Estimated LOC:** ~40
- **Change:** New CLI arg `--outline-only` (`action="store_true"`). Plumb into `build_manifest(content_dir, course_code, course_title, *, outline_only: bool = False)`. When `outline_only`: in the per-week `html_files` walk (`:192`), filter to only `*overview.html` and `*summary.html` (drop content/application/self_check/discussion). Tag the manifest `<lom_el> <general> <description>` text (`:147`) with prefix `"[OUTLINE] "`. The `course_metadata.json` augmentation happens in `generate_course.py` (Subtask 28); the packager only consumes it.
- **Verification:** `python -m Courseforge.scripts.package_multifile_imscc /tmp/outline_out/ /tmp/x.imscc COURSE_101 'T' --outline-only && unzip -l /tmp/x.imscc | grep -E "content|application|self_check|discussion" | wc -l` returns `0`; `unzip -l /tmp/x.imscc | grep -E "overview|summary"` returns ≥2.

### K. Trainforge consumer parallel path

#### Subtask 30: Add `_extract_blocks_from_jsonld` to `html_content_parser.py`
- **Files:** `/home/user/Ed4All/Trainforge/parsers/html_content_parser.py:445-456` (after `_extract_json_ld`)
- **Depends on:** Subtask 11 (schema) — independent of Courseforge changes (Trainforge can read either path)
- **Estimated LOC:** ~80
- **Change:** New method `_extract_blocks_from_jsonld(self, json_ld: Dict[str,Any]) -> List[Dict[str,Any]]`. When `json_ld.get("blocks")` is present: return it directly (callers can map to `ContentSection` / `LearningObjective`). When absent: return empty list (callers fall through to the existing regex DOM walk). Add a new `parse()` flow step (`:255-330`) that prefers `blocks[]` when present: build `ContentSection` entries from `block.block_type ∈ {"explanation","example","procedure","comparison","definition","overview","summary","exercise"}` blocks with `content_type = block.contentTypeLabel`, `key_terms = [t for t in block.keyTerms]`, `template_type = block.templateType`. The legacy `_extract_sections` regex DOM walk remains as fallback for non-Courseforge IMSCC.
- **Verification:** `python -c "from Trainforge.parsers.html_content_parser import HTMLContentParser; p=HTMLContentParser(); html='<html><script type=\"application/ld+json\">{\"@context\":\"https://ed4all.dev/ns/courseforge/v1\",\"@type\":\"CourseModule\",\"courseCode\":\"X_101\",\"weekNumber\":1,\"moduleType\":\"content\",\"pageId\":\"p\",\"blocks\":[{\"blockId\":\"x\",\"blockType\":\"explanation\",\"sequence\":0,\"contentTypeLabel\":\"explanation\"}]}</script></html>'; m=p.parse(html); assert any(s.content_type==\"explanation\" for s in m.sections)"` exits 0.

#### Subtask 31: Update `process_course._extract_section_metadata` to honor `blocks[]` when present
- **Files:** `/home/user/Ed4All/Trainforge/process_course.py:2259-2417`
- **Depends on:** Subtask 30
- **Estimated LOC:** ~30
- **Change:** Add a NEW priority tier ABOVE the existing JSON-LD `cf_meta["sections"]` lookup (`:2335`): if `cf_meta.get("blocks")`, walk those first; for each block with `block_type` in the heading-types set, match `cand` heading to the block's content (block content carries the heading for the section blocks). When a matching block is found, populate `bloom_level / content_type_label / key_terms` from it. Trace value `"jsonld_blocks_match"`. If no match in `blocks[]`, fall through to the existing `cf_meta["sections"]` path (preserves back-compat).
- **Verification:** `pytest Trainforge/tests/ -k "metadata or section_metadata" -v` reports all PASS (regression gate). Add unit: `pytest Trainforge/tests/test_metadata_extraction.py -v` ≥ existing tests still PASS.

### L. Contract test (Trainforge ⇄ Courseforge)

#### Subtask 32: Author `tests/test_block_contract_trainforge.py`
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/tests/test_block_contract_trainforge.py`
- **Depends on:** Subtasks 27, 30, 31
- **Estimated LOC:** ~140
- **Change:** Per high-level §6 plan. For each `block_type` in BLOCK_TYPES: build a Block, emit via `_build_page_metadata` + `_wrap_page` (full HTML page), parse via `Trainforge.parsers.html_content_parser.HTMLContentParser.parse()`, assert key fields equal across BOTH consume paths. Specifically: (1) parse with `COURSEFORGE_EMIT_BLOCKS=false` (legacy regex DOM walk only) → capture `bloom_level`, `content_type_label`, `key_terms`, `objective_refs`, `source_references`, `template_type` per ContentSection. (2) parse with `COURSEFORGE_EMIT_BLOCKS=true` (new JSON-LD blocks[] path) → capture same fields. (3) Assert per-section equality across the two parse runs. Skip-if-not-applicable for `block_type ∈ {"chrome","prereq_set","misconception"}` (those don't carry section equivalents in the consumer).
- **Verification:** `pytest Courseforge/scripts/tests/test_block_contract_trainforge.py -v` reports ≥13 PASSED.

### M. Round-trip integration test

#### Subtask 33: Author `tests/test_block_roundtrip.py`
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/tests/test_block_roundtrip.py`
- **Depends on:** Subtask 32
- **Estimated LOC:** ~80
- **Change:** Idempotency: `test_emit_parse_emit_byte_equal_html` — emit a representative course (one week, one content page) → parse via Trainforge → reconstruct the Block list from the parsed JSON-LD `blocks[]` → re-emit through `_build_page_metadata` + `_wrap_page` → assert HTML byte-equal AND JSON-LD payload (modulo ordered keys) byte-equal between the two emits. `test_emit_parse_emit_byte_equal_jsonld_blocks_array` — same but compare only the `blocks[]` array ordering and values.
- **Verification:** `pytest Courseforge/scripts/tests/test_block_roundtrip.py -v` reports ≥2 PASSED.

### N. Provenance audit test

#### Subtask 34: Author `tests/test_block_provenance_chain.py`
- **Files:** create `/home/user/Ed4All/Courseforge/scripts/tests/test_block_provenance_chain.py`
- **Depends on:** Subtask 27
- **Estimated LOC:** ~70
- **Change:** Synthesise a 3-tier run: build a Block, call `b1 = b.with_touch(Touch(tier="outline",...))`, then `b2 = b1.with_touch(Touch(tier="validation",...))`, then `b3 = b2.with_touch(Touch(tier="rewrite",...))`. Assert: `len(b3.touched_by) == 3`; the three timestamps are strictly monotonically increasing; `every t.decision_capture_id != ""` (Wave 112 invariant); the tier sequence is exactly `["outline","validation","rewrite"]`. Then emit `b3` via `to_jsonld_entry()` and assert `entry["touchedBy"]` is a 3-element list with the right `tier`/`provider`/`model` fields.
- **Verification:** `pytest Courseforge/scripts/tests/test_block_provenance_chain.py -v` reports ≥1 PASSED.

### O. Phase 1 wire-in update

#### Subtask 35: Widen `ContentGeneratorProvider.generate_page` to return `Block` (not `str`)
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py:259-317`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~40
- **Change:** Change return type annotation from `-> str` to `-> Block`. Update the docstring: replace "Returns rendered HTML as a `str`." with "Returns a `Block` carrying the rendered prose, parsed structure, and a single Touch entry." After `text, retry_count = self._dispatch_call(user_prompt)` and `_emit_decision`, parse the returned HTML via the same minimal regex `_parse_provider_page_html` currently in `_content_gen_helpers.py:_parse_provider_page_html` (move it into `Courseforge/scripts/blocks.py` as a module-level helper). Construct `block = Block(block_id=Block.stable_id(page_id, "explanation", _slugify(heading or page_id), 0), block_type="explanation", page_id=page_id, sequence=0, content=" ".join(paragraphs), key_terms=tuple(_slugify(k) for k in page_context.get("key_terms",[])), source_ids=tuple(...), …)`. Append a Touch: `block = block.with_touch(Touch(model=self._model, provider=self._provider, tier="outline", timestamp=datetime.utcnow().isoformat()+"Z", decision_capture_id=self._last_capture_id(), purpose="draft"))`. Return the Block. Add a private `_last_capture_id()` returning the basename:idx of the last appended decision-capture event.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -v` reports all PASS (the existing tests assert on string content; update them in Subtask 36 to assert on `Block.content`).

#### Subtask 36: Update Phase 1 wire-in `_build_content_modules_dynamic` to consume `Block` directly
- **Files:** `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py:1860-1905`
- **Depends on:** Subtask 35
- **Estimated LOC:** ~30
- **Change:** Delete `_parse_provider_page_html` from this file (moved into `blocks.py` per Subtask 35 — or kept here as a thin shim for one wave for safety). Replace `rendered_html = content_provider.generate_page(...)` then-parse pattern with: `block = content_provider.generate_page(...)`. Read `block.content` (string) directly into `sections[0]["paragraphs"]`. Read `block.key_terms` into the section. Update tests: `Courseforge/tests/test_content_generator_provider.py::test_local_backend_routes_to_local_base_url` and analogues — assert `isinstance(result, Block)` and `result.content` contains the expected paragraph text instead of `"<section>" in result`.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -v` ≥7 PASSED. `pytest Courseforge/scripts/tests/test_generate_course_sourcerefs.py -v` PASSES (regression — provider=None path unchanged).

### P. Documentation

#### Subtask 37: Add Block section to `Courseforge/CLAUDE.md`
- **Files:** `/home/user/Ed4All/Courseforge/CLAUDE.md` (insert after the "Metadata Output" section, ~line 200)
- **Depends on:** Subtasks 11, 27
- **Estimated LOC:** ~50
- **Change:** New section `### Phase 2: intermediate Block format`. Describe the dataclass briefly; link to `Courseforge/scripts/blocks.py`; describe the BLOCK_TYPES enum; describe the JSON-LD `blocks[]` field (gated behind `COURSEFORGE_EMIT_BLOCKS`); cross-link to `schemas/knowledge/courseforge_jsonld_v1.schema.json`. Add a one-row addition to the "HTML Data Attributes" table (`:Courseforge/CLAUDE.md:~250`): `data-cf-block-id` | every block emit | "stable Block ID for cross-referencing JSON-LD `blocks[]`" (gated behind `COURSEFORGE_EMIT_BLOCKS`).
- **Verification:** `grep -c "Phase 2: intermediate Block format" Courseforge/CLAUDE.md` returns 1; `grep "data-cf-block-id" Courseforge/CLAUDE.md` returns ≥1 line.

#### Subtask 38: Add `COURSEFORGE_EMIT_BLOCKS` row to root `CLAUDE.md` opt-in flags table
- **Files:** `/home/user/Ed4All/CLAUDE.md:728` (insert before `COURSEFORGE_PROVIDER` row at `:729`)
- **Depends on:** Subtask 37
- **Estimated LOC:** ~5
- **Change:** Insert one row matching prose density of the `COURSEFORGE_PROVIDER` row. Content: name (`COURSEFORGE_EMIT_BLOCKS`); values (`true` / `false`, default `false`); behavior (when truthy, `Courseforge/scripts/generate_course.py::_build_page_metadata` emits the new `blocks[]` / `provenance` / `contentHash` JSON-LD fields and stamps `data-cf-block-id` on every block-bearing wrapper; default off keeps emit byte-stable for the legacy snapshot regression suite); cross-link to `schemas/knowledge/courseforge_jsonld_v1.schema.json` and `Courseforge/scripts/blocks.py`. Note: rolls forward to `true` after one wave's byte-stable confirmation; flag drops in a later phase.
- **Verification:** `grep -B1 "COURSEFORGE_PROVIDER" CLAUDE.md | grep -c COURSEFORGE_EMIT_BLOCKS` returns 1.

#### Subtask 39: Document `--emit-mode` and `--outline-only` in `Courseforge/CLAUDE.md`
- **Files:** `/home/user/Ed4All/Courseforge/CLAUDE.md` (Scripts table, ~line 350)
- **Depends on:** Subtasks 28, 29
- **Estimated LOC:** ~10
- **Change:** Update the `generate_course.py` and `package_multifile_imscc.py` rows in the Scripts table. Add a one-paragraph note: "`--emit-mode outline` (`generate_course.py`) and `--outline-only` (`package_multifile_imscc.py`) produce a stripped-down deliverable carrying only objectives + summaries; content/example/assessment HTML bodies are dropped while their JSON-LD `blocks[]` entries persist for downstream consumers (Trainforge `process_course.py` skips `instruction_pair` extraction when `course_metadata.blocks_summary.outline_only=true`). Outline mode is the input shape Phase 3's two-pass pipeline expects from the outline tier."
- **Verification:** `grep -c "emit-mode outline" Courseforge/CLAUDE.md` returns ≥1; `grep -c "outline-only" Courseforge/CLAUDE.md` returns ≥1.

#### Subtask 40: Update `Trainforge/CLAUDE.md` "Metadata Extraction" section
- **Files:** `/home/user/Ed4All/Trainforge/CLAUDE.md` (Metadata Extraction section near top)
- **Depends on:** Subtask 31
- **Estimated LOC:** ~6
- **Change:** Add one bullet to the "extraction priority" list: "0. **JSON-LD `blocks[]`** (Phase 2, highest fidelity when `COURSEFORGE_EMIT_BLOCKS=true`): canonical projection of the Block dataclass; `block_type` carries content-type, `bloom_level`, `key_terms`, `template_type` directly per block." Add the table row showing where Phase 2's `blocks[]` reads in `process_course._extract_section_metadata`.
- **Verification:** `grep -c "JSON-LD .blocks" Trainforge/CLAUDE.md` returns ≥1.

---

## Execution sequencing

**Strict-serial within categories (must run in order):**
- A: Subtask 1 → 2 → 3 → 4 → 5.
- B: Subtask 6 → 7 → 8 → 9.
- C: Subtask 10 → 11 → 12.
- D: Subtask 13 → 14.
- F: Subtask 16 → 17 → 18 → 19 → 20 → 21 (renderer migration; B5 / B6 must serialise; B1-B4 are listed sequentially here for readability but see parallel batch below).
- G: Subtask 22 → 23 → 24 (after F lands).
- H: Subtask 25 (independent) → 26 → 27 (after G).
- I, J: Subtask 28 → 29 (each independent, but 29 depends on 28 because the packager reads `course_metadata.blocks_summary` written by 28).
- K: Subtask 30 → 31.
- L, M, N: Subtask 32 → 33 → 34 (each builds on 27 + 30).
- O: Subtask 35 → 36 (signature change + wire-in update).
- P: Subtasks 37, 38, 39, 40 (after code lands).

**Parallelisable batches:**

- **Week 1 — foundations (mostly serial):**
  - Day 1-2: Subtask 1 → 2 → 3 → 4 → 5 (dataclass + tests).
  - Day 3: Subtask 6 → 7 → 8 → 9 (BlockEmitter + byte-stable tests).
  - Day 4: Subtask 10 → 11 → 12 (JSON Schema), Subtask 13 → 14 (SHACL) — parallelisable.
  - Day 5: Subtask 15 (course_metadata schema), Subtask 25 (env-flag helper) — parallelisable.

- **Week 2 — renderer migration (mostly parallelisable; B5/B6 must serialise):**
  - **Parallel batch (two engineers possible):** Subtasks 16 (B1) + 17 (B2) + 18 (B3) + 19 (B4) — independent renderers, independent fixture files. Each lands as its own commit + snapshot test green.
  - **Solo (highest risk):** Subtask 20 (B5 — `_render_content_sections` + Wave-35 ancestor-walk wrapper). Take a full week. Strictly after B1-B4.
  - **Solo:** Subtask 21 (B6 — `generate_week` inline wrappers). Strictly after B5 because B6 builds on the same Wave-43 recap path.

- **Week 3 — JSON-LD + provenance + outline-only + consumer + tests (parallelisable):**
  - **Parallel batch A:** Subtasks 22 + 23 + 24 (JSON-LD builder migration). Two engineers possible if 22 lands first.
  - **Parallel batch B:** Subtasks 26 → 27 (new emit fields). Sequential. After batch A.
  - **Parallel batch C:** Subtask 28 (CLI flag) + Subtask 29 (packager flag) + Subtask 30 (consumer parallel path) + Subtask 31 (process_course path). Three engineers possible — independent surfaces.
  - **Sequential after:** Subtasks 32 + 33 + 34 (contract / round-trip / provenance tests). Each depends on the prior batch.
  - **Sequential after:** Subtasks 35 + 36 (Phase 1 wire-in update — provider returns Block, consumer dispatches Block).
  - **Parallel:** Subtasks 37 + 38 + 39 + 40 (docs).

**Migration rollout strategy** (mirrors high-level plan §8):
1. **Wave N** — Land Subtasks 1-24 + 35 + 36 (dataclass, BlockEmitter, schema additions, all renderer migrations, JSON-LD builder migration, Phase 1 provider widening). **`COURSEFORGE_EMIT_BLOCKS=false`** by default. Snapshot tests confirm byte-stable emit. Block dataclass is the single source of truth internally even though no new wire fields appear.
2. **Wave N+1** — Land Subtasks 25-34 + 37-40 (env-flag helper, new emit fields, outline mode, consumer parallel path, tests, docs). Default still `false`. Operators can opt in to `true` to test the new consume path.
3. **Wave N+2** — Flip `COURSEFORGE_EMIT_BLOCKS` default to `true`. Trainforge's parallel consumer path becomes primary; legacy DOM regex walk stays as fallback.
4. **Phase 2 followup (NOT this plan):** Drop the `COURSEFORGE_EMIT_BLOCKS` flag entirely — `blocks[]` becomes always-on. Drop redundant `data-cf-content-type / data-cf-bloom-level / data-cf-key-terms` HTML attributes. Keep `data-cf-block-id` and `data-cf-source-ids` and `data-cf-role="template-chrome"` for ancestor-walk and template-chrome filtering.

---

## Final smoke test

A single end-to-end verification an operator runs to prove Phase 2 landed:

```bash
# 1. Run the full unit + integration test suite for the new module:
pytest Courseforge/scripts/tests/test_block_dataclass.py \
       Courseforge/scripts/tests/test_block_emitter_html.py \
       Courseforge/scripts/tests/test_block_emitter_jsonld.py \
       Courseforge/scripts/tests/test_block_contract_trainforge.py \
       Courseforge/scripts/tests/test_block_roundtrip.py \
       Courseforge/scripts/tests/test_block_provenance_chain.py -v

# 2. Snapshot regression: emit byte-stable when flag off.
unset COURSEFORGE_EMIT_BLOCKS
pytest Courseforge/scripts/tests/ -v

# 3. End-to-end with the new emit on:
export COURSEFORGE_EMIT_BLOCKS=true
python -m Courseforge.scripts.generate_course \
  Courseforge/scripts/tests/fixtures/<sample_course_data>.json /tmp/blocks_out/ \
  --division STEM --primary-domain computer-science

# 4. Verify the new fields are present in the JSON-LD:
grep -l '"blocks":' /tmp/blocks_out/week_*/*.html | head -3
grep -l '"contentHash":' /tmp/blocks_out/week_*/*.html | head -3
grep -l '"provenance":' /tmp/blocks_out/week_*/*.html | head -3

# 5. Verify the new data-cf-block-id attribute on at least one wrapper:
grep -l 'data-cf-block-id="' /tmp/blocks_out/week_*/*.html | head -3

# 6. Verify the schema validates the emit:
python -c "import json,jsonschema; s=json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); import re; html=open('/tmp/blocks_out/week_01/week_01_overview.html').read(); m=re.search(r'<script type=\"application/ld\\+json\">(.+?)</script>', html, re.DOTALL); jsonschema.validate(json.loads(m.group(1)), s)"

# 7. Verify the Trainforge consumer reads via blocks[] preferentially:
python -c "from Trainforge.parsers.html_content_parser import HTMLContentParser; m=HTMLContentParser().parse(open('/tmp/blocks_out/week_01/week_01_content_01_intro.html').read()); print('parsed sections:', len(m.sections), 'first contentType:', m.sections[0].content_type if m.sections else None)"

# 8. Outline-mode end-to-end:
python -m Courseforge.scripts.generate_course \
  Courseforge/scripts/tests/fixtures/<sample>.json /tmp/outline_out/ --emit-mode outline
python -m Courseforge.scripts.package_multifile_imscc \
  /tmp/outline_out/ /tmp/outline.imscc COURSE_101 'T' --outline-only
unzip -l /tmp/outline.imscc | grep -E "overview|summary"   # ≥2
unzip -l /tmp/outline.imscc | grep -cE "content|application|self_check|discussion"  # 0
grep '"outline_only": true' /tmp/outline_out/course_metadata.json

# 9. Provenance chain audit on a Phase-1-provider-authored page:
export COURSEFORGE_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
ed4all run textbook_to_course --course-code DEMO_201 --weeks 1
jq -r 'select(.decision_type=="content_generator_call") | .metadata.page_id' \
  training-captures/courseforge/DEMO_201/phase_content-generator/decisions_*.jsonl | sort -u | wc -l   # ≥1
grep -l '"touchedBy":' Courseforge/exports/*DEMO_201*/03_content_development/week_01/*content*.html | head -1
```

**Acceptance criteria:** all `pytest` invocations PASS; commands 4-7 return non-empty as documented; command 8 produces an outline-only IMSCC with the file contracts above; command 9's `touchedBy[]` chain has ≥1 entry per provider-authored page with non-empty `decisionCaptureId`.

---

### Critical Files for Implementation
- `/home/user/Ed4All/Courseforge/scripts/blocks.py` (NEW — Block + Touch dataclass + emitter)
- `/home/user/Ed4All/Courseforge/scripts/generate_course.py` (renderer migration + new emit fields + `--emit-mode`)
- `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json` (Block / Touch / blocks[] / provenance / contentHash additions)
- `/home/user/Ed4All/Trainforge/parsers/html_content_parser.py` (`_extract_blocks_from_jsonld` parallel consumer)
- `/home/user/Ed4All/Courseforge/generators/_provider.py` (Phase 1 widening: returns `Block` not `str`)
