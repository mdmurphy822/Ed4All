# Phase 6 Detailed Execution Plan — ABCD Framework + Concept Extractor Decoupling + Phase 3c Env-Var Fixes

Refines `plans/courseforge_architecture_roadmap.md` §3.1 (ABCD), §3.4 (concept extractor decoupling), §6.4, §6.6 into atomic subtasks. **Depends on:** Phase 2 (Block dataclass), Phase 3 (router seam), Phase 4 (embedding seam), Phase 7a (chunker package — concept extractor consumes its output).

---

## Investigation findings (locked)

- **`lib/ontology/learning_objectives.py`** is a 245-line module exposing `mint_lo_id`, `validate_lo_id`, `hierarchy_from_id`, `split_terminal_chapter`, `assign_lo_ids`. **No ABCD support today.** Phase 6 adds `compose_abcd_prose(abcd: dict) -> str` and `BLOOMS_VERBS: Dict[str, FrozenSet[str]]` lookup.
- **`schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.LearningObjective`** at `:113-201` carries the existing LO definition with `id`, `statement`, `bloomLevel`, `bloomVerb`, `cognitiveDomain`, `keyConcepts`, etc. — no `abcd` field. Phase 6 adds `properties.abcd: {$ref: "#/$defs/AbcdObjective"}` and a new `$defs.AbcdObjective` per roadmap §6.4 recommendation.
- **`MCP/tools/pipeline_tools.py::_plan_course_structure`** at `:2517-2685` synthesizes objectives via `_cgh.synthesize_objectives_from_topics` and persists as `synthesized_objectives.json`. Phase 6 widens the schema; existing fixtures continue to validate (the new `abcd` field is optional).
- **`Courseforge/agents/course-outliner.md`** is a 488-line agent prompt. Phase 6 amends the prompt to instruct the synthesizer to emit ABCD as discrete fields.
- **`Trainforge/pedagogy_graph_builder.py`** is a 1021-line module. The entry point is `build_pedagogy_graph(chunks, objectives, course_id=None, modules=None, concept_classes=None) -> Dict[str, Any]`. Currently invoked from `Trainforge/process_course.py:3624-3628` inside `libv2_archival`-aligned graph emission.
- **No standalone `concept_extraction` workflow phase exists** — verified. Currently every concept-graph build is bundled into `libv2_archival`. Phase 6 introduces a new phase between `chunking` (Phase 7a) and `course_planning`.
- **`config/workflows.yaml::textbook_to_course`** has `objective_extraction → source_mapping → course_planning` chain at `:562-665`. Phase 6 inserts `concept_extraction` between `source_mapping` and `course_planning` (per roadmap §3.4: "between course_planning and content_generation_outline ... or, more strictly per the canonical chain, between chunking and course_planning"). Pre-resolved decision: place between `chunking` (NEW Phase 7a) and `course_planning` so the graph informs synthesizer.
- **No `lib/validators/concept_graph.py` exists** — verified. Phase 6 creates it.
- **No `lib/ontology/concept_objective_linker.py` exists** — verified. Per roadmap §6.6 recommendation, Phase 6 creates a two-stage linker as an explicit pass between concept extraction and content generation.
- **DART hardcoded models at 4 files**: `claude_processor.py:228`, `alt_text_generator.py:42`, `cli.py:99-100`, `converter.py:69`. **`MCP/orchestrator/llm_backend.py:48`** has `DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"`; the file claims env-var override but `:791` shows it falls back hardcoded with `or DEFAULT_ANTHROPIC_MODEL` — env-var-first chain is intact only when the call site sets `model` explicitly.
- **Pedagogy graph builder consumes chunks of shape v4**: `chunks` is `List[Dict]` with `concept_tags`, `learning_outcome_refs`, `source.module_id`, etc. Phase 7a's chunker package emits the same v4 shape (verified via `Trainforge/process_course.py:1432-1748`).
- **`schemas/taxonomies/`** carries `bloom_verbs.json` per roadmap citation. Phase 6's `BLOOMS_VERBS` lookup table mirrors this.

---

## Pre-resolved decisions

1. **ABCD schema location (per roadmap §6.4 recommendation).** Co-locate as `$defs.AbcdObjective` inside `courseforge_jsonld_v1.schema.json`; `$defs.LearningObjective.properties.abcd` references it.
2. **ABCD shape.** `{audience: str, behavior: {verb: str, action_object: str}, condition: str, degree: str}`. All four fields required when `abcd` is present.
3. **`compose_abcd_prose` deterministic format.** `"{audience} will {verb} {action_object} {condition}, {degree}."` Examples: `"Students will identify the parts of a cell from a labeled diagram, with 90% accuracy."` Spaces, capitalization, terminal period are mechanical.
4. **`BLOOMS_VERBS` lookup table source-of-truth.** `lib/ontology/learning_objectives.py::BLOOMS_VERBS: Dict[str, FrozenSet[str]]` keyed on canonical Bloom levels (`remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`). Values are FrozenSet of verbs from `schemas/taxonomies/bloom_verbs.json` (the existing taxonomy file). Phase 6 adds the in-Python projection; verbs themselves stay sourced from the JSON.
5. **`AbcdObjectiveValidator` location and contract.** New `lib/validators/abcd_objective.py::AbcdObjectiveValidator`. For each LO with `abcd` field present: assert `abcd.behavior.verb in BLOOMS_VERBS[lo.bloom_level]`. On miss emit `action="regenerate"` (per Phase 4 contract). Replaces a hypothetical `bloom_verb_mismatch` validator that would have lived in Phase 4 — fold-in per roadmap meta-pattern §3.12.
6. **Concept-extraction phase placement.** Between `chunking` (Phase 7a `dart_chunking`) and `course_planning`. Phase definition has `agents: [pedagogy-graph-builder]` (NEW agent spec); inputs are `dart_chunks_path` (from `chunking` phase) + `objectives_path` (NOT YET — it doesn't exist yet at this point in the chain). The phase reads ONLY `dart_chunks_path` per roadmap §6.6 recommendation (DART chunks only, not objectives).
7. **Concept-extractor → objective-synthesizer linkage (per §6.6 two-stage).** Concept extraction emits `concept_graph/concept_graph_semantic.json`. Objective synthesizer (`plan_course_structure`) reads it via `concept_graph_path` input. Then `concept_objective_linker.py` runs as a deterministic pass between objective synthesis and content generation, populating `LearningObjective.keyConcepts[]` from concept-graph slugs that match.
8. **Concept-graph hash in provenance chain.** New `concept_graph_sha256` field in `LibV2/courses/<slug>/manifest.json`. Required by `LibV2ManifestValidator` post-Phase-6 (Phase 7c extends with both chunkset hashes; this lands the concept-graph hash earlier).
9. **Per-edge provenance (per roadmap §6.1 recommendation).** Add edge-level provenance `edges[].provenance: {chunk_ids: List[str], rule_version: str}` gated behind `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE` env var (default off for legacy corpora; flip on for Phase 6 emit).
10. **Phase 3c env-var fixes.** Five files: `DART/pdf_converter/{cli,alt_text_generator,converter,claude_processor}.py` + `MCP/orchestrator/llm_backend.py`. Each reads `DART_CLAUDE_MODEL` / `MCP_ORCHESTRATOR_LLM_MODEL` env var BEFORE the hardcoded literal. Audit `_post_with_retry` paths to ensure resolved value flows through.
11. **Course-outliner agent prompt amendment.** The agent emits ABCD as constrained-decoding-validated structured output (JSON Schema sample-time per Phase 3 §2.1.1 pattern). The course-outliner agent gets a JSON-Schema-shaped output contract appended; `plan_course_structure` validates it.

---

## Atomic subtasks

Estimated total LOC: ~3,200 (200 ABCD schema + 250 BLOOMS_VERBS + 200 compose_abcd_prose + 250 AbcdObjectiveValidator + 350 course-outliner amendment + 80 plan_course_structure widening + 400 concept_extraction phase + 350 concept_graph validator + 250 concept_objective_linker + 100 phase 3c env-var + 600 tests + 150 docs).

### A. ABCD schema + ontology helpers (5 subtasks)

#### Subtask 1: Add `$defs.AbcdObjective` + `LearningObjective.properties.abcd` to courseforge JSON-LD schema
- **Files:** `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json:113-201`
- **Depends on:** none
- **Estimated LOC:** ~60
- **Change:** Add `$defs.AbcdObjective` definition: `{type: "object", required: ["audience","behavior","condition","degree"], additionalProperties: false, properties: {audience: {type: "string", minLength: 1}, behavior: {type: "object", required: ["verb","action_object"], additionalProperties: false, properties: {verb: {type:"string", minLength:1}, action_object: {type:"string", minLength:1}}}, condition: {type: "string"}, degree: {type: "string"}}}`. Add `LearningObjective.properties.abcd: {$ref: "#/$defs/AbcdObjective", description: "Phase 6: ABCD framework discrete fields. Optional for legacy LOs; required for new emit."}`.
- **Verification:** `python -c "import json,jsonschema; s=json.load(open('schemas/knowledge/courseforge_jsonld_v1.schema.json')); jsonschema.Draft202012Validator.check_schema(s); assert 'AbcdObjective' in s['\$defs']"` exits 0.

#### Subtask 2: Add `BLOOMS_VERBS` lookup table to `lib/ontology/learning_objectives.py`
- **Files:** `/home/user/Ed4All/lib/ontology/learning_objectives.py`
- **Depends on:** none
- **Estimated LOC:** ~60
- **Change:** Add module-level `BLOOMS_VERBS: Dict[str, FrozenSet[str]]` constructed from `schemas/taxonomies/bloom_verbs.json` at import (lru_cache'd). Keys: `remember`, `understand`, `apply`, `analyze`, `evaluate`, `create`. Values: frozenset of verbs from the schema's `BloomLevel` defs.
- **Verification:** `python -c "from lib.ontology.learning_objectives import BLOOMS_VERBS; assert 'remember' in BLOOMS_VERBS and 'identify' in BLOOMS_VERBS['remember']"` exits 0.

#### Subtask 3: Add `compose_abcd_prose(abcd) -> str` function
- **Files:** `/home/user/Ed4All/lib/ontology/learning_objectives.py`
- **Depends on:** Subtask 2
- **Estimated LOC:** ~80
- **Change:** Function builds prose per pre-resolved decision #3 format. Validates ABCD shape on entry (TypeError on missing fields). Strips trailing periods from inputs to avoid double-periods. Returns capitalized first letter + terminal period.
- **Verification:** `python -c "from lib.ontology.learning_objectives import compose_abcd_prose; out=compose_abcd_prose({'audience':'Students','behavior':{'verb':'identify','action_object':'cell parts'},'condition':'from a labeled diagram','degree':'with 90% accuracy'}); assert out=='Students will identify cell parts from a labeled diagram, with 90% accuracy.'"` exits 0.

#### Subtask 4: Create `lib/validators/abcd_objective.py::AbcdObjectiveValidator`
- **Files:** create `/home/user/Ed4All/lib/validators/abcd_objective.py`; edit `/home/user/Ed4All/schemas/events/decision_event.schema.json::properties.decision_type.enum`
- **Depends on:** Subtask 3
- **Estimated LOC:** ~150
- **Change:** Class with `validate(inputs)`. Reads LOs from `inputs["objectives"]` (or `synthesized_objectives_path`). For each LO: when `abcd` present, assert `abcd.behavior.verb.lower() in BLOOMS_VERBS[lo.bloom_level]`. On miss: `action="regenerate"` + GateIssue with `code="ABCD_VERB_BLOOM_MISMATCH"` and message naming the LO ID + verb + Bloom level + valid verb set. When `abcd` absent on a LO that requires it (Phase 6 contract: every newly-emitted LO has ABCD), emit `code="ABCD_MISSING"` warning.
  - Add `"abcd_verb_bloom_mismatch"` (and `"abcd_authored"` if the validator emits a positive-path event too) to `schemas/events/decision_event.schema.json::decision_type.enum` in alphabetical position. Phase 4.5 cleanup re-alphabetised this enum (commit `3184f1a`); maintain the alphabetical contract. Without the enum addition, `DECISION_VALIDATION_STRICT=true` fails closed on the first emit.
- **Verification:** `python -c "from lib.validators.abcd_objective import AbcdObjectiveValidator; v=AbcdObjectiveValidator(); assert hasattr(v, 'validate')"` exits 0. Plus enum-addition smoke: `python -c "import json; e = json.load(open('schemas/events/decision_event.schema.json'))['properties']['decision_type']['enum']; assert 'abcd_verb_bloom_mismatch' in e"` exits 0.

#### Subtask 4.5: Wire `AbcdObjectiveValidator` to `course_planning.validation_gates`
*(Added Phase 6-prep based on investigation refresh against HEAD `ae7779e`; not present in original plan authoring.)*
- **Files:** `/home/user/Ed4All/config/workflows.yaml` (edit the `textbook_to_course::course_planning` phase block, around `:677-719`)
- **Depends on:** Subtask 4 (validator must exist before being wired)
- **Estimated LOC:** ~10
- **Change:** Add a new `validation_gates` block under `course_planning` (the phase currently has no gates) with one entry:
  ```yaml
  validation_gates:
    - gate_id: abcd_verb_alignment
      validator: lib.validators.abcd_objective.AbcdObjectiveValidator
      severity: warning  # Phase 6 lands as warning; Phase 7+ promotes to critical once corpus calibration confirms safe
      threshold:
        max_critical_issues: 0
      behavior:
        on_fail: warn
        on_error: warn
  ```
  The validator operates on `synthesized_objectives.json` at the `course_planning` phase output (NOT at the inter-tier or post-rewrite seam — those seams are downstream of objective synthesis).
- **Verification:** `grep -A 5 "abcd_verb_alignment" config/workflows.yaml` returns the new entry; `python -c "import yaml; w = yaml.safe_load(open('config/workflows.yaml')); cp = next(p for p in w['workflows']['textbook_to_course']['phases'] if p['name'] == 'course_planning'); gates = [g['gate_id'] for g in cp.get('validation_gates', [])]; assert 'abcd_verb_alignment' in gates"` exits 0.

#### Subtask 5: Add `lib/validators/tests/test_abcd_objective.py`
- **Files:** create `/home/user/Ed4All/lib/validators/tests/test_abcd_objective.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~150
- **Change:** Tests: `test_passes_when_verb_matches_bloom_level`, `test_returns_regenerate_on_verb_mismatch`, `test_returns_warning_when_abcd_field_absent`, `test_handles_capitalization_normalization`, `test_compose_abcd_prose_round_trip`, `test_legacy_lo_without_abcd_skipped`. Plus integration with `compose_abcd_prose`.
- **Verification:** `pytest lib/validators/tests/test_abcd_objective.py -v` reports ≥6 PASSED.

### B. Course-outliner agent + plan_course_structure widening (4 subtasks)

#### Subtask 6: Amend `Courseforge/agents/course-outliner.md` to emit ABCD
- **Files:** `/home/user/Ed4All/Courseforge/agents/course-outliner.md`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~80
- **Change:** Append a new section "### ABCD-tagged emit format (Phase 6)" instructing the agent to emit each LO with: `id`, `statement`, `bloom_level`, `bloom_verb`, `cognitive_domain`, `key_concepts[]`, AND a new `abcd: {audience, behavior: {verb, action_object}, condition, degree}` object. Document the JSON Schema (point at `$defs.AbcdObjective`). Add 2 worked examples covering "remember" and "apply" levels.
- **Verification:** `grep -c "abcd\|AbcdObjective" Courseforge/agents/course-outliner.md` returns ≥3.

#### Subtask 7: Widen `MCP/tools/pipeline_tools.py::_plan_course_structure` to accept ABCD
*(Citations refreshed Phase 6-prep against HEAD `ae7779e`; lines drifted as Phase 3.5/4/7a landed.)*
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:3789-3957` (the `_plan_course_structure` async helper + its registry entry)
- **Depends on:** Subtask 6
- **Estimated LOC:** ~50
- **Change:** When `_cgh.synthesize_objectives_from_topics` returns LOs (or a supplied objectives JSON has them), the synthesized payload now carries `abcd` per LO. Extend the `lo_entries.append(entry)` block to pass through `abcd` when present.
- **Verification:** test the function with a stubbed agent return that includes ABCD; assert `synthesized_objectives.json` contains `abcd` per LO.

#### Subtask 8: Add `MCP/tools/_content_gen_helpers.py::synthesize_objectives_from_topics` ABCD widening
- **Files:** `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py:1214` (the function)
- **Depends on:** Subtask 7
- **Estimated LOC:** ~120
- **Change:** Function generates structured ABCD when LLM-driven path is wired (Wave 24 default is template-based). Add ABCD population: derive `audience` from course-level metadata (`course_name` → "Students"); derive `behavior.verb` from `bloom_verb`; derive `behavior.action_object` from the LO statement minus the verb; derive `condition` and `degree` from defaults (`""`,  `""`) — these are filled in by the LLM-driven outliner downstream.
- **Verification:** `pytest MCP/tools/tests/test_content_gen_helpers.py::test_synthesize_objectives_emits_abcd_skeleton -v` PASSES.

#### Subtask 9: Add `lib/ontology/tests/test_compose_abcd_prose.py`
- **Files:** create `/home/user/Ed4All/lib/ontology/tests/test_compose_abcd_prose.py`
- **Depends on:** Subtask 3
- **Estimated LOC:** ~100
- **Change:** Round-trip tests across 6 Bloom levels × 3 fixture verbs each. Asserts terminal period; capitalization; no double-spaces; no double-periods.
- **Verification:** `pytest lib/ontology/tests/test_compose_abcd_prose.py -v` reports ≥18 PASSED.

### C. Concept-extraction phase + decoupling (8 subtasks)

#### Subtask 10: Create `concept_extraction` agent spec
- **Files:** create `/home/user/Ed4All/Trainforge/agents/pedagogy-graph-builder.md`
- **Depends on:** none
- **Estimated LOC:** ~120
- **Change:** Agent spec describing the standalone graph builder. Inputs: `dart_chunks_path`. Output: `concept_graph_path`. References `Trainforge/pedagogy_graph_builder.py::build_pedagogy_graph`. Cross-link to roadmap §3.4.
- **Verification:** `grep -c "build_pedagogy_graph\|concept_graph" Trainforge/agents/pedagogy-graph-builder.md` returns ≥3.

#### Subtask 11: Add `concept_extraction` workflow phase entry
*(Citations refreshed Phase 6-prep against HEAD `ae7779e`; Phase 7a did NOT add a `chunking`/`dart_chunking` workflow phase — it only lifted the in-process `_chunk_content` helper into the `ed4all-chunker` package + added a `chunker_version` manifest field (commits `64d5e3e`, `874dd1b`). The original two-path branching has been collapsed: only the `depends_on: [source_mapping]` path is valid.)*
- **Files:** `/home/user/Ed4All/config/workflows.yaml` (the `textbook_to_course::phases` list, around `:651-719` — the `source_mapping` and `course_planning` phase blocks)
- **Depends on:** Subtask 10
- **Estimated LOC:** ~60
- **Change:** Insert new `concept_extraction` phase between `source_mapping` and `course_planning`:
  - `name: concept_extraction`
  - `agents: [pedagogy-graph-builder]`
  - `parallel: false`
  - `depends_on: [source_mapping]`
  - `outputs: [concept_graph_path, concept_graph_sha256]`
  - `timeout_minutes: 30`
  - `validation_gates`: 1 entry — `concept_graph` → `lib.validators.concept_graph.ConceptGraphValidator`, severity warning initially.
- Update `course_planning.depends_on: [source_mapping]` → `[concept_extraction]`.
- **Verification:** `python -c "import yaml; d=yaml.safe_load(open('config/workflows.yaml')); ph=next(p for p in d['workflows']['textbook_to_course']['phases'] if p['name']=='concept_extraction'); assert ph is not None and ph['depends_on']==['source_mapping']"` exits 0.

#### Subtask 12: Add `MCP/tools/pipeline_tools.py::_run_concept_extraction` helper
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py`
- **Depends on:** Subtask 11
- **Estimated LOC:** ~100
- **Change:** New async helper invoked when `concept_extraction` phase runs. Phase 7a lifted `_chunk_content` into the `ed4all-chunker` package (commits `64d5e3e`, `874dd1b`) but did NOT add a workflow phase, so this helper invokes the chunker directly via `ed4all_chunker.chunk_content(...)` to produce v4 chunks from staged DART HTML; then calls `Trainforge.pedagogy_graph_builder.build_pedagogy_graph(chunks=chunks, course_id=course_slug)`; persists graph to `LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json` and `manifest.json`; computes `sha256` and routes it through `phase_outputs.concept_extraction.concept_graph_sha256`.
- **Verification:** `pytest MCP/tests/test_pipeline_tools.py::test_run_concept_extraction_emits_graph -v` PASSES.

#### Subtask 13: Refactor `Trainforge/process_course.py` `build_pedagogy_graph` call site to skip when concept_extraction phase ran
*(Citations refreshed Phase 6-prep against HEAD `ae7779e`; lines drifted as Phase 3.5/4/7a landed.)*
- **Files:** `/home/user/Ed4All/Trainforge/process_course.py:~3300-3420` (outer block at `:3302` documents the `build_pedagogy_graph` flow; import at `:3345`; call site at `:3402-3407`; failure-fallback log at `:3410`)
- **Depends on:** Subtask 12
- **Estimated LOC:** ~50
- **Change:** Read `concept_graph_path` from input; if present, load the concept graph from there instead of re-building. Keep the existing build path as a fallback for legacy corpora.
- **Verification:** `pytest Trainforge/tests/test_process_course.py::test_consumes_phase6_concept_graph_when_present -v` PASSES.

#### Subtask 14: Create `lib/validators/concept_graph.py::ConceptGraphValidator`
- **Files:** create `/home/user/Ed4All/lib/validators/concept_graph.py`
- **Depends on:** Subtask 12
- **Estimated LOC:** ~250
- **Change:** Validator gates the emitted graph: ≥10 concept nodes; ≥5 edge types present (taxonomic + pedagogical); each node carries `class` field; each edge carries `relation_type` field; provenance per edge when `TRAINFORGE_CONCEPT_GRAPH_EDGE_PROVENANCE=true`.
- **Verification:** `pytest lib/validators/tests/test_concept_graph.py -v` reports ≥6 PASSED.

#### Subtask 15: Create `lib/ontology/concept_objective_linker.py`
- **Files:** create `/home/user/Ed4All/lib/ontology/concept_objective_linker.py`
- **Depends on:** Subtasks 12, 4
- **Estimated LOC:** ~250
- **Change:** Per roadmap §6.6 two-stage with explicit linker. Function `link_concepts_to_objectives(concept_graph: dict, objectives: List[dict]) -> List[dict]`: for each LO, find concept-graph nodes whose slug matches `keyConcepts` (substring match). Add unmatched concepts where the LO statement contains the concept slug verbatim. Returns enriched objectives list with populated `keyConcepts`.
- **Verification:** `pytest lib/ontology/tests/test_concept_objective_linker.py -v` reports ≥5 PASSED.

#### Subtask 16: Wire `concept_objective_linker` invocation into `_plan_course_structure`
*(Citations refreshed Phase 6-prep against HEAD `ae7779e`; lines drifted as Phase 3.5/4/7a landed.)*
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:3789-3957` (the same `_plan_course_structure` async helper widened in Subtask 7)
- **Depends on:** Subtask 15
- **Estimated LOC:** ~30
- **Change:** When `concept_graph_path` is supplied (Phase 6 enabled), call `link_concepts_to_objectives` after objective synthesis but before persisting `synthesized_objectives.json`. The `keyConcepts` field on each LO is populated from the linker.
- **Verification:** `pytest MCP/tests/test_pipeline_tools.py::test_plan_course_structure_links_concepts -v` PASSES.

#### Subtask 17: Add `concept_graph_sha256` to LibV2 manifest schema
- **Files:** `/home/user/Ed4All/schemas/library/course_manifest.schema.json`
- **Depends on:** Subtask 12
- **Estimated LOC:** ~10
- **Change:** Add `concept_graph_sha256: {type: "string", pattern: "^[a-f0-9]{64}$"}`. Optional in Phase 6; required in Phase 7c.

### D. Concept-graph hash + manifest gate (3 subtasks)

#### Subtask 18: Compute + persist `concept_graph_sha256` in archive helper
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py` (the `archive_to_libv2` flow)
- **Depends on:** Subtask 17
- **Estimated LOC:** ~30

#### Subtask 19: Extend `LibV2ManifestValidator` to read `concept_graph_sha256` (advisory only in Phase 6)
- **Files:** `/home/user/Ed4All/lib/validators/libv2_manifest.py`
- **Depends on:** Subtask 18
- **Estimated LOC:** ~25
- **Change:** Validator reads `manifest.concept_graph_sha256`. When absent in Phase 6, emit `severity="warning"` GateIssue. Phase 7c promotes to critical.

#### Subtask 20: Tests for concept-graph hash + manifest extension
- **Files:** create `/home/user/Ed4All/lib/validators/tests/test_libv2_manifest_concept_graph.py`
- **Depends on:** Subtask 19
- **Estimated LOC:** ~80

### E. Phase 3c env-var fixes (4 subtasks)

#### Subtask 21: Fix `DART/pdf_converter/cli.py:99-100` hardcoded model
- **Files:** `/home/user/Ed4All/DART/pdf_converter/cli.py`
- **Depends on:** none
- **Estimated LOC:** ~15
- **Change:** Replace `default='claude-sonnet-4-20250514'` with `default=os.environ.get('DART_CLAUDE_MODEL') or 'claude-sonnet-4-20250514'`. Same for help string.
- **Verification:** `python -c "import os; os.environ['DART_CLAUDE_MODEL']='custom-x'; from DART.pdf_converter.cli import _resolve_default_model; assert _resolve_default_model()=='custom-x'"` exits 0.

#### Subtask 22: Fix `DART/pdf_converter/{converter,claude_processor,alt_text_generator}.py` model resolution
- **Files:** `/home/user/Ed4All/DART/pdf_converter/converter.py:69`, `claude_processor.py:228`, `alt_text_generator.py:42`
- **Depends on:** Subtask 21
- **Estimated LOC:** ~30 (10 each)
- **Change:** Each constructor's `model: str = "claude-sonnet-4-20250514"` argument changes to `model: str = None`; `__init__` resolves to `model or os.environ.get("DART_CLAUDE_MODEL") or "claude-sonnet-4-20250514"`.

#### Subtask 23: Fix `MCP/orchestrator/llm_backend.py:48,791` env-var-first chain
- **Files:** `/home/user/Ed4All/MCP/orchestrator/llm_backend.py`
- **Depends on:** none
- **Estimated LOC:** ~25
- **Change:** Replace `DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"` resolution to read `os.environ.get("MCP_ORCHESTRATOR_LLM_MODEL")` first. Add new env var `MCP_ORCHESTRATOR_LLM_MODEL` (defaults to `claude-opus-4-7`).

#### Subtask 24: Add 2 env-var rows to root `CLAUDE.md` flag table
- **Files:** `/home/user/Ed4All/CLAUDE.md`
- **Depends on:** Subtasks 21-23
- **Estimated LOC:** ~10

### F. Documentation + smoke (3 subtasks)

#### Subtask 25: Update `Courseforge/CLAUDE.md` with ABCD section
#### Subtask 26: Update `Trainforge/CLAUDE.md` with concept-extraction phase section
#### Subtask 27: End-to-end smoke

---

## Execution sequencing

- 6-N1: A (1-5) + E (21-24) parallelisable; C (10-13) start can be parallelisable with A.
- 6-N2: B (6-9) → D (18-20).
- 6-N3: C (14-17) sequential after Phase 7a lands.
- 6-N4: F (25-27).

---

## Final smoke test

```bash
pytest lib/ontology/tests/test_compose_abcd_prose.py \
       lib/ontology/tests/test_concept_objective_linker.py \
       lib/validators/tests/test_abcd_objective.py \
       lib/validators/tests/test_concept_graph.py \
       MCP/tests/test_pipeline_tools.py -k "concept_extraction or abcd" -v

# Verify ABCD round-trip:
jq -r '.learning_outcomes[] | select(.abcd != null) | "\(.id): \(.abcd.behavior.verb) (\(.bloom_level))"' \
  Courseforge/exports/PROJ-DEMO_303-*/01_learning_objectives/synthesized_objectives.json
```

---

### Critical Files for Implementation
- `/home/user/Ed4All/lib/ontology/learning_objectives.py` (extend with `compose_abcd_prose`, `BLOOMS_VERBS`)
- `/home/user/Ed4All/lib/validators/abcd_objective.py` (NEW)
- `/home/user/Ed4All/lib/validators/concept_graph.py` (NEW)
- `/home/user/Ed4All/lib/ontology/concept_objective_linker.py` (NEW)
- `/home/user/Ed4All/Trainforge/agents/pedagogy-graph-builder.md` (NEW)
- `/home/user/Ed4All/config/workflows.yaml` (insert `concept_extraction` phase)
- `/home/user/Ed4All/schemas/knowledge/courseforge_jsonld_v1.schema.json` (add `$defs.AbcdObjective`)
- `/home/user/Ed4All/Courseforge/agents/course-outliner.md` (amend with ABCD emit contract)
