# Phase 7 Detailed Execution Plan — `ed4all-chunker` Package + Dual Chunkset (DART + IMSCC) + LibV2 Manifest Gate

Refines `plans/courseforge_architecture_roadmap.md` §3.3, §3.5, §5 into atomic subtasks. Three sub-phases: 7a (chunker package), 7b (DART chunkset), 7c (IMSCC chunkset rename + manifest gate). **Depends on:** Phase 6 (concept extractor consumes Phase 7a's chunker package output). Phase 7a lands BEFORE Phase 6.

---

## Investigation findings (locked)

- **Chunker entry points are at `Trainforge/process_course.py:1462::_chunk_content` (~108-LOC method) and `:1699::_chunk_text_block` (~120 LOC)**. Boilerplate detection is `Trainforge/rag/boilerplate_detector.py::strip_boilerplate` at `:98`. Constants `MIN_CHUNK_SIZE = 100`, `MAX_CHUNK_SIZE = 800` at `process_course.py:986-987`. *(Citations refreshed Phase 7a-prep against HEAD `84decc9`; lines drifted +30 since the original Phase 7 plan authoring as Phase 3.5 + Phase 4 landed.)*
- **Helper functions called by chunker**: `_extract_plain_text` (`:2929`), `_strip_assessment_feedback` (`:3031`), `_strip_feedback_from_text` (`:3052`), `_extract_section_html` (`:2935`), `_merge_small_sections` (`:1590`), `_merge_section_source_ids` (`:1572`), `_create_chunk` (`:1823`), `_type_from_resource` (`:2580`). All are `CourseProcessor` methods on `process_course.py`. Lift target: extract these to package functions.
- **`pyproject.toml`** at `/home/user/Ed4All/pyproject.toml` does not have `[tool.uv.workspace]` or workspace member listing — verified. Phase 7a adds workspace-member directives via `[tool.setuptools.packages.find]` or per-package install.
- **No `ed4all-chunker/` directory exists** — verified. Phase 7a creates it as `/home/user/Ed4All/ed4all-chunker/` (in-repo workspace member per roadmap §6.2 recommendation).
- **`LibV2/courses/<slug>/corpus/chunks.jsonl`** is the current IMSCC chunkset location (per `LibV2/CLAUDE.md:194`). Phase 7c renames `corpus/` → `imscc_chunks/`.
- **No `LibV2/courses/<slug>/dart_chunks/` directory exists today** — verified for any course. Phase 7b creates it.
- **`LibV2ManifestValidator`** at `lib/validators/libv2_manifest.py` already validates manifest schema via `course_manifest.schema.json`. Phase 7c extends it to require both `dart_chunks_sha256` and `imscc_chunks_sha256` fields.
- **`LibV2/CLAUDE.md`** has a directory-tree section (per roadmap citation `:194`). Phase 7c amends to reflect the rename + new dirs.
- **Backfill is operator-driven (per roadmap §5.2)**. Script: `LibV2/tools/libv2/scripts/backfill_dart_chunks.py`.

---

## Pre-resolved decisions

1. **Packaging form (per roadmap §6.2 recommendation).** In-repo workspace member at `/home/user/Ed4All/ed4all-chunker/`. Future TODO: promote to PyPI when an external consumer surfaces.
2. **Package layout.**
   ```
   ed4all-chunker/
   ├── pyproject.toml
   ├── README.md
   ├── ed4all_chunker/
   │   ├── __init__.py
   │   ├── chunker.py       # _chunk_content + _chunk_text_block ports
   │   ├── boilerplate.py   # strip_boilerplate + detection helpers
   │   ├── helpers.py       # _extract_plain_text, _strip_*, _extract_section_html
   │   ├── version.py
   │   └── schema.py        # chunk_v4 dict shape contract
   └── tests/
   ```
3. **Backwards compatibility.** Trainforge's `process_course.py` keeps `_chunk_content` and `_chunk_text_block` as thin wrappers around `ed4all_chunker.chunker.chunk_content` / `chunk_text_block` for one wave. Existing tests stay green.
4. **Version pinning.** `LibV2/courses/<slug>/manifest.json::chunker_version` field carries the package's installed version (read from `ed4all_chunker.__version__`).
5. **DART chunkset content.** Chunker fires against the DART HTML output in `LibV2/courses/<slug>/source/dart_html/*.html`. Output goes to `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` + `manifest.json`.
6. **Symmetric IMSCC chunkset.** Path migration `corpus/chunks.jsonl` → `imscc_chunks/chunks.jsonl`. Manifest sidecar emitted alongside.
7. **`LibV2ManifestValidator` extension (Phase 7c) — gate posture (per roadmap §6.3 recommendation).** Hard fail-closed when either `dart_chunks_sha256` or `imscc_chunks_sha256` is missing. Backfill via operator-driven `backfill_dart_chunks.py` script.
8. **Triangle invariant.** Manifest validates `dart_chunks → dart_html → pdf` chain AND `imscc_chunks → imscc → ... → pdf` chain trace back to the same PDF SHA. Phase 7c ships a stub of this check; full triangle validation lands in a Phase 7c-followup.
9. **Workflow integration.**
   - Phase 7a: no workflow change (package is consumable from existing chunker invocation in `process_course.py`).
   - Phase 7b: new workflow phase `chunking` (or `dart_chunking`) between `dart_conversion` and `course_planning`. Pre-resolved decision: place between `staging` and `objective_extraction` so the chunker output is available for downstream Phase 6 concept-extraction. The phase agent is `dart-chunker` (NEW).
   - Phase 7c: workflow phase `imscc_chunking` between `packaging` and `training_synthesis`, replacing the in-process chunker invocation Trainforge currently runs.

---

## Atomic subtasks

Estimated total LOC: ~3,600 (800 chunker package extraction + 600 Trainforge refactor + 250 DART chunking workflow phase + 200 imscc rename + 250 manifest validator extension + 200 backfill script + 200 LibV2 docs update + 600 tests + 150 docs + 100 misc).

### A. `ed4all-chunker` package extraction (Phase 7a, 8 subtasks)

#### Subtask 1: Create `ed4all-chunker/` package skeleton
- **Files:** create `/home/user/Ed4All/ed4all-chunker/pyproject.toml`, `/home/user/Ed4All/ed4all-chunker/README.md`, `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/__init__.py`, `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/version.py`
- **Depends on:** none
- **Estimated LOC:** ~80
- **Change:** `pyproject.toml`: name=`ed4all-chunker`, version=`1.0.0`, description="Canonical chunker for Ed4All pipeline (DART + IMSCC)". Dependencies: `beautifulsoup4>=4.12.0`, `lxml>=4.9.0`. `version.py`: `__version__ = "1.0.0"`. `__init__.py` re-exports `chunk_content`, `chunk_text_block`, `__version__`.
- **Verification:** `python -c "from ed4all_chunker import chunk_content, chunk_text_block, __version__; assert __version__=='1.0.0'"` exits 0 (after install).

#### Subtask 2: Lift `boilerplate_detector.py` to `ed4all_chunker/boilerplate.py`
- **Files:** copy `/home/user/Ed4All/Trainforge/rag/boilerplate_detector.py` to `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/boilerplate.py`
- **Depends on:** Subtask 1
- **Estimated LOC:** ~150 (move + re-namespace)
- **Change:** Move `strip_boilerplate` and supporting helpers. Replace any imports from Trainforge with self-contained imports.
- **Verification:** `python -c "from ed4all_chunker.boilerplate import strip_boilerplate; out, removed = strip_boilerplate('hello world', ['world']); assert out=='hello' and removed==1"` exits 0.

#### Subtask 3: Lift chunking helpers to `ed4all_chunker/helpers.py`
- **Files:** create `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/helpers.py`
- **Depends on:** Subtask 2
- **Estimated LOC:** ~250
- **Change:** Port `_extract_plain_text`, `_strip_assessment_feedback`, `_strip_feedback_from_text`, `_extract_section_html`, `_type_from_resource` from `Trainforge/process_course.py` as standalone module functions (drop the `self` param; inputs are explicit args).

#### Subtask 4: Lift `_chunk_content` + `_chunk_text_block` to `ed4all_chunker/chunker.py`
- **Files:** create `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/chunker.py`
- **Depends on:** Subtasks 2, 3
- **Estimated LOC:** ~450
- **Change:** Port `_chunk_content` (lines 1462-1570) + `_chunk_text_block` (lines 1699-~1820) + `_merge_small_sections` (line 1590) + `_merge_section_source_ids` (line 1572) + `_create_chunk` (line 1823) from `Trainforge/process_course.py`. Make them standalone functions: `chunk_content(parsed_items: List[Dict], course_code: str, boilerplate_spans: Optional[List[str]] = None, *, min_chunk_size: int = 100, max_chunk_size: int = 800) -> List[Dict]`. The Trainforge-specific `pages_with_misconceptions` tracking is moved to a returned-tuple element.
- **Verification:** `python -c "from ed4all_chunker.chunker import chunk_content; chunks = chunk_content([], 'TEST_101'); assert chunks == []"` exits 0.

#### Subtask 5: Add `ed4all-chunker/tests/test_chunker_smoke.py`
- **Files:** create `/home/user/Ed4All/ed4all-chunker/tests/test_chunker_smoke.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~200
- **Change:** Tests: `test_chunk_content_returns_v4_shape`, `test_chunk_text_block_respects_max_chunk_size`, `test_merged_small_sections_below_min_size`, `test_boilerplate_stripped_from_chunk_text`, `test_chunk_id_format_matches_prefix`, `test_chunk_provenance_carries_html_xpath_and_char_span`. Uses fixture HTML.
- **Verification:** `pytest ed4all-chunker/tests/test_chunker_smoke.py -v` reports ≥6 PASSED.

#### Subtask 6: Refactor `Trainforge/process_course.py` to delegate to package
- **Files:** `/home/user/Ed4All/Trainforge/process_course.py:1462-~1820`
- **Depends on:** Subtask 5
- **Estimated LOC:** ~80 (deletion + thin delegation)
- **Change:** Replace `_chunk_content` body with `from ed4all_chunker.chunker import chunk_content; return chunk_content(parsed_items, self.course_code, self._boilerplate_spans, min_chunk_size=self.MIN_CHUNK_SIZE, max_chunk_size=self.MAX_CHUNK_SIZE)`. Same for `_chunk_text_block`. Existing tests must stay green.
- **Verification:** `pytest Trainforge/tests/ -k "chunk" -v` PASSES (regression).

#### Subtask 7: Add `ed4all-chunker` as workspace dependency
- **Files:** `/home/user/Ed4All/pyproject.toml`
- **Depends on:** Subtask 6
- **Estimated LOC:** ~5
- **Change:** Add `"ed4all-chunker @ file://./ed4all-chunker"` to `[project].dependencies`. Document in `README.md`.

#### Subtask 8: Add `chunker_version` field to manifest schema + emit
- **Files:** `/home/user/Ed4All/schemas/library/course_manifest.schema.json` + `MCP/tools/pipeline_tools.py::archive_to_libv2`
- **Depends on:** Subtask 7
- **Estimated LOC:** ~25
- **Change:** Add manifest field `chunker_version: {type: "string"}`. Archive helper reads `ed4all_chunker.__version__` and writes it. Validator (Phase 7c) requires this field present.

### A.1. Phase 4 followups (Phase 7a, 1 subtask)

#### Subtask 8.5: Resolve BERT ensemble revision SHAs (Phase 4 followup)
- **Files:** `/home/user/Ed4All/lib/classifiers/bloom_bert_ensemble.py:63-76` (`_DEFAULT_ENSEMBLE_MEMBERS`)
- **Depends on:** none (independent of chunker work; bundled into Phase 7a so it doesn't get lost across the Phase 6 / Phase 7b/c boundary)
- **Estimated LOC:** ~10 (3 SHA strings + 1 capture-emit hook in `_emit_member_loaded`)
- **Background:** Phase 4 Subtask 24 shipped the `BloomBertEnsemble` with placeholder `revision="main"` for all three members. Per the canonical `bloom_bert_ensemble.py` module docstring (lines 24-32) and root `CLAUDE.md` § "BERT ensemble members" (which already documents this as "**Phase 4 followup** is to resolve concrete commit SHAs"), the placeholder must be replaced with concrete pinned SHAs so each classification's reproducibility chain is closed end-to-end. The Phase 4 review worker flagged this as HIGH severity but it currently lives only as a docs-only "not closed" item in `Courseforge/CLAUDE.md` § "Phase 4: statistical-tier validators + BERT ensemble" → "Phase 4 follow-ups intentionally not closed in this batch".
- **Change:**
  1. For each of the three members in `_DEFAULT_ENSEMBLE_MEMBERS` — `kabir5297/bloom_taxonomy_classifier`, `distilbert-base-uncased-finetuned-sst-2-english`, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — resolve the current `main` revision via `huggingface_hub.HfApi().model_info(repo_id).sha`. Pin the resolved 40-char hex SHA into the `revision` field of the corresponding registry entry (replacing the literal `"main"`).
  2. Verify that `_emit_member_loaded` (`bloom_bert_ensemble.py:389-425`) already captures `member_revision` in the `bert_ensemble_member_loaded` decision event metadata — it does (line 417). No code change needed there; the audit trail already records exactly which revision produced each classification once the registry carries the resolved SHA. (If a future cleanup wants to also surface the SHA in the rationale string for grep-ability, that's a one-line interpolation tweak in the `rationale=` block at line 408.)
  3. Add a regression test under `lib/classifiers/tests/` (or extend `lib/validators/tests/test_bloom_classifier_disagreement.py`) asserting that every entry in `_DEFAULT_ENSEMBLE_MEMBERS` has a `revision` field matching the regex `^[0-9a-f]{40}$` — guards against a future regression that re-introduces the `"main"` placeholder.
- **Verification:** `python -c "import re; from lib.classifiers.bloom_bert_ensemble import _DEFAULT_ENSEMBLE_MEMBERS; assert all(re.match(r'^[0-9a-f]{40}$', m['revision']) for m in _DEFAULT_ENSEMBLE_MEMBERS), 'revision must be a resolved 40-char SHA, not main'"` exits 0. Integration: run the Phase 4 statistical-tier smoke (per `Courseforge/CLAUDE.md` § "Operator smoke runbook (Phase 4 statistical tier)") and confirm `bert_ensemble_member_loaded` events in the resulting JSONL carry the resolved SHA in `metadata.member_revision`.
- **Prerequisites:** `pip install huggingface_hub` is required before running the SHA-resolution one-off (it is not currently installed in the dev environment per the Phase 7a investigation refresh against HEAD `84decc9`). The package is import-only — no model weights are downloaded by `HfApi().model_info()` (it queries the Hub HTTP API).

### B. DART chunkset (Phase 7b, 6 subtasks)

#### Subtask 9: Create `dart-chunker` agent spec
*(Phase 6 precedent established Trainforge-side placement for chunker-adjacent agents; investigation refresh suggests utility-agent in-code dispatcher per textbook-stager.)*
- **Files:**
  - Following the `textbook-stager` precedent (`config/agents.yaml:142-150`, type=utility, in-code dispatcher with no `.md` spec), the `dart-chunker` is a deterministic chunker transformation with no LLM dispatch. **Plan: SKIP the .md spec; register only as a utility agent in `config/agents.yaml`.** Mirror `textbook-stager`'s entry style.
  - If the `.md` spec is wanted instead (subagent-style), place at `/home/user/Ed4All/Trainforge/agents/dart-chunker.md` (NOT `Courseforge/agents/`) following the Phase 6 precedent set by `pedagogy-graph-builder.md` (commit `95e9dda`).
- **Registration (regardless of `.md` choice):**
  - Add a `config/agents.yaml` registration entry for `dart-chunker` (mirror `textbook-stager` at `:142-150` if utility-style, or `source-router` at `:161-169` if subagent-style).
  - Add a `MCP/core/executor.py::AGENT_TOOL_MAPPING` entry `"dart-chunker": "run_dart_chunking"` mirroring the Phase 6 entry `"pedagogy-graph-builder": "run_concept_extraction"` at `:160-167`.
- **Depends on:** Subtask 8
- **Estimated LOC:** ~80 (or ~20 if utility-agent-only path is taken)

#### Subtask 10: Add `chunking` workflow phase (DART chunkset emit)
*(Citations refreshed Phase 7b-prep against HEAD `4f1372a`; coordinated with Phase 6's concept_extraction insertion at workflows.yaml:685-720.)*
- **Files:** `/home/user/Ed4All/config/workflows.yaml:591-650` (the staging-to-objective_extraction window at HEAD `4f1372a`)
- **Depends on:** Subtask 9
- **Estimated LOC:** ~50
- **Change:** Insert phase:
  - `name: chunking`
  - `agents: [dart-chunker]`
  - `parallel: false`
  - `depends_on: [staging]`
  - `outputs: [dart_chunks_path, dart_chunks_sha256]`
  - `timeout_minutes: 15`
- Update `objective_extraction.depends_on: [staging]` → `[chunking]`. (Phase 6 concept_extraction will consume `dart_chunks_path` directly.)
- **Phase ordering reconciliation with Phase 6:** Phase 6 inserted `concept_extraction` between `source_mapping` and `course_planning` at `config/workflows.yaml:685-720` (commit `e0ea640`). With this subtask, the dependency chain becomes: `staging → chunking → objective_extraction → source_mapping → concept_extraction → course_planning` (insert `chunking` between `staging` and `objective_extraction`).
- **Widen `concept_extraction.depends_on`:** Phase 6's `concept_extraction.depends_on` (currently `[source_mapping]`) should be widened to `[source_mapping, chunking]` so the `chunking` phase output is consumable by `concept_extraction` once both phases are live.
- **Remove the Phase 6 placeholder comment:** the YAML comment at `config/workflows.yaml:683-684` (the *"Phase 7a did NOT add a chunking workflow phase, so depends_on: [source_mapping] is the only valid path"* note) can be REMOVED as part of ST 10 since the assumption no longer holds once `chunking` is wired in.

*(Subtask order: 9 → 10 → 12 → 11 → 13 → 14. ST 12's manifest schema must land before ST 11's helper writes against it.)*

#### Subtask 12: Add `LibV2/courses/<slug>/dart_chunks/manifest.json` schema
- **Files:** create `/home/user/Ed4All/schemas/library/chunkset_manifest.schema.json`
- **Depends on:** Subtask 10
- **Estimated LOC:** ~80
- **Change:** Manifest schema: `{required: [chunks_sha256, chunker_version, chunkset_kind, source_*_sha256], properties: {chunks_sha256, chunker_version, chunkset_kind: {enum: ["dart","imscc"]}, source_dart_html_sha256, source_imscc_sha256, chunks_count, generated_at}}`.

#### Subtask 11: Add `MCP/tools/pipeline_tools.py::_run_dart_chunking` helper
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py`
- **Depends on:** ST 10 + ST 12 (schema must exist before helper writes against it)
- **Estimated LOC:** ~120
- **Change:**
  - **Mirror the `_run_concept_extraction` template** at `MCP/tools/pipeline_tools.py:6051-6320` (Phase 6 commit `e0ea640`): async helper, `**kwargs` resolved via the workflow YAML's `inputs_from`, registered in `_build_tool_registry` as `registry["run_dart_chunking"]`, returns a dict with `dart_chunks_path` + `dart_chunks_sha256` keys for downstream phase consumption.
  - Async helper invoking `ed4all_chunker.chunk_content` against DART HTML files; persists output to `LibV2/courses/<slug>/dart_chunks/chunks.jsonl` + `manifest.json` (carrying `chunks_sha256`, `chunker_version`, `source_dart_html_sha256`); routes `dart_chunks_sha256` through `phase_outputs`.
- **Verification:** `pytest MCP/tests/test_pipeline_tools.py::test_run_dart_chunking_emits_chunks_jsonl -v` PASSES.

#### Subtask 13: Add `lib/validators/chunkset_manifest.py::ChunksetManifestValidator`
- **Files:** create `/home/user/Ed4All/lib/validators/chunkset_manifest.py`
- **Depends on:** Subtask 12
- **Estimated LOC:** ~150
- **Change:** Validator gates the chunkset manifest. Verifies `chunks_sha256` matches the on-disk file's SHA. Verifies `chunker_version` matches installed package. Verifies `source_*_sha256` resolves to a known artifact.

#### Subtask 14: Add `Courseforge/CLAUDE.md` + `Trainforge/CLAUDE.md` updates for DART chunkset

#### Subtask 14.5: **Architectural decision — post-chunking reconciliation of `_run_concept_extraction`**

**Problem:** Phase 6's `_run_concept_extraction` (`MCP/tools/pipeline_tools.py:6051-6320`, commit `e0ea640`) uses an **inline v4-chunk-projection** rather than calling `ed4all_chunker.chunk_content`, because at the `concept_extraction` phase point neither a packaged IMSCC nor a `ChunkerContext` exists yet. Phase 7b's new `chunking` phase WILL produce DART chunks earlier in the pipeline. This subtask reconciles the two surfaces so we don't carry duplicated chunk-shaping logic across the codebase.

**Two options considered:**
- **(a) Refactor `_run_concept_extraction` to consume `dart_chunks_path` from upstream `chunking` phase** — cleaner; eliminates two-surface drift; single canonical chunker.
- **(b) Keep inline-projection for back-compat; have `chunking` produce a redundant copy** — simpler; tolerates two-surface coexistence; leaves duplicate logic in place.

**Recommendation: Option (a) — refactor** (per the Phase 7b-prep investigation worker's preference for cleanliness over coexistence; eliminates a divergence-risk surface).

**Change:**
- Phase 7b ST 11 produces v4 chunks at `LibV2/courses/<slug>/dart_chunks/chunks.jsonl`.
- Phase 7b ST 14.5 (this new subtask) refactors `MCP/tools/pipeline_tools.py:6051-6320::_run_concept_extraction` to consume `dart_chunks_path` from upstream `chunking` phase output.
- Removes the inline projection block at `MCP/tools/pipeline_tools.py:6129-6135`.
- Updates `concept_extraction.depends_on` from `[source_mapping]` → `[source_mapping, chunking]` (mirrors ST 10's widening note).
- Updates `concept_extraction.inputs_from` to include `dart_chunks_path` from `chunking` phase outputs.

**Depends on:** ST 11, ST 12, ST 13 (helper + schema + validator must be live before the consumer-side refactor).
**Estimated LOC:** ~30 (delete projection block at `:6129-6135` + add input wiring through `inputs_from`).
**Verification:** the `concept_graph_semantic.json` output is **byte-stable across the refactor** (no inline-projection vs shared-chunker semantic drift). Pin via a regression test that runs `_run_concept_extraction` against a fixed corpus before and after the refactor and asserts byte equality on the emitted graph JSON.

### C. IMSCC chunkset rename + manifest gate (Phase 7c, 7 subtasks)

#### Subtask 15: Rename `LibV2/courses/<slug>/corpus/` → `imscc_chunks/` (path migration code only)
*(Blast radius expanded Phase 7b-prep based on investigation refresh against HEAD `4f1372a`; original plan listed ~5 files but tree-wide grep found 50+ load-bearing references plus 20+ documentation/schema refs.)*
- **Files:** the rename touches 50+ load-bearing code paths plus 20+ documentation / schema-comment references. Organized below:

  **Code paths needing rename:**
  - `LibV2/tools/chunk_query.py:129,132`
  - `LibV2/tools/study_pack_renderer.py:11,149-185`
  - `LibV2/tools/libv2/validator.py:105,133,179,323,496,549`
  - `LibV2/tools/libv2/cli.py:1815`
  - `MCP/tools/pipeline_tools.py:417,462,1116-1122,4694,4791,5106,5150-5158,5279-5286,5469`
  - `MCP/tools/tutoring_tools.py`
  - `MCP/tools/quiz_generator.py`
  - `MCP/tools/trainforge_tools.py`
  - `MCP/hardening/gate_input_routing.py`
  - `MCP/core/tool_schemas.py`
  - `Trainforge/process_course.py:1112` (`self.corpus_dir = self.output_dir / "corpus"`)
  - `Trainforge/synthesize_training.py:5,215,819-820,2343,2432,2438`
  - `Trainforge/training/runner.py`
  - `Trainforge/eval/slm_eval_harness.py`
  - `Trainforge/eval/chunk_labels.py`
  - `Trainforge/instruction_pair_extractor.py`
  - `Trainforge/pedagogy_graph_builder.py`
  - `Trainforge/scripts/audit_pairs.py`
  - `Trainforge/scripts/discover_curies.py`
  - `Trainforge/scripts/backfill_form_data.py`
  - `Trainforge/rag/retrieval_benchmark.py`
  - `Trainforge/generators/instruction_factory.py`

  **Docs needing update:**
  - `Trainforge/CLAUDE.md`
  - `LibV2/CLAUDE.md:55,194`
  - `ARCHITECTURE.md`
  - `VERSIONING.md`

  **Schema comment refs:**
  - `schemas/models/model_card.schema.json`
  - `schemas/knowledge/chunk_v4.schema.json`
  - `schemas/training/instruction_pair.schema.json`

  **Test fixtures:** any `tests/` path or fixture JSON / YAML referencing `corpus/chunks.jsonl` should be swept.

- **Depends on:** none
- **Estimated LOC:** ~300-500 LOC
- **Change:** Update all consumer code paths reading `corpus/chunks.jsonl` to read `imscc_chunks/chunks.jsonl`. Add back-compat fallback: try `imscc_chunks/` first, fall back to `corpus/` for one wave with a deprecation warning.
- **Back-compat read-fallback shim:** when reading chunks for downstream operations, attempt `imscc_chunks/chunks.jsonl` first; if absent, fall back to `corpus/chunks.jsonl` and log a deprecation warning. Land the shim WITH the rename so unprovisioned LibV2 archives keep working through one migration cycle. Drop the shim in Phase 8.

#### Subtask 16: Add `imscc_chunking` workflow phase (post-packaging IMSCC chunkset emit)
- **Files:** `/home/user/Ed4All/config/workflows.yaml`
- **Depends on:** Subtasks 11, 15
- **Estimated LOC:** ~50
- **Change:** Insert phase between `packaging` and `training_synthesis`:
  - `name: imscc_chunking`
  - `agents: [dart-chunker]` (same agent spec — chunker is symmetric)
  - `inputs_from`: `{imscc_path: phase_outputs.packaging.package_path}`
  - `outputs: [imscc_chunks_path, imscc_chunks_sha256]`
- Update `training_synthesis.depends_on: [packaging]` → `[imscc_chunking]`.

#### Subtask 17: Extend `LibV2ManifestValidator` to require both chunkset hashes (HARD)
*(Citations refreshed Phase 7b-prep against HEAD `4f1372a`; concept_graph_sha256 promotion folded in per Phase 6 ST 19 forward-reference.)*
- **Files:** `/home/user/Ed4All/lib/validators/libv2_manifest.py:38-46` (the `_EXPECTED_SUBDIRS`); plus `validate(...)` method
- **Depends on:** Subtask 16
- **Estimated LOC:** ~80
- **Change:**
  - **Promote `concept_graph_sha256` from warning to critical:** change the three issue codes (`MISSING_CONCEPT_GRAPH_SHA256`, `INVALID_CONCEPT_GRAPH_SHA256`, `CONCEPT_GRAPH_HASH_MISMATCH`) from `severity='warning'` to `severity='critical'`. Phase 6 commit `c3a9f72` left these as warnings explicitly noting Phase 7c promotion.
  - Add `dart_chunks_sha256` and `imscc_chunks_sha256` to required manifest keys. When either is missing, emit `severity="critical"` issue. Update `_EXPECTED_SUBDIRS` to include `dart_chunks` + `imscc_chunks` (corpus removed).
  - **Update schema `required` array:** update `schemas/library/course_manifest.schema.json::required` to include `dart_chunks_sha256`, `imscc_chunks_sha256`, AND `concept_graph_sha256` (Phase 7a's `chunker_version` may also be promoted; assess based on whether all in-flight LibV2 archives carry it). Currently `required: ["libv2_version", "slug", "import_timestamp", "sourceforge_manifest", "classification", "content_profile"]` per `:9-13`.
  - **Mirror the `_check_concept_graph_sha256` method shape** from `lib/validators/libv2_manifest.py:417-521` for both new check methods (`_check_dart_chunks_sha256` + `_check_imscc_chunks_sha256`): each emits MISSING / INVALID / MISMATCH GateIssue codes, accepts `inputs` + `course_dir`, defends against missing/unreadable files.

#### Subtask 18: Create `LibV2/tools/libv2/scripts/backfill_dart_chunks.py`
- **Files:** create `/home/user/Ed4All/LibV2/tools/libv2/scripts/backfill_dart_chunks.py`
- **Depends on:** Subtask 17
- **Estimated LOC:** ~200
- **Change:** Operator-driven script. Args: `--course-slug <slug>`. Reads DART HTML from `source/dart_html/`, runs chunker, writes `dart_chunks/`, computes hash, updates `manifest.json::dart_chunks_sha256`. Emits `decision_type="dart_chunks_backfill"` event with operator + timestamp.

#### Subtask 19: Update `LibV2/CLAUDE.md` directory tree
- **Files:** `/home/user/Ed4All/LibV2/CLAUDE.md:194` (corpus dir reference)
- **Depends on:** Subtask 17
- **Estimated LOC:** ~30
- **Change:** Update directory-tree section to show `dart_chunks/`, `imscc_chunks/`, deprecate `corpus/` with a one-wave migration note.

#### Subtask 20: Add tests for manifest extension + backfill script
- **Files:** create `/home/user/Ed4All/lib/validators/tests/test_libv2_manifest_dual_chunkset.py`, `/home/user/Ed4All/LibV2/tests/test_backfill_dart_chunks.py`
- **Depends on:** Subtasks 17, 18
- **Estimated LOC:** ~250

#### Subtask 21: End-to-end smoke

---

## Execution sequencing

- 7-N1 (Phase 7a): A (1-8) — sequentially. Subtask 8.5 (A.1, BERT SHA-pinning) lands in parallel with A; independent of chunker work.
- 7-N2 (Phase 7b): B (9-14) — after 7a.
- 7-N3 (Phase 7c): C (15-21) — after 7b. NOTE: order respects roadmap-cited "Phase 6 lands between 7a and 7b/c".

---

## Final smoke test

```bash
pip install -e .[dev]
pip install -e ./ed4all-chunker

pytest ed4all-chunker/tests/ \
       Trainforge/tests/ -k chunk \
       MCP/tests/test_pipeline_tools.py -k "chunking" \
       lib/validators/tests/test_libv2_manifest_dual_chunkset.py -v

# Smoke run + verify both chunksets present:
ed4all run textbook_to_course --course-code DEMO_303 --weeks 1
ls LibV2/courses/demo-303-2/dart_chunks/chunks.jsonl
ls LibV2/courses/demo-303-2/imscc_chunks/chunks.jsonl
jq -r '.dart_chunks_sha256, .imscc_chunks_sha256, .chunker_version' \
  LibV2/courses/demo-303-2/manifest.json

# Verify backfill works:
python LibV2/tools/libv2/scripts/backfill_dart_chunks.py --course-slug rdf-shacl-551-2
```

---

### Critical Files for Implementation
- `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/chunker.py` (NEW)
- `/home/user/Ed4All/ed4all-chunker/ed4all_chunker/boilerplate.py` (NEW)
- `/home/user/Ed4All/Trainforge/process_course.py:1462-~1820` (refactor to delegate)
- `/home/user/Ed4All/lib/validators/libv2_manifest.py` (extend for dual chunkset hashes)
- `/home/user/Ed4All/config/workflows.yaml` (add `chunking` + `imscc_chunking` phases)
- `/home/user/Ed4All/LibV2/tools/libv2/scripts/backfill_dart_chunks.py` (NEW)
