# Worker E — Cross-system contracts audit

## Summary

- **10 contracts audited** (DART→CF HTML markers, DART `.quality.json` sidecar, CF→packager per-week dir, CF→packager JSON-LD gate, IMSCC manifest, Trainforge→LibV2 `manifest.json`+`course.json`, Trainforge→LibV2 `chunks.jsonl`, decision-event write surface, validator gate I/O, workflow phase routing, `config/*.yaml`).
- **Schematized + runtime-validated:** 2 (decision events via `schemas/events/decision_event.schema.json` + `lib/validation.py`; LibV2 `manifest.json` via `schemas/library/course_manifest.schema.json` + `LibV2/tools/libv2/validator.py:147-196`).
- **Schematized, not enforced at write time:** 2 (decision enum — write path warns only, ref `lib/decision_capture.py:82-93` + `:401-414`; LibV2 `catalog_entry` schema exists at `schemas/library/catalog_entry.schema.json` but no runtime validator invocation found in importer).
- **Convention-only (no schema):** 8 (everything else).
- **Top KG-impact risks:** (a) Trainforge `chunks.jsonl` shape is the KG node layer and has *no* schema — any drift in `chunk_type`, `bloom_level`, `content_type_label`, `concept_tags` silently corrupts every downstream join; (b) the CF→packager per-week directory convention is policed by filename regex (`package_multifile_imscc.py:82-107`) with no schema — a renamed page type vanishes from the IMSCC org tree; (c) phase-to-phase data routing in `MCP/core/workflow_runner.py:37-97` is a Python dict with no schema; a rename of an output key (e.g. `output_paths`) is a runtime KeyError *or* silent None propagation into the next phase's call site.

---

## Contract inventory

### 1. DART → Courseforge (HTML markers)

- **Producer:** `DART/multi_source_interpreter.py:837-960` (emits `.dart-section`, `.dart-document`, `<a class="skip-link">`, `<main role="main">`, `<section ... aria-labelledby="...">`); `DART/templates/gold_standard.html:339`.
- **Consumer:** `Courseforge/scripts/textbook-loader/textbook_loader.py:88-227` (uses `parsers.html_content_parser.HTMLContentParser` for primary parse, falls back to basic parse); `Trainforge/parsers/html_content_parser.py` reads DART HTML too when packaged into IMSCC.
- **Contract location:** Implicit. Documented only as prose in `MCP/tools/pipeline_tools.py:368-416` (`validate_dart_markers` tool checks four markers: `skip-link`, `role="main"`, `aria-labelledby`, `dart-section`/`dart-document`), and `DART/CLAUDE.md` "WCAG 2.2 AA Features". No JSON schema, no XSD.
- **Enforcement:** Runtime string-contains check in `validate_dart_markers` (MCP tool, not auto-invoked in any workflow phase). Courseforge's `TextbookLoader._basic_parse` falls back silently if markers are absent.
- **Drift risk:** Medium. DART is semi-stable; the emit site is one file.
- **KG-impact:** A DART semantic-class rename (e.g. `.dart-section--systems` → `.dart--systems-table`) would silently degrade Courseforge's textbook ingestor → objective extraction, polluting the KG's `LearningObjective` nodes with generic headings rather than typed sections.

### 2. DART → Courseforge (`.quality.json` sidecar)

- **Producer:** `DART/multi_source_interpreter.py:1313` (`report_path = Path(output_path).with_suffix('.quality.json')`).
- **Consumer:** `Courseforge/scripts/textbook-loader/textbook_loader.py:214-219` (attaches as `content.metadata['dart_quality']` if the sidecar exists; silently skipped otherwise).
- **Contract location:** Implicit. No schema file.
- **Enforcement:** None. Consumer treats absence as "no quality data" and does not gate ingestion on presence or shape.
- **Drift risk:** Low (producer/consumer are 1:1 today).
- **KG-impact:** If DART renames the sidecar suffix or restructures the quality report, Courseforge-ingested chunks lose source-reliability weighting, and KG provenance nodes lose the `source.dart_quality` annotation needed to filter low-confidence chunks from training sets.

### 3. Courseforge → Brightspace-packager (per-week directory layout)

- **Producer:** `Courseforge/scripts/generate_course.py:806-809` (iterates `data["weeks"]`, writes `week_NN/overview.html|content_NN_*.html|application.html|self_check.html|summary.html|discussion.html`).
- **Consumer:** `Courseforge/scripts/package_multifile_imscc.py:82-107` — globs `content_dir/week_*/`, orders by hardcoded dict `{"overview":0,"content":1,"application":2,"self_check":3,"summary":4,"discussion":5}` (`:94`), and stamps `ITEM_*`/`RES_*` identifiers into `imsmanifest.xml`.
- **Contract location:** Convention only. Hardcoded in both scripts; no schema or shared constants module.
- **Enforcement:** None at the structural level. Any HTML in `week_*/` gets packaged; the order dict falls through to `(99, name)` for unrecognized stems.
- **Drift risk:** High. Any new page type added to `generate_course.py` (e.g. a "reflection.html") is silently ordered after discussion and given a title derived from stem-munging (`:113`) with no KG identity.
- **KG-impact:** The packager's regex-driven item typing is the only source of the `data-cf-*`-independent `module_type` label the KG sees from the IMSCC organization tree. A new page type added on the emit side but not taught to `package_multifile_imscc.py` order dict lands the page into the KG as a nameless `item` without pedagogical role, degrading `module_sequence` edge inference.

### 4. Courseforge → Brightspace-packager (per-week LO validation gate)

- **Producer:** `generate_course.py` `<script type="application/ld+json">` per page (Worker C scope).
- **Consumer:** `package_multifile_imscc.py:127-153` calls `validate_page_objectives.validate_page` for every `week_*/*.html` page when `--objectives` is passed; refuses to package on violation unless `--skip-validation` (`:156-179`).
- **Contract location:** `Courseforge/scripts/validate_page_objectives.py` + inline comments in both scripts. No externalized schema.
- **Enforcement:** Hard fail (`raise SystemExit(2)`) when `--objectives` provided; silently skipped when the flag is omitted.
- **Drift risk:** Medium. The gate is opt-in, not default-on.
- **KG-impact:** When the flag is omitted (the default CLI invocation), CF can ship an IMSCC whose `learningObjectives` JSON-LD references IDs outside the canonical registry. Those LOs then fan out in Trainforge's `learning_outcome_refs` normalization and corrupt the `chunk → LearningObjective` join at the KG's most load-bearing edge.

### 5. Courseforge → Trainforge (IMSCC manifest + resource layout)

- **Producer:** `package_multifile_imscc.py:33-124` writes `imsmanifest.xml` under IMS CC 1.3 namespaces (`imsccv1p3`, `LOM/resource`, `LOM/manifest`), with `<organizations>/<organization structure="rooted-hierarchy">` containing one `<item identifier="WEEK_N">` per week.
- **Consumer:** `Trainforge/parsers/imscc_parser.py` (walks the manifest + opens resource `href`s); ultimately feeds `Trainforge/process_course.py` chunk generation.
- **Contract location:** Partially standardized (IMS Common Cartridge 1.3 spec), but the Ed4All-specific shape — week-item identifier prefix `WEEK_N`, resource id prefix `RES_*`, `type="webcontent"`, file layout `week_NN/*.html` — is convention only. No Ed4All XSD or JSON schema.
- **Enforcement:** `lib/validators/imscc.py:29-157` (IMSCCValidator) gate runs in the `packaging` phase of `course_generation` and `textbook_to_course`; checks structural validity of the zip + manifest.
- **Drift risk:** Medium — IMS spec stabilizes the XML shape, but Ed4All conventions on top of it are fragile.
- **KG-impact:** IMSCC is the single hop carrying CF's curated structure into Trainforge's chunk extractor. A convention drift (e.g. `WEEK_N` → `W_N`) would break `module_id` derivation at `process_course.py:1059-1066`, desyncing every chunk's `source.module_id` from the LO-registry module identifiers and cratering `chunk → module` KG joins.

### 6. Trainforge → LibV2 (`manifest.json`)

- **Producer:** `Trainforge/process_course.py:1743-1786` builds manifest dict with `course_id`, `sourceforge_version`, `chunk_schema_version` (`CHUNK_SCHEMA_VERSION = "v4"` at `:86`), `classification`, `structure`, `pedagogy`, `processing`, `statistics`; `_write_metadata():2618` writes to `output_dir/manifest.json`.
- **Consumer:** `LibV2/tools/libv2/importer.py:61-68` (`read_sourceforge_manifest`), `:128-158` (`extract_content_profile`), ultimately validated at `LibV2/tools/libv2/validator.py:147-196` against `schemas/library/course_manifest.schema.json`.
- **Contract location:** `schemas/library/course_manifest.schema.json` — requires `libv2_version`, `slug`, `import_timestamp`, `sourceforge_manifest`, `classification`, `content_profile`.
- **Enforcement:** **Actual JSON Schema validation** via `jsonschema.Draft7Validator` in `LibV2/tools/libv2/validator.py`, invoked from `importer.import_course` when `strict_validation=True` (default).
- **Drift risk:** Low — this is the one end-to-end schematized pipe. But note: the schema validates the *LibV2-side* manifest (post-import wrap), not Trainforge's output directly. Trainforge's `sourceforge_manifest` sub-dict shape is convention.
- **KG-impact:** Relatively well-guarded. Drift in Trainforge's pre-import manifest would fail strict import, which is visible. However, the `classification.division|primary_domain|subdomains|topics` fields derive from CLI args (`import_course` params) — not from course content — which means taxonomy mis-tagging silently propagates to KG (see Worker D).

### 7. Trainforge → LibV2 (`chunks.jsonl` corpus format)

- **Producer:** `Trainforge/process_course.py:1038-1216` (`_create_chunk`) with shape `{id, schema_version, chunk_type, text, html, follows_chunk, source, concept_tags, learning_outcome_refs, difficulty, tokens_estimate, word_count, bloom_level [, bloom_level_source, content_type_label, key_terms, misconceptions, summary, retrieval_text]}`; written to `corpus/chunks.jsonl` at `:1668`.
- **Consumer:** LibV2 retrieval engine (`LibV2/tools/libv2/retriever.py`, `indexer.py`, `concept_vocabulary.py`); `schemas/knowledge/instruction_pair.schema.json:34` references a chunk id shape of the form `{slug}_chunk_NNNNN` but does *not* constrain the chunk record itself.
- **Contract location:** **No chunk JSON Schema exists**. `CHUNK_SCHEMA_VERSION` is a string constant in `process_course.py:86`; `docs/schema/chunk-schema-v4.md` describes it in prose only.
- **Enforcement:** None at the JSON level. Trainforge self-test fixtures exercise the shape, and `lib/validators/leak_check.py:124-129` and `content_facts.py:231` receive chunks by duck-typing.
- **Drift risk:** **High.** This is the most consequential un-schematized artifact in the repo.
- **KG-impact:** `chunks.jsonl` is the KG's node layer. Every `chunk_type` (`explanation|example|procedure|...`), `bloom_level` enum, `content_type_label`, `difficulty` enum, `concept_tags` value is consumed as fact by retrieval, typed-edge inference, and eval. A silent rename on either side would mis-type nodes on the KG's primary surface. Bumping `CHUNK_SCHEMA_VERSION` is the only signal — consumers that don't check will read drifted records as authoritative.

### 8. Decision-capture contract (events)

- **Producer:** `lib/decision_capture.py` — `DecisionCapture.log_decision()` at `:441-479`, `_build_record()` assembles records, `_write_to_streams()` writes JSONL.
- **Consumer:** `MCP/tools/analysis_tools.py` (`analyze_training_data`, `preview_export_filter`); `export_training_data` MCP tool; downstream training-data export pipelines.
- **Contract location:** `schemas/events/decision_event.schema.json` — `decision_type` enum has 39 values (`:63-103`, not 40; verified); `tool` enum is `{dart, courseforge, trainforge, orchestrator}` (`:58`); `bloom_levels` items enum `{remember..create}` (`:165`); rationale `minLength: 20` (`:113`).
- **Enforcement:** `lib/decision_capture.py:401-414` (`_validate_record`) calls `lib/validation.py:validate_decision` (gated by `VALIDATE_DECISIONS` env-driven constant from `lib/constants.py`). Validation **warns** on failure (`:409: logger.warning`) and **stores issues on the record** (`:410`) but does **not** refuse the write — Trainforge's custom `instruction_pair_synthesis` + `preference_pair_generation` + Worker F's `typed_edge_inference` (`decision_capture.py:87-93`) are emitted despite not being in the schema enum.
- **Drift risk:** Medium-high. The enum+registry are in two places (schema file + in-code `ALLOWED_DECISION_TYPES` tuple) and both are out-of-date relative to real emit sites.
- **KG-impact:** Decision events are the **provenance layer** for every KG assertion. An un-enumerated `decision_type` passing through the warn-only gate means any query over "all `question_generation` decisions" will silently exclude Trainforge's `instruction_pair_synthesis` variant, breaking training-data provenance audits and making "why did the KG assert X?" non-answerable.

### 9. Validator contracts (`lib/validators/*.py`)

- **Producer:** `MCP/core/workflow_runner.py` → `MCP/hardening/validation_gates.py` runs gates, passing an `inputs: Dict[str, Any]` to each validator's `validate()`.
- **Consumer:** Each validator is also a producer of `GateResult`. The 9 validators consume heterogeneous inputs — see table below. `GateResult` shape is uniform (`MCP/hardening/validation_gates.py:48-78`): `{gate_id, validator_name, validator_version, passed, score?, issues[], execution_time_ms, inputs_hash?, timestamp, waived, error?}`. `GateIssue` shape is also common (`:34-45`): `{severity, code, message, location?, suggestion?}`.

| Validator | File | Input keys consumed |
|---|---|---|
| `ContentStructureValidator` | `lib/validators/content.py:39-47` | `html_path`, `html_content`, `week`, `module`, `gate_id` |
| `IMSCCValidator` | `lib/validators/imscc.py:35-45` | `imscc_path`, `gate_id` |
| `IMSCCParseValidator` | `lib/validators/imscc.py:166` | `imscc_path`, `gate_id` |
| `OSCQRValidator` | `lib/validators/oscqr.py:26` | (stub — thin wrapper) |
| `AssessmentQualityValidator` | `lib/validators/assessment.py:53-108` | `assessment_data`, `assessment_path`, `learning_objectives`, `min_score`, `gate_id` |
| `FinalQualityValidator` | `lib/validators/assessment.py:258-272` | `assessments`, `assessments_dir`, `min_score` |
| `BloomAlignmentValidator` | `lib/validators/bloom.py:67` | (inputs dict — Bloom verbs duplicated at `:21-45`, see Worker A) |
| `LeakCheckValidator` | `lib/validators/leak_check.py:24-129` | `assessment_data`, `chunks`, `max_leaks`, `max_boilerplate_chunk_fraction`, `boilerplate_ngram_tokens`, `strict_mode` |
| `QuestionQualityValidator` | `lib/validators/question_quality.py` | (question dict list) |
| `ContentFactValidator` | `lib/validators/content_facts.py:226-231` | `chunks`, `gate_id` |

- **Contract location:** `GateResult`/`GateIssue` are `@dataclass` in `MCP/hardening/validation_gates.py`. Input shapes are documented only in per-validator docstrings. `Validator` `Protocol` at `validation_gates.py:107-114` is duck-typed.
- **Enforcement:** Output shape is enforced structurally (Python dataclass). Input shape is not enforced — each validator does `inputs.get(...)` and defaults silently (`content.py:48`, `assessment.py:62-64`).
- **Drift risk:** Medium. Silent input-key drift results in default values being used (e.g. `min_score=0.8` default masks a missing threshold config).
- **KG-impact:** Validators decide which artifacts enter the KG via the validation gates in `config/workflows.yaml`. Silent defaulting means a `min_score: 0.7` meant for `oscqr_score` landing on `assessment_quality` instead returns a validator score computed against `0.8` default — the artifact enters the KG under false pretenses.

### 10. Workflow phase-to-phase data routing (`PHASE_PARAM_ROUTING`)

- **Producer:** Each phase's task returns a dict; `MCP/core/workflow_runner.py:87-97` (`PHASE_OUTPUT_KEYS`) extracts the declared keys from phase results into `workflow_state["phase_outputs"][phase_name]`.
- **Consumer:** `PHASE_PARAM_ROUTING` at `workflow_runner.py:37-82` — a Python dict mapping `{phase_name: {param_name: (source_type, *source_path)}}`; `param_mapper.py` (same directory) looks up values at task-creation time.
- **Contract location:** **In-code Python dict only**. Not a schema, not a YAML, not a doc.
- **Enforcement:** Runtime KeyError if a referenced key is missing. No pre-flight check. `param_mapper` may substitute `None` silently for some source types.
- **Drift risk:** **High.** A rename in any phase's returned dict (e.g. `output_paths` → `html_paths`) propagates as `None` to the next phase's call.
- **KG-impact:** The KG is only as good as the artifacts each phase hands off. If `libv2_archival` receives `html_paths=None` because `dart_conversion` renamed its output key, the archive bundle silently omits DART HTML and the KG loses the HTML-xpath provenance edges (`Trainforge/process_course.py:1070`) that tie chunks back to source elements for Section 508 audits.

### 11. `config/workflows.yaml` + `config/agents.yaml` shape

- **Producer:** Human-edited YAML.
- **Consumer:** `MCP/core/config.py:97-204` (`OrchestratorConfig.load`) — reads YAML, validates structure manually via `isinstance` checks (`:124-166`) rather than a JSON Schema, warns via `ValueError` on malformed structure.
- **Contract location:** **No externalized schema.** Field expectations live only in `config.py` dataclasses (`WorkflowPhase` at `:38-49`, `AgentConfig` at `:60-65`). Workflow-level keys like `retry_policy`, `poison_pill`, `validation_gates` are read raw via `p.get(...)` at `:148-159`.
- **Enforcement:** Per-field type checks in `config.py:130-166`; `validate()` at `:214-307` checks agent references, agent source-file existence, and validator importability (`:262-292`). No JSON Schema.
- **Drift risk:** Medium. New fields added in YAML are silently ignored by the loader (no strict-mode rejection).
- **KG-impact:** Workflow config governs which validation gates run — i.e. which quality filters the KG publishes past. A typo'd gate_id (e.g. `bloom_alignment` vs `bloom-alignment`) is accepted by the YAML parser and silently treated as a new gate with no handler, degrading KG node-quality guardrails without surfacing an error.

---

## Enforcement matrix

| # | Contract | Schema file? | Runtime check? | Drift risk | KG-impact severity |
|---|---|---|---|---|---|
| 1 | DART→CF HTML markers | No | Ad-hoc MCP tool only | Medium | Medium |
| 2 | DART→CF `.quality.json` sidecar | No | No | Low | Low |
| 3 | CF→packager per-week layout | No | Filename regex + order dict | High | High |
| 4 | CF→packager per-week LO gate | No (canonical JSON only) | Yes, opt-in via `--objectives` | Medium | High |
| 5 | CF→TF IMSCC Ed4All conventions | No (IMS CC 1.3 covers XML only) | IMSCCValidator structural | Medium | High |
| 6 | TF→LibV2 `manifest.json` | `schemas/library/course_manifest.schema.json` | Yes (`validator.py:147`) | Low | Low |
| 7 | TF→LibV2 `chunks.jsonl` | **No** | No | High | **Critical** |
| 8 | Decision-event write | `schemas/events/decision_event.schema.json` | Warn-only, not blocking | Medium-high | High |
| 9 | Validator gate I/O | `GateResult` dataclass (output only) | None on inputs | Medium | Medium |
| 10 | Workflow phase routing | No | Runtime KeyError / None | High | High |
| 11 | `config/*.yaml` | No | Manual isinstance checks | Medium | Medium |

---

## Must-schematize list (for KG reliability)

1. **Contract #7 (Trainforge `chunks.jsonl`)** — the KG's node layer has no schema. A `schemas/knowledge/chunk_v4.schema.json` with enumerated `chunk_type`, `bloom_level`, `difficulty`, required `source.{course_id,module_id,lesson_id}`, required `learning_outcome_refs[]` pattern, and validation hook in `Trainforge/process_course.py` before `chunks.jsonl` write is the single highest-leverage schematization. Unlocks every downstream KG join guarantee.
2. **Contract #10 (phase routing)** — move `PHASE_PARAM_ROUTING` + `PHASE_OUTPUT_KEYS` into `config/workflows.yaml` under per-phase `inputs_from:`/`outputs:` blocks, validated against a meta-schema at load. Currently a silent rename in *any* phase's return dict corrupts `libv2_archival` provenance bundling — and `libv2_archival` is the point where KG data actually gets published.
3. **Contract #3 (CF per-week directory layout)** — formalize the page-type vocabulary (`overview|content|application|self_check|summary|discussion`) as `schemas/academic/courseforge_page_types.schema.json` + validation in both `generate_course.py` (emit) and `package_multifile_imscc.py` (consume). Today the `module_type` KG label is derived from a regex match in one script and a hardcoded dict in another.
4. **Contract #8 (decision events)** — flip `_validate_record` from warn-only to fail-closed *once the ALLOWED_DECISION_TYPES tuple is reconciled with the schema enum* (39 in schema vs. 3 in code tuple, with real emit sites outside both). Decision provenance is how we audit KG claims — warn-only here undermines every audit trail.
5. **Contract #4 (per-week LO packaging gate)** — make `--objectives` the default, not opt-in, and promote the gate to a `validation_gates:` entry on the `packaging` phase. Today an omitted flag lets LO-fanout drift into the IMSCC and corrupt Trainforge's primary chunk→LO join.

## Can-remain-convention list

- **Contract #1 (DART HTML markers)** — DART is one producer, the emit site is one file, and WCAG validation covers the accessibility-relevant invariants. A schema would duplicate the WCAG checks without KG benefit.
- **Contract #2 (DART `.quality.json` sidecar)** — low cross-system fanout; consumer treats as soft metadata. Type-stub in consumer is cheaper than a schema.
- **Contract #5 IMSCC Ed4All conventions** — IMSCC 1.3 standardizes the XML; the Ed4All-specific identifier shapes (`WEEK_N`/`RES_*`) can be captured as constants shared between `generate_course.py` and `package_multifile_imscc.py` rather than a full schema. A shared `lib/courseforge_identifiers.py` module suffices.
- **Contract #9 (validator gate I/O)** — the `GateResult` dataclass already enforces output shape. A Protocol-documented input TypedDict per validator is cheaper than per-validator JSON Schema; the risk is silent input-key defaulting which is better fixed by removing `.get(..., default)` fallbacks in favor of `KeyError`.
- **Contract #11 (`config/*.yaml`)** — the current `OrchestratorConfig.validate()` at `MCP/core/config.py:214-307` catches the high-impact drift (missing agents, unimportable validators). Adding a JSON Schema on top would duplicate that coverage; instead, extend `validate()` to reject unknown keys in strict mode.

---

## New drift surfaced during audit

- **Decision enum count mismatch:** Worker plan + Ultraplan say 40 values; `schemas/events/decision_event.schema.json:63-103` has **39** entries. `lib/decision_capture.py:87-93` `ALLOWED_DECISION_TYPES` has **3** entries (`instruction_pair_synthesis`, `preference_pair_generation`, `typed_edge_inference`). Three-way divergence between plan, schema, and in-code registry.
- **`chunk_schema_version` duplicated in manifest.json:** written under both top-level key and `processing.chunk_schema_version` (`Trainforge/process_course.py:1749` + `:1779`). Either can drift against `CHUNK_SCHEMA_VERSION`.
- **`validate_dart_markers` is orphan:** the MCP tool at `MCP/tools/pipeline_tools.py:368` is not invoked by any phase in `config/workflows.yaml` — the DART→CF marker contract has a validator with no gate wired.
- **Warn-only behavior in decision validation is undocumented:** `lib/decision_capture.py:401-414` silently stores validation issues on the record's metadata; there is no downstream consumer that reads `metadata.validation_issues` to filter training exports, so the warning signal is effectively discarded.
