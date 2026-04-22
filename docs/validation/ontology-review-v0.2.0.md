# Ontology Review v0.2.0 — KG-publish-readiness audit

## § 0 Context & scope

This is a critical review of the unified ontology that landed on `dev-v0.2.0` after Workers R/S/T/U (PRs #14–#17). It audits the state at commit `925a69a`.

**Lens.** "Could we materialize this to a knowledge graph today?" Not "is the code correct?" and not "is the content pedagogically sound?" The audit asks whether the schemas, emitters, and parsers collectively produce a joinable, typed, provenance-bearing graph. The review treats the union of `schemas/` + the Courseforge emit path + the Trainforge parse and inference paths as a single ontology, because that is the union a KG loader would have to reconcile.

**What this is.** A descriptive catalog of gaps. Codex's 5 external claims are re-verified against source, and 14 additional findings are surfaced by direct schema/code read. Each finding carries one or more `path:line` citations against commit 925a69a, a severity class (Critical / High / Medium / Low), and a KG-publish impact note. A publish-readiness table covers 13 entity types. An impact matrix maps 12 example queries to the findings that would block them.

**What this is not.** Not a target-ontology proposal. Not a plan to fix anything. Not a review of `schemas/ONTOLOGY.md` as a document — that file is descriptive, and this review is critical of what it describes. Not an OSCQR or content-quality review. Not a pedagogical-framework audit. Not RDF/SHACL/OWL authoring. Not a choice between KG technologies; the gaps apply regardless of backing store.

**Review method.** Every file:line citation below was re-verified against the commit by direct read of the cited file's line range. Where the planner's input had a stale anchor (for example, an off-by-a-few-lines range for a precedence map), the anchor was corrected before citation. Findings are organized by severity and theme. Section § 6 is the only place where prescriptive language appears; § 0–§ 5, § 7, and § 8 are descriptive throughout.

**Downstream.** This review is input to a separate v0.3 remediation planning session. The five directions in § 6 are pointers only. Sequencing, ownership, and dependency analysis are out of scope here.

## § 1 Executive summary

**Verdict — can we publish to a KG today? No.**

The ontology has three structural blockers that would cause silent data loss or unreliable joins under a naive graph materialization, plus eight consistency gaps that would compound as new courses are ingested.

A publisher can stand up nodes and edges, but the resulting graph would fail straightforward queries like "which chunks define concept X?" or "which questions target learning objective Y?" without rebuilds. The gaps are not random; they cluster in Chunk, Concept, and LearningObjective — the three entities that carry the most join traffic.

Raw ingestion appears permissive by design (additional properties allowed at every schema level, evidence polymorphic across rules, difficulty mapping in code only). That permissiveness is defensible for exploratory emit, but it means nothing prevents a KG-incompatible emit from passing validation. There is no stricter "publish profile" that tightens the contract before the graph is materialized.

**What works.**

The typed-edge provenance pattern is sound. Every edge carries `{rule, rule_version, evidence}` with an optional `confidence` bounded to `[0, 1]` (`schemas/knowledge/concept_graph_semantic.schema.json:48–64`), and the three inference rules have stable rule names plus rule-version integers that accompany each emitted edge.

Bloom's taxonomy is consistent across all surfaces. The six values `remember / understand / apply / analyze / evaluate / create` appear as an enum in the LO schema (`schemas/academic/learning_objectives.schema.json:202–206`), in `schemas/knowledge/instruction_pair.schema.json:42–44`, and in `schemas/events/trainforge_decision.schema.json:23–26`. Every enumerated value is reachable from the Courseforge emit path via `BLOOM_TO_DOMAIN` and `detect_bloom_level`.

The Courseforge → Trainforge bridge via `data-cf-*` attributes and JSON-LD works on the happy path. `Trainforge/parsers/html_content_parser.py:319–331` preferentially reads JSON-LD and falls back to `data-cf-*` without losing the field set it cares about — the exception being the `data-cf-objective-ref` attribute noted in B3 below.

`CourseManifest` (per the Worker S unification) and `DecisionEvent` are the most schema-complete entities in the repository. Phase-0 decision-capture event IDs (`EVT_<16-hex-chars>`) and monotonic per-run sequence numbers are in place at `schemas/events/decision_event.schema.json:9–18`, which gives a strong ledger foundation — although C1 and C2 below note that the ledger does not yet propagate to artifact-level provenance.

**Top three blockers.**

1. **LO ID field-name drift.** The canonical schema uses `objectiveId` as the required identifier (`schemas/academic/learning_objectives.schema.json:193, 238`); the runtime emit writes `"id"` in JSON-LD (`Courseforge/scripts/generate_course.py:523`); the parser reads `"id"` (`Trainforge/parsers/html_content_parser.py:323`). Two field-name namespaces exist for the same logical identifier. A KG loader keyed to the schema sees zero objectives; a loader keyed to the runtime passes but emits data the schema rejects.

2. **Cross-course concept collisions.** The concept node schema requires only `id` (`schemas/knowledge/concept_graph_semantic.schema.json:22–33`) — no `course_id`, no origin namespace. A concept named `accessibility` emitted from courses DIGPED_101 and WCAG_201 merges into a single node with no way to recover origin after merge.

3. **Position-based chunk IDs.** Chunk IDs are constructed as `f"{prefix}{start_id:05d}"` at `Trainforge/process_course.py:1003` and `:1027`; re-chunking a course with a different splitter configuration shifts every ID. Edge evidence records chunk IDs by value (for example `Trainforge/rag/inference_rules/is_a_from_key_terms.py:163` writes the chunk ID into the provenance dict), so a re-chunk silently invalidates every existing edge's evidence pointer.

## § 2 Codex's 5 claims: verdicts

Each claim is re-verified against source. "Verified" means the claim describes the code accurately at 925a69a. "Verified with nuance" means the claim is directionally correct but one or more specifics need refinement. "Partially refuted" means at least one specific in the claim is wrong while a related concern stands.

Of Codex's 5 claims: 3 are verified, 1 is verified with nuance, 1 is partially refuted. None is fully refuted. Even the partially-refuted Claim 5 points at a real drift — Codex just described the drift direction incorrectly.

### Claim 1 — LO ID inconsistency (`objectiveId` vs `id`)

**Status: VERIFIED.**

Evidence:

- Canonical field name in schema is `objectiveId` at `schemas/academic/learning_objectives.schema.json:193`. It is marked required at `:238` (`"required": ["objectiveId", "statement", "bloomLevel"]`), confirming this is the schema-authoritative identifier, not an optional alias.

- Courseforge JSON-LD emit writes `"id"` at `Courseforge/scripts/generate_course.py:523` inside `_build_objectives_metadata`. The dict literal constructs `{"id": o["id"], "statement": ..., "bloomLevel": ..., ...}` and never uses `objectiveId`.

- Trainforge parser reads `lo.get("id")` at `Trainforge/parsers/html_content_parser.py:323` in the JSON-LD strategy branch, then constructs a `LearningObjective` dataclass keyed by `id`.

- The HTML data-attribute surface IS consistent: emitted as `data-cf-objective-id` at `Courseforge/scripts/generate_course.py:314`, parsed with the matching regex at `Trainforge/parsers/html_content_parser.py:341`. The inconsistency lives specifically in the JSON-LD payload, not across both metadata surfaces.

- Downstream normalization at `Trainforge/process_course.py:1501–1507` applies `obj_id.lower().strip()` and then strips a week prefix via `WEEK_PREFIX_RE.sub('', normalized)`. This is a second-level identity transform that the LO schema does not describe or constrain.

Impact on KG publish: a graph loader keyed on the schema's `objectiveId` would see zero objectives from current Courseforge output. A loader keyed on the runtime `id` field would pass but would fail strict schema validation.

The existence of a second normalization step (lowercase + week-prefix strip) means even picking a field name is insufficient — the identity contract needs to cover case and prefix handling too. Nothing in the repo picks authoritatively between schema and runtime.

### Claim 2 — Edge vocabulary narrowness (only 3 edge types)

**Status: VERIFIED.**

Evidence:

- Edge-type enum at `schemas/knowledge/concept_graph_semantic.schema.json:46` is exactly `["prerequisite", "is-a", "related-to"]`. No other values pass validation.

- The precedence map at `Trainforge/rag/typed_edge_inference.py:48–52` covers the same three (`is-a: 3`, `prerequisite: 2`, `related-to: 1`) with no provision for other types.

- Rule modules are one-per-type: `Trainforge/rag/inference_rules/is_a_from_key_terms.py`, `prerequisite_from_lo_order.py`, `related_from_cooccurrence.py`. Each rule emits edges only of its own type.

- No pedagogical edge types exist. A repo-wide grep finds no emitter for `assesses` (question → LO), `exemplifies` (example → principle), `misconception-of` (wrong answer → concept), `derived-from-objective` (chunk → LO as a graph edge rather than a list reference), or `defined-by` (concept → chunk that introduces it). These relationships are present as data in JSON-LD, chunk fields, or activity attributes, but none surface as typed edges.

Impact on KG publish: the graph can express taxonomic hierarchy (via `is-a`) and curricular ordering (via `prerequisite`), plus a weak catch-all (`related-to`).

It cannot represent the pedagogical relationships that carry the training signal: which question assesses which objective, which concept a misconception mis-identifies, which chunk defines a concept first. All of those connections exist in the data; the ontology simply does not offer a typed surface for them.

A KG built today would under-represent precisely the structure a curriculum graph is most valuable for.

### Claim 3 — Permissive concept/edge contracts

**Status: VERIFIED.**

Evidence:

- `schemas/knowledge/concept_graph_semantic.schema.json` sets `additionalProperties: true` at the document root (line 8), the node object (line 27), the edge object (line 40), and the provenance sub-object (line 59). Every level is open for extension.

- Node `required` is only `["id"]` at line 26. `label` and `frequency` are optional at lines 30–31. A node with no label, no frequency, and an opaque `id` passes validation.

- Edge `required` at line 39 lists `["source", "target", "type", "provenance"]` — no structural constraint tying `source` and `target` to existing entries in `nodes[]` (see also A6 below).

- `confidence` is bounded to `[0, 1]` at lines 48–51 but has no default and is not required. An emitter may omit it, and two edges of the same `(source, target, type)` with different confidences cannot be distinguished by a loader without per-emitter knowledge.

- Provenance `evidence` is typed as `{"type": "object", "additionalProperties": true}` at line 63. The shape varies per rule:
  - `Trainforge/rag/inference_rules/prerequisite_from_lo_order.py:143–148` emits `{target_first_lo, target_first_lo_position, source_first_lo, source_first_lo_position}` — LO-position anchors.
  - `Trainforge/rag/inference_rules/is_a_from_key_terms.py:159–168` emits `{chunk_id, term, definition_excerpt, pattern}` — chunk anchor plus matched linguistic pattern.
  - `Trainforge/rag/inference_rules/related_from_cooccurrence.py:72–79` emits `{cooccurrence_weight, threshold}` — no chunk or LO anchor at all; a related-to edge is traceable to a rule and a numeric weight but not to a specific textual source.

Impact on KG publish: a loader cannot rely on any node field beyond `id`, nor on any edge field beyond `source`, `target`, `type`, and the presence of `provenance`.

Analytical queries such as "show me the chunks that support any is-a edge into concept X" are feasible only by hard-coding per-rule evidence shape. The schema offers no discriminator; a consumer either duplicates the rule catalog or treats evidence as opaque.

This is a classic tradeoff: raw-ingestion permissiveness is an asset for exploratory emit, but once the graph is being published, consumers need a tighter contract. Nothing in the current schema layout provides a route from permissive ingest to tight publish.

### Claim 4 — LO coverage coupling silently drops content

**Status: VERIFIED with nuance.**

Evidence:

- `schemas/knowledge/instruction_pair.schema.json:36–41` requires `lo_refs` with `minItems: 1` and `minLength: 1` per item. `schemas/knowledge/preference_pair.schema.json:44–48` mirrors that constraint.

- `Trainforge/synthesize_training.py:119–121` defines eligibility tersely: `_eligible(chunk)` returns `bool(chunk.get("learning_outcome_refs")) and bool(chunk.get("id") or chunk.get("chunk_id"))`.

- The synthesis loop at `Trainforge/synthesize_training.py:243–246` increments `stats.chunks_skipped_no_lo` and `continue`s on ineligible chunks. No per-chunk log line, no reason code, no chunk-ID list written out.

- Upstream, `_extract_objective_refs(item)` is invoked at `Trainforge/process_course.py:1088`. Its logic prefers structured `learning_objectives` from the parser (`lo.id` or `lo.get("id")`), then falls back to regex CO/TO extraction from key concepts, and can legitimately return `[]` with no warning when neither path yields anything.

Nuance: the exclusion behavior itself is intentional. Pairs without LO refs would not train coherently, and filtering them at synthesis time is correct.

The real gap is that upstream failure to attach LO refs is silent. A chunk entering synthesis with an empty `learning_outcome_refs` is skipped without any log identifying which chunk or why. The `_extract_objective_refs` path does not distinguish "no LO in source" from "LO in source but parse failed". Coverage regressions cannot be diagnosed from run output.

Impact on KG publish: if the KG is the seed corpus for SFT/DPO, coverage degrades invisibly as JSON-LD emission regresses upstream.

The aggregate `chunks_skipped_no_lo` counter exists but does not support a per-chunk audit, a per-course breakdown, or a comparison against the previous run. An entire module's worth of chunks can drop out of training without surfacing in the quality report.

### Claim 5 — Question-type enum drift

**Status: PARTIALLY REFUTED — direction reversed, drift is real.**

Evidence:

- `Trainforge/generators/question_factory.py:81–89` `VALID_TYPES` is `["multiple_choice", "multiple_response", "true_false", "fill_in_blank", "short_answer", "essay", "matching"]` — 7 values, and it **does** include `multiple_response`.

- `schemas/events/trainforge_decision.schema.json:62–65` `question_type` enum is `["multiple_choice", "true_false", "short_answer", "essay", "matching", "fill_in_blank", "ordering", "hotspot"]` — 8 values, and it **does** include `ordering` and `hotspot`.

- Symmetric drift: factory-only value is `multiple_response` (1 value); schema-only values are `ordering` and `hotspot` (2 values). Overlapping core: 6 values (`multiple_choice`, `true_false`, `short_answer`, `essay`, `matching`, `fill_in_blank`).

Codex described the drift with the direction reversed — claimed the schema had `multiple_response` and the factory had `ordering`/`hotspot`. That specific attribution is incorrect. The structural concern, however, is correct: the two vocabularies disagree, and by three values total.

Impact on KG publish: a `Question` node materialized from a factory call with `type="multiple_response"` would fail validation against the `trainforge_decision` event schema if passed through the event surface.

The schema enum values `ordering` and `hotspot` are unreachable from the generator — any KG consumer written to the schema's vocabulary will index slots that are never populated.

A third vocabulary (see B4 below) complicates this further: `assessmentSuggestions` on LOs uses yet another set of nine values that do not line up with either of these enums, bringing the total to at least three distinct "question kind" vocabularies in play simultaneously.

## § 3 Additional findings beyond Codex

Fourteen findings organized by theme. Each entry has an ID, severity, evidence with file:line anchors, and a KG-publish impact note. The severity rubric: **Critical** = blocks naive graph publish; **High** = produces unreliable joins or silent data loss; **Medium** = produces inconsistent analytics or requires out-of-band bookkeeping; **Low** = quality-of-life or latent risk.

### Theme A — Identity and join stability

**A1 (Critical) — Cross-course concept collisions.**

The concept node schema at `schemas/knowledge/concept_graph_semantic.schema.json:22–33` requires only `id`; there is no `course_id`, no `namespace`, no origin field, and no constraint that prevents merging nodes that originated in different courses.

The Courseforge emit does not scope concept slugs to a course. The Trainforge `normalize_tag` routine at `Trainforge/process_course.py:273–286` does not prepend a course identifier.

Concrete failure case: a course DIGPED_101 contains the key term "accessibility" and emits node `id="accessibility"`; a separate course WCAG_201 also contains "accessibility" and emits the same `id`. A KG loader that treats the concept graph as append-only merges these into one node with no retrievable origin.

Impact on KG publish: cross-course analytics such as "which concepts appear in two or more courses?" are impossible without re-scanning the source data. Course-scoped queries such as "all concepts in the pedagogy course" devolve into scans over chunk-level references. The graph cannot answer any query whose answer depends on knowing which course a concept came from.

**A2 (Critical) — Position-based chunk IDs.**

`Trainforge/process_course.py:1003` and `:1027` construct chunk IDs as `f"{prefix}{start_id:05d}"`, where `start_id` is an integer that advances by position through the document. A re-chunk with a different `MAX_CHUNK_SIZE`, a different splitter, or even a textual edit earlier in the document shifts every downstream ID by one or more positions.

Edge evidence records chunk IDs by value. `Trainforge/rag/inference_rules/is_a_from_key_terms.py:163` writes `"chunk_id": chunk.get("id")` into the provenance dict. The preference-factory misconception ID at `Trainforge/generators/preference_factory.py:140–143` embeds the chunk ID into the misconception ID.

Impact on KG publish: the graph is not rerun-stable. Every existing edge's evidence pointer, every misconception ID, and every chunk-referencing artifact becomes stale after a re-chunk. A publisher has no way to map from the old IDs to the new ones without re-running upstream inference, which means a re-chunk is effectively a full graph rebuild.

**A3 (High) — Irreversible LO ID case normalization.**

`Trainforge/process_course.py:1501–1507` applies `obj_id.lower().strip()` to the LO ID when attaching it to a chunk's `learning_outcome_refs`, then strips a week prefix (`w01-`, `w02-`) via `WEEK_PREFIX_RE.sub('', normalized)`.

Questions that entered the pipeline from external IMSCC sources, or earlier pipeline runs, may still persist their LO references in the original casing (for example `CO-04`). No schema declares that LO IDs are case-insensitive, and no emit point guarantees the same case everywhere.

Impact on KG publish: KG joins between chunks and questions on LO ID fail silently when the two sides arrive with different case. The case-insensitive invariant is baked in only on the chunk side; the rest of the graph is at the mercy of whoever emitted the ID.

**A4 (High) — Concept ID normalization mismatch between emit and consume.**

Courseforge `_slugify` at `Courseforge/scripts/generate_course.py:169–172` produces lowercase-hyphenated slugs from arbitrary-length text with no truncation.

Trainforge `normalize_tag` at `Trainforge/process_course.py:273–286` enforces stricter rules: after the same lowercase/punctuation-strip transform, it splits on `-` and truncates to 4 tokens, and rejects tags whose first character is not alphabetic.

Concrete divergence case: the phrase "Instructional Design and Technology Enhanced Learning" slugifies in Courseforge to `instructional-design-and-technology-enhanced-learning` but normalizes in Trainforge to `instructional-design-and-technology`.

Impact on KG publish: the same human-readable concept produces different graph node IDs on the two sides of the bridge. Edges emitted by Trainforge (which uses `normalize_tag`) reference node IDs that the Courseforge-supplied node set (using `_slugify`) does not contain; the graph either fragments into two concept populations or merges them incorrectly depending on load order.

**A5 (High) — Silent slug collisions.**

`Trainforge/rag/inference_rules/is_a_from_key_terms.py:52–64` provides a rule-local `_slugify` that lowercases, strips non-alphanumeric-non-hyphen characters, and collapses whitespace to hyphens. There is no collision tracking: inputs `"Cognitive Load Theory"` and `"cognitive-load-theory"` both produce the slug `cognitive-load-theory`.

When the same slug is produced from two distinct source spans, node `frequency` accumulates; distinct semantic sources collapse into a single node with no telltale in the output.

Impact on KG publish: concept frequency counts are inflated in a way that cannot be reconstructed after the fact. Slug collision is a well-known hazard for content-addressed identifiers, and the ontology has no mitigation — no `aliases[]`, no pre-collision audit log, no warning.

**A6 (Medium) — No referential integrity between edges and nodes.**

Edge `required` at `schemas/knowledge/concept_graph_semantic.schema.json:39–46` lists `["source", "target", "type", "provenance"]`, but the schema does not constrain `source` or `target` to reference an existing entry in `nodes[]`.

An edge pointing to a removed or never-emitted node passes schema validation. The rule emitters assume but do not enforce that their targets exist in the frequency-aggregated node set.

Impact on KG publish: orphaned edges require a post-load referential sweep to detect. A KG loader that trusts the schema will ingest edges whose endpoints are not nodes, leaving phantom references that fail at query time rather than load time.

### Theme B — Missing bidirectional links

**B1 (Critical) — Concept nodes carry no back-reference to chunks.**

`schemas/knowledge/concept_graph_semantic.schema.json:22–33` defines node properties as `{id, label, frequency}` and permits additional properties but does not define `occurrences[]`, `chunks[]`, `defined_in[]`, or any equivalent back-reference.

The edge set points concept → concept (via is-a / prerequisite / related-to). The chunk set points chunk → concept (via `concept_tags`). The reverse direction — concept → chunks that mention it, or concept → chunk that first defines it — is not a graph property today. It is only reconstructable by scanning the entire chunk store.

Impact on KG publish: answering "which chunks define concept X?" requires O(N) in chunks. This is arguably the single most fundamental concept-graph query, and its absence means concept navigation cannot use graph traversal.

**B2 (High) — Misconceptions are prose-only with unstable IDs.**

Courseforge emits misconceptions as `{misconception: "...", correction: "..."}` free text in the JSON-LD `misconceptions` array at `Courseforge/scripts/generate_course.py:591`. There is no stable ID, no link to the concept a misconception concerns, and no link to the LO it would undermine.

Trainforge subsequently assigns a synthetic `misconception_id` at `Trainforge/generators/preference_factory.py:140–143` as `f"{chunk_id}_mc_{index:02d}_{short}"`, where `chunk_id` is position-dependent (see A2), `index` depends on ordering within the chunk, and `short` is an 8-char SHA256 prefix of the misconception text. If chunk ordering shifts, every misconception gets a new ID.

Impact on KG publish: misconception identity is not stable across re-runs. A KG that stores misconceptions as nodes gets a new node population for the same course on re-emit. Any analytics over "recurring misconceptions" or "misconceptions targeting concept Z" needs re-indexing every run.

**B3 (High) — `data-cf-objective-ref` emitted but never parsed.**

`Courseforge/scripts/generate_course.py:378` and `:491` write `data-cf-objective-ref="..."` on `.self-check` and `.activity-card` elements respectively. This attribute binds an activity to the LO it exercises.

A recursive grep of the Trainforge source tree for `data-cf-objective-ref` returns zero files. The only match in the repository is in a metadata-extraction test at `Trainforge/tests/test_metadata_extraction.py`, which does not integrate the attribute into the runtime parser.

Impact on KG publish: the activity → LO binding is rendered in HTML for human consumption and is machine-readable in principle, but the current runtime parser does not extract it. A KG that wanted to answer "which activities exercise LO Y?" would have to add the parse. Today the edge is thrown away between emit and consume.

**B4 (Medium) — `assessmentSuggestions` is an orphaned vocabulary.**

`schemas/academic/learning_objectives.schema.json:219–225` enumerates 9 values for `assessmentSuggestions`: `exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study`.

These have no documented mapping to `question_factory.VALID_TYPES` (`multiple_choice, multiple_response, true_false, fill_in_blank, short_answer, essay, matching` — 7 values at `Trainforge/generators/question_factory.py:81–89`) or to `trainforge_decision.question_type` (8 values; see Claim 5).

Courseforge at `Courseforge/scripts/generate_course.py:531–541` emits `assessmentSuggestions` using a hard-coded Bloom → `[multiple_choice, short_answer, ...]` map that uses a fourth, ad-hoc vocabulary (overlapping with but not identical to the factory).

Impact on KG publish: three to four independent vocabularies describe the same abstract concept — "kind of assessment" — and the ontology does not document a bridge between them. KG queries that want to reason across "assessment kind" have to pick a vocabulary and translate.

### Theme C — Provenance and run tracking

**C1 (High) — No timestamps on chunks or graph nodes/edges.**

`schemas/knowledge/concept_graph_semantic.schema.json:14–15` places `generated_at` only at the document level. Individual nodes and edges carry no `created_at`.

Chunk records at `Trainforge/process_course.py:1080–1092` carry `schema_version` but no timestamp. The enrichment pass at `Trainforge/process_course.py:1094+` adds Bloom level, content type, and related fields but no creation time.

Impact on KG publish: replaying a course cannot be distinguished from an incremental update at node/edge or chunk granularity. The graph cannot age out stale assertions, cannot surface "assertions added in the last 24 hours", and cannot support a query like "all edges added after run R completed". Out-of-band bookkeeping (filesystem mtime, git commit IDs) would have to stand in.

**C2 (High) — No `run_id` or `generated_by` on chunks or graph nodes/edges.**

`schemas/events/decision_event.schema.json:19–22` defines `run_id` as a required field on every decision event, so run identity is first-class in the event ledger.

That identity does not propagate to artifacts. A chunk does not know which run produced it. A node or edge does not know either. The provenance block on edges names the rule but not the run that executed it.

Impact on KG publish: a KG assertion cannot be traced back to the run that emitted it. Rollback ("undo everything run R produced") is impossible at graph granularity. Incremental publishing is possible only if the publisher maintains external run-to-artifact bookkeeping, which duplicates what the decision event ledger already contains.

**C3 (Medium) — Trainforge uses `decision_type` values not in the schema enum.**

`Trainforge/synthesize_training.py:226, 261, 287` log `decision_type="instruction_pair_synthesis"` and `"preference_pair_generation"`. The `decision_event.schema.json` `decision_type` enum at `:63–102` lists 40 values; neither of these two appears.

`DecisionCapture` writes events without a schema gate, so the mismatch is silent. The events land in the JSONL ledger, the synthesis completes, and no error surfaces.

Impact on KG publish: any KG consumer that validates the decision-event ledger strictly against the schema would reject Trainforge's synthesis events or have to carry an undocumented extension list. The schema is under-counting the decision vocabulary it is supposed to enumerate.

### Theme D — Enum alignment and vocabulary drift

**D1 (Medium) — Content type labels are free-string across four surfaces.**

`Courseforge/scripts/generate_course.py:553–556` emits `contentType` in JSON-LD. `_infer_content_type` at `Courseforge/scripts/generate_course.py:388–405` enumerates 8 inferred labels in code: `definition, example, procedure, comparison, exercise, overview, summary, explanation`.

The HTML surface `data-cf-content-type` at `Courseforge/scripts/generate_course.py:423` is a free-string attribute. Chunk `content_type_label` propagates downstream. `schemas/knowledge/instruction_pair.schema.json:46–50` accepts `content_type` as any non-empty string with no enum constraint.

Impact on KG publish: the KG's content-type axis depends entirely on spelling discipline in `_infer_content_type`. A drift in spelling ("procedure" vs "procedures" vs "step-by-step") fragments the axis with no schema-level detection. Analytics aggregations will under-count.

**D2 (Medium) — `moduleType` enum defined but never consumed.**

`Courseforge/scripts/generate_course.py:571–584` builds JSON-LD metadata that includes `moduleType: module_type`; the enum includes `overview / content / application / assessment / summary` (plus variants) per the Worker U ontology map.

A Trainforge-wide grep for `moduleType` returns only `Trainforge/tests/test_metadata_extraction.py`. No runtime parser reads the field, and no chunk or graph element carries a module-type derived from it.

Impact on KG publish: downstream consumers receive `moduleType` in JSON-LD but the main Trainforge pipeline does not join on it. A KG built from Trainforge output will not have module-type partitioning, and questions like "show me all 'assessment' modules across courses" will have no graph-level answer.

**D3 (Low) — `BLOOM_TO_DIFFICULTY` mapping lives in code only.**

`Trainforge/process_course.py:118–125` defines the 6-level-Bloom → 3-level-difficulty reduction: `remember` and `understand` both map to `foundational`; `apply` and `analyze` to `intermediate`; `evaluate` and `create` to `advanced`.

No schema documents this mapping. Difficulty appears as a chunk field; schema-level difficulty constraints (for example the `schemas/events/trainforge_decision.schema.json:91–94` enum `[easy, medium, hard]`) use a different vocabulary entirely. A `mixed` value referenced in some dataclass usage is not produced in the emit path today.

Impact on KG publish: difficulty semantics are discoverable only by reading Python. KG consumers replicate the mapping to reason about it. The two difficulty enums (`foundational/intermediate/advanced` vs `easy/medium/hard`) have no bridge.

**D4 (Low) — `keyTerms` definition fidelity depends on JSON-LD availability.**

`Courseforge/scripts/generate_course.py:560–563` emits JSON-LD `keyTerms` as structured `{term, definition}` objects. The parallel HTML attribute `data-cf-key-terms` at `Courseforge/scripts/generate_course.py:425` is a comma-separated list of term slugs only.

If JSON-LD is absent or fails to parse, Trainforge's fallback path for key terms sees slugs with no definitions. Concept nodes produced in the fallback path therefore carry neither a definition field nor any link to the text that defined them.

Impact on KG publish: the same course re-emitted without JSON-LD yields a lower-fidelity graph. This makes the JSON-LD path implicitly load-bearing for graph quality while the schema does not require it.

## § 4 KG-publish-readiness per entity

Each entity type a KG loader would materialize is evaluated for publish-readiness. "Publish-ready" means the entity can be loaded with stable identity, typed relations, and provenance sufficient for basic queries. Ratings: **Yes** = ready now; **Mostly** = ready with minor touch-ups; **Partial** = workable but with structural gaps; **No** = publishes but fails basic queries.

| Entity | Ready? | What blocks |
|---|---|---|
| Course | Mostly | Slug-based ID is stable-ish; no `@type`, no IRI, no `created_at` at course level. `CourseManifest` covers most of the gap. |
| Module | No | No stable global ID; `moduleNumber` in the JSON-LD is a sequential int per course, not unique across courses. |
| Page | No | Not a first-class schema entity; exists only as `pageId` string in `_build_page_metadata`'s JSON-LD output. |
| LearningObjective | No | Field-name drift (Claim 1); case-normalization asymmetry (A3); no stable bridge between JSON-LD emit and chunk reference. |
| Chunk | No | Position-based ID (A2); no `created_at` (C1); no `run_id` (C2); LO refs case-lossy (A3). |
| Concept | No | Cross-course collision (A1); no chunk back-reference (B1); emit/consume slug mismatch (A4); silent slug collisions (A5). |
| TypedEdge | Partial | Provenance pattern sound; evidence polymorphic (Claim 3); no referential integrity against nodes (A6). |
| KeyTerm | No | Not a first-class schema entity; lives as nested JSON-LD object; fallback path loses definitions (D4). |
| Misconception | No | Prose-only with unstable synthetic IDs (B2); no link to concept or LO. |
| Question | Partial | Type enum drift (Claim 5); `objective_id` case ambiguity (A3). |
| Assessment | Mostly | Dataclass-based with decent fields; no schema-level definition; `questions[]` inherits Question problems. |
| CourseManifest | Yes | Most complete entity; 4/6 universal fields populated; provenance block present. |
| DecisionEvent | Yes | Schema-enforced at `schemas/events/decision_event.schema.json`; event ledger functional — although C3 values slip past the enum gate, the overall contract is the strongest in the repo. |

Aggregate: 2 Ready, 2 Mostly, 2 Partial, 7 No. The blockers cluster in the three entities that carry the most join traffic — Chunk, Concept, and LearningObjective — and in the entities that are not schema-first (Page, KeyTerm, Misconception). `CourseManifest` and `DecisionEvent` are the functional reference for what "schema-complete" looks like in this repo; the other entities sit between 1 and 3 structural gaps behind that standard.

The "schema-first" observation is worth dwelling on. Of the 13 entity types, only 4 are backed by a JSON Schema in `schemas/` (LearningObjective, Chunk-as-instruction-pair/preference-pair, Concept-graph nodes and edges, DecisionEvent). The rest live entirely in code as dataclasses, dict literals, or emergent structure in JSON-LD. A publisher wanting to reason about Page, Module, KeyTerm, or Misconception as first-class entities has no schema to validate against and no guarantee of structural stability from one pipeline run to the next.

A publisher proceeding today would get clean event ledger ingest, partial typed-edge ingest, usable course and assessment scaffolding, and broken or unusable chunk / concept / LO / misconception layers. Since those four layers are where the content and curriculum signal lives, the practical result is a graph that looks populated but answers few of the queries that motivate KG publication in the first place.

Chunk is the most-cited entity in the repo by reference count (every edge-evidence pointer, every instruction pair, every preference pair, every LO back-reference goes through Chunk). The fact that its ID is position-dependent is therefore a blast-radius-maximizing flaw: one upstream change invalidates the most-referenced identifier set in the graph.

## § 5 Impact matrix — which queries break

Representative queries a KG-backed curriculum tool would want to answer, mapped to the findings that block them today. "Difficulty to unblock" estimates the scope of change: **Easy** = localized to one emit or parse site; **Moderate** = schema + emit + parse coordination; **Hard** = structural change across multiple artifacts and retrofitting existing data.

| Query | Blocked by | Difficulty to unblock |
|---|---|---|
| "Which chunks define concept X?" | B1 (no back-reference from concept to chunk) | Moderate: add `occurrences[]` to node schema and populate from the key-terms rule emit. |
| "Which concepts appear in both courses A and B?" | A1 (cross-course collision) + B1 | Hard: add `course_id` to node; stop naive merge across courses; retrofit existing data. |
| "Which questions target LO Y?" | A3 (case drift) + Claim 1 (field-name drift) | Easy on the case axis; moderate on the field-name axis (needs schema + Courseforge + Trainforge coordination). |
| "Prerequisite paths from concept X to concept Y" | Works today (prerequisite edges exist with provenance) | — |
| "Evidence for edge (X, prerequisite, Y)" | Claim 3 (polymorphic evidence) | Moderate: per-rule evidence sub-schema with a discriminator. |
| "Assertions added in run R" | C1 + C2 (no timestamp, no `run_id` on artifacts) | Hard: retrofit provenance to every chunk and every graph node/edge. |
| "Chunks excluded from training in the last run, with reason" | Claim 4 (aggregate-only counter) | Easy: per-chunk reason log emitted into the synthesis stats. |
| "Misconceptions targeting concept Z" | B2 (prose-only, unstable IDs; no concept link) | Hard: introduce a stable Misconception entity with links to concept and LO. |
| "Activities that exercise LO Y" | B3 (`data-cf-objective-ref` unparsed) | Easy: parse the existing attribute in Trainforge. |
| "All concepts with a definition available" | D4 (fallback path loses definitions) | Moderate: require JSON-LD or promote `data-cf-key-terms` to carry definitions. |
| "Questions of type `ordering` or `hotspot`" | Claim 5 (factory cannot produce them; schema enum cannot be populated) | Moderate: align the two vocabularies in both directions. |
| "Modules partitioned by `moduleType`" | D2 (consumer ignores the field) | Easy: thread `moduleType` through the chunk record from the parser. |

Query readiness by severity: 3 queries blocked by Critical findings; 5 blocked by High; 4 blocked by Medium. Two of the 12 are Easy unblocks; four are Moderate; four are Hard; two depend on cross-concern work.

The only query in the list that works today ("prerequisite paths") is the one the typed-edge inference work was specifically designed to enable, which is consistent with the findings: typed-edge provenance is the most mature piece of the ontology.

A second readable pattern: "Easy" queries cluster on parsing gaps that already have a defined emit surface (B3, D2, Claim 4's exclusion log). The emit side has done the work; a consumer just has to connect. "Hard" queries cluster on provenance gaps (C1/C2), on cross-course identity (A1), and on stable entity typing (B2). These involve schema changes, retrofitting, or new first-class entities — which is why § 6 groups them under "Critical direction."

## § 6 Recommended direction

This section is prescriptive. It points at directions for a future v0.3 roadmap. It is not a plan, does not sequence work, and does not identify owners. Each direction names the concern it addresses and the surfaces it would touch.

1. **Canonical ID policy (Critical direction).** One field name per entity, one normalization rule per entity, case-preserved end-to-end.

   For LearningObjective, the choice between `objectiveId` (schema) and `id` (runtime) should be made deliberately and propagated — the chosen form must appear in the schema, in the JSON-LD emit, in the HTML data-attribute, and in the parser. The secondary normalization (lowercase + week-prefix strip) should either be documented as part of the identity contract or removed. Concept and Chunk should follow the same discipline.

   Surfaces affected: `schemas/academic/learning_objectives.schema.json`, `Courseforge/scripts/generate_course.py` (`_build_objectives_metadata`, `_render_objectives`), `Trainforge/process_course.py` (`_extract_objective_refs`), `Trainforge/parsers/html_content_parser.py` (`_extract_objectives`).

2. **Stable, content-addressed identifiers (Critical direction).** Chunk IDs should derive from a content hash (text or text + section path), not from sequential position, so re-chunking does not invalidate every downstream reference.

   Concept IDs should incorporate a course namespace or be derived from a content-derived identifier so cross-course emission cannot silently merge distinct concepts. Misconception IDs should not embed position-dependent chunk IDs.

   Surfaces affected: `Trainforge/process_course.py` (chunk-ID construction at `:1003, :1027`), `Trainforge/generators/preference_factory.py` (`_misconception_id`), emit sites for concept nodes in Courseforge and Trainforge.

3. **Expanded edge ontology (High-value, moderate-cost direction).** Adding the pedagogical edges the review found missing — `assesses`, `exemplifies`, `misconception-of`, `derived-from-objective`, `defined-by` — would unlock the Chunk ↔ LO, Question ↔ LO, and Misconception ↔ Concept queries that are currently impossible or require scans.

   Each new type should get its own rule module under `Trainforge/rag/inference_rules/`, a corresponding enum entry at `schemas/knowledge/concept_graph_semantic.schema.json:46`, and a precedence rank in `Trainforge/rag/typed_edge_inference.py:48–52`. The existing typed-edge pattern is the template.

4. **A "publish-to-KG" validation profile (Critical direction).** Raw ingestion must stay permissive — exploratory emit depends on being able to add fields — but a gated publish profile should require a stricter contract.

   The profile would require: no orphaned edges (every edge's `source` and `target` exist in `nodes[]`); every chunk carries `run_id` and `created_at`; every concept node carries `course_id` or an equivalent namespace and a non-empty `occurrences[]`; every edge's evidence matches its rule's documented per-rule sub-schema; and every ID field in the graph is case-consistent with its canonical emit form.

   This would be a new validator alongside the existing WCAG, OSCQR, and assessment-quality gates — a sibling gate that specifically gates KG publish rather than pipeline completion.

5. **Coverage KPIs (Moderate direction).** The `quality_report.json` and the synthesis dataset-config stats should expose ratios that surface the silent failures this review found.

   Candidate KPIs: percent chunks with ≥1 LO ref; percent edges whose evidence matches the expected per-rule shape; percent concepts linked to ≥1 LO and ≥1 chunk; percent misconceptions linked to a stable concept; question → chunk referential integrity rate.

   These numbers would let a future reviewer detect regression without re-running the audit by hand, and would let a CI gate catch a regression in a future PR before merge.

## § 7 Out of scope

Explicit non-goals for this review. Listed to keep the document's lane narrow.

- No implementation. This document is descriptive, not a refactor plan, not a patch series, and not a code diff.
- No target-ontology proposal. Earlier speculative framings from planning conversations (for example a "5-class root ontology" or a specific IRI scheme) are not repeated here; any such proposal belongs in a separate design document.
- No RDF, OWL, or SHACL authoring. The findings apply regardless of graph representation.
- No KG-technology pick. The gaps surface on Neo4j, on RDF triple stores, on property graphs, and on labeled-property graph overlays — structural problems are technology-independent.
- No enrichment-rate analysis. That is `docs/validation/enrichment-trace-report.md`'s scope.
- No typed-edge precision/recall tuning. That is tracked as `FOLLOWUP-WORKER-N-1`.
- No content-quality, OSCQR, WCAG, or pedagogical review. Those are separate validation surfaces with their own reports.
- No review of `schemas/ONTOLOGY.md` as a document. The map itself is descriptive and adequate; this review is critical of what the map describes, not of the map's organization.
- No remediation-order sequencing or dependency analysis. A future planning session takes this review as input and produces that sequencing.

## § 8 Appendix — file:line index

Citation index for every source referenced in § 0–§ 5. Grouped by role (schemas, code) and then by file.

### Schemas cited

`schemas/academic/learning_objectives.schema.json`
- `:193` — `objectiveId` property definition
- `:202–206` — `bloomLevel` enum (six Bloom values)
- `:219–225` — `assessmentSuggestions` enum (9 values)
- `:238` — `required: ["objectiveId", "statement", "bloomLevel"]`

`schemas/knowledge/concept_graph_semantic.schema.json`
- `:8, :27, :40, :59` — `additionalProperties: true` at four levels
- `:14–15` — document-level `generated_at`
- `:22–33` — nodes array schema
- `:26` — node `required: ["id"]`
- `:30–31` — optional node properties `label`, `frequency`
- `:39–46` — edge schema including `required` and `type` enum
- `:46` — edge type enum `["prerequisite", "is-a", "related-to"]`
- `:48–51` — confidence bounds `[0, 1]`
- `:56–64` — provenance structure

`schemas/knowledge/instruction_pair.schema.json`
- `:36–41` — `lo_refs` required with `minItems: 1`
- `:42–44` — `bloom_level` enum
- `:46–50` — `content_type` free-string

`schemas/knowledge/preference_pair.schema.json`
- `:44–48` — `lo_refs` required with `minItems: 1`

`schemas/events/decision_event.schema.json`
- `:7` — top-level required list
- `:9–18` — `event_id`, `seq`
- `:19–22` — `run_id`
- `:63–102` — `decision_type` enum (40 values)

`schemas/events/trainforge_decision.schema.json`
- `:23–26` — `bloom_target` enum
- `:62–65` — `question_type` enum (8 values)
- `:91–94` — `difficulty` enum (3 values)

### Code cited

`Courseforge/scripts/generate_course.py`
- `:169–172` — `_slugify`
- `:314` — `data-cf-objective-id` attribute emit
- `:378, :491` — `data-cf-objective-ref` attribute emit on self-checks and activities
- `:388–405` — `_infer_content_type` producing 8 inferred labels
- `:423` — `data-cf-content-type` attribute emit
- `:425` — `data-cf-key-terms` attribute (slugs only)
- `:523` — JSON-LD `id` emit (not `objectiveId`)
- `:531–541` — `assessmentSuggestions` ad-hoc Bloom → question-type map
- `:553–556` — JSON-LD `contentType` emit
- `:560–563` — JSON-LD `keyTerms` with definitions
- `:571–584` — `_build_page_metadata` including `moduleType`
- `:591` — JSON-LD `misconceptions` emit

`Trainforge/parsers/html_content_parser.py`
- `:319–331` — JSON-LD objectives strategy
- `:323` — `lo.get("id")` read (runtime field name)
- `:334–358` — `data-cf-*` objectives strategy
- `:341` — `data-cf-objective-id` parse

`Trainforge/process_course.py`
- `:118–125` — `BLOOM_TO_DIFFICULTY`
- `:273–286` — `normalize_tag`
- `:1003, :1027` — position-based chunk-ID construction
- `:1080–1092` — chunk record dict
- `:1088` — `_extract_objective_refs(item)` call
- `:1495+` — full `_extract_objective_refs` definition
- `:1501–1507` — LO ref normalization (`lower().strip()` + week-prefix strip)

`Trainforge/rag/typed_edge_inference.py`
- `:48–52` — edge-type precedence map

`Trainforge/rag/inference_rules/prerequisite_from_lo_order.py`
- `:135–148` — evidence dict shape

`Trainforge/rag/inference_rules/is_a_from_key_terms.py`
- `:52–64` — rule-local `_slugify`
- `:159–168` — evidence dict shape including `chunk_id`

`Trainforge/rag/inference_rules/related_from_cooccurrence.py`
- `:65–80` — evidence dict shape

`Trainforge/generators/question_factory.py`
- `:81–89` — `VALID_TYPES` (7 values)

`Trainforge/generators/preference_factory.py`
- `:140–143` — `_misconception_id` construction

`Trainforge/synthesize_training.py`
- `:119–121` — `_eligible(chunk)`
- `:226, :261, :287` — `decision_type` values not in schema enum
- `:243–246` — silent `chunks_skipped_no_lo` increment
