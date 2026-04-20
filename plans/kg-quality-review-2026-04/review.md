# KG-Quality Review — Unified Ontology & Cross-System Contracts (2026-04)

Synthesis of five parallel discovery workers (A–E) + the prior Ontology Review v0.2.0 + Ultraplan session findings, framed around a single north-star question: **which changes to our ontology and contracts would most measurably improve the quality of our final knowledge graph?** Every finding below carries a KG-impact statement (what join, dedupe, or query is affected). Every recommendation in § 3 carries an "unlocks" / "reduces" statement tying it back to a concrete KG capability.

- **Total unified findings:** 37 (across 8 themes).
- **Total recommendations:** 23 (P0: 5, P1: 7, P2: 7, P3: 4).
- **Sources merged:** Worker A [A], Worker B [B], Worker C [C], Worker D [D], Worker E [E], prior review `docs/validation/ontology-review-v0.2.0.md` [PR], Ultraplan session notes [UP].

---

## § 1 Executive summary

**Verdict — can we publish a high-quality KG today? No, but the gap is bounded and most of it is schematization work rather than re-architecture.** The emit side has done the hard work (Courseforge emits 14 JSON-LD fields + 14 `data-cf-*` attributes per page; Trainforge produces `chunks.jsonl` with a stable `v4` dict shape; LibV2 has the manifest schema and a working validator); the breakage is concentrated in **unwritten contracts**, **duplicated vocabularies**, **unreachable taxonomies**, and **unstable identifiers**. The KG backbone is assembled from code-only agreements that three independent subsystems happen to share today, and every "happen-to-share" is one rename away from silent drift.

Three structural blockers remain from the prior review and are unchanged by worker discovery: (1) LO field-name drift between schema `objectiveId` and runtime `id` [PR.Claim1]; (2) cross-course concept collisions because concept nodes require only `id` [PR.A1]; (3) position-based chunk IDs that invalidate every edge evidence pointer on re-chunk [PR.A2]. Worker E adds a fourth structural blocker: **`chunks.jsonl` — the KG's actual node layer — has no JSON Schema at all** [E.7]. Worker D adds a fifth: **the authoritative subject taxonomy and pedagogy framework are unreachable from Courseforge/Trainforge**, so every KG node enters with a CLI-fabricated or free-text classification [D].

### What's working — the green lights

- **Typed-edge provenance pattern is sound.** Every concept-graph edge carries `{rule, rule_version, evidence}` with bounded `[0,1]` confidence (`schemas/knowledge/concept_graph_semantic.schema.json:48–64`). Three rule modules each emit exactly their own type [PR.Claim2].
- **Bloom level as a six-value enum** is consistent across three schemas (`learning_objectives.schema.json:202–206`, `instruction_pair.schema.json:42–44`, `trainforge_decision.schema.json:23–26`) — the *enum* is consistent even when the *verb lists* are not [PR]. The enum consistency is what lets the 11 copies still interoperate on the happy path.
- **Decision-event ledger has `event_id` + per-run `seq`** (`schemas/events/decision_event.schema.json:9–18`), the strongest provenance foundation in the repo [PR].
- **Courseforge → Trainforge metadata bridge works end-to-end** on the JSON-LD > `data-cf-*` > regex priority chain [C, PR]. The exception is `data-cf-objective-ref`, which is emitted but never parsed [PR.B3, C].
- **`CourseManifest` + LibV2 import is the only end-to-end schema-validated pipe** in the repo (`schemas/library/course_manifest.schema.json` + `LibV2/tools/libv2/validator.py:147–196`) [E.6]. It is the functional reference for what "contract-hardened" looks like here.
- **Courseforge now emits `data-cf-role="template-chrome"`** (added in Worker Q), letting Trainforge skip page chrome during extraction [C, `generate_course.py:289,290,297`].

### Top 5 leverage points (ranked by KG-impact per unit of effort)

1. **Publish `schemas/knowledge/courseforge_jsonld_v1.schema.json`** with enum constraints on `moduleType`, `cognitiveDomain`, `contentType`, `bloomLevel`, `AssessmentType`, and normalized `bloomRange` as `array<BloomLevel>`. Single artifact; unlocks enum-safe KG joins for every page-level emit; closes the four shape gaps in Worker C's table (bloomRange str|array, misconceptions shape, moduleType enum, cognitiveDomain enum). → **REC-JSL-01** (P0).
2. **Schematize `chunks.jsonl`** (`schemas/knowledge/chunk_v4.schema.json`) + a validation hook before write. This is the KG's node layer today and has no schema. Every chunk-level KG edge (chunk→LO, chunk→concept, chunk→misconception) depends on its un-schematized shape. → **REC-CTR-01** (P0).
3. **Consolidate the 11 BLOOM_VERBS copies** into `schemas/taxonomies/bloom_verbs.json` loaded by a thin `lib/ontology/bloom.py`. Richest copy (`bloom_taxonomy_mapper.py:55`, 10 verbs/level + templates) becomes the schema source; every producer/consumer/validator loads from one place. Eliminates a whole class of silent Bloom-level flips (D1 in Worker A). → **REC-BL-01** (P0).
4. **Propagate LibV2 subject taxonomy onto every page JSON-LD + chunk record.** Today Courseforge emits no `division/primary_domain/subdomain/topics` anywhere; Trainforge fabricates them from CLI flags [D]. Fix unlocks cross-course dedupe by taxonomy node and fail-closed validation at Courseforge emit time (before IMSCC packaging). → **REC-TAX-01** (P0).
5. **Reconcile the decision_type enum** (39 schema values vs. 3 in-code allowlist vs. real emit sites outside both) and flip `_validate_record` from warn-only to fail-closed [E.8, PR.C3]. Decision events are the KG's provenance ledger — warn-only on its shape undermines every "why does the KG assert X?" audit. → **REC-CTR-04** (P0).

### Scale of the catalog

- **37 unified findings** organized by 8 themes: BL (Bloom), VOC (controlled vocabulary), JSL (JSON-LD contract), TAX (taxonomy propagation), CTR (cross-system contracts), ID (identity / join stability), PRV (provenance / run tracking), LNK (missing links).
- **23 recommendations** mapping to those findings; every one carries a file target, change shape, KG-impact statement, effort estimate, and dependencies.
- **Source coverage:** All 5 Codex claims from the prior review are carried forward (3 as-is, 2 absorbed into recs). All 14 prior-review findings (A1–A6, B1–B4, C1–C3, D1–D4) map into the catalog — some merged with worker findings, none lost. All 11 BLOOM_VERBS sites from Worker A appear in BL-01. The 7 Ultraplan recommendations are present in § 3.

### Contradictions surfaced and resolved

- **Decision-type enum count:** master plan / Ultraplan says 40 values; Worker E's re-count of `schemas/events/decision_event.schema.json:63–103` yields **39**; prior review cites 40. Worker E's re-count is the authoritative current value. The gap between 39 (schema) and 3 (in-code allowlist `lib/decision_capture.py:87–93`) + real emit sites is the contract gap; the 40/39 discrepancy is stale count, not drift.
- **Claim 5 question-type drift direction:** Codex originally said schema had `multiple_response` and factory had `ordering/hotspot`; prior review verified the direction is **reversed** (factory has `multiple_response`, schema has `ordering/hotspot`) [PR.Claim5]. Worker B re-verifies the reversed direction; catalog entry VOC-04 uses Worker B's direction.
- **moduleType enum size:** `schemas/ONTOLOGY.md:108,625` documents 5 values (`overview/content/application/self-check/summary`); Worker B found Courseforge actually emits **6** values — the sixth is `discussion` at `generate_course.py:754–759`, and `self-check` actually emits as `"assessment"` (not `"self-check"`) at `:704`. The ontology map is stale; catalog entry VOC-05 uses the 6-value vocabulary.

---

## § 2 Unified findings catalog

Each finding carries a unified ID, severity (**Critical** / High / Medium / Low), evidence (file:line or prior-review reference), one-sentence KG-impact, and source attribution. Themes in order: BL (Bloom), VOC (vocabularies), JSL (JSON-LD), TAX (taxonomy), CTR (cross-system contracts), ID (identity), PRV (provenance), LNK (missing links).

### Theme BL — Bloom taxonomy duplication and drift

Worker A enumerated **11 in-code copies** of Bloom-verb tables. Catalog entries below collapse them into structural findings rather than one-per-copy.

#### BL-01 — 11 in-code Bloom-verb copies, no canonical schema source
- **Severity:** Critical
- **Evidence:** `lib/validators/bloom.py:21`; `Trainforge/parsers/html_content_parser.py:156`; `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55`; `Courseforge/scripts/generate_course.py:136`; `LibV2/tools/libv2/query_decomposer.py:111`; `Courseforge/agents/content-quality-remediation.md:159`; `Trainforge/generators/assessment_generator.py:46`; `Trainforge/generators/question_factory.py:91`; `lib/semantic_structure_extractor/analysis/content_profiler.py:182`; `lib/semantic_structure_extractor/semantic_structure_extractor.py:126`; `Trainforge/rag/libv2_bridge.py:473`.
- **KG-impact:** An identical LO traversing DART→Courseforge→Trainforge→LibV2 can be classified at different Bloom levels by different producers, because no two copies agree on the verb set (11 vs 7 vs 10 vs 7 vs 11 at `remember`; same shape for higher levels). Bloom-level KG edges are therefore non-deterministic.
- **Sources:** [A], [UP]

#### BL-02 — Validator vs emitter verb-set divergence forces spurious `BLOOM_MISMATCH` warnings
- **Severity:** High
- **Evidence:** `lib/validators/bloom.py:138` (`BloomAlignmentValidator.validate`) detects Bloom from its 11/9/9/9/9/9 verb set and emits `BLOOM_MISMATCH` when its detection disagrees with the emitter's `data-cf-bloom-level`. Emitter uses Copy 4 (`generate_course.py:136`, 7/6/6/6/6/6 verbs) — which disagrees about `describe` (validator: `remember`; emitter: `understand`) and the cross-level ambiguity of `compare`, `distinguish`, `illustrate` [A].
- **KG-impact:** Clean input produces validation-failure KG edges — inflating `validation_failed` counts in the decision ledger without any real drift, which poisons any "low-quality run" KG query.
- **Sources:** [A], [PR.Theme-A+D1]

#### BL-03 — Query-side Bloom verb list diverges from chunk-side verb list
- **Severity:** High
- **Evidence:** `LibV2/tools/libv2/query_decomposer.py:111` (11/10/10/10/11/11 verbs at retrieval time); `Trainforge/parsers/html_content_parser.py:156` (7/6/6/6/6/6 at chunk-ingest time).
- **KG-impact:** A query `synthesize X` is routed to `create`-level chunks by the decomposer, but chunks tagged by the parser lack `synthesize` in their create list — retrieval and index are asymmetric, producing recall gaps on higher-order queries.
- **Sources:** [A]

#### BL-04 — Difficulty score and Bloom level derive from disjoint verb lists
- **Severity:** Medium
- **Evidence:** `lib/semantic_structure_extractor/analysis/content_profiler.py:182` (`BLOOM_PATTERNS`) feeds `BLOOM_DIFFICULTY_WEIGHTS:192`; `Courseforge/scripts/generate_course.py:136` emits `bloom_level`. Verbs match inconsistently across the two lists.
- **KG-impact:** Same chunk can carry `bloom_level=remember` and `difficulty=0.85` simultaneously — Bloom-level and difficulty KG axes can point in opposite directions on a single node.
- **Sources:** [A]

#### BL-05 — Agent-prompt Bloom palette narrows remediated LO vocabulary
- **Severity:** Low
- **Evidence:** `Courseforge/agents/content-quality-remediation.md:159` — inline prompt dict with 4 verbs/level.
- **KG-impact:** Remediation-agent-authored LOs use a narrower verb palette than any code-level detector; downstream classification falls through to heuristics. No audit surface catches the drift.
- **Sources:** [A]

#### BL-06 — `BLOOM_TO_DOMAIN` coupling cascades Bloom drift into cognitive-domain drift
- **Severity:** High
- **Evidence:** `Courseforge/scripts/generate_course.py:146` — `BLOOM_TO_DOMAIN` maps Bloom → `factual|conceptual|procedural|metacognitive`. Any BL-01 / BL-02 / BL-03 flip re-derives the domain edge [B.§1].
- **KG-impact:** A single mis-coded Bloom level corrupts *two* KG edges: `(LO, bloom_level, …)` and `(LO, cognitive_domain, …)`. Double-drift on one source error.
- **Sources:** [A], [B]

#### BL-07 — `BLOOM_TO_DIFFICULTY` mapping lives in code only, with a second difficulty vocabulary in the schema
- **Severity:** Low
- **Evidence:** `Trainforge/process_course.py:118–125` maps Bloom → `foundational/intermediate/advanced`; `schemas/events/trainforge_decision.schema.json:91–94` enum is `easy/medium/hard`. Two difficulty vocabularies, no bridge [PR.D3].
- **KG-impact:** Difficulty semantics are reconstructable only by reading Python; cross-surface difficulty queries need translation tables; one emit path uses `mixed` that never materializes.
- **Sources:** [PR.D3]

### Theme VOC — Controlled-vocabulary drift

#### VOC-01 — `content_type` is free-string across four surfaces; the most fragmented vocabulary in the repo
- **Severity:** **Critical**
- **Evidence:** `Courseforge/scripts/generate_course.py:388–405` `_infer_content_type` emits 8 values; `:448,452` callout emits 2 more (`application-note|note`); `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:141–173` `CONTENT_TYPE_DEFAULTS` has 20 disjoint values; `schemas/knowledge/instruction_pair.schema.json:46–50` requires `content_type` as free-string; `schemas/academic/course_metadata.schema.json:348–355` has an unrelated 8-value enum (lecture/reading/video/…). Consumers: `Trainforge/parsers/html_content_parser.py:26,292`; `Trainforge/process_course.py:1099–1336` (chunk `content_type_label`); `Trainforge/generators/instruction_factory.py:48,183–201` (template catalog keys `(bloom, content_type)`); `LibV2/tools/libv2/retriever.py:62,439–440` (`ChunkFilter.content_type_label`).
- **KG-impact:** Every Chunk→ContentType KG edge is unconstrained; LibV2's `content_type_label` retrieval filter does string-equality across a vocabulary of ≥13 values produced by disjoint mappers — instruction-factory template lookups silently fall through to `_default`, fragmenting content-type analytics and mis-partitioning retrieval.
- **Sources:** [B.§2], [C], [PR.D1]

#### VOC-02 — `cognitive_domain` is 4 values, emitted on every LO, but never schematized
- **Severity:** High
- **Evidence:** `Courseforge/scripts/generate_course.py:146` sources `BLOOM_TO_DOMAIN`; `:313,520,527` emit `data-cf-cognitive-domain` + JSON-LD `cognitiveDomain`; `Trainforge/parsers/html_content_parser.py:37,327,357` consume. No `schemas/**/*.schema.json` enumerates `{factual, conceptual, procedural, metacognitive}`. Default fallback `"conceptual"` silently absorbs unmapped Bloom levels.
- **KG-impact:** Objective→Domain edge is single-sourced from one dict; any emitter producing `"procedure"` (singular) or `"procedural-knowledge"` breaks the join silently with no schema-level guard.
- **Sources:** [B.§1], [C]

#### VOC-03 — `teaching_role` defined only in Trainforge; Courseforge emits deterministic signal that is discarded
- **Severity:** High
- **Evidence:** `Trainforge/align_chunks.py:33` `VALID_ROLES = {introduce, elaborate, reinforce, assess, transfer, synthesize}` (6 values, no schema). Producers: LLM/mock classifier at `align_chunks.py:487–582`. Consumers: `LibV2/tools/libv2/retriever.py:61,435–436` (`ChunkFilter.teaching_role`). Courseforge emits `data-cf-component ∈ {flip-card, self-check, activity}` + `data-cf-purpose ∈ {term-definition, formative-assessment, practice}` at `generate_course.py:345,374,487` — deterministic candidates that map cleanly onto `VALID_ROLES` but are ignored.
- **KG-impact:** `teaching_role` is LibV2's primary retrieval filter (ADR-002 filter-first), but the Chunk→TeachingRole KG edge is LLM-classified non-deterministically each run. Same course's KG differs across runs — dedupe on teaching_role is impossible, and Courseforge's deterministic authorial signal is thrown away.
- **Sources:** [B.§3], [UP]

#### VOC-04 — Four divergent `assessment_type` / `question_type` enums
- **Severity:** High
- **Evidence:** `schemas/academic/learning_objectives.schema.json:219–225` — 9-value `assessmentSuggestions` enum (assessment methods: `exam, quiz, assignment, project, discussion, presentation, portfolio, demonstration, case_study`). `schemas/academic/course_metadata.schema.json:365–368` — 5-value `assessments[].format` enum. `Trainforge/generators/question_factory.py:81–89` `VALID_TYPES` — 7-value list including `multiple_response`. `schemas/events/trainforge_decision.schema.json:62–65` — 8-value `question_type` enum including `ordering` and `hotspot`. `Courseforge/scripts/generate_course.py:534–541,593` JSON-LD `suggestedAssessmentTypes` picks from an implicit 5-value set `{multiple_choice, true_false, fill_in_blank, short_answer, essay}`.
- **KG-impact:** A `Question` node typed `multiple_response` by the factory fails validation when logged as a `trainforge_decision` event (enum rejects it). CF JSON-LD puts *question-format* values (`multiple_choice`) into an LO-schema field that expects *assessment-method* values (`exam`, `portfolio`) — category error the schema doesn't catch. Question→Type KG edges have two incompatible authorities; Objective→AssessmentMethod edges carry semantically wrong values.
- **Sources:** [B.§4], [B.§6], [C], [PR.Claim5], [PR.B4]

#### VOC-05 — `moduleType` actually emits 6 values; ontology map is stale at 5
- **Severity:** Medium
- **Evidence:** `Courseforge/scripts/generate_course.py:647–759` emits `{overview, content, application, assessment, summary, discussion}` (6 values, with self-check mapped to the string `"assessment"` not `"self-check"`). `schemas/ONTOLOGY.md:108,625` documents 5. No `.schema.json` enumerates. Consumer side: `Trainforge/tests/test_metadata_extraction.py:34,150` is the *only* runtime read — no production parser joins on it.
- **KG-impact:** Page→ModuleType partitioning is absent from the KG; queries "list all assessment modules for course X" have no graph-level answer; `discussion` is an emitted-but-undocumented sixth value; ontology documentation is already stale.
- **Sources:** [B.§5], [C], [PR.D2]

#### VOC-06 — `BLOOM_CHUNK_STRATEGY` pairs two free-string vocabularies with zero schema presence
- **Severity:** Medium
- **Evidence:** `Trainforge/rag/libv2_bridge.py:310–317` — 6-key dict pairing Bloom → chunk-type tuple (`(explanation|example|None, …)`). Consumed at `:341`. Not in `schemas/ONTOLOGY.md` despite the map's exhaustive tables.
- **KG-impact:** Retrieval-to-objective KG edges for `apply`/`analyze` depend on an un-versioned code-only dict pairing free-string values from VOC-01's uncontrolled content-type vocabulary. No provenance for retrieval behavior.
- **Sources:** [B.§7]

#### VOC-07 — `decision_type` enum three-way split: 39 in schema / 3 in in-code allowlist / emit sites outside both
- **Severity:** High
- **Evidence:** `schemas/events/decision_event.schema.json:63–103` — 39 enum values (Worker E re-count; prior plan cited 40). `lib/decision_capture.py:87–93` — 3-value `ALLOWED_DECISION_TYPES` tuple (`instruction_pair_synthesis, preference_pair_generation, typed_edge_inference`). Real emit sites `Trainforge/synthesize_training.py:226,261,287,312` use the 3 allowlisted values not in schema. `_validate_record` at `decision_capture.py:401–414` **warns** on failure and stores issues on the record but does NOT refuse the write.
- **KG-impact:** Decision-event provenance KG edges split into two populations (schema-validated vs in-code-allowlisted); "count all question_generation decisions" queries silently miss Trainforge's `instruction_pair_synthesis` events; warn-only means the KG's provenance audit layer accepts drift.
- **Sources:** [B.§8], [E.8], [PR.C3]

#### VOC-08 — Chunk `chunk_type` vs `content_type_label` semantic drift
- **Severity:** Medium
- **Evidence:** `Trainforge/process_course.py:874,1082` sets `chunk_type`; `:1162` sets `content_type_label`; both free-string, overlapping-but-different semantics. BLOOM_CHUNK_STRATEGY filters on one while the schema's instruction_pair field uses the other [B.§2,§7].
- **KG-impact:** A chunk can carry `chunk_type=example` and `content_type_label=explanation` — the KG has two content-type axes, neither controlled, and retrieval filters operate on different ones.
- **Sources:** [B.§2], [B.§7]

### Theme JSL — Courseforge JSON-LD contract

#### JSL-01 — CF JSON-LD namespace `https://ed4all.dev/ns/courseforge/v1` has no published schema
- **Severity:** **Critical**
- **Evidence:** `Courseforge/scripts/generate_course.py:580` hard-codes the context URL; no `schemas/knowledge/courseforge_jsonld_v1.schema.json` file exists. Peer schemas at `schemas/knowledge/` (`concept_graph_semantic`, `instruction_pair`, `preference_pair`) show the expected landing pattern.
- **KG-impact:** Every inbound KG assertion from the CF→TF bridge rests on an unschematized contract. A producer-side rename of any of the 14 JSON-LD fields fails the downstream parser silently (fallback to `data-cf-*` or regex). Publishing the schema unlocks enum-level guarantees for `moduleType`, `cognitiveDomain`, `contentType`, `bloomLevel`, `AssessmentType`, and `bloomRange`.
- **Sources:** [C], [UP]

#### JSL-02 — `prerequisitePages` consumed but never emitted
- **Severity:** **Critical**
- **Evidence:** Parser reads at `Trainforge/parsers/html_content_parser.py:219` (`json_ld.get("prerequisitePages", [])`); zero emit sites in `Courseforge/scripts/generate_course.py` (grep verified).
- **KG-impact:** Prerequisite-page KG edges are permanently empty. Inter-page dependency graph never materializes, so "which pages teach prerequisites for X" returns nothing. Parser infrastructure is in place; the gap is pure emit-side silence.
- **Sources:** [C]

#### JSL-03 — `prerequisiteObjectives` emitted but never consumed
- **Severity:** High
- **Evidence:** Emit at `Courseforge/scripts/generate_course.py:542–544`; zero consumers under `Trainforge/` (grep verified).
- **KG-impact:** LO-level prerequisite edges (`CO-02 → CO-05`) never propagate to chunks or concept graph. Bloom-progression inference has no edge data; objective-dependency graph is emit-only.
- **Sources:** [C]

#### JSL-04 — `sections[].bloomRange` emitted as `str | array<str>`; consumer reduces to single value
- **Severity:** High
- **Evidence:** Emit at `generate_course.py:564–566` (dual type). Consumer at `Trainforge/process_course.py:1302–1304` takes `[0]` when list, collapsing `["remember","apply"]` to `"remember"`.
- **KG-impact:** Bloom-span KG edges lose their upper bound; "find all chunks spanning apply or higher" queries return incomplete sets. Normalize to `array<BloomLevel>` (min 1, max 2) in schema to fix.
- **Sources:** [C]

#### JSL-05 — `misconceptions[]` shape is producer-opaque
- **Severity:** High
- **Evidence:** Emit passes through `week_data["misconceptions"]` without shape constraint (`generate_course.py:591–592, :633, :671`). Consumer at `process_course.py:1177–1183` expects `{misconception: str, correction: str}` per item; `html_content_parser.py:54` types it `List[Dict[str, str]]`.
- **KG-impact:** Misconception nodes silently drop `correction` edges when a producer ships a string instead of a dict, or uses alternate keys. Distractor-targeting joins degrade without error. Compound with PR.B2 (misconception IDs are unstable) — this contract is underspecified at both shape and identity level.
- **Sources:** [C], [PR.B2]

#### JSL-06 — `data-cf-objective-ref` emitted but never parsed
- **Severity:** High
- **Evidence:** Emit at `generate_course.py:378, :491` on `.self-check` and `.activity-card`. No consumer under `Trainforge/` (only test file `Trainforge/tests/test_metadata_extraction.py`).
- **KG-impact:** Activity→LO binding is visible in HTML for humans but discarded at KG-ingest time. "Which activities exercise LO Y?" requires HTML re-scan. Parser fix is small; this is an "Easy" unblock per the prior review's impact matrix.
- **Sources:** [C], [PR.B3]

#### JSL-07 — `data-cf-component` / `data-cf-purpose` ignored; parser uses class-regex instead
- **Severity:** Medium
- **Evidence:** Emit at `generate_course.py:345,374,487` of authoritative enum values. Parser uses class-regex (`COMPONENT_PATTERNS` at `html_content_parser.py:166–173`) rather than the attribute.
- **KG-impact:** Component-type KG classification depends on class-name stability; any template swap breaks detection silently. Authoritative `data-cf-component` enum is ignored. Provenance inversion with VOC-03 (teaching role): the deterministic signal is present but unused.
- **Sources:** [C]

#### JSL-08 — `data-cf-term` slug normalization drifts from emitter's `_slugify`
- **Severity:** Medium
- **Evidence:** Emit at `generate_course.py:346` uses `_slugify` (`:169–172`). Parser extracts flip-card terms via `<strong>`/`<dt>` regex (`html_content_parser.py:419–425`), never from `data-cf-term`. Normalization path at `Trainforge/process_course.py:273–286` truncates to 4 tokens.
- **KG-impact:** Same term produces two distinct concept-graph nodes because emit-side slug and consume-side normalization disagree (cf. PR.A4). Compound with VOC-01 — concept identity drift has both enum and normalization dimensions.
- **Sources:** [C], [PR.A4]

#### JSL-09 — `learningObjectives[].bloomLevel=null` outranks downstream verb inference
- **Severity:** Medium
- **Evidence:** Emit at `generate_course.py:519, 525` produces `None` when `detect_bloom_level` fails. Consumer at `process_course.py:1123` trusts it verbatim — null bloom level becomes authoritative.
- **KG-impact:** Chunks receive `bloom_level_source="page_jsonld"` with no level; Bloom-coverage metrics silently exclude whole modules. Fix by requiring `bloomLevel` to be non-null in the schema + tightening the emit path.
- **Sources:** [C]

#### JSL-10 — `contentType` enum is split between section-level (8 values) and callout-level (2 values) under the same attribute
- **Severity:** Medium
- **Evidence:** Section emit `:388–405` produces `{definition, example, procedure, comparison, exercise, overview, summary, explanation}`. Callout emit `:448, :452` produces `{application-note, note}`. Both via `data-cf-content-type` on different element classes.
- **KG-impact:** A single-enum schema fails both; KG queries over content-type need to know which element class they're filtering on. Schema should discriminate by element class (e.g. separate `SectionContentType` and `CalloutContentType` types).
- **Sources:** [C], [PR.D1]

#### JSL-11 — `data-cf-objectives-count` emit-only, advisory
- **Severity:** Low
- **Evidence:** Emit at `:327`; no consumer.
- **KG-impact:** Advisory only. Zero KG impact, but adds contract surface that drifts silently.
- **Sources:** [C]

### Theme TAX — Taxonomy propagation (LibV2 orphaning)

#### TAX-01 — Courseforge emits zero subject-taxonomy classification
- **Severity:** **Critical**
- **Evidence:** `generate_course.py::_build_page_metadata:571` emits `courseCode, weekNumber, moduleType, learningObjectives, sections, misconceptions, suggestedAssessmentTypes` but no `division/primary_domain/subdomain/topics`. No `data-cf-*` carries taxonomy coordinates. `schemas/taxonomies/taxonomy.json` has 2 production consumers (both LibV2): `LibV2/tools/libv2/concept_vocabulary.py:112, 383, 411`; `LibV2/tools/libv2/validator.py:197`. No references from Courseforge, Trainforge, or `lib/`.
- **KG-impact:** Cross-course dedupe by taxonomy node is impossible — two courses on "kinematics" have no shared `primary_domain=physics, subdomain=mechanics` node to merge on. Federated queries like "all STEM/physics/mechanics chunks across courses" only work after a human supplies `--division/--primary-domain` at Trainforge ingestion.
- **Sources:** [D]

#### TAX-02 — Trainforge fabricates `classification` from CLI flags, not upstream signal
- **Severity:** **Critical**
- **Evidence:** `Trainforge/process_course.py:2731–2735, 1757–1764` derive `classification` from `--division/--domain/--subdomain` argparse args; default `--division STEM` applies even to ARTS content.
- **KG-impact:** Automation surfaces (`create_textbook_pipeline_tool`, orchestrator workflows) must pass correct flags manually. Silent misclassification when defaults kick in. KG acquires wrong edges with no audit trail. `validate_taxonomy_compliance` runs at LibV2 import — by then IMSCC + chunks already exist with whatever classification the CLI supplied.
- **Sources:** [D]

#### TAX-03 — `pedagogy_framework.yaml` has zero runtime consumers
- **Severity:** High
- **Evidence:** `schemas/taxonomies/pedagogy_framework.yaml` is loaded by no code in the repo. Referenced only in prose (`schemas/README.md:41,48`, `schemas/ONTOLOGY.md:704,1117`, `LibV2/README.md:38`). `retrieval_scoring.py:240 load_pedagogy_model()` loads a per-course `pedagogy/pedagogy_model.json` — a different artifact sharing a name root.
- **KG-impact:** 12-tier framework (`foundational_theories, learning_sciences, instructional_design, …`) never flows into any KG node. "Which tiers of the 12-tier framework does this course cover?" and "find gaps in pedagogy coverage" are unanswerable programmatically.
- **Sources:** [D], [UP]

#### TAX-04 — `ontology_mappings.acm_ccs` and `.lcsh` are schema slots with zero producers
- **Severity:** High
- **Evidence:** `schemas/library/course_manifest.schema.json:90, 103` define the cross-walk slots. No code populates them in observed manifests.
- **KG-impact:** Cross-walk joins to external ontologies (ACM Digital Library, Library of Congress subject headings) are impossible. Federated KG scenarios across external library catalogs are blocked at the first join. The schema advertises intent the pipeline never fulfills.
- **Sources:** [D]

#### TAX-05 — `CrossCourseRAG(domain=…)` accepts unvalidated free-text
- **Severity:** High
- **Evidence:** `Trainforge/rag/libv2_bridge.py:532–572, 610`. Docstring example at `Trainforge/rag/__init__.py:23` uses `"pedagogy"` — not a canonical domain in `taxonomy.json` (closest is `educational-technology:279`). `search_catalog` does case-insensitive equality at `LibV2/tools/libv2/catalog.py:111–116` and returns `[]` on miss with no error.
- **KG-impact:** Typo or free-text domain returns empty result set without warning; downstream assessment generation silently loses cross-course grounding. Any KG query layered on CrossCourseRAG inherits this silent-failure mode.
- **Sources:** [D]

#### TAX-06 — Chunk records inconsistently carry `_domain`
- **Severity:** Medium
- **Evidence:** `LibV2/tools/libv2/retriever.py:691` reads `chunk.get("_domain", "")`; no producer contract guarantees the field is set on chunks.
- **KG-impact:** Chunk-level domain filters silently downgrade to catalog-level filters; retrieval scoring loses a dimension.
- **Sources:** [D]

#### TAX-07 — No taxonomy-validation gate runs at Courseforge emit time
- **Severity:** Medium
- **Evidence:** `validate_taxonomy_compliance` (`LibV2/tools/libv2/validator.py:197`) runs at LibV2 import only. By then the IMSCC + chunks already exist.
- **KG-impact:** Errors caught late → full-pipeline re-run required. Provenance auditability suffers (can't tell whether misclassification originated at Courseforge planning, Trainforge ingestion, or LibV2 import).
- **Sources:** [D]

#### TAX-08 — `subtopics` auto-derived from concept-graph frequency with no taxonomy grounding
- **Severity:** Medium
- **Evidence:** `Trainforge/process_course.py:1705–1721` derives `subtopics` from concept frequency, not from `taxonomy.json`.
- **KG-impact:** Subtopic KG edges are frequency artifacts not taxonomy nodes. Two courses may produce `subtopics=["bloom", "udl"]` without touching canonical topics — blocks subtopic-grained cross-course joins.
- **Sources:** [D]

### Theme CTR — Cross-system contracts (non-JSON-LD)

#### CTR-01 — `Trainforge/chunks.jsonl` is the KG node layer with NO JSON Schema
- **Severity:** **Critical**
- **Evidence:** `Trainforge/process_course.py:86` — `CHUNK_SCHEMA_VERSION = "v4"` string constant. `:1038–1216` `_create_chunk` produces ~14-key dict. `:1668` writes JSONL. `docs/schema/chunk-schema-v4.md` is prose only. `schemas/knowledge/instruction_pair.schema.json:34` references chunk ID shape but does not constrain the chunk record itself. Consumers: LibV2 retriever/indexer/concept_vocabulary + `lib/validators/leak_check.py:124–129` + `content_facts.py:231` (all duck-typed).
- **KG-impact:** Every KG node's `chunk_type, bloom_level, content_type_label, difficulty, concept_tags` is consumed as fact by retrieval, typed-edge inference, and eval. A silent rename on either side mis-types the KG's primary surface. Bumping `CHUNK_SCHEMA_VERSION` is the only signal — consumers not checking read drifted records as authoritative. This is the single highest-leverage schematization in the repo.
- **Sources:** [E.7]

#### CTR-02 — Courseforge → Brightspace-packager per-week directory layout is filename-regex convention
- **Severity:** High
- **Evidence:** Producer `generate_course.py:806–809` writes `week_NN/{overview|content_NN_*|application|self_check|summary|discussion}.html`. Consumer `package_multifile_imscc.py:82–107` globs `content_dir/week_*/` and orders by hard-coded dict `{overview:0, content:1, application:2, self_check:3, summary:4, discussion:5}` at `:94`. Unrecognized stems fall to `(99, name)`.
- **KG-impact:** Any new page type added on emit (e.g. `reflection.html`) lands into the IMSCC + KG as a nameless `item` after `discussion`, with no pedagogical role — degrading `module_sequence` KG edge inference. The packager's regex is the only derivation of module-type label that the KG sees from the IMSCC organization tree.
- **Sources:** [E.3]

#### CTR-03 — Per-week LO validation gate is opt-in (`--objectives`), not default
- **Severity:** High
- **Evidence:** `package_multifile_imscc.py:127–153` calls `validate_page_objectives.validate_page` only when `--objectives` is passed; hard-fails `SystemExit(2)` on violation, but silently skips when flag omitted.
- **KG-impact:** Default CLI invocation can ship IMSCC with `learningObjectives` JSON-LD referencing IDs outside the canonical registry. These fan out in Trainforge's `learning_outcome_refs` normalization and corrupt the `chunk → LearningObjective` join — the KG's most load-bearing edge.
- **Sources:** [E.4]

#### CTR-04 — Decision-event validation is warn-only
- **Severity:** High
- **Evidence:** `lib/decision_capture.py:401–414` — `_validate_record` calls `lib/validation.py:validate_decision` (gated by `VALIDATE_DECISIONS`); failure warns + stores `metadata.validation_issues` but does NOT refuse the write. Trainforge's 3 extension values + real emit sites outside both the schema and the in-code tuple pass through.
- **KG-impact:** Decision provenance is how we audit KG claims; warn-only means every audit trail can contain un-schematized decisions silently. No downstream consumer reads `metadata.validation_issues` to filter training exports — the warning signal is discarded.
- **Sources:** [E.8], [PR.C3]

#### CTR-05 — Workflow phase-to-phase routing is an in-code Python dict with no schema
- **Severity:** High
- **Evidence:** `MCP/core/workflow_runner.py:37–82` `PHASE_PARAM_ROUTING` maps `{phase_name: {param_name: (source_type, *source_path)}}`. `:87–97` `PHASE_OUTPUT_KEYS` extracts declared keys. No schema, no YAML, no doc. Runtime KeyError or silent `None` propagation on rename.
- **KG-impact:** KG is only as good as each phase's handoff. If `libv2_archival` receives `html_paths=None` because `dart_conversion` renamed its output key, the archive bundle omits DART HTML and the KG loses HTML-xpath provenance edges (`process_course.py:1070`) needed for Section 508 audits.
- **Sources:** [E.10]

#### CTR-06 — `config/workflows.yaml` + `config/agents.yaml` validated by manual isinstance checks, not a JSON Schema
- **Severity:** Medium
- **Evidence:** `MCP/core/config.py:97–204, 214–307`. No externalized schema. Unknown keys silently ignored.
- **KG-impact:** Workflow config governs which validation gates run — i.e. which quality filters the KG publishes past. A typo'd `gate_id` (`bloom_alignment` vs `bloom-alignment`) is accepted and silently treated as a new gate with no handler, degrading guardrails.
- **Sources:** [E.11]

#### CTR-07 — `validate_dart_markers` is an orphan MCP tool
- **Severity:** Medium
- **Evidence:** `MCP/tools/pipeline_tools.py:368–416` — tool exists and validates 4 DART markers (`skip-link, role="main", aria-labelledby, dart-section/dart-document`). Not invoked by any phase in `config/workflows.yaml`.
- **KG-impact:** DART→CF marker contract has a validator with no gate wired. A DART semantic-class rename silently degrades Courseforge's textbook ingestor → objective extraction, polluting `LearningObjective` KG nodes with generic headings.
- **Sources:** [E.1]

#### CTR-08 — Validator gate input shape is not enforced
- **Severity:** Medium
- **Evidence:** `MCP/hardening/validation_gates.py:107–114` — `Validator` Protocol is duck-typed. 9 validators in `lib/validators/*.py` each do `inputs.get(..., default)` silently. `GateResult` output shape (dataclass) IS enforced.
- **KG-impact:** Silent input-key drift → default values (e.g. `min_score=0.8`). A `min_score: 0.7` config meant for `oscqr_score` landing on `assessment_quality` returns validator score computed against `0.8` default — artifact enters KG under false pretenses.
- **Sources:** [E.9]

#### CTR-09 — DART `.quality.json` sidecar contract is implicit
- **Severity:** Low
- **Evidence:** Producer `DART/multi_source_interpreter.py:1313`. Consumer `Courseforge/scripts/textbook-loader/textbook_loader.py:214–219` attaches as soft metadata.
- **KG-impact:** Rename of sidecar suffix → Courseforge-ingested chunks lose source-reliability weighting; KG provenance loses `source.dart_quality` annotation needed to filter low-confidence chunks from training sets.
- **Sources:** [E.2]

#### CTR-10 — `chunk_schema_version` written twice in `manifest.json`
- **Severity:** Low
- **Evidence:** `Trainforge/process_course.py:1749, 1779` — top-level key and `processing.chunk_schema_version`. Either can drift against the other or against the `CHUNK_SCHEMA_VERSION` constant at `:86`.
- **KG-impact:** Provenance drift risk: a KG consumer reading one path sees `v4`, reading the other sees something else if only one is bumped.
- **Sources:** [E]

### Theme ID — Identity / join stability (carried from prior review)

#### ID-01 — LO ID field-name drift: schema `objectiveId` vs runtime `id`
- **Severity:** **Critical**
- **Evidence:** `schemas/academic/learning_objectives.schema.json:193, 238` require `objectiveId`. Runtime emit at `Courseforge/scripts/generate_course.py:523` writes `"id"`. Parser reads `lo.get("id")` at `Trainforge/parsers/html_content_parser.py:323`. `data-cf-objective-id` attribute surface IS consistent.
- **KG-impact:** A KG loader keyed to schema's `objectiveId` sees zero objectives from current CF output; a loader keyed to runtime `id` passes but emits data the schema rejects.
- **Sources:** [PR.Claim1]

#### ID-02 — Cross-course concept collisions
- **Severity:** **Critical**
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:22–33` requires only `id`. No `course_id`, no `namespace`, no origin field. CF emit doesn't scope concept slugs per-course; `Trainforge/process_course.py:273–286 normalize_tag` doesn't prepend course identifier.
- **KG-impact:** Concepts named `accessibility` from DIGPED_101 and WCAG_201 merge into a single node with no retrievable origin. "Which concepts appear in both courses A and B?" is unanswerable; course-scoped queries devolve to chunk-level scans.
- **Sources:** [PR.A1]

#### ID-03 — Position-based chunk IDs invalidate every edge evidence pointer on re-chunk
- **Severity:** **Critical**
- **Evidence:** `Trainforge/process_course.py:1003, 1027` construct chunk IDs as `f"{prefix}{start_id:05d}"`. Edge evidence records chunk IDs by value (`is_a_from_key_terms.py:163`); preference-factory misconception IDs embed chunk IDs (`preference_factory.py:140–143`).
- **KG-impact:** Graph is not rerun-stable. Re-chunk with different `MAX_CHUNK_SIZE` shifts every ID → every edge's evidence pointer, every misconception ID goes stale. Re-chunk = effective full graph rebuild.
- **Sources:** [PR.A2]

#### ID-04 — Irreversible LO ID case normalization on chunk side only
- **Severity:** High
- **Evidence:** `Trainforge/process_course.py:1501–1507` — `obj_id.lower().strip()` + `WEEK_PREFIX_RE.sub('', normalized)`. No schema declares LO IDs case-insensitive. External IMSCC sources persist original casing.
- **KG-impact:** Chunk↔Question joins on LO ID fail silently when sides arrive with different case. Case-insensitive invariant is unilateral.
- **Sources:** [PR.A3]

#### ID-05 — Concept ID normalization mismatch between emit (`_slugify`) and consume (`normalize_tag`)
- **Severity:** High
- **Evidence:** `Courseforge/scripts/generate_course.py:169–172 _slugify` — no truncation. `Trainforge/process_course.py:273–286 normalize_tag` — truncates to 4 tokens + rejects non-alphabetic-first.
- **KG-impact:** "Instructional Design and Technology Enhanced Learning" → CF slug `instructional-design-and-technology-enhanced-learning`; TF normalize `instructional-design-and-technology`. Same human-readable concept produces different graph node IDs; emitter-produced nodes don't match consumer-produced edges.
- **Sources:** [PR.A4]

#### ID-06 — Silent slug collisions
- **Severity:** High
- **Evidence:** `Trainforge/rag/inference_rules/is_a_from_key_terms.py:52–64` rule-local `_slugify` — no collision tracking. Inputs "Cognitive Load Theory" and "cognitive-load-theory" both produce `cognitive-load-theory`.
- **KG-impact:** Concept frequency counts are inflated; distinct semantic sources collapse into one node with no telltale; no `aliases[]` or pre-collision audit.
- **Sources:** [PR.A5]

#### ID-07 — No referential integrity between edges and nodes
- **Severity:** Medium
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:39–46` — edge `required` doesn't constrain `source`/`target` to exist in `nodes[]`.
- **KG-impact:** Orphaned edges pass schema validation. KG loader trusting schema ingests edges whose endpoints aren't nodes — phantom references fail at query time not load time.
- **Sources:** [PR.A6]

### Theme PRV — Provenance / run tracking (carried from prior review)

#### PRV-01 — No `created_at` on chunks, nodes, or edges
- **Severity:** High
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:14–15` puts `generated_at` at document level only; nodes/edges carry no `created_at`. Chunk records `process_course.py:1080–1092` carry `schema_version` but no timestamp.
- **KG-impact:** Replaying a course indistinguishable from incremental update at node/edge or chunk granularity. Cannot age out stale assertions; cannot answer "all edges added after run R"; out-of-band bookkeeping (mtime, git IDs) required.
- **Sources:** [PR.C1]

#### PRV-02 — No `run_id` / `generated_by` on chunks, nodes, or edges
- **Severity:** High
- **Evidence:** `schemas/events/decision_event.schema.json:19–22` defines `run_id` as required on decision events, but identity doesn't propagate to artifacts. Chunk doesn't know its run; node/edge doesn't either.
- **KG-impact:** KG assertion cannot be traced to emitting run. Rollback ("undo everything run R produced") impossible at graph granularity. Incremental publishing requires external bookkeeping that duplicates the decision-event ledger.
- **Sources:** [PR.C2]

#### PRV-03 — Edge evidence is polymorphic per rule, not schematized per rule
- **Severity:** Medium
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:56–64` — `evidence` is `{type: object, additionalProperties: true}`. `prerequisite_from_lo_order.py:143–148` emits LO-position anchors; `is_a_from_key_terms.py:159–168` emits chunk + pattern; `related_from_cooccurrence.py:72–79` emits weight only, no anchor.
- **KG-impact:** "Show me chunks supporting any is-a edge into concept X" requires hard-coded per-rule evidence-shape knowledge. No discriminator in schema — consumers either duplicate the rule catalog or treat evidence as opaque.
- **Sources:** [PR.Claim3]

#### PRV-04 — Coverage regressions invisible: `chunks_skipped_no_lo` aggregate counter only
- **Severity:** Medium
- **Evidence:** `Trainforge/synthesize_training.py:119–121, 243–246` — `_eligible` returns False for chunks without `learning_outcome_refs`; the skip is logged as aggregate count with no per-chunk reason code or ID. `_extract_objective_refs(item)` at `process_course.py:1088` returns `[]` silently when parse fails.
- **KG-impact:** Coverage regressions invisible in run output. Entire module's worth of chunks can drop from training without surfacing in quality report. Blocks "chunks excluded from training in last run, with reason" query.
- **Sources:** [PR.Claim4]

### Theme LNK — Missing links (carried from prior review)

#### LNK-01 — Concept nodes carry no back-reference to chunks
- **Severity:** **Critical**
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:22–33` — node properties `{id, label, frequency}` only. No `occurrences[]`, no `chunks[]`, no `defined_in[]`.
- **KG-impact:** "Which chunks define concept X?" requires O(N) scan in chunks. This is arguably the most fundamental concept-graph query, and its absence means concept navigation cannot use graph traversal.
- **Sources:** [PR.B1]

#### LNK-02 — Misconceptions are prose-only with unstable synthetic IDs and no concept/LO link
- **Severity:** High
- **Evidence:** CF emits `{misconception: str, correction: str}` free text at `generate_course.py:591`. Trainforge synthesizes `misconception_id = f"{chunk_id}_mc_{index:02d}_{short}"` at `preference_factory.py:140–143` — chunk_id is position-dependent (ID-03), index is ordering-dependent, short is SHA256 prefix of text.
- **KG-impact:** Misconception identity not stable across runs. KG storing misconceptions as nodes gets a new node population for the same course on re-emit. "Misconceptions targeting concept Z" needs re-indexing every run.
- **Sources:** [PR.B2]

#### LNK-03 — `keyTerms` definitions lost in non-JSON-LD fallback path
- **Severity:** Medium
- **Evidence:** CF emits structured `{term, definition}` in JSON-LD at `generate_course.py:560–563`. Fallback `data-cf-key-terms` attribute at `:425` is comma-separated slugs only.
- **KG-impact:** Same course re-emitted without JSON-LD yields lower-fidelity graph (concept nodes carry no definition field, no link to defining text). JSON-LD path is implicitly load-bearing for KG quality while schema doesn't require it.
- **Sources:** [PR.D4]

#### LNK-04 — Pedagogical edge vocabulary is 3 types; missing `assesses, exemplifies, misconception-of, derived-from-objective, defined-by`
- **Severity:** High
- **Evidence:** `schemas/knowledge/concept_graph_semantic.schema.json:46` edge-type enum is `["prerequisite", "is-a", "related-to"]`. Precedence map at `Trainforge/rag/typed_edge_inference.py:48–52` covers same three.
- **KG-impact:** Graph can express taxonomic hierarchy and curricular ordering; cannot express pedagogical relationships that carry training signal (question→LO, example→principle, misconception→concept, chunk→LO as edge, concept→chunk-that-introduces-it). All data present, no typed surface. KG built today under-represents the structure a curriculum graph is most valuable for.
- **Sources:** [PR.Claim2]

---

## § 3 Ranked recommendations

23 recommendations in 4 priority buckets. Each carries: target files; change shape (≤ 3 lines); KG-impact statement; effort (S = ≤ 1 day, M = 1–3 days, L = > 3 days); dependencies on other recs. Every P0/P1 rec cites the finding IDs it closes.

### P0 — Foundational (unblocks multiple downstream recs)

#### REC-JSL-01 — Publish `schemas/knowledge/courseforge_jsonld_v1.schema.json`
- **Closes:** JSL-01, JSL-04, JSL-05, JSL-09, JSL-10, VOC-02, VOC-05
- **Targets:** new file `schemas/knowledge/courseforge_jsonld_v1.schema.json`; `$id: https://ed4all.dev/ns/courseforge/v1/CourseModule.schema.json`; draft 2020-12 to match peers.
- **Change shape:** Define top-level `CourseModule` shape per Worker C outline. Sub-shapes: `LearningObjective`, `Section`, `KeyTerm`, `Misconception` (tighten to required-both-keys), `AssessmentType` (reconciled — see REC-VOC-01). Normalize `bloomRange` to `array<BloomLevel>` (min 1, max 2). `$ref` into `schemas/taxonomies/*` for enums.
- **Unlocks:** Enum-safe KG joins on `moduleType`, `cognitiveDomain`, `contentType`, `bloomLevel`, `assessmentSuggestions`. Schema-level detection of producer renames. Collapses 4 silent-drift gaps (bloomRange, misconception shape, moduleType enum, cognitiveDomain enum) into one validated contract.
- **Effort:** M
- **Depends on:** REC-VOC-01 (AssessmentType union), REC-BL-01 (BloomLevel canonical source)
- **Sources:** [C], [UP]

#### REC-CTR-01 — Schematize `chunks.jsonl` as `schemas/knowledge/chunk_v4.schema.json` + validation hook
- **Closes:** CTR-01, VOC-08
- **Targets:** new file `schemas/knowledge/chunk_v4.schema.json`; validation call before `chunks.jsonl` write in `Trainforge/process_course.py:1668`.
- **Change shape:** Enumerate `chunk_type`, `bloom_level`, `difficulty`; make `source.{course_id, module_id, lesson_id}` required; require `learning_outcome_refs[]` with item pattern; include `schema_version: const "v4"`.
- **Unlocks:** Every downstream KG join guarantee — this is the KG's node layer. Closes the biggest structural gap Worker E surfaced. Makes `CHUNK_SCHEMA_VERSION` bumps detectable by consumers.
- **Effort:** M
- **Depends on:** REC-VOC-03 (content_type union needed for chunk_type enum), REC-BL-01 (BloomLevel canonical)
- **Sources:** [E.7]

#### REC-BL-01 — Consolidate 11 BLOOM_VERBS copies into `schemas/taxonomies/bloom_verbs.json` + `lib/ontology/bloom.py`
- **Closes:** BL-01, BL-02, BL-03, BL-04, BL-05
- **Targets:** new data file `schemas/taxonomies/bloom_verbs.json` (derived from richest copy at `bloom_taxonomy_mapper.py:55`, preserving `verb + usage_context + example_template`); new loader `lib/ontology/bloom.py` with `get_verbs()`, `get_verb_objects()`, `detect_bloom_level()`; migrate 11 call sites per Worker A's migration table.
- **Change shape:** Single JSON source of truth; Python adapter exposes `Dict[str, Set[str]]`, `Dict[str, List[str]]`, and enriched dataclass views to minimize per-caller churn.
- **Unlocks:** Single-vocabulary Bloom-level KG edges; deterministic validator vs emitter agreement; query-side and chunk-side verb symmetry. Eliminates whole class of silent Bloom-level flips (D1–D10 in Worker A).
- **Effort:** L (11 call sites; LibV2 cross-package caveat needs vendored JSON copy)
- **Depends on:** none (enables others)
- **Sources:** [A], [UP]

#### REC-TAX-01 — Propagate LibV2 subject taxonomy onto course manifest stub + page JSON-LD + chunks
- **Closes:** TAX-01, TAX-02, TAX-06, TAX-07
- **Targets:** `Courseforge/scripts/generate_course.py` (emit `classification` block into course.json stub + inherit `division/primary_domain/subdomains` into `_build_page_metadata`); `Trainforge/process_course.py:2731–2735, 1757–1764` (consume from stub, not CLI flags); `lib/ontology/taxonomy.py` (new — reusable loader lifted from `LibV2/tools/libv2/concept_vocabulary.py:112`).
- **Change shape:** Course-planning phase emits `classification` + `ontology_mappings` into a course.json stub at IMSCC package root; Trainforge loads stub when `--division` etc. are absent; promote `validate_taxonomy_compliance` to run at Courseforge emit time (fail-closed).
- **Unlocks:** Cross-course dedupe by taxonomy node; federated queries like "all STEM/physics/mechanics chunks"; fail-closed misclassification at emit time (before IMSCC exists). Turns `_domain` chunk field into a reliable filter axis.
- **Effort:** L
- **Depends on:** `lib/ontology/` landing (shares path with REC-BL-01)
- **Sources:** [D]

#### REC-CTR-04 — Reconcile decision_type enum (39 schema / 3 allowlist / real emit sites) + flip validator to fail-closed
- **Closes:** CTR-04, VOC-07, PRV-04 (partial)
- **Targets:** `schemas/events/decision_event.schema.json:63–103` (extend enum to include the 3 currently-allowlisted values: `instruction_pair_synthesis, preference_pair_generation, typed_edge_inference`, plus any real emit sites outside both); `lib/decision_capture.py:87–93, 401–414` (remove in-code allowlist, flip `_validate_record` from warn-only to fail-closed when `VALIDATE_DECISIONS=true`).
- **Change shape:** Single source of truth (schema enum). In-code tuple removed. Validator rejects writes on enum violation instead of recording `metadata.validation_issues`.
- **Unlocks:** Decision-event KG edges form one provenance population, not two. "Why did the KG assert X?" queries become answerable. `metadata.validation_issues` signal is no longer discarded.
- **Effort:** S
- **Depends on:** none
- **Sources:** [B.§8], [E.8], [PR.C3]

### P1 — High leverage (single-surface or small-surface changes with clear KG payoff)

#### REC-VOC-01 — Separate `question_type` (item-level) and `assessment_method` (course-level) vocabularies; converge enums
- **Closes:** VOC-04 (partial), JSL-05 (partial)
- **Targets:** new `schemas/taxonomies/question_type.json` (9-value union: `multiple_choice, multiple_response, true_false, short_answer, essay, matching, fill_in_blank, ordering, hotspot`) and `schemas/taxonomies/assessment_method.json` (LO-schema 9-value method enum). Update `question_factory.py:81–89`, `trainforge_decision.schema.json:62–65`, `generate_course.py:534–541`, `learning_objectives.schema.json:219–225` to `$ref` the respective file. Stop emitting question-format values into LO-schema `assessmentSuggestions`.
- **Change shape:** Two vocabularies where one ambiguous surface existed; both $ref'd from a single schemas/taxonomies file.
- **Unlocks:** Question→Type KG edges have one authority; Objective→AssessmentMethod edges carry semantically correct values; unreachable schema values (`ordering, hotspot`) become emit-reachable.
- **Effort:** M
- **Depends on:** none (prerequisite for REC-JSL-01)
- **Sources:** [B.§4], [B.§6], [C], [PR.Claim5], [PR.B4]

#### REC-VOC-02 — Emit `teaching_role` deterministically from `data-cf-component` + `data-cf-purpose`
- **Closes:** VOC-03, JSL-07
- **Targets:** `Courseforge/scripts/generate_course.py:345,374,487` (emit `data-cf-teaching-role` alongside component/purpose); new deterministic mapping in `lib/ontology/teaching_roles.py`; `Trainforge/align_chunks.py:487–582` (prefer `data-cf-teaching-role` when present, fall back to LLM).
- **Change shape:** `{flip-card, term-definition} → introduce`, `{activity, practice} → transfer`, `{self-check, formative-assessment} → assess`, etc. (full mapping in Worker B). Courseforge becomes producer of record; Trainforge LLM path becomes fallback only.
- **Unlocks:** Chunk→TeachingRole KG edge is deterministic and reproducible across runs. LibV2's primary retrieval filter (ADR-002) gains provenance. Same course's KG stable across runs.
- **Effort:** M
- **Depends on:** REC-JSL-01 (publishes the new attribute)
- **Sources:** [B.§3], [C], [UP]

#### REC-CTR-02 — Formalize Courseforge page-type vocabulary + gate on both emit and consume
- **Closes:** CTR-02, VOC-05
- **Targets:** new `schemas/academic/courseforge_page_types.schema.json` (6-value enum `{overview, content, application, assessment, summary, discussion}` — matching actual emit at `generate_course.py:647–759`); validation in both `generate_course.py` and `package_multifile_imscc.py:82–107`.
- **Change shape:** Shared constants module `lib/courseforge_identifiers.py`; page-type emit and packager-order dict both $ref the enum; new page types require schema update.
- **Unlocks:** Page→ModuleType KG partition becomes schema-validated. New page types added on emit without packager update are rejected at package time, not silently landed as nameless items. Closes ontology-map staleness (`schemas/ONTOLOGY.md:108` still says 5 values).
- **Effort:** S
- **Depends on:** none
- **Sources:** [B.§5], [E.3]

#### REC-CTR-03 — Default `--objectives` on in `package_multifile_imscc.py` (promote LO gate to always-on)
- **Closes:** CTR-03
- **Targets:** `Courseforge/scripts/package_multifile_imscc.py:127–179`; `config/workflows.yaml` (add `page_objectives` as a `validation_gates:` entry on the `packaging` phase of `course_generation` and `textbook_to_course`).
- **Change shape:** `--skip-validation` becomes explicit opt-out; workflow gate promotes it into the orchestrator's severity/threshold system.
- **Unlocks:** Chunk→LearningObjective KG edge (the most load-bearing edge) no longer corrupted by IMSCC shipping with LO IDs outside the canonical registry. The KG's primary join gains a hard enforcement point.
- **Effort:** S
- **Depends on:** none
- **Sources:** [E.4]

#### REC-JSL-02 — Emit `prerequisitePages` from Courseforge (close the consume-only gap)
- **Closes:** JSL-02
- **Targets:** `Courseforge/scripts/generate_course.py:_build_page_metadata:571`.
- **Change shape:** Extend `_build_page_metadata` to accept + emit a `prerequisitePages: [pageId]` list derived from course planner output or LO prerequisite chain.
- **Unlocks:** Prerequisite-page KG edges populate for the first time. Inter-page dependency graph becomes queryable — adaptive-learning / prerequisite-traversal queries return non-empty sets.
- **Effort:** S
- **Depends on:** REC-JSL-01 (to formalize the field in schema)
- **Sources:** [C]

#### REC-JSL-03 — Parse `data-cf-objective-ref` on activities and self-checks (close the emit-only gap)
- **Closes:** JSL-06
- **Targets:** `Trainforge/parsers/html_content_parser.py` — add attribute scan on `.self-check` and `.activity-card` elements.
- **Change shape:** When present, attach to the activity's chunk record as `learning_outcome_refs[]`.
- **Unlocks:** Activity→LearningObjective KG edge materializes. "Which activities exercise LO Y?" becomes a direct graph query, not an HTML rescan.
- **Effort:** S
- **Depends on:** none (data already emitted)
- **Sources:** [C], [PR.B3]

#### REC-PRV-01 — Attach `run_id` + `created_at` to every chunk, node, and edge
- **Closes:** PRV-01, PRV-02
- **Targets:** `schemas/knowledge/concept_graph_semantic.schema.json:22–46` (add both fields to nodes + edges); `schemas/knowledge/chunk_v4.schema.json` (add both — part of REC-CTR-01); `Trainforge/process_course.py` (thread `run_id` from decision-event ledger + set `created_at` at emit); edge rule emitters.
- **Change shape:** Both fields required on new artifacts; schema `$ref`s a shared `run_id` type matching `decision_event.schema.json:19–22`.
- **Unlocks:** "All edges added after run R" becomes answerable; rollback at graph granularity becomes possible; KG ages out stale assertions; incremental publish without external bookkeeping.
- **Effort:** M
- **Depends on:** REC-CTR-01
- **Sources:** [PR.C1], [PR.C2]

### P2 — Mid leverage (useful, smaller KG-impact or larger effort)

#### REC-VOC-03 — Publish content_type union + element-class-discriminated enums
- **Closes:** VOC-01, VOC-08, JSL-10
- **Targets:** new `schemas/taxonomies/content_type.json` with union of CF `_infer_content_type` 8-value + TOG `CONTENT_TYPE_DEFAULTS` 20-value + callout 2-value + `chunk_type` union. Element-class discriminator (`SectionContentType` vs `CalloutContentType`).
- **Change shape:** One schema file hosting the content-type namespace; `instruction_pair.schema.json:46–50` free-string becomes enum-constrained; LibV2 `ChunkFilter.content_type_label` validates.
- **Unlocks:** Chunk→ContentType KG edge is controlled-vocabulary. `instruction_factory` template catalog stops falling to `_default` silently. LibV2 retrieval filter partitions reliably. Closes the most fragmented vocabulary in the repo (Worker B's "critical" item).
- **Effort:** L (Courseforge + Trainforge + LibV2 coordination; historical data may need migration)
- **Depends on:** none (enables REC-CTR-01)
- **Sources:** [B.§2], [C], [PR.D1]

#### REC-LNK-01 — Add `occurrences[]` / `chunks[]` back-reference on concept nodes
- **Closes:** LNK-01
- **Targets:** `schemas/knowledge/concept_graph_semantic.schema.json:22–33` (add `occurrences: array<chunk_id>`, optional); `Trainforge/rag/inference_rules/is_a_from_key_terms.py` + any concept-node emitter to populate.
- **Change shape:** Every concept node carries a list of chunk IDs that reference it. Populated from existing chunk `concept_tags` inverted index.
- **Unlocks:** "Which chunks define concept X?" from O(N) scan to O(1) graph traversal — the most fundamental concept-graph query. Concept navigation becomes a graph-first operation.
- **Effort:** M
- **Depends on:** ID-03 stable chunk IDs (else back-reference rots on re-chunk)
- **Sources:** [PR.B1]

#### REC-LNK-02 — Introduce first-class Misconception entity with stable content-derived ID + concept/LO links
- **Closes:** LNK-02, JSL-05 (full closure), PRV-04 (partial)
- **Targets:** new `schemas/knowledge/misconception.schema.json`; change `Trainforge/generators/preference_factory.py:140–143` to derive `misconception_id` from a content hash (not chunk position); wire explicit `{concept_id, lo_id}` fields.
- **Change shape:** Misconception becomes first-class entity with stable identity across re-chunks and typed links.
- **Unlocks:** "Misconceptions targeting concept Z" becomes stable. KG storing misconceptions as nodes gets the same node population across re-emits. Enables the `misconception-of` edge type (REC-LNK-04).
- **Effort:** M
- **Depends on:** REC-ID-01 (stable content-hash IDs), REC-JSL-01
- **Sources:** [PR.B2], [C]

#### REC-ID-01 — Content-hash chunk IDs (replace position-based)
- **Closes:** ID-03 (partial — misconception IDs need REC-LNK-02), PRV-02 (partial)
- **Targets:** `Trainforge/process_course.py:1003, 1027` — replace `f"{prefix}{start_id:05d}"` with `f"{prefix}{content_hash(chunk_text):016x}"` using stable canonicalization (e.g. text + section path + schema_version).
- **Change shape:** Chunk IDs derive from content, not position.
- **Unlocks:** Re-chunking no longer invalidates every edge evidence pointer. Graph becomes rerun-stable. Edge evidence survives across runs provided the underlying content is unchanged.
- **Effort:** L (migration path for historical chunks; all edge-evidence consumers need simultaneous update)
- **Depends on:** REC-CTR-01 (chunk schema with explicit `id` contract)
- **Sources:** [PR.A2]

#### REC-ID-02 — Scope concept IDs to course namespace OR use content-hash
- **Closes:** ID-02, ID-06 (partial)
- **Targets:** `schemas/knowledge/concept_graph_semantic.schema.json:22–33` (add `course_id` as required, or change `id` to be `{course_id}:{slug}`); concept-node emit sites.
- **Change shape:** Concept nodes gain origin scoping. Cross-course merge requires explicit `aliases[]` or a separate equivalence edge type.
- **Unlocks:** Cross-course dedupe becomes explicit and auditable. "Which concepts appear in both courses A and B?" answerable via join on alias/equivalence structure rather than silent merge.
- **Effort:** L (historical data migration; cross-course retrieval logic update)
- **Depends on:** none
- **Sources:** [PR.A1]

#### REC-ID-03 — Unify `_slugify` and `normalize_tag`; move to `lib/ontology/slugs.py`
- **Closes:** ID-05, ID-06 (full closure), JSL-08
- **Targets:** `Courseforge/scripts/generate_course.py:169–172` + `Trainforge/process_course.py:273–286` + `Trainforge/rag/inference_rules/is_a_from_key_terms.py:52–64`. Move to shared `lib/ontology/slugs.py` with collision-tracking (optional `aliases[]`).
- **Change shape:** Single slug function; emit-side and consume-side agree by construction.
- **Unlocks:** Same human-readable concept produces the same graph node ID from every surface. No phantom concept population.
- **Effort:** M
- **Depends on:** REC-ID-02 (may change slug signature)
- **Sources:** [PR.A4], [PR.A5], [C]

#### REC-LNK-04 — Expand concept-graph edge-type enum with pedagogical edges
- **Closes:** LNK-04
- **Targets:** `schemas/knowledge/concept_graph_semantic.schema.json:46` (add `assesses, exemplifies, misconception-of, derived-from-objective, defined-by` — 5 new values); `Trainforge/rag/typed_edge_inference.py:48–52` (extend precedence map); add rule modules under `Trainforge/rag/inference_rules/` for each new edge type.
- **Change shape:** Each new edge type gets its own rule module, schema enum entry, precedence rank — following the existing pattern.
- **Unlocks:** Question→LO, example→principle, misconception→concept, chunk→LO-as-edge, concept→first-defining-chunk become graph-native queries. Curriculum graph gains the pedagogical connectivity that carries its training signal.
- **Effort:** L (5 new rule modules; evidence shapes need per-rule design)
- **Depends on:** REC-LNK-02 (misconception identity), REC-LNK-01 (defined-by requires back-reference surface)
- **Sources:** [PR.Claim2]

### P3 — Lower priority / deferred (useful but low KG-impact-per-effort)

#### REC-PRV-02 — Publish per-rule evidence schemas with discriminator
- **Closes:** PRV-03
- **Targets:** extend `schemas/knowledge/concept_graph_semantic.schema.json:56–64` with `oneOf` discriminator keyed by `rule` name; each rule's evidence shape becomes a separate sub-schema.
- **Change shape:** Edge evidence becomes typed per rule; consumers stop hard-coding per-rule evidence shape.
- **Unlocks:** "Show me chunks supporting any is-a edge into concept X" becomes schema-queryable without per-rule knowledge.
- **Effort:** M
- **Depends on:** none
- **Sources:** [PR.Claim3]

#### REC-CTR-05 — Move `PHASE_PARAM_ROUTING` + `PHASE_OUTPUT_KEYS` into `config/workflows.yaml` + meta-schema
- **Closes:** CTR-05, CTR-06
- **Targets:** `MCP/core/workflow_runner.py:37–97` — rewrite against per-phase `inputs_from:`/`outputs:` blocks in `config/workflows.yaml`. Add meta-schema `schemas/config/workflows_meta.schema.json` validated at load.
- **Change shape:** Phase routing becomes declarative YAML; rename of a phase's return key fails at config-load time, not mid-pipeline.
- **Unlocks:** Phase handoff failures become pre-flight detectable. `libv2_archival` no longer silently drops HTML-xpath provenance edges. Gate typos (`bloom_alignment` vs `bloom-alignment`) caught at load time.
- **Effort:** M
- **Depends on:** none
- **Sources:** [E.10], [E.11]

#### REC-CTR-06 — Wire `validate_dart_markers` into the `dart_conversion` phase gate
- **Closes:** CTR-07
- **Targets:** `config/workflows.yaml` — add DART marker validation as `validation_gates:` entry on `dart_conversion` phase of `batch_dart` and `textbook_to_course`.
- **Change shape:** The orphan MCP tool becomes a real gate.
- **Unlocks:** DART semantic-class renames caught at phase boundary, not silently degrading downstream LO extraction.
- **Effort:** S
- **Depends on:** none
- **Sources:** [E.1]

#### REC-JSL-04 — Remove emit-only / advisory attributes or convert to JSON-LD
- **Closes:** JSL-11
- **Targets:** `Courseforge/scripts/generate_course.py:327` (`data-cf-objectives-count`); either wire a consumer or remove to reduce contract surface.
- **Change shape:** Eliminate no-op attributes that drift silently without KG benefit.
- **Unlocks:** Smaller drift surface; cleaner contract.
- **Effort:** S
- **Depends on:** none
- **Sources:** [C]

---

## § 4 Proposed implementation roadmap

Sketch only. Wave sequencing depends on the outcome of Wave 1 foundations; commits to Wave 3+ would be premature.

### Wave 1 — Foundations (unblock everything else)

**Goal:** Land the schemas and shared libraries that every downstream wave depends on.

- **Bundle 1.A — Taxonomy layer:**
  - REC-BL-01 (bloom_verbs.json + lib/ontology/bloom.py)
  - REC-VOC-01 (question_type + assessment_method enums)
  - REC-VOC-03 (content_type.json — if scoped to union-only, not full enum enforcement yet)
  - New `schemas/taxonomies/` additions: `cognitive_domain.json` (4 values), `teaching_role.json` (6 values), `module_type.json` (6 values).
- **Bundle 1.B — Schema publication:**
  - REC-JSL-01 (courseforge_jsonld_v1.schema.json)
  - REC-CTR-01 (chunk_v4.schema.json)
  - REC-CTR-02 (courseforge_page_types.schema.json)
- **Bundle 1.C — Process:**
  - REC-CTR-04 (decision_type reconciliation + fail-closed)

Parallelizable: 1.A enables both 1.B items; 1.C is independent. All of Wave 1 is "new artifact + loader" work; no historical data touched. Estimated: 3 workers × 2 waves within Wave 1.

### Wave 2 — Emit-side hardening (Courseforge is producer-of-record)

**Goal:** Make Courseforge emit everything downstream consumers already assume; close all emit-only / consume-only gaps.

- **REC-VOC-02** — CF emits `data-cf-teaching-role` (+ JSON-LD `teachingRole`).
- **REC-JSL-02** — CF emits `prerequisitePages` per page JSON-LD.
- **REC-TAX-01** — CF emits `classification` + `ontology_mappings` into course.json stub; inherits into page JSON-LD; Trainforge consumes stub not CLI flags.
- **REC-CTR-03** — `--objectives` default-on in packager + `packaging` phase gate.

Depends on Wave 1 completion (schemas need to be in place to validate emits). Estimated: 3 workers in parallel; most changes localized to `generate_course.py` + `package_multifile_imscc.py`.

### Wave 3 — Consume-side alignment (Trainforge honors the emit-side truth)

**Goal:** Trainforge stops doing its own re-inference when CF provides the signal deterministically.

- **REC-JSL-03** — Trainforge parses `data-cf-objective-ref` on activities/self-checks.
- **REC-VOC-02 (Phase 2)** — Trainforge `align_chunks.py` prefers `data-cf-teaching-role`; LLM path becomes fallback.
- Stop lowercasing LO IDs if the ID policy (REC for ID-01 — deferred to Wave 4 structural) hasn't landed; carry case end-to-end.
- Honor `classification` from stub (part of REC-TAX-01).

Depends on Wave 2. Estimated: 2 workers.

### Wave 4 — Structural / identity (higher-risk, historical migration)

**Goal:** Fix the structural blockers. This is the wave most likely to require migration of historical chunks / edges.

- **REC-ID-01** — Content-hash chunk IDs. Cascades: every existing edge evidence, every misconception ID must be rewritten. Coordinated change across `process_course.py`, rule modules, preference factory.
- **REC-ID-02** — Course-scoped concept IDs. Historical concept-graph data needs course-scoping migration.
- **REC-ID-03** — Unified slug function (after ID-01 stabilizes).
- **REC-LNK-02** — Misconception as first-class entity with content-derived ID.
- **REC-PRV-01** — `run_id` + `created_at` on chunks/nodes/edges.

Depends on Waves 1–3. Highest migration risk; recommend a staged rollout with a v4/v5 chunk schema version bump. Estimated: L-size, full-team wave.

### Wave 5 — Graph expansion (unlocks new queries)

**Goal:** Extend the concept graph to carry the pedagogical signal.

- **REC-LNK-01** — `occurrences[]` back-reference on concept nodes.
- **REC-LNK-04** — Pedagogical edge types (`assesses, exemplifies, misconception-of, derived-from-objective, defined-by`). 5 new rule modules under `Trainforge/rag/inference_rules/`.
- **REC-VOC-03 (Phase 2)** — `content_type` full enum enforcement (not just union publication).

Depends on Waves 1–4. Speculative: the exact rule-module design should be re-scoped after Wave 1 data on emit-side content_type stability.

### Wave 6 — Provenance + governance

**Goal:** Tighten the operational surface that governs KG publish.

- **REC-CTR-05** — Phase routing into YAML + meta-schema.
- **REC-CTR-06** — `validate_dart_markers` wired into `dart_conversion` phase gate.
- **REC-PRV-02** — Per-rule evidence sub-schemas with discriminator.
- **REC-JSL-04** — Remove advisory / emit-only attributes that have no KG value.
- Agent-doc sweep: update `Courseforge/agents/content-quality-remediation.md:159` to reference `schemas/taxonomies/bloom_verbs.json` (part of REC-BL-01); audit all 12 agent docs for stale schema references.
- Update `schemas/ONTOLOGY.md` to reflect new canonical vocabularies (6-value `moduleType`, new edge types, etc.).

Depends on Waves 1–5. Estimated: 2 workers.

### Wave speculation caveats

- **Waves 1–3 are concretely dispatchable** from the current finding set. The recommendations cite specific files + line numbers; the wave bundles align with single-concern worker scopes.
- **Wave 4 requires a migration plan** that this review doesn't design. Re-chunk-stability work (REC-ID-01) has to be paired with an audit of every edge-evidence consumer; that audit needs to happen *before* the wave dispatches, not during.
- **Wave 5 is speculative** — LNK-04's 5 edge types each need an evidence-shape design and precedence-rank justification. That design is its own sub-planning session.
- **Wave 6 sweeps are low-risk but low-signal** — defer until data from Waves 1–4 shows which advisory surfaces actually had KG impact.

Recommend: commit to Waves 1 + 2 in the next implementation plan; treat Wave 3 as a dependent follow-up with go/no-go gate at the end of Wave 2; treat Waves 4+ as separately-planned efforts each with their own review + scope document.

---

## § 5 Non-goals / deferred

Explicit non-goals for the next implementation session (and this review):

- **No v0.3 target-ontology proposal.** This review is catalog + recommendations, not a forward-looking ontology redesign. Any "5-class root ontology" or specific IRI scheme belongs in a separate design document.
- **No RDF / OWL / SHACL authoring.** Findings apply regardless of backing store. KG-technology selection (Neo4j, RDF triple store, property graph, LPG overlay) is out of scope.
- **No cross-package concept-index work.** Several findings (ID-02, LNK-01) would benefit from a cross-package concept index, but designing one requires its own scope + planning.
- **No main-branch changes.** All of this work lands on `dev-v0.2.0` or its successor branch(es).
- **No changes to `schemas/ONTOLOGY.md` structure.** The ontology map is descriptive and adequate; this review criticizes what the map describes, not the map's organization. Data updates (stale `moduleType` 5-value → 6-value) are in-scope for REC-CTR-02; structural reorganization of the map document is not.
- **No enrichment-rate / retrieval-quality tuning.** Those are `docs/validation/enrichment-trace-report.md` and `FOLLOWUP-WORKER-N-1` scopes.
- **No content-quality, OSCQR, WCAG, or pedagogical review.** Separate validation surfaces.
- **No orchestrator / MCP-tool refactor.** Beyond the specific phase-routing work in REC-CTR-05, the MCP-tool surface is out of scope.
- **No gitignore-policy enforcement.** The `plans/README.md` proposes a convention; actual `.gitignore` edits are deferred.
- **No removal of legacy `PLAN_NOTES.md` at repo root.** That is a user-discretion cleanup after this effort lands.

---

## § 6 Source attribution appendix

Per-finding source-ID mapping. Legend: [A] = Worker A, [B] = Worker B (+ section ref), [C] = Worker C, [D] = Worker D, [E] = Worker E (+ contract number), [PR] = prior ontology review (+ claim/finding), [UP] = Ultraplan session notes.

### Bloom (BL)

- BL-01: sources=[A, UP]
- BL-02: sources=[A, PR.Theme-A+D1]
- BL-03: sources=[A]
- BL-04: sources=[A]
- BL-05: sources=[A]
- BL-06: sources=[A, B.§1]
- BL-07: sources=[PR.D3]

### Vocabularies (VOC)

- VOC-01: sources=[B.§2, C, PR.D1]
- VOC-02: sources=[B.§1, C]
- VOC-03: sources=[B.§3, UP]
- VOC-04: sources=[B.§4, B.§6, C, PR.Claim5, PR.B4]
- VOC-05: sources=[B.§5, C, PR.D2]
- VOC-06: sources=[B.§7]
- VOC-07: sources=[B.§8, E.8, PR.C3]
- VOC-08: sources=[B.§2, B.§7]

### JSON-LD (JSL)

- JSL-01: sources=[C, UP]
- JSL-02: sources=[C]
- JSL-03: sources=[C]
- JSL-04: sources=[C]
- JSL-05: sources=[C, PR.B2]
- JSL-06: sources=[C, PR.B3]
- JSL-07: sources=[C]
- JSL-08: sources=[C, PR.A4]
- JSL-09: sources=[C]
- JSL-10: sources=[C, PR.D1]
- JSL-11: sources=[C]

### Taxonomy (TAX)

- TAX-01: sources=[D]
- TAX-02: sources=[D]
- TAX-03: sources=[D, UP]
- TAX-04: sources=[D]
- TAX-05: sources=[D]
- TAX-06: sources=[D]
- TAX-07: sources=[D]
- TAX-08: sources=[D]

### Cross-system contracts (CTR)

- CTR-01: sources=[E.7]
- CTR-02: sources=[E.3]
- CTR-03: sources=[E.4]
- CTR-04: sources=[E.8, PR.C3]
- CTR-05: sources=[E.10]
- CTR-06: sources=[E.11]
- CTR-07: sources=[E.1]
- CTR-08: sources=[E.9]
- CTR-09: sources=[E.2]
- CTR-10: sources=[E]

### Identity (ID)

- ID-01: sources=[PR.Claim1]
- ID-02: sources=[PR.A1]
- ID-03: sources=[PR.A2]
- ID-04: sources=[PR.A3]
- ID-05: sources=[PR.A4]
- ID-06: sources=[PR.A5]
- ID-07: sources=[PR.A6]

### Provenance (PRV)

- PRV-01: sources=[PR.C1]
- PRV-02: sources=[PR.C2]
- PRV-03: sources=[PR.Claim3]
- PRV-04: sources=[PR.Claim4]

### Missing links (LNK)

- LNK-01: sources=[PR.B1]
- LNK-02: sources=[PR.B2]
- LNK-03: sources=[PR.D4]
- LNK-04: sources=[PR.Claim2]

### Coverage verification

- **All 5 Codex claims from [PR]** carried forward: Claim 1 → ID-01; Claim 2 → LNK-04; Claim 3 → PRV-03; Claim 4 → PRV-04; Claim 5 → VOC-04.
- **All 14 prior-review findings** (A1–A6, B1–B4, C1–C3, D1–D4) carried forward: A1 → ID-02; A2 → ID-03; A3 → ID-04; A4 → ID-05 (+ JSL-08); A5 → ID-06; A6 → ID-07; B1 → LNK-01; B2 → LNK-02 (+ JSL-05); B3 → JSL-06; B4 → VOC-04; C1 → PRV-01; C2 → PRV-02; C3 → VOC-07 (+ CTR-04); D1 → VOC-01 (+ JSL-10); D2 → VOC-05; D3 → BL-07; D4 → LNK-03.
- **All 11 BLOOM_VERBS sites from [A]** enumerated in BL-01 evidence field.
- **All 7 Ultraplan recommendations** present in § 3: BLOOM consolidation → REC-BL-01; JSON-LD schema → REC-JSL-01; assessment-type enum → REC-VOC-01; teaching-role emit → REC-VOC-02; taxonomy propagation → REC-TAX-01; pedagogy framework tagging → deferred (TAX-03 has no direct implementation rec in P0/P1 — flagged as P2/P3 candidate requiring its own scope); CI guard → covered by REC-CTR-04 (fail-closed decision validation) + REC-CTR-02 (page-type schema) + REC-CTR-05 (config meta-schema) + REC-CTR-06 (wire validator).

_End of review._
