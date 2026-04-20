# Worker A — BLOOM_VERBS consolidation audit

## Summary

- **11 Bloom-verb copies found** (5 expected + 6 additional):
  1. `lib/validators/bloom.py:21` — `BLOOM_VERBS` (flat `Dict[str, Set[str]]`)
  2. `Trainforge/parsers/html_content_parser.py:156` — `HTMLContentParser.BLOOM_VERBS` (class-level `Dict[str, List[str]]`)
  3. `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55` — `BLOOM_VERBS: Dict[BloomLevel, List[BloomVerb]]` (**richest — dataclass + templates**)
  4. `Courseforge/scripts/generate_course.py:136` — module-level `BLOOM_VERBS: Dict[str, List[str]]`
  5. `LibV2/tools/libv2/query_decomposer.py:111` — `QueryDecomposer.BLOOM_VERBS` (class-level, regex-flavor list)
  6. `Courseforge/agents/content-quality-remediation.md:159` — inline `bloom_verbs` in prompt code sample (agent-spec, not executed)
  7. `Trainforge/generators/assessment_generator.py:46` — `BLOOM_LEVELS` (nested dict: verbs + patterns + question_types)
  8. `Trainforge/generators/question_factory.py:91` — `BLOOM_QUESTION_MAP` (bloom → question-type list; structural sibling, no verbs)
  9. `lib/semantic_structure_extractor/analysis/content_profiler.py:182` — `BLOOM_PATTERNS: Dict[str, List[str]]` (verb list for difficulty weighting)
  10. `lib/semantic_structure_extractor/semantic_structure_extractor.py:126` — `BLOOM_PATTERNS` (regex-wrapped verb list)
  11. `Trainforge/rag/libv2_bridge.py:473` — `bloom_verbs` (regex `|`-joined alternation, 20 verbs, for objective preamble stripping)
- Ancillary constant repeats (level name ordering only, no verbs): `Trainforge/generators/instruction_factory.py:56` `_BLOOM_LEVELS`, `LibV2/tools/libv2/query_decomposition.py:114` `BLOOM_LEVELS`.
- **Richest:** `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55` — 10 verbs/level, each a `BloomVerb` dataclass carrying `usage_context` + `example_template`; has companion `CONTENT_TYPE_DEFAULTS` (20 types) and `HIGHER_ORDER_PATTERNS`.
- **Canonical-location recommendation:** `schemas/taxonomies/bloom_verbs.json` (schema-first, loaded by a thin `lib/ontology/bloom.py` module that also publishes `detect_bloom_level`). Rationale below.
- **KG-impact headline:** Bloom level is a first-class KG edge label applied at multiple producer sites with divergent verb sets, so identical LO text traversing DART→Courseforge→Trainforge→LibV2 can be classified at different levels depending on which detector fires, corrupting both Bloom-level joins and Bloom-aware retrieval.

## Inventory

### Copy 1 — `lib/validators/bloom.py:21`
- Structural shape: module-level `Dict[str, Set[str]]` (unordered sets).
- Verb counts: `{remember: 11, understand: 9, apply: 9, analyze: 9, evaluate: 9, create: 9}` (prior plan said 9/level — actually 11 for remember).
- Callers: `lib/validators/bloom.py:55` (`detect_bloom_level`), `lib/validators/bloom.py:138` (`BloomAlignmentValidator.validate`), `lib/validators/question_quality.py:19,280` (re-export), `config/workflows.yaml:296` (`rag_training` gate `bloom_alignment`).
- Distinguishing feature: canonical detector referenced by the `rag_training` validation gate; prioritizes higher-order levels via explicit level order at `bloom.py:53`.
- KG-impact: Authoritative per workflow config, but its verb set disagrees with every producer (e.g., `label`/`match` in `remember` but not in `generate_course.py`), so producer-declared levels that the validator re-detects may flip — producing the `BLOOM_MISMATCH` warning at `bloom.py:145` on clean input.

### Copy 2 — `Trainforge/parsers/html_content_parser.py:156`
- Structural shape: class-level `Dict[str, List[str]]` on `HTMLContentParser`.
- Verb counts: `{remember: 7, understand: 6, apply: 6, analyze: 6, evaluate: 6, create: 6}` — thinnest set among live code.
- Callers: `Trainforge/parsers/html_content_parser.py:400` (`_detect_bloom_level`), invoked at `:353, :372, :386` during objective + concept + section extraction; class consumed by `Trainforge/process_course.py:43,560,2180`, `Courseforge/scripts/textbook-loader/textbook_loader.py:121`, plus test modules `test_parsers.py`, `test_metadata_extraction.py`.
- Distinguishing feature: only copy with duplicated verbs across levels (`compare` in both `understand:158` and `analyze:160`; `describe` in both `remember:157` and `understand:158`) — ambiguity resolved by iteration order in `_detect_bloom_level`, so first-match wins.
- KG-impact: Parser is the ingest surface for LibV2 chunks, so the thin verb set drives `bloom_level` tagging when Courseforge-emitted `data-cf-bloom-level` is absent — silent under-classification (LOs with verbs like `calculate`, `interpret`, `categorize`, `formulate` get `None` and fall through to heuristic defaults).

### Copy 3 — `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55`
- Structural shape: module-level `Dict[BloomLevel, List[BloomVerb]]` keyed by `BloomLevel` enum (`bloom_taxonomy_mapper.py:18`), values are `BloomVerb` dataclasses (`verb`, `level`, `usage_context`, `example_template`).
- Verb counts: `{remember: 10, understand: 10, apply: 10, analyze: 10, evaluate: 10, create: 10}` — only uniformly-rich copy.
- Callers: `Courseforge/scripts/textbook-objective-generator/objective_formatter.py:18,330`, `Courseforge/scripts/textbook-objective-generator/__init__.py:9,11,31,32` (re-export), `bloom_taxonomy_mapper.py:197,255,380`, `textbook_objective_generator.py:32,96`.
- Distinguishing feature: only copy carrying generation affordances — usage-context strings and Jinja-style templates (e.g., `"Analyze {topic} to identify {components}"`). Also paired with `CONTENT_TYPE_DEFAULTS:141` (20 content_type → BloomLevel) and `HIGHER_ORDER_PATTERNS:177` (regex patterns per level) in the same module.
- KG-impact: The richest verb vocabulary lives in the objective-generator (upstream) but never reaches downstream consumers — Trainforge/LibV2 cannot reuse the same verb catalog to re-verify emitted bloom levels, so the KG loses provenance for "this LO was classified `analyze` because verb `deconstruct` was matched with usage_context=`into parts`."

### Copy 4 — `Courseforge/scripts/generate_course.py:136`
- Structural shape: module-level `Dict[str, List[str]]`.
- Verb counts: `{remember: 7, understand: 6, apply: 6, analyze: 6, evaluate: 6, create: 6}` — identical to Copy 2 verb-for-verb.
- Callers: `generate_course.py:162` (`detect_bloom_level`), `:312, :519, :532` (`_render_objectives`, `_render_page`, `_build_objectives_metadata`). Imported externally at `tests/test_pipeline_integration.py:112`, `Courseforge/scripts/validate_page_objectives.py:37`, `Courseforge/scripts/tests/test_template_chrome_emit.py:27`, `Courseforge/scripts/tests/test_generate_course_lo_specificity.py:30` (though downstream callers import `generate_week`/`load_canonical_objectives`, not `BLOOM_VERBS` directly).
- Distinguishing feature: primary emit-side detector; paired with `BLOOM_TO_DOMAIN:146` (bloom → `factual|conceptual|procedural|metacognitive`) so this copy also governs `data-cf-cognitive-domain` emission. Single `BLOOM_VERBS` source for `data-cf-bloom-verb`, `data-cf-bloom-level`, and JSON-LD `bloom_level`.
- KG-impact: Producer truth for Courseforge `data-cf-bloom-*` + JSON-LD — every drift between this copy and `lib/validators/bloom.py` or the Trainforge parser rewrites KG edge labels during ingest, since downstream sites re-detect rather than trust.

### Copy 5 — `LibV2/tools/libv2/query_decomposer.py:111`
- Structural shape: class-level `Dict[str, List[str]]` on `QueryDecomposer`.
- Verb counts: `{remember: 11, understand: 10, apply: 10, analyze: 10, evaluate: 11, create: 11}` — second-richest by count.
- Callers: `query_decomposer.py:283` (`_detect_bloom_level`), invoked at `:227` during query decomposition; class consumed by `LibV2/tools/libv2/multi_retriever.py:12,65` and `Trainforge/rag/libv2_bridge.py:239,248`.
- Distinguishing feature: query-side (not LO-side) detection — detects Bloom level of a user *query* to influence chunk-type routing. Iterates levels in reverse at `:282` (create-first) for higher-order priority. Adjacent `LibV2/tools/libv2/query_decomposition.py:114` exposes just the level-name ordering.
- KG-impact: Retrieval-time Bloom detection uses a *different* verb set than chunk-time Bloom tagging (Copy 2). Queries asking "synthesize X" get `create`-routing; the corresponding chunks tagged by Copy 2 lack `synthesize` in their `create` list — retrieval/index Bloom axes are asymmetric.

### Copy 6 — `Courseforge/agents/content-quality-remediation.md:159`
- Structural shape: inline `bloom_verbs` dict inside a Python code block in the agent prompt (Markdown). Not imported; consumed by Claude when the agent runs.
- Verb counts: `{remember: 4, understand: 4, apply: 4, analyze: 4, evaluate: 4, create: 4}` — smallest.
- Callers: the agent itself (`content-quality-remediation`), which may prompt-generate LOs without going through `bloom_taxonomy_mapper.py`.
- Distinguishing feature: prompt-resident. Drift risk is invisible to grep once the agent generates HTML (no import chain to audit).
- KG-impact: Agent-produced LOs can use verbs absent from every other detector (e.g., if the agent improvises), and this small seed set biases remediation to only the 24 most-common verbs — narrowing Bloom-level coverage in remediated courses.

### Copy 7 — `Trainforge/generators/assessment_generator.py:46`
- Structural shape: module-level `Dict[str, Dict[str, List[str]]]` — nested: each level carries `verbs`, `patterns` (question stems), `question_types`.
- Verb counts: `{remember: 5, understand: 5, apply: 5, analyze: 5, evaluate: 5, create: 5}`.
- Callers: `assessment_generator.py:407` (`AssessmentGenerator._select_question_type`); exported by `Trainforge/generators/__init__.py:19,53`; consumed via `MCP/tools/pipeline_tools.py:1476`, `MCP/tools/trainforge_tools.py:33`, `Trainforge/tests/test_content_grounded_generation.py:157`.
- Distinguishing feature: only copy that couples verbs to question stems and question-type affordances; structural sibling of Copy 3 but uses plain dict not dataclass.
- KG-impact: Generation-time Bloom rationale ("stem chosen because `Compare and contrast...` matches `analyze`") lives here, but the KG records only the emitted `bloom_level` — evidence chain is lost because this structure isn't persisted alongside the generated question.

### Copy 8 — `Trainforge/generators/question_factory.py:91`
- Structural shape: class-level `BLOOM_QUESTION_MAP: Dict[str, List[str]]` on `QuestionFactory` (no verbs; bloom → question-type list).
- Callers: `question_factory.py:446` (`QuestionFactory._validate_bloom_alignment`).
- Distinguishing feature: verb-free sibling; included because it encodes Bloom-level affordances and would need to travel with any consolidation. Compare to Copy 7's `question_types` sub-key — disagreement: Copy 7 `remember:50` has no `matching` on-ramp; Copy 8 includes `matching` for `remember:92`.
- KG-impact: Disagreement on which question types are valid for a Bloom level means generated assessments get different Bloom provenance depending on factory path — KG assertion `(Q, targets, apply)` carries different confidence under each.

### Copy 9 — `lib/semantic_structure_extractor/analysis/content_profiler.py:182`
- Structural shape: class-level `BLOOM_PATTERNS: Dict[str, List[str]]` on `ContentProfiler`.
- Verb counts: `{remember: 9, understand: 7, apply: 8, analyze: 5, evaluate: 7, create: 7}`.
- Callers: `content_profiler.py:539` (`_analyze_bloom_levels`), invoked at `:264` during `profile_text`; class consumed by `lib/semantic_structure_extractor/semantic_structure_extractor.py:29,160,405,419`, `DART/multi_source_interpreter.py:52`.
- Distinguishing feature: feeds `BLOOM_DIFFICULTY_WEIGHTS:192` (remember=0.1 → create=1.0) — this copy converts Bloom → scalar difficulty used by DART multi-source synthesis.
- KG-impact: Difficulty scores attached to chunks derive from verb counts against *this* vocabulary; drift vs. Copy 4 (emit-side) means the `difficulty` KG property and `bloom_level` KG property can point in opposite directions on the same chunk.

### Copy 10 — `lib/semantic_structure_extractor/semantic_structure_extractor.py:126`
- Structural shape: class-level `BLOOM_PATTERNS: Dict[str, List[str]]` where values are regex strings (not plain verbs).
- Verb counts: `{remember: 9, understand: 7, apply: 8, analyze: 6, evaluate: 7, create: 7}` (counted inside the alternation groups; `analyze` includes `compare.*contrast` multi-word).
- Callers: `semantic_structure_extractor.py:891` (one internal pattern scan); class consumed by `DART/multi_source_interpreter.py:1091`, agent prompt `Courseforge/agents/textbook-ingestor.md:192`.
- Distinguishing feature: regex-wrapped, so consumable with `re.search` instead of word tokenization; near-duplicate of Copy 9 but shipped as regex alternations.
- KG-impact: DART-side Bloom inference during multi-source synthesis derives from a *third* vocabulary — any LO extracted from PDF via this path carries a Bloom assertion that Courseforge then overwrites with its own detection, and the KG has no record of the original DART-side inference.

### Copy 11 — `Trainforge/rag/libv2_bridge.py:473`
- Structural shape: regex alternation string `(define|list|recall|…)` — 20 verbs flat across all levels, no per-level grouping.
- Callers: `libv2_bridge.py:479` (`_extract_query_concepts` helper inside `CrossCourseRAG`).
- Distinguishing feature: strips Bloom preamble verbs from queries to focus embedding on concept words; treats all 20 verbs equivalently, *discards* level information.
- KG-impact: Cross-course retrieval intentionally throws away Bloom signal here — but the selection of which 20 verbs to strip is hard-coded and disjoint from every other verb list (e.g., `calculate`, `interpret`, `categorize`, `formulate` aren't stripped), so query-embedding consistency silently varies by verb choice in the LO.

## Semantic diff matrix

Verb counts per Bloom level per live code site (Copies 7–11 shown in compact form).

| Bloom level | Copy 1 `lib/validators/bloom.py` | Copy 2 `html_content_parser.py` | Copy 3 `bloom_taxonomy_mapper.py` | Copy 4 `generate_course.py` | Copy 5 `query_decomposer.py` |
|---|---|---|---|---|---|
| remember    | 11 | 7 | 10 | 7 | 11 |
| understand  |  9 | 6 | 10 | 6 | 10 |
| apply       |  9 | 6 | 10 | 6 | 10 |
| analyze     |  9 | 6 | 10 | 6 | 10 |
| evaluate    |  9 | 6 | 10 | 6 | 11 |
| create      |  9 | 6 | 10 | 6 | 11 |

Supplementary (non-primary copies): Copy 6 (`content-quality-remediation.md`) = 4/level; Copy 7 (`assessment_generator.py`) = 5/level; Copy 9 (`content_profiler.py`) = {9,7,8,5,7,7}; Copy 10 (`semantic_structure_extractor.py`) = {9,7,8,6,7,7}; Copy 11 (`libv2_bridge.py`) = 20 verbs flat.

**Verbs present in the richest copy (3) but absent from the emit copy (4):**
- remember: `label`, `match`, `recognize`, `select`  (4 missing at emit)
- understand: `discuss`, `paraphrase`, `distinguish`, `illustrate`  (4 missing)
- apply: `compute`, `calculate`, `practice`, `perform`  (4 missing)
- analyze: `relate`, `categorize`, `deconstruct`, `investigate`, `attribute`  (5 missing)
- evaluate: `defend`, `recommend`, `prioritize`, `support`  (4 missing)
- create: `invent`, `produce`, `generate`  (3 missing)

**Verbs present in emit (Copy 4) but absent from validator (Copy 1):**
- apply: `solve` is in both; but `execute` present only in validator; `implement`/`execute` split differs
- analyze: `compare`, `contrast` — validator lacks `contrast`; Copy 2/4 include both
- Copy 1 `remember` includes `describe` (which Copy 4 places in `understand`) — direct level disagreement.

**Cross-level duplicates (ambiguity sources):**
- `compare`: appears in `understand` (Copies 2, 4, 5) AND `analyze` (Copies 2, 4, 5) — first-match wins varies by iteration order.
- `describe`: appears in `remember` (Copy 1) AND `understand` (Copies 2, 3, 4).
- `distinguish`: appears in `understand` (Copies 1, 3, 5) AND `analyze` (Copies 1, 5).
- `illustrate`: appears in `apply` (Copy 1) AND `understand` (Copy 3).

## KG-impact per drift

- **D1 (validator vs emit verb set divergence):** LO "Describe the ADDIE model" is emitted by Courseforge with `bloom_level="understand"` (Copy 4), then re-validated by `BloomAlignmentValidator` (Copy 1), whose `remember` set also contains `describe` — producing `BLOOM_MISMATCH` warnings on clean input, inflating the KG's "validation failed" edge count without real drift.
- **D2 (parser thinness):** Trainforge chunk ingest (Copy 2) can't classify LOs using `calculate`, `formulate`, `categorize`, `deconstruct`, `synthesize` — KG `bloom_level` property is silently null on these chunks, so Bloom-faceted retrieval misses them.
- **D3 (query vs chunk asymmetry):** Query decomposer (Copy 5) tags a query `synthesize X` as `create`; chunks indexed by Copy 2 lack `synthesize` → recall gap on `create`-level queries.
- **D4 (richest set unreachable):** `BloomTaxonomyMapper` (Copy 3) knows verb usage-context and generation templates, but Trainforge/LibV2 can't query it — no `(LO, classified_by_verb, X)` provenance edge is creatable in the KG.
- **D5 (difficulty vs bloom inconsistency):** DART/profiler (Copies 9, 10) drive a `difficulty` score from one verb list while Courseforge emits `bloom_level` from another — KG can hold chunks with `bloom_level=remember` and `difficulty=0.85` simultaneously.
- **D6 (agent-prompt drift):** `content-quality-remediation.md` (Copy 6) can emit LOs with its 24-verb palette; no audit surface exists to catch when remediated LOs use verbs the downstream parser won't classify.
- **D7 (preamble stripping hard-coded):** `libv2_bridge.py` (Copy 11) strips 20 verbs for query embedding; drift vs. the true verb universe means LOs starting with `formulate` or `categorize` keep their Bloom preamble inside the embedding — contaminating nearest-neighbor clusters in the KG retrieval layer.
- **D8 (cognitive-domain coupling):** `BLOOM_TO_DOMAIN:146` in Copy 4 maps Bloom → cognitive domain (`factual|conceptual|procedural|metacognitive`). Any level-flip caused by D1/D2/D3 cascades into the KG's `cognitive_domain` edge too (double-drift on a single miscoded LO).
- **D9 (level ordering):** `lib/validators/bloom.py:53` and `query_decomposer.py:282` both iterate levels create-first for ambiguous verbs; `generate_course.py:162` iterates in dict order (remember-first). For `compare`, `describe`, `distinguish`, `illustrate` the emit and validate sides can *guarantee* disagreement — structural, not drift.
- **D10 (question-type affordance disagreement):** Copies 7 & 8 disagree on which question types a Bloom level supports — KG assertion `(bloom_level, permits, question_type)` has two different truth sources.

## Canonical-location recommendation

- **Path:** `schemas/taxonomies/bloom_verbs.json` (data) + `lib/ontology/bloom.py` (loader + `detect_bloom_level` + `BloomLevel` enum re-export).
- **Rationale:** Existing unified schema registry at `schemas/taxonomies/` already hosts the subject taxonomy and pedagogy framework that LibV2 publishes; placing Bloom alongside makes it a peer first-class taxonomy. JSON schema lets LibV2/Courseforge/Trainforge/DART depend on data not code, crossing the no-cross-package-imports boundary the workspaces currently enforce. Dataclass richness (usage_context, templates) from Copy 3 is preserved as optional JSON fields. A thin `lib/ontology/bloom.py` adapter provides the `Dict[str, Set[str]]` and `Dict[str, List[str]]` views that existing callers want, so per-caller churn is minimized.
- **Per-caller migration notes:**

| Caller | Current access pattern | Post-consolidation |
|---|---|---|
| `lib/validators/bloom.py` | owns `BLOOM_VERBS`, exports `detect_bloom_level` | re-export from `lib/ontology/bloom`; validator becomes pure logic |
| `lib/validators/question_quality.py:19` | imports `detect_bloom_level` | unchanged public surface |
| `Trainforge/parsers/html_content_parser.py:156` | class-level dict | class loads via `lib.ontology.bloom.get_verbs()`; keeps `HTMLContentParser.BLOOM_VERBS` alias |
| `Courseforge/scripts/generate_course.py:136` | module-level dict + `BLOOM_TO_DOMAIN` | imports from `lib.ontology.bloom`; `BLOOM_TO_DOMAIN` moves alongside as Worker B artifact |
| `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py:55` | owns `BloomVerb` dataclass + `BLOOM_VERBS` | becomes canonical writer of the JSON schema; loads back via `lib.ontology.bloom.get_verb_objects()` for usage_context/templates |
| `Courseforge/scripts/textbook-objective-generator/objective_formatter.py:18` | imports `BLOOM_VERBS, BloomLevel, BloomTaxonomyMapper` | re-export preserved |
| `LibV2/tools/libv2/query_decomposer.py:111` | class-level list | imports via `lib.ontology.bloom` — but LibV2 currently has no cross-package import; will need either a vendored JSON copy at `LibV2/schemas/taxonomies/bloom_verbs.json` or a published package |
| `Trainforge/generators/assessment_generator.py:46` | `BLOOM_LEVELS` (verbs + patterns + question_types) | verbs move to ontology; patterns + question_types stay here as assessment-specific affordances |
| `Trainforge/generators/question_factory.py:91` | `BLOOM_QUESTION_MAP` | stays local (not a verb set); reconciled against Copy 7 in Worker B vocab audit |
| `lib/semantic_structure_extractor/analysis/content_profiler.py:182` | `BLOOM_PATTERNS` + `BLOOM_DIFFICULTY_WEIGHTS` | patterns sourced from ontology; weights stay local |
| `lib/semantic_structure_extractor/semantic_structure_extractor.py:126` | regex alternations | derive regex at import time from ontology verb list |
| `Trainforge/rag/libv2_bridge.py:473` | flat regex `|`-joined | generated from `lib.ontology.bloom.all_verbs()` at module init |
| `Courseforge/agents/content-quality-remediation.md:159` | inline prompt dict | agent prompt references the ontology file path; no code change but prompt edit required |
