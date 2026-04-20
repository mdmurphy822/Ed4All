# Worker C — Courseforge JSON-LD contract formal audit

## Summary
- Namespace: `https://ed4all.dev/ns/courseforge/v1` (hard-coded at `Courseforge/scripts/generate_course.py:580`; never published as a JSON Schema or JSON-LD `@context` document).
- JSON-LD fields emitted: **14** (6 page-level, 6 on `learningObjectives[]` items, 3 on `sections[]` items — with `assessmentSuggestions` counted once; `misconceptions[]` shape is producer-opaque, see gap M1).
- JSON-LD fields reliably consumed by Trainforge: **9** (`pageId`, `learningObjectives[].id/statement/bloomLevel/bloomVerb/cognitiveDomain/keyConcepts/assessmentSuggestions`, `sections[].heading/contentType/bloomRange/keyTerms`, `misconceptions`, `suggestedAssessmentTypes`). **1** consume-only field (`prerequisitePages`) is read but never emitted.
- `data-cf-*` attributes emitted: **14 distinct**. Consumed by Trainforge: **5** (`data-cf-role=template-chrome`, `data-cf-objective-id`, `data-cf-bloom-level`, `data-cf-bloom-verb`, `data-cf-cognitive-domain`, `data-cf-content-type`, `data-cf-key-terms`). Emit-only: `data-cf-bloom-range`, `data-cf-objectives-count`, `data-cf-component`, `data-cf-purpose`, `data-cf-term`, `data-cf-objective-ref`.
- Schema publication: **not done**. Proposed path: `schemas/knowledge/courseforge_jsonld_v1.schema.json` (directory exists at `schemas/knowledge/`; peers: `concept_graph_semantic.schema.json`, `instruction_pair.schema.json`, `preference_pair.schema.json`).
- Top KG-impact items: (1) `misconceptions[]` shape is undocumented — downstream misconception-targeted distractor joins drift silently. (2) `prerequisitePages` consumed but never emitted — prerequisite-graph edges are permanently empty. (3) `sections[].bloomRange` shape is `str | List[str]` — dual-type emission breaks typed range queries. (4) `assessmentSuggestions` vocabulary is not enumerated — assessment-type joins with `question_factory.VALID_TYPES` degrade.

## JSON-LD field inventory (emit side)

All fields emitted via `_build_page_metadata` (`generate_course.py:571–595`), `_build_objectives_metadata` (`:512–546`), `_build_sections_metadata` (`:549–568`). Rendered as `<script type="application/ld+json">` inside `<head>` by `_wrap_page` (`:273–279`).

| Path | Type | Required | Example | Emit site |
|---|---|---|---|---|
| `@context` | string (const) | Yes | `"https://ed4all.dev/ns/courseforge/v1"` | `generate_course.py:580` |
| `@type` | string (const) | Yes | `"CourseModule"` | `generate_course.py:581` |
| `courseCode` | string | Yes | `"SAMPLE_101"` | `generate_course.py:582` |
| `weekNumber` | integer | Yes | `3` | `generate_course.py:583` |
| `moduleType` | string enum | Yes | `"overview"`, `"content"`, `"application"`, `"assessment"`, `"summary"`, `"discussion"` | `generate_course.py:584`; callers: `:648,668,686,704,729,755` |
| `pageId` | string | Yes | `"week_03_overview"`, `"week_03_content_01_<slug>"` | `generate_course.py:585`; callers: `:649,666,687,705,730,756` |
| `learningObjectives` | array of object | Optional (elided when empty/None) | see below | `generate_course.py:587–588` |
| `learningObjectives[].id` | string | Yes (per item) | `"CO-05"`, `"TO-01"` | `generate_course.py:523` |
| `learningObjectives[].statement` | string | Yes (per item) | `"Describe the principles of visual hierarchy…"` | `generate_course.py:524` |
| `learningObjectives[].bloomLevel` | string enum \| null | Yes (per item; may be null) | `"apply"` | `generate_course.py:525` |
| `learningObjectives[].bloomVerb` | string \| null | Optional | `"apply"` | `generate_course.py:526` |
| `learningObjectives[].cognitiveDomain` | string enum | Yes (defaults to `"conceptual"`) | `"procedural"` | `generate_course.py:527` |
| `learningObjectives[].keyConcepts` | array of string (slugs) | Optional | `["visual-hierarchy","contrast"]` | `generate_course.py:529–530` |
| `learningObjectives[].assessmentSuggestions` | array of string | Optional | `["multiple_choice","short_answer","essay"]` | `generate_course.py:532–541` |
| `learningObjectives[].prerequisiteObjectives` | array of string | Optional | `["CO-02"]` | `generate_course.py:542–544` |
| `sections` | array of object | Optional | see below | `generate_course.py:589–590` |
| `sections[].heading` | string | Yes (per item) | `"Grouping by Proximity"` | `generate_course.py:555` |
| `sections[].contentType` | string enum | Yes (per item; defaults via `_infer_content_type` `:388–405`) | `"example"`, `"definition"`, `"procedure"`, `"comparison"`, `"exercise"`, `"overview"`, `"summary"`, `"explanation"` | `generate_course.py:556` |
| `sections[].keyTerms` | array of `{term, definition}` | Optional | `[{"term":"Proximity","definition":"Close elements appear related."}]` | `generate_course.py:559–563` |
| `sections[].bloomRange` | string \| array of string | Optional | `"apply"` **or** `["remember","apply"]` | `generate_course.py:564–566` **[SHAPE DRIFT]** |
| `misconceptions` | array (producer-opaque — shape not constrained at emit site) | Optional | `[{"misconception":"…","correction":"…"}]` (inferred from Trainforge consumers) | `generate_course.py:591–592` |
| `suggestedAssessmentTypes` | array of string | Optional | `["short_answer","essay"]` (application), `["multiple_choice","true_false"]` (self-check) | `generate_course.py:593–594`; callers: `:689,707` |

## data-cf-* attribute inventory (emit side)

| Attribute | Element / class | Value type | Example | Emit site |
|---|---|---|---|---|
| `data-cf-role` | `a.skip-link`, `header[role=banner]`, `footer[role=contentinfo]` | const `"template-chrome"` | `"template-chrome"` | `generate_course.py:289,290,297` |
| `data-cf-objective-id` | `ul > li` in `.objectives` | canonical LO ID | `"CO-05"`, `"TO-01"` | `generate_course.py:314` |
| `data-cf-bloom-level` | `li` (objective); `.self-check`; `.activity-card` | string enum (6 Bloom levels) | `"apply"` | `generate_course.py:316,375,488` |
| `data-cf-bloom-verb` | `li` (objective) | string (free) | `"apply"` | `generate_course.py:318` |
| `data-cf-cognitive-domain` | `li` (objective) | enum: `factual\|conceptual\|procedural\|metacognitive` | `"procedural"` | `generate_course.py:320` |
| `data-cf-objectives-count` | `div.objectives` | integer | `"4"` | `generate_course.py:327` |
| `data-cf-component` | `.flip-card`, `.self-check`, `.activity-card` | enum: `flip-card\|self-check\|activity` | `"flip-card"` | `generate_course.py:345,374,487` |
| `data-cf-purpose` | `.flip-card`, `.self-check`, `.activity-card` | enum: `term-definition\|formative-assessment\|practice` | `"practice"` | `generate_course.py:345,374,487` |
| `data-cf-term` | `.flip-card` | slug | `"visual-hierarchy"` | `generate_course.py:346` |
| `data-cf-objective-ref` | `.self-check`, `.activity-card` | objective ID | `"CO-05"` | `generate_course.py:378,491` |
| `data-cf-content-type` | `h2/h3`; `.callout` | enum (8 values above + `"application-note"`, `"note"` on callouts) | `"example"` | `generate_course.py:423,452` |
| `data-cf-key-terms` | `h2/h3` | comma-separated slugs | `"proximity,similarity"` | `generate_course.py:425` |
| `data-cf-bloom-range` | `h2/h3` | string or string-array-as-string | `"apply"` | `generate_course.py:427` |

Chrome attribute `data-cf-role="template-chrome"` was added in Worker Q (referenced by `Trainforge/parsers/html_content_parser.py:64–69`) to let the text extractor skip repeated page chrome.

## Consume-side map (Trainforge)

Priority chain documented at `Trainforge/parsers/html_content_parser.py:179` and in `Trainforge/CLAUDE.md` (Metadata Extraction section): **JSON-LD > data-cf-* > regex heuristics**.

| Field / Attribute | Parser path | Fallback behavior | Consume site |
|---|---|---|---|
| whole JSON-LD block | regex-match first `<script type="application/ld+json">`, `json.loads` | if parse fails or non-dict → `None` (Strategy 2 kicks in) | `html_content_parser.py:236–247` |
| `pageId` | `json_ld.get("pageId")` | `None` if missing | `html_content_parser.py:217` |
| `misconceptions` | `json_ld.get("misconceptions", [])` | `[]` (→ `ParsedHTMLModule.misconceptions`) | `html_content_parser.py:218`; further consumed in `process_course.py:1174–1184` |
| `prerequisitePages` | `json_ld.get("prerequisitePages", [])` | `[]` | `html_content_parser.py:219` **[CONSUME-ONLY — never emitted]** |
| `suggestedAssessmentTypes` | `json_ld.get("suggestedAssessmentTypes", [])` | `[]` | `html_content_parser.py:220` |
| `learningObjectives[].id/statement/bloomLevel/bloomVerb/cognitiveDomain/keyConcepts/assessmentSuggestions` | iterated from `json_ld["learningObjectives"]` (Strategy 1) | If absent → Strategy 2 (data-cf-*) → Strategy 3 (regex + `data-objective-id` legacy) | `html_content_parser.py:320–331`; also re-read in `process_course.py:1119–1125`, `:1544–1550`, `:2576–2585` for difficulty + bloom source gates |
| `sections[].heading/contentType/bloomRange/keyTerms` | `cf_meta["sections"]` walk with lowercased heading match | If heading mismatch / empty → `data-cf-*` section fallback then `none_*` trace codes (H1/H2/H3/H4 hypotheses) | `process_course.py:1288–1316` |
| `data-cf-role="template-chrome"` | HTMLParser attr scan during text extraction | no chrome-skip → boilerplate detector takes over (`Trainforge/rag/boilerplate_detector.py`) | `html_content_parser.py:82–86, 94–96, 109–110` |
| `data-cf-objective-id` / `data-cf-bloom-level` / `data-cf-bloom-verb` / `data-cf-cognitive-domain` on `<li>` | regex on captured attr string when JSON-LD absent | regex list fallback + legacy `data-objective-id` scan | `html_content_parser.py:334–358` |
| `data-cf-content-type`, `data-cf-key-terms` on `<h1–h6>` | regex on attr string | `None` / `[]` on absence | `html_content_parser.py:292–297`; also `process_course.py:1323–1332` |
| `data-cf-role`, `data-cf-objective-id`, `data-cf-content-type` as atomic chunking boundary | prefix list | no atomic boundary → split-through possible | `process_course.py:338–340` |

## Round-trip gap table

| Item | Status | Evidence | KG-impact |
|---|---|---|---|
| `prerequisitePages` | consume-only | read at `html_content_parser.py:219`, stored at `:232`; zero emit site in `generate_course.py` (grep: 0 hits) | Prerequisite-page KG edges are permanently `[]`; inter-page dependency graph silently empty, so any "which pages teach prerequisites for X" query returns nothing. |
| `prerequisiteObjectives` (LO-level) | emit-only | emitted `generate_course.py:542–544`; **no** consumer reads `prerequisiteObjectives` anywhere in `Trainforge/` (grep confirms) | LO-level prerequisite edges (e.g., `CO-02 → CO-05`) never propagate to chunks or the concept graph; Bloom progression inference has no edge data to ride on. |
| `misconceptions[]` shape | shape-mismatch (undocumented contract) | emit side passes through whatever `week_data["misconceptions"]` contains (`generate_course.py:591–592`, `:633`, `:671`) — no canonical shape. Consume side expects `{"misconception": str, "correction": str}` per item (`process_course.py:1177–1183`; `html_content_parser.py:54` types it `List[Dict[str, str]]`). | Misconception nodes in the KG will silently drop `correction` edges when a producer ships a string or alternate key; distractor-targeting joins degrade without error. |
| `sections[].bloomRange` dual type | shape-mismatch | emitted as `str | List[str]` (`generate_course.py:566`); consumer at `process_course.py:1302–1304` only uses `[0]` when list, else direct value — reducing a multi-level range to a single value. | A range `["remember","apply"]` collapses to `"remember"` in chunk metadata; Bloom-span KG edges lose their upper bound, breaking "find all chunks spanning apply+" queries. |
| `data-cf-bloom-range` (attribute) | emit-only | emitted `generate_course.py:427`; no regex consumer in parser (only `content_type` and `key_terms` are scanned at `html_content_parser.py:292–297`) | When JSON-LD is absent (non-Courseforge IMSCC) there's no fallback path for section Bloom range; data-cf-only documents silently lose Bloom info for every section. |
| `data-cf-objective-ref` | emit-only | emitted `:378, :491`; no consumer anywhere in `Trainforge/` for this exact attr name | Self-check and activity chunks can't be joined to their target LO without JSON-LD — activity→LO KG edges rely entirely on page-level JSON-LD, which doesn't identify which activity targets which LO. |
| `data-cf-component` / `data-cf-purpose` | emit-only | emitted `:345,374,487`; consumer uses class-regex (`COMPONENT_PATTERNS` `html_content_parser.py:166–173`) not the attribute | Component-type KG classification is class-regex-based, so any template swap breaks detection silently — the authoritative `data-cf-component` enum is ignored. |
| `data-cf-term` | emit-only | emitted `:346`; parser extracts key-term slugs from h2/h3 `data-cf-key-terms` + dt/strong regex (`:419–425`), never from flip-card `data-cf-term` | Flip-card terms are captured only via `<strong>`/`<dt>` regex heuristics after chrome skip — slug normalization drifts from emitter's `_slugify` (`generate_course.py:169–172`), so the same term can produce two distinct concept-graph nodes. |
| `data-cf-objectives-count` | emit-only | emitted `:327`; no consumer | Advisory only — no KG impact, but adds contract surface area that drifts silently. |
| `assessmentSuggestions` enum | enum-undocumented | emit vocabulary `{multiple_choice, true_false, fill_in_blank, short_answer, essay}` (`:534–539`); Trainforge's `question_factory.VALID_TYPES` is 7 values and `course_metadata_schema.json` is 7; consumed verbatim at `html_content_parser.py:329` | Assessment-type joins across the three enums fail for `fill_in_blank` (missing from one) and any producer-side rename; cross-system question-type roll-ups silently undercount. |
| `cognitiveDomain` enum | enum-undocumented | default `"conceptual"` hardcoded (`:520, :527`); enum `{factual, conceptual, procedural, metacognitive}` never schematized — see Worker B for cross-repo drift | "All procedural chunks" query pivots on free-text string match; any producer emitting `"procedure"` (singular) breaks the join silently. |
| `moduleType` enum | enum-undocumented | 6 literals at emit callers `:648,668,686,704,729,756` — never validated | A producer shipping `"self_check"` vs `"self-check"` vs `"assessment"` fractures the module-type KG axis. |
| `learningObjectives[].bloomLevel=null` | semantic-drift | emitted as `None` when `detect_bloom_level` fails (`:519, :525`); consumer at `process_course.py:1123` assigns it verbatim — a null bloom level is now "authoritative" | Null bloom levels from the emitter outrank downstream verb-inference fallbacks; chunks get `bloom_level_source="page_jsonld"` with no level, breaking Bloom-coverage metrics. |
| `contentType` enum mismatch on callouts | enum-split | `_infer_content_type` emits 8 section values (`:388–405`); callout emits separate `{application-note, note}` under the **same** attribute (`:448, :452`) | `data-cf-content-type` is a union of two disjoint vocabularies; a single-enum schema would fail both; KG queries over content-type need to know which element class to filter on. |

## Proposed schema outline

**Filename:** `schemas/knowledge/courseforge_jsonld_v1.schema.json`
**$id:** `https://ed4all.dev/ns/courseforge/v1/CourseModule.schema.json`
**Draft:** 2020-12 (match peers in `schemas/knowledge/`)

Top-level shape:

```
{
  @context:  const "https://ed4all.dev/ns/courseforge/v1"   [required]
  @type:     const "CourseModule"                           [required]
  courseCode: string, pattern "[A-Z]{2,}_?\\d{3,}"          [required]
  weekNumber: integer, minimum 0                            [required]
  moduleType: enum ["overview","content","application",
                    "assessment","summary","discussion"]    [required]
  pageId:    string, pattern "week_\\d{2}_[a-z0-9_]+"       [required]
  learningObjectives: array<LearningObjective>              [optional]
  sections:  array<Section>                                 [optional]
  misconceptions: array<Misconception>                      [optional]
  suggestedAssessmentTypes: array<AssessmentType>           [optional]
  prerequisitePages: array<string>                          [optional; consume-only TODAY — see gap]
}
```

Sub-shapes:

- **LearningObjective** — required `{id, statement}`; optional `{bloomLevel, bloomVerb, cognitiveDomain, keyConcepts[], assessmentSuggestions[], prerequisiteObjectives[]}`. `bloomLevel` must be `oneOf [enum, null]` if null is kept permissible OR tighten to required-enum (recommend the latter).
- **Section** — required `{heading, contentType}`; optional `{keyTerms[], bloomRange}`. `bloomRange` should be normalized to `array<BloomLevel>` (min 1, max 2) — the current `str | array` union is the KG-damaging shape gap.
- **KeyTerm** — `{term: string, definition: string}`, both required.
- **Misconception** — `{misconception: string, correction: string}`, both required (matches what `process_course.py:1177–1180` expects). Today's emit site passes through without shape validation — schema should force the shape.
- **AssessmentType** — enum union reconciled with `question_factory.VALID_TYPES` and `course_metadata_schema.json` (see Worker B).

Controlled enums to schematize (each should `$ref` a shared `schemas/taxonomies/` file to avoid the duplication Worker A/B will flag):

- `BloomLevel`: `remember|understand|apply|analyze|evaluate|create`.
- `CognitiveDomain`: `factual|conceptual|procedural|metacognitive`.
- `ContentType` (for sections): `explanation|example|procedure|comparison|exercise|overview|summary|definition`.
- `CalloutContentType` (distinct from section): `application-note|note` — or merge under a broader union with element-class discriminator.
- `ModuleType`: 6 values above.
- `AssessmentType`: reconciled union.

Companion schema (out of this worker's scope, flagged): `schemas/knowledge/courseforge_data_cf_v1.schema.json` documenting the 14 `data-cf-*` attributes, their element-class context, and their consume status.

## KG-impact summary

Top 5 gaps ranked by KG-quality damage:

1. **`prerequisitePages` consume-only** — parser infrastructure ready but emitter silent; prerequisite-graph is a permanent no-op, blocking any adaptive-learning or prerequisite-traversal KG query.
2. **`misconceptions[]` shape uncontrolled** — emitter passes through unvalidated; correction edges silently drop when producer changes keys, so misconception-targeted distractor retrieval degrades without a test ever failing.
3. **`bloomRange` str|array drift** — `["remember","apply"]` collapses to `"remember"` at consume, erasing the upper Bloom bound; range-based KG queries ("chunks spanning apply or higher") return incomplete sets.
4. **`prerequisiteObjectives` emit-only** — LO-level prerequisite edges emitted but never parsed into chunk metadata; the objective dependency graph never materializes at chunk granularity.
5. **Enum namespaces undocumented** — `moduleType`, `cognitiveDomain`, `assessmentSuggestions`, `contentType` are all string-typed at emit with no schema; any rename by a producer fractures KG joins silently. Publishing the schema with enum constraints is the single highest-leverage contract-hardening step.
