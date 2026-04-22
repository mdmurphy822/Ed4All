# Courseforge Source-Attribution Audit

Scope: `dev-v0.2.0` branch. Read-only audit for the v0.3.0 provenance wave.

## Current state — content-generator inputs

The `content_generation` phase is the weakest provenance link in the pipeline. It receives **only an opaque `project_id`** — no pointer to source material, staging dir, or DART chunk map.

- `config/workflows.yaml:514-532` (`textbook_to_course → content_generation`) declares exactly one `inputs_from` entry:
  ```yaml
  inputs_from:
    - param: project_id
      source: phase_outputs
      phase: objective_extraction
      output: project_id
  ```
  Legacy fallback in `MCP/core/workflow_runner.py:70-72` confirms: `"content_generation": { "project_id": ("phase_outputs", "objective_extraction", "project_id") }`. The `staging_dir` emitted by the `staging` phase (`workflows.yaml:462-465`) is **never routed into content_generation**, despite being available as a `phase_outputs` key.
- The course-generation branch (`course_generation` workflow, `workflows.yaml:39-57`) has no `inputs_from` at all for `content_generation` — agents inherit nothing explicitly.
- The content-generator spec `Courseforge/agents/content-generator.md` makes this worse: the 532-line prompt template references "course structure", "objectives", "template foundation", "Bootstrap", "WCAG", Pattern 22 depth floors — **zero references to DART output, textbook source paths, chapter maps, or source chunks**. The example parallel-batch prompt at `content-generator.md:415-448` lists "Template Foundation / Content Requirements / Module Structure / Template Integration / Pattern 22 Prevention" — source material is not a category.
- Runtime behavior in `MCP/tools/courseforge_tools.py:194-272` (`generate_course_content`) only creates empty `week_XX/` directories and flips project status; it does not bind any source data to the child agent tasks.

**Net**: content-generator agents are generating content from (learning objectives, week structure, templates, agent prior). DART synthesized HTML sits in `Courseforge/inputs/textbooks/{run_id}/` (from `stage_dart_outputs`, `MCP/tools/pipeline_tools.py:234-281`) and is never explicitly handed to the agent that writes the prose.

## Current state — emitted metadata

Per-page JSON-LD is assembled in `Courseforge/scripts/generate_course.py:_build_page_metadata` (lines 622-661). The top-level slots emitted today:

| Slot | Source line | Note |
|---|---|---|
| `@context`, `@type`, `courseCode`, `weekNumber`, `moduleType`, `pageId` | 642-648 | Required by `schemas/knowledge/courseforge_jsonld_v1.schema.json:8` |
| `learningObjectives[]` | 650, built at 527-561 | Each LO carries `prerequisiteObjectives[]` (557-559) — object-level prereq refs |
| `sections[]` | 652, built at 595-619 | Each section has `heading`, `contentType`, `keyTerms[]`, optional `teachingRole[]`, `bloomRange[]` |
| `misconceptions[]` | 654 | |
| `suggestedAssessmentTypes[]` | 656 | |
| `classification` | 658 | Wave 2 course-level taxonomy inheritance |
| `prerequisitePages[]` | 659-660 | Wave 2 page-level refs |

**There is no source-attribution slot anywhere in the schema or the emitter.** No `sourceReferences`, no `derivedFrom`, no `provenance` field at page or section level. Grep across `Courseforge/` for `data-cf-source|sourceRef|source_chunk|provenance` returns zero hits in `generate_course.py` and in any `data-cf-*` emit site.

The `data-cf-*` surface (`Courseforge/CLAUDE.md:242-258`, emitters scattered around `generate_course.py:319`, `438`, `500-506`) covers: `role, objective-id, bloom-level, bloom-verb, bloom-range, cognitive-domain, content-type, teaching-role, key-terms, term, component, purpose, objective-ref`. No `data-cf-source*`.

## Textbook ingestor & source routing

The ingestor is designed as a one-way structural extractor, not a source-to-module mapper.

- `Courseforge/agents/textbook-ingestor.md:182-251`: pipeline is `DART HTML → textbook_structure.json`. Output (per schema `schemas/academic/textbook_structure.schema.json`) lists `chapters[]`, `sections[]`, `extractedConcepts[]`, `reviewQuestions[]`. Each chapter/section carries its own `id` and `headingText`, which are **candidate source-chunk identifiers** but are not currently propagated forward.
- Downstream usage (ingestor.md:265-269): output feeds `objective-synthesizer → course-outliner`. The only thing that survives this chain into content generation is the **learning objective** (objective IDs + statements + Bloom level). The chapter→objective provenance is lost at the synthesizer boundary.
- `objective-synthesizer` (referenced in `Courseforge/agents/objective-synthesizer.md`, one of the 5 files that contains `sourceFormat`) presumably knows which chapter informed each objective, but nothing in `course.json` / `*_course_data.json` consumed by `generate_course.py` carries a `source_chunks` or `derived_from` key — `data = json.loads(...)` at `generate_course.py:892` reads `classification`, `prerequisite_map`, `weeks[]` only.
- `stage_dart_outputs` (`MCP/tools/pipeline_tools.py:213-310`) writes a `staging_manifest.json` that lists HTML + synthesized-JSON sidecar pairs per run, but this manifest is never consumed by content_generation — it's intended for human inspection / auditing.

**No step today maps "DART chunk X informed Week 3 Content Module 2".** The closest artifact is the per-objective chapter provenance inside the ingestor workspace, which dies before content generation starts.

## Where source attribution should live

Three surfaces, each additive:

1. **JSON-LD page-level** (`schemas/knowledge/courseforge_jsonld_v1.schema.json`, add to `properties` at `:10-63`): new optional `sourceReferences: SourceRef[]` where `SourceRef` mirrors the existing `prerequisitePages` pattern (free string ids, elided when empty). Emitted from `_build_page_metadata` (`generate_course.py:622-661`) — add a parallel `source_references` kwarg and propagate through all six call-sites at `:724, 746, 767, 788, 816, 845`. This is the exact plumbing pattern Wave 2 used for `prerequisitePages`.

2. **JSON-LD section-level** (`SourceRef[]` inside `Section` at `schemas/knowledge/courseforge_jsonld_v1.schema.json:114-139`): per-section attribution — a content-block synthesizing a definition from chapter 3 vs. an example from chapter 7 should be distinguishable. Emitted from `_build_sections_metadata` (`generate_course.py:595-619`).

3. **`data-cf-*` HTML surface**: propose `data-cf-source-ids` (comma-separated slug list, mirrors existing `data-cf-key-terms` pattern from `CLAUDE.md:253`) and optionally `data-cf-source-primary` (single-id string, for the "dominant" source). Applied at `<section>` and major content blocks — same emit sites that already carry `data-cf-bloom-range` / `data-cf-teaching-role`.

`data-cf-source-ids` is naming-consistent: existing plural/slug-list attributes (`data-cf-key-terms`) use the same shape. `data-cf-source-id` (singular) is already implicitly claimed by the "one-slug-per-element" convention seen in `data-cf-term`.

## Multi-source attribution shape

The ONTOLOGY-style pattern hint is `prerequisiteObjectives: string[]` (`generate_course.py:557-559`) — an **unweighted array of IDs**. That pattern is the minimum viable shape and matches how the rest of the JSON-LD treats many-to-one refs.

Recommended shape (additive, does not block future enrichment):

```jsonc
"sourceReferences": [
  {
    "sourceId": "dart:networking_ch03#sec-2-4",   // required
    "role": "primary",                              // enum: primary|contributing|corroborating
    "weight": 0.7,                                  // optional, [0,1], elided when absent
    "confidence": 0.95                              // optional, ingestor's mapping confidence
  }
]
```

Constraints:
- `role` must be required so consumers (Trainforge KG) can distinguish "this block IS chapter 3" from "this block touches chapter 3". Enum keeps it tractable; defaults to `"contributing"` when the generator can't decide.
- `weight` + `confidence` are optional and additive — Wave 1 can emit `role` only; Wave 2 can add weights when the router learns how to produce them.
- If a simpler v1 is preferred to ship faster, fall back to `{ primary: string, contributing: string[] }` at page level — this matches the human mental model and is the shape that `prerequisitePages` (flat array) could NOT cleanly express if we need role differentiation.

## Routing step proposal

A dedicated **`source_mapping`** phase between `objective_extraction` and `course_planning` in `config/workflows.yaml:467-513` (`textbook_to_course`).

Why not an extension of `course_planning`: `course_planning` runs `course-outliner` (the existing agent that invents week structure from objectives). Overloading it would conflate pedagogical structuring with source-to-module binding, and the existing agent spec has no source-handling affordances. A separate phase keeps responsibilities clean and makes the artifact (a `source_module_map.json`) diffable.

Proposed shape:

```yaml
- name: source_mapping
  agents: [source-router]        # new agent
  parallel: false
  depends_on: [objective_extraction]
  inputs_from:
    - { param: project_id, source: phase_outputs, phase: objective_extraction, output: project_id }
    - { param: staging_dir, source: phase_outputs, phase: staging, output: staging_dir }
    - { param: textbook_structure_path, source: phase_outputs, phase: objective_extraction, output: textbook_structure_path }
  outputs: [source_module_map_path, source_chunk_ids]
```

Then extend `course_planning`'s `inputs_from` to include `source_module_map_path`, and most critically, extend `content_generation`'s `inputs_from` (currently just `project_id`) with `source_module_map_path` and `staging_dir` so child agents can cite as they write.

The map should be a JSON keyed by `(week, module_type, page_id)` → `SourceRef[]`, with the same schema as the JSON-LD slot. That way `generate_course.py` reads it like it reads `prerequisite_map` at `:913` — a single additional lookup through `_build_page_metadata`.

## Preservation through packaging

**Good news**: `package_multifile_imscc.py` is a byte-copy packager. It does NOT parse/rewrite HTML. At `package_multifile_imscc.py:239-241` it simply `zf.write(html_file, ...)` for every `week_*/*.html`. Arbitrary `data-cf-*` attributes, JSON-LD `<script>` blocks, and any custom markup survive packaging intact. No sanitizer, no BeautifulSoup round-trip, no attribute whitelist.

The only content-gating is LO validation at `:138-164` (`validate_content_objectives`) via `validate_page_objectives.validate_page` — it reads JSON-LD but does not write it. The JSON-LD `course_metadata.json` stub is passed through at `:230-233`.

**Packaging is therefore transparent to source refs**: whatever `generate_course.py` emits reaches Trainforge's IMSCC parser byte-identical.

## Key code paths

- `config/workflows.yaml:514-532` — content_generation `inputs_from` (`project_id` only).
- `MCP/core/workflow_runner.py:70-72, 105` — legacy PHASE_PARAM_ROUTING confirming no source plumbing.
- `MCP/tools/courseforge_tools.py:194-272` — `generate_course_content` runtime (no source binding).
- `MCP/tools/pipeline_tools.py:213-314` — `stage_dart_outputs` (source material landing zone).
- `Courseforge/scripts/generate_course.py:527-561` — `_build_objectives_metadata` (prereq-obj pattern template).
- `Courseforge/scripts/generate_course.py:595-619` — `_build_sections_metadata` (section-level emit site).
- `Courseforge/scripts/generate_course.py:622-661` — `_build_page_metadata` (insert page-level sourceRefs here).
- `Courseforge/scripts/generate_course.py:689-857` — `generate_week` (six `_build_page_metadata` call-sites that need the new kwarg).
- `Courseforge/scripts/generate_course.py:911-913` — `prerequisite_map` load pattern to mirror.
- `Courseforge/scripts/package_multifile_imscc.py:239-241` — byte-level HTML copy; attributes preserved.
- `schemas/knowledge/courseforge_jsonld_v1.schema.json:10-63, 114-139` — insertion points (page + Section).
- `Courseforge/agents/textbook-ingestor.md:182-251` — structural extraction that should also emit a chapter-id table.
- `Courseforge/agents/content-generator.md:415-448` — prompt template that needs a new "Source Material" section.

## Risks / things to watch

- **Bad routing poisons every downstream consumer.** If `source_module_map.json` misattributes chapter 3 to Week 5, every JSON-LD page and every Trainforge KG edge for that week will carry the wrong `sourceId`. Mitigation: confidence scores emitted by the router, plus a `min_confidence` gate at `content_generation` entry (warning, not block).
- **JSON-LD size inflation.** Six pages/week × 12 weeks × N sourceRefs can balloon the inline `<script>` blocks. Current pages already carry `learningObjectives` + `sections` + `prerequisitePages`. Dedup by emitting `sourceReferences` once per page (page-level) and only overriding at section-level when it differs.
- **`data-cf-source-ids` attribute bloat on nested elements.** If applied to every `<p>` or `<li>`, HTML will double in size. Restrict to `<section>`, `<h2>`/`<h3>` headings, and component wrappers — same elements that already carry `data-cf-content-type` or `data-cf-teaching-role`.
- **LO validator doesn't yet understand sourceRefs.** `validate_page_objectives.py` is strict about LO IDs. A parallel `validate_source_refs.py` (check ID resolvability against the staging manifest) should land in the same wave the emitter does, or bad refs will silently pass packaging.
- **Agent prompt hallucination.** Until content-generator is retooled to cite what it reads, adding a `sourceReferences` slot will encourage the model to **invent** plausible-looking chapter IDs. The emit-side schema change and the prompt/agent change MUST ship together; emit-side alone is worse than today.
- **`course_generation` workflow (non-textbook branch)** has no DART source at all. The schema change must make `sourceReferences` optional; emitting an empty array for pure-LO courses is acceptable.

## Open questions for synthesis phase

- **DART audit**: does DART emit stable chunk IDs today (e.g., `dart:{course}_{pdf}#sec-{n}` or page-anchor-based), or would a new canonical ID scheme need to be introduced? The synthesized-JSON sidecar staged at `MCP/tools/pipeline_tools.py:267-281` is the obvious place to look.
- **DART audit**: when DART merges multi-source extractions (pdftotext + pdfplumber + OCR per `batch_dart` description in `workflows.yaml:201`), what granularity survives into the `_synthesized.json` file? Per-page? Per-section? That sets the granularity ceiling for sourceRefs.
- **Trainforge audit**: is there already a consumer for `prerequisitePages` that we can use as a template for sourceRefs consumption, or is that still consume-only as the schema `$comment` at `courseforge_jsonld_v1.schema.json:61` suggests?
- **Trainforge audit**: does the html_content_parser already extract `<script type="application/ld+json">` JSON, and if so, how does it handle unknown keys? Additive schema changes only work if consumers are permissive.
- **Cross-cutting**: should `sourceId` schema live in a shared namespace (e.g., `schemas/taxonomies/v1/source_ref.schema.json`) so DART-emit and Courseforge-emit can reference the same `$def`? Recommended — prevents the exact drift flagged in `courseforge_jsonld_v1.schema.json:5-6`.
