# GPT Feedback (12 May) Item 5 — Cognitive Task Type Axis

**Worker 5 of 5.** GPT feedback line item: *"Not only Bloom level, but observable
task type: classify, compute, critique, debug, explain, compare, construct."*

## Motivation

Today every assessment-bearing chunk and assessment-bearing misconception carries
`bloom_level` (the 6-value cognitive process tier) and every question artifact
carries `question_type` (the 9-value format / instrument tier). Those two axes
together still don't disambiguate the actual cognitive task: two `apply`-tier
multiple-choice questions can be entirely different cognitively — one is
`compute`, another is `debug`, another is `classify`. The action verb is baked
into the stem but never extracted as a tagged enum, so:

- Per-task coverage metrics ("is this corpus light on `debug` probes?") can't be
  computed.
- Per-task adapter diversity can't be tuned ("more `critique` examples this
  rebuild").
- Bloom-axis-orthogonal training signals stay hidden.

The fix is a new axis: `cognitive_task_type`. Wave 60/69 precedent for Bloom
shows how to land a parallel axis additively (chunk + misconception + JSON-LD,
optional, normalized camelCase → snake_case, inferred from text with upstream
override). Same shape here.

## Taxonomy choice

The 60-verb canonical bloom_verbs.json union, narrowed to 15 pedagogically
distinct task verbs (one per cognitive task, each clearly different from the
others). Pulled from across the 6 Bloom buckets so the axis is genuinely
orthogonal to level:

| Task verb | Bloom-bucket home | Why distinct |
|-----------|-------------------|--------------|
| `define`    | remember   | Lexicographic recall, no inference. |
| `identify`  | remember   | Recognition / labeling within a set. |
| `explain`   | understand | Causal / process exposition. |
| `summarize` | understand | Compression / restatement. |
| `classify`  | understand | Assign to a category by criteria. |
| `compare`   | understand | Side-by-side similarity/difference. |
| `apply`     | apply      | Generic use-in-new-situation. |
| `compute`   | apply      | Numerical / formulaic derivation. |
| `analyze`   | analyze    | Decompose / examine components. |
| `debug`     | analyze    | (synonymous with `examine` for failure cases) — added because GPT feedback specifically called it out and it's the dominant CS task verb. |
| `predict`   | analyze    | Forward-inferential outcome. |
| `infer`     | analyze    | Backward-inferential cause. |
| `critique`  | evaluate   | Strength/weakness judgment. |
| `evaluate`  | evaluate   | Generic merit-based judgment. |
| `construct` | create     | Build / produce artifact. |

(`design` is dropped in favor of `construct` because it overlaps too much with
`construct`; `compose`, `formulate`, `invent` are dropped for the same reason.
`debug` is new — not in the bloom_verbs canonical list — added by GPT feedback
explicitly and seeded into the taxonomy.)

Final enum: `define | identify | explain | summarize | classify | compare |
apply | compute | analyze | debug | predict | infer | critique | evaluate |
construct`. 15 values.

## Schema choice

NEW file: `schemas/taxonomies/cognitive_task_type.json`, modeled on
`question_type.json` (simple `$defs/CognitiveTaskType` enum + `$ref` at root).
Not a bolt-on to `bloom_verbs.json` because the verbs there are bucketed by
Bloom level — folding a parallel axis in would conflate the two and break the
loader contract that `properties.<bloom>.default` is the verb list.

## Schema diffs

### 1. `schemas/knowledge/chunk_v4.schema.json`

- Add `cognitive_task_type` at chunk top-level (mirrors `bloom_level` placement,
  optional). Refs the new taxonomy file's `$defs/CognitiveTaskType`.
- Add `cognitive_task_type` inside `$defs.Misconception.properties` (mirrors
  the existing `bloom_level` field there, optional). Refs the same taxonomy.

Both fields are optional — legacy chunks (no field) continue to validate
because the chunk's `required` list isn't extended and the misconception
subobject doesn't add `cognitive_task_type` to its `required` array either.

### 2. `schemas/knowledge/courseforge_jsonld_v1.schema.json`

JSON-LD adds `cognitiveTaskType` (camelCase emit) on the misconception
sub-shape (mirroring the existing `bloomLevel` on misconceptions). Optional;
when absent the consumer continues as before. Trainforge's
`html_content_parser` normalizes camelCase → snake_case at parse time (see
threading section below).

### 3. Question artifact

`assessment_quality_report.build_assessment_dimension` reads per-question
records. Question artifacts emitted by Trainforge live in
`Trainforge/output/assessments.json` (per course) and individual question
records carry `question_type` already; adding a new optional
`cognitive_task_type` follows the same pattern.

**Decision:** the question artifact emit is gated behind a behavior flag
(`TRAINFORGE_COGNITIVE_TASK_TYPE`) for this wave to keep blast radius minimal.
The chunk + misconception fields land unconditionally because they're behind
the already-existing optional-field contract and don't change generation
behavior.

## Inference path

Helper: `lib/ontology/cognitive_task.py::detect_cognitive_task_type(text) ->
Optional[str]`. Same shape as `lib.ontology.bloom.detect_bloom_level`:

- Loads the 15-value enum from the taxonomy JSON once (lru_cache).
- Pre-computes verb iteration order: longest verb first, alphabetical for
  stable ordering (no level priority — there's no level for this axis).
- For each verb, whole-word match (`\b{verb}\b`) on lowered text.
- Returns the first verb-string match, or `None`.

Decision rationale: an LO/question/misconception stem typically leads with a
single primary task verb ("Classify the species…", "Compute the molarity…",
"Debug the recursion…"). The detector mirrors `detect_bloom_level`'s
"first whole-word match wins" contract; downstream consumers that want a
multi-task signal can call a future `detect_cognitive_task_types(text)`
(modeled on `detect_bloom_verbs`) — out of scope for this wave.

## Generator threading

### Chunk emit (`Trainforge/process_course.py::_create_chunk`)

Right after the bloom_level resolution block (around line 1907), call:

```python
task_type = detect_cognitive_task_type(text)
if task_type:
    chunk["cognitive_task_type"] = task_type
    capture.log_decision(
        decision_type="cognitive_task_type_detection",
        decision=f"Assigned cognitive_task_type={task_type}",
        rationale=(
            f"Heuristic match against chunk text prefix; verb={task_type!r} "
            f"surfaced via lib.ontology.cognitive_task.detect_cognitive_task_type "
            f"on a {len(text)}-char chunk. Adds an axis orthogonal to "
            f"bloom_level={chunk['bloom_level']!r} so downstream coverage "
            "validators can audit per-task diversity."
        ),
    )
```

`decision_type="cognitive_task_type_detection"` is a new value — add it to
`schemas/events/decision_event.schema.json`'s canonical enum so
`DECISION_VALIDATION_STRICT=true` doesn't fail-close. Skip emit when
`task_type is None` so legacy chunks without an action verb don't grow a null
field.

### Misconception emit (`Trainforge/parsers/html_content_parser.py`)

In the JSON-LD misconception loop (line 350-374), add:

```python
task = mc.get("cognitiveTaskType") or mc.get("cognitive_task_type")
if isinstance(task, str) and task:
    entry["cognitive_task_type"] = task
```

Mirrors the existing `bloomLevel`/`cognitiveDomain` camelCase normalization
arm. Trainforge can also fall back to detection over the correction statement
if neither is provided — but that's the chunk's emit-time call, not the
parser's job (parser stays pass-through).

### Question artifact emit (gated)

Trainforge `AssessmentGenerator` already attaches `bloom_level`. When
`TRAINFORGE_COGNITIVE_TASK_TYPE=true`, also attach `cognitive_task_type`
inferred from the question stem. Default off this wave (the generator surface
is large enough that opt-in protects existing corpora).

## Behavior flag

| Flag | Default | Purpose |
|------|---------|---------|
| `TRAINFORGE_COGNITIVE_TASK_TYPE` | unset | When on, Trainforge assessment generator additionally tags emitted questions with `cognitive_task_type` (the chunk + misconception fields land unconditionally because they're additive on optional schema slots). |

Add a row to `Trainforge/CLAUDE.md` § Opt-In Behavior Flags. Per the root
CLAUDE.md maintenance contract, **skip** the `docs/LICENSING.md` row — this
flag does not select a provider / model / synthesis backend.

## Test plan

`lib/ontology/tests/test_cognitive_task.py` (new file):

1. **Canonical detection.** A table of 15 stems, one per task verb, each
   expected to map to its task: "Classify the species…" → `classify`,
   "Compute the molarity…" → `compute`, etc.
2. **No-match returns None.** "no task verb here" → `None`.
3. **Whole-word match.** "classifier" (substring of `classify`) must NOT match.
4. **Longest-verb-first tie.** A stem with both `compute` (7 chars) and
   `apply` (5 chars) prefers `compute`.

`schemas/tests/test_chunk_v4_cognitive_task_type.py` (new file):

5. **Chunk validates with the field present.** Build a minimal chunk dict
   with `cognitive_task_type: "classify"`, validate against
   `chunk_v4.schema.json`, expect pass.
6. **Chunk validates with the field absent.** Same chunk without the field,
   expect pass (back-compat).
7. **Misconception validates with the field.** A chunk with
   `misconceptions[].cognitive_task_type: "explain"` passes.
8. **Invalid task value rejected.** `cognitive_task_type: "frobnicate"`
   fails validation (proves the enum is wired).

`Trainforge/tests/test_html_content_parser_cognitive_task.py` (new file):

9. **JSON-LD camelCase normalized.** A JSON-LD page block with
   `misconceptions: [{"misconception": "...", "correction": "...",
   "cognitiveTaskType": "debug"}]` parses to a dict carrying
   `cognitive_task_type: "debug"` (snake_case).

All tests must run via `pytest` and pass before commit.

## Validator follow-up (DEFERRED)

A `lib/validators/cognitive_task_coverage.py` gate would assess per-LO and
per-objective-coverage on the task-type axis (e.g. "no `debug` examples for
TO-05" warning, or "task-type distribution dominated by `explain`"
warning). Mirroring the W6.A per-question-type bucket pattern. Wave-scope for
this worker is NOT to implement the validator — only the data substrate.
File a follow-up in the GPT-feedback-may12 series.

## Out of scope

- Multi-task detection (`detect_cognitive_task_types`) — analogue to
  `detect_bloom_verbs`. Single-match is enough for the chunk emit; the helper
  module is structured so adding the plural form later is a one-function
  addition.
- Surfacing into `pedagogy_graph_builder` / `concept_graph` as edge metadata
  — defer until the chunk-level axis stabilizes across at least one corpus
  rebuild.
- The cross-axis interaction matrix (Bloom × task-type heatmaps in
  `eval_report.json`) — proposed but separate wave.

## Files touched

- ADD `schemas/taxonomies/cognitive_task_type.json`
- ADD `lib/ontology/cognitive_task.py`
- ADD `lib/ontology/tests/test_cognitive_task.py`
- ADD `schemas/tests/test_chunk_v4_cognitive_task_type.py`
- ADD `Trainforge/tests/test_html_content_parser_cognitive_task.py`
- EDIT `schemas/knowledge/chunk_v4.schema.json` (two field additions)
- EDIT `Trainforge/process_course.py::_create_chunk` (detection + decision capture)
- EDIT `Trainforge/parsers/html_content_parser.py` (camelCase → snake_case)
- EDIT `schemas/events/decision_event.schema.json` (new decision_type enum value)
- EDIT `Trainforge/CLAUDE.md` (flag table row)
