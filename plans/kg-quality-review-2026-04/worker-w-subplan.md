# Worker W sub-plan — Wave 6 schema + docs cleanup

Scope: REC-PRV-02 (evidence discriminator) + REC-JSL-04 (emit-only attribute removal) + agent-doc sweep + ONTOLOGY refresh. All four sit in disjoint files from Worker V's workflow-governance work.

Branch: `worker-w/wave6-schema-docs-cleanup` off `dev-v0.2.0`.

---

## 1. REC-PRV-02 — per-rule evidence discriminator

### Current state (verified against tree)

`schemas/knowledge/concept_graph_semantic.schema.json` already declares 8 edge types at `type.enum`: `prerequisite | is-a | related-to | assesses | exemplifies | misconception-of | derived-from-objective | defined-by` (lines 66–74). `provenance.evidence` is currently `{type: object, additionalProperties: true}` (line 91) — a free-form object with no per-rule shape.

The 8 rule emitters in `Trainforge/rag/inference_rules/*.py` each emit a specific `provenance.evidence` shape. I re-read each rule module and canonicalized the shapes below.

### Evidence shape inventory (verified from rule modules)

| Rule | Type | Evidence fields | Notes |
|------|------|-----------------|-------|
| `is_a_from_key_terms` | `is-a` | `chunk_id` (string), `term` (string), `definition_excerpt` (string ≤ 200 chars), `pattern` (string, regex pattern used) | `is_a_from_key_terms.py:205–214` |
| `prerequisite_from_lo_order` | `prerequisite` | `target_first_lo` (string), `target_first_lo_position` (int), `source_first_lo` (string), `source_first_lo_position` (int) | `prerequisite_from_lo_order.py:144–157` |
| `related_from_cooccurrence` | `related-to` | `cooccurrence_weight` (int), `threshold` (int) | `related_from_cooccurrence.py:75–78` |
| `derived_from_lo_ref` | `derived-from-objective` | `chunk_id` (string), `objective_id` (string) | `derived_from_lo_ref.py:69–73` |
| `defined_by_from_first_mention` | `defined-by` | `chunk_id` (string), `concept_slug` (string), `first_mention_position` (int, always 0) | `defined_by_from_first_mention.py:91–96` |
| `exemplifies_from_example_chunks` | `exemplifies` | `chunk_id` (string), `concept_slug` (string), `content_type` (string, one of `content_type_label` / `chunk_type`) | `exemplifies_from_example_chunks.py:100–107` |
| `misconception_of_from_misconception_ref` | `misconception-of` | `misconception_id` (string), `concept_id` (string) | `misconception_of_from_misconception_ref.py:80–84` |
| `assesses_from_question_lo` | `assesses` | `question_id` (string), `objective_id` (string), optional `source_chunk_id` (string) | `assesses_from_question_lo.py:70–77` |

Plus one fallback:

- `FreeFormEvidence` — `{type: "object", additionalProperties: true}`. Lenient; preserves backward-compat for any future rule that isn't yet modeled and for legacy fixtures.

### Discriminator design

Inline `$defs` inside `concept_graph_semantic.schema.json` (single file; no separate folder). Rationale: 8 sub-schemas at ~8 lines each = ~64 lines. Well below the "bloat" threshold where a separate directory pays off. Keeps validation self-contained.

Each specific `$def` sets `additionalProperties: false` so unknown fields on a known shape fail validation (in strict mode). `FreeFormEvidence` stays permissive.

`provenance.evidence.oneOf` list order matches the review / plan directives: `[IsA, Prerequisite, Related, Assesses, Exemplifies, MisconceptionOf, DerivedFromObjective, DefinedBy, FreeForm]`.

### Lenient vs strict

**Default: lenient.** The free-form `FreeFormEvidence` arm in the `oneOf` matches any evidence object. This preserves backward-compat with legacy graphs and with rules whose evidence predated this wave (e.g. older LibV2 regenerations). Keeping it lenient matches the Waves 1–5 opt-in policy.

**Strict (opt-in): `TRAINFORGE_STRICT_EVIDENCE=true`.** Not a schema-level flag — schemas don't read env vars. Instead we gate the strict mode at validation time: consumers that want strict validation (e.g. a future `test_evidence_strict` gate) remove `FreeFormEvidence` from the `oneOf` before validating. For this wave the schema ships the lenient `oneOf`. The strict mode is documented in ONTOLOGY.md's opt-in flag table; actual validator implementation can land later without further schema churn.

Concrete implementation: `lib/validators/evidence.py` exposes `get_schema(strict: bool = False)` that:
1. Loads the schema from `schemas/knowledge/concept_graph_semantic.schema.json`.
2. When `strict=True` or `os.environ.get("TRAINFORGE_STRICT_EVIDENCE") == "true"`, deep-copies the schema and strips the last `oneOf` arm (`FreeFormEvidence`) from `provenance.evidence`.
3. Returns the (possibly modified) schema dict.

This is the minimal plumbing; no validator callsites are retrofitted in this wave. The test file calls `get_schema()` explicitly with `strict=True` to exercise strict mode.

### New test file

`lib/tests/test_evidence_discriminator.py`:

- Per-rule happy-path (8): synth an edge with matching `rule` + correct evidence shape → validates.
- Per-rule mismatched-evidence rejects in strict mode (8): same `rule` but wrong evidence (extra field or wrong type) → strict rejects; lenient passes (falls through to FreeForm).
- FreeForm fallback passes: unknown-rule edge with arbitrary evidence shape → lenient passes; strict rejects (when FreeForm is stripped, no arm matches since the unknown rule's evidence doesn't match any specific `$def`).
- Load any existing `concept_graph_semantic.json` from LibV2 courses and validate — smoke-level check. No LibV2 corpora exist in this worktree (confirmed via `ls LibV2/courses/`); test skips gracefully if absent.

Test count: ~17 cases. Target: all pass.

---

## 2. REC-JSL-04 — remove `data-cf-objectives-count`; audit siblings

### Target line

`Courseforge/scripts/generate_course.py:332` — the `data-cf-objectives-count="{len(objectives)}"` attribute inside `_render_objectives`. Confirmed: line 332 (no drift).

### Consumer audit

`Trainforge/parsers/html_content_parser.py` is the canonical consumer of `data-cf-*` attributes. Grep pattern `data-cf-[a-z-]+` across emit (`generate_course.py`) and consume (`html_content_parser.py`) sides yields:

**Emitted (14 unique):**
bloom-level, bloom-range, bloom-verb, cognitive-domain, component, content-type, key-terms, objective-id, objective-ref, objectives-count, purpose, role, teaching-role, term.

**Consumed (9 unique):**
bloom-level, bloom-verb, cognitive-domain, content-type, key-terms, objective-id, objective-ref, role, teaching-role.

**Emit-only candidates (5):**
`bloom-range`, `component`, `purpose`, `term`, `objectives-count`.

### Decision per emit-only

| Attribute | Decision | Rationale |
|-----------|----------|-----------|
| `data-cf-objectives-count` | **REMOVE** | Count of a list that's already emitted in full right below. Zero KG value; flagged by review. |
| `data-cf-bloom-range` | KEEP | Emitted on `<h2>`/`<h3>` to record the span of Bloom levels addressed by a section. Future consumer intent for section-level bloom analytics; also appears in Trainforge test fixtures. |
| `data-cf-component` | KEEP | Covered by `Trainforge/tests/test_teaching_role_emit.py:64,85,101` + `test_activity_objective_ref.py`. Structural discriminator (flip-card / self-check / activity) used during teaching-role validation. |
| `data-cf-purpose` | KEEP | Same — paired with `data-cf-component` in the same tests. Pedagogical-purpose discriminator. |
| `data-cf-term` | KEEP | Carries the term slug on flip cards (`test_teaching_role_emit.py:286,290`). Gives downstream consumers a deterministic term→card mapping. |

Only `objectives-count` removed this wave. Other 4 stay — they have either test fixtures asserting them or clear semantic utility. Documented here so future reviews have traceable rationale.

### Test impact

One grep in `Trainforge/tests/test_metadata_extraction.py:77` references `data-cf-objectives-count` in an inline HTML fixture. The fixture's parse path doesn't currently read the attribute (it's not in html_content_parser consumer list). Removing from the fixture is a cosmetic change — but safest to leave the fixture intact since it doesn't break anything and represents "example real HTML" (tests parse for other attrs). Alternative: strip from fixture too, for cleanliness. **Decision: strip from fixture** — matches the intent of the removal and keeps fixtures truthful to emit-side reality.

---

## 3. Agent-doc sweep

### Grep results (executed)

`grep -rn 'schemas/' Courseforge/agents/*.md` ran; results show only canonical paths:

- `course-outliner.md:294`: `schemas/academic/learning_objectives.schema.json` — canonical, keep.
- `textbook-ingestor.md:210`: `schemas/academic/textbook_structure.schema.json` — canonical, keep.
- `objective-synthesizer.md:30`: `schemas/academic/textbook_structure.schema.json` — canonical, keep.

`grep -rn 'Courseforge/schemas/'` → 0 matches (no stale pre-unification paths).

`grep -rin 'BLOOM_VERBS\|slugify'` → only one hit:
- `content-quality-remediation.md:158–171` — `bloom_verbs = {...}` inline dict inside a `generate_learning_objectives` example, with a TODO comment already pointing to `schemas/taxonomies/bloom_verbs.json`.

The "hot spot" the master plan flags at `content-quality-remediation.md:159` is this TODO comment. It currently says `load from schemas/taxonomies/bloom_verbs.json at orchestrator templating layer`. The canonical loader is now `lib/ontology/bloom.py` (not just the raw JSON). Fix: update the TODO to reference the loader.

**Minimal prose edits (single file):**

- `Courseforge/agents/content-quality-remediation.md:159` — update TODO comment from `load from schemas/taxonomies/bloom_verbs.json at orchestrator templating layer` to `load from lib.ontology.bloom.get_verbs_list() (canonical loader over schemas/taxonomies/bloom_verbs.json)`.

No other stale references found. Sweep is surgical; don't rewrite agent behavior.

### Optional drive-by — `data-cf-objectives-count` references in Courseforge/

Not flagged in agent docs (grep over `Courseforge/agents/*.md` for that string yields 0). No edits needed there.

---

## 4. ONTOLOGY.md refresh

Append a new section `## § 17 v0.2.0 changes (Waves 1–6 summary)` at the end of `schemas/ONTOLOGY.md`. ~150 lines. **Additive only — do not rewrite existing sections.**

### Structure (subsection inventory)

1. **Wave overview** — brief table: wave, scope, representative PRs, KG impact.
2. **Taxonomies added** (`schemas/taxonomies/`) — list: bloom_verbs, question_type, assessment_method, content_type, cognitive_domain, teaching_role, module_type.
3. **Knowledge schemas added** — courseforge_jsonld_v1, chunk_v4, misconception, instruction_pair.strict (opt-in).
4. **Config meta-schema** — `workflows_meta.schema.json` (landed by Worker V in same wave; reference-only).
5. **6-value `moduleType`** — surfaced by Worker B, schematized by Worker F; note the added `discussion` value.
6. **8 edge types** — 3 taxonomic (is-a, prerequisite, related-to) + 5 pedagogical (assesses, exemplifies, misconception-of, derived-from-objective, defined-by).
7. **First-class `Misconception` entity** — content-hash `mc_<16hex>` IDs, optional `concept_id` / `lo_id` links.
8. **`occurrences[]`** — back-reference on concept nodes.
9. **Opt-in flags table** — 7 flags: TRAINFORGE_CONTENT_HASH_IDS, TRAINFORGE_SCOPE_CONCEPT_IDS, TRAINFORGE_PRESERVE_LO_CASE, TRAINFORGE_VALIDATE_CHUNKS, TRAINFORGE_ENFORCE_CONTENT_TYPE, TRAINFORGE_STRICT_EVIDENCE (introduced this wave), DECISION_VALIDATION_STRICT.
10. **Always-emit provenance** — `run_id`, `created_at` on chunks, concept nodes, concept edges.
11. **Canonical helpers** (`lib/ontology/`) — slugs.canonical_slug, bloom loader, teaching_roles mapper, taxonomy loader.
12. **Validators consolidated** (`lib/validators/`) — page_objectives, content_type.
13. **New validation gates** — page_objectives (packaging), dart_markers (dart_conversion, landing alongside Worker V).
14. **Decision type enum** — 44 values (was 39 pre-Wave-1-G).

Each subsection is 5–15 lines. No embedded code. Cross-references to existing `§ N` sections where an item is elaborated.

### Style

Match existing ONTOLOGY.md voice: descriptive ("what exists now"), not prescriptive ("what should be"). No implementation plans; cross-link to individual wave sub-plans via footer when relevant.

---

## 5. `schemas/README.md` small update

Add a tiny mention of the new `schemas/config/` folder (arriving in same wave via Worker V) and expand the `knowledge/` entry to list all current files (chunk_v4, courseforge_jsonld_v1, misconception, instruction_pair.schema.json, instruction_pair.strict.schema.json, preference_pair). Keep it surgical — the current README is ~90 lines; I'll add maybe 10–15.

---

## 6. Coordination

- Worker V's files: `schemas/config/*`, `MCP/core/workflow_runner.py`, `config/workflows.yaml`, `MCP/tools/pipeline_tools.py`. Disjoint from mine.
- README.md is shared in a trivial sense — I only mention the `config/` folder existence, using the same phrasing as other subfolder entries. V creates the file; I reference it. No merge conflict.
- ONTOLOGY.md v0.2.0 section mentions `workflows_meta.schema.json` as a reference-only pointer to V's work. No overlap.

---

## 7. Verification plan

```bash
# 1. CI integrity
python3 -m ci.integrity_check

# 2. New discriminator tests
pytest lib/tests/test_evidence_discriminator.py -x

# 3. Existing typed-edge tests still pass (regression proof the discriminator
#    accepts current rule outputs)
pytest Trainforge/tests/test_pedagogical_edges.py Trainforge/tests/test_typed_edge_inference.py -x

# 4. Full suite
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ LibV2/tools/libv2/tests/ -q

# 5. Smoke: validate any existing semantic graph (none exist in worktree; loop is no-op)
for f in LibV2/courses/*/graph/concept_graph_semantic.json; do
  python3 -c "import json, jsonschema; s=json.load(open('schemas/knowledge/concept_graph_semantic.schema.json')); d=json.load(open('$f')); jsonschema.validate(d, s); print('OK: $f')"
done
```

Acceptance: 8/8 CI gates green, discriminator tests pass, existing tests pass (no regressions), full suite ≥967 (Wave 5 baseline 962 + ~5 from new test file).

---

## 8. File change summary

**Modified (5):**
- `schemas/knowledge/concept_graph_semantic.schema.json` — evidence `oneOf` discriminator + 9 `$defs`
- `Courseforge/scripts/generate_course.py` — drop `data-cf-objectives-count="..."` at L332
- `Courseforge/agents/content-quality-remediation.md` — TODO comment refresh on L159
- `schemas/ONTOLOGY.md` — append § 17 v0.2.0 changes
- `schemas/README.md` — small structural update
- `Trainforge/tests/test_metadata_extraction.py` — drop removed attr from inline fixture

**New (2):**
- `lib/tests/test_evidence_discriminator.py` — new test suite
- `lib/validators/evidence.py` — thin strict-mode schema loader

**Total:** 6 modified + 2 new = 8 files. Effort: M. Zero file-level overlap with Worker V.
