# Pre-Wave-113 Baseline — rdf-shacl-551-2 Gates

_Run date: 2026-04-28T10:57:20-04:00_
_Commit: 1c1bab6d423bd8b2d4cb41f516b314129d6d949c_ (branch `dev-v0.3.0`)
_Specs evaluated:_
- `LibV2/courses/rdf-shacl-551-2/training_specs/instruction_pairs.jsonl` (mtime 2026-04-27T16:49:43-04:00, 885 pairs, 1,071,210 bytes)
- `LibV2/courses/rdf-shacl-551-2/training_specs/preference_pairs.jsonl` (mtime 2026-04-27T16:49:43-04:00, 254 pairs, 361,377 bytes)
- Snapshots `*.pre-wave113.bak` are byte-identical to the live files at run time (size match).

Validators executed: `synthesis_quota`, `min_edge_count`, `synthesis_diversity`,
`property_coverage`. Each was invoked in-process by importing the
class from `lib/validators/` and calling `validate(inputs)`. No
subprocess, no shell escaping. Results captured into `/tmp/baseline_results.json`
during the run.

## Summary

| Validator | Severity wiring | Outcome | Issues (critical / warning) |
|-----------|-----------------|---------|------------------------------|
| `synthesis_quota`     | warning  | PASS | 0 / 0 |
| `min_edge_count`      | critical | PASS | 0 / 0 |
| `synthesis_diversity` | critical | PASS (marginal) | 0 / 0 |
| `property_coverage`   | critical | **FAIL** | 1 / 0 |

Single failing gate is `property_coverage`. `synthesis_diversity` passes
but with score 0.5288 and a top-1 prefix bigram (`('week', 'co')`)
sitting at 13.9% — only 1.1 pp below the 15% top-1 ceiling. See
"Surprises" below.

## Per-validator detail

### synthesis_quota (severity: warning, validator v1.0.0)

- passed: **True**, score: 1.0
- inputs: `{ "course_dir": "LibV2/courses/rdf-shacl-551-2" }`
- issues: none
- Quota math (re-derived): 295 chunks, all 295 carry `learning_outcome_refs`
  → eligible=295. Default `instruction_variants_per_chunk=1` →
  estimated dispatches = 295 × (1 + 1) = **590**, well below the
  default 1500 ceiling. Validator returned `score=1.0` and no issues.
- notes: validator reads the corpus from `<course_dir>/corpus/chunks.jsonl`,
  so the `chunks.jsonl` we have on disk (3,317,286 bytes, 295 records)
  is what was sampled.

### min_edge_count (severity: critical, validator v1.0.0)

- passed: **True**, score: 1.0
- inputs: `{ pedagogy_graph_path: ".../graph/pedagogy_graph.json", concept_graph_path: ".../graph/concept_graph.json" }`
- issues: none
- Graph stats (re-derived):
  - pedagogy edges: 8,735 (floor 100 — well above)
  - distinct relation types: 13 (floor 4) — `assesses`,
    `assessment_validates_outcome`, `at_bloom_level`,
    `belongs_to_module`, `chunk_at_difficulty`,
    `concept_supports_outcome`, `derived_from_objective`,
    `exemplifies`, `follows`, `interferes_with`, `prerequisite_of`,
    `supports_outcome`, `teaches`
  - concept graph nodes: 672 (floor 50)
- notes: graphs at `LibV2/courses/rdf-shacl-551-2/graph/{pedagogy_graph,concept_graph}.json`.
  Under the rebuild, expect this gate to remain green — graph emit
  is upstream of the synthesis call and isn't the regression class
  Wave 113 targets.

### synthesis_diversity (severity: critical, validator v1.0.0)

- passed: **True (marginal)**, score: 0.5288
- inputs: `{ instruction_pairs_path: ".../training_specs/instruction_pairs.jsonl" }`
- issues: none — but every check is close to the threshold:
  - distinct templates: passes the `min_distinct_templates=8` floor.
  - top-3 template share: passes `max_top3_share=0.60` (score 0.5288 implies top-3 ≈ 47%).
  - top-1 template share: passes `max_single_share=0.35`.
  - **prefix-bigram top-1 share: 13.90%** (`('week', 'co')`, 123 of
    885 pairs). Threshold `max_prefix_top1_share=0.15`. **1.1 pp of
    margin.** Any small synthesis-output shift can flip this to a
    critical fail.
  - prefix-bigram top-3 share: 18.64% (threshold 30%). Wide margin.
- notes: The corpus is exhibiting the exact failure mode Wave 105
  was designed to catch — completion text starts with the same
  bigram in 1-in-7 pairs — but the threshold was tuned just above
  this corpus's characteristics. The validator does NOT flag it,
  yet the corpus is empirically poor (per the Wave 113 plan's
  diagnosis). Effective verdict from the gate is "barely-OK" while
  the corpus is "not OK".
- Top prefix bigrams (from CLI helper `python -m lib.validators.synthesis_diversity`):
  ```
  ('week', 'co'): 123 (13.9%)
  ('describe', 'the'): 21 (2.4%)
  ('apply', 'owl'): 21 (2.4%)
  ('interpret', 'datatyped'): 21 (2.4%)
  ('you', 'are'): 18 (2.0%)
  ```

### property_coverage (severity: critical, validator v1.0.0)

- passed: **False**, score: 0.9
- inputs: `{ course_dir: "LibV2/courses/rdf-shacl-551-2", course_slug: "rdf-shacl-551-2" }`
- issues (1):
  - `[critical] PROPERTY_COVERAGE_BELOW_FLOOR`: "Synthesis output is
    missing minimum coverage for 3 of 6 declared properties
    (`sh_nodeshape`: 3/8; `sh_propertyshape`: 3/8; `owl_sameas`:
    3/8). Re-run synthesis with a property-aware provider, or
    revise the manifest floors if intentional."
  - location: `LibV2/courses/rdf-shacl-551-2/training_specs/instruction_pairs.jsonl`
- notes: Manifest resolved via family slug `rdf_shacl` → loaded
  `schemas/training/property_manifest.rdf_shacl.yaml`. Three of six
  declared properties (the SHACL shape predicates and `owl:sameAs`)
  fall to **3/8** of their per-property floor. This is the exact
  regression class the Wave 113 plan calls out: the paraphrase
  pass dropped hard surface forms in favour of natural-English
  rewrites, leaving the SLM with no signal for `sh:NodeShape`,
  `sh:PropertyShape`, or `owl:sameAs`. The other three properties
  meet their floors.

## Comparison hooks

To diff against post-rebuild gates, e.g. once Wave 113 lands a
rebuilt corpus and re-runs the same four validators into a
companion file:

```
diff state/runs/2026-04-28-pre-wave113-baseline.md state/runs/<post-rebuild-baseline>.md
```

Key signals to watch on the post-rebuild file:
- `property_coverage` flips PASS, with `sh_nodeshape`,
  `sh_propertyshape`, `owl_sameas` all at-or-above their floors.
- `synthesis_diversity` score moves up (prefix-bigram top-1 share
  drops well below 13.9%) so the validator stops being "marginal".
- `synthesis_quota` and `min_edge_count` should remain PASS
  (regression of either would indicate upstream graph/chunk damage).

## Surprises / flags

1. **`synthesis_diversity` passes despite empirical poison.** The
   top-1 prefix bigram `('week', 'co')` is 13.9% of completions,
   1.1 pp under the configured `max_prefix_top1_share=0.15`. Wave 105
   tightened this threshold *because of* the exact phenomenon visible
   here ("the treatment of …" in an earlier corpus dominated 80% of
   outputs). The current 15% threshold catches only severe collapse;
   the current corpus shows mild collapse + the validator gives a
   green light. **Not a validator bug — a tuning gap.** Consider
   tightening to ~0.10 if Wave 113 confirms the prefix-collapse
   regression class needs to fail harder.
2. **`property_coverage` is the only critical-failing gate**, and
   it points cleanly at the documented Wave 113 regression
   (paraphrase pass dropped surface forms). The 3/8 ratio across
   three of six declared properties confirms the failure was
   uniform across SHACL shape vocabulary.
3. **`min_edge_count` is healthy** — pedagogy graph has 8,735
   edges across 13 relation types, concept graph has 672 nodes.
   Wave 113 rebuild should NOT touch the upstream graph emit; if
   the post-rebuild numbers shift materially that's a separate
   regression to investigate.
4. **No validator crashed.** All four imported and ran cleanly
   against the on-disk artifacts. No missing-file errors, no schema
   mismatches.

## Verdict

These specs are **NOT** in good enough shape to skip the Wave 113
rebuild. Evidence:

- `property_coverage` fails closed (critical) with three SHACL
  surface-form properties at 3/8 of their min-pair floor. Training
  on this corpus would produce a model that has never seen
  `sh:NodeShape`, `sh:PropertyShape`, or `owl:sameAs` in 5/8 of
  the expected pair contexts.
- `synthesis_diversity` passes by a 1.1 pp margin on the top-1
  prefix bigram. The corpus is qualitatively template-collapsed
  even though it formally clears the gate.
- The two PASSING gates (`synthesis_quota`, `min_edge_count`) are
  upstream-of-synthesis signals; their PASS status confirms the
  rebuild can re-use the existing chunks + graphs and only needs
  to redo the synthesis call.

The Wave 113 rebuild is justified. After it lands, re-running this
exact validator set should produce a companion baseline that
flips `property_coverage` to PASS and increases the
`synthesis_diversity` score (lower top-1 prefix-bigram share).
