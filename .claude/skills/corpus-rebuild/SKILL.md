---
name: corpus-rebuild
description: Rebuild a reference corpus and run the standard verification pytest combo. Use to verify Trainforge/Courseforge changes against a small archived corpus. Pass the course slug as the argument.
disable-model-invocation: true
---

# Corpus Rebuild

Rebuild a canonical small reference corpus and run the standard
verification pytest combo. This is the loop used during plan-verification
work: small enough to iterate quickly, broad enough to exercise the
Trainforge → LibV2 contract surface end-to-end.

## Arguments

- `$ARGUMENTS` (or the first positional argument) — the **course slug**
  to rebuild, e.g. `<course-slug>`. Resolve it once at the top and use it
  everywhere below as `SLUG`. Pick the smallest archived corpus that
  exercises the surfaces your change touches (rule emit, concept-graph
  build, graph-assisted retrieval, etc.).

```bash
SLUG="${ARGUMENTS:?pass the course slug as the skill argument}"
COURSE_DIR="LibV2/courses/$SLUG"
ls "$COURSE_DIR/source/imscc/"   # discover the source IMSCC filename
```

## Reference corpus

- Source IMSCC: `LibV2/courses/<slug>/source/imscc/<COURSE>.imscc`
- Canonical course code: the `$SLUG` you passed in
- How to choose: the smallest archived corpus that exercises the rule
  emit, concept-graph build, and graph-assisted retrieval surfaces. The
  RDF/SHACL calibration corpus is the canonical example — the zero-edge
  rule-emit regression class shows up there first.

## Workflow

### 1. Discover the actual entry-point invocation

Do NOT assume the command line — orchestrator wiring drifts. Surface the
real entry point first:

```bash
grep -nE 'def main|argparse|--imscc|--output-dir|--reuse-objectives' \
  Trainforge/pipeline/process_course.py | head -n 30
grep -rnE 'process_course\.py' MCP/ cli/ | head -n 20
# Cross-check the canonical CLI path
grep -nE 'rag_training|trainforge_assessment' cli/commands/run.py config/workflows.yaml | head -n 30
```

Surface the resolved command in the report before running it. If the wiring
is via `ed4all run rag_training ...`, prefer that over a direct
`process_course.py` call.

### 2. Rebuild into a tmp dir

Do not overwrite `LibV2/courses/<slug>/` — write into a tmp dir and
diff. Sketch (substitute the real entry-point flags discovered in step 1
and the source IMSCC filename discovered above):

```bash
TMP=$(mktemp -d)
IMSCC=$(ls "$COURSE_DIR"/source/imscc/*.imscc | head -n 1)
python3 -m Trainforge.pipeline.process_course \
  --imscc "$IMSCC" \
  --output-dir "$TMP" \
  # [--reuse-objectives <path-to-prior-synthesized_objectives.json>] if stable LO regen is needed
```

Or, via the canonical CLI (preferred when wired):

```bash
ed4all run rag_training \
  --corpus "$IMSCC" \
  --course-name "$SLUG" \
  --mode local
```

If `ANTHROPIC_API_KEY` is unset, prefer `--mode local`.

### 3. Run the verification pytest combo

```bash
pytest Trainforge/tests/ \
       lib/validators/tests/ \
       lib/ontology/tests/ \
       LibV2/tools/libv2/tests/ \
       schemas/tests/ \
       -x --tb=short
```

`-x` stops on first failure (faster iteration); drop `-x` for a full sweep.

### 4. Diff the regenerated concept graph

Key regression check: rule edge counts in the concept graph. The
canonical regression class manifested as zero-edge rules.

```bash
PRIOR="$COURSE_DIR/graph/concept_graph_semantic.json"
NEW="$TMP/graph/concept_graph_semantic.json"

# Edge count by rule
jq '[.edges[] | select(.kind=="shacl_rule")] | length' "$PRIOR"
jq '[.edges[] | select(.kind=="shacl_rule")] | length' "$NEW"

# Or if the structure differs, inspect both shapes
jq 'keys' "$PRIOR" "$NEW"
diff <(jq -S . "$PRIOR") <(jq -S . "$NEW") | head -n 80
```

If the new edge count drops to zero (or sharply below the prior value), the
zero-edge regression has reappeared. Stop and report.

### 5. Run the retrieval-simulation queries

Pull the canonical query list from the corpus's own enrichment notes /
retrieval-simulation fixtures (search the repo for "retrieval simulation"
near the retrieval tooling). Run them through whatever retrieval
entry-point is documented (typically `LibV2/tools/libv2/retrieval/`);
confirm the graph-assisted hits still resolve. Surface the discovered
command rather than hardcoding it:

```bash
grep -rnE 'retrieval_simulation|retrieval-simulation' \
  LibV2/tools/libv2/ | head -n 20
```

### 6. Report

Surface in the final report:

- The exact entry-point command that was run (resolved in step 1).
- Pytest pass/fail counts per test directory.
- Concept-graph edge-count delta (prior vs new).
- Retrieval-simulation pass/fail per query.
- Any drift worth flagging to the human.

## What this skill does NOT do

- Does not commit the regenerated corpus.
- Does not overwrite `LibV2/courses/<slug>/` — always builds into a tmp
  dir.
- Does not invent CLI flags. If the entry-point grep doesn't find a flag,
  ask before guessing.
