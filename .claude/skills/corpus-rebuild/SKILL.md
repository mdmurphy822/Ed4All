---
name: corpus-rebuild
description: Rebuild the rdf-shacl-551-2 reference corpus and run the standard verification pytest combo. Use to verify Trainforge/Courseforge changes against the canonical small corpus.
disable-model-invocation: true
---

# Corpus Rebuild

Rebuild the canonical small reference corpus (`rdf-shacl-551-2`) and run the
standard verification pytest combo. This is the loop used during Wave 85
plan-verification work: small enough to iterate quickly, broad enough to
exercise the Trainforge → LibV2 contract surface end-to-end.

## Reference corpus

- Source IMSCC: `LibV2/courses/rdf-shacl-551-2/source/imscc/RDF_SHACL_551.imscc`
- Canonical course code: `rdf-shacl-551-2`
- Why this one: smallest archived corpus that exercises the SHACL rule
  emit, concept-graph build, and graph-assisted retrieval surfaces. The
  Wave 82 regression class (zero-edge SHACL rules) shows up here first.

## Workflow

### 1. Discover the actual entry-point invocation

Do NOT assume the command line — orchestrator wiring drifts. Surface the
real entry point first:

```bash
grep -nE 'def main|argparse|--imscc|--output-dir|--reuse-objectives' \
  Trainforge/process_course.py | head -n 30
grep -rnE 'process_course\.py' MCP/ cli/ | head -n 20
# Cross-check the canonical CLI path
grep -nE 'rag_training|trainforge_assessment' cli/commands/run.py config/workflows.yaml | head -n 30
```

Surface the resolved command in the report before running it. If the wiring
is via `ed4all run rag_training ...`, prefer that over a direct
`process_course.py` call.

### 2. Rebuild into a tmp dir

Do not overwrite `LibV2/courses/rdf-shacl-551-2/` — write into a tmp dir and
diff. Sketch (substitute the real entry-point flags discovered in step 1):

```bash
TMP=$(mktemp -d)
python3 Trainforge/process_course.py \
  --imscc LibV2/courses/rdf-shacl-551-2/source/imscc/RDF_SHACL_551.imscc \
  --output-dir "$TMP" \
  # [--reuse-objectives <path-to-prior-synthesized_objectives.json>] if stable LO regen is needed
```

Or, via the canonical CLI (preferred when wired):

```bash
ed4all run rag_training \
  --corpus LibV2/courses/rdf-shacl-551-2/source/imscc/RDF_SHACL_551.imscc \
  --course-name rdf-shacl-551-2 \
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

Key regression check: SHACL rule edge counts in the concept graph. The
Wave 82 regression class manifested as zero-edge rules.

```bash
PRIOR=LibV2/courses/rdf-shacl-551-2/graph/concept_graph_semantic.json
NEW="$TMP/graph/concept_graph_semantic.json"

# Edge count by rule
jq '[.edges[] | select(.kind=="shacl_rule")] | length' "$PRIOR"
jq '[.edges[] | select(.kind=="shacl_rule")] | length' "$NEW"

# Or if the structure differs, inspect both shapes
jq 'keys' "$PRIOR" "$NEW"
diff <(jq -S . "$PRIOR") <(jq -S . "$NEW") | head -n 80
```

If the new edge count drops to zero (or sharply below the prior value), the
Wave 82 regression has reappeared. Stop and report.

### 5. Run the 8 retrieval-simulation queries

Pull the canonical query list from
`plans/rdf-shacl-enrichment-2026-04-26.md` (search for "retrieval
simulation" / "8 queries"). Run them through whatever retrieval entry-point
the plan documents (typically `LibV2/tools/libv2/retrieval/`); confirm the
graph-assisted hits still resolve. Surface the discovered command rather
than hardcoding it:

```bash
grep -rnE 'retrieval_simulation|retrieval-simulation|8 queries' \
  plans/ LibV2/tools/libv2/ | head -n 20
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
- Does not overwrite `LibV2/courses/rdf-shacl-551-2/` — always builds into
  a tmp dir.
- Does not invent CLI flags. If the entry-point grep doesn't find a flag,
  ask before guessing.
