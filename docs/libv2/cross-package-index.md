# Cross-package concept index

The **cross-package concept index** is a LibV2-level catalog artifact that records which concepts appear across which courses. It is produced by the indexer added in Worker G (`worker-g/cross-package-index`) and lives at:

```
LibV2/catalog/cross_package_concepts.json
```

The artifact is a **read-only derivative** of each course's `graph/concept_graph.json` (and, when present, Worker F's `graph/concept_graph_semantic.json`). Nothing in the retrieval pipeline or course-import pipeline mutates it; it is rebuilt on demand from the committed course graphs.

## Why it exists

When a caller issues a retrieval query against LibV2, they usually target one course. The cross-package index answers the adjacent question: **"given a concept from course X, which other courses in this repo cover the same concept?"** That lets a downstream tool (for example, a query expander or a cross-course evidence aggregator) widen its retrieval scope without crawling every course's chunk file.

The index does not replace retrieval. It is a *navigation* layer for concept IDs.

## How to build it

```bash
# From the repo root (or anywhere inside it — repo root is auto-detected).
python -m LibV2.tools.libv2.cli cross-index

# Explicit paths.
python -m LibV2.tools.libv2.cli cross-index \
  --repo-root /path/to/Ed4All \
  --output    /path/to/Ed4All/LibV2/catalog/cross_package_concepts.json
```

The subcommand is pure-Python, has no network calls, and takes under a second on the current course set.

A freshness check is wired into `lib/libv2_fsck.py`: when any course's `graph/concept_graph.json` has a newer mtime than the committed catalog, fsck reports a `stale_catalog` warning. If the catalog file does not exist, fsck is silent — not every repo needs a built index.

## How to read it

The top-level shape is:

```json
{
  "catalog_version": 1,
  "generated_at": "ISO-8601 UTC timestamp",
  "repo_root": "/absolute/path/to/Ed4All",
  "course_count": 5,
  "concept_count": 247,
  "concepts": { "<concept-id>": { ... } }
}
```

Concepts are keyed by their canonical id (e.g. `accessibility`, `screen-reader`) and **sorted by `total_courses` descending, then alphabetically** so the ranking is deterministic across runs on the same input.

### Field reference

**`catalog_version`**
Integer that bumps on any breaking shape change. Consumers should hard-fail on an unknown value rather than silently downgrading.

**`generated_at`**
UTC ISO-8601 timestamp of the build. This is the one non-deterministic field in the document; tests that want byte-stable comparisons strip it via `canonical_payload()` in `cross_package_indexer.py`.

**`repo_root`**
Absolute path the indexer resolved against at build time. Useful for debugging which checkout produced a given artifact; never load-bearing for correctness.

**`course_count`**
Number of courses under `LibV2/courses/` that had a readable `graph/concept_graph.json`. Courses without a graph are silently skipped — they contribute nothing to the index but do not cause a failure.

**`concept_count`**
Number of distinct concept ids observed across all included courses. Equal to `len(concepts)`.

**`concepts.<id>.label`**
Human-readable label for the concept, taken from the first course that supplied one. Labels are informational; consumers should match on `id`.

**`concepts.<id>.total_courses`**
Count of courses in which this concept id appeared with any non-zero frequency.

**`concepts.<id>.courses[]`**
Per-course presence list, sorted alphabetically by `slug`. Each entry carries `{slug, frequency, label}` where `frequency` is the raw per-course count from that course's `concept_graph.json`.

**`concepts.<id>.cross_package_edges[]`**
Typed edges (from Worker F's `concept_graph_semantic.json`, when present) that originate at this concept and target another concept that is **also shared across at least two courses**. Each entry carries `{source_concept, target_concept, type, course_slug, confidence?, weight?}`. Edges whose endpoints are single-course concepts are filtered out — by definition they do not cross package boundaries.

When none of the included courses carry a semantic graph (the case for any course built before Worker F landed), `cross_package_edges` is an empty list on every concept. That is a graceful degradation, not an error.

## Example use case

> Given a retrieval query against `foundations-of-digital-pedagogy` that returned the concept `accessibility`, identify which other LibV2 courses also cover that concept and what typed relationships connect it to sibling concepts.

```python
import json

with open("LibV2/catalog/cross_package_concepts.json") as f:
    index = json.load(f)

entry = index["concepts"].get("accessibility")
if entry is None:
    neighbours = []
else:
    neighbours = [c["slug"] for c in entry["courses"]
                  if c["slug"] != "foundations-of-digital-pedagogy"]

# neighbours -> list of other course slugs to consider for a widened retrieval.
```

If `entry["cross_package_edges"]` is non-empty, the caller can additionally traverse typed relationships (`related-to`, `is-a`, `prerequisite`) to pick sibling concepts worth retrieving alongside the primary hit.

## Non-goals

- **No LLM involvement.** The index is a pure aggregation of committed JSON.
- **No chunk-schema change.** Worker G adds a catalog artifact; it does not touch chunk files or their schema version.
- **No retrieval-engine change.** Consumers read the JSON; the retriever itself is unchanged.
- **No cross-repo federation.** The index covers one `LibV2/courses/` tree per build.
