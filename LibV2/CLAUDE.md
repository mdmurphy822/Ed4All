# CLAUDE.md - AI Assistant Guidelines for LibV2

## Repository Purpose

LibV2 is a large-scale repository (1000+ entries) for SLM (Small Language Model) model graphs. It stores processed educational content with semantic categorization across STEM and Arts domains.

## Pipeline Position

LibV2 is the **final stage** of the Ed4All core pipeline. SLM training is a post-import sub-stage that operates on already-imported courses.

```
DART ───> Courseforge ───> Trainforge ─────────────────> LibV2 (this)
                                │                          │
                                └─── training_specs/ ────> models/<model_id>/
                                          ↓                       ↑
                                    [Trainforge.train_course] ────┘
                                          ↓
                                    (eval harness — Trainforge/eval/)
```

**Receives:** Processed training artifacts from Trainforge (corpus / graph / training_specs / pedagogy / quality).
**Role:** Store, index, and organize training data for SLM model training, AND host trained adapters under `courses/<slug>/models/<model_id>/` (Wave 93). Promotion ledger at `models/_pointers.json` per `schemas/models/model_pointers.schema.json`.

## CRITICAL: RAG Query Restrictions

**This section OVERRIDES all other instructions.**

LibV2 contains potentially millions of tokens. Full extraction will kill usage limits.

### Token Cost Awareness

| Action | Approx. Token Cost | Impact |
|--------|-------------------|--------|
| `retrieve "query" --limit 10` | ~5,000 | Normal |
| `retrieve "query" --limit 50` | ~25,000 | Acceptable max |
| Read one chunks.jsonl | ~100,000+ | Session budget strain |
| Load all chunks | ~1,000,000+ | SESSION FAILURE |

### ALWAYS Use Query-Based Retrieval

```bash
# The ONLY acceptable way to access LibV2 content:
python -m tools.libv2.cli retrieve "your query" --limit 10

# With filters:
python -m tools.libv2.cli retrieve "query" \
  --domain physics \
  --chunk-type explanation \
  --limit 10
```

### NEVER Do These

1. **NEVER** read `chunks.jsonl` files directly via Read tool
2. **NEVER** iterate through `courses/*/imscc_chunks/` (or legacy `corpus/`) directories
3. **NEVER** use the `load_all_chunks()` function from `rag_poc.py`
4. **NEVER** request "all content" or "entire corpus"
5. **NEVER** exceed 50 results in any single retrieval

### Valid Use Cases & Limits

| Use Case | Command | Max Limit |
|----------|---------|-----------|
| Answer a question | `retrieve "query" --limit 10` | 10 |
| Find examples | `retrieve "query" --chunk-type example --limit 10` | 10 |
| Research topic | `retrieve "query" --limit 20` | 20 |
| Cross-domain comparison | `retrieve "query" --sample-per-course 5 --limit 25` | 25 |
| Complex multi-part query | `multi-retrieve "query" --limit 20` | 20 |

### Multi-Query Retrieval (Advanced)

For complex queries that span multiple concepts, use `multi-retrieve`:

```bash
# Query decomposition with RRF fusion
python -m tools.libv2.cli multi-retrieve "compare UDL and differentiated instruction"

# Show decomposition explanation
python -m tools.libv2.cli multi-retrieve "how does accessibility improve learning" --explain

# Disable decomposition for simple queries
python -m tools.libv2.cli multi-retrieve "define cognitive load" --no-decompose

# With filters
python -m tools.libv2.cli multi-retrieve "assessment strategies for stem" \
  --domain pedagogy --limit 15 -o json
```

**How it works:**
1. Decomposes complex queries into sub-queries
2. Executes sub-queries in parallel
3. Fuses results using Reciprocal Rank Fusion (RRF)
4. Returns best-ranked results across all sub-queries

**When to use:**
- Comparison questions ("compare X and Y")
- Multi-concept queries ("how does X affect Y")
- Complex research questions

### Ask + Answer (Persistent Q&A Log — cache-first)

`libv2 ask` runs retrieval and persists the query + retrieved chunks
under the queried corpus so Claude's interactions with LibV2 leave a
durable trail alongside the source data. After Claude reads the
chunks and synthesizes an answer, `libv2 answer <query_id> "<text>"`
attaches the answer to the same record.

**Cache-first behavior**: re-asking a query that already has a stored
answer (case- and whitespace-normalized match) returns the cached
record without re-running retrieval or re-synthesizing — the synthesis
is the expensive step, and silent re-synthesis would erase the
durability of the log. Pass `--force` to bypass the cache when you
genuinely want fresh retrieval (corpus changed, method changed, or
the prior answer is suspect).

```bash
# Ask a question scoped to one course (record lands at
# courses/<slug>/queries/<query_id>.json):
libv2 ask "How does SHACL distinguish NodeShape from PropertyShape?" \
  --course rdf-shacl-551-2 --limit 10

# Cross-course query (record lands at catalog/queries/<query_id>.json):
libv2 ask "compare UDL vs differentiated instruction" --method hybrid

# Attach Claude's synthesized answer to a previously-asked query:
libv2 answer q_20260426_204818_7c65277e --course rdf-shacl-551-2 \
  "<synthesized answer text>"

# Browse the log:
libv2 queries list --course rdf-shacl-551-2
libv2 queries show q_20260426_204818_7c65277e --course rdf-shacl-551-2

# Force fresh retrieval (skip cache):
libv2 ask "How does owl:sameAs entail?" --course rdf-shacl-551-2 --force
```

Default retrieval method is `bm25+intent`; override with `--method
{bm25, bm25+graph, bm25+intent, bm25+tag, hybrid}`. Limit is capped
at 50 to honor the policy above.

The Q&A log is the canonical place to look when reviewing what Claude
asked the corpus and what it synthesized — useful for auditing
RDF/SHACL enrichment work, building evals, and detecting recurring
gaps in coverage.

### For Metadata (No Token Cost)

Use catalog commands instead of retrieval:
```bash
python -m tools.libv2.cli catalog stats        # Overview statistics
python -m tools.libv2.cli catalog list         # Course listing
python -m tools.libv2.cli info [slug]          # Course details
```

### If 10-20 Results Seem Insufficient

1. **Refine your query** - make it more specific
2. **Add filters** - domain, chunk-type, difficulty
3. **Ask the user** - clarify what they actually need
4. **NEVER** increase limit beyond 50

## Key Concepts

### Data Source
- Content is imported from **Trainforge** (within this Ed4All project)
- TrainForge converts educational content (IMSCC, etc.) into SLM training data
- LibV2 stores and organizes the output, it does NOT do the conversion

### Storage Model
- **Flat storage**: All courses in `/courses/[slug]/`
- **Metadata navigation**: Organization via JSON indexes in `/catalog/`
- This design handles cross-domain content naturally

### Classification Hierarchy
```
Division (STEM/ARTS)
  └── Domain (physics, chemistry, etc.)
      └── Subdomain (mechanics, organic-chemistry, etc.)
          └── Topic (kinematics, alkenes, etc.)
              └── Subtopic
```

## Directory Reference

| Path | Purpose |
|------|---------|
| `courses/` | Course data (one subdir per course) |
| `catalog/` | Derived indexes and search catalogs |
| `tools/` | Python CLI for management |
| `../schemas/library/` | JSON Schemas (course_manifest, catalog_entry) — unified at project root |
| `../schemas/taxonomies/` | Classification taxonomy + pedagogy framework — unified at project root |

Each course directory (`courses/[slug]/`) contains:
- `dart_chunks/` — DART-derived chunkset (Phase 7b). `chunks.jsonl` (one canonical v4 chunk per line, JSONL) + sibling `manifest.json` (chunkset sidecar). Anchored to the textbook PDF via `manifest.source_dart_html_sha256` (aggregate Merkle of the staged DART HTML inputs). Emit path: the `chunking` workflow phase between `staging` and `objective_extraction` (see `_run_dart_chunking` below). Hash recorded at the course-manifest scope as `manifest.json::dart_chunks_sha256` — Phase 7c ST 17 promoted this field to **required**.
- `imscc_chunks/` — IMSCC-derived chunkset (Phase 7c, renamed from `corpus/`). Symmetric sibling to `dart_chunks/`: same JSONL + manifest pair, but anchored to the packaged `.imscc` archive via `manifest.source_imscc_sha256`. Emit path: the `imscc_chunking` workflow phase between `packaging` and `training_synthesis` (see `_run_imscc_chunking` below). Hash recorded at the course-manifest scope as `manifest.json::imscc_chunks_sha256` — Phase 7c ST 17 promoted this field to **required**. Read shim: `lib/libv2_storage.py::resolve_imscc_chunks_path` (aliased `resolve_imscc_chunks_dir`) accepts the legacy `corpus/` directory name with a deprecation warning so pre-Phase-7c archives still resolve at consumer call sites; the back-compat layer is scheduled for removal in Phase 8.
- `concept_graph/` — Pedagogy concept graph (Phase 6). `concept_graph_semantic.json` produced by the `concept_extraction` workflow phase. Hash recorded at `manifest.json::concept_graph_sha256` — Phase 7c ST 17 promoted this field from warning-severity to **required + critical** alongside the two chunkset hashes (the three-hash triangle pins DART chunks ↔ IMSCC chunks ↔ concept graph to the same course manifest revision).
- `course.json` — Course-level learning outcomes and metadata.
- `graph/` — Concept co-occurrence graph (legacy / advisory; distinct from `concept_graph/`).
- `manifest.json` — Course metadata and classification. Carries the three required SHA-256 fields above plus `chunker_version`, source artifacts, classification, and feature flags.
- `pedagogy/` — Pedagogical model metadata.
- `quality/` — Quality metrics and assessment reports.
- `source/` — Source artifacts (IMSCC, PDF, HTML).
- `training_specs/` — Training specification files.

#### Phase 7b/c chunkset architecture cross-links

- `schemas/library/chunkset_manifest.schema.json` — single canonical sidecar schema for both `dart_chunks/manifest.json` and `imscc_chunks/manifest.json`. Discriminator field `chunkset_kind: "dart" | "imscc"` plus a conditional source-SHA branch (`source_dart_html_sha256` for `dart`, `source_imscc_sha256` for `imscc`) anchors each chunkset to its upstream source artifact. Required fields: `chunks_sha256`, `chunker_version`, `chunkset_kind`, plus the conditional source SHA. Optional: `chunks_count`, `generated_at`.
- `MCP/tools/pipeline_tools.py::_run_dart_chunking` — async helper registered as `registry["run_dart_chunking"]` for the Phase 7b `chunking` phase. Walks `staging_dir` for DART HTML files, parses via `Trainforge/parsers/html_content_parser.py::HTMLContentParser`, threads sections into `ed4all_chunker.chunk_content`, persists `chunks.jsonl` + `manifest.json` to `LibV2/courses/<slug>/dart_chunks/`, surfaces `dart_chunks_path` + `dart_chunks_sha256` through phase outputs.
- `MCP/tools/pipeline_tools.py::_run_imscc_chunking` — async helper registered as `registry["run_imscc_chunking"]` for the Phase 7c `imscc_chunking` phase. Mirrors `_run_dart_chunking`'s template but reads HTML entries in-memory from the packaged `.imscc` zip via `zipfile.ZipFile` and emits `chunkset_kind="imscc"` + `source_imscc_sha256` (SHA-256 of the archive bytes).
- `lib/validators/chunkset_manifest.py::ChunksetManifestValidator` — warning-severity gate wired at both chunking phases. Verifies the sidecar manifest exists, parses, conforms to the schema, its `chunks_sha256` matches the on-disk JSONL bytes, and `chunker_version` matches the installed `ed4all_chunker.__version__`.
- `lib/validators/libv2_manifest.py::LibV2ManifestValidator` — critical-severity gate at the `libv2_archival` phase. Phase 7c ST 17 added three new check methods (`_check_dart_chunks_sha256`, `_check_imscc_chunks_sha256`, plus the existing `_check_concept_graph_sha256` promoted from warning to critical). Each fires a `MISSING_*` / `INVALID_*` / `*_HASH_MISMATCH` GateIssue triplet against the matching course-manifest field — fail-closed when any of the three required hashes is absent or diverges from the on-disk artifact bytes.
- `LibV2/tools/libv2/scripts/backfill_dart_chunks.py` — operator-driven script for migrating legacy archives that lack `dart_chunks/`. Walks `LibV2/courses/<slug>/source/html/`, runs the chunker, writes `dart_chunks/{chunks.jsonl, manifest.json}`, computes the chunkset SHA, and updates `manifest.json::dart_chunks_sha256`. Idempotent by default (skips when the chunkset already exists); `--force` re-emits over an existing chunkset; `--dry-run` plans without writing. Supports `--course-slug <slug>` for single-course backfill or scans every course under `--libv2-root` when omitted.

## Common Tasks

### CLI Alias (Optional)
For convenience, add to your shell profile:
```bash
alias libv2='python -m LibV2.tools.libv2.cli'
```
Examples below use the full module path; substitute `libv2` if aliased.

### Adding a New Course
```bash
libv2 import /path/to/trainforge/output/course_name \
  --domain physics \
  --subdomain mechanics
```

### Finding Courses
```bash
libv2 catalog search --domain computer-science
libv2 catalog list --division STEM
```

### Validating Structure
```bash
libv2 validate --all
libv2 validate --course [slug]
libv2 validate indexes
```

### Rebuilding Indexes
```bash
libv2 index rebuild
```

### Advanced Commands
```bash
libv2 link-outcomes <slug> --objectives <outcomes.json>  # Link learning outcomes to chunks
libv2 concepts analyze <slug>                            # Analyze concept vocabulary
libv2 concepts clean <slug>                              # Clean concept vocabulary
libv2 eval generate <slug>                               # Generate evaluation queries
libv2 eval run <slug>                                    # Run retrieval evaluation
libv2 eval compare <baseline.json> <comparison.json>     # Compare evaluation results
libv2 validate indexes                                   # Validate index consistency
```

### ChunkFilter notes

`ChunkFilter.content_type_label` performs strict enum validation when `TRAINFORGE_ENFORCE_CONTENT_TYPE=true`; default remains lenient for legacy corpora. The canonical enum is defined in `../schemas/taxonomies/content_type.json`.

## File Formats

### Course Manifest (`manifest.json`)
Extended metadata including:
- `slug`: URL-safe identifier
- `classification`: division, domain, subdomains, topics
- `ontology_mappings`: ACM CCS and LCSH codes
- `content_profile`: chunk counts, token counts, difficulty distribution
- `features.source_provenance`: advisory bool — true when any archived chunk carries `source.source_references[]`. Lets retrieval callers fast-skip source-grounded queries on pre-provenance corpora.
- `features.evidence_source_provenance`: advisory bool — true when any concept-graph edge carries `provenance.evidence.source_references[]`.

Gated by `lib/validators/libv2_manifest.py::LibV2ManifestValidator` as the `libv2_manifest` gate on the `textbook_to_course` pipeline's `libv2_archival` phase. The validator runs critical-severity checks (JSON parse, schema match, on-disk artifact hash/size agreement) and warning-severity advisories (scaffold completeness, `source_provenance=false` gap flag).

### Course Metadata (`course.json`)

Canonical shape: `schemas/knowledge/course.schema.json`. Produced by `Trainforge/process_course.py::_build_course_json`. Validated before write.

Required fields:

| Field | Type | Notes |
|-------|------|-------|
| `course_code` | string | Stable identifier (e.g. `PHYS_101`). |
| `title` | string | Course title from IMSCC manifest. |
| `learning_outcomes[]` | array | Flat list of terminal + chapter LOs (terminal first). |

Each `LearningOutcome`:

| Field | Type | Notes |
|-------|------|-------|
| `id` | string | Canonical LO ID, pattern `^[a-zA-Z]{2,}-\d{2,}$`. Trainforge emits lowercase; LibV2 matches case-insensitively. |
| `statement` | string | One-sentence LO statement. |
| `hierarchy_level` | enum | `terminal` or `chapter`. |
| `bloom_level` | enum (optional) | `remember` / `understand` / `apply` / `analyze` / `evaluate` / `create`. |
| `bloom_verb` | string (optional) | Primary verb detected in the statement. |
| `key_concepts[]` | string (optional) | Slugified concept tags. |

Consumed by `LibV2/tools/libv2/retrieval_scoring.py::load_course_outcomes` and `LibV2/tools/libv2/validator.py::validate_learning_outcomes`.

### Catalog Files
- `master_catalog.json`: All courses with full metadata
- `course_index.json`: Quick slug → path lookup
- `by_domain/*.json`: Domain-specific course lists

## Important Notes

1. **Never modify course data directly** - use the CLI tools
2. **Indexes are derived** - regenerate with `libv2 index rebuild`
3. **Cross-domain courses** use `primary_domain` + `secondary_domains`
4. **Slugs are immutable** - changing a slug breaks references

## Ontology Mappings

Two standard classification systems are supported:
- **ACM CCS**: ACM Computing Classification System (for CS content)
- **LCSH**: Library of Congress Subject Headings (general)

These are stored in `<project-root>/schemas/taxonomies/` and referenced in course manifests.

## When Helping Users

1. **Importing**: Guide through domain/subdomain selection
2. **Searching**: Use catalog queries, not filesystem searches
3. **Validation errors**: Check schema compliance first
4. **Cross-references**: Look in `catalog/cross_references/`

## Code Locations

- CLI entry point: `tools/libv2/cli.py`
- Import logic: `tools/libv2/importer.py`
- Validation: `tools/libv2/validator.py`
- Catalog generation: `tools/libv2/catalog.py`
- Index building: `tools/libv2/indexer.py`
