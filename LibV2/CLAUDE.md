# CLAUDE.md - AI Assistant Guidelines for LibV2

## Repository Purpose

LibV2 is the course library for SLM (Small Language Model) training and retrieval. It stores processed educational content with semantic categorization across STEM and Arts domains, and is designed for flat storage at library scale (one directory per course, navigation via derived catalog indexes rather than directory nesting).

## Pipeline Position

LibV2 is the **final stage** of the Ed4All core pipeline. SLM training is a post-import sub-stage that operates on already-imported courses.

```
SemantiK ───> Courseforge ───> Trainforge ──────────────> LibV2 (this)
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
python -m LibV2.tools.libv2.cli retrieve "your query" --limit 10

# With filters:
python -m LibV2.tools.libv2.cli retrieve "query" \
  --domain physics \
  --chunk-type explanation \
  --limit 10
```

### NEVER Do These

1. **NEVER** read `chunks.jsonl` files directly via Read tool
2. **NEVER** iterate through `courses/*/semantik_chunks/`, `courses/*/imscc_chunks/`
   (or the legacy `dart_chunks/` / `corpus/` aliases) directories
3. **NEVER** write an ad-hoc "load every chunk" helper — the whole retrieval
   surface exists so no caller ever materializes a full chunkset in context
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
python -m LibV2.tools.libv2.cli multi-retrieve "compare UDL and differentiated instruction"

# Show decomposition explanation
python -m LibV2.tools.libv2.cli multi-retrieve "how does accessibility improve learning" --explain

# Disable decomposition for simple queries
python -m LibV2.tools.libv2.cli multi-retrieve "define cognitive load" --no-decompose

# With filters
python -m LibV2.tools.libv2.cli multi-retrieve "assessment strategies for stem" \
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
under the queried corpus so the assistant's interactions with LibV2 leave a
durable trail alongside the source data. After the assistant reads the
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
  --course demo-course-1 --limit 10

# Cross-course query (record lands at catalog/queries/<query_id>.json):
libv2 ask "compare UDL vs differentiated instruction" --method hybrid

# Attach the synthesized answer to a previously-asked query:
libv2 answer q_20260426_204818_7c65277e --course demo-course-1 \
  "<synthesized answer text>"

# Browse the log:
libv2 queries list --course demo-course-1
libv2 queries show q_20260426_204818_7c65277e --course demo-course-1

# Force fresh retrieval (skip cache):
libv2 ask "How does owl:sameAs entail?" --course demo-course-1 --force
```

Default retrieval method is `bm25+intent`; override with `--method
{bm25, bm25+graph, bm25+intent, bm25+tag, hybrid}`. Limit is capped
at 50 to honor the policy above.

The Q&A log is the canonical place to look when reviewing what the assistant
asked the corpus and what it synthesized — useful for auditing
RDF/SHACL enrichment work, building evals, and detecting recurring
gaps in coverage.

### For Metadata (No Token Cost)

Use catalog commands instead of retrieval:
```bash
python -m LibV2.tools.libv2.cli catalog stats        # Overview statistics
python -m LibV2.tools.libv2.cli catalog list         # Course listing
python -m LibV2.tools.libv2.cli info [slug]          # Course details
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
      └── Subdomain (mechanics, synthetic-topic-delta, etc.)
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
- `semantik_chunks/` — staged-conversion chunkset (SemantiK-derived). **This is the active emit directory.** `chunks.jsonl` (one canonical v4 chunk per line, JSONL) + sibling `manifest.json` (chunkset sidecar, `chunkset_kind: "semantik"`), anchored to the staged accessible HTML via `manifest.source_semantik_html_sha256`. Emit path: the `chunking` workflow phase between `staging` and `objective_extraction` (staged-chunkset emitter in `MCP/tools/pipeline_tools.py`; see below). Hash recorded at the course-manifest scope as `manifest.json::semantik_chunks_sha256`.
- `dart_chunks/` — legacy pre-SemantiK layout for the same staged chunkset. **Read-only fallback** — no emitter writes it; resolving here does NOT warn (the deprecation warning is deliberately deferred until on-disk corpora migrate). Present only on legacy archives; a staged-chunkset backfill script under `LibV2/tools/libv2/scripts/` (re)builds the chunkset for archives that lack it.
- `imscc_chunks/` — IMSCC-derived chunkset. Symmetric sibling: same JSONL + manifest pair, but `chunkset_kind: "imscc"`, anchored to the packaged `.imscc` archive via `manifest.source_imscc_sha256`. Emit path: the `imscc_chunking` workflow phase between `packaging` and `training_synthesis` (see `_run_imscc_chunking` below). Hash recorded at the course-manifest scope as `manifest.json::imscc_chunks_sha256` — **required by the manifest schema**.
- `corpus/` — legacy pre-Phase-7c name for the IMSCC chunkset. Read-only fallback; resolving here DOES emit a `DeprecationWarning` naming the migration script.
- `{semantik,imscc}_chunks/dedup_ledger.jsonl` — optional sidecar written **only** when `ED4ALL_CHUNK_DEDUP` is on and at least one within-package duplicate was skipped. One JSON object per skipped unit, exactly five keys: `dropped_index` (0-based ordinal of the dropped unit in the chunker's source-ordered unit sequence — **not** a chunk index; a dropped unit never gets one), `kept_chunk_index` (0-based index into `chunks.jsonl` of the surviving first occurrence's first chunk), `kept_chunk_id` (that chunk's `id`, always resolvable in `chunks.jsonl`), `normalized_hash` (sha256 of the exact-normalized heading-scoped unit key), and `source_item_path` (the DROPPED unit's locator). It is a sibling FILE rather than a manifest key because `schemas/library/chunkset_manifest.schema.json` is `additionalProperties: false` at the top level; the count rides in `source_coverage.drop_reasons.within_package_duplicate` instead. Ledger line count equals the number of skipped UNITS, and one unit could have produced several chunks had it been kept — so it is a unit count, not a chunk count.
- `concept_graph/` — Pedagogy concept graph. `concept_graph_semantic.json` produced by the `concept_extraction` workflow phase. Hash recorded at `manifest.json::concept_graph_sha256` — **required by the manifest schema**. The three-hash triangle pins the staged chunkset ↔ IMSCC chunkset ↔ concept graph to the same course manifest revision.
- `course.json` — Course-level learning outcomes and metadata.
- `objectives.json` — Archived projection of the run's synthesized objectives (`lib/libv2_storage.py::project_objectives_for_archive`, filename constant `OBJECTIVES_ARCHIVE_FILENAME`).
- `graph/` — Concept co-occurrence graph (legacy / advisory; distinct from `concept_graph/`).
- `manifest.json` — Course metadata and classification. Carries the chunkset + concept-graph SHA-256 fields above plus `chunker_version`, source artifacts, classification, and feature flags.
- `models/` — Trained adapters, one subdir per `model_id`, plus the `_pointers.json` promotion ledger.
- `pedagogy/` — Pedagogical model metadata.
- `quality/` — Quality metrics and assessment reports.
- `queries/` — Persistent Q&A log written by `libv2 ask` / `answer` / `answer-grounded --log` (one `<query_id>.json` per record). Cross-course records land in `catalog/queries/` instead.
- `source/` — Source artifacts, split `source/pdf`, `source/html`, `source/imscc`.
- `training_specs/` — Training specification files.

Resolution shims (single source of truth `lib/libv2_storage.py`): `resolve_imscc_chunks_dir` / `resolve_imscc_chunks_path` walk `imscc_chunks/` → `semantik_chunks/` → `dart_chunks/` → `corpus/`; `resolve_staged_chunks_dir` / `resolve_staged_chunks_path` cover the staged side; `resolve_chunks_path_for_query` is the query-path entry point. The dirname ↔ `chunkset_kind` map (`DIRNAME_TO_CHUNKSET_KIND` / `CHUNKSET_KIND_TO_DIRNAME`) lives there too, so the build path (`vector_index.py`) and the query path cannot drift.

The LibV2 gate's scaffold-completeness advisory (`_EXPECTED_SUBDIRS` in `lib/validators/libv2/manifest.py`) currently expects `dart_chunks` (the legacy staged-chunkset subdir name), `imscc_chunks`, `graph`, `training_specs`, `quality`, `source/pdf`, `source/html`, `source/imscc` — a missing one is a warning-severity `MISSING_SCAFFOLD_SUBDIR`, never a block.
- `vector_index/` — On-device semantic vector index (built by `libv2 vector-index build`). Three artifacts: `embeddings.npy` (float32 `[N, dim]`, C-order, L2-normalized rows; row `i` ↔ `id_map[i]`), `id_map.json` (load-bearing chunk-id order), and `manifest.json` (provenance manifest per `schemas/library/vector_index_manifest.schema.json` — embedding provider/kind/model/dim, `source_chunks_sha256`, `embeddings_sha256`, `id_map_sha256`, `chunkset_kind`, `text_field_policy`, the replay triple `device` / `batch_size` / optional `dtype` (real measurements read off the client's `model_fingerprint()`, never defaulted — `device` legitimately carries `cpu` / `cuda` / `cuda:N` / the `server` sentinel for a remote encoder; an absent `dtype` means unrecorded, never "assume fp32"), the optional `embed_overflow` accounting block, the optional `parent_chunks_count` (present only when the `ED4ALL_EMBED_OVERFLOW_SPLIT` arm sliced over-window chunks into sub-window rows — `chunks_count` stays the ROW count and sub-piece ids `{parent}#p{n}` are collapsed back onto their parent inside `VectorIndex.search`), and the asymmetric-retrieval `document_prefix` / `query_prefix` recorded for replay — passages are embedded with `document_prefix` prepended at build time, queries get `query_prefix` at search time; both empty for symmetric models). Pure-numpy exact cosine search; backs `libv2 retrieve --engine semantic` / `--engine hybrid-rrf`. The query path is fail-closed: a missing index raises `SemanticIndexMissing`, a chunkset-sha drift raises `SemanticIndexStale`, and a `provider="fake"` manifest is refused unless `ED4ALL_EMBEDDING_ALLOW_FAKE=true` — never a silent BM25 fallback. Verified by `lib/validators/vector_index_manifest.py::VectorIndexManifestValidator` (`libv2 vector-index verify`).
- `retrieval_eval/` — Retrieval gold sets + benchmark reports. `gold_set.json` (WS1-authored; `schemas/retrieval/gold_set.schema.json`) and `benchmark_<ts>.json` siblings emitted by `libv2 retrieval-benchmark` (BM25 vs semantic vs hybrid-rrf Recall@{1,3,5,10} + MRR + latency + per-engine deltas vs the BM25 baseline). Distinct from the SLM-eval `eval/` dir.

#### Semantic retrieval + vector index

The semantic retrieval path is fail-closed end to end (no lexical fallback ever masquerades as semantic). Operator surface:

```bash
# Build (or rebuild) the per-course on-device vector index. Downloads happen
# here (provision-time) unless --offline; the query path is always offline.
libv2 vector-index build --course <slug> [--provider st] [--model <id>] \
  [--chunkset imscc|dart|corpus-legacy] [--device cpu|cuda|cuda:N] [--batch-size N] \
  [--offline] [--force]
# --device accepts cpu | cuda | cuda:N (validated by the one resolver,
# lib/embedding/providers.py::normalize_device_token — NOT a click.Choice, so
# cuda:N is accepted and recorded verbatim in the manifest). 'auto' is
# deliberately not a token. Default: ED4ALL_EMBEDDING_DEVICE, else the registry
# default (`cuda`). --device / --batch-size are applied through a RESTORING
# env scope, so a per-course override cannot pin a later build in the same
# process. Build + status print the recorded (device, batch_size, dtype).
# --chunkset pins one chunkset; default precedence follows
# resolve_imscc_chunks_dir: imscc_chunks -> semantik_chunks -> dart_chunks
# -> legacy corpus. (`dart` and `corpus-legacy` are the legacy fallback
# kinds; `semantik_chunks/` is resolved by default when no flag is passed.)
libv2 vector-index status --course <slug>     # manifest summary + staleness check
libv2 vector-index verify --course <slug>     # full sha re-verification (exit 1 on drift)

# Query with the semantic / hybrid-rrf engine (default lexical = BM25):
libv2 retrieve "<query>" --course <slug> --engine semantic
libv2 retrieve "<query>" --course <slug> --engine hybrid-rrf
libv2 multi-retrieve "<query>" --course <slug> --engine semantic

# Benchmark BM25 vs semantic vs hybrid-rrf over a course gold set. Emits a
# human-readable comparison table + a benchmark_<ts>.json report under
# retrieval_eval/. --build-index provisions the canonical index inline.
libv2 retrieval-benchmark --course <slug> [--engines bm25,semantic,hybrid-rrf] \
  [--gold-set PATH] [--k 1,3,5,10] [--limit N] [--out PATH] \
  [--build-index] [--provider st] [--model <id>] \
  [--device cpu|cuda|cuda:N] [--batch-size N]
# --device / --batch-size apply to --build-index (and to every arm of the
# --models sweep), scoped so a per-arm pin cannot leak into the QUERY encoder
# that keeps running in the same process afterwards. Without them the inline
# build silently used ambient env and the report could not say which device
# produced the index it measured.

# Multi-model sweep: build one temp index per model
# (vector_index.bench-<tag>/ alongside the canonical dir), benchmark each,
# and write one report per model (config.models records the requested list).
# The canonical vector_index/ is preserved across the sweep; temp dirs are
# cleaned unless --keep is passed. Mutually exclusive with --model.
libv2 retrieval-benchmark --course <slug> \
  --models BAAI/bge-base-en-v1.5,BAAI/bge-large-en-v1.5 [--keep]
```

Embedding providers are registry entries in `lib/embedding/providers.py` (kinds `st` / `openai-embeddings` / `fake`), selected via `ED4ALL_EMBEDDING_PROVIDER` (default `st`).

**Device + precision.** `ED4ALL_EMBEDDING_DEVICE` defaults to **`cuda`** for index builds and query encoding alike; CPU is a fully-supported explicit selection (`ED4ALL_EMBEDDING_DEVICE=cpu`). There is **no automatic CUDA→CPU fallback** — a selected-but-absent device raises `EmbeddingBackendUnavailable` naming the CPU opt-out, never a silent downgrade. `ED4ALL_EMBEDDING_DTYPE` (`fp32` default \| `bf16` \| `fp16`) selects the ENCODER compute precision only; `encode_batch` still returns float32 and the persisted matrix stays float32. Non-`fp32` with `device=cpu` raises rather than being ignored.

**Determinism (restated).** Byte-identical `embeddings.npy` / `id_map.json` holds only WITHIN a fixed `(device, dtype, batch_size)` triple — same machine + venv + provider + model + that triple. A cuda-built index is **not** bit-reproducible against a cpu-built one (GPU reductions are not associative). That is exactly why all three are recorded in the manifest: a mixed-provenance comparison can be refused instead of silently trusted. `libv2 vector-index status` prints the triple; `dtype` is optional and prints as `unrecorded` when the client exposed no precision seam — absent means unknown, never "assume fp32".

**Pre-existing indexes.** Manifests written before the provenance change recorded a fabricated `device: "cpu"` / `batch_size: 1` rather than a measurement. Those values are wrong-by-record; rebuild with `--force` rather than comparing a new index against them.

**Query-side matching.** When a client is passed explicitly, `semantic_retriever._verify_client_matches_index` refuses a device/dtype mismatch against the index it is querying. The auto-built query client resolves from ambient env and can only be pinned through the environment.

**Search-path knobs** (both byte-identical in output; latency/parity only): `ED4ALL_RETRIEVAL_BLAS_THREADS` (default `8`) caps the BLAS thread team around the search GEMV only — never the build encode — and `ED4ALL_RETRIEVAL_TOPK_LEGACY` restores the pre-`argpartition` full-Python top-k sort as a parity escape hatch. The default top-k widens the `argpartition` candidate set to cover the whole score tie-group before a `lexsort` on `(-score, chunk_id)`, so the deterministic chunk-ID tie-break is preserved exactly.

#### Grounded answer + citation-back

`libv2 answer-grounded` runs the fully-automated grounded-answer pipeline
(retrieve → calibrated refusal → local-model compose → WS1 citation gate),
distinct from the Claude-in-the-loop `ask`/`answer` log. The single entry
point owns the refusal policy and the citation gate; the gate is NOT bypassable
from the CLI by design (emitting an ungrounded claim is the
hallucination-by-construction path). Local-only: no cloud calls ever — the
answer backend is loopback-enforced (`ED4ALL_ANSWER_PROVIDER` resolving to a
non-loopback base_url raises `AnswerProviderNotLocal`).

```bash
# Answer one course question, grounded + citation-gated:
libv2 answer-grounded "What is a SHACL NodeShape?" --course demo-course-1
libv2 answer-grounded "Explain RRF fusion" -c demo-course-2 --engine semantic
libv2 answer-grounded "Define a derivative" -c demo-course-3 --json --with-groundedness

# --engine auto picks hybrid-rrf when a vector index exists, else lexical — the
# benchmark-selected default (pure semantic never beat BM25). Resolved by the ONE
# shared `lib.libv2_storage.resolve_auto_engine`, which the GUI ask service also
# calls; an explicit engine is never rewritten. Pass `semantic` for that arm alone.
# --log persists the Q&A under courses/<slug>/queries/ (answered_by=grounded:<model_id>).

# Eval harness over a course gold set (BM25/semantic) — emits
# grounded_answer_eval_<ts>.json under retrieval_eval/:
libv2 answer-eval --course demo-course-1 --engine lexical

# Calibrate the refusal threshold for one (course, engine) — measures
# answerable (gold-set) vs unanswerable (refusal-probe) score distributions,
# emits refusal_calibration.json under retrieval_eval/:
libv2 refusal-calibrate --course demo-course-1 --engine semantic
```

`answer-grounded` exit codes: **0** answered (or answered-with-warnings), **2**
refused (low-confidence or model-side `not_in_course` — an honest "no"), **3**
blocked by the citation gate / invalid-citation contradiction (answer withheld),
**1** typed backend/index/compose failure (operator guidance names the
`ED4ALL_ANSWER_*` env triple). `answer-eval` / `refusal-calibrate` are thin
delegations to the `python -m lib.retrieval.{grounded_eval,refusal}` entry
points (same logic, one CLI; `answer-eval` passes through 3 = pipeline absent,
2 = gold refused, 0 = ok). The grounded backend is local-only (loopback); set
`ED4ALL_ANSWER_PROVIDER` / `ED4ALL_ANSWER_MODEL` / `ED4ALL_ANSWER_TIMEOUT_SECONDS`
(default `local` / `LOCAL_SYNTHESIS_MODEL` chain / `120`). `retrieval_eval/`
artifacts owned by this surface: `refusal_probes.json`, `refusal_calibration.json`,
`grounded_answer_eval_*.json`, `groundedness_review_sample.json`.

#### Chunkset architecture cross-links

- `schemas/library/chunkset_manifest.schema.json` — single canonical sidecar schema for the staged-conversion (`semantik_chunks/manifest.json`, legacy `dart_chunks/manifest.json`) and `imscc_chunks/manifest.json` chunksets. Discriminator field `chunkset_kind: "semantik" | "imscc"` (with the legacy `"dart"` value still accepted read-only) plus a conditional source-SHA branch anchoring each chunkset to its upstream source artifact (`source_semantik_html_sha256` for `semantik`, `source_imscc_sha256` for `imscc`; legacy corpora dual-read the pre-SemantiK staged-source-SHA key). Required fields: `chunks_sha256`, `chunker_version`, `chunkset_kind`, plus the conditional source SHA. Optional (non-exhaustive): `extraction_contract`, `chunks_count`, `generated_at`, `overlap_words`, `lo_linkage`, `source_coverage`.
- `MCP/tools/pipeline_tools.py` — the staged-chunkset emitter for the `chunking` phase (registered in `_build_tool_registry`). Walks `staging_dir` for staged HTML files (case-insensitive `.html`), parses via `Trainforge/parsers/html_content_parser.py::HTMLContentParser`, threads sections into `Trainforge.chunker.chunk_content`, persists `chunks.jsonl` + `manifest.json` (`chunkset_kind="semantik"`, `source_semantik_html_sha256`) to `LibV2/courses/<slug>/semantik_chunks/`, and surfaces `semantik_chunks_path` + `semantik_chunks_sha256` through phase outputs. The parse is dispatched through `_parse_staged_html_files`, which runs the SAME per-file worker (`Trainforge/parsers/parallel_html.py::parse_html_path`) either serially or across a `spawn` process pool depending on `ED4ALL_HTML_PARSE_WORKERS` — serial and pooled emit are identical **by construction**, not coincidence. Every discovered file lands in a per-file outcome ledger surfaced as `source_html_parse_outcomes` (a histogram summing to `source_html_count`), and file-level failures widen `source_coverage` under three named drop reasons — `asset_rejected` (byte-signature reject, `ED4ALL_HTML_ASSET_REJECT`), `html_read_error`, `html_parse_error` — so a file that contributes to the source digest but zero blocks can never silently shrink the corpus. `chunks.jsonl` is published via `os.replace` (no `.tmp` residue, no partially-written chunkset). `_run_imscc_chunking` mirrors the ledger, coverage accounting, atomic write, strict-utf-8 decode and stop-boundary handling, but keeps its parse serial: its inputs are in-memory zip entries, not files on disk.
- `MCP/tools/pipeline_tools.py::_run_imscc_chunking` — async helper registered as `registry["run_imscc_chunking"]` for the `imscc_chunking` phase. Mirrors the staged-chunkset emitter's template but reads HTML entries in-memory from the packaged `.imscc` zip via `zipfile.ZipFile` and emits `chunkset_kind="imscc"` + `source_imscc_sha256` (SHA-256 of the archive bytes).
- `lib/validators/chunkset_manifest.py::ChunksetManifestValidator` — warning-severity gate wired at both chunking phases. Verifies the sidecar manifest exists, parses, conforms to the schema, its `chunks_sha256` matches the on-disk JSONL bytes, and `chunker_version` matches `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`.
- `lib/validators/libv2/manifest.py::LibV2ManifestValidator` — critical-severity gate at the `libv2_archival` phase (this is the path wired in `config/workflows.yaml`; `lib/validators/libv2_manifest.py` is a deprecated back-compat shim that re-exports it with a `PendingDeprecationWarning`). Three check methods — one each for the staged-chunkset, IMSCC-chunkset, and concept-graph hashes — each fire a `MISSING_*` / `INVALID_*` / `*_HASH_MISMATCH` GateIssue triplet against the matching course-manifest field, fail-closed when any of the three hashes is absent or diverges from the on-disk artifact bytes. The staged-chunkset check prefers `semantik_chunks/chunks.jsonl` on disk, falling back to the legacy `dart_chunks/chunks.jsonl`, and reads `semantik_chunks_sha256` from the manifest (dual-reading the legacy pre-SemantiK key on old archives).
- A staged-chunkset backfill script under `LibV2/tools/libv2/scripts/` — for legacy archives that lack a staged chunkset. Walks `LibV2/courses/<slug>/source/html/`, runs the chunker, writes the chunkset `{chunks.jsonl, manifest.json}`, computes the chunkset SHA, and updates the course manifest's staged-chunkset hash. Idempotent by default (skips when the chunkset already exists); `--force` re-emits over an existing chunkset; `--dry-run` plans without writing. Supports `--course-slug <slug>` for single-course backfill or scans every course under `--libv2-root` when omitted.

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
`validate` is a command GROUP with three subcommands (not flags). All three
exit 1 on failure:

```bash
libv2 validate all             # every course under courses/
libv2 validate course <slug>   # one course
libv2 validate indexes         # catalog/index consistency
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
libv2 eval run <slug> <model_id>                         # ED4ALL-Bench dispatch — fresh adapter eval (judge=none)
libv2 eval compare <baseline.json> <comparison.json>     # Compare evaluation results
libv2 models eval <slug> <model_id>                      # Print the cached eval_report.json for a trained adapter
libv2 models eval <slug> <model_id> --fresh [--smoke] [--replace]  # Re-run a fresh adapter eval
libv2 validate indexes                                   # Validate index consistency
libv2 vector-index build --course <slug>                 # Build on-device semantic vector index
libv2 vector-index status --course <slug>                # Index manifest summary + staleness
libv2 vector-index verify --course <slug>                # Full sha re-verification (exit 1 on drift)
libv2 retrieve "<q>" --course <slug> --engine semantic   # Semantic (vs default lexical/BM25)
libv2 retrieval-benchmark --course <slug>                # BM25 vs semantic vs hybrid-rrf benchmark
libv2 answer-grounded "<q>" --course <slug>              # Grounded + citation-gated answer (local-only)
libv2 answer-eval --course <slug>                        # Grounded-answer eval harness over the gold set
libv2 refusal-calibrate --course <slug>                  # Calibrate the refusal threshold (measure-then-pin)
libv2 cross-index                                        # Build the cross-package concept index (catalog/cross_package_concepts.json)
libv2 cross-discover "<query>"                           # Route a topic query to candidate courses via the cross-package concept index
libv2 migrate <slug>                                     # OP4 — dry-run plan a course's library_format_version migration
libv2 migrate <slug> --apply                             # Apply the migration (backup manifest + validate + rollback on failure)
libv2 migrate --all [--apply]                            # Plan (or apply) across every discovered course
libv2 remove <slug>                                      # Permanently delete a course from the library
libv2 backup                                             # Snapshot the LibV2 metadata spine (catalog + manifests)
libv2 restore <path>                                     # Verify + restore a LibV2 metadata backup
libv2 export-rdf <slug>                                  # Export a course's JSON artifacts as RDF
libv2 import-model <dir>                                 # Import a TrainingRunner output dir as a course adapter
libv2 models list <slug>                                 # List adapters attached to a course
libv2 models promote <slug> <model_id>                   # Update the models/_pointers.json promotion ledger
libv2 catalog backfill                                   # Backfill catalog entries
libv2 eval init | validate                               # Scaffold / validate an eval set
libv2 retrieval-eval                                     # Run hand-curated gold queries
libv2 retrieval-compare                                  # A/B compare retrieval-method presets
libv2 answer-eval-diff <a.json> <b.json>                 # Diff two grounded_answer_eval reports
libv2 attribution-calibrate --course <slug>              # Calibrate the citation-attribution support threshold
libv2 probe-candidates --course <slug>                   # Build refusal-probe candidates + per-engine scores
libv2 gold-validate | gold-repin | gold-candidates | gold-promote
libv2 gold-metadata-backfill | gold-key-points | gold-difficulty-regrade
libv2 gold-parts | gold-enrich-passages                  # Gold-set authoring / maintenance family
```

`libv2 backup` / `libv2 restore` snapshot the LibV2 **metadata spine** (catalog +
manifests) only. That is a different surface from the top-level
`ed4all backup` / `ed4all backup --verify`, which archives the whole Ed4All data
dir. Do not treat one as a substitute for the other.

### Integrity check (fsck)

Storage-integrity checking lives at the Ed4All level, not under `libv2`:

```bash
ed4all fsck [--fix] [--verbose] [--json]
```

Backed by `lib/libv2_fsck.py::LibV2Fsck.check_all` (convenience wrapper
`run_fsck`). Six checks run in order: blob hash integrity, catalog consistency,
run manifests (incl. lockfile verification), symlink targets, orphaned files,
and cross-package concept-index freshness
(`check_cross_package_index_freshness`). Issues carry `severity`
(`error` / `warning` / `info`), `category`, `path`, and a `fixable` flag;
`--fix` attempts only the fixable ones. `result.passed` is `error_count == 0`,
and the command exits 1 when it fails.

The LibV2 root is resolved to an **absolute** path up front (honoring
`ED4ALL_LIBV2_ROOT` / `ED4ALL_HOME` via `lib/paths.py::libv2_path` when no root
is passed). This is load-bearing, not cosmetic: `course_index.json` stores
course paths RELATIVE to that root, so a cwd-relative root would make a `--fix`
run from another working directory resolve every entry against the caller's cwd
and wipe the index.

OP4 (stage 2): `libv2 migrate [<slug>|--all] [--apply]` is the on-disk `library_format_version` migration framework (`LibV2/tools/libv2/migrate.py`). Dry-run by default (`plan_course_migration` never writes); `--apply` backs up `manifest.json` to a timestamped `.bak` sibling BEFORE writing, re-runs the LibV2 validate check on the migrated course, and **rolls the manifest back** on validation failure (never a silent half-migrated course). A manifest with no `library_format_version` is the pre-1.0 `legacy` baseline; the baseline step registered is `legacy -> 1.0` (stamp-only, no directory-layout change). An already-current course plans as the empty "already current" plan.

W4.5: the cross-package concept index (`catalog/cross_package_concepts.json`, written by `libv2 cross-index`) is now CONSUMED by `LibV2/tools/libv2/cross_package_discovery.py` (via the additive `libv2 cross-discover` command) — no longer dead data. Additive read-only consumption: no env flag, no config/workflows.yaml/gate/schema change, and the `cross-index` writer path stays byte-identical.

**Fresh-eval bridge (Wave-92 deferral CLOSED).** `libv2 models eval <slug> <model_id>` prints the cached `eval_report.json` the training harness wrote alongside the model card. Adding `--fresh` re-runs a NEW evaluation from the saved adapter via `LibV2/tools/libv2/model_eval_bridge.py::run_fresh_eval` — it rebuilds an `AdapterCallable` from the model dir (`model_card.json` `base_model.{huggingface_repo,name,revision}` + `eval_config` gen knobs) and scores it with Trainforge's `SLMEvalHarness`. The fresh report is NON-destructive by default: it lands at `models/<model_id>/eval_report.fresh-<ts>.json` under the model dir. `--replace` instead overwrites the canonical `eval_report.json` after a `.json.bak` backup, so `get_model_eval_report` / `EvalGatingValidator` only pick up fresh scores deliberately (the canonical report stays training-time unless replaced). `--smoke` runs the harness in smoke mode (N=3 probes/stage). A real fresh run needs `pip install -e '.[training]'` and, on a shared-GPU box, the `scripts/ops/gpu_guard.sh run --task libv2-fresh-eval -- libv2 models eval <slug> <model_id> --fresh` wrap (`ED4ALL_GPU_LIFECYCLE` only sweeps inside `ed4all run`, not a standalone CLI). `libv2 eval run <slug> <model_id>` (default `--judge none`) now dispatches the SAME fresh bridge; `--judge anthropic` / `--judge local_nli` remain scaffold (the qualitative scorer is not yet wired). `run_fresh_eval` emits one best-effort `fresh_eval_invocation` decision-capture event (rationale interpolates `model_id`, `course_slug`, base repo, profile, `smoke`, gen knobs, `replace`, output name).

### ChunkFilter notes

`ChunkFilter.content_type_label` performs strict enum validation when `TRAINFORGE_ENFORCE_CONTENT_TYPE=true`; default remains lenient for legacy corpora. The canonical enum is defined in `../schemas/taxonomies/content_type.json`.

## File Formats

### Course Manifest (`manifest.json`)

Canonical shape: `schemas/library/course_manifest.schema.json`.

Schema-**required** fields: `libv2_version`, `slug`, `import_timestamp`,
`sourceforge_manifest`, `classification`, `content_profile`,
`imscc_chunks_sha256`, `concept_graph_sha256`. Note the staged-chunkset hash
(`semantik_chunks_sha256`) is **schema-optional but gate-required** —
`LibV2ManifestValidator` raises a critical missing-staged-chunkset-hash issue
when it is absent (dual-reading the legacy pre-SemantiK key on old archives),
so a pipeline-built archive still fails closed without it.

Notable fields:
- `slug`: URL-safe identifier. Immutable.
- `classification`: division, domain, subdomains, topics
- `ontology_mappings`: ACM CCS and LCSH codes
- `content_profile`: chunk counts, token counts, difficulty distribution
- `semantik_chunks_sha256`: SHA-256 of the staged chunkset `chunks.jsonl`, written by the archival step. Legacy archives store the same digest under a pre-SemantiK key, which the gate dual-reads.
- `imscc_chunks_sha256` / `concept_graph_sha256`: the other two legs of the hash triangle.
- `chunker_version`: version of the chunker that produced the archived chunks — the chunk-EMIT-SHAPE contract, resolved at archive time via `MCP/tools/pipeline_tools.py::_resolve_chunker_version` (which mirrors `Trainforge.chunker.CHUNKER_SCHEMA_VERSION`). Both the course-manifest and chunkset-manifest schemas accept the old `MAJOR.MINOR.PATCH` form alongside the current one.
- `extraction_contract` (optional, integer): chunk-TEXT extraction-contract version — ORTHOGONAL to `chunker_version` (that is the emit SHAPE; this versions WHAT TEXT lands in a chunk's `text`).
- `course_package_version` / `rulepack_version` / `graph_build_hash` (optional): mirrors of the same-named `concept_graph_semantic.json` fields, copied onto the manifest at archive time so an auditor reads one file.
- `source_documents_sha256_index` (optional, array): SHA-256 of every upstream source document (PDFs, IMSCC archive) that contributed to the archive.
- `eval_profile` (optional): explicit Trainforge eval-harness profile; when set it overrides the harness's default-profile resolution.
- `features.source_provenance`: advisory bool — true when any archived chunk carries `source.source_references[]`. Lets retrieval callers fast-skip source-grounded queries on pre-provenance corpora.
- `features.evidence_source_provenance`: advisory bool — true when any concept-graph edge carries `provenance.evidence.source_references[]`.
- `attestation` (optional): I5 human-review sign-off — `{reviewed_by, reviewed_at (server-stamped when omitted), scope: objectives|content|full, note?}`. Stamped via the GUI PATCH `/api/courses/{id}/attestation` (`gui.services.course_service.save_attestation`); records that a person reviewed the course. Absent on legacy manifests (still validate).
- `license` (optional): B3 license declaration for the archived source corpus — `{spdx_or_name, note?}`. Threaded from the `--license-note` run flag (`MCP/tools/pipeline_tools.py::_normalize_license_field`). Absent = byte-identical legacy manifest; when present, the archival step also writes a human-readable `NOTICE` file into the course dir.
- `attribution` (optional): B3 source-attribution declaration — `{statement, ...}`. Threaded from the `--attribution` run flag (`_normalize_attribution_field`); the `statement` is mirrored into the `NOTICE` file at archive time. Absent = byte-identical legacy manifest.
- `library_format_version` (optional): OP4 on-disk LibV2 course-layout contract version (starts `1.0`, pattern `^\d+\.\d+$`), distinct from `libv2_version` (manifest-document schema) and `chunker_version` (chunk-emit contract). Stamped `LIBRARY_FORMAT_VERSION` at archive time. A missing field is treated as the pre-1.0 `legacy` baseline (serve read-only + warn, never silent). The in-place upgrader has SHIPPED (OP4 stage 2) as the `libv2 migrate` command (`LibV2/tools/libv2/migrate.py`) — dry-run-by-default with backup + validate + rollback on `--apply`; the baseline step is `legacy -> 1.0`. Contract: `docs/operations/library-versioning.md`.

Gated by `lib/validators/libv2/manifest.py::LibV2ManifestValidator` as the `libv2_manifest` gate on the `textbook_to_course` pipeline's `libv2_archival` phase (`severity: critical`, `on_fail: block`, `on_error: fail_closed`, `max_critical_issues: 0`). The validator runs critical-severity checks (JSON parse, schema match, on-disk artifact hash agreement) and warning-severity advisories (scaffold completeness, `source_provenance=false` gap flag).

### Course Metadata (`course.json`)

Canonical shape: `schemas/knowledge/course.schema.json`. Produced by `Trainforge/process_course.py::_build_course_json`. Validated before write.

Required fields:

| Field | Type | Notes |
|-------|------|-------|
| `course_code` | string | Stable identifier (for example, `<COURSE_CODE>`). |
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

`tools/libv2/` (the `python -m LibV2.tools.libv2.cli` package):

- CLI entry point (click group, one command per surface): `cli.py`
- Import / removal: `importer.py`, `remove.py`, `backup.py`
- Validation: `validator.py`; SHACL shapes: `_shacl_validator.py`
- Catalog + indexes: `catalog.py`, `indexer.py`, `models/catalog.py`, `models/course.py`
- Retrieval: `retriever.py`, `semantic_retriever.py`, `multi_retriever.py`, `result_fusion.py`, `retrieval_scoring.py`, `vector_index.py`
- Query decomposition: `query_decomposer.py`, `query_decomposition.py`; Q&A log: `query_log.py`
- Eval: `eval_generator.py`, `eval_harness.py`, `model_eval_bridge.py`
- Concepts / outcomes: `concept_vocabulary.py`, `outcome_linker.py`, `_bloom_verbs.py`
- Cross-package: `cross_package_indexer.py` (writer), `cross_package_discovery.py` (reader)
- Export: `rdf_export.py`, `jsonld_emit.py`
- Format migration: `migrate.py`
- Operator scripts: `scripts/` — staged-chunkset backfill for legacy archives

Standalone modules directly under `tools/` (not part of the `libv2` CLI package):
`chunk_query.py`, `intent_router.py`, `study_pack_renderer.py`.

Outside `LibV2/`: the archival gate is `lib/validators/libv2/manifest.py`,
storage/path resolution is `lib/libv2_storage.py`, and integrity checking is
`lib/libv2_fsck.py`.
