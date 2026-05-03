# Post-Phase-8 Architecture Review & Chunker Migration Plan

> Date: 2026-05-03. Branch: `claude/dev0.3.0-courseforge-1o4fS`.
>
> Reviewed against the two original plan documents:
> - `plans/courseforge_architecture_v2.docx`
> - `plans/courseforge_architecture_handoff.docx`
>
> Companion to `plans/courseforge_architecture_roadmap.md` (the
> implementation-time roadmap; this doc audits *delivery* against the
> original plans).

---

## 1. Architecture review — original plan vs. Phase 1–8 landed

### 1.1 Pipeline ordering

The v2 plan specified the canonical chain:

```
DART → Chunker → Source analyzer → Concept extractor → Objective synthesizer (LARGE)
  → Block generator (SMALL) → Block validator → Block rewriter (LARGE) → IMSCC
```

Landed: `config/workflows.yaml::textbook_to_course` runs in this exact
order. Two notes:

- **Source analyzer + Concept extractor are folded into one phase**
  (`concept_extraction`, Phase 6). The plan specified them as separate
  stages; landed as a single stage that pulls per-chunk `topic`,
  `content_type`, `cognitive_demand`, `significance`, and concept mentions
  into the `pedagogy_graph_builder`. Functionally equivalent; would only
  matter if a future caller wanted to run source-analysis without
  emitting a concept graph.
- **Block validator runs at two seams** (Phase 3.5 promotion):
  `inter_tier_validation` (between outline and rewrite) and
  `post_rewrite_validation` (between rewrite and packaging). The plan
  specified one validator pass; the symmetric two-seam design is a
  strict superset.

### 1.2 Two-stage chunker (structural deterministic + enriched)

Plan: `chunk_structural(html) -> List[StructuralChunk]` (deterministic,
no models) followed by `enrich_chunks(structural_chunks) -> List[EnrichedChunk]`
(adds `content_type`, `cognitive_demand`, embeddings).

Landed: `ed4all_chunker.chunk_content` does the structural pass — pure,
deterministic, content-hash IDs gated by `TRAINFORGE_CONTENT_HASH_IDS=true`.
The "enrich" pass is **not a separate stage** — instead, downstream
phases (`concept_extraction`, `course_planning`) read the structural
chunks and add their own enrichments. Embeddings live in `lib/embedding/`
and are consumed by Phase 4 statistical-tier validators rather than
materialised onto chunks.

Gap: there is no single `enriched_chunks.jsonl` artifact. The
enrichments are scattered across `concept_graph_semantic.json`,
`pedagogy_graph.json`, and the per-block `data-cf-*` attributes.
Whether to consolidate is a design call — the current scatter pattern
keeps each consumer's enrichments scoped to its own concern, which is
arguably better than a fat enriched chunk.

### 1.3 Provenance chain

Plan:
```
PDF(sha256) → DART(version, config_hash) → HTML(sha256)
  → Chunker(version, config_hash) → Chunks(sha256)
  → Synthesizer → Objectives(sha256) → Generator → Blocks(sha256) → IMSCC(sha256)
```

Landed: `course_manifest.json` carries:

| Plan field | Manifest field | Status |
|---|---|---|
| PDF sha256 | (per-PDF in `archive[].pdfs[]`) | ✅ |
| DART version | (DART CLI version + config in DART output) | ✅ |
| HTML sha256 | `dart_html_sha256` (per file in `dart_chunks/manifest.json`) | ✅ |
| Chunker version | top-level `chunker_version` | ✅ |
| Chunks sha256 (DART) | `dart_chunks_sha256` | ✅ |
| Chunks sha256 (IMSCC) | `imscc_chunks_sha256` | ✅ (Phase 7c addition) |
| Concept graph sha256 | `concept_graph_sha256` | ✅ (Phase 6 addition) |
| Objectives sha256 | NOT IN MANIFEST | ⚠️ Gap — emitted at `synthesized_objectives.json` but no manifest hash |
| Blocks sha256 | NOT IN MANIFEST | ⚠️ Gap — emitted at `blocks_validated.json` / `blocks_final.json` but no manifest hash |
| IMSCC sha256 | NOT IN MANIFEST | ⚠️ Gap — only `archive[].imscc.size_bytes` |

**Three provenance gaps**: objectives, blocks, IMSCC sha256 fields are
not stamped on the LibV2 course manifest. The artifacts exist; the
hashes don't carry through. The plan called these out as part of the
end-to-end reproducibility guarantee. Fixing this is a small schema
extension + three hash computations at archive time.

### 1.4 Four-layer validation stack

Plan:

| Layer | Purpose | Landing |
|---|---|---|
| Layer 1: Constrained decoding | Schema-enforced sampling via Outlines / GBNF / lm-format-enforcer | ⚠️ Partial — `Courseforge/generators/_provider.py` builds a `constrained_decoding_payload` (`Courseforge/generators/tests/test_constrained_decoding_payload.py` covers it) but no concrete Outlines/GBNF wiring. The payload is sent in the prompt; whether the receiving model honors it depends on the provider. |
| Layer 2: SHACL/structural | Reference integrity, cardinality, controlled vocab | ✅ Phase 4 — 5 SHACL gates |
| Layer 3: Semantic (statistical) | Embedding similarity, BERT classifier disagreement, round-trip | ✅ Phase 4 — 3 embedding validators + BERT ensemble |
| Layer 4: Self-consistency / regen loop | N attempts, escalation to large model | ✅ Phase 3 — `route_with_self_consistency` + 10-attempt budget |

**Layer 1 gap**: the original plan said "Model literally cannot emit
invalid structure" — token-level vocabulary masking via Outlines or
GBNF. Landed as a structured payload added to the prompt rather than a
hard sampling-time constraint. For local/Ollama backends this can still
be tightened (Ollama supports `format: "json"` constrained decoding;
vLLM supports Outlines; LM Studio supports JSON mode); for Anthropic
the payload is a soft directive only. Worth noting in the migration
docs but not a regression — operationally the prompt-side payload + the
two-tier validation chain catches what Layer 1 would have caught at the
sampling step.

### 1.5 BERT classifier as disagreement detector

Plan: kabir5297/bloom_taxonomy_classifier (off-the-shelf MVP) + DistilBERT
SST-2 (independent, contributes to dispersion) + lightly fine-tuned in
v3.

Landed: Phase 4 BERT ensemble (`lib/classifiers/bloom_bert_ensemble.py`)
with three SHA-pinned classifiers:
1. `dyzwj/cip29-bloom-taxonomy-classifier` (replaces kabir5297 — Phase 8 ST 4)
2. `distilbert-base-uncased-finetuned-sst-2-english` (sentiment, dispersion contribution)
3. A third pinned model

The Phase 8 cip29 swap closes the v2 plan's "primary classifier needs
~95% accuracy" requirement on RDF/SHACL technical material where
kabir5297 underperformed. Decision capture: `bert_ensemble_replacement_selection`
(Phase 8 ST 9 commit `c8ef35b`).

The plan's path-3 fine-tuned classifier is a future option but not
required: the dispersion-detection signal is what actually fires
escalation, and three SHA-pinned independent classifiers already give
that.

### 1.6 Block-level provenance metadata

Plan: every block carries which model generated it, attempts, validators
passed/failed, rewrite model, final state.

Landed: ✅ Phase 2 `Block.touched_by[]` (list of `Touch{tier, model, ...}`)
+ `escalation_marker` + `validation_attempts`. Verified by the smoke
runbook in `Courseforge/CLAUDE.md` § "Operator smoke runbook (post-Phase-3.5)"
step 3 (`jq '.[0].touched_by[].tier'` shows `local, outline, outline_val,
rewrite, rewrite_val`).

### 1.7 Independent stage execution (Handoff plan)

Plan: `courseforge outline / validate / classify / rewrite / run` CLI
subcommands + Python API.

Landed: ✅ Phase 5 — four `ed4all run courseforge-*` subcommands:
- `courseforge-outline` (outline tier only)
- `courseforge-validate` (inter-tier + post-rewrite validation)
- `courseforge-rewrite` (rewrite tier with `--blocks` filter)
- `courseforge` (full Courseforge slice, all 4 phases)

The plan's standalone `classify` subcommand was rolled into
`courseforge-validate` (the BERT ensemble fires inside the validation
chain). Functionally equivalent.

### 1.8 LLM-agnosticism via env vars

Plan:
```
COURSEFORGE_OUTLINE_MODEL=qwen2.5:7b
COURSEFORGE_CLASSIFIER_MODEL=distilbert-blooms
COURSEFORGE_REWRITE_PROVIDER=deepinfra
COURSEFORGE_REWRITE_MODEL=Qwen/Qwen2.5-72B-Instruct
COURSEFORGE_REWRITE_API_KEY=...
```

Landed: **✅ implemented** at the router layer
(`Courseforge/router/router.py:154-160`, consumed at `:556-560`):

```python
_ENV_OUTLINE_PROVIDER = "COURSEFORGE_OUTLINE_PROVIDER"
_ENV_OUTLINE_MODEL = "COURSEFORGE_OUTLINE_MODEL"
_ENV_REWRITE_PROVIDER = "COURSEFORGE_REWRITE_PROVIDER"
_ENV_REWRITE_MODEL = "COURSEFORGE_REWRITE_MODEL"
_ENV_LEGACY_PROVIDER = "COURSEFORGE_PROVIDER"
```

Resolution chain (`Courseforge/router/router.py` module docstring,
lines 12-23):

1. Per-call `**overrides` (operator / test override).
2. Loaded `block_routing.yaml` policy entry (Subtask 34 — schema at
   `schemas/courseforge/block_routing.schema.json`; not yet wired,
   router falls through when policy is absent).
3. **Tier-default env vars** (`COURSEFORGE_OUTLINE_PROVIDER` /
   `COURSEFORGE_OUTLINE_MODEL` / `COURSEFORGE_REWRITE_PROVIDER` /
   `COURSEFORGE_REWRITE_MODEL`).
4. Module-level `_HARDCODED_DEFAULTS` table (one entry per
   `(block_type, tier)` pair).
5. Legacy `COURSEFORGE_PROVIDER` fallback.

So the user's intent (outline tier local + rewrite tier API; or
different rewrite models for different block types) is supported. The
pipeline orchestration layer (`MCP/tools/pipeline_tools.py:4221-4226`,
`MCP/core/executor.py:893-945`) only reads `COURSEFORGE_PROVIDER` —
that's the **legacy fallback path**, not a collapse. When operator-set
per-tier env vars resolve, the router uses them per the chain above.

The original review claimed the per-tier vars were collapsed; that was
wrong. Documenting here for the audit trail.

**Outstanding gap (the actual one)**: the resolution chain's step 2
(`block_routing.yaml` policy) is **not yet wired**. Schema exists
(`schemas/courseforge/block_routing.schema.json`); per-block-type model
selection is the v0.4.0 feature. See §1.9.

### 1.9 Block-level model routing (`block_routing.yaml`)

Plan:
```yaml
objectives: { outline: qwen2.5:7b, rewrite: Qwen/Qwen2.5-72B-Instruct }
examples:   { outline: qwen2.5:7b, rewrite: claude-sonnet-4-6 }
assessments: { outline: qwen2.5:7b, classify: distilbert-blooms, rewrite: deepseek-v3 }
prereqs:     { infer: Qwen/Qwen2.5-72B-Instruct, verify: claude-sonnet-4-6 }
```

Landed: **not implemented**. There is no per-block-type model routing.
The router (`Courseforge/router/router.py::CourseforgeRouter`) routes
all blocks of a given tier through the same provider/model. The
`schemas/courseforge/block_routing.schema.json` exists but only
documents the inter-tier routing decision (which block went which way),
not per-block-type model selection.

**Recommendation**: this is a v0.4.0 feature, not a regression. The
v0.3.0 deliverable is the two-pass ToS-unblocked surface; per-block
model routing was a stretch goal in the handoff plan. Document as
"deferred to v0.4.0" with a clear migration path: extend
`Courseforge/generators/_provider.py` to accept a `block_type → model`
map sourced from `block_routing.yaml`.

### 1.10 Summary scorecard

| Plan area | Status | Notes |
|---|---|---|
| Pipeline ordering | ✅ | Source-analyzer + concept-extractor folded |
| Two-stage chunker | ✅ structural; ⚠️ no separate `enriched_chunks` artifact | Acceptable — enrichments live with their consumers |
| 7-step provenance chain | ⚠️ 3 missing manifest hashes (objectives, blocks, IMSCC) | Small schema extension closes |
| ABCD + Bloom's | ✅ Phase 6 | |
| 4-layer validation | ⚠️ Layer 1 partial | Prompt-payload only; not sampling-time enforcement |
| BERT disagreement detector | ✅ Phase 4 + Phase 8 cip29 swap | |
| Block-level provenance | ✅ Phase 2 | |
| Independent stage subcommands | ✅ Phase 5 | classify rolled into validate |
| Per-tier env vars | ✅ wired in router (`COURSEFORGE_OUTLINE_*` / `COURSEFORGE_REWRITE_*`) | Resolution chain in `Courseforge/router/router.py:154-160` |
| Block-routing policy file | ❌ deferred | v0.4.0 feature |

**Net**: Phase 1-8 delivered the architectural intent. Two concrete
gaps to address before declaring v0.3.0 complete:
1. Provenance hashes for objectives / blocks / IMSCC in
   `course_manifest.json`.
2. Decision: ship Layer 1 as prompt-payload only (current state) or
   wire concrete constrained-decoding for at least the local provider
   (`format: "json"` is already wired in `_local_provider.py` per
   Wave 113; could be promoted to a Layer 1 contract).

The block-routing policy file (§1.9) is the v0.4.0 stretch goal that
unlocks per-block-type model selection on top of the already-wired
per-tier env vars.

---

## 2. Chunker migration plan: `ed4all-chunker` → `Trainforge/`

### 2.1 Why move it

Per user direction: Trainforge owns chunking. The dual-chunkset contract
(DART chunks for course material + IMSCC chunks for training material,
both sourced from the same document) stays — it's only a code-location
change.

### 2.2 Current state

`ed4all-chunker/` is a workspace package declared as a relative
`file:` direct-reference dependency in the parent `pyproject.toml:44`.
It carries 1164 LOC across four modules:

```
ed4all-chunker/
├── pyproject.toml      (32 LOC — separate package metadata)
├── README.md           (41 LOC)
├── ed4all_chunker/
│   ├── __init__.py     (87 LOC — re-exports)
│   ├── boilerplate.py  (136 LOC — strip_boilerplate)
│   ├── chunker.py      (701 LOC — chunk_content + chunk_text_block + merges)
│   └── helpers.py      (240 LOC — extract_plain_text, extract_section_html, etc.)
└── tests/
    └── test_chunker_smoke.py  (614 LOC, 32 tests)
```

### 2.3 Caller surface

```
Trainforge/process_course.py:72          from ed4all_chunker import (CANONICAL_CHUNK_TYPES, ...)
Trainforge/process_course.py:2527        from ed4all_chunker.helpers import type_from_resource
Trainforge/process_course.py:2876        from ed4all_chunker.helpers import extract_plain_text
Trainforge/process_course.py:2894        from ed4all_chunker.helpers import extract_section_html
Trainforge/process_course.py:2907        from ed4all_chunker.helpers import strip_assessment_feedback
Trainforge/process_course.py:2920        from ed4all_chunker.helpers import strip_feedback_from_text
Trainforge/rag/boilerplate_detector.py:20  from ed4all_chunker.boilerplate import (
MCP/tools/pipeline_tools.py:6800         from ed4all_chunker import ChunkerContext, chunk_content
MCP/tools/pipeline_tools.py:7131         from ed4all_chunker import ChunkerContext, chunk_content
LibV2/tools/libv2/scripts/backfill_dart_chunks.py:30,379,502  doc + log strings
```

8 production import sites + 32 tests + the `archive_to_libv2`
`chunker_version` resolution (`importlib.metadata.version("ed4all-chunker")`,
2 sites in `pipeline_tools.py`).

### 2.4 Existing circular-import constraint

The chunker package already has lazy imports back into Trainforge:

- `ed4all_chunker/chunker.py:410` → `from Trainforge.parsers.xpath_walker import (select_block_by_xpath, compute_xpath)`
- `ed4all_chunker/helpers.py:82` → `from Trainforge.parsers.html_content_parser import HTMLTextExtractor`

These were lazy specifically *because* the chunker was extracted *out*
of Trainforge in Phase 7a. **Moving the package back inside Trainforge
removes this constraint** — the lazy imports become direct module
imports.

### 2.5 Destination layout

```
Trainforge/
└── chunker/
    ├── __init__.py     (re-exports from chunker / boilerplate / helpers)
    ├── boilerplate.py  (verbatim from ed4all_chunker/boilerplate.py)
    ├── chunker.py      (verbatim from ed4all_chunker/chunker.py — drop lazy imports)
    └── helpers.py      (verbatim from ed4all_chunker/helpers.py — drop lazy imports)
```

Test relocation: `ed4all-chunker/tests/test_chunker_smoke.py` →
`Trainforge/tests/test_chunker_smoke.py` (rewrite imports).

### 2.6 Migration steps

**Phase A — Move the code (single commit)**

1. Create `Trainforge/chunker/` directory; copy the four package
   modules into it.
2. Rewrite imports inside the copied modules:
   - `from ed4all_chunker.boilerplate` → `from Trainforge.chunker.boilerplate`
   - `from ed4all_chunker.helpers` → `from Trainforge.chunker.helpers`
   - Drop the lazy imports — replace with direct top-level imports of
     `Trainforge.parsers.xpath_walker` and
     `Trainforge.parsers.html_content_parser`. Verify no import cycle.
3. Move `ed4all-chunker/tests/test_chunker_smoke.py` →
   `Trainforge/tests/test_chunker_smoke.py`; rewrite imports
   (`from ed4all_chunker import` → `from Trainforge.chunker import`).
4. Delete `ed4all-chunker/` directory.
5. `pyproject.toml`: remove `ed4all-chunker @ file:./ed4all-chunker` from
   `dependencies`. The pip-friction note in `README.md` becomes obsolete
   — drop it.

**Phase B — Update callers (same commit)**

Eight production import-rewrites:

```
Trainforge/process_course.py:72,2527,2876,2894,2907,2920
  from ed4all_chunker import ...           → from Trainforge.chunker import ...
  from ed4all_chunker.helpers import ...   → from Trainforge.chunker.helpers import ...

Trainforge/rag/boilerplate_detector.py:20
  from ed4all_chunker.boilerplate import ... → from Trainforge.chunker.boilerplate import ...

MCP/tools/pipeline_tools.py:6800,7131
  from ed4all_chunker import ChunkerContext, chunk_content
    → from Trainforge.chunker import ChunkerContext, chunk_content
```

**Phase C — `chunker_version` resolution (same commit)**

`importlib.metadata.version("ed4all-chunker")` no longer resolves once
the separate package is gone. Two options:

- Option 1 (preferred): expose a `__version__` constant in
  `Trainforge/chunker/__init__.py`, sourced from a Trainforge package
  version field. Replace the two `importlib.metadata` call sites with
  `Trainforge.chunker.__version__`.
- Option 2: stamp a hardcoded `CHUNKER_SCHEMA_VERSION = "v4"` constant
  and use it as the manifest field. Decouples chunker version from
  Python-package version.

Recommend Option 2 — chunker schema version is a pipeline contract,
not a package version. The two are conceptually different.

**Phase D — Documentation sync (same commit)**

- `CLAUDE.md:92`: drop the `├── ed4all-chunker/` line from the
  project-structure tree.
- `CLAUDE.md:323`: update `archive_to_libv2` chunker_version note —
  cite `Trainforge.chunker.__version__` (or the schema-version
  constant) in place of `importlib.metadata.version("ed4all-chunker")`.
- `CLAUDE.md:834`: update the "ed4all-chunker" reference doc bullet
  → "Trainforge chunker (`Trainforge/chunker/`)".
- `Trainforge/CLAUDE.md` § "Chunking (Phase 7a delegation)": rewrite to
  describe in-Trainforge chunker (no longer "delegation"; the four
  helpers ARE the chunker).
- `LibV2/CLAUDE.md:208`: same import-path fix.
- `Courseforge/CLAUDE.md` § "Phase 7b: DART chunkset" + § "Phase 7b/7c":
  update import-path references.
- `README.md:31,43`: drop the workspace-member rationale + the pip
  ≥24 caveat (no longer applies once the relative `file:` direct
  reference is gone).
- `LibV2/tools/libv2/scripts/backfill_dart_chunks.py:30,379,502`:
  update doc/log strings from "ed4all-chunker" to "Trainforge chunker".
- `ed4all-chunker/README.md`: gone with the directory.

**Phase E — Verification**

1. `.venv/bin/pip install -e .` — should succeed without the
   workspace-member friction (pre-Wave-current-state README friction
   note becomes obsolete).
2. `.venv/bin/pytest Trainforge/tests/test_chunker_smoke.py -v` —
   all 32 tests pass.
3. `.venv/bin/pytest --no-header -q` — full suite passes (modulo
   pre-existing failures excluded above).
4. End-to-end smoke (`ed4all run textbook-to-course --dry-run`):
   workflow plan still resolves; the `chunking` and `imscc_chunking`
   phases still wire the new import path.
5. `LibV2/tools/libv2/scripts/backfill_dart_chunks.py --dry-run`:
   operator script still resolves the chunker.

### 2.7 Risks

- **External consumers**: anyone importing `ed4all_chunker` from outside
  this repo (PyPI was deferred per roadmap §6.2). Greppable mitigation:
  `git log --all -p | grep "ed4all_chunker"` shows no public-facing
  references. Internal-only — safe to break.
- **Lazy-import unwinding**: when the lazy imports become direct, double-
  check no transient import cycle surfaces during `python -c "import
  Trainforge.chunker"`. The chunker pulls
  `Trainforge.parsers.{xpath_walker,html_content_parser}`; those modules
  must not import from `Trainforge.process_course` (which imports
  chunker). Verified by inspection: `xpath_walker` imports stdlib only;
  `html_content_parser` imports stdlib + bs4 only. **No cycle**.
- **`chunker_version` manifest field**: pre-Wave-current-state archives
  carry `"chunker_version": "0.1.0"` from the old `importlib.metadata`
  resolution. The Option 2 swap to a `CHUNKER_SCHEMA_VERSION = "v4"`
  constant produces a different value; auditors comparing pre-vs-post-
  migration archives will see drift. Surface this in the migration
  commit message.
- **Test placement**: 32 tests in one file might overflow
  `Trainforge/tests/`. Acceptable — the file is self-contained and
  marker-tagged.

### 2.8 Estimated scope

| Phase | Effort | Risk |
|---|---|---|
| A. Move code | 1 hour (mechanical copy + import rewrite) | Low |
| B. Update callers | 30 min (8 grep-able sites) | Low |
| C. `chunker_version` | 15 min (Option 2 const) | Low |
| D. Documentation | 1 hour (8 doc files) | Low |
| E. Verification | 30 min (suite + smoke) | Medium — may surface latent issue |

Total: ~3-4 hours; one PR; reviewable as a single commit.

---

## 3. Open hygiene items (post-migration follow-ups)

These are not regressions, but worth tracking:

### 3.1 Drop legacy `_chunk_*` wrappers on `CourseProcessor`

`Trainforge/process_course.py:1642-1747` carries thin delegating
wrappers for `_chunk_content`, `_chunk_text_block`, `_merge_small_sections`,
`_merge_section_source_ids`. Kept for back-compat with:
- `Trainforge/tests/test_merge_small_sections_zero_word.py` calls
  `proc._merge_small_sections(...)` directly.
- External `scripts/wave81_reclassify_chunks.py` imports
  `CANONICAL_CHUNK_TYPES` from the module.

Once the migration lands, these wrappers can be dropped — direct test
imports from `Trainforge.chunker` are cleaner. Migration: rewrite the
1-2 test sites; preserve the `CANONICAL_CHUNK_TYPES` re-export at the
module top.

### 3.2 Promote `chunkset_manifest` gate to critical

Currently warning-severity (`config/workflows.yaml`, `chunking` and
`imscc_chunking` phases). Promotion to critical was promised "after a
clean corpus rebuild calibrates the thresholds" (`Courseforge/CLAUDE.md`
§ "Phase 7b/7c"). When the next clean rebuild lands, flip both gates
to critical.

### 3.3 Add provenance hashes for objectives, blocks, IMSCC

Per §1.3 above — three small `course_manifest.json` field additions:
- `objectives_sha256` (over `synthesized_objectives.json` bytes)
- `blocks_validated_sha256` (over `blocks_final.json` bytes)
- `imscc_sha256` (over the packaged `.imscc` zip bytes)

Closes the v2 plan's reproducibility chain.

### 3.4 Layer 1 constrained decoding promotion

For the local provider, `format: "json"` is already wired (Wave 113).
Promote it to a documented Layer 1 contract: emit a
`structured_output_enforced=true` flag in the decision-capture record
when the provider supports sampling-time JSON schema enforcement,
`false` otherwise. Per-provider:
- `local` (Ollama / vLLM via OpenAI-compatible): `format: "json"` ✅
- `together`: OpenAI-compatible JSON mode ✅
- `anthropic`: tool-use enforcement (could be wired)
- `claude_session`: subagent prompt-side only

Documents the gap honestly without claiming a contract that doesn't
exist.

---

## 4. Recommended next steps

1. **Land the chunker migration** as a single PR (Phases A-E above).
   Estimated ~3-4 hours. Low risk, mechanical, single reviewable diff.
2. **Defer the three §3 hygiene items** to v0.4.0 unless one becomes
   load-bearing for an in-flight workstream.
3. **Open follow-up issues** for §1.8 (per-tier env vars) and §1.3
   (manifest hash gaps) — both are small, both improve the v0.3.0
   shipping surface.
4. **Mark the two .docx plans as historical** by adding a header at the
   top of each: "Original plan (2026-04-XX); see
   `plans/post-phase8-review-2026-05.md` for delivery audit and
   `plans/courseforge_architecture_roadmap.md` for the Phase 1-8
   implementation roadmap."

The `.docx` files are binary; the headers go in companion `.md`
sidecars (`courseforge_architecture_v2.STATUS.md` and
`courseforge_architecture_handoff.STATUS.md`) that operators read first.
