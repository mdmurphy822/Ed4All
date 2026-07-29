# SemantiK

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules, decision capture requirements, and error handling. This file contains SemantiK-specific guidance only.

This file provides guidance to coding agents working in this repository.

## Overview

SemantiK is Ed4All's **PDF → WCAG 2.2 AA accessible-HTML conversion
engine** — a local-only pipeline that is **license-clean by construction**:
the extraction stack is built entirely on permissively-licensed tooling
(pypdfium2 + pdfplumber + pikepdf + Tesseract), so the code ships Apache-2.0
(see `LICENSE`). The runtime needs no
cloud LLM — the local GGUF specialists run fully offline; a hosted large-model
endpoint is an opt-in quality seat, not a dependency.

SemantiK emits a stable **source-provenance wire contract** — the
`data-semantik-*` block attributes and the `semantik:{slug}#{block_id}`
sourceId CURIE — that every downstream Ed4All consumer (Courseforge staging,
source-mapping, the chunker, the Ask path) reads to thread block-level
provenance through the pipeline. Legacy pre-SemantiK sourceId and attribute
spellings are still accepted on **READ** (dual-read:
`lib/validators/source_refs.py::SOURCE_ID_RE` and the chunker's attribute
harvest both admit the older prefix) so pre-purge corpora keep resolving;
every EMITTER mints the `semantik:` form. The full block-provenance contract
is in [`architecture.md`](architecture.md) §12.

Core principle: **learned models are narrow candidate generators;
deterministic code orchestrates, gates, and assembles.** BERTs classify,
Qwens generate, deterministic code owns composition / hierarchy / ARIA /
validation. That is what lets SemantiK make a WCAG conformance claim — the
rules that produce conformance are auditable code, not weights.

Code layout: the cascade + models live under `SemantiK/semantik_structure/`; the
Ed4All-facing adapter seam (output-contract normalizer + deterministic
front-matter filter) lives under `lib/semantik/`; the MCP bridge wiring lives
in `MCP/tools/pipeline_tools.py`.

Trainer layout: the seven council/theta head-trainers live under
`SemantiK/training/` (`train_classifier.py`, `train_structure.py`,
`train_semantic.py`, `train_merge_or_split.py`, `train_table_specialist.py`,
`train_math_specialist.py`, `train_reasoner.py`). Three trainers remain under
`SemantiK/scripts/` by design, not as an unfinished migration:
`train_qwen_lora.py` is a thin CLI wrapper, while `train_figure_router.py` and
`train_semantic_preservation.py` are substantial standalone trainers (the
theta cross-encoder + the figure-router head) that are run as one-off scripts
rather than part of the council DAG.

## The 13-stage v2 cascade at a glance

Entry: `semantik_structure/cascade.py::run_full_cascade` (reachable as
`pipeline_v2.run(pdf, mode="v2")`). Full per-stage depth (input → behavior →
output) is in [`architecture.md`](architecture.md); the one-liner map:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | extract | pikepdf + pypdfium2 + pdfplumber + Tesseract → per-page text/bbox |
| 2 | features | font/geometry/column/reading-order features per block |
| 3 | council | BERT specialists: Structure · Semantic · MergeOrSplit · TableSpecialist · Math (shared backbone, one-resident LoRA adapter swap) |
| 4 | cross-BERT reranker | arbitrate conflicting council signals → routing decisions |
| 5 | structure_graph | deterministic 6-pass grouping → typed `Region` objects |
| 5b | GLM-OCR enrich | optional table-cell OCR enrichment (cache/live) |
| 5c | figure bbox → PNG | render figure Region bboxes to PNG bytes |
| 5d | structure reviewer | **off by default** — conservative large-model heading/structure corrector under a strict text-conservation invariant |
| 5e | block join/split | **regroup arm default-ON** (`SEMANTIK_UNIT_REGROUP`, ITEM1); same-kind join/split + fused-title + the optional large-model op-proposal layer stay off by default — a deterministic-first FB-boundary re-partitioner (merge shattered/continued same-kind regions, fuse cross-kind pedagogical label+body units, split over-merged/pedagogical-label regions) under the R-PART partition + text-conservation invariant + bounded audited MOVE (shadow by default) |
| 6 | Qwen specialists | prose/table/math HTML generation; local GGUF or hosted large-model endpoint; batched two-phase |
| 6b | figure captioner | SmolVLM2 alt-text + extended description per figure |
| 7 | per-region hard gate | axe-wcag22aa · html5 · text_preserve · mathml · table_structure · heading_tree (eliminating) |
| 8 | per-region soft reranker | pick top surviving candidate per region |
| 9 | assembler | role→HTML via ontology, heading-tree normalize, gap-fill splice (`pass_9a/9b/9c`). **Whole-document heading-contiguity pass**: `assembler/heading_contiguity.py::normalize_document_heading_levels`, wired into `assemble_document` after 9a/9c, re-levels EVERY `<hN>` (structural + specialist-EMBEDDED + gap-filled) so the Stage-10 `heading_tree` gate sees a contiguous hierarchy — fixes a prose-embedded `<h6>` that 9a's `kind=="heading"`-only normalization misses (text/ids/attrs byte-preserved, idempotent). |
| 10 | document hard gate | document-scope axe · lang · title · landmark · heading contiguity |
| 11 | document soft reranker | document quality / lane-selection signal |
| 12 | theta | DeBERTa-v3-small + LoRA semantic-preservation cross-encoder (Stage-12 score) |
| 13 | exit decider | one capped offline retry, then stamp `ship_with_confidence` / `ship_with_flag` / `offline_qwen_lane` / `non_certified_stamp`. **Theta-stub bypass**: when theta runs in stub mode (`SEMANTIK_ALLOW_THETA_STUB=1`, the v8 mode-collapse fallback's flat 0.7 placeholder), the exit-decider does NOT let the placeholder trip the theta-`<TAU>` offline retry / non-certified path — `theta_is_stubbed` (keyed off `semantic_preservation.method == "stub_v1"`, in `theta/evaluator.py`) makes `theta/offline_retry.py::_needs_retry` skip the theta trigger (still retries on a real `wcag=failed`) and `theta/exits.py::decide_exit` ships `ship_with_flag` + appends the `THETA_UNVERIFIED_STUB` flag (byte-stable when theta is real). |

## Runtime modes

- **Local GGUF specialists (default).** `SEMANTIK_SPECIALIST_PROVIDER=local`
  (or unset). The Stage-6 specialists run in-process on `llama-cpp-python`
  (built against CUDA), fully offline, no key. This is the byte-stable
  default.
- **Hosted large-model endpoint seat (opt-in, never silent).** `SEMANTIK_SPECIALIST_PROVIDER=nvidia`
  (any non-local value) CONFIGURES an OpenAI-compatible hosted endpoint —
  defaults to the NVIDIA large-model seat (`NVIDIA_API_KEY` / `NVIDIA_BASE_URL`).
  The Stage-5d reviewer / Stage-5e resegment seats route there directly when
  their own flags are on. For **Stage-6 authoring**, selecting the provider
  alone leaves the LOCAL specialists as the authoring tier; the large-model seat displaces
  them only on an explicit opt-in — `SEMANTIK_SPECIALIST_REFINE=1` (hybrid) or
  `SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE=1` (pure-endpoint). This is the
  **only** part of SemantiK that selects an LLM provider, so it is the only
  flag carrying a `docs/LICENSING.md` row.
- **Batched two-phase Stage-6.** Phase 1 generates local drafts **batched by
  adapter** (one PROSE/TABLE/MATH adapter resident at a time — never
  interleaved on 8 GB VRAM) and is the DEFAULT authoring tier even when an
  endpoint provider is configured. Phase 2 is the optional hosted-endpoint
  pass, reached only on an explicit opt-in: `SEMANTIK_SPECIALIST_ENDPOINT_DISPLACE=1`
  (pure-endpoint) skips Phase 1 and fans all regions concurrently; the hybrid
  `SEMANTIK_SPECIALIST_REFINE=1` mode runs Phase 1 then sends the drafts to
  the large-model seat for a polish pass. Region-index keying is identical across modes, so
  Stages 7+ never see the difference. On the endpoint lane the Phase-2 POSTs
  are **multi-region BATCHED by default** (`SEMANTIK_SPECIALIST_BATCH`, on) —
  many regions per POST via `generate_multi` so the hosted seat's
  ~40-requests/minute cap is not exhausted by the ~197×K per-region POSTs
  that otherwise 429 on ~every call (set `SEMANTIK_SPECIALIST_BATCH=0` for the
  byte-stable per-region path). The batched lane caps candidate-K to ≤2 and
  uses low concurrency (2) so a 16-page slice fires ~34 POSTs.
- **Stage-5d structure reviewer (off by default).** `SEMANTIK_STRUCTURE_REVIEW=1`
  enables a conservative, cluster-aware large-model reviewer that corrects
  chapter/section headings before structure is finalized. It never alters
  text or re-partitions FeatureBlocks, runs a document-level
  token-conservation check, and **fails closed** (reverts to the unreviewed
  region list) on any mismatch. The load-bearing front-matter fix is the
  always-on deterministic detector (see § Front-matter), not this reviewer.
- **Stage-5d full-block reviewer — content-type re-typing (off by default).**
  `SEMANTIK_BLOCK_REVIEW=1` (on top of `SEMANTIK_STRUCTURE_REVIEW=1`) widens the
  reviewer from heading-only into a *structural editor over block IDs*: it
  re-types the council's known-weak content kinds (`code_block`, `table` —
  dispatched unconditionally) and drops furniture, emitting ops keyed by index,
  **never text** (verbatim source rides deterministic assembly; token-conservation
  fails closed). The 7B sees an edge-windowed view (head/tail tokens per block)
  and returns `corrected_kind`. Live-validated on a 3-chapter algebra-textbook slice
  (2026-06-27): **197 re-types, 0 text reverts** — all 35
  council `code_block`s (which were "TRY IT" exercises) → `paragraph`/`math`;
  over-detected `table`s (prose callouts, single definitions) → `paragraph`,
  section titles → `heading`, while **real data tables stay `table`**.

  **Canonical run config (default).** The reviewer rides the endpoint seat
  (`make_runtime("endpoint", …)`), so point it at a LOCAL Ollama model:
  ```bash
  SEMANTIK_STRUCTURE_REVIEW=on SEMANTIK_BLOCK_REVIEW=on \
  SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE=0 \
  SEMANTIK_SPECIALIST_BASE_URL=http://localhost:11434/v1 \
  SEMANTIK_SPECIALIST_API_KEY=ollama \
  SEMANTIK_STRUCTURE_REVIEW_MODEL=qwen2.5-7b-16k:latest \
  SEMANTIK_SPECIALIST_CONCURRENCY=1     # serialize: one Ollama model can't run windows in parallel
  ```
  **On an 8GB dev box use the 16k model, NOT 32k, for the *reviewer*** — the
  reviewer's windows are ~2k tokens, but `qwen2.5-7b-32k` allocates the full 32k
  KV (≈8.7GB > 8GB → ~27% spills to CPU → 120s timeouts); `qwen2.5-7b-16k` is
  fully GPU-resident (~6.3GB). 32k stays the *authoring* seat. `CONCURRENCY=1` is
  load-bearing: the windowed dispatch fans all windows at once, but one Ollama
  model serializes them, so concurrent windows trip the client timeout while
  queued. **GPU lifecycle:** the cascade keeps the council BERTs resident until
  Stage-5e (`release_council_gpu()` only fires before Stage-6), so a full-cascade
  run on 8GB has council + reviewer coexisting at Stage-5d. For an isolated
  reviewer run/validation that keeps one model on the card at a time
  (council → `release_council_gpu()` → reviewer), see the council/structure/clean
  → release → `run_structure_review` sequence. Run the council on GPU (sequential
  cascade phases time-share the card — do NOT force the council to CPU).
  On top of that, the deterministic **GPU-lifecycle stage-lease** seams
  (`ED4ALL_GPU_LIFECYCLE`, root-owned, default ON — the cross-venv twin
  `semantik_structure/gpu_lifecycle.py` since the bridge cannot import Ed4All's `lib/`)
  hand the card between STAGES rather than relying on contention: `cascade.py::
  _gpu_lifecycle_release` fires an ollama `keep_alive:0` sweep post-Stage-5e /
  pre-Stage-6 (the 5d reviewer + 5e resegment are ONE lease — never released
  between 5d and 5e), a torch `empty_cache` post-Stage-6b captioner /
  post-Stage-6 GGUF (belt-and-suspenders over the AdapterSwap `runtime.free`) /
  post-Stage-12 theta, and a second ollama sweep post-second-pass+ocr_repair
  (one lease over both same-seat consumers). It WRAPS, never replaces,
  `release_council_gpu()` (still the council's canonical end-of-stage release)
  and the AdapterSwap/theta scope-drops. Every seam is idempotent +
  lazy-reload-safe so `maybe_offline_retry`'s `_run_inner` re-entry is safe; the
  seams gate on `ED4ALL_GPU_LIFECYCLE` (flag-off → zero release calls,
  byte-identical) and are fail-soft. No new `SEMANTIK_*` flag — the gate is the
  root-owned `ED4ALL_*` flag.

## The cross-venv bridge

SemantiK's runtime pulls in heavy ML deps (torch, transformers, peft, a
CUDA-built `llama-cpp-python`, sentence-transformers for theta) that must NOT
live in Ed4All's MCP/orchestrator venv. So the cascade runs **out of
process**, in its own venv, behind a JSON bridge:

- `SemantiK/scripts/run_cascade_json.py` is the subprocess entry point: PDF
  in → `run_full_cascade` → HTML + `region_provenance` + conformance audit as
  JSON on stdout.
- `MCP/tools/pipeline_tools.py` invokes it (`_run_semantik_bridge_subprocess`,
  the conversion seam for the `semantik_conversion` phase). `SEMANTIK_PYTHON` =
  absolute path to the SemantiK venv python; `SEMANTIK_RUNTIME_DIR` = SemantiK
  repo root used as the subprocess `cwd` (so model/cache dirs resolve). When
  the in-process deps are absent and `SEMANTIK_PYTHON` is unset, the bridge
  **fails closed with operator guidance** (no silent stub).

Everything crossing the bridge is the § Output contract wire contract, so the
rest of Ed4All consumes conversion output through one stable interface.

## In-process install (Option A)

As an alternative to the cross-venv bridge, SemantiK's deps can be installed
*directly* into Ed4All's venv so the cascade runs **in-process**
(`MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion` takes its arm (a)
when `from SemantiK.semantik_structure.cascade import run_pipeline_v2` succeeds).
The bare `semantik_structure.*` imports resolve because that function inserts
`<repo>/SemantiK` onto `sys.path` before the import (mirroring
`scripts/run_cascade_json.py`).

Provision into a venv that already carries the heavy ML stack
(torch≥2.2 / transformers≥4.45 / peft≥0.10 / scikit-learn / scipy /
bitsandbytes — all satisfied by Ed4All's `[training]`+`[embedding]` extras):

```bash
# 1. The 10 missing pure-pip deps (the [semantik] extra).
pip install -e '.[semantik]'

# 2. The headless Chromium used by the axe-core a11y audit.
playwright install chromium

# 3. llama-cpp-python for the LOCAL Stage-6 GGUF specialists — a CUDA
#    *source* build, NOT the CPU wheel:
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
  pip install --no-binary llama-cpp-python llama-cpp-python
```

Step 3 is **optional for structure-only conversion**: the Stage-6 specialists
can run on the hosted NVIDIA large-model endpoint instead
(`SEMANTIK_SPECIALIST_PROVIDER=nvidia`, key `NVIDIA_API_KEY`), so a box without
a CUDA-built `llama-cpp-python` can still run the full cascade with Stage-6 via
the endpoint seat.

## Output contract

SemantiK emits a stable, downstream-facing shape (full detail:
[`architecture.md`](architecture.md) §12):

- **Source-provenance block attributes + `semantik:{slug}#{block_id}` sourceId.**
  The adapter seam (`lib/semantik/adapter.py`, `cascade_ir.py`) wraps each
  block in a provenance-stamped `<section data-semantik-block-id=…
  data-semantik-source=… data-semantik-pages=… data-semantik-page-kind="physical"
  data-semantik-confidence=… data-semantik-wcag=…>` — `data-semantik-block-id`
  carries the block's stable id, `data-semantik-source` the
  `synthesized`/`vendor` provenance, `data-semantik-pages` the physical PDF
  page span, and `data-semantik-wcag` the per-region gate verdict. Optional
  attributes ride the SAME opening tag when their feature fired
  (`data-semantik-block-role`, `-demoted-role`,
  `-repair` / `-repair-count` / `-repair-annotated`,
  `-opener` / `-opener-group`, `-flow`, `-unit`, `-subclass`, `-box-title`).
  The sourceId `block_id` is
  **deterministic** (first raw FeatureBlock index, or a content hash under
  `TRAINFORGE_CONTENT_HASH_IDS=1`) — same PDF in → same sourceIds out, so
  chunk refs / `source_module_map.json` / citation deep-links survive re-runs.
- **`region_provenance`.** Per-region list in emission order
  (`cascade.py::_build_region_provenance`): `region_index`, `region_kind`,
  `role`, `confidence`, `wcag_status`, `first_raw_block_index`, `pages`,
  `heading_text`/`level`, `figure_alt`, `raw_text`, and an OPTIONAL `review`
  block (Stage-5d corrections only).
- **`*.conformance_audit.json`** (`conformance_audit.py`, schema
  `conformance-audit/1.0`): exit action + WCAG status, per-region + document
  gate logs **with skip counts** (skips are first-class — "no measurement",
  not "verified safe"), theta report, thresholds, heading tree, and rule-id →
  WCAG SC coverage map.

## Front-matter handling (deterministic)

Phantom-TOC / front-matter contamination (a book's front-matter Table of
Contents getting classified as real chapter headings) is fixed at the adapter
seam by the **deterministic, CPU-only** detector
`lib/semantik/toc_frontmatter_detector.py`, applied inside `build_chapters_ir`
before chapters assemble. Three signals: a front-matter zone anchor (first
real chapter heading not in a TOC run), a monotonic-page-number TOC run
(≥`_MIN_TOC_RUN` contiguous "Title … pageN" entries with increasing page
numbers), and a page-density cluster (many chapter-level headings packed into
a tiny early-page span). It reuses tested predicates from
`lib/semantic_structure_extractor`. This deterministic detector is the
load-bearing front-matter fix; the off-by-default Stage-5d large-model reviewer is the
secondary, conservative defensive layer.

## Standalone reasoning-QC runner (omni-vs-Super seat A/B, task #39)

`semantik_structure/reasoning_qc_standalone.py` judges an ALREADY-EMITTED
accessible-HTML file with the Stage-9b reasoning-QC machinery **without
re-running conversion** — so two reasoning seats (e.g. a Nemotron-Omni vs a
Super-120B text seat) can be compared on the SAME combined HTMLs a full run
produced. It is **REPORT-ONLY**: no reconcile op is applied, the input HTML is
never rewritten (the A/B needs verdicts, not mutations).

It **reuses** the production judgment path verbatim — parse → one document-level
`QCWindow` → `reasoning_qc._fan_out_page_verifies` (the same window/junction-seam
partition, bounded fan-out, reasoning-preserving split ladder, verdict stitch,
and `FlaggedBlock` conversion) → findings — so NO judgment logic is duplicated
and `reasoning_qc.run_reasoning_qc`'s cascade/apply behavior (and its tests) are
untouched. `parse_accessible_html` walks each provenance-stamped `<section>` with
BeautifulSoup (`lxml`) into the QC record shape `{type, role, level, page, text}`
(+ `block_id`); type is coarse-inferred from the block's first significant
descendant (`hN`→heading+level, `table`→table, `figure`/`img`→figure,
`ul`/`ol`/`dl`→list, `pre`→code, `blockquote`→blockquote, else paragraph), role =
the block-role attribute, page = first int of the pages attribute. It reads BOTH
the emitted `data-semantik-*` names and the legacy pre-SemantiK spelling, so it
parses pre-purge and post-purge HTML alike. Seat resolution is
the normal env chain (`SEMANTIK_REASONING_QC_BASE_URL`/`_MODEL` > specialist seat
> legacy VLM seat); `lib.vllm_container_lifecycle.ensure_serving` is called for
the resolved seat first (fail-soft, records `load_seconds`).

```bash
# omni run (point the QC seat at the omni endpoint), report per file
SEMANTIK_REASONING_QC_BASE_URL=http://localhost:8000/v1 \
SEMANTIK_REASONING_QC_MODEL=nemotron-omni \
python3 -m semantik_structure.reasoning_qc_standalone \
  <OUTPUT_DIR>/<CORPUS>_accessible.html --out-dir qc/omni/

# Super-120B run over the SAME HTML (a directory of *_accessible.html also works)
SEMANTIK_REASONING_QC_BASE_URL=http://localhost:8001/v1 \
SEMANTIK_REASONING_QC_MODEL=super-120b \
python3 -m semantik_structure.reasoning_qc_standalone <OUTPUT_DIR>/ --out-dir qc/super/

# compare the two reports (writes qc_compare.json + prints a summary)
python3 -m semantik_structure.reasoning_qc_standalone --compare \
  qc/omni/<CORPUS>.qc_report.json \
  qc/super/<CORPUS>.qc_report.json --out-dir qc/
```

Each run writes `<out-dir>/<stem>.qc_report.json` (`html_path`, `seat`, unit
stats, all findings with absolute indices + block ids, `wall_clock_seconds`,
`load_seconds`, timestamp). `compare_reports` aligns findings by
`(block_id, finding_kind)` → both / only-A / only-B counts + qc_incomplete rates
+ wall-clock delta. API: `parse_accessible_html`, `run_standalone_qc`,
`compare_reports`.

## Testing

| Area | Tests |
|------|-------|
| Cascade IR / adapter seam | `lib/semantik/tests/test_cascade_ir.py`, `test_adapter.py`, `test_vendor_ingest.py` |
| Standalone reasoning-QC runner (task #39) | `semantik_structure/tests/test_reasoning_qc_standalone.py` (parse ordered records, report shape + no-mutation, compare alignment, never-non-thinking + no-image guards on the standalone path) |
| Stage-5d reviewer | `semantik_structure/qwen_specialists/tests/test_reviewer.py`, `test_reviewer_prompt.py`, `test_cascade_stage5d.py`; bridge: `lib/semantik/tests/test_structure_review_bridge.py` |
| Theta device | `lib/semantik/tests/test_theta_device.py` |
| Front-matter detector | `lib/semantik/tests/test_toc_frontmatter_detector.py` |
| Bridge / MCP seam | `MCP/tests/test_semantik_v2_seam.py`, `test_semantik_bridge_e2e.py`, `test_semantik_bridge_subprocess.py`, `test_semantik_dispatch_flip.py`, `lib/semantik/tests/test_run_cascade_json_env.py`, `test_run_cascade_json_alloc_conf.py` |
| Chunker enrichment | `Trainforge/chunker/tests/test_semantik_chunk_enrichment.py` |
| Specialist runtime | `semantik_structure/qwen_specialists/tests/` |
| Tables (structural confirm) + Figures (detect/route/sidecar) | `semantik_structure/council/tests/test_table_figure_detection.py`, `semantik_structure/tests/test_figure_detection.py`, `test_figure_sidecar.py`, `test_image_extract_yflip.py`, `lib/semantik/tests/test_figure_img_src.py` |
| VLM figure-DETECT lane (task #56) | `semantik_structure/tests/test_vlm_figure_detect.py` (flag + DETECT_FIGURES dual-gate; the accept gate incl. the PAGE-RASTER backstop + the TEXT-COLUMN / WCAG-1.4.5 guard; the **GRID rejector** — a ruled table is rejected, a solid-fill photo is not mistaken for a lattice, **a NUMBER LINE is never rejected** (the trap), the numeric table that passes the word guard is caught by grid structure, the markdown arm reuses `table_structure`, and the flag-off path is byte-identical; the OCR-lane pixel-vs-point coordinate-space contract; DETECT call shape + tolerant parse; **DecisionCapture fires with a dynamic rationale**; the injection seam consumed end-to-end by the real `detect_image_region_candidates` + `is_page_raster_candidate`) |

## MCP Tools

SemantiK has no dedicated `@mcp.tool()` surface of its own — it is wired in as
the **conversion backend** in `MCP/tools/pipeline_tools.py`. The
`semantik_conversion` phase routes through the cross-venv bridge
(`_run_semantik_bridge_subprocess`) when the SemantiK runtime is configured,
producing the § Output contract shape that the downstream output-staging +
marker-validation pipeline tools consume. Bridge env (`SEMANTIK_PYTHON`
/ `SEMANTIK_RUNTIME_DIR` / timeout / alloc-conf) is in the flag table below.

## Decision Capture

The MCP bridge emits one **`structure_review`** DecisionCapture event per
converted document (the doc-level Stage-5d verdict list — block_id, verdict,
kind/level before→after, reason code, reverted flag — surfaced off
`conformance_audit["structure_review"]`; resolves to `None`/not-run when the
reviewer is off). Rationale is dynamic and replayable (per-doc verdict
counts). Regression: `lib/semantik/tests/test_structure_review_bridge.py`
asserts the JSONL row fires with a dynamic rationale. See root `/CLAUDE.md`
§ Decision Capture for the contract.

The Stage-6b figure captioner (the SmolVLM2 VLM call site,
`semantik_structure/figure_captioner.py::caption_figure_regions`) emits one
**`alt_text_generation`** DecisionCapture event per captioned figure (W7.6),
constructed best-effort under the `semantik` capture tool / `semantik_conversion`
phase (`_build_caption_capture`). Rationale is dynamic + replayable — image
content hash (sha256), figure geometry (`px_size`) + stable min-FB id, the
SmolVLM2 model id + per-prompt `max_new_tokens`, and the produced alt/extended
caption lengths — so a replay can attribute the alt text to its exact input.
Best-effort: a capture-construction / log failure is non-fatal (never breaks
captioning), mirroring the bridge's `structure_review` posture. Regression:
`semantik_structure/tests/test_figure_captioner_capture.py` asserts one
`alt_text_generation` decision fires per figure with a dynamic rationale (VLM
boundary mocked).

The MCP conversion seam also emits one **`block_resegment`** DecisionCapture
event per converted doc when the Stage-5e re-partition pass fired
(`MCP/tools/pipeline_tools.py` section 2c → `_emit_block_resegment_capture`,
off the cascade's `block_resegment` audit rows). It resolves the audit off
BOTH cascade arms — the in-process result-dict top-level `block_resegment` key
AND the cross-venv bridge (`run_cascade_json._build_bridge_dict` now forwards
it via `_resolve_block_resegment`; the same-touch `second_pass_verify` arm is
forwarded alongside). Rationale is dynamic + replayable: merge / split /
regroup op tallies, the fused-title-split count, the folded-region tally, the
merged-unit `semantic_class` set, a bounded `source_ids` sample, and
`conservation_verified`. It is a deterministic-pass capture (the resegment ops
are deterministic-first, so it fires even with no LLM op-proposal layer) riding
the canonical `block_resegment` `decision_type` enum (no schema change), and is
best-effort (a capture failure logs a warning, never breaks conversion) —
mirroring the `structure_review` posture. Regression:
`lib/semantik/tests/test_structure_review_bridge.py` § 3c
(`test_block_resegment_capture_row_emitted_dynamic_rationale`).

The **VLM figure-DETECT lane** (task #56, `semantik_structure/vlm_figure_detect.py::inject_figure_candidates`
→ `_emit_detect_capture`) emits one **`structure_detection`** DecisionCapture event
per converted document — it is a NEW LLM call site (one multimodal DETECT POST per
page on the page-arranger seat). It rides the EXISTING `structure_detection` enum
(the page-arranger precedent) so there is **no `decision_event` schema change**.
Rationale is dynamic + replayable: the model id, the total bboxes PROPOSED vs the
number the deterministic accept gate ADMITTED, every guard-rejection tally
(`page_raster` — the whole-page backstop; `text_column` — the WCAG 1.4.5
images-of-text guard; `degenerate` / `too_small` / `bad_aspect` / `out_of_page` /
`malformed` / `duplicate` / `over_cap`), the accepted bboxes' page-area fractions,
and the busiest pages — so a replay can attribute every accepted crop to its exact
page and gate decision. Best-effort: a capture-construction / log failure is
non-fatal (never breaks conversion), mirroring the `structure_review` /
`alt_text_generation` posture. Regression:
`semantik_structure/tests/test_vlm_figure_detect.py::test_decision_capture_fires_with_a_dynamic_rationale`
(asserts the capture FIRES on the call path and that the rationale interpolates the
dynamic signals) + `test_capture_failure_never_breaks_detection`.

## Opt-In Behavior Flags

SemantiK owns the `SEMANTIK_*` env-var prefixes. The full per-flag table
(name, default, behavior, guardrails) lives in
[`docs/operations/behavior-flags-semantik.md`](../docs/operations/behavior-flags-semantik.md) —
read or grep that file before adding, removing, or changing a flag.

## Honest constraints

- **Council VRAM on 8 GB.** The full cascade is GPU-flaky on an 8 GB card —
  council BERTs share one ModernBERT-base backbone (one-resident LoRA adapter
  swap) and the Qwen specialists batch **by adapter** rather than fanning out,
  precisely because parallel adapter contexts + a concurrent Chromium/axe-core
  process poison CUDA on 8 GB. Mitigated, not eliminated; gate GPU-heavy work
  on a dev box. The DGX Spark-class deployment target is the real fix.
- **Structure quality is council-bound.** Block-ID quality of *pedagogical*
  elements is only as good as BERT-Structure's `structural_role` / `is_heading`
  heads; heading over-detection is patched defensively (the always-on
  deterministic front-matter detector is load-bearing; the Stage-5d large-model
  reviewer is conservative and off by default).

See [`architecture.md`](architecture.md) §14 for the full limitations section.

## License

Apache-2.0 (`LICENSE`, "Copyright 2026 Ed4All"). License-clean by
construction: the PDF stack is pypdfium2 (Apache-2/BSD-3) + pdfplumber (MIT) +
pikepdf (MPL-2.0) + pytesseract/Tesseract (Apache-2); the ML stack is
transformers/peft/llama-cpp-python — **every dependency on the path is
permissively licensed** (see the licensing inventory in
`semantik_structure/extract_shared.py` and `image_extract.py`). Model weights
(council BERTs, Qwen GGUFs, theta head) are separate artifacts, not shipped in
this tree. The provider/model licensing for the opt-in hosted large-model endpoint seat
lives in `docs/LICENSING.md`.
