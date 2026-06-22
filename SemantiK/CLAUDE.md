# SemantiK

> **Universal Protocols**: See root `/CLAUDE.md` for orchestrator protocol, execution rules, decision capture requirements, and error handling. This file contains SemantiK-specific guidance only.

This file provides guidance to coding agents working in this repository.

## Overview

SemantiK is the **license-clean replacement for DART** — a local-only
PDF → WCAG 2.2 AA accessible-HTML pipeline. It exists because the whole
point of replacing DART was **license cleanliness**: SemantiK's extraction
stack carries no PyMuPDF/MuPDF (AGPL-3) and no Poppler (GPL-2), so the code
ships Apache-2.0 (see `LICENSE`). The runtime needs no cloud LLM — the local
GGUF specialists run fully offline; a hosted 70B endpoint is an opt-in
quality seat, not a dependency.

SemantiK is a drop-in for DART: it preserves DART's `data-dart-*` HTML
markers and the `dart:{slug}#{block_id}` sourceId **wire contract** so every
downstream Ed4All consumer (Courseforge staging, source-mapping, the chunker,
the Ask path) is unchanged. The full contract is in
[`architecture.md`](architecture.md) §12.

Core principle: **learned models are narrow candidate generators;
deterministic code orchestrates, gates, and assembles.** BERTs classify,
Qwens generate, deterministic code owns composition / hierarchy / ARIA /
validation. That is what lets SemantiK make a WCAG conformance claim — the
rules that produce conformance are auditable code, not weights.

Code layout: the cascade + models live under `SemantiK/dart_semantic/`; the
Ed4All-facing adapter seam (output-contract normalizer + deterministic
front-matter filter) lives under `lib/semantik/`; the MCP bridge wiring lives
in `MCP/tools/pipeline_tools.py`.

## The 13-stage v2 cascade at a glance

Entry: `dart_semantic/cascade.py::run_full_cascade` (reachable as
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
| 5d | structure reviewer | **off by default** — conservative 70B heading/structure corrector under a strict text-conservation invariant |
| 6 | Qwen specialists | prose/table/math HTML generation; local GGUF or hosted 70B endpoint; batched two-phase |
| 6b | figure captioner | SmolVLM2 alt-text + extended description per figure |
| 7 | per-region hard gate | axe-wcag22aa · html5 · text_preserve · mathml · table_structure · heading_tree (eliminating) |
| 8 | per-region soft reranker | pick top surviving candidate per region |
| 9 | assembler | role→HTML via ontology, heading-tree normalize, gap-fill splice (`pass_9a/9b/9c`) |
| 10 | document hard gate | document-scope axe · lang · title · landmark · heading contiguity |
| 11 | document soft reranker | document quality / lane-selection signal |
| 12 | theta | DeBERTa-v3-small + LoRA semantic-preservation cross-encoder (Stage-12 score) |
| 13 | exit decider | one capped offline retry, then stamp `ship_with_confidence` / `ship_with_flag` / `offline_qwen_lane` / `non_certified_stamp` |

## Runtime modes

- **Local GGUF specialists (default).** `SEMANTIK_SPECIALIST_PROVIDER=local`
  (or unset). The Stage-6 specialists run in-process on `llama-cpp-python`
  (built against CUDA), fully offline, no key. This is the byte-stable
  default.
- **Hosted 70B endpoint seat.** `SEMANTIK_SPECIALIST_PROVIDER=nvidia` (any
  non-local value) routes Stage-6 generation (and the Stage-5d reviewer) to
  an OpenAI-compatible hosted endpoint — defaults to the NVIDIA hosted 70B
  (`NVIDIA_API_KEY` / `NVIDIA_BASE_URL`). This is the **only** part of
  SemantiK that selects an LLM provider, so it is the only flag carrying a
  `docs/LICENSING.md` row.
- **Batched two-phase Stage-6.** Phase 1 generates local drafts **batched by
  adapter** (one PROSE/TABLE/MATH adapter resident at a time — never
  interleaved on 8 GB VRAM). Phase 2 is the optional hosted-endpoint pass:
  pure-endpoint mode skips Phase 1 and fans all regions concurrently; the
  hybrid `SEMANTIK_SPECIALIST_REFINE=1` mode sends the local drafts to the
  70B for a polish pass. Region-index keying is identical across modes, so
  Stages 7+ never see the difference.
- **Stage-5d structure reviewer (off by default).** `SEMANTIK_STRUCTURE_REVIEW=1`
  enables a conservative, cluster-aware 70B reviewer that corrects
  chapter/section headings before structure is finalized. It never alters
  text or re-partitions FeatureBlocks, runs a document-level
  token-conservation check, and **fails closed** (reverts to the unreviewed
  region list) on any mismatch. The load-bearing front-matter fix is the
  always-on deterministic detector (see § Front-matter), not this reviewer.

## The cross-venv bridge

SemantiK's runtime pulls in heavy ML deps (torch, transformers, peft, a
CUDA-built `llama-cpp-python`, sentence-transformers for theta) that must NOT
live in Ed4All's MCP/orchestrator venv. So the cascade runs **out of
process**, in its own venv, behind a JSON bridge:

- `SemantiK/scripts/run_cascade_json.py` is the subprocess entry point: PDF
  in → `run_full_cascade` → HTML + `region_provenance` + conformance audit as
  JSON on stdout.
- `MCP/tools/pipeline_tools.py` invokes it (`_run_semantik_bridge_subprocess`,
  the DART-conversion seam). `SEMANTIK_PYTHON` = absolute path to the SemantiK
  venv python; `SEMANTIK_RUNTIME_DIR` = SemantiK repo root used as the
  subprocess `cwd` (so model/cache dirs resolve). When the in-process deps are
  absent and `SEMANTIK_PYTHON` is unset, the bridge **fails closed with
  operator guidance** (no silent stub).

Everything crossing the bridge is the § Output contract wire contract, so the
rest of Ed4All cannot tell SemantiK from DART.

## In-process install (Option A)

As an alternative to the cross-venv bridge, SemantiK's deps can be installed
*directly* into Ed4All's venv so the cascade runs **in-process**
(`MCP/tools/pipeline_tools.py::_run_semantik_v2_conversion` takes its arm (a)
when `from SemantiK.dart_semantic.cascade import run_pipeline_v2` succeeds).
The bare `dart_semantic.*` imports resolve because that function inserts
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
can run on the hosted NVIDIA 70B endpoint instead
(`SEMANTIK_SPECIALIST_PROVIDER=nvidia`, key `NVIDIA_API_KEY`), so a box without
a CUDA-built `llama-cpp-python` can still run the full cascade with Stage-6 via
the endpoint seat.

## Output contract

SemantiK emits the same shape DART did (full detail:
[`architecture.md`](architecture.md) §12):

- **`data-dart-*` markers + `dart:{slug}#{block_id}` sourceId.** The adapter
  seam (`lib/semantik/adapter.py`, `cascade_ir.py`) wraps each block in
  `<section class="dart-section" data-dart-block-id=… data-dart-source=…
  data-dart-pages=… data-dart-page-kind="physical" data-dart-confidence=…
  data-dart-wcag=…>`. The sourceId `block_id` is **deterministic** (first raw
  FeatureBlock index, or a content hash under `TRAINFORGE_CONTENT_HASH_IDS=1`)
  — same PDF in → same sourceIds out, so chunk refs / `source_module_map.json`
  / citation deep-links survive re-runs.
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
load-bearing front-matter fix; the off-by-default Stage-5d 70B reviewer is the
secondary, conservative defensive layer.

## Testing

| Area | Tests |
|------|-------|
| Cascade IR / adapter seam | `lib/semantik/tests/test_cascade_ir.py`, `test_cascade_ir_real_ea2e.py`, `test_adapter.py`, `test_vendor_ingest.py` |
| Stage-5d reviewer | `dart_semantic/qwen_specialists/tests/test_reviewer.py`, `test_reviewer_prompt.py`, `test_cascade_stage5d.py`; bridge: `lib/semantik/tests/test_structure_review_bridge.py` |
| Theta device | `lib/semantik/tests/test_theta_device.py` |
| Front-matter detector | `lib/semantik/tests/test_toc_frontmatter_detector.py` |
| Bridge / MCP seam | `MCP/tests/test_semantik_v2_seam.py`, `test_semantik_bridge_e2e.py`, `test_semantik_bridge_subprocess.py`, `test_semantik_dispatch_flip.py`, `lib/semantik/tests/test_run_cascade_json_env.py`, `test_run_cascade_json_alloc_conf.py` |
| Chunker enrichment | `Trainforge/chunker/tests/test_semantik_chunk_enrichment.py` |
| Specialist runtime | `dart_semantic/qwen_specialists/tests/` |

## MCP Tools

SemantiK has no dedicated `@mcp.tool()` surface of its own — it is wired in as
the **DART-conversion backend** in `MCP/tools/pipeline_tools.py`. The
`dart_conversion` phase routes through the cross-venv bridge
(`_run_semantik_bridge_subprocess`) when the SemantiK runtime is configured,
producing the same output contract DART emitted (§ Output contract), so the
`stage_dart_outputs` / `validate_dart_markers` pipeline tools downstream are
unchanged. Bridge env (`SEMANTIK_PYTHON` / `SEMANTIK_RUNTIME_DIR` / timeout /
alloc-conf) is in the flag table below.

## Decision Capture

The MCP bridge emits one **`structure_review`** DecisionCapture event per
converted document (the doc-level Stage-5d verdict list — block_id, verdict,
kind/level before→after, reason code, reverted flag — surfaced off
`conformance_audit["structure_review"]`; resolves to `None`/not-run when the
reviewer is off). Rationale is dynamic and replayable (per-doc verdict
counts). Regression: `lib/semantik/tests/test_structure_review_bridge.py`
asserts the JSONL row fires with a dynamic rationale. See root `/CLAUDE.md`
§ Decision Capture for the contract.

## Opt-In Behavior Flags

SemantiK (the DART replacement — a license-clean semantic-cascade
PDF→structured-content converter under `SemantiK/dart_semantic/` +
`lib/semantik/`) owns the `SEMANTIK_*` env-var prefix. All toggles default to
byte-stable / off-or-local behaviour to preserve backward compatibility; only
the two model/provider selectors below (`SEMANTIK_SPECIALIST_PROVIDER` + the
two `*_MODEL` seats) carry a `docs/LICENSING.md` row. Resolution everywhere is
parse-with-fallback (unset / garbage / out-of-range → the documented default;
a run never crashes on a malformed flag), mirroring the root `ED4ALL_*`
posture.

| Flag | Default | Purpose |
|------|---------|---------|
| `SEMANTIK_SPECIALIST_PROVIDER` | `local` | **Selects the Stage-6 specialist + Stage-5d structure-reviewer generation backend** (`SemantiK/dart_semantic/qwen_specialists/runtime.py::resolve_specialist_provider`, ~L97). Resolution: explicit arg > `SEMANTIK_SPECIALIST_PROVIDER` (lowercased/stripped) > `local`. The `_LOCAL_PROVIDER_VALUES` set (`{"", "local", "gguf", "llama_cpp", "llamacpp"}`) all resolve to the byte-stable in-process local-GGUF council adapters (no network, no key); ANY other non-empty value (e.g. `nvidia` / `endpoint`) routes generation to the hosted OpenAI-compatible endpoint seat (`endpoint_runtime.py`). **Selects an LLM provider → has a `docs/LICENSING.md` row.** The endpoint seat defaults to the NVIDIA hosted 70B (key `NVIDIA_API_KEY`, base `NVIDIA_BASE_URL`). Garbage / unknown → treated as a provider name (fail-loud at POST time if no base_url/key resolves). |
| `SEMANTIK_SPECIALIST_BASE_URL` | unset → `NVIDIA_BASE_URL` | Base URL of the hosted OpenAI-compatible endpoint for Stage-6 specialist + Stage-5d reviewer generation (`endpoint_runtime.py::_resolve_base_url`, ~L113). Resolution: `SEMANTIK_SPECIALIST_BASE_URL` > `NVIDIA_BASE_URL` > none. No machine default is hardcoded — when the provider is non-local and neither resolves, the endpoint runtime fails loud at generation time (no silent fallback). URL only — no `docs/LICENSING.md` row. |
| `SEMANTIK_SPECIALIST_API_KEY` | unset → `NVIDIA_API_KEY` | Bearer token for the hosted endpoint seat (`endpoint_runtime.py::_resolve_api_key`, ~L120). Resolution: `SEMANTIK_SPECIALIST_API_KEY` > `NVIDIA_API_KEY` > none; a missing key on a non-local provider raises `EndpointRuntimeError` before any dispatch (fail-closed). Credential, never hardcoded — no `docs/LICENSING.md` row (the seat it unlocks is covered by the NVIDIA hosted-70B licensing row). |
| `SEMANTIK_SPECIALIST_MODEL` | `meta/llama-3.3-70b-instruct` | **Model ID for Stage-6 specialist generation** (math / table / prose / gap-fill) on the endpoint seat (`endpoint_runtime.py::_resolve_model`, ~L127). Resolution: `SEMANTIK_SPECIALIST_MODEL` > `NVIDIA_LARGE_MODEL` > the literal `meta/llama-3.3-70b-instruct` default. **Selects an LLM model → has a `docs/LICENSING.md` row** (the NVIDIA hosted-70B seat). No-op when `SEMANTIK_SPECIALIST_PROVIDER` resolves local. |
| `SEMANTIK_STRUCTURE_REVIEW` | unset (off) | Stage-5d 70B **structure-reviewer** gate (`SemantiK/dart_semantic/qwen_specialists/reviewer.py::resolve_structure_review_mode`, ~L322). Default OFF → the cascade's heading/structure output is byte-identical (no reviewer dispatch). Truthy (`1`/`true`/`yes`/`on`, case-insensitive) → a conservative, cluster-aware 70B reviewer corrects chapter/section headings before the structure is finalized. Falsey / unset / garbage → off. Gate only (the model it dispatches is the `SEMANTIK_STRUCTURE_REVIEW_MODEL` / specialist seat) — no separate `docs/LICENSING.md` row beyond the model seat's. |
| `SEMANTIK_STRUCTURE_REVIEW_MODEL` | `meta/llama-3.3-70b-instruct` | **Model ID for the Stage-5d structure-reviewer seat** (`runtime.py::resolve_structure_review_model`, ~L126; mirrored in `MCP/tools/pipeline_tools.py` ~L6380). Resolution: `SEMANTIK_STRUCTURE_REVIEW_MODEL` > `SEMANTIK_SPECIALIST_MODEL` > `NVIDIA_LARGE_MODEL` > the literal `meta/llama-3.3-70b-instruct`. **Selects an LLM model → has a `docs/LICENSING.md` row** (same NVIDIA hosted-70B seat as the specialist). No-op when `SEMANTIK_STRUCTURE_REVIEW` is off. |
| `SEMANTIK_SPECIALIST_REFINE` | unset (off) | Hybrid two-phase Stage-6 refine gate (`SemantiK/dart_semantic/qwen_specialists/runner.py::resolve_refine_mode`, ~L105). Default OFF. Truthy (`1`/`true`/`yes`/`on`) → Phase-2 sends the local-adapter drafts + directive to the 70B endpoint for a polish pass (only meaningful when `SEMANTIK_SPECIALIST_PROVIDER` resolves to the endpoint). Falsey / garbage → off. Routes to the already-licensed specialist seat — no separate licensing row. |
| `SEMANTIK_SPECIALIST_CONCURRENCY` | `8` | Thread-pool `max_workers` for concurrent endpoint POSTs in batched Stage-6 generation (`endpoint_runtime.py`, ~L101). Parse-with-fallback: non-int / non-positive / garbage → `8`. No provider/model selection — no licensing row. |
| `SEMANTIK_SPECIALIST_TIMEOUT_SECONDS` | `120.0` | Per-request HTTP timeout (float seconds) for endpoint specialist/reviewer POSTs (`endpoint_runtime.py`, ~L135). Non-finite / non-positive / garbage → `120.0`. No licensing row. |
| `SEMANTIK_DROP_FRONTMATTER_ZONE` | `1` (on) | Deterministic **page-density front-matter-zone** detector gate (`lib/semantik/toc_frontmatter_detector.py::_zone_flag_enabled`, ~L73/446). Default ON → a page-density pass detects dense chapter-heading clusters on early pages and drops the front-matter zone (catches the preface-summary contamination defect). Default-on parse semantics: explicit falsey (`0`/`false`/`no`/`off`) → off; unset / truthy / garbage → on. No provider/model — no licensing row. |
| `SEMANTIK_DROP_FRONTMATTER_TOC` | `1` (on) | Deterministic **phantom-TOC** front-matter detector gate (`lib/semantik/toc_frontmatter_detector.py::_flag_enabled`, ~L66/170). Default ON → the phantom-TOC + chapter-index-cluster + boilerplate detection runs inside the front-matter zone. Default-on parse semantics identical to `SEMANTIK_DROP_FRONTMATTER_ZONE` (explicit falsey → off; else on). No licensing row. |
| `SEMANTIK_THETA_DEVICE` | `cpu` | Torch device for the theta cross-encoder (DeBERTa-v3-small + LoRA semantic-preservation head; `SemantiK/dart_semantic/theta/semantic_preservation.py`, `ENV_THETA_DEVICE`, ~L74). Resolution: explicit `device` arg > `SEMANTIK_THETA_DEVICE` > legacy `DART_THETA_DEVICE` > `cpu`. Accepts `cpu` / `cuda` / `cuda:N`; empty / garbage → `cpu`. Graceful CUDA-unavailable fallback to CPU at load time (never crashes a GPU-less box). Mirrors `ED4ALL_NLI_DEVICE`. Device knob, not a provider/model — no licensing row. |
| `SEMANTIK_EXPANDABLE_SEGMENTS` | unset (off) | CUDA-allocator opt-in for the out-of-process bridge (`MCP/tools/pipeline_tools.py::_semantik_expandable_segments_enabled`, ~L6061). Truthy (`1`/`true`/`yes`/`on`) → the bridge subprocess env gets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (OOM mitigation; opt-in because the setting is driver/torch-version dependent). Falsey / unset / garbage → off (unset). No licensing row. |
| `SEMANTIK_BRIDGE_TIMEOUT_SECONDS` | `3600.0` | Subprocess timeout (float seconds) for the out-of-process SemantiK cascade JSON bridge (`MCP/tools/pipeline_tools.py::_resolve_semantik_bridge_timeout`, ~L6068). Non-float / non-positive / garbage → `3600.0`. No licensing row. |
| `SEMANTIK_PYTHON` | unset (required for bridge) | Absolute path to the SemantiK runtime venv's python interpreter, used when the cascade runs out-of-process via the JSON bridge (`MCP/tools/pipeline_tools.py`, `_SEMANTIK_PYTHON_ENV` ~L6041/6134/6579). No machine default — when the in-process SemantiK deps are absent and this is unset, the bridge fails closed with operator guidance. Exec path, not a provider/model — no licensing row. |
| `SEMANTIK_RUNTIME_DIR` | unset (required for bridge) | SemantiK runtime repo root, used as the subprocess `cwd` for the out-of-process cascade so model/cache dirs resolve relative to it (`MCP/tools/pipeline_tools.py`, `_SEMANTIK_RUNTIME_DIR_ENV` ~L6042/6135). No machine default; fail-closed when the bridge is needed and this is unset. Path knob — no licensing row. |
| `SEMANTIK_HOME` | unset (repo-relative) | **Relocatable data root** (`SemantiK/dart_semantic/paths.py`, ~L80). When set, every SemantiK data dir defaults to `<SEMANTIK_HOME>/<basename>` (`models`, `data`); the per-dir overrides below keep higher precedence (per-dir env > `SEMANTIK_HOME` > package-relative `<SemantiK>/<basename>`). Mirrors `ED4ALL_HOME`. Read at call time (tests/Docker can flip it without re-import). Byte-stable to the package-relative default when unset. No licensing row. |
| `SEMANTIK_MODEL_DIR` / `SEMANTIK_CACHE_DIR` / `SEMANTIK_DATA_DIR` / `SEMANTIK_CONFIG_DIR` | `<SemantiK>/models` (model) · `<SemantiK>/data` (cache/data/config) | Per-dir relocation overrides for the SemantiK data roots (`SemantiK/dart_semantic/paths.py::_resolve`, ~L108/118/127/137). Precedence per dir: the per-dir env var > `SEMANTIK_HOME/<basename>` > package-relative. `SEMANTIK_MODEL_DIR` points at model weights / LoRA council adapters / Qwen GGUFs / theta head; the cache/data/config roots hold extract/glm-ocr/prerender caches, eval reports/datasets/labels, and calibration/config artifacts respectively. No provider/model selection — no licensing row. (Counted as four flags in the prefix-ownership tally.) |

**Dev-scripts-only flag (kept out of the table above to avoid noise, mirroring the root `ED4ALL_*` test-discovery posture):** `SEMANTIK_ARXIV_REPO` (default `~/arxiv-repo/papers`) points the offline eval-corpus builders under `SemantiK/scripts/` (`build_eval_corpus_manifest.py`, `pair_from_arxiv.py`, `measure_stage5_heading_rate.py`) at a local arxiv paper repo. Not a production pipeline path; selects no provider/model.

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
  deterministic front-matter detector is load-bearing; the Stage-5d 70B
  reviewer is conservative and off by default).

See [`architecture.md`](architecture.md) §14 for the full limitations section.

## License

Apache-2.0 (`LICENSE`, "Copyright 2026 Ed4All"). License-clean by
construction: the PDF stack is pypdfium2 (Apache-2/BSD-3) + pdfplumber (MIT) +
pikepdf (MPL-2.0) + pytesseract/Tesseract (Apache-2); the ML stack is
transformers/peft/llama-cpp-python (all permissive). **No PyMuPDF/MuPDF
(AGPL-3), no Poppler (GPL-2)** anywhere on the path (see the avoidance notes in
`dart_semantic/extract_shared.py` and `image_extract.py`). Model weights
(council BERTs, Qwen GGUFs, theta head) are separate artifacts, not shipped in
this tree. The provider/model licensing for the opt-in hosted 70B endpoint seat
lives in `docs/LICENSING.md`.
