# SemantiK behavior flags

SemantiK converts source documents into accessible HTML through the preferred
GLM-OCR SDK path: extraction, deterministic normalization, accessibility and
document enrichment, then the Super heading judge. This page documents the
public `SEMANTIK_*` operator controls without embedding deployment-specific
hosts, ports, paths, identifiers, or model inventory.

> **Private by design.** Source documents, converted HTML, layout sidecars,
> course and run identifiers, caches, model artifacts, credentials, endpoint
> values, evaluation data, and logs are always operator-private. Keep them in
> ignored or external storage and never commit them to code, documentation,
> examples, fixtures, or comments.

For installation, use [Installation and local dependencies](installation.md).
Provider and model choices must follow [Licensing and terms posture](../LICENSING.md).
For invocation, stop/resume, and failure handling, use
[Run Ed4All pipelines](pipeline-invocation.md). SemantiK architecture and its
output contract are described in the [SemantiK architecture](../../SemantiK/architecture.md).

## Preferred conversion profile

`SEMANTIK_GLMOCR_LANE=1` selects the preferred conversion path. The lane sends
rendered pages through the GLM-OCR SDK, normalizes the returned layout into the
SemantiK provenance contract, enriches document structure and accessibility,
and invokes the Super heading judge for headings that remain unresolved.

The two defaults are intentionally different:

- the GLM-OCR lane is **opt-in** at the resolver level, so
  `SEMANTIK_GLMOCR_LANE` is off when unset;
- the heading judge is **default-on** whenever its phase has GLM-OCR layout
  sidecars. Only `0`, `false`, `no`, or `off` disables it. A document without
  pending headings is a natural no-op.

Use deployment-managed environment or secret configuration for endpoint,
credential, and model values. Do not copy concrete operator values into this
document.

## Flag reference

Defaults below are resolver defaults. “Within lane” means the control has no
effect unless its owning master lane is active. Invalid numeric input follows
the repository parse-with-fallback convention and resolves to the stated
default.

### GLM-OCR conversion and enrichment

| Flag | Default | Purpose |
|---|---|---|
| `SEMANTIK_GLMOCR_LANE` | off | Selects the preferred whole-document GLM-OCR SDK conversion path. Truthy values are `1`, `true`, `yes`, and `on`; all other values resolve off. |
| `SEMANTIK_GLMOCR_BASE_URL` | local OpenAI-compatible endpoint | GLM-OCR service endpoint. Configure the concrete endpoint outside tracked files. |
| `SEMANTIK_GLMOCR_MODEL` | `glm-ocr` | Served GLM-OCR model identifier. A changed model must remain covered by `docs/LICENSING.md`. |
| `SEMANTIK_GLMOCR_WORKERS` | `4` | Parallel SDK worker count; positive integers only. |
| `SEMANTIK_GLMOCR_RENDER_DPI` | `300` | Page-raster resolution, with a minimum of 72 DPI. |
| `SEMANTIK_GLMOCR_LAYOUT_DEVICE` | `cpu` | Device used by the SDK layout model. |
| `SEMANTIK_GLMOCR_OUTPUT_DIR` | operator-managed output | Private layout and escalation sidecar destination. Keep it ignored or external. |
| `SEMANTIK_GLMOCR_MATH_NORMALIZE` | on within lane | Applies deterministic OCR math, ordinal, and placeholder normalization. Explicit falsey values disable it. |
| `SEMANTIK_CHAPTER_LADDER_RECONCILE` | on | Reconciles multi-chapter heading roots and prevents duplicate or phantom chapter ladders; natural no-op on unrelated document shapes. |
| `SEMANTIK_ALTTEXT_PROVIDER` | `off` | Optional figure alt-text provider. `qwen30` selects the implemented provider path; unknown values resolve off. |
| `SEMANTIK_ALTTEXT_BASE_URL` | local OpenAI-compatible endpoint | Alt-text service endpoint; private deployment configuration. |
| `SEMANTIK_ALTTEXT_MODEL` | `qwen3-vl-30b` | Alt-text model identifier. Model changes require licensing review. |
| `SEMANTIK_ALTTEXT_API_KEY` | unset | Optional bearer credential. Supply only through secret management. |
| `SEMANTIK_ALTTEXT_CONCURRENCY` | `8` | Alt-text request fan-out width. |
| `SEMANTIK_ALTTEXT_TIMEOUT_SECONDS` | `120` | Per-request timeout in seconds. |

### Super heading judge

| Flag | Default | Purpose |
|---|---|---|
| `SEMANTIK_HEADING_JUDGE` | on | Master heading-judge gate. Unset, blank, truthy, or malformed values keep it on; only explicit falsey values disable it. |
| `SEMANTIK_HEADING_JUDGE_BASE_URL` | local OpenAI-compatible endpoint | Super heading-judge endpoint. Keep the concrete value private. |
| `SEMANTIK_HEADING_JUDGE_MODEL` | served Nemotron-3 Super identifier | Model identifier reported by the configured service. Model changes require licensing review. |
| `SEMANTIK_HEADING_JUDGE_API_KEY` | unset | Optional bearer credential supplied through secret management. |
| `SEMANTIK_HEADING_JUDGE_TIMEOUT` | `1200` | Timeout for one judge request, in seconds. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_TIMEOUT` | `5400` | Pipeline subprocess timeout for one chapter. A timeout retains the pre-judge result and records a warning. |
| `SEMANTIK_HEADING_JUDGE_CHECKPOINT` | on | Content-addressed per-window resume cache. The site value overrides `ED4ALL_GENERATION_CHECKPOINT`. |
| `SEMANTIK_HEADING_JUDGE_ENABLE_THINKING` | off | Enables model reasoning explicitly. Heading assignment defaults to compact classification without a reasoning block. |
| `SEMANTIK_HEADING_JUDGE_REASONING_EFFORT` | `high` when thinking is enabled | Selects `low`, `medium`, `high`, or `off`; ignored while thinking is disabled. |
| `SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY` | `0.3` thinking-off; omitted thinking-on | Anti-repetition request setting. Any finite explicit value is honored; zero omits the request key. |
| `SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF` | `4096` | Thinking-off completion ceiling. |
| `SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF` | `512` | Thinking-off completion floor. |
| `SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF` | `64` | Thinking-off completion estimate per pending heading. |
| `SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT` | `auto` | Reads the configured service context and derives safe prompt/completion budgets. `off` uses fixed compatibility budgets; a positive integer pins context directly. |
| `SEMANTIK_HEADING_JUDGE_CTX_MARGIN` | `4096` | Context headroom used by automatic budget derivation. |
| `SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION` | `0.7` | Completion share of usable context, clamped to `[0.4, 0.9]`. |
| `SEMANTIK_HEADING_JUDGE_CTX_BUDGET` | seat-derived | Explicit prompt-plus-completion budget override; fixed compatibility value is `31500`. |
| `SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET` | seat-derived | Explicit digest budget override; fixed compatibility value is `24000`. |
| `SEMANTIK_HEADING_JUDGE_MAX_TOKENS` | seat-derived when thinking is on | Thinking-on completion ceiling override; fixed compatibility value is `30000`. Thinking-off uses `SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF`. |
| `SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR` | `20480` when thinking is on | Thinking-on completion floor. Thinking-off uses the dedicated 512-token default. |
| `SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT` | `300` when thinking is on | Thinking-on per-heading estimate. Thinking-off uses the dedicated 64-token default. |
| `SEMANTIK_HEADING_JUDGE_MAX_PENDING_PER_WINDOW` | `96` | Hard pending-heading ceiling per window. The effective thinking-off cap is normally lower because it is budget-derived. |
| `SEMANTIK_HEADING_JUDGE_MIN_PENDING_WINDOW_CAP` | `8` | Lower bound for the budget-derived pending count. |
| `SEMANTIK_HEADING_JUDGE_CONCURRENCY` | `4` | Concurrent independent judge requests. |
| `SEMANTIK_HEADING_JUDGE_MAX_COVERAGE_RESPLIT_ROUNDS` | `3` | Maximum coverage-recovery split rounds. |
| `SEMANTIK_HEADING_JUDGE_MIN_PENDING_PER_SPLIT` | `2` | Minimum pending count eligible for another split. |
| `SEMANTIK_HEADING_JUDGE_ANCHOR_TRUNCATE` | `80` | Content-anchor character limit in judge digests. |
| `SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE` | `90` | Heading-text character limit in judge digests. |
| `SEMANTIK_HEADING_JUDGE_CONTEXT_TEXT_TRUNCATE` | `40` | Fixed-anchor context character limit. |
| `SEMANTIK_HEADING_JUDGE_TOKENIZER` | `auto` | Uses the configured judge model tokenizer when locally available; supported fallbacks remain deterministic. |
| `SEMANTIK_HEADING_JUDGE_TOKENIZER_ID` | judge model identifier | Optional explicit tokenizer identifier for offline loading. |
| `SEMANTIK_HEADING_JUDGE_AUDIT` | on | Deterministic post-judge structural audit. Explicit falsey values disable it. |
| `SEMANTIK_HEADING_JUDGE_REJUDGE` | off | Enables bounded targeted re-judging of audit failures. |
| `SEMANTIK_HEADING_JUDGE_REJUDGE_MAX_ATTEMPTS` | `1` | Maximum targeted re-judge attempts. |
| `SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT` | off | Adds a read-only whole-document heading skeleton to each window. |
| `SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS` | off | Adds content anchors to full-document context; no-op unless full-document context is enabled. |
| `SEMANTIK_HEADING_JUDGE_FINAL_REVIEW` | off | Runs one bounded whole-document consistency review after initial judgments. Failures retain the pre-review tree. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_MODE` | off | Uses one content-aware work unit per chapter instead of standard windows. |
| `SEMANTIK_HEADING_JUDGE_DOC_SCHEMA` | on within chapter mode | Adds a compact document-derived hierarchy convention to chapter work units. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_CONTENT_WORDS` | `60` | Per-region word cap when chapter content must be reduced to fit. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE` | off | Enables bounded overlapping slices for chapters exceeding the selected work-unit size. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW` | adaptive | Optional fixed normalized-window size. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE` | `100` | Percentile used to derive the adaptive window. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN` | `4096` | Lower clamp for the adaptive window. |
| `SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP` | `2` | Boundary headings repeated across adjacent normalized slices. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW` | on within normalized mode | Reconciles overlapping slice judgments; deterministic reconciliation remains available when disabled or unavailable. |

### Supported enrichment and compatibility controls

These controls remain supported for specialized inputs and compatibility
testing. They do not replace the preferred GLM-OCR route. Keep them unset
unless the owning architecture or operations guide calls for them.

| Flag | Default | Purpose |
|---|---|---|
| `SEMANTIK_STRUCTURE_CLEAN` | on | Deterministic structure cleanup with conservation checks. |
| `SEMANTIK_BLOCK_RESEGMENT` | off | Deterministic join/split pass over formed regions. |
| `SEMANTIK_BLOCK_RESEGMENT_LLM` | off | Adds bounded model proposals to resegmentation; deterministic conservation remains authoritative. |
| `SEMANTIK_UNIT_REGROUP` | on | Regroups adjacent pedagogical label/body regions under partition conservation. |
| `SEMANTIK_CONTAINMENT` | on | Builds materialized containment relationships used by accessible assembly. |
| `SEMANTIK_REGION_ORDER` | `fb` | Selects the supported region-order policy; `geom` and `off` are compatibility choices. |
| `SEMANTIK_COLUMN_EXTRACT` | off | Enables column-aware extraction. |
| `SEMANTIK_COLUMN_ORDER` | off | Enables column-major ordering; column-aware extraction implies it. |
| `SEMANTIK_DETECT_FIGURES` | off | Enables deterministic image-candidate extraction. |
| `SEMANTIK_VLM_FIGURE_DETECT` | off | Enables model-proposed sub-page figure boxes followed by deterministic acceptance guards. |
| `SEMANTIK_VLM_FIGURE_DETECT_CONCURRENCY` | `4` | Figure-detection request fan-out. |
| `SEMANTIK_VLM_FIGURE_DETECT_TIMEOUT` | `600` | Per-request figure-detection timeout in seconds. |
| `SEMANTIK_VLM_FIGURE_DETECT_CHECKPOINT` | on | Private content-addressed page cache with stop-aware dispatch. |
| `SEMANTIK_VLM_FIGURE_DETECT_DISABLE_THINKING` | off | Explicitly disables reasoning for figure localization; default behavior keeps it enabled. |
| `SEMANTIK_VLM_FIGURE_DETECT_MAX_WORDS` | `20` | Text-density rejection ceiling; zero disables this arm. |
| `SEMANTIK_VLM_FIGURE_DETECT_GRID_REJECT` | on within detector | Rejects table-like grids from the figure path. |
| `SEMANTIK_VLM_FIGURE_DETECT_MAX_PER_PAGE` | `8` | Hard accepted-figure cap per page. |
| `SEMANTIK_PAGE_ARRANGER` | off | Compatibility structure path for OCR-heavy pages; not part of the preferred profile. |
| `SEMANTIK_PAGE_ARRANGER_CONCURRENCY` | `4` | Page-arrangement request fan-out. |
| `SEMANTIK_REASONING_QC` | off | Optional report/reconcile quality-control pass. |
| `SEMANTIK_REASONING_QC_DISABLE_THINKING` | off | Explicit reasoning opt-out for quality-control requests. |
| `SEMANTIK_REASONING_QC_CONCURRENCY` | `8` | Quality-control request fan-out. |
| `SEMANTIK_SPECIALIST_PROVIDER` | `local` | Compatibility generation backend selector. Non-local providers require licensing and credential review. |
| `SEMANTIK_SPECIALIST_MODEL` | provider default | Compatibility generation model selector; changes require licensing review. |
| `SEMANTIK_SPECIALIST_BASE_URL` | provider-derived | Private OpenAI-compatible endpoint configuration. |
| `SEMANTIK_SPECIALIST_API_KEY` | provider-derived | Private credential configuration. |
| `SEMANTIK_THETA_DEVICE` | `cpu` | Device for the optional semantic-preservation evaluator; unavailable CUDA falls back to CPU. |
| `SEMANTIK_ALLOW_THETA_STUB` | off | Development/evaluation-only placeholder opt-in. Stub output is non-production and must never be treated as completed validation. |
| `SEMANTIK_MODEL_DIR` | package model root | Private model-artifact location. |
| `SEMANTIK_CACHE_DIR` | package data root | Private cache location. |
| `SEMANTIK_DATA_DIR` | package data root | Private evaluation/training-data location. |
| `SEMANTIK_CONFIG_DIR` | package data root | Private calibration/configuration location. |
| `SEMANTIK_HOME` | unset | Relocates the four private SemantiK data roots by basename. |
| `SEMANTIK_COURSE_CODE` | generic telemetry label | Decision-capture context only; never controls conversion. The resolved identifier is private. |
| `SEMANTIK_RUN_ID` | Ed4All run context when available | Decision-capture context only; never controls conversion. The resolved identifier is private. |

## Maintenance contract

- Resolver code and tests are authoritative for defaults and parsing.
- A new or changed `SEMANTIK_*` control must update this table in the same
  change.
- Provider or model selectors must also update `docs/LICENSING.md`.
- Flags must not silently downgrade required behavior. Optional operations may
  retain an input unchanged only when that fail-open behavior is explicit and
  observable.
- Never add source-specific measurements, deployment history, machine-specific
  values, private identifiers, or temporary implementation rationale to this
  public guide.
