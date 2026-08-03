# SemantiK Behavior Flags

SemantiK converts source documents through the preferred conversion path in
this order: **GLM-OCR SDK extraction → deterministic normalization → document
and accessibility enrichment → Super heading judge**. This registry documents
the public `SEMANTIK_*` controls used by that path and its rendered-output
contract.

> **Private by design.** Source documents, converted HTML, course and run
> identifiers, caches, model artifacts, credentials, endpoint values,
> evaluation data, and logs are always operator-private. Keep them in ignored
> or external storage and never commit them to source, documentation, fixtures,
> examples, or comments.

See [installation](installation.md), [pipeline invocation](pipeline-invocation.md),
[licensing](../LICENSING.md), and the
[SemantiK architecture](../../SemantiK/architecture.md) before changing a
provider, model, or execution profile.

## Preferred conversion profile

`SEMANTIK_GLMOCR_LANE=1` selects the preferred conversion path. The GLM-OCR
lane is **opt-in**: unset, blank, falsey, or malformed values leave it off. The
heading judge is **default-on** when its workflow stage receives GLM-OCR
sidecars: only `0`, `false`, `no`, or `off` disables it. With no pending
headings, the judge is a natural no-op.

The tables use these parsing classes:

- **opt-in boolean** — only `1`, `true`, `yes`, or `on` enables; all other
  values resolve off;
- **default-on boolean** — only `0`, `false`, `no`, or `off` disables; malformed
  values retain the enabled default;
- **bounded number** — malformed, non-finite, or out-of-range input resolves to
  the documented default unless the row states a clamp; and
- **string** — blank input resolves to the documented default.

Service failures are always observable. GLM-OCR conversion failures fail the
producing task. Heading-judge request or chapter-timeout failures retain the
pre-judge structure and emit warnings; they never masquerade as successful
judgments. Deterministic audits and enrichments either report their failure or
leave the input unchanged according to the row below.

## Flag reference

Each row gives the live resolver default, accepted behavior, failure behavior,
and owning source. Endpoint values and credentials are configured privately;
the public default is described by role rather than exposing deployment data.

### GLM-OCR SDK and figure enrichment

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_GLMOCR_LANE` | off | Opt-in boolean selecting the whole-document SDK lane. Conversion errors fail the task. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_BASE_URL` | local OpenAI-compatible service | Nonblank URL override; blank uses the resolver default. Connection and protocol failures fail conversion. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_MODEL` | `glm-ocr` | Nonblank served-model override. An unavailable or mismatched model fails the SDK request. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_WORKERS` | `4` | Positive integer worker count; invalid or non-positive input uses `4`. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_RENDER_DPI` | `300` | Integer page-raster DPI with minimum `72`; invalid or smaller input uses `300`. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_LAYOUT_DEVICE` | `cpu` | Nonblank SDK layout-device string. Device startup failures fail conversion rather than silently changing device. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_GLMOCR_OUTPUT_DIR` | source-adjacent private output | Nonblank path redirects private layout and escalation sidecars. Unwritable output fails the producing step. Source: `SemantiK/semantik_structure/cascade.py` and `MCP/tools/pipeline_tools.py`. |
| `SEMANTIK_GLMOCR_MATH_NORMALIZE` | on within lane | Default-on boolean for deterministic OCR math, ordinal, and placeholder normalization. Disabled means the original transformed text is retained. Source: `SemantiK/semantik_structure/glmocr/math_normalize.py`. |
| `SEMANTIK_CHAPTER_LADDER_RECONCILE` | on | Default-on boolean reconciling chapter-root ladders during rendering. A nonmatching document shape is a no-op. Source: `lib/semantik/adapter.py`. |
| `SEMANTIK_ALTTEXT_PROVIDER` | `off` | Accepts `off` or `qwen30`; unknown input resolves `off`. Provider failure retains harvested captions or an honest placeholder. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_ALTTEXT_BASE_URL` | local OpenAI-compatible service | Nonblank private endpoint override. Used only when the provider is enabled. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_ALTTEXT_MODEL` | `qwen3-vl-30b` | Nonblank model override; changes require licensing review. Unavailable models follow the alt-text failure contract. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_ALTTEXT_API_KEY` | unset | Nonblank bearer credential; blank means no credential. Supply only through secret management. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_ALTTEXT_CONCURRENCY` | `8` | Positive integer request fan-out; invalid input uses `8`. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_ALTTEXT_TIMEOUT_SECONDS` | `120` | Positive finite request timeout; invalid input uses `120`. Timeout follows the alt-text failure contract. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |

### Super heading judge: service and request policy

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_HEADING_JUDGE` | on | Default-on boolean. Request failures retain the pre-judge hierarchy with a warning. Source: `SemantiK/semantik_structure/glmocr/__init__.py` and `MCP/tools/pipeline_tools.py`. |
| `SEMANTIK_HEADING_JUDGE_BASE_URL` | local OpenAI-compatible Super service | Nonblank private endpoint override; transport failures are classified and retained in reports. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_HEADING_JUDGE_MODEL` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Nonblank served-model override; changes require licensing review. A service/model mismatch is a request failure. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_HEADING_JUDGE_API_KEY` | unset | Nonblank bearer credential; blank means no credential. Source: `SemantiK/semantik_structure/glmocr/heading_judge_standalone.py`. |
| `SEMANTIK_HEADING_JUDGE_TIMEOUT` | `1200` | Positive finite per-request seconds; invalid input uses `1200`. Timeout triggers bounded split recovery, then a surfaced unresolved result. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_TIMEOUT` | `5400` | Positive finite subprocess seconds; invalid input uses `5400`. Timeout keeps that chapter's pre-judge result and records a warning. Source: `MCP/tools/pipeline_tools.py`. |
| `SEMANTIK_HEADING_JUDGE_CHECKPOINT` | on | Default-on content-addressed cache; the site value overrides `ED4ALL_GENERATION_CHECKPOINT`. Cache failure cannot fabricate a verdict. Source: `SemantiK/semantik_structure/glmocr/__init__.py`. |
| `SEMANTIK_HEADING_JUDGE_ENABLE_THINKING` | off | Opt-in boolean. Off sends compact classification requests; malformed input remains off. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_REASONING_EFFORT` | `high` when thinking is enabled | Accepts `low`, `medium`, `high`, or `off`; invalid input uses `high`; ignored while thinking is off. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY` | `0.3` thinking-off; omitted thinking-on | Any finite float is accepted while thinking is off; invalid input uses `0.3`; zero omits the request key. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |

### Super heading judge: context and capacity

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT` | `auto` | `auto` probes service context, `off` uses fixed compatibility budgets, and a positive integer pins context; invalid input uses `auto`. Probe failure uses explicit compatibility budgeting with a warning. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_SEAT_TIERS` | built-in ordered tiers | Comma-separated `name:context:sequences`; malformed entries are warned and skipped, and no valid entries restores defaults. Source: `SemantiK/semantik_structure/glmocr/seat_profile.py`. |
| `SEMANTIK_HEADING_JUDGE_SEAT_KV_FRONTIER` | unset | Positive integer derives sequence counts from context; invalid or non-positive input leaves tier counts unchanged. Source: `SemantiK/semantik_structure/glmocr/seat_profile.py`. |
| `SEMANTIK_HEADING_JUDGE_SEAT_MAX_SEQS` | `64` | Positive integer ceiling used with the KV frontier; invalid input uses `64`. Source: `SemantiK/semantik_structure/glmocr/seat_profile.py`. |
| `SEMANTIK_HEADING_JUDGE_SEAT_SELECT_SAFETY` | `1.3` | Positive finite fit multiplier; invalid input uses `1.3`. Overflow selects the largest tier and is surfaced. Source: `SemantiK/semantik_structure/glmocr/seat_profile.py`. |
| `SEMANTIK_HEADING_JUDGE_CTX_MARGIN` | `4096` | Positive integer context headroom; invalid or non-positive input uses `4096`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION` | `0.7` | Finite fraction clamped to `[0.4, 0.9]`; invalid input uses `0.7`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CTX_BUDGET` | seat-derived | Positive integer prompt-plus-completion override; invalid input uses the derived budget, or `31500` in fixed mode. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET` | seat-derived | Positive integer digest override; invalid input uses the derived budget, or `24000` in fixed mode. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MAX_TOKENS` | seat-derived thinking-on ceiling | Positive integer override; invalid input uses the derived ceiling, or `30000` in fixed mode. Thinking-off uses its dedicated control. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR` | `20480` thinking-on | Positive integer floor; invalid or non-positive input uses `20480`. Thinking-off uses its dedicated control. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT` | `300` thinking-on | Positive integer token estimate; invalid input uses `300`. Thinking-off uses its dedicated control. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF` | `4096` | Positive integer thinking-off ceiling; invalid input uses `4096`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF` | `512` | Positive integer thinking-off floor; invalid or non-positive input uses `512`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF` | `64` | Positive integer thinking-off estimate; invalid input uses `64`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MAX_PENDING_PER_WINDOW` | `96` | Positive integer hard ceiling; invalid input uses `96`. Effective capacity may be lower after budget calculation. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MIN_PENDING_WINDOW_CAP` | `8` | Positive integer lower bound; invalid input uses `8`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CONCURRENCY` | `4` | Positive integer independent-request fan-out; invalid input uses `4`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MAX_COVERAGE_RESPLIT_ROUNDS` | `3` | Positive integer recovery-round limit; invalid or non-positive input uses `3`. Exhaustion leaves uncovered headings explicit. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MIN_PENDING_PER_SPLIT` | `2` | Positive integer minimum eligible pending count; invalid input uses `2`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH` | `3` | Non-negative integer timeout-split depth; invalid input uses `3`. Exhaustion reports the failure mechanism. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |

### Super heading judge: digest, tokenizer, and document modes

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_HEADING_JUDGE_ANCHOR_TRUNCATE` | `80` | Positive integer content-anchor character cap; invalid input uses `80`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE` | `90` | Positive integer heading-text character cap; invalid input uses `90`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CONTEXT_TEXT_TRUNCATE` | `40` | Positive integer fixed-anchor context cap; invalid input uses `40`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_TOKENIZER` | `auto` | Accepts `auto`, falsey/off compatibility mode, or an explicit local tokenizer ID/path. Load failure uses a conservative estimator and warns; it never downloads. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_TOKENIZER_ID` | heading-judge model tokenizer | Nonblank local tokenizer override for `auto`; a local cache miss uses the conservative estimator. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT` | off | Opt-in boolean adding the read-only whole-document heading skeleton. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS` | off | Opt-in boolean adding content anchors; no-op unless full-document context is on. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_MODE` | off | Opt-in boolean selecting chapter work units. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_DOC_SCHEMA` | on within chapter mode | Default-on boolean adding a document-derived hierarchy convention; no-op outside chapter mode. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_CONTENT_WORDS` | `60` | Positive integer region-word cap used during fit reduction; invalid input uses `60`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE` | off | Opt-in boolean enabling bounded overlapping slices for oversized chapters. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW` | adaptive | Positive integer fixed slice size; invalid or absent input uses the adaptive resolver. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_PERCENTILE` | `100` | Integer percentile clamped to `[1, 100]`; invalid input uses `100`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_NORMALIZE_WINDOW_MIN` | `4096` | Positive integer adaptive lower clamp; invalid input uses `4096`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_SLICE_OVERLAP` | `2` | Non-negative integer repeated-boundary heading count; invalid input uses `2`. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_CHAPTER_REVIEW` | on within normalized mode | Default-on boolean reconciling slice judgments. Failure uses deterministic overlap reconciliation and warns. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_FINAL_REVIEW` | off | Opt-in boolean for bounded whole-document consistency review. Failure retains the pre-review tree. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |
| `SEMANTIK_HEADING_JUDGE_FINAL_REVIEW_MIN_CHAPTERS` | `2` | Positive integer minimum in normalized mode; invalid input uses `2`. Below the floor, chapter-review results stand. Source: `SemantiK/semantik_structure/glmocr/heading_judge.py`. |

### Super heading judge: audit and targeted recovery

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_HEADING_JUDGE_AUDIT` | on | Default-on deterministic post-judge audit. Audit failure is warned and never rewrites the judged tree. Source: `MCP/tools/pipeline_tools.py`. |
| `SEMANTIK_HEADING_JUDGE_AUDIT_COLLAPSE_SHARE` | `0.95` | Finite fraction in `[0, 1]`; invalid input uses `0.95`. Source: `SemantiK/semantik_structure/glmocr/heading_judge_audit.py`. |
| `SEMANTIK_HEADING_JUDGE_AUDIT_MIN_HEADINGS` | `4` | Positive integer minimum for collapse checks; invalid input uses `4`. Source: `SemantiK/semantik_structure/glmocr/heading_judge_audit.py`. |
| `SEMANTIK_HEADING_JUDGE_REJUDGE` | off | Opt-in boolean re-running only audit-flagged chapters. Failure retains the prior judged output and warns. Source: `MCP/tools/pipeline_tools.py`. |
| `SEMANTIK_HEADING_JUDGE_REJUDGE_MAX_ATTEMPTS` | `1` | Positive integer targeted-recovery limit; invalid input uses `1`. Source: `MCP/tools/pipeline_tools.py`. |

### Deterministic rendering and enrichment

| Flag | Default | Behavior, failure, and source |
|---|---|---|
| `SEMANTIK_DROP_FRONTMATTER_TOC` | on | Default-on detector for printed front-matter contents runs. Explicit falsey input preserves them. Source: `lib/semantik/toc_frontmatter_detector.py`. |
| `SEMANTIK_DROP_FRONTMATTER_ZONE` | on | Default-on suppression of non-content front-matter zones; falsey input preserves the zone. Source: `lib/semantik/toc_frontmatter_detector.py`. |
| `SEMANTIK_EMIT_TOC` | on | Default-on accessible contents navigation; explicit falsey input disables it. Source: `lib/semantik/adapter.py`. |
| `SEMANTIK_BOX_TITLE_HEADINGS` | off | Opt-in boolean carving presentational callout titles into headings under deterministic guards. Ineligible blocks remain unchanged. Source: `lib/semantik/adapter.py`. |
| `SEMANTIK_TABLE_STRUCTURE` | off | Opt-in deterministic table-cell topology reconstruction. Unconfirmed topology remains ordinary text/table content. Source: `lib/semantik/adapter.py`. |
| `SEMANTIK_LATEX_MATHML` | off | Opt-in LaTeX-to-presentation-MathML rendering. Missing validity support or invalid output fails loudly instead of emitting unvalidated MathML. Source: `lib/semantik/adapter.py` and `lib/semantik/latex_mathml.py`. |
| `SEMANTIK_RENDER_TIKZ_FIGURES` | off | Opt-in deterministic rendering of accepted TikZ figure sources. Render failure preserves the source representation and is surfaced. Source: `lib/semantik/adapter.py`. |
| `SEMANTIK_SEMANTIC_SUBCLASS` | off | Opt-in model-assisted composite-unit subclass label. Call or parse failure leaves the unit unlabelled; no prose is invented. Source: `lib/semantik/subclassifier.py`. |
| `SEMANTIK_SUBCLASS_SAMPLES` | `1` | Positive integer self-consistency sample count; invalid input uses `1`; only used when subclassing is enabled. Source: `lib/semantik/subclassifier.py`. |
| `SEMANTIK_AFFORDANCE_GATE` | off | Opt-in deterministic affordance-conservation audit. Findings are explicit and do not fabricate repaired content. Source: `lib/semantik/affordance_conservation.py`. |
| `SEMANTIK_AFFORDANCE_SECTION_RECALL_MIN` | `0.80` | Finite fraction in `[0, 1]`; invalid input uses `0.80`; only used by the affordance audit. Source: `lib/semantik/affordance_conservation.py`. |

## Registry boundary

This page intentionally excludes compatibility-only extraction paths, internal
subprocess plumbing, usage-meter destinations, run identifiers, private cache
locations, and controls with no production source reader. Those are not public
SemantiK behavior contracts.

Resolver code and focused tests are authoritative. A public flag change must
update its row in the same change; provider and model selectors must also update
`docs/LICENSING.md`. Required behavior must fail loudly. Optional behavior may
retain its input only when that outcome is explicit, observable, and documented.
