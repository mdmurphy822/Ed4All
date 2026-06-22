# Image alt-text & figure descriptions (WCAG 1.1.1) — routed, caption-first

**Plan version:** 2026-05-31 (rev 3 — SHIPPED, §7 decisions resolved, build + tests + eval landed)
**Branch:** `snapshot/runtime-2026-05-28` (uncommitted)
**Status:** SHIPPED 2026-06-09 — tested (17 unit tests for the two new modules
+ 16 eval-harness tests), evaluated (`figure_captioner_v1.json`: axe 1.0 /
regression 0.0; `figure_captioner_v2_guard.json`: no_hallucination 1.0,
ship=true), with the numeric-claim guard wired at
`assembler/fallbacks.py` (~line 276). Resolves task #20. The scope check
(#19) confirmed image *detection* works but alt-text *generation* was unbuilt.
The runtime path now exists end-to-end (Stage 5c `image_extract.py` + Stage 6b
`figure_captioner.py`, wired into `cascade.py` and `assembler/fallbacks.py`),
with the `extended_description → aria-describedby` emission in place. **All §7
decisions are RESOLVED (see §7).** See `Plans/10_completion_punchlist.md` for
the dispatchable task list.

Cross-links: realizes the `enrich.py` Stage-6 contract (`FIGURE -> alt_text
+ extended_description`); supersedes / re-scopes the Phase 3f
`BERT-ImageSpecialist` design (`architecture.md` §11 ~856-860,
`Plans/01` §3f) — see §6. Honors `feedback_no_external_llms` (local only),
`feedback_license_policy` (commercial-OK weights only), and
`feedback_no_silent_fallbacks` (no hallucinated descriptions).

---

## 0. Current state

| layer | state |
|---|---|
| detect image regions | ✅ Structure `is_image_block` head → cross_reranker `image_block_demoted` → Stage-5 `kind="figure"` |
| two-tier output contract | ✅ designed in `enrich.py:11,19-24` (`alt_text` short + `extended_description` for complex) |
| `summarize_table` (aria-describedby) | ✅ deterministic, shipped (`enrich.py:110`) |
| `alt_text_for_image` / `describe_figure_extended` | ✅ **`NotImplementedError` stubs removed 2026-06-09** — figure alt-text now flows through the Stage-6b cascade (`figure_captioner.caption_figure_regions` → `assembler/fallbacks.fallback_figure`) |
| **image bytes at Stage 6** | ⬜ **BLOCKER** — Stage-1 `extract.py` emits text blocks only; image bytes aren't carried through (`enrich.py:56` + docstring) |

**The model is not the first problem.** Nothing can caption an image until
image bytes flow from Stage 1 to Stage 6. That plumbing is deterministic and
model-agnostic — do it first (§1).

## 0.1 Why this matters for DART specifically

The corpus is scientific (arXiv/PMC): figures are overwhelmingly **charts,
plots, and diagrams**, not photos. A generic captioner emits *"a line graph
with a blue line"* — useless as accessible alt text. Winning on the
accessibility axis means the **chart tier** has to actually read the figure
(*"line plot of decoherence rate vs. coupling strength; exponential decay
above M≈10"*). That single fact drives the model choice.

## 1. Prerequisite — image extraction (CPU, deterministic, model-agnostic)

Carry figure pixels from Stage 1 to Stage 6. No model.

1. `extract.py`: for each detected figure region, either (a) pull the
   embedded raster via `pymupdf` (`Page.get_images` / `extract_image`) when
   the figure is a single embedded image, or (b) render the figure bbox to
   PNG at print DPI (`Page.get_pixmap(clip=bbox)`) for vector/composite
   figures (most scientific plots).
2. Carry the bytes on the block/IR (e.g. `ResolvedBlock.image_bytes` or a
   figure-region payload) through to Stage 6.
3. **No silent fallback:** if a figure region yields no extractable pixels,
   raise/flag explicitly — don't emit an empty figure as if captioned.

This unblocks *any* captioner and is independently shippable.

## 2. Strategy — caption-first, routed, no-hallucination

Do **not** "send every figure to one VLM and trust it." Route:

1. **Caption-first (no model).** When the source doc carries a
   `<figcaption>` / caption run, that text is the highest-precision alt
   source — zero hallucination risk. (`parse_ar5iv` already *drops*
   caption-less figures in training "we have no alt-text ground truth" →
   most academic figures carry captions.) Use the caption; the model's job
   shrinks to the *extended* description / "what the visual adds beyond the
   caption."
2. **Decorative → `alt=""`** (WCAG-correct), gated by detection — no model,
   no generation.
3. **Generate only when needed**, two tiers (§3):
   - simple / photo / diagram → short factual `alt_text`
   - chart / plot → richer `extended_description` from a chart-capable model
4. **No hallucinated data.** If the model can't describe a chart
   confidently, emit a conservative *type-level* alt ("Figure: scatter plot,
   see caption") + reference the caption/surrounding text — never invented
   numbers. A wrong data description is worse than a modest one for a
   screen-reader user.

## 3. Model options (filtered: local-only, ≤~8GB, commercial-OK license)

| model | size | license | role for DART | weakness |
|---|---|---|---|---|
| **Florence-2** base/large | 230M / 770M | MIT ✓ | designed-in; OCR + region caption; decorative/simple tier | generic on charts |
| **Moondream2** | ~1.9B | Apache-2.0 ✓ | small promptable VQA; better reasoning than Florence | limited on dense plots |
| **SmolVLM2** | 256M–2.2B | Apache-2.0 ✓ | on-device, decent doc/chart for size | weaker on complex figures |
| **Qwen2.5-VL-3B/7B** | 3B / 7B | 7B **Apache-2.0 ✓** / 3B **VERIFY** | strong chart+doc+OCR; same family as the Qwen specialists; 3B-4bit ~3GB fits | 7B-4bit (~5-6GB) tight; latency |
| **DePlot / MatCha** (Pix2Struct) | ~300M | Apache-2.0 ✓ | chart-specialist: extracts the chart's data table → feed deterministic summary | charts only |
| PaliGemma-2 | 3B | **Gemma license ⚠️** | strong VQA | custom use-restrictions — likely fails strict policy |

Out of scope: cloud vision (GPT-4V/Gemini/Claude) per `feedback_no_external_llms`;
Llama/Vicuna-derived weights need a license check.

**Recommended (rev 1):** single image model = **Qwen2.5-VL-3B (4-bit)** for
both tiers — one model, same family/tooling as the specialists, chart-capable,
fits 8GB, promptable with caption + surrounding text for *contextual* alt.
Keep **Florence-2-base** as the ultra-light fallback (and for the decorative/
simple tier if 3B latency hurts). **Gate on verifying the Qwen2.5-VL-3B
license** against `feedback_license_policy`; if it fails, fall back to
Florence-2 (simple tier) + **DePlot→deterministic summary** (chart tier),
both clean Apache/MIT.

Runtime note: the pipeline runs serial (one model resident at a time), so the
image model loads when figures are processed and unloads after — even a 7B-4bit
fits in isolation, but the small models avoid load/unload churn.

## 3a. Data sourcing — image content with alt-text ground truth (expansion)

The current corpus is text/HTML-derived; figures are effectively dropped
(`parse_ar5iv`: "drop figures without `<figcaption>` — no alt-text ground
truth"). To eval the captioner (and to fine-tune one if we go that route) we
need pairs of **(figure image → ground-truth alt/description)**. Two uses,
two license bars:

- **Generative use** (fine-tuning a captioner, or any target set the model is
  optimized toward) → strict: CC-BY / CC0 / ODC-By / US-gov-PD only.
  **CC-BY-SA (Wikipedia / Wikimedia Commons) is OFF-LIMITS here** — share-alike
  on a generative model is precisely the exposure `feedback_license_policy`
  blocks. (CC-BY-SA was fine for the *discriminative* Structure BERT — it
  classifies, doesn't reproduce content — but NOT for a generative captioner.)
- **Eval-only** (measuring axe / alt quality on held-out figures) → could be a
  touch broader, but keep it clean to avoid contaminating a future fine-tune.

Ranked sources (license-clean, accessibility-relevant):

| source | license | why | ground truth |
|---|---|---|---|
| **US-gov Section-508 PDFs** (govinfo, agency reports) | US-gov PD ✓ | **professionally authored alt text** (508 is legally required) — best alt-text ground truth for our axis | PDF `/Alt` + figure captions |
| **ar5iv + PMC OA figures w/ figcaption** | CC-cleared / CC-BY ✓ | already paired; **scientific charts = the hard case**; caption = description | `<figcaption>` |
| **OpenStax figures** | CC-BY ✓ | textbook figures + captions, often with real alt | alt + caption |
| CC0 / PD image+caption sets | CC0 ✓ | clean generative use | caption |
| ~~Wikipedia / Wikimedia Commons~~ | CC-BY-SA ✗ | off-limits for a generative captioner | — |

**Prerequisite is the same as §1:** we can't build any of these pairs until
image extraction carries figure pixels through the pipeline. §1 unblocks both
the runtime captioner *and* this dataset expansion.

The standout is **Section-508 government PDFs**: public-domain *and* they carry
human-written alt text (the PDF `/Alt` attribute), i.e. real accessibility
ground truth, directly on DART's strategic axis. Worth a dedicated extractor
(pull `/Alt` + the figure image) once §1 lands.

Outward-facing fetches (govinfo, PMC OA bulk, OpenStax) — confirm before
running, per license policy.

## 4. WCAG output mapping

- `alt_text` → the `alt` attribute (short, ≤ ~125 chars convention).
- `extended_description` → `aria-describedby` target (e.g. a visually-hidden
  block or `<details>`), mirroring the existing `summarize_table`
  aria-describedby pattern. Not `longdesc` (deprecated in HTML5).
- Decorative → `alt=""` + `role="presentation"` where appropriate.
- All figure output must pass the existing per-region axe gate.

## 5. Acceptance gates (before shipping the captioner)

| gate | target |
|---|---|
| image extraction | every detected figure yields pixels OR an explicit raise/flag (no silent empty) |
| decorative | decorative images get `alt=""`, not a generated caption |
| caption-first precision | when a `<figcaption>` exists, alt derives from it (no model override) |
| axe / WCAG | figure fragments pass the per-region hard gate; 0 new violations vs baseline (mirror the table-adapter eval: `axe_pass`, `axe_regression`) |
| no-hallucination | a held-out chart set: generated descriptions contain no numeric claims absent from the chart/caption (manual or rule-spot-check); low-confidence → conservative type-level alt |
| license | chosen weights are Apache/MIT/clearly commercial-OK; Qwen2.5-VL-3B license verified or fallback used |

## 6. Re-scoping the Phase 3f `BERT-ImageSpecialist`

The original plan (`architecture.md` §11, `Plans/01` §3f) specced a 3-head
BERT (`caption_role`, `caption_position`, `is_alt_candidate`) — a
*classification/routing* model. With a VLM doing *generation*, that BERT is
likely **over-scoped**: the routing it provides (decorative vs
figure-with-caption vs chart; where the caption sits) can come from
`is_image_block` + lightweight heuristics + the caption-first check, not a
trained BERT. **Recommendation: descope the standalone BERT-ImageSpecialist;**
fold its routing job into deterministic heuristics + the VLM. Revisit a
trained router only if heuristic routing proves insufficient on a held-out
figure set. (Decision belongs in §7.)

## 7. Decision axes — RESOLVED (2026-05-30/31)

All four were resolved in code; recorded here so the plan is not stale.

1. **Caption-first-then-generate** vs **always-generate** → **caption-first**
   (the assembler's `fallback_figure` reads `<figcaption>` when present and
   only falls back to the model `alt_text`; `dart_semantic/assembler/fallbacks.py:237-250`).
2. **One vision model vs split** → **one model**, but NOT the rev-1 pick.
   **Qwen2.5-VL-3B was REJECTED** — its weights are under the Qwen Research
   License (non-commercial), which fails `feedback_license_policy`. **Florence-2
   + DePlot was REJECTED** — two transformers-5.x compatibility breaks in
   Microsoft's checkpoint (config + processor) needing fragile monkey-patches.
   **Chosen: SmolVLM2-256M-Video-Instruct** (`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`,
   Apache-2.0, HF-maintained so transformers-native, ~512MB fp16). See the
   `dart_semantic/figure_captioner.py` module docstring for the full rationale.
3. **Chart-description aggressiveness** → the captioner emits a short `alt_text`
   and a longer `extended_description` from generic prompts (no chart-specific
   data extraction yet). DePlot chart-tier is deferred until a chart-vs-photo
   router exists (then evaluate against the `data/figure_alt_dataset`
   19,357-pair set). The no-hallucination gate (§5) is NOT yet enforced in code.
4. **Descope Phase 3f BERT-ImageSpecialist** → **yes, descoped.** Detection is
   carried by Structure's new `is_image_block` head (present in
   `structure_v2/final`, pos-F1 0.738) → cross-reranker `image_block_demoted`
   → Stage-5 `kind="figure"`; no standalone ImageSpecialist BERT is built.

## 8. Sequencing — STATUS (2026-05-31)

1. ✅ **Stage-1/5c image extraction → carry bytes to Stage 6 (CPU).**
   DONE — `dart_semantic/image_extract.py` (`render_figure_regions_to_bytes`,
   pypdfium2 bbox→PNG, `FigureRenderError` no-silent-fallback), wired into
   `cascade.py:266-277`. **Untested** (no `tests/` file).
2. ✅ **Resolve §7 decisions + verify model license.** DONE — see §7.
3. ◐ **Wire the chosen captioner.** DONE for the runtime path —
   `dart_semantic/figure_captioner.py` (`caption_figure_regions`, SmolVLM2,
   `FigureCaptionError`), wired into `cascade.py:286-297`; `fallback_figure`
   consumes `alt_text`. **REMAINING:** `extended_description → aria-describedby`
   emission (the assembler currently emits `alt` only; the `extended_description`
   payload field is captured but never rendered to HTML). The `enrich.py`
   `NotImplementedError` stubs (`enrich.py:107,134`) are now superseded by the
   cascade Stage-6b path but are NOT removed. **Untested** (no `tests/` file).
4. ⬜ **Eval — NOT STARTED.** No figure eval harness exists in `scripts/`
   (only the three `fetch_*_figure_assets.py` fetchers + `build_figure_alt_dataset.py`).
   Need the §5 gate harness: axe pass/regression on figure fragments +
   no-hallucination spot-check, stratified by figure subtype, mirroring
   `scripts/eval_qwen_table_adapter.py`.

### 8.1 Data note — coverage_report.json is stale

`data/figure_catalog/coverage_report.json` reports `"image_local": 0` /
`"needs_image_fetch": 20922` because it was written when the catalog was built
(2026-05-28), BEFORE the image fetch. The images were then fetched —
`data/figure_images/` is 2.0 GB across arxiv/pmc/openstax with a 20,961-line
`_fetch_manifest.jsonl`, and `data/figure_alt_dataset/` (train/val/test) was
assembled from the join. Regenerate the catalog coverage report (or trust the
`figure_alt_dataset/coverage_report.json` instead) so `image_local` is accurate.

## Do-NOT list
- No cloud/external vision APIs (`feedback_no_external_llms`).
- No hallucinated chart data — conservative type-level alt + caption reference
  when uncertain (`feedback_no_silent_fallbacks`).
- Don't generate alt for decorative images — emit `alt=""`.
- Don't override an existing `<figcaption>` with a model guess (caption-first).
- License: Apache/MIT/clearly-commercial-OK weights only; verify Qwen2.5-VL-3B
  or use the Florence-2 + DePlot fallback (`feedback_license_policy`).
- Don't build the Phase 3f BERT until heuristic routing is shown insufficient.
