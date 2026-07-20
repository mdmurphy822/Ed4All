# Hybrid vision extraction for scanned corpora

**Status:** **SUPERSEDED (2026-07-20)** — this was a design proposal (2026-07-11). The problem analysis in
§ 1 remains the canonical statement of why OCR is the quality ceiling, and § 1.3 (fidelity-blind gates) is
still true. The *solution* in §§ 2–8 was not built as described; a different architecture was adopted
instead. **See § 9 — What actually shipped** at the end of this file before acting on anything above it.
**Workstream:** scan-fidelity
**Measurement harness:** `scripts/integration/vision_ocr_probe.py`
**Relates to:** SemantiK cascade (`SemantiK/semantik_structure/cascade.py`),
`docs/LICENSING.md`, `docs/operations/nemotron-spark-serving.md`

## 1. Problem: Tesseract is the quality ceiling

For **image-only scanned** corpora (e.g. the OpenStax *Elementary Algebra 2e*
scan), Tesseract OCR at Stage 1 is the **sole source of text**. Every
downstream stage — the BERT council (structure/semantic/table/math heads),
`structure_graph`'s Region typing, and Stage-6 HTML authoring by the Omni
specialist — operates on that text. Whatever OCR loses is lost for the entire
pipeline; the Omni specialist cannot author a correct `<table>` from a table
whose structure OCR already shredded, and it cannot recover a superscript OCR
turned into a digit.

We stood up a multimodal model (Nemotron-Omni-30B-A3B, validated as a
near-perfect page transcriber — bring-up task #9) and then wired it in **behind
Tesseract as a text-only author** (Stage-6 prompts carry no image; confirmed in
`qwen_specialists/prompts.py`). So the model's defining capability — vision — is
unused on the exact corpus where OCR fails hardest.

### 1.1 Evidence (probe, ch01 p.57 / book p.61)

A three-column exercise page, math-dense. Tesseract at 216 DPI
(`SEMANTIK_OCR_RENDER_SCALE=3.0`, mean word-confidence 77.9 — the lowest of the
sampled pages) produced two classes of unrecoverable damage:

**A. Reading-order collapse (multi-column salad).** The third column detached
entirely: exercise *numbers* landed in one block and their *content* in a
separate block ~22 lines later —

```
211.            <- line 47          48 + (-16)        <- line 69
214.                                -17+(-18)+6
217.                                19 + 2(-3 + 8)
```

The number↔content binding is destroyed *before the council runs*, so no
downstream structure decision can restore it. Multi-column salad is an
extraction-level defect: it is not recoverable by any later stage, because the
information needed to re-pair a number with its content (spatial adjacency on
the page) is gone from the text stream by the time structure typing begins.

**B. Math corruption that silently yields wrong answers.**

| Printed | Tesseract | Effect |
|---|---|---|
| `5² − 6²` | `5* — 67` | superscripts destroyed |
| `6² − 7²` | `67-77` | reads as *67 minus 77* |
| `−|a|` | `—lal` | abs-value bars → letters |
| `−105 − (−68)` | `—105 — (68)` | negative sign dropped |

Answer keys synthesized from this are wrong by construction — the origin of a
recurring "wrong answer key" defect class on scanned math corpora. **The defect
is in extraction, not generation**, so no amount of generation-side fixing can
close it.

### 1.2 Empirical confirmation: the corruption already shipped

Scanning the `dart_chunks/chunks.jsonl` of a previously-delivered scanned
algebra course (525 chunks) finds the corruption **in shipped downstream
artifacts**, not just the raw page:
- **7 chunks** with symbol-injected math in factor-pair tables —
  `\(-1 + 5 = 4*\)`, `\(1 + (-5) = -4*\)`, `\(1 + (-9) = -8*\)` (spurious `*`).
- Running-text garbage — `Chapter 1 Foundations lll Decimals` (a rule / `|||`
  misread as `lll` mid-sentence).

So this is not a hypothetical — extraction corruption is demonstrably present in
courses we already built.

### 1.3 Why it went undetected: the gates are fidelity-blind

Every quality gate — `text_preserve`, NLI groundedness, sympy math-validity,
citation-anchoring — compares downstream artifacts against the **extracted
text**, never against the page pixels. So once `6² − 7²` becomes `67-77` at
extraction:

```
pixels ──Tesseract──▶ "67-77" ──▶ chunk ──▶ objective ──▶ assessment ──▶ answer
              ▲                                                         
        corruption      text_preserve ✓  groundedness ✓  sympy-valid ✓  anchor ✓
        enters here     (every gate treats "67-77" as ground truth → PASS)
```

`67-77` is valid arithmetic, faithfully preserved, grounded, and citable — so
**every gate passes** and the corruption is *laundered into "validated" output*.
No downstream gate can catch an extraction error because none of them look at the
original pixels. This is why the defect bled through weeks of downstream
symptom-fixing (wrong answer keys, TO over-segmentation, fragmented chunks)
invisibly. The corollary: the highest-leverage quality investment is **faithful
extraction or a vision fidelity check at extraction**, not another downstream
gate.

### 1.4 Scope (do not over-attribute)

This is **scan-specific**. Born-digital PDFs and vendor-HTML corpora carry a
text layer and bypass Tesseract, so they are clean of
*this* defect. And some downstream problems have unrelated roots — e.g. limited
block-type variety is a Courseforge **static-plan** issue, generation-side, not
extraction. Attribute to this root cause only: math corruption, multi-column
reading-order collapse, source fragmentation, and the scan TO over-segmentation
that follows from fragmentation.

## 2. Why it is wired text-only today (the contracts vision must respect)

Three load-bearing properties assume deterministic extraction:

1. **Text-preservation gate** (`gates/text_preserve.py`) — Stage-7/10 verify the
   authored HTML's text matches the *extracted source text*. This is what backs
   the WCAG conformance claim: the VLM only *reformats*, never *invents*. It
   presupposes one authoritative source text.
2. **Council geometric features** — the BERT heads consume per-block
   font/geometry/bbox/column features from pdfplumber+Tesseract (Stage 2). A raw
   VLM page transcription has no per-block bboxes, so it cannot *replace* the
   geometric extractor wholesale.
3. **License-clean-by-construction** — SemantiK ships Apache-2.0 because its
   extraction stack (pypdfium2 + pdfplumber + pikepdf + Tesseract) is
   permissively licensed. The **extracted text** is not a model derivative.
   Making a hosted/So-vendor model authoritative for extraction changes the
   licensing posture of the corpus (`docs/LICENSING.md`), which today treats an
   LLM as an *opt-in authoring seat*, never a *source*.

Any vision injection must be evaluated against all three.

## 3. Injection points

| # | Where | Omni's job | Leverage | Gate / license interaction |
|---|-------|-----------|----------|----------------------------|
| A | **Extraction (Stage 1–2), hybrid** | Tesseract keeps **bbox/geometry**; Omni transcribes **correct text** per block/region from its crop | Highest — fixes text for council + structure + authoring at once | ✅ Omni text *becomes* the source of truth → text-preservation is faithful by construction. ⚠️ licensing: Omni-derived source text (see §5) |
| B | **Structure reviewer (5d)** + page image | Omni looks at the page to fix headings / columns / reading-order the council mis-read from OCR features | Targets *structure* (the p.57 column-collapse class); cheap (~1 call/page) | ✅ Already text-conserving (edits Region IDs, never text). Lowest-risk win |
| C | **Math + table specialists (Stage 6), image-grounded** | Only the known-weak kinds get their region crop alongside OCR text | High per-region payoff (OCR fails worst here) | ⚠️ collides with text_preserve unless the gate allows visual correction for these kinds |
| D | **Full Stage-6 authoring** with region crops | Every region authored from text **+** image | Broad but expensive (visual tokens every region) | ❌ collides with text_preserve as-is; also collapses the two-source check (§5.2) |

The machinery is half-present: Stage 5c already renders figure Region bboxes →
PNG; the endpoint accepts `image_url` (task #9). The gap is cropping non-figure
regions and resolving the contracts.

## 4. Recommended design

**Two moves, sequenced by risk:**

### 4.1 P1 — image-grounded Stage-5d reviewer (cheap, no contract change)

Pass the **page image** to the existing 5d structure/block reviewer. It already
edits over Region IDs under a token-conservation invariant and fails closed, so
vision here can only *re-type / merge / split / re-order* — never inject text.
This directly attacks the p.57 **reading-order collapse** class without touching
text-preservation or licensing. Ship first.

### 4.2 P2 — hybrid extraction lane (opt-in, scan-specific)

A new extraction sub-lane, **off by default** (`SEMANTIK_VISION_EXTRACTION`,
proposed), that keeps the license-clean Tesseract path as the default:

```
Stage 1  Tesseract  -> per-block TEXT + bbox/geometry          (as today)
Stage 1.5 (opt-in)   for each block/region crop (from bbox):
                       Omni faithful transcription  ->  corrected TEXT
                       (bbox/geometry retained from Tesseract)
Stage 2+  council/structure/author run on the CORRECTED text
```

Key properties:
- **Geometry stays deterministic** — Tesseract still supplies the bboxes the
  council's features need; Omni only supplies text content. Resolves §2.2.
- **Text-preservation stays meaningful** — Omni's corrected text is the
  *source of truth*, so authoring is checked against faithful text, not
  mangled OCR. Resolves §2.1 — but see the two-source caveat in §5.2.
- **Granularity trade-off** — per-block crops = most faithful but many vision
  calls/page; per-Region crops (post-Stage-5) = fewer calls but the council
  already ran on OCR text, so structure is still OCR-shaped. Start per-Region
  for cost; escalate to per-block where the probe shows structure damage.

## 5. Contract changes required

### 5.1 Provenance + decision-capture
- Record the extraction method per block on the `data-dart-*` wire contract
  (`tesseract` vs `omni-vision`) so downstream + audits know the source.
- Every Omni extraction call is an LLM call site → must wire `DecisionCapture`
  with dynamic rationale (page, bbox, model, token counts) and a regression test
  asserting capture fires (root CLAUDE.md § LLM call-site instrumentation).

### 5.2 Text-preservation: keep a second witness
If **both** extraction and authoring are Omni, the text-preservation gate
degrades from *faithfulness* to *self-consistency* (Omni vs Omni). Mitigation:
retain the Tesseract text as a **secondary witness** — text_preserve checks the
authored HTML against a fuzzy union/agreement of {Omni-source, Tesseract}, and
divergence where Tesseract and Omni disagree is flagged for review rather than
silently trusted. This preserves an independent-ish anchor and surfaces vision
hallucination.

### 5.3 Licensing (`docs/LICENSING.md`)
Omni-extracted text makes the corpus text a **model derivative**. Therefore:
- The license-clean **default stays Tesseract** (opt-in flag, never silent).
- The vision lane lands with a `docs/LICENSING.md` row per the maintenance
  contract (any flag selecting an LLM provider/model for a synthesis/source
  surface needs one). For a self-hosted Nemotron seat the posture differs from a
  hosted vendor seat — document both.

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| **Hallucination** — vision invents text absent from the page | Tesseract second-witness cross-check (§5.2); numeric-grounding of computational lines; low temperature (0.0) |
| **Multi-column reading order** — does Omni actually get it right? | *Measure first* (the probe's whole point); explicit column-order directive in the transcription prompt |
| **Cost / throughput** — per-block vision on one seat | Per-Region granularity; scan-only opt-in; batch via vLLM continuous batching; only-weak-kinds (C) variant |
| **License drift** | opt-in default-off + `docs/LICENSING.md` row (§5.3) |
| **Faithfulness gate collapse** | second-witness design (§5.2) |

## 7. Phased rollout

- **P0 — measure** (now): `vision_ocr_probe.py` on ~5 OCR-hostile pages;
  quantify text/structure/math recovered by Omni vs Tesseract. Decision gate for
  P1/P2. CPU half (render+Tesseract) already staged; GPU half fires after the
  baseline conversion frees the seat.
- **P0.5 — extraction-fidelity gate** (visibility, independent of the fix):
  a vision check that samples extracted text, diffs it against the page image
  (Omni), and flags divergence as a warning-severity signal. This is the first
  gate in the architecture that is NOT fidelity-blind (§1.3) — it gives a
  *corruption rate* per corpus even before extraction is fixed, so the laundered
  errors become visible. Cheap (sampled, not every block).
- **P1 — 5d image review** (cheap win): page image → structure reviewer.
- **P2 — hybrid extraction lane** (opt-in): Stage 1.5 per-Region Omni text,
  Tesseract geometry retained; provenance + decision-capture + licensing row.
- **P3 — evaluate + promote**: re-run the ch01-03 subset with the lane on;
  compare against the text-grounded baseline (the current run) on math-answer
  correctness, reading-order, table structure. Promote scan-default only if the
  faithfulness gate holds.

## 8. Measurement plan

The probe writes, per page: `page.png`, `tesseract.txt`, `omni.md`,
`metrics.json` (math-marker counts, table-row counts, char deltas, garbage
counts) and a roll-up `probe_report.json`. The metrics rank pages and quantify
the delta; the **decisive read is visual** — page image vs Tesseract text vs
Omni transcription. Decision criterion for P2: Omni must recover reading order
and math on the hostile pages **without** introducing hallucinated text the
Tesseract second-witness can't corroborate.

---

## 9. What actually shipped (2026-07-20)

The P0 measurement ran and its conclusion held: **extraction is the ceiling.** But the P1/P2 design above —
keep Tesseract authoritative, bolt vision on as a corrector — was **not** what was built. A bake-off replaced
it with a stronger architecture, and the difference matters enough that §§ 2–8 should be read as a rejected
alternative rather than as a plan.

### 9.1 The adopted architecture

Vision does not *correct* Tesseract; it *replaces* the extractor outright. The GLM-OCR lane
(`SEMANTIK_GLMOCR_LANE`, opt-in, default off) is a whole-document converter that owns structure end to end:

```
PDF ──▶ 300-DPI pdftoppm page renders
    ──▶ GLM-OCR SDK  (PP-DocLayoutV3 layout model + per-region OCR on a seat)
    ──▶ DETERMINISTIC transform  (glmocr/transform.py)
    ──▶ region_provenance  ──▶ existing lib/semantik adapter ──▶ accessible HTML
```

Code lives at `SemantiK/semantik_structure/glmocr/` (`lane.py`, `transform.py`, `sdk_client.py`,
`math_normalize.py`, `heading_judge.py`, `alttext.py`), branching from `cascade.py::run_pipeline_v2` via
`_run_glmocr_lane_v2`. Default off means the branch never imports the lane, so the legacy cascade stays
byte-identical.

### 9.2 How this differs from the proposal above — and why

| § above | Proposed | Shipped |
|---|---|---|
| §2.2 "council needs Tesseract geometry" | Retain Tesseract for bboxes; vision supplies text only | **Council bypassed entirely.** The lane skips the council BERTs, cross-reranker, `structure_graph`, Stage-5d/5e, and theta. The layout model supplies regions + bboxes; the deterministic transform supplies typing. Learned heads are narrow candidate generators; auditable code beat them on the structural decisions. |
| §4.1 "P1: image-grounded Stage-5d reviewer" | Page image → the existing 5d reviewer | **Not built as such.** The reviewer stage the lane bypasses is the one this would have improved. The reasoning-model role instead landed as the **heading-level judge** (`SEMANTIK_HEADING_JUDGE`), which re-levels `heading_level_pending` headings over a chapter's ordered heading skeleton — text is never touched, only levels, under a deterministic clamp and a fail-open keep-current on any transport or parse failure. |
| §4.2 "P2: Stage-1.5 per-region Omni correction, Tesseract retained" | Hybrid two-source lane | **Rejected.** Single-source extraction from a purpose-built OCR stack, no Tesseract in the lane at all. |
| §5.2 "second witness" | Tesseract as an independent corroborating witness against vision hallucination | **Moot in the lane** — there is no Tesseract text to corroborate against. The faithfulness posture rests instead on (a) the extractor being a dedicated OCR model rather than a general VLM asked to transcribe, and (b) explicit escalation records: an unreachable alt-text seat leaves a placeholder and records a loud escalation, never fabricated text. |

The §5.2 concern was legitimate and is worth restating plainly: **the lane has no independent second witness on
extracted text.** That is a real, accepted residual risk, not a solved problem. The §1.3 fidelity-blind-gates
critique therefore still applies to the lane — no downstream gate compares the lane's output against page
pixels.

### 9.3 Contract dispositions

- **§5.1 provenance** — honored. The lane writes `{stem}.glmocr_layout.json` (per-page region layout, the
  provenance backbone) and `{stem}.glmocr_escalations.jsonl`. The existing `data-semantik-*` /
  `{slug}#{block_id}` wire contract is rendered unchanged by the existing adapter.
- **§5.1 decision-capture** — honored on the LLM surface. The heading judge emits one `structure_review`
  capture per chapter with a `heading_level_judge` metadata discriminator and a dynamic rationale (model,
  pending count, applied/clamped/dropped/kept tallies, `max_tokens`, `finish_reason`, cache hit/miss). The
  deterministic transform makes no model call and correctly wires no capture.
- **§5.3 licensing** — honored, and the posture came out *better* than the proposal feared. The proposal
  worried that model-derived extraction makes the corpus a model derivative. The adopted stack is
  permissively licensed throughout: GLM-OCR weights MIT, the `glmocr` SDK Apache-2.0, PP-DocLayoutV3
  Apache-2.0, and the alt-text seat (Qwen3-VL-30B-A3B-Instruct) Apache-2.0. Each model-selecting flag carries
  its `docs/LICENSING.md` row per the maintenance contract. The lane remains opt-in and default off.

### 9.4 Pipeline position

The heading judge is now a permanent `textbook_to_course` phase named `heading_judge`, sitting between
`semantik_conversion` and `staging` (`MCP/tools/pipeline_tools.py::_run_heading_judge`, routed by phase name
through `_PHASE_TOOL_MAPPING`). It runs the standalone judge with `--apply` per layout sidecar as a
subprocess, copies judged HTML and corrected escalations back over the conversion output (keeping `.prejudge.bak`
/ `.bak`; the layout sidecar is never overwritten), and **fail-opens per chapter** — a judge failure never
blocks a build. With the flag off, or on a born-digital corpus, the phase is skip-with-pass.

### 9.5 Still open

- **No pixel-level fidelity gate.** §7's P0.5 proposal — sample extracted text, diff it against the page
  image, report a per-corpus corruption rate — was not built. It remains the only design in this document
  that would make §1.3's laundering visible, and it is independent of which extractor wins. Worth building.
- **`region_kind` vs. the L1 block ontology.** The transform carries its own 25-class layout vocabulary,
  unreconciled with `schemas/taxonomies/block_kinds.json`. See `docs/architecture/block-ontology.md`
  § Adoption status.
